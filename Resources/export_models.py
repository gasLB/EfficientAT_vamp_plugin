"""Exports the pretrained EfficientAT models to TorchScript files for the Vamp plugin.

The EfficientAT sources (https://github.com/fschmid56/EfficientAT) and the pretrained
weights are downloaded automatically into the `.cache` directory of the project, and the
TorchScript models are written into the `models` directory of the project.

See the README for the setup of the Python environment.
"""

import contextlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")

# The EfficientAT sources are pinned to a commit for reproducibility
EFFICIENTAT_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
EFFICIENTAT_URL = f"https://github.com/fschmid56/EfficientAT/archive/{EFFICIENTAT_COMMIT}.zip"
EFFICIENTAT_DIR = os.path.join(CACHE_DIR, f"EfficientAT-{EFFICIENTAT_COMMIT}")
WEIGHTS_DIR = os.path.join(CACHE_DIR, "weights")

# The version of LibTorch used by the plugin (see Resources/libtorch.cmake)
LIBTORCH_VERSION = "2.4.1"

# The output file names and the EfficientAT pretrained model names (AudioSet, 527 classes)
MODELS = [
    ("01_mn04_as.ts", "mn04_as"),
    ("02_mn10_as.ts", "mn10_as"),
    ("03_mn20_as.ts", "mn20_as"),
]

SAMPLE_RATE = 32000
# 10 seconds, the length of the AudioSet clips used to train the models
SEGMENT_LENGTH = 320000


def download_efficientat() -> None:
    """Downloads and extracts the EfficientAT sources if they are not in the cache."""
    if os.path.isdir(EFFICIENTAT_DIR):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Downloading EfficientAT sources from {EFFICIENTAT_URL}")
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as tmp_dir:
        archive_path = os.path.join(tmp_dir, "EfficientAT.zip")
        urllib.request.urlretrieve(EFFICIENTAT_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp_dir)
        shutil.move(os.path.join(tmp_dir, os.path.basename(EFFICIENTAT_DIR)), EFFICIENTAT_DIR)


class MelSpectrogram(nn.Module):
    """Eval-mode equivalent of EfficientAT's AugmentMelSTFT for the export.

    AugmentMelSTFT builds the mel filter bank on the CPU at each call, so a trace freezes it
    as a CPU constant and the model can't run on another device (MPS, CUDA). Here, all the
    constant tensors are buffers, so they follow the module when it is moved to a device.
    """

    def __init__(self, reference: nn.Module):
        super().__init__()
        self.n_fft: int = reference.n_fft
        self.hop_length: int = reference.hopsize
        self.win_length: int = reference.win_length
        self.register_buffer("preemphasis", reference.preemphasis_coefficient.clone())
        self.register_buffer("window", reference.window.clone())
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            reference.n_mels, reference.n_fft, reference.sr, reference.fmin, reference.fmax,
            vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0)
        self.register_buffer("mel_basis", F.pad(mel_basis, (0, 1), mode="constant", value=0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv1d(x.unsqueeze(1), self.preemphasis).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=True)
        x = x.real.square() + x.imag.square()  # power magnitude
        melspec = torch.matmul(self.mel_basis, x)
        melspec = (melspec + 0.00001).log()
        return (melspec + 4.5) / 5.0  # fast normalization


class InnerPipeline(nn.Module):
    """Combines the mel spectrogram and the MobileNetV3 for a clean TorchScript trace."""

    def __init__(self, mel: nn.Module, net: nn.Module):
        super().__init__()
        self.mel = mel
        self.net = net

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The plugin passes the shape [batch_size, 1, segment_length]
        x = x.squeeze(1)  # -> [batch_size, segment_length]
        spec = self.mel(x).unsqueeze(1)  # -> [batch_size, 1, n_mels, frames]
        out = self.net(spec)
        return out[0], out[1]


class EfficientATWrapper(nn.Module):
    """Exposes the methods required by the plugin (see source/eat_vamp_plugin.cpp)."""

    def __init__(self, traced_pipeline: torch.jit.ScriptModule, sr: int, seg_len: int, classes: List[str]):
        super().__init__()
        self.pipeline = traced_pipeline
        self._sample_rate: int = sr
        self._segment_length: int = seg_len
        self._class_names: List[str] = classes

    @torch.jit.export
    def sample_rate(self) -> int:
        return self._sample_rate

    @torch.jit.export
    def segment_length(self) -> int:
        return self._segment_length

    @torch.jit.export
    def class_names(self) -> List[str]:
        return self._class_names

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pipeline(x)


def check_model(path: str, reference_mel: nn.Module, reference_net: nn.Module) -> None:
    """Reloads the exported model and compares it with the original EfficientAT model on CPU and MPS."""
    torch.manual_seed(0)
    x = 0.1 * torch.randn(2, 1, SEGMENT_LENGTH)
    with torch.no_grad():
        expected = reference_net(reference_mel(x.squeeze(1)).unsqueeze(1))[0].reshape(2, -1)
        model = torch.jit.load(path, map_location="cpu").eval()
        result = model(x)[0].reshape(2, -1)
        cpu_error = (result - expected).abs().max().item()
        if cpu_error > 1e-3:
            raise RuntimeError(f"{path}: CPU output differs from EfficientAT (max error {cpu_error})")
        print(f"  CPU: OK (max error {cpu_error:.2e})")

        if not torch.backends.mps.is_available():
            print("  MPS: not available, skipped")
            return
        model.to("mps")
        result = model(x.to("mps"))[0].to("cpu").reshape(2, -1)
        mps_error = (result - expected).abs().max().item()
        if mps_error > 1e-2:
            raise RuntimeError(f"{path}: MPS output differs from EfficientAT (max error {mps_error})")
        print(f"  MPS: OK (max error {mps_error:.2e})")


def main() -> None:
    if not torch.__version__.startswith(LIBTORCH_VERSION):
        print(f"Warning: torch {torch.__version__} is used but the plugin uses LibTorch {LIBTORCH_VERSION}, "
              "the exported models may not load in the plugin", file=sys.stderr)

    download_efficientat()
    sys.path.insert(0, EFFICIENTAT_DIR)
    # EfficientAT loads the class labels with a path relative to its root directory
    with contextlib.chdir(EFFICIENTAT_DIR):
        import models.mn.model as mn_model
        from models.preprocess import AugmentMelSTFT
        from helpers.utils import NAME_TO_WIDTH, labels as audioset_labels
    mn_model.model_dir = WEIGHTS_DIR

    # EfficientAT's mel spectrogram configuration of the AudioSet models
    reference_mel = AugmentMelSTFT(n_mels=128, sr=SAMPLE_RATE, win_length=800, hopsize=320, n_fft=1024,
                                   fmin=0.0, fmax=None).eval()
    mel = MelSpectrogram(reference_mel).eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for file_name, pretrained_name in MODELS:
        print(f"Exporting {pretrained_name}")
        # EfficientAT prints the whole network when it is created
        with contextlib.redirect_stdout(None):
            net = mn_model.get_model(width_mult=NAME_TO_WIDTH(pretrained_name), pretrained_name=pretrained_name)
        net.eval()

        # Traces the audio -> spectrogram -> logits pipeline with a dummy input on the CPU
        pipeline = InnerPipeline(mel, net).eval()
        dummy_input = torch.zeros(1, 1, SEGMENT_LENGTH, dtype=torch.float32)
        with torch.no_grad():
            traced_pipeline = torch.jit.trace(pipeline, dummy_input)

        # Scripts the outer wrapper so the plugin can call sample_rate(), segment_length() and class_names()
        wrapper = EfficientATWrapper(traced_pipeline, SAMPLE_RATE, SEGMENT_LENGTH, list(audioset_labels)).eval()
        scripted_model = torch.jit.script(wrapper)

        out_path = os.path.join(OUTPUT_DIR, file_name)
        scripted_model.save(out_path)
        print(f"  Saved to {out_path}")
        check_model(out_path, reference_mel, net)


if __name__ == "__main__":
    main()
