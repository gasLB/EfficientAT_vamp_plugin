import os
import sys
from typing import List, Tuple
import torch
import torch.nn as nn

# 1. Switch into EfficientAT so its relative paths ('metadata/...', 'resources/...') work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EFFICIENTAT_DIR = os.path.join(PROJECT_ROOT, "EfficientAT")
os.chdir(EFFICIENTAT_DIR)
sys.path.insert(0, EFFICIENTAT_DIR)

from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT
from helpers.utils import labels as audioset_labels


class InnerPipeline(nn.Module):
    """Combines Mel Spectrogram + MobileNetV3 for clean TorchScript tracing."""
    def __init__(self, mel: nn.Module, net: nn.Module):
        super().__init__()
        self.mel = mel
        self.net = net

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # C++ passes shape: [batch_size, 1, segment_length]
        x = x.squeeze(1)  # -> [batch_size, segment_length]
        spec = self.mel(x).unsqueeze(1)  # -> [batch_size, 1, n_mels, frames]
        out = self.net(spec)
        return out[0], out[1]


class EfficientATWrapper(nn.Module):
    """Exposes the exact methods required by classifier_vamp_plugin.cpp."""
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


def main():
    sample_rate = 32000
    # 10 seconds = 320000 samples (or 2.5 seconds = 80000 samples)
    segment_length = 320000

    # 1. Initialize EfficientAT Mel Spectrogram preprocessor
    mel = AugmentMelSTFT(
        n_mels=128,
        sr=sample_rate,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        fmin=0.0,
        fmax=None,
    )
    mel.eval()

    # 2. Initialize mn10_as model
    net = get_mobilenet(width_mult=1.0, pretrained_name="mn10_as")
    net.eval()

    # 3. Trace the audio -> spectrogram -> logits pipeline with a dummy input
    pipeline = InnerPipeline(mel, net)
    pipeline.eval()
    dummy_input = torch.zeros(1, 1, segment_length, dtype=torch.float32)

    with torch.no_grad():
        traced_pipeline = torch.jit.trace(pipeline, dummy_input)

    # 4. Script the outer wrapper so C++ can call sample_rate(), segment_length(), class_names()
    wrapper = EfficientATWrapper(
        traced_pipeline=traced_pipeline,
        sr=sample_rate,
        seg_len=segment_length,
        classes=list(audioset_labels),
    )
    wrapper.eval()
    scripted_model = torch.jit.script(wrapper)

    # 5. Save into BOTH Resources/models and Contents/Resources/models so Partiels and tests find it
    for rel_dir in ["Resources/models", "Contents/Resources/models"]:
        out_dir = os.path.join(PROJECT_ROOT, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "01_mn10_as.ts")
        scripted_model.save(out_path)
        print(f"Saved TorchScript model to: {out_path}")


if __name__ == "__main__":
    main()
