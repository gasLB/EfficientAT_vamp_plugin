# Efficient AT

## Model export

The plugin runs TorchScript versions of the [EfficientAT](https://github.com/fschmid56/EfficientAT) models pretrained on AudioSet (527 classes). The script [Resources/export_ts.py](Resources/export_ts.py) generates them:

- it downloads the EfficientAT sources (pinned to a commit) and the pretrained weights into the `.cache` directory,
- it exports the models `01_mn04_as.ts`, `02_mn10_as.ts` and `03_mn20_as.ts` into the `models` directory,
- it checks that each exported model gives the same results as the original EfficientAT model on the CPU and, if available, on the GPU (MPS on macOS).

The `models` and `.cache` directories are ignored by git.

### Requirements

- **Python 3.12**: the PyTorch version required by the plugin doesn't support more recent versions of Python. On macOS, it can be installed with `brew install python@3.12`.
- **PyTorch 2.4.1**: the version must match the LibTorch version used by the plugin (see `LIBTORCH_VERSION` in [Resources/libtorch.cmake](Resources/libtorch.cmake)), otherwise the exported models may not load in the plugin. The dependencies are listed in [Resources/requirements.txt](Resources/requirements.txt).
- An internet connection for the first run (about 100 MB of sources and weights).

### Setup

From the root of the repository, create a virtual environment and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r Resources/requirements.txt
```

### Export

```bash
source .venv/bin/activate
python Resources/export_ts.py
```

To export other models, edit the `MODELS` list of the script (the names are the keys of `pretrained_models` in the EfficientAT file `models/mn/model.py`).

### Use in the plugin

At configure time, CMake downloads the models of the [models release](https://github.com/gasLB/EfficientAT_vamp_plugin/releases/tag/models) into `<build directory>/models` (see [Resources/download_model.cmake](Resources/download_model.cmake)) and only downloads the files that are missing. To test locally exported models, copy them into `<build directory>/models` before building. To distribute them, upload them to the `models` release.
