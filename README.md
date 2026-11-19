# Efficient AT

The EfficientAT Vamp plugin is an implementation of [EfficientAT](https://github.com/fschmid56/EfficientAT), a set of pre-trained models ready for audio tagging, developed at [CPJKU](https://www.jku.at/) as a [Vamp Plugin](https://www.vamp-plugins.org/).

The EfficientAT Vamp Plugin has been designed for use in the free audio analysis application Partiels.

## Model export

The plugin runs TorchScript versions of the [EfficientAT](https://github.com/fschmid56/EfficientAT) models pretrained on AudioSet (527 classes). The script [Resources/export_models.py](Resources/export_ts.py) generates them:

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
python Resources/export_models.py
```

To export other models, edit the `MODELS` list of the script (the names are the keys of `pretrained_models` in the EfficientAT file `models/mn/model.py`).

### Use in the plugin

At configure time, CMake downloads the models of the [models release](https://github.com/gasLB/EfficientAT_vamp_plugin/releases/tag/models) into `<build directory>/models` (see [Resources/download_models.cmake](Resources/download_models.cmake)) and only downloads the files that are missing. To test locally exported models, copy them into `<build directory>/models` before building. To distribute them, upload them to the `models` release.

## Compilation

The compilation system is based on [CMake](https://cmake.org/), for example:
```
cmake . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest -C Debug -VV --test-dir build
```

## Credits

- **[EfficientAT Vamp Plugin](https://www.ircam.fr/)** by Gaston Fontagneu, Pierre Guillot at IRCAM IMR Department.
- **[EfficientAT](https://github.com/fschmid56/EfficientAT)** models by Florian Schmid, Khaled Koutini and Gerhard Widmer at Institute of Computational Perception of Johannes Kepler University (https://www.jku.at/)
- **[LibTorch](https://github.com/pytorch/pytorch)** originally developed by Meta Platforms. 
- **[Vamp SDK](https://github.com/vamp-plugins/vamp-plugin-sdk)** by Chris Cannam, copyright (c) 2005-2024 Chris Cannam and Centre for Digital Music, Queen Mary, University of London.
- **[Ircam Vamp Extension](https://github.com/Ircam-Partiels/ircam-vamp-extension)** by Pierre Guillot at [IRCAM IMR department](https://www.ircam.fr/).
