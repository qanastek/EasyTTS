<p align="center">
  <img src="https://raw.githubusercontent.com/qanastek/EasyTTS/main/ressources/images/logo_name_transparent.png" alt="drawing" width="250"/>
</p>

[![PyPI version](https://badge.fury.io/py/EasyTTS.svg)](https://badge.fury.io/py/EasyTTS)
[![GitHub Issues](https://img.shields.io/github/issues/qanastek/EasyTTS.svg)](https://github.com/qanastek/EasyTTS/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

EasyTTS is an open-source and easy to use all-in-one huggingface wrapper for computer vision.

The goal is to create a fast, flexible and user-friendly toolkit that can be used to easily develop **state-of-the-art** computer vision technologies, including systems for Image Classification, Semantic Segmentation, Object Detection, Image Generation, Denoising and much more.

⚠️ EasyTTS is currently in beta. ⚠️

# Quick installation

EasyTTS is constantly evolving. New features, tutorials, and documentation will appear over time. EasyTTS can be installed via PyPI to rapidly use the standard library. Moreover, a local installation can be used by those users than want to run experiments and modify/customize the toolkit. EasyTTS supports both CPU and GPU computations. For most recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.

## Anaconda setup

```bash
conda create --name EasyTTS python=3.7 -y
conda activate EasyTTS
```

More information on managing environments with Anaconda can be found in [the conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

## Install via PyPI

Once you have created your Python environment (Python 3.7+) you can simply type:

```bash
pip install EasyTTS
```

## Install with GitHub

Once you have created your Python environment (Python 3.7+) you can simply type:

```bash
git clone https://github.com/qanastek/EasyTTS.git
cd EasyTTS
pip install -r requirements.txt
pip install --editable .
```

Any modification made to the `EasyTTS` package will be automatically interpreted as we installed it with the `--editable` flag.

# Example Usage

```python
from EasyTTS.inference import TTS

audio = TTS(lang="fr", text="Bonjour à tous")

sf.write('./audio.wav', audio, 22050, "PCM_16") # Save to a .WAV file
```

# Model architectures

1. **[Tacotron 2](https://github.com/NVIDIA/tacotron2)** (from Google Research &  University of California, Berkeley) released with the paper [NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM PREDICTIONS](https://arxiv.org/pdf/1712.05884.pdf), by Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis and Yonghui Wu.

# Datasets

1. **[SynPaFlex](http://synpaflex.irisa.fr/fr/)** (from IRISA, LLF (Laboratoire de Linguistique Formelle de Nantes), LIUM (Le Mans Université) and ATILF (Analyse et Traitement Informatique de la Langue Française)) released with the paper [SynPaFlex-Corpus: An Expressive French Audiobooks Corpus Dedicated to Expressive Speech Synthesis](https://hal.archives-ouvertes.fr/hal-01826690), by Aghilas Sini, Damien Lolive, Gaëlle Vidal, Marie Tahon and Élisabeth Delais-Roussarie.

# Build PyPi package

Build: `python setup.py sdist bdist_wheel`

Upload: `twine upload dist/*`