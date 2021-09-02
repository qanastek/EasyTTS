#!/usr/bin/env python3
import os
import setuptools
from distutils.core import setup

with open("README.md") as f:
    long_description = f.read()

with open(os.path.join("EasyTTS", "version.txt")) as f:
    version = f.read().strip()

setup(
    name = "EasyTTS",
    version = version,
    description = "Ready-to-use Multilingual Text-To-Speech (TTS) package.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author = "Yanis Labrak & Others",
    author_email = "yanis.labrak@univ-avignon.fr",
    packages = setuptools.find_packages(),
    package_data = {
        "EasyTTS": [
            "version.txt"
        ]
    },
    install_requires = [
        "numpy",
        "timm",
        "torch",
        "tensorflow",
        "soundfile",
        "TensorFlowTTS",
    ],
    python_requires = ">=3.7",
    url = "https://EasyTTS.github.io/",
    keywords = ["python","transformers","huggingface","wrapper","toolkit","speech","text-to-speech","text2speech","text-2-speech","T2S","easy","voice","vocal synthesis","synthesis","Speech synthesis"],
)