#!/bin/bash

pyenv virutalenv 3.8 intent-inference
pyenv local intent-inference

pip install --upgrade pip

poetry install
