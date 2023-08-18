#!/bin/bash

pyenv virtualenv 3.8 intent-inference
pyenv local intent-inference

pip install --upgrade pip

poetry install
