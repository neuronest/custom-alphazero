#!/usr/bin/env bash

conda remove --name alphazero --all
conda create --name alphazero python=3.7
conda activate alphazero

pip install --upgrade pip
pip install -Ur requirements.txt

pre-commit install