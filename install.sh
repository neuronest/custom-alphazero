#!/usr/bin/env bash

rm -rf .ipynb_checkpoints

conda remove --name alphazero --all
conda create --name alphazero python=3.7
conda activate alphazero
conda install ipython
conda install jupyter

pip install -Ur requirements.txt
pre-commit install
