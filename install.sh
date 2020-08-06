#!/usr/bin/env bash

rm -rf ./env

conda create -p ./env python=3.7.7
conda activate ./env

pip install --upgrade pip
pip install -Ur requirements.txt
./env/bin/pre-commit install
