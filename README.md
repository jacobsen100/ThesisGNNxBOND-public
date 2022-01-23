# Speciale_GNNxBONDS
Prediction of bonds using Graph Neural Networks

Connect to ThinLinc directory: 
From Finder select Go | Connect to Server (⌘K) and enter the following as the Server Address:
``smb://home.cc.dtu.dk/<username>``

## Install src
``pip install -e .``

## Data rules
Dimension 0 (rows): timesteps
Dimension 1 (cols): different timeseries

Makes sense because normalizing data normally happens on axis 0, so each timeseries follows axis 0 here. 

## Adding model to setup:
- Add model file in `src/model`
- Import model name in `src/model/__init__.py`
- Add model configs in `src/config/model` (create net file for each model)
- Add model import in `src/utils/builders.py` -> `model_builder()`
- To run experiment, select model name in `src/config/config.yaml` and bingo