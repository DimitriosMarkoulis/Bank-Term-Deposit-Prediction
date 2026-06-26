# src/config.py

Central project settings used by the pipeline.

## Purpose

`config.py` keeps shared constants in one place:

- project root
- data folder
- raw dataset path
- target column
- train/validation/test split sizes
- random state

## Why It Exists

Data loading, preprocessing, training, and prediction should use the same basic settings. This avoids repeated values across scripts.

## Used By

- `src/data/load_data.py`
- `src/preprocess/preprocessing.py`
- future training and prediction scripts

