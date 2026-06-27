# src

Source code for the project pipeline.

## Purpose

The `src` folder separates the project into configuration, data loading, preprocessing, training, metrics, prediction, and result saving.

## Files

```text
src/
|-- config/
|   |-- config.py
|-- data/
|   |-- load_data.py
|-- preprocess/
|   |-- preprocessing.py
|-- models/
|   |-- train.py
|   |-- predict.py
|   |-- metrics.py
|   |-- model_pipeline.py
|   |-- results.py
```

## Configuration

`config/config.py` keeps shared constants in one place:

- project root
- data folder
- results folder
- raw dataset path
- target column
- train/validation/test split sizes
- random state

## Why It Exists

Data loading, preprocessing, training, and prediction should use the same basic settings. This avoids repeated values across scripts.

## Main Flow

```text
load_data.py
  -> preprocessing.py
  -> model_pipeline.py
  -> train.py
  -> predict.py
```

`metrics.py` is reused by model scripts for evaluation.
`results.py` is reused by model scripts for saving local artifacts.
