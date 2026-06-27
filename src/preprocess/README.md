# src/preprocess/preprocessing.py

Preprocessing pipeline for model training.

## Purpose

`preprocessing.py` prepares model-ready data before training.

## Pipeline

```text
raw data
  -> train/validation/test split
  -> feature engineering
  -> preprocessor definition
  -> model pipeline
```

## Main Functions

- `prepare_data()` loads, splits, and engineers features.
- `build_preprocessor()` defines numeric, binary, and categorical preprocessing.
- `fit_preprocessor()` is used for preprocessing inspection only.
- `transform_to_dataframe()` converts transformed arrays back to named columns.

## What It Does

- uses a stratified `0.6/0.2/0.2` train/validation/test split
- separates features from the target column
- converts `pdays = -1` into a valid numeric value
- creates `was_previously_contacted` from the original `pdays` value
- encodes binary features as `0/1`
- scales continuous numeric features with `StandardScaler`
- one-hot encodes multi-class categorical features
- keeps `"unknown"` as a valid category
- provides a preprocessor that model pipelines fit only on training data

## What It Does Not Do

- it does not train models
- it does not calculate model metrics
- it does not save model artifacts

## Why This Matters

The preprocessor is fitted only on training data inside each model pipeline. This avoids data leakage and lets future scripts swap in different algorithms.

## Run

```powershell
python -m src.preprocess.preprocessing
```

Expected output:

```text
Preprocessing pipeline fitted.
Rows train=27126, val=9042, test=9043
Transformed train shape: (27126, 49)
```

## Future Feature Ideas

These are not implemented yet, but they are good candidates to test later:

- train two model variants: with `duration` and without `duration`
- create age bands for non-linear age effects
- create balance bands or a signed log balance feature
- cap extreme values in `campaign`, `previous`, and `balance`
- add month seasonality features if one-hot month is not enough
- add loan exposure features from `housing`, `loan`, and `default`

## Used Later By

Future model scripts should import this preprocessing pipeline instead of recreating split or transformation logic.
