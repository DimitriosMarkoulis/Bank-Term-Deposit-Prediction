# src/preprocess/preprocessing.py

Preprocessing pipeline for model training.

## Purpose

`preprocessing.py` prepares model-ready data before training.

## Pipeline

```text
raw data
  -> train/validation/test split
  -> feature engineering
  -> numeric preprocessing
  -> categorical preprocessing
  -> transformed model-ready data
```

## What It Does

- uses a stratified `0.6/0.2/0.2` train/validation/test split
- separates features from the target column
- converts `pdays = -1` into a valid numeric value
- creates `was_previously_contacted` from the original `pdays` value
- encodes binary features as `0/1`
- scales continuous numeric features with `StandardScaler`
- one-hot encodes multi-class categorical features
- keeps `"unknown"` as a valid category
- fits the transformer only on the training set

## Why This Matters

The transformer is fitted only on training data to avoid data leakage. Validation, test, and future prediction data must reuse the fitted transformer.

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

Future `training.py` should import this preprocessing pipeline instead of recreating split or transformation logic.
