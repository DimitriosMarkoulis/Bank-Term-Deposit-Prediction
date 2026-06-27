# src/models

Model training and evaluation scripts.

## Purpose

The `models` folder contains scripts that train and evaluate algorithms. These scripts reuse preprocessing instead of rebuilding data preparation logic.

## Pipeline

```text
raw train/validation/test features
  -> shared preprocessing
  -> model-specific algorithm
  -> evaluate on train, validation, and test data
```

## Shared Files

- `model_pipeline.py` builds a full sklearn pipeline from preprocessing plus any model.
- `metrics.py` calculates classification metrics and confusion matrices.
- `results.py` saves and loads model artifacts, reports, and prediction files under a folder based on the model ID from `train.py`.

`metrics.py` reports:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- average precision
- PR-AUC
- confusion matrix

## Training

`train.py` trains the Version 3 model: Logistic Regression with `max_iter=1000` and `class_weight="balanced"`.

In Version 4, `MODEL_CONFIGS` can be extended with additional naive algorithms such as Decision Tree, Random Forest, and MLP.

The main model metrics are average precision and PR-AUC because the target is imbalanced.

The script saves the trained model and model results:

```text
results/<model_id>/model.joblib
results/<model_id>/results.json
results/<model_id>/results.md
```

For Version 3, the current `model_id` is `logistic_regression`.

The model artifact is used by `predict.py`. The JSON file is useful for programmatic comparison. The Markdown file is useful for reading and documenting the experiment.

Model artifacts are generated locally and ignored by Git.

Run:

```powershell
python -m src.models.train
```

## Prediction

`predict.py` loads the saved model artifact and generates predictions on the test split.

It saves a local CSV artifact:

```text
results/<model_id>/test_predictions.csv
```

Prediction CSV files are useful for local inspection but are ignored by Git.

The prediction file contains:

- `row_id`
- `actual`
- `prediction`
- `probability_yes`

Run:

```powershell
python -m src.models.predict
```

## Separation of Concerns

Model scripts choose the algorithm. Preprocessing remains in `src/preprocess/preprocessing.py`. Results saving remains in `src/models/results.py`.
