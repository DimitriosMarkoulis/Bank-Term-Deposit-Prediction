# Bank Term Deposit Prediction

This is an end-to-end machine learning project that predicts whether a bank client is likely to subscribe to a term deposit after a marketing campaign.

The goal of this project is to build a complete ML workflow step by step, starting from data loading and exploratory analysis, then moving to model training, experiment tracking, hyperparameter tuning, API deployment, and containerization.

## Problem Statement

Banks run marketing campaigns by contacting existing clients, but outreach has a cost: time, call-center capacity, and customer fatigue. The objective is to predict which clients are more likely to subscribe to a term deposit so the campaign can prioritize higher-potential leads.

This is an imbalanced binary classification problem:

```text
yes -> client subscribed to a term deposit
no  -> client did not subscribe
```

Because the positive class is relatively rare, the project focuses on metrics such as average precision, PR-AUC, recall, and precision instead of accuracy alone.

## Current Baseline Findings

Version 3 uses Logistic Regression as the baseline model.

| Split | Accuracy | Precision | Recall | Average Precision | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Train | 0.8440 | 0.4155 | 0.8216 | 0.5505 | 0.5503 |
| Validation | 0.8446 | 0.4161 | 0.8138 | 0.5481 | 0.5471 |
| Test | 0.8456 | 0.4184 | 0.8185 | 0.5369 | 0.5359 |

The train, validation, and test scores are close, so the baseline does not show strong overfitting.

## Business Interpretation

The baseline model has high recall on the test set, meaning it identifies many clients who actually subscribe. Precision is lower, meaning the model still produces false positives. In business terms, this model is more useful for broad lead prioritization than for highly selective targeting.

The next modeling step is to compare additional algorithms and tune the decision threshold based on the campaign goal: higher recall for wider outreach, or higher precision for a smaller and more selective contact list.

## Repository Structure

```text
bank-marketing-mlops/
|-- data/
|-- results/
|-- notebooks/
|   |-- 01_eda.ipynb
|-- src/
|   |-- config/
|   |   |-- config.py
|   |-- data/
|   |   |-- load_data.py
|   |-- preprocess/
|   |   |-- preprocessing.py
|   |-- models/
|   |   |-- train.py
|   |   |-- predict.py
|   |   |-- metrics.py
|   |   |-- model_pipeline.py
|   |   |-- results.py
|-- CHANGELOG.md
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Versioning

The project uses:

```text
README.md       -> project overview and roadmap
CHANGELOG.md    -> version-by-version changes
Git commits     -> code history
Git tags        -> stable milestones
```

## Version 1

```powershell
python -m src.data.load_data
```

Open `notebooks/01_eda.ipynb` for exploratory data analysis.

## Version 2

```powershell
python -m src.preprocess.preprocessing
```

The preprocessing script prints the split sizes, transformed training shape, and a compact preview of the transformed training data.

## Version 3

```powershell
python -m src.models.train
```

Version 3 trains and evaluates a Logistic Regression baseline model with `max_iter=1000` and `class_weight="balanced"`.

The training run saves local results:

```text
results/<model_id>/model.joblib
results/<model_id>/results.json
results/<model_id>/results.md
```

For Version 3, the current `model_id` is `logistic_regression`.

The model artifact is generated locally and ignored by Git.

```powershell
python -m src.models.predict
```

Use `predict.py` after training to load the saved model and generate test-set predictions for inspection.

The prediction run saves a local CSV file:

```text
results/<model_id>/test_predictions.csv
```

Prediction CSV files are local inspection artifacts and are ignored by Git.
