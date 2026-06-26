# Bank Term Deposit Prediction

This is an end-to-end machine learning project that predicts whether a bank client is likely to subscribe to a term deposit after a marketing campaign.

The goal of this project is to build a complete ML workflow step by step, starting from data loading and exploratory analysis, then moving to model training, experiment tracking, hyperparameter tuning, API deployment, and containerization.

## Repository Structure

```text
bank-marketing-mlops/
|-- data/
|-- notebooks/
|   |-- 01_eda.ipynb
|-- src/
|   |-- config.py
|   |-- data/
|   |   |-- load_data.py
|   |-- preprocess/
|   |   |-- preprocessing.py
|   |-- models/
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
