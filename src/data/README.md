# src/data/load_data.py

Data loading script.

## Purpose

`load_data.py` reads the raw Bank Marketing CSV file from the `data/` folder.

## Responsibility

This script only loads data. It does not clean, split, transform, or train anything.

## Pipeline Role

```text
data/bank-full.csv
        |
        v
src/data/load_data.py
        |
        v
src/preprocess/preprocessing.py
```

## Run

```powershell
python -m src.data.load_data
```

Expected output:

```text
Loaded 45211 rows and 17 columns.
```

