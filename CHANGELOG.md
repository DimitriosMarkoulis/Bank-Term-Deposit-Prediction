# Changelog

All notable changes to this project will be documented in this file.

## v2.0-preprocessing-analysis

Version 2 adds preprocessing analysis.

Added:

- preprocessing module in `src/preprocess/preprocessing.py`
- shared configuration in `src/config.py`
- script-level README files for configuration, data loading, and preprocessing
- stratified train/validation/test split with 0.6/0.2/0.2
- `pdays` feature handling with `was_previously_contacted`
- binary feature encoding as 0/1
- continuous numeric scaling
- multi-class categorical one-hot encoding

## v1.0-eda

Version 1 adds data loading and exploratory analysis.

Added:

- `src/data/load_data.py`
- exploratory notebook in `notebooks/01_eda.ipynb`
- EDA summary in `notebooks/01_eda_analysis.md`

## v0.0-project-setup

Version 0 creates the project foundation.

Added:

- clean folder structure
- `README.md` project overview and roadmap
- `CHANGELOG.md` version history
- `requirements.txt` placeholder for future dependencies
- `.gitignore` for local environment and generated files
