# Bank Marketing EDA Analysis

## Dataset Overview

- Rows: 45,211
- Columns: 17
- Target column: `y`
- Missing values: 0
- Duplicate rows: 0

The dataset is clean from a missing-value and duplicate-row perspective. Most work in later versions should focus on feature understanding, class imbalance, preprocessing, and avoiding leakage.

## Target Distribution

| Class | Count | Share |
|---|---:|---:|
| `no` | 39,922 | 88.30% |
| `yes` | 5,289 | 11.70% |

This is an imbalanced binary classification problem. Accuracy alone will not be enough for evaluation because a model can look accurate by mostly predicting `no`.

## Numeric Feature Notes

| Feature | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| `age` | 40.94 | 39 | 18 | 95 |
| `balance` | 1362.27 | 448 | -8019 | 102127 |
| `duration` | 258.16 | 180 | 0 | 4918 |
| `campaign` | 2.76 | 2 | 1 | 63 |
| `pdays` | 40.20 | -1 | -1 | 871 |
| `previous` | 0.58 | 0 | 0 | 275 |

`balance`, `duration`, `campaign`, `pdays`, and `previous` are skewed and contain large outliers. Later preprocessing should consider scaling for linear models and robust validation for outlier-sensitive methods.

## Target-Level Differences

Average values by target:

| Feature | `no` mean | `yes` mean |
|---|---:|---:|
| `balance` | 1303.71 | 1804.27 |
| `duration` | 221.18 | 537.29 |
| `campaign` | 2.85 | 2.14 |
| `previous` | 0.50 | 1.17 |

Clients who subscribed had higher average balances, longer call durations, fewer campaign contacts, and more previous contacts.

Important caution: `duration` is only known after a call happens. It can be useful for analysis, but it may cause data leakage if the goal is to predict before calling a client.

## Categorical Feature Notes

Most common categories:

| Feature | Most common value | Share |
|---|---|---:|
| `job` | `blue-collar` | 21.53% |
| `marital` | `married` | 60.19% |
| `education` | `secondary` | 51.32% |
| `housing` | `yes` | 55.58% |
| `contact` | `cellular` | 64.77% |
| `month` | `may` | 30.45% |
| `poutcome` | `unknown` | 81.75% |

`poutcome` is mostly `unknown`, which likely means many clients had no previous campaign outcome. This feature should be handled carefully during preprocessing.

## High-Level Patterns

Highest subscription rates by selected categories:

- `poutcome = success`: 64.73%
- `month = mar`: 51.99%
- `month = dec`: 46.73%
- `job = student`: 28.68%
- `job = retired`: 22.79%
- `education = tertiary`: 15.01%
- `contact = cellular`: 14.92%

Lower subscription rates appear for:

- `housing = yes`: 7.70%
- `loan = yes`: 6.68%
- `contact = unknown`: 4.07%

## Modeling Implications

- Use stratified train, validation, and test splits because the target is imbalanced.
- Prefer evaluation metrics such as precision, recall, F1, ROC AUC, and PR AUC.
- Encode categorical variables before modeling.
- Treat `duration` as a special feature because it may not be available for pre-call prediction.
- Keep the test set untouched until final evaluation.
