# LogisticRegression Results

## Model

| Setting | Value |
|---|---|
| Model ID | logistic_regression |
| Model | LogisticRegression |
| Positive class | yes |
| max_iter | 1000 |
| class_weight | balanced |

## Metrics

| Split | accuracy | precision | recall | f1 | roc_auc | average_precision | pr_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.8440 | 0.4155 | 0.8216 | 0.5519 | 0.9107 | 0.5505 | 0.5503 |
| validation | 0.8446 | 0.4161 | 0.8138 | 0.5507 | 0.9127 | 0.5481 | 0.5471 |
| test | 0.8456 | 0.4184 | 0.8185 | 0.5537 | 0.9079 | 0.5369 | 0.5359 |

## Confusion Matrices

### Train

| Actual / Predicted | predicted_no | predicted_yes |
|---|---:|---:|
| actual_no | 20286 | 3667 |
| actual_yes | 566 | 2607 |

### Validation

| Actual / Predicted | predicted_no | predicted_yes |
|---|---:|---:|
| actual_no | 6776 | 1208 |
| actual_yes | 197 | 861 |

### Test

| Actual / Predicted | predicted_no | predicted_yes |
|---|---:|---:|
| actual_no | 6781 | 1204 |
| actual_yes | 192 | 866 |
