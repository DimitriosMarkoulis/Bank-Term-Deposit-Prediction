"""Classification metrics for model evaluation."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_positive_probabilities(model, X) -> list[float]:
    """Return predicted probabilities for the positive class `yes`."""

    yes_index = list(model.classes_).index("yes")
    return model.predict_proba(X)[:, yes_index]


def calculate_metrics(model, X, y) -> dict:
    """Calculate classification metrics for the positive class `yes`."""

    predictions = model.predict(X)
    probabilities = get_positive_probabilities(model, X)
    y_binary = (y == "yes").astype(int)
    pr_precision, pr_recall, _ = precision_recall_curve(y_binary, probabilities)

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, pos_label="yes", zero_division=0),
        "recall": recall_score(y, predictions, pos_label="yes", zero_division=0),
        "f1": f1_score(y, predictions, pos_label="yes", zero_division=0),
        "roc_auc": roc_auc_score(y_binary, probabilities),
        "average_precision": average_precision_score(y_binary, probabilities),
        "pr_auc": auc(pr_recall, pr_precision),
    }


def calculate_confusion_matrix(model, X, y) -> pd.DataFrame:
    """Calculate a labeled confusion matrix."""

    matrix = confusion_matrix(y, model.predict(X), labels=["no", "yes"])

    return pd.DataFrame(
        matrix,
        index=["actual_no", "actual_yes"],
        columns=["predicted_no", "predicted_yes"],
    )
