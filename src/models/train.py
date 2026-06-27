"""Train and evaluate models."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.metrics import calculate_confusion_matrix, calculate_metrics
from src.models.model_pipeline import build_model_pipeline
from src.models.results import save_model, save_results_report
from src.preprocess.preprocessing import prepare_data

BASELINE_MODEL_ID = "logistic_regression"

MODEL_CONFIGS = {
    "logistic_regression": {
        "model_name": "LogisticRegression",
        "estimator": LogisticRegression,
        "hyperparameters": {
            "max_iter": 1000,
            "class_weight": "balanced",
        },
    },
}


def train_model(data: dict, model_id: str = BASELINE_MODEL_ID):
    """Train one model by ID."""

    model_config = MODEL_CONFIGS[model_id]
    model = model_config["estimator"](**model_config["hyperparameters"])
    pipeline = build_model_pipeline(model, data["X_train"])
    pipeline.fit(data["X_train"], data["y_train"])
    return pipeline


def train_logistic_regression(data: dict):
    """Train the Logistic Regression baseline model."""

    return train_model(data, BASELINE_MODEL_ID)


def build_results_report(
    model,
    data: dict,
    model_id: str = BASELINE_MODEL_ID,
) -> dict:
    """Build a JSON-serializable evaluation report."""

    model_config = MODEL_CONFIGS[model_id]
    splits = {
        "train": ("X_train", "y_train"),
        "validation": ("X_val", "y_val"),
        "test": ("X_test", "y_test"),
    }

    metrics = {}
    confusion_matrices = {}

    for split_name, (features_key, target_key) in splits.items():
        split_metrics = calculate_metrics(
            model,
            data[features_key],
            data[target_key],
        )
        split_confusion_matrix = calculate_confusion_matrix(
            model,
            data[features_key],
            data[target_key],
        )

        metrics[split_name] = {
            metric_name: round(float(metric_value), 4)
            for metric_name, metric_value in split_metrics.items()
        }
        confusion_matrices[split_name] = {
            row_name: {
                column_name: int(value)
                for column_name, value in row_values.items()
            }
            for row_name, row_values in split_confusion_matrix.to_dict(
                orient="index"
            ).items()
        }

    return {
        "model_id": model_id,
        "model": model_config["model_name"],
        "positive_class": "yes",
        "hyperparameters": model_config["hyperparameters"],
        "metrics": metrics,
        "confusion_matrices": confusion_matrices,
    }


def main() -> None:
    """Run training and print metrics."""

    data = prepare_data()
    model = train_model(data, BASELINE_MODEL_ID)
    report = build_results_report(model, data, BASELINE_MODEL_ID)
    model_path = save_model(model, BASELINE_MODEL_ID)
    result_paths = save_results_report(report)

    results = pd.DataFrame(
        [
            {"split": split_name, **split_metrics}
            for split_name, split_metrics in report["metrics"].items()
        ]
    )

    print("Logistic Regression training complete.")
    print(results.round(4).to_string(index=False))
    print("\nValidation confusion matrix:")
    print(calculate_confusion_matrix(model, data["X_val"], data["y_val"]))
    print("\nTest confusion matrix:")
    print(calculate_confusion_matrix(model, data["X_test"], data["y_test"]))
    print(f"\nModel artifact saved to: {model_path}")
    print(f"\nJSON results saved to: {result_paths['json']}")
    print(f"Markdown results saved to: {result_paths['markdown']}")


if __name__ == "__main__":
    main()
