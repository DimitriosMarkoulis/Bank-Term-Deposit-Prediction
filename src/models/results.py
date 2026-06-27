"""Save model result artifacts."""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.config.config import RESULTS_DIR


def get_model_results_dir(model_id: str) -> Path:
    """Return the results folder for one model."""

    return RESULTS_DIR / model_id


def get_results_json_path(model_id: str) -> Path:
    """Return the JSON results path for one model."""

    return get_model_results_dir(model_id) / "results.json"


def get_results_markdown_path(model_id: str) -> Path:
    """Return the Markdown results path for one model."""

    return get_model_results_dir(model_id) / "results.md"


def get_predictions_path(model_id: str) -> Path:
    """Return the prediction artifact path for one model."""

    return get_model_results_dir(model_id) / "test_predictions.csv"


def get_model_artifact_path(model_id: str) -> Path:
    """Return the trained model artifact path for one model."""

    return get_model_results_dir(model_id) / "model.joblib"


def build_markdown_report(report: dict) -> str:
    """Build a readable Markdown report from model results."""

    lines = [
        f"# {report['model']} Results",
        "",
        "## Model",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Model ID | {report['model_id']} |",
        f"| Model | {report['model']} |",
        f"| Positive class | {report['positive_class']} |",
    ]

    for parameter, value in report["hyperparameters"].items():
        lines.append(f"| {parameter} | {value} |")

    lines.extend(["", "## Metrics", ""])

    metric_names = list(next(iter(report["metrics"].values())).keys())
    header = "| Split | " + " | ".join(metric_names) + " |"
    separator = "|---|" + "|".join("---:" for _ in metric_names) + "|"
    lines.extend([header, separator])

    for split_name, split_metrics in report["metrics"].items():
        values = [f"{split_metrics[metric_name]:.4f}" for metric_name in metric_names]
        lines.append("| " + split_name + " | " + " | ".join(values) + " |")

    lines.extend(["", "## Confusion Matrices"])

    for split_name, matrix in report["confusion_matrices"].items():
        row_labels = list(matrix.keys())
        column_labels = list(next(iter(matrix.values())).keys())
        lines.extend(
            [
                "",
                f"### {split_name.title()}",
                "",
                "| Actual / Predicted | " + " | ".join(column_labels) + " |",
                "|---|" + "|".join("---:" for _ in column_labels) + "|",
            ]
        )

        for actual_label in row_labels:
            predicted_values = matrix[actual_label]
            values = [str(predicted_values[label]) for label in column_labels]
            lines.append(
                "| " + actual_label + " | " + " | ".join(values) + " |"
            )

    return "\n".join(lines) + "\n"


def save_results_report(
    report: dict,
    model_id: str | None = None,
) -> dict[str, Path]:
    """Save model results as JSON and Markdown files."""

    model_id = report["model_id"] if model_id is None else model_id
    json_path = get_results_json_path(model_id)
    markdown_path = get_results_markdown_path(model_id)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")

    return {
        "json": json_path,
        "markdown": markdown_path,
    }


def save_predictions(
    predictions: pd.DataFrame,
    model_id: str,
) -> Path:
    """Save model predictions as a CSV file."""

    path = get_predictions_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path


def save_model(model, model_id: str) -> Path:
    """Save a trained model pipeline."""

    path = get_model_artifact_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_id: str):
    """Load a trained model pipeline."""

    path = get_model_artifact_path(model_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}. "
            "Run `python -m src.models.train` first."
        )

    return joblib.load(path)
