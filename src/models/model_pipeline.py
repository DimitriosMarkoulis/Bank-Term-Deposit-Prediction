"""Reusable model pipeline helpers."""

from sklearn.pipeline import Pipeline

from src.preprocess.preprocessing import build_preprocessor


def build_model_pipeline(model, X_train):
    """Attach preprocessing to any sklearn-compatible model."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ]
    )
