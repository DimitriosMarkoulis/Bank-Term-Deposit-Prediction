"""Generate model predictions."""

import pandas as pd

from src.models.metrics import get_positive_probabilities
from src.models.results import load_model, save_predictions
from src.models.train import BASELINE_MODEL_ID
from src.preprocess.preprocessing import prepare_data


def build_predictions(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Create a prediction table for inspection."""

    return pd.DataFrame(
        {
            "row_id": X.index,
            "actual": y.values,
            "prediction": model.predict(X),
            "probability_yes": get_positive_probabilities(model, X),
        }
    )


def main() -> None:
    """Load the current model and generate test predictions."""

    data = prepare_data()
    model = load_model(BASELINE_MODEL_ID)
    predictions = build_predictions(model, data["X_test"], data["y_test"])
    predictions_path = save_predictions(predictions, BASELINE_MODEL_ID)

    print("Predictions generated.")
    print(predictions.head().round(4).to_string(index=False))
    print(f"\nPredictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
