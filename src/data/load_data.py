"""Load the Bank Marketing dataset."""

from pathlib import Path
import pandas as pd

from src.config import DATA_PATH


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Read the semicolon-delimited Bank Marketing CSV file."""

    data_path = DATA_PATH if path is None else Path(path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return pd.read_csv(data_path, sep=";")


if __name__ == "__main__":
    dataset = load_data()
    print(f"Loaded {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
