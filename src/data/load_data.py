"""Load the Bank Marketing dataset."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the semicolon-delimited Bank Marketing CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, sep=";")


if __name__ == "__main__":
    dataset = load_data()
    print(f"Loaded {dataset.shape[0]} rows and {dataset.shape[1]} columns.")