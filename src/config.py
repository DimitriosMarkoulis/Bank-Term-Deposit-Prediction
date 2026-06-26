"""Project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "bank-full.csv"

TARGET_COLUMN = "y"
TRAIN_SIZE = 0.60
VAL_SIZE = 0.20
TEST_SIZE = 0.20
RANDOM_STATE = 42


def validate_split_sizes() -> None:
    """Validate train/validation/test sizes."""

    split_total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if round(split_total, 10) != 1.0:
        raise ValueError("TRAIN_SIZE, VAL_SIZE, and TEST_SIZE must sum to 1.0")
