"""Preprocessing utilities for model training."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    validate_split_sizes,
)
from src.data.load_data import load_data


def split_train_val_test(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into stratified train, validation, and test sets."""

    validate_split_sizes()

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    val_fraction = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Handle dataset-specific feature rules before sklearn preprocessing."""

    X = X.copy()
    X["was_previously_contacted"] = (X["pdays"] != -1).astype(int)
    X["pdays"] = X["pdays"].replace(-1, 0)
    for column in ["default", "housing", "loan"]:
        X[column] = X[column].map({"no": 0, "yes": 1}).astype(int)
    return X


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Create the numeric and categorical preprocessor."""

    binary_features = ["default", "housing", "loan", "was_previously_contacted"]
    numeric_features = [
        column
        for column in X_train.select_dtypes(include="number").columns
        if column not in binary_features
    ]
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("binary", "passthrough", binary_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )


def prepare_data(data: pd.DataFrame | None = None) -> dict:
    """Load, split, and engineer features without fitting a model."""

    source_data = load_data() if data is None else data
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(source_data)

    X_train = engineer_features(X_train)
    X_val = engineer_features(X_val)
    X_test = engineer_features(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def fit_preprocessor(data: pd.DataFrame | None = None) -> dict:
    """Fit the preprocessor on the training features only."""

    prepared_data = prepare_data(data)
    preprocessor = build_preprocessor(prepared_data["X_train"])
    preprocessor.fit(prepared_data["X_train"])

    return {
        "preprocessor": preprocessor,
        **prepared_data,
    }


def transform_to_dataframe(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Transform features and keep readable column names."""

    return pd.DataFrame(
        preprocessor.transform(X),
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )


def main() -> None:
    """Run preprocessing and print a compact preview."""

    result = fit_preprocessor()
    X_train_transformed = transform_to_dataframe(
        result["preprocessor"],
        result["X_train"],
    )

    print("Preprocessing pipeline fitted.")
    print(
        "Rows "
        f"train={len(result['X_train'])}, "
        f"val={len(result['X_val'])}, "
        f"test={len(result['X_test'])}"
    )
    print(f"Transformed train shape: {X_train_transformed.shape}")
    print(X_train_transformed.iloc[:, :10].head().to_string())


if __name__ == "__main__":
    main()
