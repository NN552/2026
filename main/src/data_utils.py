from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_config import FEATURE_SETS, TARGET_COL, TIME_COL


def load_pricing_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).sort_values(TIME_COL)
    return df.reset_index(drop=True)


def validate_feature_set(name: str) -> list[str]:
    if name not in FEATURE_SETS:
        valid = ", ".join(FEATURE_SETS)
        raise ValueError(f"Unknown feature set {name!r}. Valid values: {valid}")
    return FEATURE_SETS[name]


def time_series_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = df.dropna(subset=feature_cols + [target_col]).copy()
    split_index = int(len(clean) * (1.0 - test_size))
    if split_index <= 0 or split_index >= len(clean):
        raise ValueError("Invalid split. Adjust test_size or dataset length.")

    train = clean.iloc[:split_index]
    test = clean.iloc[split_index:]

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train[target_col]
    y_test = test[target_col]
    return X_train, X_test, y_train, y_test
