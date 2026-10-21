from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from experiment_config import TARGET_COL


def build_sequence_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    window_size: int = 24,
    test_size: float = 0.2,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler,
]:
    clean = df.dropna(subset=feature_cols + [target_col]).copy()
    split_index = int(len(clean) * (1.0 - test_size))
    if split_index <= window_size or split_index >= len(clean):
        raise ValueError("Invalid split. Adjust test_size, window_size, or dataset length.")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(clean.iloc[:split_index][feature_cols])
    target_scaler.fit(clean.iloc[:split_index][[target_col]])

    X_scaled = feature_scaler.transform(clean[feature_cols]).astype(np.float32)
    y_scaled = target_scaler.transform(clean[[target_col]]).reshape(-1).astype(np.float32)

    X_seq = []
    y_seq = []
    baseline_seq = []
    target_indices = []
    for target_idx in range(window_size, len(clean)):
        start = target_idx - window_size + 1
        X_seq.append(X_scaled[start : target_idx + 1])
        y_seq.append(y_scaled[target_idx])
        baseline_seq.append(float(np.mean(y_scaled[target_idx - window_size : target_idx])))
        target_indices.append(target_idx)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)
    baseline_seq = np.asarray(baseline_seq, dtype=np.float32)
    target_indices = np.asarray(target_indices)

    train_mask = target_indices < split_index
    test_mask = target_indices >= split_index

    return (
        X_seq[train_mask],
        X_seq[test_mask],
        y_seq[train_mask],
        y_seq[test_mask],
        baseline_seq[train_mask],
        baseline_seq[test_mask],
        feature_scaler,
        target_scaler,
    )


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(-1),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).unsqueeze(-1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def make_residual_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    baseline_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_test: np.ndarray,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(-1),
        torch.from_numpy(baseline_train).unsqueeze(-1),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).unsqueeze(-1),
        torch.from_numpy(baseline_test).unsqueeze(-1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
