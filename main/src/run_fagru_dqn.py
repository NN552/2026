from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_pricing_dataset, validate_feature_set
from deep_models import build_deep_model
from experiment_config import ExperimentConfig
from metrics import regression_metrics
from sequence_data import build_sequence_arrays, make_loaders


ACTION_VALUES = np.array([-0.02, -0.01, 0.0, 0.01, 0.02], dtype=np.float32)


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    config = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Train FA-GRU-DQN price calibrator.")
    parser.add_argument("--data", default=str(config.data_path))
    parser.add_argument("--results-dir", default=str(config.results_dir))
    parser.add_argument("--feature-set", default="feedback")
    parser.add_argument("--test-size", type=float, default=config.test_size)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fagru-epochs", type=int, default=80)
    parser.add_argument("--fagru-patience", type=int, default=12)
    parser.add_argument("--fagru-lr", type=float, default=1e-3)
    parser.add_argument("--dqn-epochs", type=int, default=120)
    parser.add_argument("--dqn-lr", type=float, default=1e-3)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_validation(X_train, y_train, validation_size: float = 0.15):
    split_index = int(len(X_train) * (1.0 - validation_size))
    return X_train[:split_index], X_train[split_index:], y_train[:split_index], y_train[split_index:]


def train_fagru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_dim: int,
    window_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    X_fit, X_val, y_fit, y_val = split_train_validation(X_train, y_train)
    train_loader, val_loader = make_loaders(
        X_fit,
        y_fit,
        X_val,
        y_val,
        batch_size=args.batch_size,
    )
    model = build_deep_model("fa_gru", window_size=window_size, feature_dim=feature_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.fagru_lr, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for _ in range(args.fagru_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                val_losses.append(criterion(model(X_batch), y_batch).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.fagru_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_scaled(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.zeros((len(X), 1), dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds.append(model(X_batch).detach().cpu().numpy())
    return np.vstack(preds).reshape(-1)


def make_rl_states(
    X_seq: np.ndarray,
    base_price: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    last_features = X_seq[:, -1, :]
    feature_index = {name: idx for idx, name in enumerate(feature_cols)}
    selected_names = [
        "quality_score",
        "timeliness_score",
        "market_demand",
        "risk_score",
    ]
    selected = [last_features[:, feature_index[name]] for name in selected_names if name in feature_index]
    price_change = np.concatenate([[0.0], np.diff(base_price) / np.maximum(np.abs(base_price[:-1]), 1e-6)])
    state_parts = [base_price, price_change] + selected
    return np.vstack(state_parts).T.astype(np.float32)


def reward_for_action(
    base_price: float,
    y_true: float,
    risk_score: float,
    action: float,
) -> float:
    adjusted = base_price * (1.0 + action)
    error = abs(adjusted - y_true)
    risk_penalty = 0.2 * abs(action) * max(risk_score, 0.0)
    action_penalty = 0.05 * abs(action)
    return float(-error - risk_penalty - action_penalty)


def train_dqn(
    states: np.ndarray,
    base_price: np.ndarray,
    y_true: np.ndarray,
    risk_values: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> DQN:
    state_mean = states.mean(axis=0, keepdims=True)
    state_std = states.std(axis=0, keepdims=True) + 1e-6
    states_norm = (states - state_mean) / state_std

    rewards = np.zeros((len(states_norm), len(ACTION_VALUES)), dtype=np.float32)
    for action_idx, action in enumerate(ACTION_VALUES):
        adjusted = base_price * (1.0 + action)
        error = np.abs(adjusted - y_true)
        rewards[:, action_idx] = (
            -error
            - 0.2 * np.abs(action) * np.maximum(risk_values, 0.0)
            - 0.05 * np.abs(action)
        )

    dqn = DQN(state_dim=states.shape[1], action_dim=len(ACTION_VALUES)).to(device)
    optimizer = torch.optim.AdamW(dqn.parameters(), lr=args.dqn_lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    states_tensor = torch.tensor(states_norm, dtype=torch.float32, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    indices = np.arange(len(states_norm))

    for _ in range(args.dqn_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            state_batch = states_tensor[batch_idx]
            reward_batch = rewards_tensor[batch_idx]
            prediction = dqn(state_batch)
            loss = criterion(prediction, reward_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    dqn.state_mean = state_mean
    dqn.state_std = state_std
    return dqn


def apply_dqn(dqn: DQN, states: np.ndarray, base_price: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    states_norm = (states - dqn.state_mean) / dqn.state_std
    with torch.no_grad():
        state_tensor = torch.tensor(states_norm, dtype=torch.float32, device=device)
        action_indices = torch.argmax(dqn(state_tensor), dim=1).cpu().numpy()
    actions = ACTION_VALUES[action_indices]
    adjusted = base_price * (1.0 + actions)
    return adjusted, actions


def choose_fixed_action(base_price: np.ndarray, y_true: np.ndarray) -> float:
    best_action = 0.0
    best_mae = float("inf")
    for action in ACTION_VALUES:
        adjusted = base_price * (1.0 + action)
        mae = float(np.mean(np.abs(adjusted - y_true)))
        if mae < best_mae:
            best_mae = mae
            best_action = float(action)
    return best_action


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(test_size=args.test_size)
    set_seed(config.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_pricing_dataset(args.data)
    feature_cols = validate_feature_set(args.feature_set)
    X_train, X_test, y_train, y_test, _, _, _, target_scaler = build_sequence_arrays(
        df=df,
        feature_cols=feature_cols,
        target_col=config.target_col,
        window_size=args.window_size,
        test_size=config.test_size,
    )

    fagru = train_fagru(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_dim=len(feature_cols),
        window_size=args.window_size,
        args=args,
        device=device,
    )

    X_fit, X_cal, y_fit, y_cal = split_train_validation(X_train, y_train)
    cal_base_scaled = predict_scaled(fagru, X_cal, args.batch_size, device)
    test_base_scaled = predict_scaled(fagru, X_test, args.batch_size, device)

    y_cal_true = target_scaler.inverse_transform(y_cal.reshape(-1, 1)).reshape(-1)
    y_test_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    cal_base = target_scaler.inverse_transform(cal_base_scaled.reshape(-1, 1)).reshape(-1)
    test_base = target_scaler.inverse_transform(test_base_scaled.reshape(-1, 1)).reshape(-1)

    cal_states = make_rl_states(X_cal, cal_base, feature_cols)
    test_states = make_rl_states(X_test, test_base, feature_cols)
    risk_idx = feature_cols.index("risk_score")
    cal_risk = X_cal[:, -1, risk_idx]
    test_risk = X_test[:, -1, risk_idx]

    dqn = train_dqn(
        states=cal_states,
        base_price=cal_base,
        y_true=y_cal_true,
        risk_values=cal_risk,
        args=args,
        device=device,
    )
    test_adjusted, test_actions = apply_dqn(dqn, test_states, test_base, device)
    fixed_action = choose_fixed_action(cal_base, y_cal_true)
    fixed_adjusted = test_base * (1.0 + fixed_action)

    base_metrics = regression_metrics(y_test_true, test_base)
    fixed_metrics = regression_metrics(y_test_true, fixed_adjusted)
    dqn_metrics = regression_metrics(y_test_true, test_adjusted)

    rows = [
        {"model": "FA-GRU", **base_metrics},
        {
            "model": "FA-GRU-Fixed",
            **fixed_metrics,
            "avg_abs_action": abs(fixed_action),
            "avg_action": fixed_action,
        },
        {
            "model": "FA-GRU-DQN",
            **dqn_metrics,
            "avg_abs_action": float(np.mean(np.abs(test_actions))),
            "avg_action": float(np.mean(test_actions)),
        },
    ]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "fagru_dqn_results.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8")

    prediction_path = results_dir / "fagru_dqn_predictions.csv"
    pd.DataFrame(
        {
            "sample_index": np.arange(len(y_test_true)),
            "y_true": y_test_true,
            "fagru_pred": test_base,
            "fixed_action": fixed_action,
            "fagru_fixed_pred": fixed_adjusted,
            "dqn_action": test_actions,
            "fagru_dqn_pred": test_adjusted,
        }
    ).to_csv(prediction_path, index=False, encoding="utf-8")

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved FA-GRU-DQN metrics to {metrics_path}")
    print(f"Saved FA-GRU-DQN predictions to {prediction_path}")


if __name__ == "__main__":
    main()
