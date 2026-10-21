from __future__ import annotations

import torch
from torch import nn


class WindowMLP(nn.Module):
    def __init__(self, window_size: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalCNN(nn.Module):
    def __init__(self, feature_dim: int, hidden_channels: int = 64):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels + feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        last_features = x[:, -1, :]
        x = x.transpose(1, 2)
        encoded = self.encoder(x).squeeze(-1)
        return self.head(torch.cat([encoded, last_features], dim=-1))


class GRURegressor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        output, _ = self.gru(x)
        last_state = output[:, -1, :]
        last_features = x[:, -1, :]
        return self.head(torch.cat([last_state, last_features], dim=-1))


class FAGRURegressor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_logits = self.feature_attention(x)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attended_x = x * attention_weights

        output, _ = self.gru(attended_x)
        last_state = output[:, -1, :]
        last_features = attended_x[:, -1, :]
        return self.head(torch.cat([last_state, last_features], dim=-1))


def build_deep_model(
    model_name: str,
    window_size: int,
    feature_dim: int,
) -> nn.Module:
    if model_name == "mlp":
        return WindowMLP(window_size=window_size, feature_dim=feature_dim)
    if model_name == "cnn":
        return TemporalCNN(feature_dim=feature_dim)
    if model_name == "gru":
        return GRURegressor(feature_dim=feature_dim)
    if model_name == "fa_gru":
        return FAGRURegressor(feature_dim=feature_dim)
    raise ValueError(f"Unknown deep model: {model_name}")
