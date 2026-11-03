"""LSTM that forecasts the next `n_predictions` telemetry values from the last `l_s` steps."""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 80, num_layers: int = 2,
                 dropout: float = 0.3, n_predictions: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_predictions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, l_s, F)
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1]))        # (B, n_predictions)
