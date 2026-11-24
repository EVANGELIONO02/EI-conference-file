"""Fixed bidirectional LSTM forecaster using historical power only."""
from __future__ import annotations
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Predict future power channels from an observed history window.

    Weather and wind-regime labels are intentionally excluded. The encoder is
    always bidirectional, operating only within the available history window.
    """
    def __init__(self, forecast_steps=12, hidden_dim=128,
                 layers=2, dropout=0.1, power_channels=1):
        super().__init__()
        self.forecast_steps = forecast_steps
        if power_channels < 1:
            raise ValueError("power_channels must be positive")
        self.power_channels = power_channels
        self.encoder = nn.LSTM(
            power_channels, hidden_dim, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True, bidirectional=True,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps * power_channels),
        )

    def forward(self, power_history):
        encoded, _ = self.encoder(power_history)
        return self.decoder(encoded[:, -1, :]).view(-1, self.forecast_steps, self.power_channels)

    def loss_per_sample(self, power_history, power_future):
        return (self(power_history) - power_future).square().mean(dim=(1, 2))
