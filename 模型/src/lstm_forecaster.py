"""Fixed bidirectional LSTM forecaster using historical power only."""
from __future__ import annotations
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Predict future four-channel power from an observed history window.

    Weather and wind-regime labels are intentionally excluded. The encoder is
    always bidirectional, operating only within the available history window.
    """
    def __init__(self, input_steps=12, forecast_steps=12, hidden_dim=128,
                 layers=2, dropout=0.1):
        super().__init__()
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.encoder = nn.LSTM(
            4, hidden_dim, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True, bidirectional=True,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps * 4),
        )

    def forward(self, power_history):
        encoded, _ = self.encoder(power_history)
        return self.decoder(encoded[:, -1, :]).view(-1, self.forecast_steps, 4)

    def loss_per_sample(self, power_history, power_future):
        return (self(power_history) - power_future).square().mean(dim=(1, 2))
