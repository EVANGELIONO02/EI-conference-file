"""LSTM forecasting model for wind power prediction.

This module implements a separate LSTM forecaster that predicts future power
values based on historical power and weather data. Unlike the LSTMImputer in
train_and_evaluate.py (which fills missing values), this model performs
multi-step-ahead forecasting.
"""
from __future__ import annotations
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """LSTM model for wind power forecasting.

    Takes a sequence of historical power and weather observations and predicts
    future power values across all four channels (Power_0, Power_15, Power_30, Power_45).

    Args:
        input_steps: number of historical timesteps to use as input
        forecast_steps: number of future timesteps to predict
        hidden_dim: LSTM hidden dimension
        layers: number of LSTM layers
        dropout: dropout rate
        use_weather: whether to include weather features as input
        bidirectional: whether to use bidirectional LSTM
    """
    def __init__(
        self,
        input_steps=12,
        forecast_steps=12,
        hidden_dim=128,
        layers=2,
        dropout=0.1,
        use_weather=True,
        bidirectional=False,
    ):
        super().__init__()
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.use_weather = use_weather
        self.bidirectional = bidirectional

        # Input: [power (4 channels) + weather (13 channels if enabled)]
        input_dim = 4 + (13 if use_weather else 0)
        recurrent_dropout = dropout if layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Decoder: takes encoder output and produces multi-step forecasts
        self.decoder = nn.Sequential(
            nn.LayerNorm(encoder_output_dim),
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps * 4),  # 4 power channels
        )

    def forward(self, power_history, weather_history=None, weather_future=None):
        """Forward pass for forecasting.

        Args:
            power_history: [B, input_steps, 4] historical power values
            weather_history: [B, input_steps, 13] historical weather (optional)
            weather_future: [B, forecast_steps, 13] future weather (optional, for now unused)

        Returns:
            [B, forecast_steps, 4] predicted power values
        """
        # Concatenate power and weather as input features
        if self.use_weather:
            if weather_history is None:
                raise ValueError("weather-enabled forecaster requires weather_history")
            encoder_input = torch.cat([power_history, weather_history], dim=-1)
        else:
            encoder_input = power_history

        # Encode the historical sequence
        encoded, _ = self.encoder(encoder_input)  # [B, input_steps, encoder_output_dim]

        # Use the last encoder output to generate forecasts
        last_hidden = encoded[:, -1, :]  # [B, encoder_output_dim]

        # Decode to multi-step forecast
        forecast_flat = self.decoder(last_hidden)  # [B, forecast_steps * 4]
        forecast = forecast_flat.view(-1, self.forecast_steps, 4)  # [B, forecast_steps, 4]

        return forecast

    def loss_per_sample(self, power_history, power_future, weather_history=None, weather_future=None):
        """Compute MSE loss per sample for training.

        Args:
            power_history: [B, input_steps, 4]
            power_future: [B, forecast_steps, 4] ground truth
            weather_history: [B, input_steps, 13]
            weather_future: [B, forecast_steps, 13]

        Returns:
            [B] loss per sample
        """
        prediction = self(power_history, weather_history, weather_future)
        # MSE across all forecast steps and channels, averaged per sample
        return (prediction - power_future).square().mean(dim=(1, 2))
