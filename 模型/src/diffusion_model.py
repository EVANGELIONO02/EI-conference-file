from __future__ import annotations
import math
import torch
from torch import nn


def time_embedding(t, dim):
    """Encode a diffusion timestep as a Transformer conditioning vector."""
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                     * (-math.log(10000) / max(half - 1, 1)))
    x = t.float().unsqueeze(1) * freq.unsqueeze(0)
    out = torch.cat([x.sin(), x.cos()], 1)
    return torch.nn.functional.pad(out, (0, dim - out.shape[1])) if out.shape[1] < dim else out


class ConditionalDiffusion(nn.Module):
    """Conditional diffusion model for configurable power channels.

    ``valid_mask`` marks real sequence positions when a batch contains padded
    variable-length windows.  Padded positions are ignored by attention,
    training loss, and sampling, while observed values remain clamped exactly.
    """

    def __init__(self, model_dim=128, layers=4, heads=4, timesteps=100, max_length=24,
                 power_channels=1, weather_channels=24, use_weather=False):
        super().__init__()
        self.use_weather = use_weather
        self.timesteps = timesteps
        if power_channels < 1 or weather_channels < 1 or max_length < 1:
            raise ValueError("power_channels, weather_channels, and max_length must be positive")
        self.input_dim = 3 * power_channels + (weather_channels if use_weather else 0)
        if model_dim % heads:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by heads ({heads})")
        self.input = nn.Linear(self.input_dim, model_dim)
        self.num_wind_regimes = 4
        self.regime_embedding = nn.Embedding(self.num_wind_regimes, model_dim) if use_weather else None
        self.position = nn.Parameter(torch.zeros(1, max_length, model_dim))
        self.time = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2), nn.SiLU(), nn.Linear(model_dim * 2, model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            model_dim, heads, model_dim * 4, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(model_dim), enable_nested_tensor=False
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, power_channels)
        )
        beta = torch.linspace(1e-4, 0.02, timesteps)
        self.register_buffer("alpha_bar", (1 - beta).cumprod(0))

    def _valid(self, x_t, valid_mask):
        if valid_mask is None:
            return torch.ones(x_t.shape[:2], dtype=torch.bool, device=x_t.device)
        if valid_mask.shape != x_t.shape[:2]:
            raise ValueError("valid_mask must have shape [B, L]")
        return valid_mask.to(device=x_t.device, dtype=torch.bool)

    def forward(self, x_t, observed, mask, t, weather=None, regimes=None, valid_mask=None):
        # Power tensors have shape [B, L, power_channels].
        valid = self._valid(x_t, valid_mask)
        if x_t.shape[1] > self.position.shape[1]:
            raise ValueError("input sequence is longer than the model's max_length")
        fields = [x_t, observed, mask]
        if self.use_weather:
            if weather is None:
                raise ValueError("weather-conditioned model requires weather")
            fields.append(weather)
        h = (self.input(torch.cat(fields, -1))
             + self.position[:, :x_t.shape[1]]
             + self.time(time_embedding(t, self.position.shape[-1])).unsqueeze(1))
        if self.use_weather:
            if regimes is None:
                raise ValueError("weather-conditioned model requires per-timestep wind regime labels")
            if regimes.shape != x_t.shape[:2] or regimes.dtype != torch.long:
                raise ValueError("wind regime labels must be a torch.long tensor of shape [B, L]")
            if torch.any((regimes < 0) | (regimes >= self.regime_embedding.num_embeddings)):
                raise ValueError(f"wind regime labels must be in [0, {self.num_wind_regimes - 1}]")
            h = h + self.regime_embedding(regimes)
        return self.output(self.encoder(h, src_key_padding_mask=~valid))

    def loss_per_sample(self, clean, observed, mask, weather=None, missing_weight=1.0,
                        regimes=None, valid_mask=None, *, generator=None):
        """Compute noise loss only at real, artificially missing coordinates."""
        valid = self._valid(clean, valid_mask)
        valid3 = valid.unsqueeze(-1)
        batch = clean.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
        alpha_bar = self.alpha_bar[timesteps].view(batch, 1, 1)
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        x_t = torch.where(mask.bool(), observed, noisy)
        x_t = torch.where(valid3, x_t, torch.zeros_like(x_t))
        pred = self(x_t, observed, mask, timesteps, weather, regimes, valid)
        missing = (1 - mask) * valid3.to(mask.dtype)
        base = ((pred - noise).square() * missing).sum((1, 2)) / missing.sum((1, 2)).clamp_min(1)
        return base * (1 + missing_weight)

    @torch.no_grad()
    def sample(self, observed, mask, weather=None, steps=50, draws=1, regimes=None, valid_mask=None, *, generator=None):
        """Run DDIM-style sampling while restoring observed coordinates each step."""
        valid = self._valid(observed, valid_mask)
        valid3 = valid.unsqueeze(-1)
        if observed.shape[1] > self.position.shape[1]:
            raise ValueError("input sequence is longer than the model's max_length")
        grid = torch.linspace(
            self.timesteps - 1, 0, min(max(int(steps), 1), self.timesteps),
            device=observed.device
        ).long()
        # Gather the full schedule on-device; no scalar timestep readback per step.
        alphas = self.alpha_bar.index_select(0, grid).unbind()
        timesteps = grid.unbind()
        observed_mask = mask.bool()
        results = []
        for _ in range(draws):
            noise = torch.randn(observed.shape, dtype=observed.dtype, device=observed.device, generator=generator)
            x = torch.where(observed_mask, observed, noise)
            x = torch.where(valid3, x, 0.0)
            for i, (timestep, alpha_bar) in enumerate(zip(timesteps, alphas)):
                t = timestep.expand(observed.shape[0])
                eps = self(x, observed, mask, t, weather, regimes, valid)
                x0 = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
                if i == len(alphas) - 1:
                    x = x0
                else:
                    previous = alphas[i + 1]
                    x = previous.sqrt() * x0 + (1 - previous).sqrt() * eps
                x = torch.where(observed_mask, observed, x)
                x = torch.where(valid3, x, 0.0)
            results.append(x)
        return torch.stack(results).mean(0)
