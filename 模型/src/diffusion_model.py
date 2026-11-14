from __future__ import annotations
import math
import torch
from torch import nn

def time_embedding(t, dim):
    """Encode a diffusion timestep as a Transformer conditioning vector."""
    half = dim // 2; freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000) / max(half - 1, 1)))
    x = t.float().unsqueeze(1) * freq.unsqueeze(0); out = torch.cat([x.sin(), x.cos()], 1)
    return torch.nn.functional.pad(out, (0, dim - out.shape[1])) if out.shape[1] < dim else out

class ConditionalDiffusion(nn.Module):
    """Conditional diffusion model for configurable power channels.
    The weather-conditioned variant receives dataset-defined weather features and
    wind-regime labels (0=below cut-in, 1=below rated,
    2=rated-to-high, 3=above high-wind threshold). The unconditional
    variant receives neither weather features nor wind-regime labels.

    ``model_dim``, ``layers`` and ``heads`` are supplied by the caller so the
    two experiments use the same capacity for a fair comparison.
    """
    use_weather = False  # Fixed by the model class, not a runtime option.

    def __init__(self, model_dim=128, layers=4, heads=4, timesteps=100, max_length=24,
                 power_channels=1, weather_channels=24):
        super().__init__(); use_weather = self.use_weather; self.timesteps = timesteps
        if power_channels < 1 or weather_channels < 1:
            raise ValueError("power_channels and weather_channels must be positive")
        self.power_channels = power_channels
        self.weather_channels = weather_channels
        self.input_dim = 3 * power_channels + (weather_channels if use_weather else 0)
        if model_dim % heads:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by heads ({heads})")
        self.input = nn.Linear(self.input_dim, model_dim)
        self.num_wind_regimes = 4
        self.regime_embedding = nn.Embedding(self.num_wind_regimes, model_dim) if use_weather else None
        self.position = nn.Parameter(torch.zeros(1, max_length, model_dim))
        self.time = nn.Sequential(nn.Linear(model_dim, model_dim * 2), nn.SiLU(), nn.Linear(model_dim * 2, model_dim))
        layer = nn.TransformerEncoderLayer(model_dim, heads, model_dim * 4, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(model_dim), enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, power_channels))
        beta = torch.linspace(1e-4, .02, timesteps); alpha = 1 - beta; self.register_buffer("alpha_bar", alpha.cumprod(0))

    def forward(self, x_t, observed, mask, t, weather=None, regimes=None):
        # Power tensors have shape [B, L, power_channels].
        fields = [x_t, observed, mask]
        if self.use_weather:
            if weather is None: raise ValueError("weather-conditioned model requires weather")
            fields.append(weather)
        h = self.input(torch.cat(fields, -1)) + self.position[:, :x_t.shape[1]] + self.time(time_embedding(t, self.position.shape[-1])).unsqueeze(1)
        if self.use_weather:
            if regimes is None:
                raise ValueError("weather-conditioned model requires per-timestep wind regime labels")
            if regimes.shape != x_t.shape[:2] or regimes.dtype != torch.long:
                raise ValueError("wind regime labels must be a torch.long tensor of shape [B, L]")
            if torch.any((regimes < 0) | (regimes >= self.regime_embedding.num_embeddings)):
                raise ValueError(f"wind regime labels must be in [0, {self.num_wind_regimes - 1}]")
            h = h + self.regime_embedding(regimes)
        return self.output(self.encoder(h))

    def loss_per_sample(self, clean, observed, mask, weather=None, missing_weight=1.0, regimes=None):
        """Compute noise loss only at artificially missing coordinates."""
        b = clean.shape[0]; t = torch.randint(0, self.timesteps, (b,), device=clean.device); noise = torch.randn_like(clean)
        ab = self.alpha_bar[t].view(b, 1, 1); noisy = ab.sqrt() * clean + (1 - ab).sqrt() * noise
        x_t = torch.where(mask.bool(), observed, noisy); pred = self(x_t, observed, mask, t, weather, regimes)
        miss = 1 - mask; base = ((pred - noise).square() * miss).sum((1, 2)) / miss.sum((1, 2)).clamp_min(1)
        return base * (1 + missing_weight)

    @torch.no_grad()
    def sample(self, observed, mask, weather=None, steps=50, draws=1, regimes=None):
        """Run DDIM-style sampling while restoring observed coordinates each step."""
        b = observed.shape[0]; grid = torch.linspace(self.timesteps - 1, 0, min(steps, self.timesteps), device=observed.device).long(); results = []
        for _ in range(draws):
            x = torch.where(mask.bool(), observed, torch.randn_like(observed))
            for i, tv in enumerate(grid):
                t = torch.full((b,), int(tv), device=observed.device, dtype=torch.long); eps = self(x, observed, mask, t, weather, regimes); ab = self.alpha_bar[tv]
                x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                if i == len(grid) - 1: x = x0
                else:
                    prev = self.alpha_bar[grid[i + 1]]; x = prev.sqrt() * x0 + (1 - prev).sqrt() * eps
                x = torch.where(mask.bool(), observed, x)
            results.append(torch.where(mask.bool(), observed, x))
        return torch.stack(results).mean(0)


class WeatherConditionalDiffusion(ConditionalDiffusion):
    """Weather- and wind-regime-conditioned diffusion method."""
    use_weather = True
