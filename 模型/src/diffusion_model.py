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
    """Four-channel conditional diffusion model.
    The weather variant additionally receives a [B, L, 13] weather tensor.
    """
    def __init__(self, model_dim=128, layers=4, heads=4, timesteps=100, use_weather=True, max_length=24):
        super().__init__(); self.use_weather = use_weather; self.timesteps = timesteps
        self.input = nn.Linear(12 + (13 if use_weather else 0), model_dim)
        self.position = nn.Parameter(torch.zeros(1, max_length, model_dim))
        self.time = nn.Sequential(nn.Linear(model_dim, model_dim * 2), nn.SiLU(), nn.Linear(model_dim * 2, model_dim))
        layer = nn.TransformerEncoderLayer(model_dim, heads, model_dim * 4, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(model_dim), enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 4))
        beta = torch.linspace(1e-4, .02, timesteps); alpha = 1 - beta; self.register_buffer("alpha_bar", alpha.cumprod(0))

    def forward(self, x_t, observed, mask, t, weather=None):
        # x_t, observed, and mask have shape [B, L, 4].
        fields = [x_t, observed, mask]
        if self.use_weather:
            if weather is None: raise ValueError("weather-conditioned model requires weather")
            fields.append(weather)
        h = self.input(torch.cat(fields, -1)) + self.position[:, :x_t.shape[1]] + self.time(time_embedding(t, self.position.shape[-1])).unsqueeze(1)
        return self.output(self.encoder(h))

    def loss_per_sample(self, clean, observed, mask, weather=None, missing_weight=1.0):
        """Compute noise loss only at artificially missing coordinates."""
        b = clean.shape[0]; t = torch.randint(0, self.timesteps, (b,), device=clean.device); noise = torch.randn_like(clean)
        ab = self.alpha_bar[t].view(b, 1, 1); noisy = ab.sqrt() * clean + (1 - ab).sqrt() * noise
        x_t = torch.where(mask.bool(), observed, noisy); pred = self(x_t, observed, mask, t, weather)
        miss = 1 - mask; base = ((pred - noise).square() * miss).sum((1, 2)) / miss.sum((1, 2)).clamp_min(1)
        return base * (1 + missing_weight)

    @torch.no_grad()
    def sample(self, observed, mask, weather=None, steps=50, draws=1):
        """Run DDIM-style sampling while restoring observed coordinates each step."""
        b = observed.shape[0]; grid = torch.linspace(self.timesteps - 1, 0, min(steps, self.timesteps), device=observed.device).long(); results = []
        for _ in range(draws):
            x = torch.where(mask.bool(), observed, torch.randn_like(observed))
            for i, tv in enumerate(grid):
                t = torch.full((b,), int(tv), device=observed.device, dtype=torch.long); eps = self(x, observed, mask, t, weather); ab = self.alpha_bar[tv]
                x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                if i == len(grid) - 1: x = x0
                else:
                    prev = self.alpha_bar[grid[i + 1]]; x = prev.sqrt() * x0 + (1 - prev).sqrt() * eps
                x = torch.where(mask.bool(), observed, x)
            results.append(torch.where(mask.bool(), observed, x))
        return torch.stack(results).mean(0)
