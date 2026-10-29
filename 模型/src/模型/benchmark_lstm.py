"""
Find optimal LSTM configuration by testing different hidden_dim and layer combinations.
Measure: parameter count, training speed, validation performance.
"""
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lstm_forecaster import LSTMForecaster

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_config(input_steps, forecast_steps, hidden_dim, layers):
    """Test one configuration."""
    model = LSTMForecaster(
        input_steps=input_steps,
        forecast_steps=forecast_steps,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=0.1
    )

    params = count_parameters(model)

    # Test forward pass speed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    batch_size = 128
    power_history = torch.randn(batch_size, input_steps, 4, device=device)
    power_future = torch.randn(batch_size, forecast_steps, 4, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(power_history)

    # Benchmark forward
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    iterations = 50
    for _ in range(iterations):
        _ = model(power_history)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = (time.time() - start) / iterations * 1000  # ms

    # Benchmark forward + backward
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = model.loss_per_sample(power_history, power_future).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    train_time = (time.time() - start) / iterations * 1000  # ms

    return {
        'params': params,
        'forward_ms': forward_time,
        'train_ms': train_time
    }

if __name__ == '__main__':
    print("=" * 80)
    print("LSTM Configuration Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch size: 128, Input steps: 12, Forecast steps: 12, Bidirectional: True, Power-only: True")
    print()

    # Target: diffusion model has 6,661,636 parameters
    target_params = 6661636

    # Test different configurations
    configs = []

    # Compare fixed bidirectional models across hidden dimensions and layer counts.
    for hidden in range(350, 500, 10):
        for layers in [1, 2, 3]:
            result = benchmark_config(12, 12, hidden, layers)
            diff = abs(result['params'] - target_params)
            diff_pct = (result['params'] - target_params) / target_params * 100
            configs.append({
                'hidden': hidden,
                'layers': layers,
                'params': result['params'],
                'diff': diff,
                'diff_pct': diff_pct,
                'forward_ms': result['forward_ms'],
                'train_ms': result['train_ms']
            })

    # Sort by parameter difference
    configs.sort(key=lambda x: x['diff'])

    print(f"Target parameters (diffusion model): {target_params:,}")
    print()
    print("Top 10 closest configurations:")
    print("-" * 80)
    print(f"{'Hidden':<8} {'Layers':<8} {'Params':<12} {'Diff %':<10} {'Fwd(ms)':<10} {'Train(ms)':<10}")
    print("-" * 80)

    for cfg in configs[:10]:
        print(f"{cfg['hidden']:<8} {cfg['layers']:<8} {cfg['params']:<12,} "
              f"{cfg['diff_pct']:>+7.2f}%  {cfg['forward_ms']:>8.2f}  {cfg['train_ms']:>9.2f}")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best = configs[0]
    print(f"Best match: hidden={best['hidden']}, layers={best['layers']}, bidirectional=True")
    print(f"Parameters: {best['params']:,} (diff: {best['diff_pct']:+.3f}%)")
    print(f"Forward: {best['forward_ms']:.2f}ms, Train: {best['train_ms']:.2f}ms per batch")
    print()

    # Also show the project's current configuration.
    current = benchmark_config(12, 12, 512, 2)
    print(f"Current config (hidden=512, layers=2, bidir=True):")
    print(f"Parameters: {current['params']:,} (diff: {(current['params']-target_params)/target_params*100:+.3f}%)")
    print(f"Forward: {current['forward_ms']:.2f}ms, Train: {current['train_ms']:.2f}ms per batch")
