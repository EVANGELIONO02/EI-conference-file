"""Quick calculation of LSTM parameter counts for different configurations."""

def lstm_params(input_steps, forecast_steps, hidden_dim, layers, use_weather, bidirectional):
    """
    Calculate LSTMForecaster parameters.

    Architecture:
    - LSTM encoder: input_dim -> hidden_dim, num_layers, bidirectional
    - Decoder: encoder_output_dim -> hidden_dim -> forecast_steps * 4
    """
    input_dim = 4 + (13 if use_weather else 0)  # power(4) + weather(13)
    encoder_output_dim = hidden_dim * (2 if bidirectional else 1)

    # LSTM parameters per layer per direction
    # 4 gates * (input_size * hidden + hidden * hidden + bias)
    lstm_params_total = 0
    for layer_idx in range(layers):
        layer_input = input_dim if layer_idx == 0 else hidden_dim * (2 if bidirectional else 1)
        # Forward direction
        lstm_params_total += 4 * (layer_input * hidden_dim + hidden_dim * hidden_dim + hidden_dim)
        # Backward direction
        if bidirectional:
            lstm_params_total += 4 * (layer_input * hidden_dim + hidden_dim * hidden_dim + hidden_dim)

    # Decoder parameters
    # LayerNorm(encoder_output_dim)
    ln_params = encoder_output_dim * 2  # weight + bias
    # Linear(encoder_output_dim, hidden_dim)
    linear1_params = encoder_output_dim * hidden_dim + hidden_dim
    # Linear(hidden_dim, forecast_steps * 4)
    linear2_params = hidden_dim * (forecast_steps * 4) + (forecast_steps * 4)

    decoder_params = ln_params + linear1_params + linear2_params

    total = lstm_params_total + decoder_params
    return total

# Target: diffusion model has 6,661,636 parameters
target = 6661636

print("=" * 100)
print("LSTM Parameter Count Analysis")
print("=" * 100)
print(f"Target (Diffusion model): {target:,} parameters")
print(f"Config: input_steps=12, forecast_steps=12, use_weather=True")
print()

# Current config
current = lstm_params(12, 12, 427, 2, True, True)
print(f"Current (hidden=427, layers=2, bidir=True): {current:,} ({(current-target)/target*100:+.3f}%)")
print()

# Search for best match
print("=" * 100)
print("Searching optimal configurations...")
print("=" * 100)

results = []
for hidden in range(350, 500, 1):
    for layers in [1, 2, 3]:
        for bidir in [True, False]:
            params = lstm_params(12, 12, hidden, layers, True, bidir)
            diff = abs(params - target)
            diff_pct = (params - target) / target * 100
            results.append({
                'hidden': hidden,
                'layers': layers,
                'bidir': bidir,
                'params': params,
                'diff': diff,
                'diff_pct': diff_pct
            })

# Sort by absolute difference
results.sort(key=lambda x: x['diff'])

print()
print("Top 15 closest configurations:")
print("-" * 100)
print(f"{'Hidden':<10} {'Layers':<10} {'Bidir':<12} {'Parameters':<15} {'Difference':<20}")
print("-" * 100)

for i, r in enumerate(results[:15]):
    print(f"{r['hidden']:<10} {r['layers']:<10} {str(r['bidir']):<12} {r['params']:<15,} {r['diff_pct']:>+7.3f}%")

print()
print("=" * 100)
print("RECOMMENDATION")
print("=" * 100)
best = results[0]
print(f"Best match: hidden={best['hidden']}, layers={best['layers']}, bidirectional={best['bidir']}")
print(f"Parameters: {best['params']:,} (difference: {best['diff_pct']:+.3f}%)")
print()

# Analysis by configuration type
print("=" * 100)
print("Analysis by configuration type:")
print("=" * 100)

# Best for each type
for layers in [1, 2, 3]:
    for bidir in [True, False]:
        subset = [r for r in results if r['layers'] == layers and r['bidir'] == bidir]
        if subset:
            best_subset = min(subset, key=lambda x: x['diff'])
            config_name = f"{layers} layer{'s' if layers > 1 else ''}, {'bidirectional' if bidir else 'unidirectional'}"
            print(f"{config_name:40} → hidden={best_subset['hidden']}, params={best_subset['params']:,} ({best_subset['diff_pct']:+.3f}%)")

print()
print("=" * 100)
print("PRACTICAL CONSIDERATIONS")
print("=" * 100)
print("1. Bidirectional LSTM is generally better for sequence modeling (can see future context)")
print("2. 2 layers is a good balance between capacity and speed")
print("3. Parameter match within ±1% is sufficient for fair comparison")
print()

# Show a few practical choices
print("Recommended configurations:")
print("-" * 100)
practical = [r for r in results if r['layers'] == 2 and r['bidir'] == True and abs(r['diff_pct']) < 1.0]
for r in practical[:5]:
    print(f"  hidden={r['hidden']}, layers=2, bidir=True → {r['params']:,} params ({r['diff_pct']:+.2f}%)")
