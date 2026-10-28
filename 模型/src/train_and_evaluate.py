"""Training and evaluation pipeline with row-level masking and imputation.

This module implements the corrected experiment design:
1. Diffusion models are trained for data imputation
2. Row-level masking is applied (not overlapping windows)
3. Three imputation methods fill the same masks
4. LSTM forecasters are trained on imputed data (input segment only)
5. Labels come from clean observations (target segment)
6. Final comparison shows LSTM prediction accuracy

Fixes all known defects from previous version.
"""
from __future__ import annotations
import csv, json, math, random, traceback
from pathlib import Path
import numpy as np, torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from data_pipeline import POWER, WEATHER, build_dataset
from diffusion_model import ConditionalDiffusion
from lstm_forecaster import LSTMForecaster

REGIMES = ["low_wind", "medium_wind", "high_wind"]
IMPUTATION_METHODS = ["linear_interpolation", "diffusion_no_weather", "diffusion_with_weather"]

def seed_all(seed):
    """Fix Python, NumPy, and PyTorch randomness."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def scenarios(cfg):
    """Return all masking scenarios."""
    return (
        [("random_point", rate) for rate in cfg["point_missing_rates"]] +
        [("random_continuous", length) for length in cfg["continuous_lengths"]]
    )

def create_row_mask(length, mask_type, mask_value, n_channels, rng):
    """Create mask for entire sequence (row-level, not per-window).

    Args:
        length: sequence length (number of timesteps)
        mask_type: 'random_point' or 'random_continuous'
        mask_value: missing rate (0-1) for point, or length for continuous
        n_channels: number of channels (4 for power)
        rng: numpy random generator

    Returns:
        mask: [length, n_channels] array, 1=observed, 0=missing
    """
    mask = np.ones((length, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        if mask_type == "random_point":
            n_missing = max(1, int(math.floor(length * mask_value + 0.5)))
            missing_indices = rng.choice(length, n_missing, replace=False)
            mask[missing_indices, ch] = 0
        elif mask_type == "random_continuous":
            block_len = int(mask_value)
            if block_len >= length:
                mask[:, ch] = 0
            else:
                start = int(rng.integers(0, length - block_len + 1))
                mask[start:start + block_len, ch] = 0

    return mask

def linear_interpolation(power, mask):
    """Fill missing values by linear interpolation.

    Args:
        power: [length, 4] array with NaN at missing positions
        mask: [length, 4] array, 1=observed, 0=missing

    Returns:
        filled: [length, 4] array with NaNs replaced
    """
    filled = power.copy()
    length, n_channels = power.shape

    for ch in range(n_channels):
        known_idx = np.where(mask[:, ch] > 0)[0]
        missing_idx = np.where(mask[:, ch] == 0)[0]

        if len(known_idx) == 0:
            # All missing, fill with 0
            filled[missing_idx, ch] = 0.0
        elif len(missing_idx) > 0:
            # Interpolate
            filled[missing_idx, ch] = np.interp(missing_idx, known_idx, power[known_idx, ch])

    return filled

def train_diffusion_imputer(arrays, indexes, cfg, work_dir, seed, use_weather):
    """Train diffusion model for imputation.

    Args:
        use_weather: if True, condition on weather; else no conditioning
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    method_name = "diffusion_with_weather" if use_weather else "diffusion_no_weather"
    L = cfg["window_length"]

    # Dataset for training (standard windowing)
    class ImputationWindows(Dataset):
        def __init__(self, arrays, starts, window_len):
            self.power = arrays["power"]
            self.weather = arrays["weather"]
            self.starts = starts
            self.L = window_len

        def __len__(self):
            return len(self.starts)

        def __getitem__(self, i):
            s = int(self.starts[i])
            e = s + self.L
            p = torch.from_numpy(self.power[s:e])
            w = torch.from_numpy(self.weather[s:e])
            return p, w

    train_dataset = ImputationWindows(arrays["train"], indexes["train"], L)
    val_dataset = ImputationWindows(arrays["val"], indexes["val"], L)

    train_loader = DataLoader(train_dataset, cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # Model
    net = ConditionalDiffusion(
        model_dim=cfg["diffusion_model_dim"],
        layers=cfg["diffusion_layers"],
        heads=cfg["diffusion_heads"],
        timesteps=cfg["diffusion_train_steps"],
        use_weather=use_weather,
        max_length=L
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_val, best_state, stale = float("inf"), None, 0
    logs = []
    rng = np.random.default_rng(seed + 8000)

    print(f"  [{method_name}] Training on {len(indexes['train'])} windows, {len(train_loader)} batches/epoch")

    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"{method_name} seed={seed}", unit="epoch", ncols=100):
        net.train()
        train_loss = 0.0

        for p, w in train_loader:
            p, w = p.to(device), w.to(device)

            # Random masks for training
            masks = []
            for _ in range(len(p)):
                mask_type, mask_val = scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))]
                m = create_row_mask(L, mask_type, mask_val, 4, rng)
                masks.append(m)
            masks = torch.from_numpy(np.stack(masks)).to(device)

            loss = net.loss_per_sample(p, p * masks, masks, w if use_weather else None).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            train_loss += float(loss.detach())

        # Validation
        net.eval()
        val_sq, val_n = 0.0, 0
        with torch.no_grad():
            val_rng = np.random.default_rng(seed + epoch)
            for p, w in tqdm(val_loader, desc="validation", leave=False, unit="batch"):
                p, w = p.to(device), w.to(device)

                masks = []
                for _ in range(len(p)):
                    mask_type, mask_val = scenarios(cfg)[int(val_rng.integers(len(scenarios(cfg))))]
                    m = create_row_mask(L, mask_type, mask_val, 4, val_rng)
                    masks.append(m)
                masks = torch.from_numpy(np.stack(masks)).to(device)

                pred = net.sample(p * masks, masks, w if use_weather else None, cfg["validation_ddim_steps"], 1)

                val_sq += float((((pred - p) ** 2) * (1 - masks)).sum())
                val_n += int((1 - masks).sum())

        val_rmse = math.sqrt(val_sq / max(val_n, 1))
        avg_train_loss = train_loss / max(len(train_loader), 1)

        logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_rmse": val_rmse})

        status = "★BEST" if val_rmse < best_val else f"stale={stale}/{cfg['early_stopping_patience']}"
        tqdm.write(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_rmse={val_rmse:.6f} {status}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                tqdm.write(f"  Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"{method_name} training produced no checkpoint")

    net.load_state_dict(best_state)
    net.eval()

    # Save checkpoint
    ckpt_path = work_dir / "checkpoints" / f"{method_name}_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "method": method_name,
        "seed": seed,
        "use_weather": use_weather,
        "best_val_rmse": best_val
    }, ckpt_path)

    # Save training log
    log_path = work_dir / "logs" / f"train_{method_name}_seed{seed}.csv"
    write_csv(log_path, logs)

    return net, device, {"best_val_rmse": best_val, "epochs": len(logs), "checkpoint": str(ckpt_path)}

def impute_row_level_batched(method, diffusion_net, device, power_clean, weather, mask, cfg):
    """Impute missing values using batched processing.

    Args:
        method: imputation method name
        diffusion_net: trained diffusion model (or None for linear)
        device: torch device
        power_clean: [length, 4] clean power array (for reference)
        weather: [length, 13] weather array
        mask: [length, 4] mask array
        cfg: config dict

    Returns:
        imputed: [length, 4] array with missing values filled
    """
    length = len(power_clean)
    L = cfg["window_length"]

    if method == "linear_interpolation":
        # Apply mask to create masked version
        power_masked = power_clean.copy()
        power_masked[mask == 0] = np.nan
        return linear_interpolation(power_masked, mask)

    # Diffusion-based imputation: process in non-overlapping windows
    use_weather = (method == "diffusion_with_weather")

    # Create masked power
    power_masked = power_clean * mask

    # Split into non-overlapping windows
    n_windows = length // L
    imputed = power_clean.copy()

    # Batch process windows
    batch_size = min(256, n_windows)

    for batch_start in tqdm(range(0, n_windows, batch_size), desc=f"Imputing ({method})", leave=False):
        batch_end = min(batch_start + batch_size, n_windows)
        batch_windows = []
        batch_masks = []
        batch_weathers = []

        for w_idx in range(batch_start, batch_end):
            start = w_idx * L
            end = start + L
            batch_windows.append(power_masked[start:end])
            batch_masks.append(mask[start:end])
            batch_weathers.append(weather[start:end])

        # Stack into batches
        p_batch = torch.from_numpy(np.stack(batch_windows)).to(device)
        m_batch = torch.from_numpy(np.stack(batch_masks)).to(device)
        w_batch = torch.from_numpy(np.stack(batch_weathers)).to(device)

        # Impute
        with torch.no_grad():
            pred_batch = diffusion_net.sample(
                p_batch, m_batch,
                w_batch if use_weather else None,
                cfg["ddim_steps"],
                cfg["ddim_samples"]
            )

        # Write back
        pred_np = pred_batch.cpu().numpy()
        for i, w_idx in enumerate(range(batch_start, batch_end)):
            start = w_idx * L
            end = start + L
            imputed[start:end] = pred_np[i]

    # Handle remainder (partial window at end)
    remainder_start = n_windows * L
    if remainder_start < length:
        # For simplicity, use linear interpolation for the tail
        power_masked_tail = power_clean[remainder_start:].copy()
        power_masked_tail[mask[remainder_start:] == 0] = np.nan
        imputed[remainder_start:] = linear_interpolation(power_masked_tail, mask[remainder_start:])

    return imputed

def create_scenario_masks(arrays, split, cfg, scenario_name, mask_type, mask_value):
    """Create and save masks for one scenario.

    Returns:
        mask: [length, 4] array saved to disk
    """
    length = len(arrays[split]["power"])
    rng = np.random.default_rng(cfg["mask_seed"])

    mask = create_row_mask(length, mask_type, mask_value, 4, rng)

    return mask

class ForecastWindowsCleanTarget(Dataset):
    """Dataset that uses imputed input but clean target labels."""

    def __init__(self, power_imputed, power_clean, weather, starts, input_len, forecast_len):
        self.power_imputed = power_imputed
        self.power_clean = power_clean
        self.weather = weather
        self.starts = starts
        self.input_len = input_len
        self.forecast_len = forecast_len
        self.total_len = input_len + forecast_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])

        # Input: imputed values
        p_input = torch.from_numpy(self.power_imputed[s:s + self.input_len])
        w_input = torch.from_numpy(self.weather[s:s + self.input_len])

        # Target: clean values
        p_target = torch.from_numpy(self.power_clean[s + self.input_len:s + self.total_len])
        w_target = torch.from_numpy(self.weather[s + self.input_len:s + self.total_len])

        return p_input, p_target, w_input, w_target

def train_lstm_forecaster(power_imputed, arrays, indexes, cfg, work_dir, scenario_name, method, seed):
    """Train LSTM forecaster on imputed data."""
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_len = cfg["forecast_input_steps"]
    forecast_len = cfg["forecast_steps"]
    total_len = input_len + forecast_len

    # Filter valid starts (must have enough length)
    train_starts = [s for s in indexes["train"] if s + total_len <= len(power_imputed)]
    val_starts = [s for s in indexes["val"] if s + total_len <= len(arrays["val"]["power"])]

    # Training uses imputed input, clean target
    train_dataset = ForecastWindowsCleanTarget(
        power_imputed, arrays["train"]["power"], arrays["train"]["weather"],
        train_starts, input_len, forecast_len
    )

    # Validation uses clean data (no imputation on val)
    val_dataset = ForecastWindowsCleanTarget(
        arrays["val"]["power"], arrays["val"]["power"], arrays["val"]["weather"],
        val_starts, input_len, forecast_len
    )

    train_loader = DataLoader(train_dataset, cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # Model
    net = LSTMForecaster(
        input_steps=input_len,
        forecast_steps=forecast_len,
        hidden_dim=cfg["lstm_hidden_dim"],
        layers=cfg["lstm_layers"],
        dropout=cfg["lstm_dropout"],
        use_weather=cfg["lstm_use_weather"],
        bidirectional=cfg["lstm_bidirectional"]
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_val, best_state, stale = float("inf"), None, 0
    logs = []

    print(f"  [LSTM {scenario_name}/{method}] Training on {len(train_starts)} windows")

    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"LSTM {method[:15]} seed={seed}", unit="epoch", ncols=100):
        net.train()
        train_loss = 0.0

        for p_in, p_out, w_in, w_out in train_loader:
            p_in, p_out = p_in.to(device), p_out.to(device)
            w_in, w_out = w_in.to(device), w_out.to(device)

            loss = net.loss_per_sample(
                p_in, p_out,
                w_in if cfg["lstm_use_weather"] else None,
                w_out if cfg["lstm_use_weather"] else None
            ).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            train_loss += float(loss.detach())

        # Validation
        net.eval()
        val_sq, val_n = 0.0, 0
        with torch.no_grad():
            for p_in, p_out, w_in, w_out in tqdm(val_loader, desc="validation", leave=False, unit="batch"):
                p_in, p_out = p_in.to(device), p_out.to(device)
                w_in, w_out = w_in.to(device), w_out.to(device)

                pred = net(
                    p_in,
                    w_in if cfg["lstm_use_weather"] else None,
                    w_out if cfg["lstm_use_weather"] else None
                )

                val_sq += float(((pred - p_out) ** 2).sum())
                val_n += int(p_out.numel())

        val_rmse = math.sqrt(val_sq / max(val_n, 1))
        avg_train_loss = train_loss / max(len(train_loader), 1)

        logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_rmse": val_rmse})

        status = "★BEST" if val_rmse < best_val else f"stale={stale}/{cfg['early_stopping_patience']}"
        tqdm.write(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_rmse={val_rmse:.6f} {status}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                tqdm.write(f"  Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"LSTM training produced no checkpoint")

    net.load_state_dict(best_state)
    net.eval()

    # Save checkpoint
    ckpt_path = work_dir / "checkpoints" / f"lstm_{scenario_name}_{method}_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "scenario": scenario_name,
        "method": method,
        "seed": seed,
        "best_val_rmse": best_val
    }, ckpt_path)

    # Save log
    log_path = work_dir / "logs" / f"train_lstm_{scenario_name}_{method}_seed{seed}.csv"
    write_csv(log_path, logs)

    return net, device, {"best_val_rmse": best_val, "epochs": len(logs), "checkpoint": str(ckpt_path)}

def evaluate_forecaster(net, device, arrays, indexes, cfg, scales, scenario_name, method, seed):
    """Evaluate LSTM forecaster on clean test set."""
    input_len = cfg["forecast_input_steps"]
    forecast_len = cfg["forecast_steps"]
    total_len = input_len + forecast_len

    test_starts = [s for s in indexes["test"] if s + total_len <= len(arrays["test"]["power"])]

    test_dataset = ForecastWindowsCleanTarget(
        arrays["test"]["power"], arrays["test"]["power"], arrays["test"]["weather"],
        test_starts, input_len, forecast_len
    )

    test_loader = DataLoader(test_dataset, min(cfg["batch_size"], 256), shuffle=False, num_workers=cfg["num_workers"])

    # Collect all predictions and targets
    all_pred = []
    all_target = []

    net.eval()
    with torch.no_grad():
        for p_in, p_out, w_in, w_out in tqdm(test_loader, desc=f"Eval {method[:15]}", leave=False, unit="batch"):
            p_in = p_in.to(device)
            w_in = w_in.to(device) if cfg["lstm_use_weather"] else None
            w_out = w_out.to(device) if cfg["lstm_use_weather"] else None

            pred = net(p_in, w_in, w_out)

            all_pred.append(pred.cpu().numpy())
            all_target.append(p_out.numpy())

    # Concatenate
    all_pred = np.concatenate(all_pred, axis=0)  # [N, forecast_len, 4]
    all_target = np.concatenate(all_target, axis=0)

    # Denormalize and compute metrics
    def inverse(x, field):
        scale = scales["power"][field]
        return x * (scale["max"] - scale["min"] or 1) + scale["min"]

    # Aggregate across all channels and timesteps
    all_errors = []
    for ch, field in enumerate(POWER):
        pred_ch = inverse(all_pred[:, :, ch].flatten(), field)
        target_ch = inverse(all_target[:, :, ch].flatten(), field)
        errors = np.abs(pred_ch - target_ch)
        all_errors.extend(errors.tolist())

    # Overall metrics
    all_errors = np.array(all_errors)
    result = {
        "scenario": scenario_name,
        "method": method,
        "seed": seed,
        "abs_error_max": float(np.max(all_errors)),
        "abs_error_min": float(np.min(all_errors)),
        "MAE": float(np.mean(all_errors)),
        "RMSE": float(np.sqrt(np.mean(all_errors ** 2))),
        "n_points": len(all_errors)
    }

    return result

def write_csv(path, rows):
    """Write CSV with UTF-8 BOM."""
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8-sig")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

def pivot_results(eval_rows):
    """Pivot results: one row per scenario, columns for each method."""
    # Group by scenario and seed
    grouped = {}
    for row in eval_rows:
        key = (row["scenario"], row["seed"])
        if key not in grouped:
            grouped[key] = {"scenario": row["scenario"], "seed": row["seed"]}

        method = row["method"]
        for metric in ["abs_error_max", "abs_error_min", "MAE", "RMSE"]:
            col = f"{method}_{metric}"
            grouped[key][col] = row[metric]
        grouped[key]["n_points"] = row["n_points"]

    # Convert to list and sort
    output = sorted(grouped.values(), key=lambda x: (x["scenario"], x["seed"]))

    return output

def run(config_path: Path, quick=False):
    """Main experiment pipeline."""
    root = config_path.parent
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))

    if quick:
        print("⚡ QUICK MODE: reduced epochs and steps for testing")
        cfg.update({
            "train_max_epochs": 2,
            "early_stopping_patience": 1,
            "batch_size": 64,
            "ddim_steps": 3,
            "ddim_samples": 1,
            "train_seeds": [cfg["train_seeds"][0]],  # Only first seed
            "point_missing_rates": [cfg["point_missing_rates"][0]],  # Only first rate
            "continuous_lengths": [cfg["continuous_lengths"][0]]  # Only first length
        })

    work_dir = root / "work"
    results_dir = root / "results"
    data_path = (root / cfg["data_path"]).resolve()

    # Phase 0: Data preparation
    print("\n" + "=" * 100)
    print("PHASE 0: DATA PREPARATION")
    print("=" * 100)

    arrays, indexes, scales, audit = build_dataset(data_path, work_dir, cfg)

    print(f"Train: {len(arrays['train']['power'])} timesteps, {len(indexes['train'])} windows")
    print(f"Val:   {len(arrays['val']['power'])} timesteps, {len(indexes['val'])} windows")
    print(f"Test:  {len(arrays['test']['power'])} timesteps, {len(indexes['test'])} windows")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    log = {
        "device": device_name,
        "audit": audit,
        "config": cfg,
        "runs": []
    }

    # Phase 1: Train diffusion imputers
    print("\n" + "=" * 100)
    print("PHASE 1: TRAIN DIFFUSION IMPUTERS")
    print("=" * 100)

    diffusion_models = {}

    for seed in cfg["train_seeds"]:
        for use_weather in [False, True]:
            method = "diffusion_with_weather" if use_weather else "diffusion_no_weather"

            try:
                print(f"\n{'=' * 100}")
                print(f"Training {method} | seed={seed}")
                print('=' * 100)

                net, device, info = train_diffusion_imputer(
                    arrays, indexes, cfg, work_dir, seed, use_weather
                )

                diffusion_models[(seed, use_weather)] = (net, device)

                log["runs"].append({
                    "phase": "diffusion_training",
                    "method": method,
                    "seed": seed,
                    "status": "completed",
                    **info
                })

                print(f"✓ Completed: val_rmse={info['best_val_rmse']:.6f}, epochs={info['epochs']}")

            except Exception as e:
                print(f"✗ FAILED: {repr(e)}")
                traceback.print_exc()
                log["runs"].append({
                    "phase": "diffusion_training",
                    "method": method,
                    "seed": seed,
                    "status": "failed",
                    "error": repr(e)
                })

    # Phase 2-4: For each scenario, mask → impute → train LSTM → evaluate
    print("\n" + "=" * 100)
    print("PHASE 2-4: MASK → IMPUTE → TRAIN LSTM → EVALUATE")
    print("=" * 100)

    eval_rows = []

    for mask_type, mask_value in scenarios(cfg):
        scenario_name = f"{mask_type}_{mask_value}"

        print(f"\n{'=' * 100}")
        print(f"SCENARIO: {scenario_name}")
        print('=' * 100)

        for seed in cfg["train_seeds"]:
            print(f"\nSeed: {seed}")

            # Create masks (same for all three methods)
            print("  Creating masks...")
            train_mask = create_scenario_masks(arrays, "train", cfg, scenario_name, mask_type, mask_value)
            val_mask = create_scenario_masks(arrays, "val", cfg, scenario_name, mask_type, mask_value)

            # Save masks
            mask_dir = work_dir / "masked" / scenario_name
            mask_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_dir / f"train_mask_seed{seed}.npz", mask=train_mask)
            np.savez_compressed(mask_dir / f"val_mask_seed{seed}.npz", mask=val_mask)

            n_missing_train = int((train_mask == 0).sum())
            n_total_train = train_mask.size
            print(f"  Train mask: {n_missing_train}/{n_total_train} missing ({n_missing_train/n_total_train*100:.2f}%)")

            # Three imputation methods
            for method in IMPUTATION_METHODS:
                try:
                    print(f"\n  Method: {method}")

                    # Impute
                    if method == "linear_interpolation":
                        diffusion_net, device = None, None
                    elif method == "diffusion_no_weather":
                        diffusion_net, device = diffusion_models.get((seed, False), (None, None))
                        if diffusion_net is None:
                            print(f"    ✗ Diffusion model not available, skipping")
                            continue
                    elif method == "diffusion_with_weather":
                        diffusion_net, device = diffusion_models.get((seed, True), (None, None))
                        if diffusion_net is None:
                            print(f"    ✗ Diffusion model not available, skipping")
                            continue

                    print(f"    Imputing training data...")
                    train_imputed = impute_row_level_batched(
                        method, diffusion_net, device,
                        arrays["train"]["power"], arrays["train"]["weather"],
                        train_mask, cfg
                    )

                    # Save imputed data
                    imputed_dir = work_dir / "imputed" / scenario_name / method
                    imputed_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(imputed_dir / f"train_imputed_seed{seed}.npz", power=train_imputed)

                    # Train LSTM
                    print(f"    Training LSTM forecaster...")
                    lstm_net, lstm_device, lstm_info = train_lstm_forecaster(
                        train_imputed, arrays, indexes, cfg, work_dir,
                        scenario_name, method, seed
                    )

                    log["runs"].append({
                        "phase": "lstm_training",
                        "scenario": scenario_name,
                        "method": method,
                        "seed": seed,
                        "status": "completed",
                        **lstm_info
                    })

                    # Evaluate
                    print(f"    Evaluating on test set...")
                    result = evaluate_forecaster(
                        lstm_net, lstm_device, arrays, indexes, cfg, scales,
                        scenario_name, method, seed
                    )

                    eval_rows.append(result)

                    print(f"    ✓ MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}")

                    del lstm_net

                except Exception as e:
                    print(f"    ✗ FAILED: {repr(e)}")
                    traceback.print_exc()
                    log["runs"].append({
                        "phase": "lstm_training_or_eval",
                        "scenario": scenario_name,
                        "method": method,
                        "seed": seed,
                        "status": "failed",
                        "error": repr(e)
                    })

    # Phase 5: Export results
    print("\n" + "=" * 100)
    print("PHASE 5: EXPORT RESULTS")
    print("=" * 100)

    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        if eval_rows:
            pivoted = pivot_results(eval_rows)
            results_path = results_dir / "forecast_imputation_comparison.csv"
            write_csv(results_path, pivoted)
            print(f"✓ Results exported: {results_path}")
            print(f"  {len(pivoted)} rows (scenarios × seeds × methods)")
        else:
            print("✗ No evaluation results to export")

        # Save run log
        log_path = work_dir / "logs" / "run_summary.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ Run summary: {log_path}")

    except Exception as e:
        print(f"✗ Export failed: {repr(e)}")
        traceback.print_exc()

    # Summary
    completed = sum(1 for r in log["runs"] if r.get("status") == "completed")
    failed = sum(1 for r in log["runs"] if r.get("status") == "failed")

    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    print(f"Device: {device_name}")
    print(f"Total runs: {len(log['runs'])}")
    print(f"  ✓ Completed: {completed}")
    print(f"  ✗ Failed: {failed}")
    print("=" * 100 + "\n")

    return log
