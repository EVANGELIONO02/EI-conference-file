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
from data_pipeline import POWER, REGIME_NAMES, WIND_REGIME_THRESHOLDS, WIND_REGIME_KEY, build_dataset
from diffusion_model import ConditionalDiffusion, WeatherConditionalDiffusion
from lstm_forecaster import LSTMForecaster

REGIMES = REGIME_NAMES
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
        mask_value: missing rate (0-1) for point, or block length for continuous
        n_channels: number of power channels
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
            block_len = max(1, int(mask_value))
            for region_start in range(0, length, 24):
                region_end = min(region_start + 24, length)
                width = min(block_len, region_end - region_start)
                start = region_start + int(rng.integers(0, region_end - region_start - width + 1))
                mask[start:start + width, ch] = 0
        else:
            raise ValueError(f"unknown mask type: {mask_type}")

    return mask

def linear_interpolation(power, mask):
    """Fill missing values by linear interpolation.

    Args:
        power: [length, channels] array with NaN at missing positions
        mask: [length, channels] array, 1=observed, 0=missing

    Returns:
        filled: [length, channels] array with NaNs replaced
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


def train_diffusion_imputer(arrays, indexes, cfg, work_dir, seed, method_name):
    """Train diffusion model for imputation.

    Args:
        method_name: one of the two fixed diffusion experiment methods
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_classes = {"diffusion_no_weather": ConditionalDiffusion,
                     "diffusion_with_weather": WeatherConditionalDiffusion}
    model_class = model_classes[method_name]
    use_weather = model_class.use_weather
    L = cfg["window_length"]

    # Dataset for training (standard windowing)
    class ImputationWindows(Dataset):
        def __init__(self, arrays, starts, window_len):
            self.power = arrays["power"]
            self.weather = arrays["weather"]
            self.cluster = arrays["cluster"]
            self.starts = starts
            self.L = window_len

        def __len__(self):
            return len(self.starts)

        def __getitem__(self, i):
            s = int(self.starts[i])
            e = s + self.L
            p = torch.from_numpy(self.power[s:e])
            w = torch.from_numpy(self.weather[s:e])
            regimes = torch.from_numpy(self.cluster[s:e]).long()
            return p, w, regimes

    train_dataset = ImputationWindows(arrays["train"], indexes["train"], L)
    val_dataset = ImputationWindows(arrays["val"], indexes["val"], L)

    train_loader = DataLoader(train_dataset, cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # Model
    net = model_class(
        model_dim=cfg["diffusion_model_dim"],
        layers=cfg["diffusion_layers"],
        heads=cfg["diffusion_heads"],
        timesteps=cfg["diffusion_train_steps"],
        max_length=L,
        power_channels=len(POWER),
        weather_channels=len(arrays["train"]["weather"][0])
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_val, best_state, stale = float("inf"), None, 0
    logs = []
    rng = np.random.default_rng(seed + 8000)

    print(f"  [{method_name}] Training on {len(indexes['train'])} windows, {len(train_loader)} batches/epoch")

    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"{method_name} seed={seed}", unit="epoch", ncols=100):
        net.train()
        train_loss = 0.0

        for p, w, regimes in train_loader:
            p, w = p.to(device), w.to(device)
            if use_weather:
                regimes = regimes.to(device)

            # Random masks for training
            masks = []
            for _ in range(len(p)):
                mask_type, mask_val = scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))]
                m = create_row_mask(L, mask_type, mask_val, len(POWER), rng)
                masks.append(m)
            masks = torch.from_numpy(np.stack(masks)).to(device)

            loss = net.loss_per_sample(p, p * masks, masks, w if use_weather else None,
                                       regimes=regimes if use_weather else None).mean()

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
            for p, w, regimes in tqdm(val_loader, desc="validation", leave=False, unit="batch"):
                p, w = p.to(device), w.to(device)
                regimes = regimes.to(device) if use_weather else None

                masks = []
                for _ in range(len(p)):
                    mask_type, mask_val = scenarios(cfg)[int(val_rng.integers(len(scenarios(cfg))))]
                    masks.append(create_row_mask(L, mask_type, mask_val, len(POWER), val_rng))
                masks = torch.from_numpy(np.stack(masks)).to(device)

                pred = net.sample(p * masks, masks, w if use_weather else None, cfg["validation_ddim_steps"], 1,
                                  regimes=regimes)

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
        "uses_wind_regime_labels": use_weather,
        "regime_names": REGIMES if use_weather else [],
        "wind_regime_key": cfg.get("wind_regime_key", WIND_REGIME_KEY),
        "wind_regime_thresholds_m_s": cfg.get("wind_regime_thresholds_m_s", WIND_REGIME_THRESHOLDS),
        "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
        "best_val_rmse": best_val
    }, ckpt_path)

    # Save training log
    log_path = work_dir / "logs" / f"train_{method_name}_seed{seed}.csv"
    write_csv(log_path, logs)

    return net, device, {"best_val_rmse": best_val, "epochs": len(logs), "checkpoint": str(ckpt_path)}

def impute_row_level_batched(method, diffusion_net, device, power_clean, weather, mask, cfg, regimes=None, segment_ids=None):
    """Impute missing values using batched processing.

    Args:
        method: imputation method name
        diffusion_net: trained diffusion model (or None for linear)
        device: torch device
        power_clean: [length, channels] clean power array (for reference)
        mask: [length, channels] mask array
        cfg: config dict

    Returns:
        imputed: [length, channels] array with missing values filled
    """
    length = len(power_clean)
    L = cfg["window_length"]

    if method == "linear_interpolation":
        # Forecasting is causal: do not use a right-hand value after the
        # forecast origin to fill the last history point. Forward-fill each
        # channel within each segment instead of retrospective interpolation.
        segment = (np.asarray(segment_ids) if segment_ids is not None
                   else np.zeros(length, dtype=np.int32))
        if segment.shape != (length,):
            raise ValueError("segment_ids must have one value per row")
        filled = power_clean.copy()
        for sid in np.unique(segment):
            positions = np.flatnonzero(segment == sid)
            for ch in range(power_clean.shape[1]):
                last = None
                for pos in positions:
                    if mask[pos, ch] > 0:
                        last = power_clean[pos, ch]
                    elif last is not None:
                        filled[pos, ch] = last
                    else:
                        filled[pos, ch] = 0.0
        return filled

    # Diffusion imputation is applied independently to each causal history
    # prefix. It never receives the target suffix or later rows.
    if method not in {"diffusion_no_weather", "diffusion_with_weather"}:
        raise ValueError(f"unknown imputation method: {method}")

    use_weather = diffusion_net.use_weather
    if use_weather:
        if regimes is None or np.asarray(regimes).shape != (length,):
            raise ValueError("weather-conditioned imputation requires one wind regime label per timestep")
        regimes = np.asarray(regimes, dtype=np.int64)
        if np.any((regimes < 0) | (regimes >= len(REGIMES))):
            raise ValueError(f"wind regime labels must be in [0, {len(REGIMES) - 1}]")

    # Create masked power
    power_masked = power_clean * mask
    weather = np.asarray(weather)

    segment = (np.asarray(segment_ids) if segment_ids is not None
               else np.zeros(length, dtype=np.int32))
    if segment.shape != (length,):
        raise ValueError("segment_ids must have one value per row")
    input_len = int(cfg.get("forecast_input_steps", L // 2))
    imputed = power_clean.copy()
    starts = []
    for sid in np.unique(segment):
        positions = np.flatnonzero(segment == sid)
        seg_start, seg_end = int(positions[0]), int(positions[-1]) + 1
        origins = list(range(seg_start + input_len, seg_end + 1, input_len))
        if not origins or origins[-1] != seg_end:
            origins.append(seg_end)
        starts.extend((seg_start, seg_end, int(s)) for s in origins)

    batch_size = min(256, len(starts))
    if batch_size == 0:
        return imputed

    for batch_start in tqdm(range(0, len(starts), batch_size), desc=f"Imputing ({method})", leave=False):
        batch = starts[batch_start:batch_start + batch_size]
        windows, masks, weathers, regimes_batch = [], [], [], []
        for seg_start, seg_end, origin in batch:
            # The model receives only the causal history prefix ending at origin.
            context_start = max(seg_start, origin - input_len)
            context_end = origin
            left_pad = input_len - (origin - context_start)
            n_channels = power_clean.shape[1]
            p = np.zeros((L, n_channels), dtype=np.float32)
            m = np.ones((L, n_channels), dtype=np.float32)
            w = np.zeros((L, weather.shape[1]), dtype=np.float32)
            p[left_pad:left_pad + context_end - context_start] = power_masked[context_start:context_end]
            m[left_pad:left_pad + context_end - context_start] = mask[context_start:context_end]
            w[left_pad:left_pad + context_end - context_start] = weather[context_start:context_end]
            windows.append(p); masks.append(m); weathers.append(w)
            if use_weather:
                r = np.zeros(L, dtype=np.int64)
                r[left_pad:left_pad + context_end - context_start] = regimes[context_start:context_end]
                regimes_batch.append(r)

        p_batch = torch.from_numpy(np.stack(windows)).to(device)
        m_batch = torch.from_numpy(np.stack(masks)).to(device)
        w_batch = torch.from_numpy(np.stack(weathers)).to(device)
        r_batch = torch.as_tensor(np.stack(regimes_batch), dtype=torch.long, device=device) if use_weather else None
        with torch.no_grad():
            pred_batch = diffusion_net.sample(p_batch, m_batch, w_batch if use_weather else None,
                                              cfg["ddim_steps"], cfg["ddim_samples"], regimes=r_batch)
        pred_np = pred_batch.cpu().numpy()
        for i, (seg_start, seg_end, origin) in enumerate(batch):
            end = origin
            context_start = max(seg_start, origin - input_len)
            width = end - context_start
            left_pad = input_len - width
            output_offset = left_pad
            write_mask = mask[context_start:end] == 0
            chunk = imputed[context_start:end]
            pred = np.clip(pred_np[i, output_offset:output_offset + width], 0.0, 1.0)
            chunk[write_mask] = pred[write_mask]
            imputed[context_start:end] = chunk

    return imputed

def create_scenario_masks(arrays, split, cfg, mask_type, mask_value, seed=None):
    """Create and save masks for one scenario.

    Returns:
        mask: [length, channels] array saved to disk
    """
    length = len(arrays[split]["power"])
    n_channels = arrays[split]["power"].shape[1]
    rng = np.random.default_rng(cfg["mask_seed"] if seed is None else seed)

    mask = create_row_mask(length, mask_type, mask_value, n_channels, rng)

    return mask

class ForecastWindowsCleanTarget(Dataset):
    """Dataset that uses imputed input but clean target labels."""

    def __init__(self, power_imputed, power_clean, starts, input_len, forecast_len):
        self.power_imputed = power_imputed
        self.power_clean = power_clean
        self.starts = starts
        self.input_len = input_len
        self.forecast_len = forecast_len
        self.total_len = input_len + forecast_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])

        p_input = torch.from_numpy(self.power_imputed[s:s + self.input_len])
        p_target = torch.from_numpy(self.power_clean[s + self.input_len:s + self.total_len])
        return p_input, p_target

def train_lstm_forecaster(power_imputed, val_imputed, arrays, indexes, cfg, work_dir, scenario_name, method, seed):
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
        power_imputed, arrays["train"]["power"], train_starts, input_len, forecast_len
    )

    # Validation uses the same imputed-history/clean-target condition as train.
    val_dataset = ForecastWindowsCleanTarget(
        val_imputed, arrays["val"]["power"], val_starts, input_len, forecast_len
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
        power_channels=len(POWER)
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_val, best_state, stale = float("inf"), None, 0
    logs = []

    print(f"  [LSTM {scenario_name}/{method}] Training on {len(train_starts)} windows")

    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"LSTM {method[:15]} seed={seed}", unit="epoch", ncols=100):
        net.train()
        train_loss = 0.0

        for p_in, p_out in train_loader:
            p_in, p_out = p_in.to(device), p_out.to(device)

            loss = net.loss_per_sample(p_in, p_out).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            train_loss += float(loss.detach())

        # Validation
        net.eval()
        val_sq, val_n = 0.0, 0
        with torch.no_grad():
            for p_in, p_out in tqdm(val_loader, desc="validation", leave=False, unit="batch"):
                p_in, p_out = p_in.to(device), p_out.to(device)

                pred = net(p_in)

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
        arrays["test"]["power"], arrays["test"]["power"], test_starts, input_len, forecast_len
    )

    test_loader = DataLoader(test_dataset, min(cfg["batch_size"], 256), shuffle=False, num_workers=cfg["num_workers"])

    # Collect all predictions and targets
    all_pred = []
    all_target = []

    net.eval()
    with torch.no_grad():
        for p_in, p_out in tqdm(test_loader, desc=f"Eval {method[:15]}", leave=False, unit="batch"):
            p_in = p_in.to(device)
            pred = net(p_in)

            all_pred.append(pred.cpu().numpy())
            all_target.append(p_out.numpy())

    # Concatenate
    all_pred = np.concatenate(all_pred, axis=0)  # [N, forecast_len, channels]
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
        "MAE": float(np.mean(all_errors)),
        "RMSE": float(np.sqrt(np.mean(all_errors ** 2))),
        "n_points": len(all_errors)
    }

    return result

def evaluate_imputation(power_clean, power_imputed, mask, scales, regimes, regime_names=REGIMES, thresholds=WIND_REGIME_THRESHOLDS):
    """Evaluate masked validation cells overall and by their wind regime."""
    if power_clean.shape != power_imputed.shape or power_clean.shape != mask.shape:
        raise ValueError("clean, imputed, and mask arrays must have the same shape")
    regimes = np.asarray(regimes, dtype=np.int64)
    regime_names = list(regime_names)
    thresholds = tuple(float(value) for value in thresholds)
    if len(regime_names) != len(thresholds) + 1 or len(regime_names) != len(REGIMES):
        raise ValueError("regime names and thresholds do not match the model's fixed embedding size")
    if regimes.shape != (power_clean.shape[0],):
        raise ValueError("one wind regime label is required per validation timestep")
    if np.any((regimes < 0) | (regimes >= len(REGIMES))):
        raise ValueError("wind regime labels are outside the configured range")

    missing = mask == 0
    rows = []
    for regime_id, regime_name in enumerate(regime_names):
        errors = []
        for ch, field in enumerate(POWER):
            selected = missing[:, ch] & (regimes == regime_id)
            if not np.any(selected):
                continue
            scale = scales["power"][field]
            span = scale["max"] - scale["min"] or 1
            truth = power_clean[selected, ch] * span + scale["min"]
            estimate = power_imputed[selected, ch] * span + scale["min"]
            errors.append(estimate - truth)
        values = np.concatenate(errors) if errors else np.array([], dtype=np.float64)
        rows.append({
            "regime_id": regime_id,
            "regime": regime_name,
            "wind_speed_range_m_s": _regime_range_label(regime_id, thresholds),
            "imputation_MAE": float(np.mean(np.abs(values))) if values.size else float("nan"),
            "imputation_RMSE": float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan"),
            "imputation_n_missing": int(values.size),
        })
    return rows


def _regime_range_label(regime_id, thresholds=WIND_REGIME_THRESHOLDS):
    bounds = [0.0, *thresholds, float("inf")]
    lower, upper = bounds[regime_id:regime_id + 2]
    lower_text = f"{lower:g}"
    upper_text = "inf" if not np.isfinite(upper) else f"{upper:g}"
    left = "[" if regime_id == 0 else "("
    return f"{left}{lower_text}, {upper_text}] m/s"


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

def pivot_imputation_results(imputation_rows):
    """Pivot validation metrics by scenario, seed, and wind regime."""
    grouped = {}
    for row in imputation_rows:
        key = (row["scenario"], row["seed"], row["regime_id"])
        grouped.setdefault(key, {
            "scenario": row["scenario"],
            "seed": row["seed"],
            "split": "validation",
            "regime_id": row["regime_id"],
            "regime": row["regime"],
            "wind_speed_range_m_s": row["wind_speed_range_m_s"],
        })
        method = row["method"]
        for metric in ("imputation_MAE", "imputation_RMSE", "imputation_n_missing"):
            grouped[key][f"{method}_{metric}"] = row[metric]

    return sorted(grouped.values(), key=lambda row: (row["scenario"], row["seed"], row["regime_id"]))


def pivot_results(eval_rows):
    """Pivot results: one row per scenario, columns for each method."""
    # Group by scenario and seed
    grouped = {}
    for row in eval_rows:
        key = (row["scenario"], row["seed"])
        if key not in grouped:
            grouped[key] = {"scenario": row["scenario"], "seed": row["seed"]}

        method = row["method"]
        for metric in ["MAE", "RMSE"]:
            col = f"{method}_{metric}"
            grouped[key][col] = row[metric]
        grouped[key]["n_points"] = row["n_points"]

    # Convert to list and sort
    output = sorted(grouped.values(), key=lambda x: (x["scenario"], x["seed"]))

    return output

def run(config_path: Path):
    """Main experiment pipeline."""
    root = config_path.parent
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))

    work_dir = (root / cfg.get("work_dir", "work")).resolve()
    results_dir = (root / cfg.get("results_dir", "results")).resolve()
    data_path = (root / cfg["data_path"]).resolve()
    legacy_dirs = {(root.resolve() / "work").resolve(), (root.resolve() / "results").resolve()}
    if work_dir in legacy_dirs or results_dir in legacy_dirs:
        raise ValueError("New-dataset runs must not write into the legacy work/ or results/ directories")
    if work_dir == results_dir or work_dir in results_dir.parents or results_dir in work_dir.parents:
        raise ValueError("work_dir and results_dir must be separate directories")
    if data_path == work_dir or data_path == results_dir:
        raise ValueError("data_path cannot be an output directory")

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
        "wind_regime": {
            "feature": cfg.get("wind_regime_key", WIND_REGIME_KEY),
            "thresholds_m_s": cfg.get("wind_regime_thresholds_m_s", WIND_REGIME_THRESHOLDS),
            "names": cfg.get("wind_regime_names", REGIMES),
            "threshold_source": "generic reference thresholds; calibrate to the turbine power curve",
        },
        "runs": []
    }

    # Phase 1: Train diffusion imputers
    print("\n" + "=" * 100)
    print("PHASE 1: TRAIN DIFFUSION IMPUTERS")
    print("=" * 100)

    diffusion_models = {}

    for seed in cfg["train_seeds"]:
        for method in ["diffusion_no_weather", "diffusion_with_weather"]:

            try:
                print(f"\n{'=' * 100}")
                print(f"Training {method} | seed={seed}")
                print('=' * 100)

                net, device, info = train_diffusion_imputer(
                    arrays, indexes, cfg, work_dir, seed, method
                )

                diffusion_models[(seed, method)] = (net, device)

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
    imputation_rows = []

    for mask_type, mask_value in scenarios(cfg):
        scenario_name = f"{mask_type}_{mask_value}"

        print(f"\n{'=' * 100}")
        print(f"SCENARIO: {scenario_name}")
        print('=' * 100)

        for seed in cfg["train_seeds"]:
            print(f"\nSeed: {seed}")

            # Create one shared mask per scenario/seed for all methods.
            print("  Creating masks...")
            mask_seed = int(cfg["mask_seed"]) + int(seed)
            train_mask = create_scenario_masks(arrays, "train", cfg, mask_type, mask_value, mask_seed)
            val_mask = create_scenario_masks(arrays, "val", cfg, mask_type, mask_value, mask_seed + 1)

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

                    diffusion_net, device = diffusion_models.get((seed, method), (None, None))
                    if method != "linear_interpolation" and diffusion_net is None:
                        print(f"    Diffusion model unavailable, skipping")
                        continue

                    print(f"    Imputing training and validation histories...")
                    train_imputed = impute_row_level_batched(
                        method, diffusion_net, device,
                        arrays["train"]["power"], arrays["train"]["weather"],
                        train_mask, cfg, regimes=arrays["train"]["cluster"],
                        segment_ids=arrays["train"]["segment"]
                    )
                    val_imputed = impute_row_level_batched(
                        method, diffusion_net, device,
                        arrays["val"]["power"], arrays["val"]["weather"],
                        val_mask, cfg, regimes=arrays["val"]["cluster"],
                        segment_ids=arrays["val"]["segment"]
                    )
                    imputation_metrics = evaluate_imputation(
                        arrays["val"]["power"], val_imputed, val_mask, scales,
                        arrays["val"]["cluster"],
                        cfg.get("wind_regime_names", REGIMES),
                        cfg.get("wind_regime_thresholds_m_s", WIND_REGIME_THRESHOLDS),
                    )

                    # Save imputed data
                    imputed_dir = work_dir / "imputed" / scenario_name / method
                    imputed_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(imputed_dir / f"train_imputed_seed{seed}.npz", power=train_imputed)
                    np.savez_compressed(imputed_dir / f"val_imputed_seed{seed}.npz", power=val_imputed)

                    # Train LSTM
                    print(f"    Training LSTM forecaster...")
                    lstm_net, lstm_device, lstm_info = train_lstm_forecaster(
                        train_imputed, val_imputed, arrays, indexes, cfg, work_dir,
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
                    for regime_metrics in imputation_metrics:
                        imputation_rows.append({
                            "scenario": scenario_name,
                            "seed": seed,
                            "method": method,
                            **regime_metrics,
                        })

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

        if imputation_rows:
            imputation_pivoted = pivot_imputation_results(imputation_rows)
            imputation_path = results_dir / "imputation_comparison.csv"
            write_csv(imputation_path, imputation_pivoted)
            print(f"✓ Imputation results exported: {imputation_path}")
            print(f"  {len(imputation_pivoted)} rows (validation scenarios × seeds)")
        else:
            print("✗ No imputation results to export")

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
