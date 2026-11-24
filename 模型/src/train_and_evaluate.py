"""Training and evaluation pipeline with row-level masking and imputation.

Three methods reconstruct the same masked training and validation power rows.
Both linear interpolation and diffusion imputation use offline, bidirectional
observed context within each continuous segment. Diffusion inputs are gap-centered
windows with real samples only. BiLSTM learns from reconstructed histories and
clean labels, and is tested on clean histories and labels.
"""
from __future__ import annotations
import csv, json, math, random, traceback
from pathlib import Path
import numpy as np, torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from data_pipeline import POWER, REGIME_NAMES, WIND_REGIME_THRESHOLDS, WIND_REGIME_KEY, build_dataset
from diffusion_model import ConditionalDiffusion
from lstm_forecaster import LSTMForecaster

REGIMES = REGIME_NAMES
IMPUTATION_METHODS = ["linear_interpolation", "diffusion_no_weather", "diffusion_with_weather"]

def seed_all(seed):
    """Fix Python, NumPy, and PyTorch randomness."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def scenarios(cfg):
    """Return all masking scenarios from config."""
    return cfg.get("scenarios", [])

def _segment_bounds(length, segment_ids):
    """Return half-open bounds of contiguous, regularly sampled runs."""
    if segment_ids is None:
        return [(0, length)] if length else []
    segment_ids = np.asarray(segment_ids)
    if segment_ids.shape != (length,):
        raise ValueError("segment_ids must have one value per row")
    edges = np.r_[0, np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1, length]
    return [(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:]) if start < end]


def _sample_feasible_gap_location(scenario, segments, rng, min_context=1):
    """Sample only gaps that fit at least one segment and two real anchors."""
    if min_context < 1:
        raise ValueError("min_context must be at least one observed row per side")
    max_gap = max((end - start - 2 * min_context for start, end in segments), default=0)
    if max_gap < 1:
        raise ValueError("no segment is long enough to contain a gap and two observed anchors")

    if scenario["type"] == "point":
        gap_length = 1
    elif scenario["type"] == "continuous":
        buckets = [name for name in ("short", "medium", "long") if f"{name}_lengths" in scenario]
        probabilities = np.asarray(scenario.get("length_probabilities", []), dtype=np.float64)
        if (not buckets or probabilities.shape != (len(buckets),)
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("gap length probabilities must match configured length buckets and sum to 1")
        feasible = []
        for bucket, probability in zip(buckets, probabilities):
            low, high = map(int, scenario[f"{bucket}_lengths"])
            if low < 1 or high < low:
                raise ValueError(f"invalid {bucket} gap range: {(low, high)}")
            feasible_high = min(high, max_gap)
            if probability > 0 and low <= feasible_high:
                feasible.append((bucket, float(probability), low, feasible_high))
        if not feasible:
            raise ValueError(f"no configured gap-length bucket fits segment maximum {max_gap}")
        weights = np.asarray([item[1] for item in feasible], dtype=np.float64)
        weights /= weights.sum()
        bucket, _, low, high = feasible[int(rng.choice(len(feasible), p=weights))]
        gap_length = int(rng.integers(low, high + 1))
    else:
        raise ValueError(f"unknown scenario type: {scenario['type']}")

    eligible = [(start, end) for start, end in segments
                if end - start >= gap_length + 2 * min_context]
    counts = np.asarray([end - start - gap_length - 2 * min_context + 1
                         for start, end in eligible], dtype=np.float64)
    if not len(eligible) or counts.sum() <= 0:
        raise ValueError(f"no segment can fit configured {gap_length}-point gap and context")
    selected = int(rng.choice(len(eligible), p=counts / counts.sum()))
    seg_start, seg_end = eligible[selected]
    first_start = seg_start + min_context
    gap_start = int(rng.integers(first_start, seg_end - min_context - gap_length + 1))
    return gap_length, seg_start, seg_end, gap_start


def _dynamic_gap_sample(split_arrays, cfg, rng):
    """Sample a clean gap and up to the configured real context on both sides."""
    by_type = {scenario["type"]: scenario for scenario in scenarios(cfg)}
    probabilities = cfg.get("diffusion_scenario_probabilities", {"point": 0.1, "continuous": 0.9})
    kinds = [kind for kind in ("point", "continuous") if kind in by_type]
    weights = np.asarray([probabilities.get(kind, 0.0) for kind in kinds], dtype=np.float64)
    if not kinds or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("diffusion_scenario_probabilities must select configured scenarios")
    weights /= weights.sum()
    scenario = by_type[kinds[int(rng.choice(len(kinds), p=weights))]]

    power = split_arrays["power"]
    segments = _segment_bounds(len(power), split_arrays.get("segment"))
    gap_length, seg_start, seg_end, gap_start = _sample_feasible_gap_location(
        scenario, segments, rng, min_context=1
    )
    gap_end = gap_start + gap_length
    context = int(cfg.get("diffusion_context_points", 18))
    window_start = max(seg_start, gap_start - context)
    window_end = min(seg_end, gap_end + context)

    clean = np.asarray(power[window_start:window_end], dtype=np.float32).copy()
    mask = np.ones_like(clean, dtype=np.float32)
    mask[gap_start - window_start:gap_end - window_start] = 0
    observed = clean * mask
    weather = np.asarray(split_arrays["weather"][window_start:window_end], dtype=np.float32)
    regimes = np.asarray(split_arrays["cluster"][window_start:window_end], dtype=np.int64)
    return clean, observed, mask, weather, regimes

def _collate_dynamic_windows(samples):
    """Pad variable-length examples and return a real-position mask."""
    batch_size = len(samples)
    max_len = max(len(sample[0]) for sample in samples)
    channels = samples[0][0].shape[1]
    weather_channels = samples[0][3].shape[1]
    clean = torch.zeros((batch_size, max_len, channels), dtype=torch.float32)
    observed = torch.zeros_like(clean)
    mask = torch.zeros_like(clean)
    weather = torch.zeros((batch_size, max_len, weather_channels), dtype=torch.float32)
    regimes = torch.zeros((batch_size, max_len), dtype=torch.long)
    valid = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, (p, obs, m, w, r) in enumerate(samples):
        n = len(p)
        clean[i, :n] = torch.as_tensor(p)
        observed[i, :n] = torch.as_tensor(obs)
        mask[i, :n] = torch.as_tensor(m)
        weather[i, :n] = torch.as_tensor(w)
        regimes[i, :n] = torch.as_tensor(r)
        valid[i, :n] = True
    return clean, observed, mask, weather, regimes, valid


def _dynamic_window_batches(split_arrays, sample_count, cfg, rng, batch_size):
    """Yield length-bucketed microbatches under an attention-memory budget."""
    reference_length = int(cfg["window_length"])
    max_attention_elements = int(batch_size * reference_length ** 2 * 4)
    chunk_size = batch_size * 4
    remaining = sample_count
    while remaining:
        count = min(chunk_size, remaining)
        samples = [_dynamic_gap_sample(split_arrays, cfg, rng) for _ in range(count)]
        samples.sort(key=lambda item: len(item[0]))
        batches = []
        start = 0
        while start < count:
            # Always allow one sample, even if its window alone exceeds the budget.
            end = start + 1
            limit = min(start + batch_size, count)
            while end < limit:
                padded_cost = (end - start + 1) * len(samples[end][0]) ** 2
                if padded_cost > max_attention_elements:
                    break
                end += 1
            batches.append(samples[start:end])
            start = end
        rng.shuffle(batches)
        for batch in batches:
            yield _collate_dynamic_windows(batch)
        remaining -= count


def create_row_mask(scenario, length, rng, n_channels=1, *, segment_ids=None, min_context=1):
    """Build isolated point or feasible continuous masks for each scenario."""
    if scenario["type"] not in {"point", "continuous"}:
        raise ValueError(f"unknown scenario type: {scenario['type']}")
    if length < 1 or n_channels < 1 or min_context < 1:
        raise ValueError("length, n_channels, and min_context must be positive")
    mask = np.ones((length, n_channels), dtype=np.float32)
    segments = _segment_bounds(length, segment_ids)

    if scenario["type"] == "point":
        low, high = map(float, scenario["missing_rates"])
        if not (0 <= low <= high <= 1):
            raise ValueError("missing_rates must be an ordered pair in [0, 1]")
        rate = rng.uniform(low, high)
        target = max(1, int(round(length * rate)))
        candidates = [(start + min_context, max(0, end - start - 2 * min_context))
                      for start, end in segments]
        candidates = [(first, count) for first, count in candidates if count > 0]
        capacity = sum((count + 1) // 2 for _, count in candidates)
        if capacity == 0:
            raise ValueError("no segment has an eligible isolated point-missing position")
        target = min(target, capacity)
        order = rng.permutation(len(candidates))
        remaining = target
        for index in order:
            first, count = candidates[int(index)]
            take = min(remaining, (count + 1) // 2)
            if take:
                compact = np.sort(rng.choice(count - take + 1, size=take, replace=False))
                selected = first + compact + np.arange(take)
                mask[selected, :] = 0
                remaining -= take
            if remaining == 0:
                break
        return mask

    gap_length, _, _, gap_start = _sample_feasible_gap_location(
        scenario, segments, rng, min_context=min_context
    )
    mask[gap_start:gap_start + gap_length, :] = 0
    return mask
def linear_interpolation(power, mask, segment_ids=None):
    """Fill only bounded gaps using the nearest observed values on both sides.

    Anchors can lie beyond a forecast history window, but not beyond the
    current continuous segment. No extrapolation or fallback is allowed.
    """
    power = np.asarray(power)
    mask = np.asarray(mask)
    if power.ndim != 2 or mask.shape != power.shape:
        raise ValueError("power and mask must have the same [length, channels] shape")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0 (missing) or 1 (observed)")
    if not np.all(np.isfinite(power[mask == 1])):
        raise ValueError("observed power must be finite")
    filled = power.astype(np.result_type(power.dtype, np.float32), copy=True)
    for start, end in _segment_bounds(len(power), segment_ids):
        for ch in range(power.shape[1]):
            observed = mask[start:end, ch] == 1
            missing = np.flatnonzero(~observed)
            if not missing.size:
                continue
            if not (observed[0] and observed[-1]):
                raise ValueError(f"linear interpolation requires two observed endpoints: rows [{start}, {end}), channel {ch}")
            known = np.flatnonzero(observed)
            filled[start + missing, ch] = np.interp(missing, known, power[start + known, ch])
    return filled


def train_diffusion_imputer(arrays, indexes, cfg, work_dir, seed, method_name):
    """Train diffusion model for imputation.

    Args:
        method_name: one of the two fixed diffusion experiment methods
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_classes = {
        "diffusion_no_weather": lambda **kw: ConditionalDiffusion(use_weather=False, **kw),
        "diffusion_with_weather": lambda **kw: ConditionalDiffusion(use_weather=True, **kw)
    }
    use_weather = method_name == "diffusion_with_weather"
    max_length = int(cfg.get(
        "max_diffusion_window_length",
        cfg["window_length"] + 2 * int(cfg.get("diffusion_context_points", 18))
    ))
    expected_max = max(
        1 if scenario["type"] == "point" else max(
            int(scenario[f"{bucket}_lengths"][1])
            for bucket in ("short", "medium", "long") if f"{bucket}_lengths" in scenario
        )
        for scenario in scenarios(cfg)
    )
    required_max = expected_max + 2 * int(cfg.get("diffusion_context_points", 18))
    if max_length < required_max:
        raise ValueError(f"max_diffusion_window_length={max_length} is below required {required_max}")

    # Model
    net = model_classes[method_name](
        model_dim=cfg["diffusion_model_dim"],
        layers=cfg["diffusion_layers"],
        heads=cfg["diffusion_heads"],
        timesteps=cfg["diffusion_train_steps"],
        max_length=max_length,
        power_channels=len(POWER),
        weather_channels=len(arrays["train"]["weather"][0])
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    train_samples = max(1, int(cfg.get("diffusion_train_samples_per_epoch", len(indexes["train"]))))
    val_samples = max(1, int(cfg.get("diffusion_validation_samples", min(len(indexes["val"]), 2048))))
    batch_size = int(cfg["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    best_val, best_state, stale = float("inf"), None, 0
    logs = []
    rng = np.random.default_rng(seed + 8000)

    print(f"[{method_name} seed={seed}] Training started: {train_samples} dynamic windows/epoch")

    for epoch in range(1, cfg["train_max_epochs"] + 1):
        net.train()
        train_loss_sum, train_seen = 0.0, 0
        for clean, observed, masks, weather, regimes, valid in _dynamic_window_batches(
                arrays["train"], train_samples, cfg, rng, batch_size):
            clean, observed = clean.to(device), observed.to(device)
            masks, weather = masks.to(device), weather.to(device)
            valid = valid.to(device)
            regimes = regimes.to(device) if use_weather else None
            loss = net.loss_per_sample(
                clean, observed, masks, weather if use_weather else None,
                regimes=regimes, valid_mask=valid
            ).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            current_batch = len(clean)
            train_loss_sum += float(loss.detach()) * current_batch
            train_seen += current_batch
        avg_train_loss = train_loss_sum / max(train_seen, 1)

        # Validate on freshly sampled dynamic gaps every 3 epochs and at epoch 1.
        if epoch == 1 or epoch % 3 == 0:
            net.eval()
            val_sq, val_n = 0.0, 0
            with torch.no_grad():
                val_rng = np.random.default_rng(seed + epoch)
                for clean, observed, masks, weather, regimes, valid in _dynamic_window_batches(
                        arrays["val"], val_samples, cfg, val_rng, batch_size):
                    clean, observed = clean.to(device), observed.to(device)
                    masks, weather = masks.to(device), weather.to(device)
                    valid = valid.to(device)
                    regimes = regimes.to(device) if use_weather else None
                    pred = net.sample(
                        observed, masks, weather if use_weather else None,
                        cfg["validation_ddim_steps"], 1, regimes=regimes, valid_mask=valid
                    )
                    missing = (1 - masks) * valid.unsqueeze(-1).to(masks.dtype)
                    val_sq += float((((pred - clean) ** 2) * missing).sum())
                    val_n += int(missing.sum())
            val_rmse = math.sqrt(val_sq / max(val_n, 1))
            logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_rmse": val_rmse})
            status = "★BEST" if val_rmse < best_val else f"stale={stale}/{cfg['early_stopping_patience']}"
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_rmse={val_rmse:.6f} {status}")
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg["early_stopping_patience"]:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_rmse": None})
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
        "max_diffusion_window_length": max_length,
        "diffusion_context_points": int(cfg.get("diffusion_context_points", 18)),
        "best_val_rmse": best_val
    }, ckpt_path)

    # Save training log
    log_path = work_dir / "logs" / f"train_{method_name}_seed{seed}.csv"
    write_csv(log_path, logs)

    return net, device, {"best_val_rmse": best_val, "epochs": len(logs), "checkpoint": str(ckpt_path)}

def impute_row_level_batched(method, diffusion_net, device, power_clean, weather, mask, cfg, regimes=None, segment_ids=None):
    """Impute each full masked run with the configured real context on each side."""
    power_clean = np.asarray(power_clean)
    mask = np.asarray(mask)
    length = len(power_clean)

    if method == "linear_interpolation":
        return linear_interpolation(power_clean, mask, segment_ids)
    if method not in {"diffusion_no_weather", "diffusion_with_weather"}:
        raise ValueError(f"unknown imputation method: {method}")
    if power_clean.ndim != 2 or mask.shape != power_clean.shape:
        raise ValueError("power and mask must have the same [length, channels] shape")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0 (missing) or 1 (observed)")
    if not np.all(np.isfinite(power_clean[mask == 1])):
        raise ValueError("observed power must be finite")

    weather = np.asarray(weather)
    if weather.ndim != 2 or weather.shape[0] != length:
        raise ValueError("weather must have shape [length, weather_channels]")
    use_weather = diffusion_net.use_weather
    if use_weather:
        if regimes is None or np.asarray(regimes).shape != (length,):
            raise ValueError("weather-conditioned imputation requires one wind regime label per timestep")
        regimes = np.asarray(regimes, dtype=np.int64)
        if np.any((regimes < 0) | (regimes >= len(REGIMES))):
            raise ValueError(f"wind regime labels must be in [0, {len(REGIMES) - 1}]")

    segment = (np.asarray(segment_ids) if segment_ids is not None
               else np.zeros(length, dtype=np.int32))
    if segment.shape != (length,):
        raise ValueError("segment_ids must have one value per row")
    context = int(cfg.get("diffusion_context_points", 18))
    max_window = min(
        int(cfg.get("max_diffusion_window_length", diffusion_net.position.shape[1])),
        int(diffusion_net.position.shape[1]),
    )
    if context < 0 or max_window < 3:
        raise ValueError("diffusion context must be nonnegative and max window must be at least 3")

    # Zero only explicitly missing coordinates; all context rows remain real values.
    power_masked = np.where(mask.astype(bool), power_clean, 0.0).astype(np.float32, copy=False)
    imputed = np.where(mask.astype(bool), power_clean, np.nan).astype(np.float32, copy=False)
    windows = {}
    for seg_start, seg_end in _segment_bounds(length, segment):
        if seg_end - seg_start < 3:
            if np.any(mask[seg_start:seg_end] == 0):
                raise ValueError(f"segment [{seg_start}, {seg_end}) is too short for bidirectional imputation")
            continue
        for channel in range(power_clean.shape[1]):
            missing_positions = np.flatnonzero(mask[seg_start:seg_end, channel] == 0) + seg_start
            if not missing_positions.size:
                continue
            breaks = np.flatnonzero(np.diff(missing_positions) > 1) + 1
            for run in np.split(missing_positions, breaks):
                run_start, run_end = int(run[0]), int(run[-1]) + 1
                if (run_start <= seg_start or run_end >= seg_end
                        or mask[run_start - 1, channel] != 1
                        or mask[run_end, channel] != 1):
                    raise ValueError(f"missing run [{run_start}, {run_end}) lacks two observed anchors")

                left_count = 0
                cursor = run_start - 1
                while cursor >= seg_start and mask[cursor, channel] == 1 and left_count < context:
                    left_count += 1
                    cursor -= 1
                right_count = 0
                cursor = run_end
                while cursor < seg_end and mask[cursor, channel] == 1 and right_count < context:
                    right_count += 1
                    cursor += 1
                window_start = run_start - left_count
                window_end = run_end + right_count
                window_len = window_end - window_start
                if window_len > max_window:
                    raise ValueError(
                        f"missing run [{run_start}, {run_end}) needs a {window_len}-point window, "
                        f"but the model maximum is {max_window}"
                    )
                key = (window_start, window_end)
                windows.setdefault(key, []).append((channel, run_start, run_end))

    if not windows:
        return power_clean.copy()

    window_items = list(windows.items())
    batch_size = max(1, int(cfg.get("diffusion_imputation_batch_size", 32)))
    for current_window_len in sorted({end - start for (start, end), _ in window_items}):
        length_items = [item for item in window_items if item[0][1] - item[0][0] == current_window_len]
        for batch_start in tqdm(
                range(0, len(length_items), batch_size),
                desc=f"Imputing ({method}, L={current_window_len})", leave=False):
            batch = length_items[batch_start:batch_start + batch_size]
            p_batch, m_batch, w_batch, regimes_batch = [], [], [], []
            for (window_start, window_end), _ in batch:
                p_batch.append(power_masked[window_start:window_end])
                m_batch.append(mask[window_start:window_end].astype(np.float32, copy=False))
                w_batch.append(weather[window_start:window_end].astype(np.float32, copy=False))
                if use_weather:
                    regimes_batch.append(regimes[window_start:window_end])

            p_tensor = torch.from_numpy(np.stack(p_batch)).to(device)
            m_tensor = torch.from_numpy(np.stack(m_batch)).to(device)
            w_tensor = torch.from_numpy(np.stack(w_batch)).to(device)
            r_tensor = (torch.as_tensor(np.stack(regimes_batch), dtype=torch.long, device=device)
                        if use_weather else None)
            with torch.no_grad():
                pred_batch = diffusion_net.sample(
                    p_tensor, m_tensor, w_tensor if use_weather else None,
                    cfg["ddim_steps"], cfg["ddim_samples"], regimes=r_tensor
                )
            predictions = pred_batch.detach().cpu().numpy()
            for batch_index, ((window_start, window_end), jobs) in enumerate(batch):
                for channel, run_start, run_end in jobs:
                    local_start = run_start - window_start
                    local_end = run_end - window_start
                    imputed[run_start:run_end, channel] = np.clip(
                        predictions[batch_index, local_start:local_end, channel], 0.0, 1.0
                    )

    if not np.all(np.isfinite(imputed)):
        raise RuntimeError("diffusion imputation left masked values unfilled")
    clean_float = power_clean.astype(np.float32, copy=False)
    if not np.array_equal(imputed[mask == 1], clean_float[mask == 1]):
        raise RuntimeError("diffusion imputation unexpectedly changed observed power values")
    return imputed
def create_scenario_masks(arrays, split, cfg, scenario, seed=None):
    """Build a shared artificial mask for one forecast-data split."""
    if split not in {"train", "val"}:
        raise ValueError("artificial masks are restricted to training and validation splits")
    length = len(arrays[split]["power"])
    n_channels = arrays[split]["power"].shape[1]
    rng = np.random.default_rng(cfg["mask_seed"] if seed is None else seed)

    return create_row_mask(scenario, length, rng,
                           n_channels=n_channels,
                           segment_ids=arrays[split]["segment"])

class ForecastWindowsCleanTarget(Dataset):
    """Dataset that uses imputed input but clean target labels."""

    def __init__(self, power_imputed, power_clean, starts, input_len, forecast_len):
        self.power_imputed = power_imputed
        self.power_clean = power_clean
        self.starts = starts
        self.input_len = input_len
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
        forecast_steps=forecast_len,
        hidden_dim=cfg["lstm_hidden_dim"],
        layers=cfg["lstm_layers"],
        dropout=cfg["lstm_dropout"],
        power_channels=len(POWER)
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_val, best_state, stale = float("inf"), None, 0
    logs = []

    print(f"[LSTM {scenario_name}/{method} seed={seed}] Training started: {len(train_starts)} windows")

    for epoch in range(1, cfg["train_max_epochs"] + 1):
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
            for p_in, p_out in val_loader:
                p_in, p_out = p_in.to(device), p_out.to(device)

                pred = net(p_in)

                val_sq += float(((pred - p_out) ** 2).sum())
                val_n += int(p_out.numel())

        val_rmse = math.sqrt(val_sq / max(val_n, 1))
        avg_train_loss = train_loss / max(len(train_loader), 1)

        logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_rmse": val_rmse})

        status = "★BEST" if val_rmse < best_val else f"stale={stale}/{cfg['early_stopping_patience']}"
        if epoch % 5 == 0 or val_rmse < best_val:  # Print every 5 epochs or on new best
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_rmse={val_rmse:.6f} {status}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch}")
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
    """Pivot validation metrics by scenario, seed, and wind regime.
    Columns grouped by metric: all MAE together, all RMSE together, all n_missing together.
    """
    grouped = {}
    for row in imputation_rows:
        key = (row["scenario"], row["seed"], row["regime_id"])
        if key not in grouped:
            grouped[key] = {
                "scenario": row["scenario"],
                "seed": row["seed"],
                "split": "validation",
                "regime_id": row["regime_id"],
                "regime": row["regime"],
                "wind_speed_range_m_s": row["wind_speed_range_m_s"],
            }
        method = row["method"]
        grouped[key][f"{method}_imputation_MAE"] = row["imputation_MAE"]
        grouped[key][f"{method}_imputation_RMSE"] = row["imputation_RMSE"]
        grouped[key][f"{method}_imputation_n_missing"] = row["imputation_n_missing"]

    # Reorder columns: group by metric (MAE, RMSE, n_missing)
    methods = ["linear_interpolation", "diffusion_no_weather", "diffusion_with_weather"]
    output = []
    for row in sorted(grouped.values(), key=lambda row: (row["scenario"], row["seed"], row["regime_id"])):
        new_row = {k: row[k] for k in ["scenario", "seed", "split", "regime_id", "regime", "wind_speed_range_m_s"]}
        for m in methods:
            new_row[f"{m}_imputation_MAE"] = row.get(f"{m}_imputation_MAE", "")
        for m in methods:
            new_row[f"{m}_imputation_RMSE"] = row.get(f"{m}_imputation_RMSE", "")
        for m in methods:
            new_row[f"{m}_imputation_n_missing"] = row.get(f"{m}_imputation_n_missing", "")
        output.append(new_row)

    return output


def pivot_results(eval_rows):
    """Pivot available forecasts without requiring every method to finish."""
    grouped = {}
    for row in eval_rows:
        key = (row["scenario"], row["seed"])
        if key not in grouped:
            grouped[key] = {"scenario": row["scenario"], "seed": row["seed"]}
        method = row["method"]
        for metric in ["MAE", "RMSE"]:
            grouped[key][f"{method}_{metric}"] = row[metric]
        grouped[key]["n_points"] = row["n_points"]

    methods = ["linear_interpolation", "diffusion_no_weather", "diffusion_with_weather"]
    output = []
    for row in sorted(grouped.values(), key=lambda x: (x["scenario"], x["seed"])):
        new_row = {"scenario": row["scenario"], "seed": row["seed"]}
        for m in methods:
            new_row[f"{m}_MAE"] = row.get(f"{m}_MAE", "")
        for m in methods:
            new_row[f"{m}_RMSE"] = row.get(f"{m}_RMSE", "")
        new_row["n_points"] = row["n_points"]
        output.append(new_row)

    return output


def export_results(results_dir, work_dir, eval_rows, imputation_rows, log):
    """Persist completed metrics and run state so interruptions retain usable outputs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    if eval_rows:
        write_csv(results_dir / "forecast_imputation_comparison.csv", pivot_results(eval_rows))
    if imputation_rows:
        write_csv(results_dir / "imputation_comparison.csv", pivot_imputation_results(imputation_rows))
    log_path = work_dir / "logs" / "run_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

def run(config_path: Path):
    """Main experiment pipeline."""
    root = config_path.parent
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))

    work_dir = (root / cfg.get("work_dir", "work")).resolve()
    results_dir = (root / cfg.get("results_dir", "results")).resolve()
    diffusion_path = (root / cfg["diffusion_data_path"]).resolve(strict=True)
    forecast_path = (root / cfg["forecast_data_path"]).resolve(strict=True)
    if diffusion_path.samefile(forecast_path):
        raise ValueError("diffusion and forecast datasets must be separate files")
    if work_dir == results_dir or work_dir in results_dir.parents or results_dir in work_dir.parents:
        raise ValueError("work_dir and results_dir must be separate directories")

    diffusion_work_dir = work_dir / "diffusion"
    forecast_work_dir = work_dir / "forecast"
    print(f"Diffusion training and validation: {diffusion_path.name}")
    diffusion_arrays, diffusion_indexes, source_scales, diffusion_audit = build_dataset(
        diffusion_path, diffusion_work_dir, cfg
    )
    print(f"BiLSTM training, validation and test: {forecast_path.name}")
    arrays, indexes, scales, forecast_audit = build_dataset(
        forecast_path, forecast_work_dir, cfg, scalers_override=source_scales
    )
    if diffusion_audit["data_sha256"] == forecast_audit["data_sha256"]:
        raise ValueError("diffusion and forecast datasets contain identical data")
    for stage, starts, splits in (
        ("diffusion", diffusion_indexes, ("train", "val")),
        ("forecast", indexes, ("train", "val", "test")),
    ):
        for split in splits:
            if not len(starts[split]):
                raise ValueError(f"{stage} {split} split contains no usable windows")
            print(f"  {stage} {split}: {len(starts[split])} windows")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    log = {
        "device": device_name,
        "status": "running",
        "audit": {"diffusion": diffusion_audit, "forecast": forecast_audit},
        "config": cfg,
        "wind_regime": {
            "feature": cfg.get("wind_regime_key", WIND_REGIME_KEY),
            "thresholds_m_s": cfg.get("wind_regime_thresholds_m_s", WIND_REGIME_THRESHOLDS),
            "names": cfg.get("wind_regime_names", REGIMES),
            "threshold_source": "generic reference thresholds; calibrate to the turbine power curve",
        },
        "runs": []
    }
    export_results(results_dir, work_dir, [], [], log)

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
                    diffusion_arrays, diffusion_indexes, cfg, diffusion_work_dir, seed, method
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
                export_results(results_dir, work_dir, [], [], log)

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
                export_results(results_dir, work_dir, [], [], log)

    # Phase 2-4: For each scenario, mask → impute → train LSTM → evaluate
    print("\n" + "=" * 100)
    print("PHASE 2-4: MASK → IMPUTE → TRAIN LSTM → EVALUATE")
    print("=" * 100)

    eval_rows = []
    imputation_rows = []
    export_results(results_dir, work_dir, eval_rows, imputation_rows, log)
    print(f"Partial result files will be updated under: {results_dir}")

    for scenario in scenarios(cfg):
        scenario_name = scenario["name"]

        print(f"\n{'=' * 100}")
        print(f"SCENARIO: {scenario_name}")
        print('=' * 100)

        for seed in cfg["train_seeds"]:
            print(f"\nSeed: {seed}")

            # Create one shared mask per scenario/seed for all methods.
            print("  Creating masks...")
            mask_seed = int(cfg["mask_seed"]) + int(seed)
            train_mask = create_scenario_masks(arrays, "train", cfg, scenario, mask_seed)
            val_mask = create_scenario_masks(arrays, "val", cfg, scenario, mask_seed + 1)

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

                    for regime_metrics in imputation_metrics:
                        imputation_rows.append({
                            "scenario": scenario_name,
                            "seed": seed,
                            "method": method,
                            **regime_metrics,
                        })
                    log["last_completed_imputation"] = {
                        "scenario": scenario_name,
                        "seed": seed,
                        "method": method,
                    }
                    export_results(results_dir, work_dir, eval_rows, imputation_rows, log)

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
                    export_results(results_dir, work_dir, eval_rows, imputation_rows, log)

                    # Evaluate
                    print(f"    Evaluating on test set...")
                    result = evaluate_forecaster(
                        lstm_net, lstm_device, arrays, indexes, cfg, scales,
                        scenario_name, method, seed
                    )

                    eval_rows.append(result)
                    for run_record in reversed(log["runs"]):
                        if (run_record.get("phase") == "lstm_training"
                                and run_record.get("scenario") == scenario_name
                                and run_record.get("method") == method
                                and run_record.get("seed") == seed):
                            run_record["status"] = "completed"
                            run_record["metrics"] = {key: result[key] for key in ("MAE", "RMSE", "n_points")}
                            break
                    export_results(results_dir, work_dir, eval_rows, imputation_rows, log)

                    print(f"    ✓ MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}")

                    del lstm_net

                except Exception as e:
                    print(f"    ✗ FAILED: {repr(e)}")
                    traceback.print_exc()
                    for run_record in reversed(log["runs"]):
                        if (run_record.get("phase") == "lstm_training"
                                and run_record.get("scenario") == scenario_name
                                and run_record.get("method") == method
                                and run_record.get("seed") == seed):
                            run_record["status"] = "failed"
                            run_record["error"] = repr(e)
                            break
                    else:
                        log["runs"].append({
                            "phase": "lstm_training_or_eval",
                            "scenario": scenario_name,
                            "method": method,
                            "seed": seed,
                            "status": "failed",
                            "error": repr(e)
                        })
                    export_results(results_dir, work_dir, eval_rows, imputation_rows, log)

    # Mark the run complete only when every expected forecast has a result.
    completed_keys = {
        (row["scenario"], row["method"], row["seed"]) for row in eval_rows
    }
    expected_keys = {
        (scenario["name"], method, seed)
        for scenario in scenarios(cfg)
        for method in IMPUTATION_METHODS
        for seed in cfg["train_seeds"]
    }
    completed = len(completed_keys)
    failed = sum(1 for r in log["runs"] if r.get("status") == "failed")
    log["status"] = "completed" if completed_keys == expected_keys else "partial"
    log["completed_forecasts"] = completed
    log["expected_forecasts"] = len(expected_keys)
    log["failed_runs"] = failed
    export_results(results_dir, work_dir, eval_rows, imputation_rows, log)

    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    print(f"Device: {device_name}")
    print(f"Total runs: {len(log['runs'])}")
    print(f"  ✓ Completed: {completed}")
    print(f"  ✗ Failed: {failed}")
    print("=" * 100 + "\n")

    return log
