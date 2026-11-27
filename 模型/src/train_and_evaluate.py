"""Training and evaluation pipeline with row-level masking and imputation.

Diffusion training and forecast training use the same row-mask generator on
separate time splits. Three methods reconstruct the same masked forecast-training
rows using offline, bidirectional context within each continuous segment.
LSTM learns from reconstructed histories and clean labels; validation and test
histories remain clean.
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


def _continuous_gap_buckets(scenario):
    keys = [f"{name}_lengths" for name in ("short", "medium", "long")
            if f"{name}_lengths" in scenario]
    weights = np.asarray(scenario["length_probabilities"], dtype=np.float64)
    if (not keys or weights.shape != (len(keys),)
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("gap length probabilities must match configured length buckets and sum to 1")
    buckets = []
    for key, weight in zip(keys, weights):
        low, high = map(int, scenario[key])
        if not 2 <= low <= high:
            raise ValueError(f"invalid {key}: {(low, high)}")
        if weight > 0:
            buckets.append((low, high, weight))
    return buckets


def _gap_windows(mask, segment_ids, context, max_window):
    """Yield the same whole-gap windows for Diffusion training and imputation."""
    if context < 1 or max_window < 3:
        raise ValueError("Diffusion windows need at least one observed row on each side")
    for seg_start, seg_end in _segment_bounds(len(mask), segment_ids):
        for channel in range(mask.shape[1]):
            missing = mask[seg_start:seg_end, channel] == 0
            edges = np.flatnonzero(np.diff(np.r_[False, missing, False])) + seg_start
            starts, ends = edges[::2], edges[1::2]
            for i, (run_start, run_end) in enumerate(zip(starts, ends)):
                if run_start == seg_start or run_end == seg_end:
                    raise ValueError(f"missing run [{run_start}, {run_end}) lacks two observed anchors")
                left = ends[i - 1] if i else seg_start
                right = starts[i + 1] if i + 1 < len(starts) else seg_end
                start = max(left, run_start - context)
                end = min(right, run_end + context)
                if end - start > max_window:
                    raise ValueError(
                        f"missing run [{run_start}, {run_end}) needs a {end - start}-point window, "
                        f"but the model maximum is {max_window}"
                    )
                yield start, end, channel, run_start, run_end


def _masked_gap_samples(split_arrays, scenario, cfg, rng):
    """Visit each gap once per mask, then draw a fresh mask for this scenario."""
    power = split_arrays["power"]
    segment_ids = split_arrays.get("segment")
    context = int(cfg.get("diffusion_context_points", 18))
    max_window = int(cfg.get("max_diffusion_window_length", cfg["window_length"] + 2 * context))
    while True:
        mask = create_row_mask(scenario, len(power), rng, power.shape[1], segment_ids=segment_ids)
        windows = list(_gap_windows(mask, segment_ids, context, max_window))
        rng.shuffle(windows)
        for start, end, _, _, _ in windows:
            clean = np.asarray(power[start:end], dtype=np.float32)
            local_mask = mask[start:end]
            weather = np.asarray(split_arrays["weather"][start:end], dtype=np.float32)
            regimes = np.asarray(split_arrays["cluster"][start:end], dtype=np.int64)
            yield clean, clean * local_mask, local_mask, weather, regimes

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
    by_type = {scenario["type"]: scenario for scenario in scenarios(cfg)}
    probabilities = cfg.get("diffusion_scenario_probabilities", {"point": 0.1, "continuous": 0.9})
    kinds = list(by_type)
    weights = np.asarray([probabilities.get(kind, 0.0) for kind in kinds], dtype=np.float64)
    if not kinds or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("diffusion_scenario_probabilities must select configured scenarios")
    weights /= weights.sum()
    pools = [_masked_gap_samples(split_arrays, by_type[kind], cfg, rng) for kind in kinds]
    remaining = sample_count
    while remaining:
        count = min(chunk_size, remaining)
        choices = rng.choice(len(pools), size=count, p=weights)
        samples = [next(pools[i]) for i in choices]
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


def _continuous_gap_limits(target, buckets, max_gap):
    """Choose integer block quotas before drawing lengths or positions."""
    lows, highs, weights = map(np.asarray, zip(*buckets))
    lows, highs = lows.astype(int), np.minimum(highs.astype(int), max_gap)
    if np.any(lows > highs) or target < lows.sum():
        raise ValueError(f"continuous missing budget ({target} rows) or segment lengths cannot fit every configured gap tier")

    # Prefer a block count that keeps the length draws near their configured midpoints.
    typical_count = target / np.dot(weights, (lows + highs) / 2)
    candidates = range(max(len(buckets), math.ceil(target / int(highs.max()))),
                       target // int(lows.min()) + 1)
    for count in sorted(candidates, key=lambda n: abs(n - typical_count)):
        ideal = count * weights
        quotas = np.floor(ideal).astype(int)
        order = np.argsort(quotas - ideal, kind="stable")
        quotas[order[:count - int(quotas.sum())]] += 1
        # Preserve every positive-weight tier when integer rounding would omit it.
        for tier in np.flatnonzero(quotas == 0):
            donor = int(np.argmax(np.where(quotas > 1, quotas - ideal, -np.inf)))
            quotas[donor] -= 1
            quotas[tier] = 1
        if np.dot(quotas, lows) <= target <= np.dot(quotas, highs):
            return np.repeat(lows, quotas), np.repeat(highs, quotas)
    raise ValueError(f"continuous missing budget ({target} rows) cannot meet the configured gap quotas")


def _plan_continuous_gaps(target, buckets, segments, rng, min_context):
    """Draw exact-budget lengths, then pack and randomly place the blocks."""
    capacity = np.asarray([max(0, end - start - min_context) for start, end in segments])
    max_gap = int(capacity.max(initial=0)) - min_context
    lows, highs = _continuous_gap_limits(target, buckets, max_gap)
    if target + len(lows) * min_context > capacity.sum():
        raise ValueError("continuous gap quotas and observed anchors exceed the available segment capacity")

    # Retry if a random length draw/allocation does not pack into fragmented segments.
    for _ in range(32):
        order = rng.permutation(len(lows))
        remaining, low_left, high_left = target, int(lows.sum()), int(highs.sum())
        lengths = []
        for low, high in zip(lows[order], highs[order]):
            low_left -= int(low)
            high_left -= int(high)
            draw = int(rng.integers(max(int(low), remaining - high_left),
                                    min(int(high), remaining - low_left) + 1))
            lengths.append(draw)
            remaining -= draw

        free = capacity.copy()
        assigned = [[] for _ in segments]
        for gap in sorted(lengths, reverse=True):
            cost = gap + min_context
            eligible = np.flatnonzero(free >= cost)
            if not eligible.size:
                break
            starts = free[eligible] - cost + 1
            index = int(rng.choice(eligible, p=starts / starts.sum()))
            assigned[index].append(gap)
            free[index] -= cost
        else:
            placed = []
            for (start, _), block_lengths, slack in zip(segments, assigned, free):
                if not block_lengths:
                    continue
                rng.shuffle(block_lengths)
                # Sample a random partition of the slack before, between and after blocks.
                n = len(block_lengths)
                bars = np.sort(rng.choice(int(slack) + n, size=n, replace=False))
                extras = np.diff(np.r_[-1, bars]) - 1
                cursor = start + min_context
                for gap, extra in zip(block_lengths, extras):
                    gap_start = cursor + int(extra)
                    placed.append((gap_start, gap_start + gap))
                    cursor = gap_start + gap + min_context
            return placed
    raise ValueError("could not place continuous gap quotas within the available segments after 32 attempts")


def create_row_mask(scenario, length, rng, n_channels=1, *, segment_ids=None, min_context=1):
    """Build one isolated-point or multi-block mask over an entire split."""
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

    rate = float(scenario["missing_rate"])
    if not 0 < rate < 1:
        raise ValueError("continuous missing_rate must be in (0, 1)")
    target = int(round(length * rate))
    buckets = _continuous_gap_buckets(scenario)
    for gap_start, gap_end in _plan_continuous_gaps(target, buckets, segments, rng, min_context):
        mask[gap_start:gap_end, :] = 0
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


@torch.no_grad()
def _evaluate_diffusion_validation(net, device, split_arrays, cfg, sample_count, batch_size,
                                   validation_seed, *, steps, draws):
    """Regenerate the same examples, batch order and initial noise at every check."""
    net.eval()
    val_rng = np.random.default_rng(validation_seed)
    val_noise = torch.Generator(device=device).manual_seed(validation_seed)
    val_abs, val_sq, val_n = 0.0, 0.0, 0
    for clean, observed, masks, weather, regimes, valid in _dynamic_window_batches(
            split_arrays, sample_count, cfg, val_rng, batch_size):
        clean, observed, masks, valid = (x.to(device) for x in (clean, observed, masks, valid))
        pred = net.sample(
            observed, masks, weather.to(device) if net.use_weather else None,
            steps, draws, regimes=regimes.to(device) if net.use_weather else None,
            valid_mask=valid, generator=val_noise,
        )
        missing = (1 - masks) * valid.unsqueeze(-1).to(masks.dtype)
        error = pred - clean
        val_abs += float((error.abs() * missing).sum())
        val_sq += float((error.square() * missing).sum())
        val_n += int(missing.sum())
    return val_abs / max(val_n, 1), math.sqrt(val_sq / max(val_n, 1))


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
    recheck_top_k = cfg.get("diffusion_validation_recheck_top_k", 0)
    if type(recheck_top_k) is not int or recheck_top_k < 0:
        raise ValueError("diffusion_validation_recheck_top_k must be a nonnegative integer")
    best_val, best_state, best_epoch, stale = float("inf"), None, None, 0
    candidates, logs = [], []
    rng = np.random.default_rng(seed + 8000)
    train_noise_seed = seed + 12000
    train_noise = torch.Generator(device=device).manual_seed(train_noise_seed)
    validation_seed = seed + 1

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
                regimes=regimes, valid_mask=valid, generator=train_noise
            ).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            current_batch = len(clean)
            train_loss_sum += float(loss.detach()) * current_batch
            train_seen += current_batch
        avg_train_loss = train_loss_sum / max(train_seen, 1)

        # Pair validation examples and initial noise across methods and epochs.
        if epoch == 1 or epoch % 3 == 0:
            val_mae, val_rmse = _evaluate_diffusion_validation(
                net, device, arrays["val"], cfg, val_samples, batch_size, validation_seed,
                steps=cfg["validation_ddim_steps"], draws=1,
            )
            keep_candidate = recheck_top_k > 0 and math.isfinite(val_mae) and (
                len(candidates) < recheck_top_k or val_mae < candidates[-1][0]
            )
            if val_mae < best_val or keep_candidate:
                snapshot = {k: v.detach().to(device="cpu", copy=True) for k, v in net.state_dict().items()}
                if keep_candidate:
                    candidates.append((val_mae, epoch, snapshot))
                    candidates.sort(key=lambda item: (item[0], item[1]))
                    del candidates[recheck_top_k:]
            logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_mae": val_mae, "val_rmse": val_rmse})
            status = "★BEST" if val_mae < best_val else f"stale={stale}/{cfg['early_stopping_patience']}"
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_mae={val_mae:.6f}, val_rmse={val_rmse:.6f} {status}")
            if val_mae < best_val:
                best_val, best_epoch, best_state = val_mae, epoch, snapshot
                stale = 0
            else:
                stale += 1
                if stale >= cfg["early_stopping_patience"]:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_mae": None, "val_rmse": None})
    if best_state is None:
        raise RuntimeError(f"{method_name} training produced no checkpoint")

    selected_epoch, selected_mae, selected_state = best_epoch, best_val, best_state
    rechecks = []
    for _, candidate_epoch, candidate_state in candidates:
        net.load_state_dict(candidate_state)
        mae, rmse = _evaluate_diffusion_validation(
            net, device, arrays["val"], cfg, val_samples, batch_size, validation_seed,
            steps=cfg["ddim_steps"], draws=cfg["ddim_samples"],
        )
        if not math.isfinite(mae) or not math.isfinite(rmse):
            raise RuntimeError(f"{method_name} produced nonfinite validation metrics at epoch {candidate_epoch}")
        rechecks.append({"epoch": candidate_epoch, "mae": mae, "rmse": rmse})
        print(f"  DDIM recheck epoch {candidate_epoch}: val_mae={mae:.6f}, val_rmse={rmse:.6f}")
    if rechecks:
        selected = min(rechecks, key=lambda row: (row["mae"], row["epoch"]))
        selected_epoch, selected_mae = selected["epoch"], selected["mae"]
        selected_state = next(state for _, epoch, state in candidates if epoch == selected_epoch)

    net.load_state_dict(selected_state)
    net.eval()
    selection = {
        "best_val_mae": best_val, "best_validation_epoch": best_epoch,
        "selected_epoch": selected_epoch, "selected_val_mae": selected_mae,
        "validation_seed": validation_seed, "train_noise_seed": train_noise_seed,
        "validation_rechecks": rechecks,
    }

    # Save checkpoint
    ckpt_path = work_dir / "checkpoints" / f"{method_name}_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": selected_state,
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
        **selection,
    }, ckpt_path)

    # Save training log
    log_path = work_dir / "logs" / f"train_{method_name}_seed{seed}.csv"
    write_csv(log_path, logs)

    return net, device, {**selection, "epochs": len(logs), "checkpoint": str(ckpt_path)}

def impute_row_level_batched(method, diffusion_net, device, power_clean, weather, mask, cfg, regimes=None, segment_ids=None, *, noise_seed=None):
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

    context = int(cfg.get("diffusion_context_points", 18))
    max_window = min(
        int(cfg.get("max_diffusion_window_length", diffusion_net.position.shape[1])),
        int(diffusion_net.position.shape[1]),
    )

    # Zero only explicitly missing coordinates; all context rows remain real values.
    power_masked = np.where(mask.astype(bool), power_clean, 0.0).astype(np.float32, copy=False)
    imputed = np.where(mask.astype(bool), power_clean, np.nan).astype(np.float32, copy=False)
    windows = {}
    for start, end, channel, run_start, run_end in _gap_windows(mask, segment_ids, context, max_window):
        windows.setdefault((start, end), []).append((channel, run_start, run_end))

    if not windows:
        return power_clean.copy()

    # Restart the same independent noise stream for each paired imputation method.
    noise_generator = (torch.Generator(device=device).manual_seed(noise_seed)
                       if noise_seed is not None else None)
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
                    cfg["ddim_steps"], cfg["ddim_samples"], regimes=r_tensor, generator=noise_generator
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
def create_forecast_train_mask(arrays, scenario, seed):
    """Mask only the LSTM training split, identically for all imputation methods."""
    train = arrays["train"]
    length, n_channels = train["power"].shape
    return create_row_mask(scenario, length, np.random.default_rng(seed),
                           n_channels=n_channels, segment_ids=train["segment"])

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

def _new_lstm_forecaster(cfg):
    return LSTMForecaster(
        forecast_steps=cfg["forecast_steps"],
        hidden_dim=cfg["lstm_hidden_dim"],
        layers=cfg["lstm_layers"],
        dropout=cfg["lstm_dropout"],
        power_channels=len(POWER),
    )


def train_lstm_forecaster(power_imputed, arrays, indexes, cfg, work_dir, scenario_name, method, seed,
                          *, initial_state):
    """Train on repaired histories and select weights using clean validation data."""
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

    # Validation uses the original power for both histories and targets.
    val_power = arrays["val"]["power"]
    val_dataset = ForecastWindowsCleanTarget(val_power, val_power, val_starts, input_len, forecast_len)

    shuffle_seed = seed + 16000
    train_loader = DataLoader(
        train_dataset, cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
        generator=torch.Generator().manual_seed(shuffle_seed),
    )
    val_loader = DataLoader(
        val_dataset, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
        generator=torch.Generator().manual_seed(seed + 17000),
    )

    net = _new_lstm_forecaster(cfg)
    net.load_state_dict(initial_state)
    net.to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    fixed_epochs = cfg.get("lstm_fixed_epochs")
    if fixed_epochs is not None and (type(fixed_epochs) is not int or fixed_epochs < 1):
        raise ValueError("lstm_fixed_epochs must be null or a positive integer")
    max_epochs = cfg["train_max_epochs"] if fixed_epochs is None else fixed_epochs
    best_val, best_state, best_epoch, stale = float("inf"), None, None, 0
    logs = []

    print(f"[LSTM {scenario_name}/{method} seed={seed}] Training started: {len(train_starts)} windows")

    for epoch in range(1, max_epochs + 1):
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
        val_abs, val_sq, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for p_in, p_out in val_loader:
                p_in, p_out = p_in.to(device), p_out.to(device)

                pred = net(p_in)

                error = pred - p_out
                val_abs += float(error.abs().sum())
                val_sq += float(error.square().sum())
                val_n += int(p_out.numel())

        val_mae = val_abs / max(val_n, 1)
        val_rmse = math.sqrt(val_sq / max(val_n, 1))
        avg_train_loss = train_loss / max(len(train_loader), 1)

        logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_mae": val_mae, "val_rmse": val_rmse})

        status = "★BEST" if val_mae < best_val else f"stale={stale}/{cfg['early_stopping_patience']}"
        if epoch % 5 == 0 or val_mae < best_val:  # Print every 5 epochs or on new best
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_mae={val_mae:.6f}, val_rmse={val_rmse:.6f} {status}")

        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            if fixed_epochs is None:
                best_state = {k: v.detach().to(device="cpu", copy=True) for k, v in net.state_dict().items()}
            stale = 0
        elif fixed_epochs is None:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_epoch is None:
        raise RuntimeError(f"{method} training produced no checkpoint")

    if fixed_epochs is None:
        selected_epoch, selected_mae, selected_state = best_epoch, best_val, best_state
        net.load_state_dict(selected_state)
    else:
        # A fixed budget evaluates the final weights, not an earlier best epoch.
        selected_epoch, selected_mae = len(logs), logs[-1]["val_mae"]
        if not math.isfinite(selected_mae):
            raise RuntimeError(f"{method} produced nonfinite validation MAE at the fixed budget")
        selected_state = {k: v.detach().to(device="cpu", copy=True) for k, v in net.state_dict().items()}
    net.eval()
    selection = {
        "best_val_mae": best_val, "best_validation_epoch": best_epoch,
        "selected_epoch": selected_epoch, "selected_val_mae": selected_mae,
        "optimizer_steps": len(logs) * len(train_loader), "shuffle_seed": shuffle_seed,
        "checkpoint_selection": "best_validation_mae" if fixed_epochs is None else "last_fixed_epoch",
    }

    # Save checkpoint
    ckpt_path = work_dir / "checkpoints" / f"lstm_{scenario_name}_{method}_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": selected_state,
        "scenario": scenario_name,
        "method": method,
        "seed": seed,
        **selection,
    }, ckpt_path)

    # Save log
    log_path = work_dir / "logs" / f"train_lstm_{scenario_name}_{method}_seed{seed}.csv"
    write_csv(log_path, logs)

    return net, device, {**selection, "epochs": len(logs), "checkpoint": str(ckpt_path)}

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
    # Match np.digitize(..., right=False): each finite lower bound is inclusive.
    bounds = [float("-inf"), *thresholds, float("inf")]
    lower, upper = bounds[regime_id:regime_id + 2]
    left = "(" if regime_id == 0 else "["
    return f"{left}{lower:g}, {upper:g}) m/s"


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
    data_path = (root / cfg["data_path"]).resolve(strict=True)
    if work_dir == results_dir or work_dir in results_dir.parents or results_dir in work_dir.parents:
        raise ValueError("work_dir and results_dir must be separate directories")

    diffusion_work_dir = work_dir / "diffusion"
    print(f"Single-turbine experiment: {data_path.name} (first half Diffusion, second half LSTM)")
    diffusion_arrays, diffusion_indexes, arrays, indexes, scales, audit = build_dataset(data_path, work_dir, cfg)
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
        "audit": audit,
        "metric_splits": {"imputation": "forecast/train", "forecast": "forecast/test"},
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

                net.requires_grad_(False)
                diffusion_models[(seed, method)] = (net, device)

                log["runs"].append({
                    "phase": "diffusion_training",
                    "method": method,
                    "seed": seed,
                    "status": "completed",
                    **info
                })

                print(f"✓ Completed: val_mae={info['best_val_mae']:.6f}, epochs={info['epochs']}")
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

    lstm_initial_states = {}
    for scenario in scenarios(cfg):
        scenario_name = scenario["name"]

        print(f"\n{'=' * 100}")
        print(f"SCENARIO: {scenario_name}")
        print('=' * 100)

        for seed in cfg["train_seeds"]:
            print(f"\nSeed: {seed}")
            if seed not in lstm_initial_states:
                seed_all(seed)
                lstm_initial_states[seed] = _new_lstm_forecaster(cfg).state_dict()

            # Create one training mask per scenario/seed, shared by all methods.
            mask_seed = int(cfg["mask_seed"]) + int(seed)
            train_mask = create_forecast_train_mask(arrays, scenario, mask_seed)
            mask_dir = work_dir / "masked" / scenario_name
            mask_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_dir / f"train_mask_seed{seed}.npz", mask=train_mask)

            n_missing_train = int((train_mask == 0).sum())
            n_total_train = train_mask.size
            print(f"  Train mask: {n_missing_train}/{n_total_train} missing ({n_missing_train/n_total_train*100:.2f}%)")

            if scenario["type"] == "continuous":
                windows = _gap_windows(train_mask[:, :1], arrays["train"]["segment"], 1,
                                       int(cfg["max_diffusion_window_length"]))
                lengths = [int(end - start) for _, _, _, start, end in windows]
                counts = {}
                for tier in ("short", "medium", "long"):
                    if f"{tier}_lengths" in scenario:
                        low, high = scenario[f"{tier}_lengths"]
                        counts[tier] = sum(low <= gap <= high for gap in lengths)
                log.setdefault("forecast_train_mask_stats", []).append({
                    "scenario": scenario_name, "seed": seed, "missing_rows": sum(lengths),
                    "missing_rate": sum(lengths) / len(train_mask), "gap_counts": counts,
                })
                print(f"  Continuous gap counts: {counts}")

            # Three imputation methods
            for method in IMPUTATION_METHODS:
                try:
                    print(f"\n  Method: {method}")

                    diffusion_net, device = diffusion_models.get((seed, method), (None, None))
                    if method != "linear_interpolation" and diffusion_net is None:
                        print(f"    Diffusion model unavailable, skipping")
                        continue

                    print("    Imputing LSTM training histories only...")
                    train_imputed = impute_row_level_batched(
                        method, diffusion_net, device,
                        arrays["train"]["power"], arrays["train"]["weather"],
                        train_mask, cfg, regimes=arrays["train"]["cluster"],
                        segment_ids=arrays["train"]["segment"], noise_seed=mask_seed
                    )
                    imputation_metrics = evaluate_imputation(
                        arrays["train"]["power"], train_imputed, train_mask, scales,
                        arrays["train"]["cluster"],
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

                    # Train LSTM
                    print(f"    Training LSTM forecaster...")
                    lstm_net, lstm_device, lstm_info = train_lstm_forecaster(
                        train_imputed, arrays, indexes, cfg, work_dir,
                        scenario_name, method, seed, initial_state=lstm_initial_states[seed],
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
