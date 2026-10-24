from __future__ import annotations
import csv, json, math, random, traceback
from pathlib import Path
import numpy as np, torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from data_pipeline import POWER, build_dataset
from diffusion_model import ConditionalDiffusion

REGIMES = ["low_wind", "medium_wind", "high_wind"]
TRAIN_METHODS = ["lstm_imputer", "diffusion_all_weather", "diffusion_low_wind", "diffusion_medium_wind", "diffusion_high_wind"]
EVAL_METHODS = ["linear_interpolation", *TRAIN_METHODS]

def seed_all(seed):
    """Fix Python, NumPy, and PyTorch randomness for reproducible seeds."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def mask_window(L, mask_type, value, rng):
    """Create an independent four-channel random-point or continuous mask."""
    out = np.ones((L, 4), dtype=np.float32)
    m = max(1, int(math.floor(L * value + .5))) if mask_type == "random_point" else int(value)
    for ch in range(4):
        if mask_type == "random_point":
            out[rng.choice(L, m, replace=False), ch] = 0
        else:
            start = int(rng.integers(0, L - m + 1)); out[start:start + m, ch] = 0
    return out

def scenarios(cfg):
    """Return the fixed point and continuous-mask test scenarios."""
    return [("random_point", rate) for rate in cfg["point_missing_rates"]] + [("random_continuous", length) for length in cfg["continuous_lengths"] if length <= cfg["window_length"]]

class Windows(Dataset):
    """Expose continuous arrays as PyTorch windows."""
    def __init__(self, arrays, starts, L): self.a, self.starts, self.L = arrays, starts, L
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = int(self.starts[i]); e = s + self.L
        return torch.from_numpy(self.a["power"][s:e]), torch.from_numpy(self.a["weather"][s:e]), torch.tensor(int(self.a["cluster"][e - 1]))

# Every method is stratified by weather regime for a like-for-like comparison.
# The three regime-specific diffusion models each train on one wind regime only.
GROUPED_METRICS = ("abs_error_max", "abs_error_min", "MAE")
GROUPED_KEYS = ("target_field", "mask_type", "weather_regime", "train_seed",
                "n_windows", "n_missing_points", "missing_rate", "missing_length")


def model(cfg):
    """Build a weather-input diffusion model.

    All four diffusion methods share this single architecture and therefore the
    same parameter count.  The only difference between them is which training
    windows each one sees: the three regime-specific models each train on one
    wind-speed bin, while ``diffusion_all_weather`` trains on a stratified
    mixture of all three.
    """
    model_dim = cfg.get("diffusion_model_dim", cfg.get("diffusion_weather_model_dim", cfg["model_dim"]))
    layers = cfg.get("diffusion_layers", cfg.get("diffusion_weather_layers", cfg["transformer_layers"]))
    heads = cfg.get("diffusion_heads", cfg.get("diffusion_weather_heads", cfg["attention_heads"]))
    return ConditionalDiffusion(model_dim, layers, heads, cfg["diffusion_train_steps"], True, cfg["window_length"])


def train_regime_specific_diffusion(arrays, indexes, cfg, work, seed, regime_name, cluster_id, train_starts):
    """Train one weather-input diffusion model on a specific wind regime.

    Args:
        regime_name: "low_wind", "medium_wind", or "high_wind"
        cluster_id: 0, 1, or 2 corresponding to the regime
        train_starts: aligned window starts for this regime, already reduced to
            the common per-model budget so every method trains on the same count
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = cfg["window_length"]

    # Validation keeps the full regime so early stopping sees a stable signal;
    # only the training budget is aligned across models.
    val_starts = np.asarray([s for s in indexes["val"] if arrays["val"]["cluster"][int(s) + L - 1] == cluster_id], dtype=np.int32)

    train = DataLoader(Windows(arrays["train"], train_starts, L), cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val = DataLoader(Windows(arrays["val"], val_starts, L), cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    method_name = f"diffusion_{regime_name}"
    net = model(cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best, state, stale, logs = float("inf"), None, 0, []
    rng = np.random.default_rng(seed + 9001 + cluster_id * 100)
    print(f"  [Diffusion-{regime_name}] Training on {len(train_starts)} windows, {len(train)} batches/epoch")

    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"Diffusion-{regime_name} seed={seed}", unit="epoch", ncols=100):
        net.train(); total = 0.0
        for p, w, _ in train:
            p, w = p.to(device), w.to(device)
            mask = torch.from_numpy(_random_masks(len(p), cfg, rng)).to(device)
            loss = net.loss_per_sample(p, p * mask, mask, w).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            total += float(loss.detach())
        val_score = validate(net, val, cfg, device, True, seed + epoch, method_name)
        logs.append({"epoch": epoch, "train_loss": total / max(len(train), 1), "val_rmse": val_score})
        avg_loss = total / max(len(train), 1)
        status = "★BEST" if val_score < best else f"stale={stale}/{cfg['early_stopping_patience']}"
        tqdm.write(f"  Epoch {epoch:3d}/{cfg['train_max_epochs']}: train_loss={avg_loss:.6f}, val_rmse={val_score:.6f} {status}")
        if val_score < best:
            best, state, stale = val_score, {k: v.detach().cpu() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                tqdm.write(f"  Early stopping triggered at epoch {epoch}")
                break

    if state is None:
        raise RuntimeError(f"{method_name} training produced no validation checkpoint")
    net.load_state_dict(state); net.eval()
    path = work / "checkpoints" / f"{method_name}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "method": method_name, "seed": seed,
                "uses_weather": True, "classified": True, "regime": regime_name,
                "cluster_id": cluster_id, "n_train_windows": len(train_starts)}, path)
    write_csv(work / "logs" / f"train_{method_name}_seed{seed}.csv", logs)
    return net, device, {"best_val_rmse": best, "epochs_run": len(logs), "checkpoint": str(path),
                         "uses_weather": True, "classified": True, "regime": regime_name,
                         "n_train_windows": len(train_starts)}


def regime_of_window(arrays, split, start, L):
    """Return the wind regime of a window, taken from its last timestamp."""
    return int(arrays[split]["cluster"][int(start) + L - 1])


def windows_by_regime(arrays, indexes, split, cfg):
    """Bucket every window start of one split by its wind regime."""
    L = cfg["window_length"]
    out = {c: [] for c in range(len(REGIMES))}
    for s in indexes[split]:
        out[regime_of_window(arrays, split, s, L)].append(int(s))
    return {c: np.asarray(v, dtype=np.int32) for c, v in out.items()}


def get_min_cluster_count(arrays, indexes, cfg):
    """Report the per-regime training window counts and the alignment target.

    The target is the largest multiple of the regime count that no single
    regime exceeds, so the stratified draw for the global models divides evenly.
    """
    buckets = windows_by_regime(arrays, indexes, "train", cfg)
    counts = [len(buckets[c]) for c in range(len(REGIMES))]
    # Round down to a multiple of the regime count so the stratified draw for
    # the global models divides evenly across the three regimes.
    per_regime = min(counts) - (min(counts) % len(REGIMES))
    print(f"Training windows per regime: {dict(zip(REGIMES, counts))}")
    print(f"Alignment target: {per_regime} windows for every model "
          f"({per_regime // len(REGIMES)} per regime for the stratified global models)")
    return per_regime, buckets


def sample_regime_windows(buckets, regime_id, target, rng):
    """Draw ``target`` windows from one regime without replacement."""
    pool = buckets[regime_id]
    if len(pool) < target:
        raise ValueError(f"regime {REGIMES[regime_id]} has only {len(pool)} windows, "
                         f"fewer than the alignment target {target}")
    return np.sort(rng.choice(pool, target, replace=False)).astype(np.int32)


def balanced_starts_for(method, target, buckets, cfg):
    """Return the training window starts one method is allowed to use.

    Regime-specific models train on ``target`` windows drawn from their own
    regime.  Global models train on the same ``target`` count, stratified so
    each regime contributes an equal share and the wind-speed distribution is
    preserved rather than left to sampling luck.
    """
    rng = np.random.default_rng(cfg.get("balance_seed", 42))
    if method.startswith("diffusion_") and method != "diffusion_all_weather":
        regime = REGIMES.index(method[len("diffusion_"):])
        return sample_regime_windows(buckets, regime, target, rng)
    per = target // len(REGIMES)
    picked = [sample_regime_windows(buckets, c, per, rng) for c in range(len(REGIMES))]
    return np.sort(np.concatenate(picked)).astype(np.int32)


class LSTMImputer(nn.Module):
    """Unified recurrent baseline for four-channel power reconstruction.

    One model is trained on all training windows.  Weather is provided as a
    normal sequence feature when ``use_weather`` is enabled; the data are
    never split into separate low-, medium- or high-wind training jobs.
    """
    def __init__(self, hidden=128, layers=2, dropout=0.1, use_weather=True, bidirectional=True):
        super().__init__()
        self.use_weather = use_weather
        self.bidirectional = bidirectional
        input_dim = 8 + (13 if use_weather else 0)
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(input_dim, hidden, num_layers=layers,
                               dropout=recurrent_dropout, batch_first=True,
                               bidirectional=bidirectional)
        output_dim = hidden * (2 if bidirectional else 1)
        self.output = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim), nn.GELU(), nn.Linear(output_dim, 4)
        )

    def forward(self, observed, mask, weather=None):
        fields = [observed, mask]
        if self.use_weather:
            if weather is None:
                raise ValueError("weather-enabled LSTM requires weather")
            fields.append(weather)
        hidden, _ = self.encoder(torch.cat(fields, dim=-1))
        generated = self.output(hidden)
        # Observations are copied through exactly; only masked values are generated.
        return torch.where(mask.bool(), observed, generated)

    def loss_per_sample(self, clean, observed, mask, weather=None):
        prediction = self(observed, mask, weather)
        missing = 1 - mask
        return ((prediction - clean).square() * missing).sum((1, 2)) / missing.sum((1, 2)).clamp_min(1)


def train_lstm(arrays, indexes, cfg, work, seed, balanced_starts=None):
    """Train one LSTM on the complete training split without weather grouping."""
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_starts = balanced_starts if balanced_starts is not None else indexes["train"]
    train = DataLoader(Windows(arrays["train"], train_starts, cfg["window_length"]),
                       cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val = DataLoader(Windows(arrays["val"], indexes["val"], cfg["window_length"]),
                     cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    use_weather = bool(cfg.get("lstm_use_weather", True))
    bidirectional = bool(cfg.get("lstm_bidirectional", True))
    net = LSTMImputer(cfg.get("lstm_hidden_dim", cfg["model_dim"]),
                      cfg.get("lstm_layers", 2), cfg.get("lstm_dropout", 0.1),
                      use_weather=use_weather, bidirectional=bidirectional).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best, state, stale, logs = float("inf"), None, 0, []
    rng = np.random.default_rng(seed + 7000)
    print(f"  [LSTM] Training on {len(train_starts)} windows, {len(train)} batches/epoch, max_epochs={cfg['train_max_epochs']}")
    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"LSTM seed={seed}", unit="epoch", ncols=100):
        net.train(); total = 0.0
        for p, w, _ in train:
            p, w = p.to(device), w.to(device)
            pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(len(p))]
            mask = torch.from_numpy(np.stack([mask_window(cfg["window_length"], a, b, rng) for a, b in pairs])).to(device)
            loss = net.loss_per_sample(p, p * mask, mask, w if use_weather else None).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            total += float(loss.detach())
        val_score = validate(net, val, cfg, device, use_weather, seed + epoch, "lstm_imputer")
        logs.append({"epoch": epoch, "train_loss": total / max(len(train), 1), "val_rmse": val_score})
        avg_loss = total / max(len(train), 1)
        status = "★BEST" if val_score < best else f"stale={stale}/{cfg['early_stopping_patience']}"
        tqdm.write(f"  Epoch {epoch:3d}/{cfg['train_max_epochs']}: train_loss={avg_loss:.6f}, val_rmse={val_score:.6f} {status}")
        if val_score < best:
            best, state, stale = val_score, {k: v.detach().cpu() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                tqdm.write(f"  Early stopping triggered at epoch {epoch}")
                break
    if state is None:
        raise RuntimeError("LSTM training produced no validation checkpoint")
    net.load_state_dict(state); net.eval()
    path = work / "checkpoints" / f"lstm_imputer_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "method": "lstm_imputer", "seed": seed,
                "use_weather": use_weather, "bidirectional": bidirectional,
                "n_train_windows": len(train_starts)}, path)
    write_csv(work / "logs" / f"train_lstm_imputer_seed{seed}.csv", logs)
    return net, device, {"best_val_rmse": best, "epochs_run": len(logs), "checkpoint": str(path),
                         "n_train_windows": len(train_starts)}

def _random_masks(batch_size, cfg, rng):
    """Draw one configured point/block scenario independently per sample."""
    pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(batch_size)]
    return np.stack([mask_window(cfg["window_length"], kind, value, rng) for kind, value in pairs])


def train_unclassified_diffusion(arrays, indexes, cfg, work, seed, balanced_starts=None):
    """Train one weather-input diffusion model on all wind regimes.

    ``diffusion_all_weather`` is the unclassified baseline: it receives all
    meteorological variables but trains one model across all wind regimes.
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_starts = balanced_starts if balanced_starts is not None else indexes["train"]
    train = DataLoader(Windows(arrays["train"], train_starts, cfg["window_length"]), cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val = DataLoader(Windows(arrays["val"], indexes["val"], cfg["window_length"]), cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    net = model(cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best, state, stale, logs = float("inf"), None, 0, []
    rng = np.random.default_rng(seed + 9001)
    print(f"  [Diffusion-ALL] Training on {len(train_starts)} windows (stratified), {len(train)} batches/epoch")
    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"Diffusion-ALL seed={seed}", unit="epoch", ncols=100):
        net.train(); total = 0.0
        for p, w, _ in train:
            p, w = p.to(device), w.to(device)
            mask = torch.from_numpy(_random_masks(len(p), cfg, rng)).to(device)
            loss = net.loss_per_sample(p, p * mask, mask, w).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            total += float(loss.detach())
        val_score = validate(net, val, cfg, device, True, seed + epoch, "diffusion_all_weather")
        logs.append({"epoch": epoch, "train_loss": total / max(len(train), 1), "val_rmse": val_score})
        avg_loss = total / max(len(train), 1)
        status = "★BEST" if val_score < best else f"stale={stale}/{cfg['early_stopping_patience']}"
        tqdm.write(f"  Epoch {epoch:3d}/{cfg['train_max_epochs']}: train_loss={avg_loss:.6f}, val_rmse={val_score:.6f} {status}")
        if val_score < best:
            best, state, stale = val_score, {k: v.detach().cpu() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                tqdm.write(f"  Early stopping triggered at epoch {epoch}")
                break
    if state is None:
        raise RuntimeError("unclassified diffusion training produced no validation checkpoint")
    net.load_state_dict(state); net.eval()
    path = work / "checkpoints" / f"diffusion_all_weather_seed{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "method": "diffusion_all_weather", "seed": seed,
                "uses_weather": True, "classified": False, "n_train_windows": len(train_starts)}, path)
    write_csv(work / "logs" / f"train_diffusion_all_weather_seed{seed}.csv", logs)
    return net, device, {"best_val_rmse": best, "epochs_run": len(logs), "checkpoint": str(path),
                         "uses_weather": True, "classified": False, "n_train_windows": len(train_starts)}


def train_one(arrays, indexes, cfg, work, method, seed, train_starts):
    """Train one method on the supplied aligned window budget.

    Every method receives exactly ``len(train_starts)`` training windows, so
    the five models are compared at equal sample size as well as equal
    parameter count.
    """
    if method == "lstm_imputer":
        return train_lstm(arrays, indexes, cfg, work, seed, train_starts)
    if method == "diffusion_all_weather":
        return train_unclassified_diffusion(arrays, indexes, cfg, work, seed, train_starts)
    if method.startswith("diffusion_"):
        regime_name = method[len("diffusion_"):]
        if regime_name in REGIMES:
            return train_regime_specific_diffusion(arrays, indexes, cfg, work, seed,
                                                   regime_name, REGIMES.index(regime_name), train_starts)
    raise ValueError(f"Unknown training method: {method}")

@torch.no_grad()
def validate(net, loader, cfg, device, weather, seed, method):
    """Evaluate normalized missing-position RMSE for early stopping."""
    rng = np.random.default_rng(seed); sq = n = 0
    for p, w, _ in tqdm(loader, desc="validation", leave=False, unit="batch"):
        p, w = p.to(device), w.to(device)
        pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(len(p))]
        masks = torch.from_numpy(np.stack([mask_window(cfg["window_length"], a, b, rng) for a, b in pairs])).to(device)
        if method == "lstm_imputer":
            pred = net(p * masks, masks, w if weather else None)
        else:
            pred = net.sample(p * masks, masks, w, cfg["validation_ddim_steps"], 1)
        sq += float((((pred - p) ** 2) * (1 - masks)).sum()); n += int((1 - masks).sum())
    return math.sqrt(sq / max(n, 1))

def inverse(x, scale): return np.asarray(x) * (scale["max"] - scale["min"] or 1) + scale["min"]
def metrics(y, p):
    """Return absolute-error statistics on the original power scale."""
    y, p = np.asarray(y), np.asarray(p); abs_error = np.abs(p - y)
    if not len(y):
        return {"abs_error_max": "", "abs_error_min": "", "MAE": "", "n_points": 0}
    mean_error = float(np.mean(abs_error))
    return {"abs_error_max": float(np.max(abs_error)), "abs_error_min": float(np.min(abs_error)), "MAE": mean_error, "n_points": int(len(y))}

def linear_fill(observed, mask):
    """Fill each channel by linear interpolation, using an edge value at boundaries."""
    out = np.asarray(observed, dtype=np.float32).copy(); L, C = out.shape
    for ch in range(C):
        known = np.flatnonzero(mask[:, ch] > 0)
        if not len(known):
            continue
        missing = np.flatnonzero(mask[:, ch] <= 0)
        out[missing, ch] = np.interp(missing, known, out[known, ch])
    return out

@torch.no_grad()
def reconstruct_batch(method, net, device, power, weather, masks, cfg):
    """Apply one reconstruction method while preserving every observed value."""
    if method == "linear_interpolation":
        return np.stack([linear_fill(power[i] * masks[i], masks[i]) for i in range(len(power))])
    pt = torch.as_tensor(power * masks, device=device, dtype=torch.float32)
    mt = torch.as_tensor(masks, device=device, dtype=torch.float32)
    wt = torch.as_tensor(weather, device=device, dtype=torch.float32)
    if method == "lstm_imputer":
        pred = net(pt, mt, wt if net.use_weather else None)
    else:
        pred = net.sample(pt, mt, wt, cfg["ddim_steps"], cfg["ddim_samples"])
    return pred.detach().cpu().numpy()


def evaluate(net, device, arrays, indexes, cfg, scales, method, seed, regime_filter=None):
    """Evaluate one method for every mask scenario and wind regime.

    Masks are drawn for the whole test split before any regime filtering, so a
    given window always carries the same missing positions no matter which
    method is being scored.  Regimes are assigned per window rather than per
    missing point, which keeps the window set and the point count identical
    across methods for every reported row.

    Args:
        regime_filter: when set, score only the windows of this regime.  A
            regime-specific diffusion model is only meaningful on the wind
            speed range it was trained on.
    """
    starts = indexes["test"]; L = cfg["window_length"]; grouped = []
    window_regimes = np.asarray([regime_of_window(arrays, "test", s, L) for s in starts], dtype=np.int64)
    batch_size = min(cfg["batch_size"], 256)
    for mask_type, value in tqdm(scenarios(cfg), desc=f"evaluate {method} seed={seed}", unit="scenario"):
        mask_rng = np.random.default_rng(cfg["mask_seed"] + int(seed))
        masks_all = np.stack([mask_window(L, mask_type, value, mask_rng) for _ in range(len(starts))])
        regime = {(field, c): {"y": [], "p": [], "windows": set()} for field in POWER for c in range(len(REGIMES))}
        for offset in tqdm(range(0, len(starts), batch_size), desc=f"mask {mask_type}:{value}", leave=False, unit="batch"):
            sl = slice(offset, offset + batch_size)
            ss, masks, wreg = starts[sl], masks_all[sl], window_regimes[sl]
            if regime_filter is not None:
                keep = wreg == regime_filter
                if not keep.any(): continue
                ss, masks, wreg = ss[keep], masks[keep], wreg[keep]
            p = np.stack([arrays["test"]["power"][s:s + L] for s in ss])
            w = np.stack([arrays["test"]["weather"][s:s + L] for s in ss])
            pred = reconstruct_batch(method, net, device, p, w, masks, cfg)
            pred = np.where(masks > 0, p, pred)
            for j, s in enumerate(ss):
                c = int(wreg[j])
                for ch, field in enumerate(POWER):
                    miss = masks[j, :, ch] == 0
                    if not miss.any(): continue
                    y = inverse(p[j, miss, ch], scales["power"][field]); q = inverse(pred[j, miss, ch], scales["power"][field])
                    bucket = regime[(field, c)]
                    bucket["y"].extend(y.tolist()); bucket["p"].extend(q.tolist()); bucket["windows"].add(int(s))
        for field in POWER:
            for c in range(len(REGIMES)):
                if regime_filter is not None and c != regime_filter: continue
                b = regime[(field, c)]
                gm = metrics(b["y"], b["p"])
                grow = {"method_id": method, "target_field": field, "mask_type": mask_type, "weather_regime": REGIMES[c], "train_seed": seed, "n_windows": len(b["windows"]), "n_missing_points": gm.pop("n_points"), **gm}
                grow["missing_rate" if mask_type == "random_point" else "missing_length"] = value
                grouped.append(grow)
    return grouped
def write_csv(path, rows):
    """Write training logs or the grouped result table as UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f: w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)

def pivot_grouped_rows(rows):
    """Place all methods side by side for every matched test condition.

    Regime-specific models only report results for their own regime, so those
    columns will be empty for other regimes.
    """
    combined = {}
    for row in rows:
        method = row["method_id"]
        if method not in EVAL_METHODS:
            raise ValueError(f"Unexpected method in grouped results: {method}")
        key = tuple("" if row.get(field) is None else str(row.get(field, "")) for field in GROUPED_KEYS)
        entry = combined.setdefault(key, dict(zip(GROUPED_KEYS, key)))
        for metric in GROUPED_METRICS:
            column = f"{method}_{metric}"
            if column in entry:
                raise ValueError(f"Duplicate grouped metric: {column} for {key}")
            entry[column] = row[metric]

    # Build expected columns, but allow regime-specific models to be missing in non-matching regimes
    output = []
    for key, entry in combined.items():
        # Extract the weather_regime from the key (index 2 in GROUPED_KEYS)
        weather_regime = key[2]  # "low_wind", "medium_wind", or "high_wind"

        # Fill in missing columns with empty strings for regime-specific models in non-matching regimes
        for method in EVAL_METHODS:
            # Check if this is a regime-specific method and if it matches the current regime
            is_regime_specific = method.startswith("diffusion_") and method != "diffusion_all_weather"
            if is_regime_specific:
                method_regime = method.replace("diffusion_", "")
                if method_regime != weather_regime:
                    # This regime-specific model doesn't evaluate on this regime, fill with empty
                    for metric in GROUPED_METRICS:
                        column = f"{method}_{metric}"
                        if column not in entry:
                            entry[column] = ""

        # Check that all general methods (lstm, diffusion_all_weather, linear_interpolation) have results
        general_methods = ["lstm_imputer", "diffusion_all_weather", "linear_interpolation"]
        expected_general = {f"{method}_{metric}" for method in general_methods for metric in GROUPED_METRICS}
        missing_general = expected_general.difference(entry)
        if missing_general:
            raise ValueError(f"Grouped comparison is missing general methods/metrics for {key}: {sorted(missing_general)}")

        output.append(entry)
    return sorted(output, key=lambda row: tuple(str(row.get(field, "")) for field in GROUPED_KEYS[:4]))

def run(config_path: Path, quick=False):
    """Train and evaluate all methods, exporting only the grouped comparison."""
    root = config_path.parent; cfg = json.loads(config_path.read_text(encoding="utf-8"));
    if quick: cfg.update({"train_max_epochs": 1, "early_stopping_patience": 1, "batch_size": 16, "ddim_steps": 3, "ddim_samples": 1})
    work, results = root / "work_v2", (root / "results_v2" / "quick" if quick else root / "results"); data = (root / cfg["data_path"]).resolve(); arrays, indexes, scales, audit = build_dataset(data, work, cfg); log = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "audit": audit, "runs": []}; grouped = []

    # Align every model to one common training budget before any training starts.
    print("\n" + "=" * 80)
    print("TRAINING DATA ALIGNMENT")
    print("=" * 80)
    target, buckets = get_min_cluster_count(arrays, indexes, cfg)

    total_runs = (len(TRAIN_METHODS) + 1) * len(cfg["train_seeds"])
    run_number = 0

    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENTS: {len(TRAIN_METHODS)} methods × {len(cfg['train_seeds'])} seeds + linear baseline")
    print(f"Total runs: {total_runs}")
    print("=" * 80)

    for method in TRAIN_METHODS:
        train_starts = balanced_starts_for(method, target, buckets, cfg)
        regime_filter = None
        if method.startswith("diffusion_") and method != "diffusion_all_weather":
            regime_filter = REGIMES.index(method[len("diffusion_"):])
        for seed in cfg["train_seeds"]:
            run_number += 1
            print(f"\n{'=' * 80}")
            print(f"[{run_number}/{total_runs}] METHOD: {method.upper()} | SEED: {seed}")
            print(f"Training windows: {len(train_starts)} | Device: {log['device']}")
            print('=' * 80)
            try:
                net, device, info = train_one(arrays, indexes, cfg, work, method, seed, train_starts)
                print(f"\n  ✓ Training completed: best_val_rmse={info['best_val_rmse']:.6f}, epochs={info['epochs_run']}")
                print(f"  → Starting evaluation on test set...")
                grouped.extend(evaluate(net, device, arrays, indexes, cfg, scales, method, seed, regime_filter))
                print(f"  ✓ Evaluation completed")
                del net
                log["runs"].append({"method": method, "seed": seed, "status": "completed", **info})
            except Exception as e:
                print(f"\n  ✗ FAILED: {repr(e)}")
                log["runs"].append({"method": method, "seed": seed, "status": "failed", "error": repr(e), "traceback": traceback.format_exc()})

    # Reuse each neural-model seed so all methods see identical test masks.
    print("\n" + "=" * 80)
    print("EVALUATING LINEAR INTERPOLATION BASELINE")
    print("=" * 80)
    for seed in cfg["train_seeds"]:
        run_number += 1
        print(f"\n[{run_number}/{total_runs}] Linear interpolation | mask seed={seed}")
        try:
            method_grouped = evaluate(None, torch.device("cpu"), arrays, indexes, cfg, scales, "linear_interpolation", seed)
            grouped.extend(method_grouped)
            log["runs"].append({"method": "linear_interpolation", "seed": seed, "status": "completed", "mask_seed": cfg["mask_seed"] + int(seed)})
            print(f"  ✓ Completed")
        except Exception as e:
            print(f"  ✗ FAILED: {repr(e)}")
            log["runs"].append({"method": "linear_interpolation", "seed": seed, "status": "failed", "error": repr(e), "traceback": traceback.format_exc()})

    results.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("GENERATING RESULTS")
    print("=" * 80)
    print(f"Pivoting {len(grouped)} evaluation rows into comparison table...")
    grouped_wide = pivot_grouped_rows(grouped)
    expected_rows = len(POWER) * len(scenarios(cfg)) * len(REGIMES) * len(cfg["train_seeds"])
    if len(grouped_wide) != expected_rows:
        raise ValueError(f"Expected {expected_rows} complete grouped rows, found {len(grouped_wide)}")
    write_csv(results / "reconstruction_grouped_comparison.csv", grouped_wide)
    print(f"  ✓ Written {len(grouped_wide)} rows to: {results / 'reconstruction_grouped_comparison.csv'}")
    # Remove the previous table set so results/ contains only the requested comparison.
    for name in ("reconstruction_comparison.csv", "forecast_support_comparison.csv", "ablation_comparison.csv"):
        stale_result = results / name
        if stale_result.exists():
            stale_result.unlink()
    (work / "logs").mkdir(parents=True, exist_ok=True)
    log["grouped_comparison"] = {"status": "completed", "rows": len(grouped_wide), "methods": EVAL_METHODS}
    (work / "logs" / "run_summary.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    completed = sum(1 for r in log["runs"] if r.get("status") == "completed")
    failed = sum(1 for r in log["runs"] if r.get("status") == "failed")
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total runs: {len(log['runs'])}")
    print(f"  ✓ Completed: {completed}")
    print(f"  ✗ Failed: {failed}")
    print(f"Results exported: {results / 'reconstruction_grouped_comparison.csv'}")
    print("=" * 80 + "\n")
    return log



