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
TRAIN_METHODS = ["lstm_imputer", "diffusion_all_weather", "proposed_weather_diffusion"]
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
# Only the proposed diffusion model trains separate networks by regime.
GROUPED_METRICS = ("abs_error_max", "abs_error_min", "MAE")
GROUPED_KEYS = ("target_field", "mask_type", "weather_regime", "train_seed",
                "n_windows", "n_missing_points", "missing_rate", "missing_length")


def model(cfg):
    """Build a weather-input diffusion model.

    Both diffusion variants receive the same 13 meteorological variables.
    Their only experimental difference is whether the proposed variant is
    trained as three separate wind-regime models.
    """
    model_dim = cfg.get("diffusion_model_dim", cfg.get("diffusion_weather_model_dim", cfg["model_dim"]))
    layers = cfg.get("diffusion_layers", cfg.get("diffusion_weather_layers", cfg["transformer_layers"]))
    heads = cfg.get("diffusion_heads", cfg.get("diffusion_weather_heads", cfg["attention_heads"]))
    return ConditionalDiffusion(model_dim, layers, heads, cfg["diffusion_train_steps"], True, cfg["window_length"])


class ClusteredDiffusion(nn.Module):
    """Three independent weather-conditioned diffusion models.

    The cluster id is assigned from the window-end weather regime.  Keeping
    the models in a ModuleList makes checkpointing and device transfer work
    exactly like a normal PyTorch module.
    """
    def __init__(self, cfg):
        super().__init__()
        self.models = nn.ModuleList([model(cfg) for _ in REGIMES])
        self.use_weather = True
        self.clustered = True

    @torch.no_grad()
    def sample(self, observed, mask, weather, clusters, steps=50, draws=1):
        if clusters is None:
            raise ValueError("clustered diffusion requires one weather cluster per window")
        output = torch.empty_like(observed)
        for cluster_id, net in enumerate(self.models):
            selected = torch.where(clusters == cluster_id)[0]
            if selected.numel():
                output[selected] = net.sample(observed[selected], mask[selected], weather[selected], steps, draws)
        return output


def sample_diffusion(net, observed, mask, weather, clusters, steps, draws):
    """Sample either the universal baseline or the three clustered models."""
    if isinstance(net, ClusteredDiffusion):
        return net.sample(observed, mask, weather, clusters, steps, draws)
    return net.sample(observed, mask, weather, steps, draws)


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


def train_lstm(arrays, indexes, cfg, work, seed):
    """Train one LSTM on the complete training split without weather grouping."""
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(Windows(arrays["train"], indexes["train"], cfg["window_length"]),
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
    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"train lstm_imputer seed={seed}", unit="epoch"):
        net.train(); total = 0.0
        for p, w, _ in tqdm(train, desc=f"lstm epoch {epoch}", leave=False, unit="batch"):
            p, w = p.to(device), w.to(device)
            pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(len(p))]
            mask = torch.from_numpy(np.stack([mask_window(cfg["window_length"], a, b, rng) for a, b in pairs])).to(device)
            loss = net.loss_per_sample(p, p * mask, mask, w if use_weather else None).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            total += float(loss.detach())
        val_score = validate(net, val, cfg, device, use_weather, seed + epoch, "lstm_imputer")
        logs.append({"epoch": epoch, "train_loss": total / max(len(train), 1), "val_rmse": val_score})
        tqdm.write(f"lstm_imputer seed={seed} epoch={epoch}: val_rmse={val_score:.6f}")
        if val_score < best:
            best, state, stale = val_score, {k: v.detach().cpu() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                break
    if state is None:
        raise RuntimeError("LSTM training produced no validation checkpoint")
    net.load_state_dict(state); net.eval()
    path = work / "checkpoints" / f"lstm_imputer_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "method": "lstm_imputer", "seed": seed,
                "use_weather": use_weather, "bidirectional": bidirectional}, path)
    write_csv(work / "logs" / f"train_lstm_imputer_seed{seed}.csv", logs)
    return net, device, {"best_val_rmse": best, "epochs_run": len(logs), "checkpoint": str(path)}

def _random_masks(batch_size, cfg, rng):
    """Draw one configured point/block scenario independently per sample."""
    pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(batch_size)]
    return np.stack([mask_window(cfg["window_length"], kind, value, rng) for kind, value in pairs])


def train_unclassified_diffusion(arrays, indexes, cfg, work, seed):
    """Train one weather-input diffusion model on all wind regimes.

    ``diffusion_all_weather`` is the unclassified baseline: it receives all
    meteorological variables but trains one model across all wind regimes.
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(Windows(arrays["train"], indexes["train"], cfg["window_length"]), cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val = DataLoader(Windows(arrays["val"], indexes["val"], cfg["window_length"]), cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    net = model(cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best, state, stale, logs = float("inf"), None, 0, []
    rng = np.random.default_rng(seed + 9001)
    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"train diffusion_all_weather seed={seed}", unit="epoch"):
        net.train(); total = 0.0
        for p, w, _ in tqdm(train, desc=f"epoch {epoch} batches", leave=False, unit="batch"):
            p, w = p.to(device), w.to(device)
            mask = torch.from_numpy(_random_masks(len(p), cfg, rng)).to(device)
            loss = net.loss_per_sample(p, p * mask, mask, w).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            total += float(loss.detach())
        val_score = validate(net, val, cfg, device, True, seed + epoch, "diffusion_all_weather")
        logs.append({"epoch": epoch, "train_loss": total / max(len(train), 1), "val_rmse": val_score})
        tqdm.write(f"diffusion_all_weather seed={seed} epoch={epoch}: val_rmse={val_score:.6f}")
        if val_score < best:
            best, state, stale = val_score, {k: v.detach().cpu() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]: break
    if state is None:
        raise RuntimeError("unclassified diffusion training produced no validation checkpoint")
    net.load_state_dict(state); net.eval()
    path = work / "checkpoints" / f"diffusion_all_weather_seed{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "method": "diffusion_all_weather", "seed": seed, "uses_weather": True, "classified": False}, path)
    write_csv(work / "logs" / f"train_diffusion_all_weather_seed{seed}.csv", logs)
    return net, device, {"best_val_rmse": best, "epochs_run": len(logs), "checkpoint": str(path), "uses_weather": True, "classified": False}


def train_clustered_diffusion(arrays, indexes, cfg, work, seed):
    """Train one independent weather-input diffusion model per wind regime."""
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = cfg["window_length"]
    train_loaders, cluster_counts = {}, {}
    for cluster_id in range(len(REGIMES)):
        starts = np.asarray([s for s in indexes["train"] if arrays["train"]["cluster"][int(s) + L - 1] == cluster_id], dtype=np.int32)
        cluster_counts[REGIMES[cluster_id]] = int(len(starts))
        train_loaders[cluster_id] = DataLoader(Windows(arrays["train"], starts, L), cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val = DataLoader(Windows(arrays["val"], indexes["val"], L), cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    net = ClusteredDiffusion(cfg).to(device)
    optimizers = [torch.optim.AdamW(child.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]) for child in net.models]
    best, state, stale, logs = float("inf"), None, 0, []
    rng = np.random.default_rng(seed + 9001)
    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"train proposed_weather_diffusion seed={seed}", unit="epoch"):
        net.train(); total = 0.0; batches_seen = 0
        for cluster_id, loader in train_loaders.items():
            child = net.models[cluster_id]
            for p, w, _ in tqdm(loader, desc=f"regime {REGIMES[cluster_id]} epoch {epoch}", leave=False, unit="batch"):
                p, w = p.to(device), w.to(device)
                mask = torch.from_numpy(_random_masks(len(p), cfg, rng)).to(device)
                loss = child.loss_per_sample(p, p * mask, mask, w).mean()
                optimizers[cluster_id].zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(child.parameters(), 1.0); optimizers[cluster_id].step()
                total += float(loss.detach()); batches_seen += 1
        val_score = validate(net, val, cfg, device, True, seed + epoch, "proposed_weather_diffusion")
        logs.append({"epoch": epoch, "train_loss": total / max(batches_seen, 1), "val_rmse": val_score})
        tqdm.write(f"proposed_weather_diffusion seed={seed} epoch={epoch}: val_rmse={val_score:.6f}")
        if val_score < best:
            best, state, stale = val_score, {k: v.detach().cpu() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]: break
    if state is None:
        raise RuntimeError("clustered diffusion training produced no validation checkpoint")
    net.load_state_dict(state); net.eval()
    path = work / "checkpoints" / f"proposed_weather_diffusion_seed{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "method": "proposed_weather_diffusion", "seed": seed, "uses_weather": True, "classified": True, "cluster_counts": cluster_counts}, path)
    write_csv(work / "logs" / f"train_proposed_weather_diffusion_seed{seed}.csv", logs)
    return net, device, {"best_val_rmse": best, "epochs_run": len(logs), "checkpoint": str(path), "uses_weather": True, "classified": True, "cluster_counts": cluster_counts}


def train_one(arrays, indexes, cfg, work, method, seed):
    """Train one method and save its best validation checkpoint."""
    if method == "lstm_imputer":
        return train_lstm(arrays, indexes, cfg, work, seed)
    if method == "proposed_weather_diffusion":
        return train_clustered_diffusion(arrays, indexes, cfg, work, seed)
    return train_unclassified_diffusion(arrays, indexes, cfg, work, seed)

@torch.no_grad()
def validate(net, loader, cfg, device, weather, seed, method):
    """Evaluate normalized missing-position RMSE for early stopping."""
    rng = np.random.default_rng(seed); sq = n = 0
    for p, w, clusters in tqdm(loader, desc="validation", leave=False, unit="batch"):
        p, w, clusters = p.to(device), w.to(device), clusters.to(device)
        pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(len(p))]
        masks = torch.from_numpy(np.stack([mask_window(cfg["window_length"], a, b, rng) for a, b in pairs])).to(device)
        if method == "lstm_imputer":
            pred = net(p * masks, masks, w if weather else None)
        else:
            pred = sample_diffusion(net, p * masks, masks, w, clusters, cfg["validation_ddim_steps"], 1)
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
def reconstruct_batch(method, net, device, power, weather, masks, cfg, clusters=None):
    """Apply one reconstruction method while preserving every observed value."""
    if method == "linear_interpolation":
        return np.stack([linear_fill(power[i] * masks[i], masks[i]) for i in range(len(power))])
    pt = torch.as_tensor(power * masks, device=device, dtype=torch.float32)
    mt = torch.as_tensor(masks, device=device, dtype=torch.float32)
    wt = torch.as_tensor(weather, device=device, dtype=torch.float32)
    if method == "lstm_imputer":
        pred = net(pt, mt, wt if net.use_weather else None)
    else:
        ct = None if clusters is None else torch.as_tensor(clusters, device=device, dtype=torch.long)
        pred = sample_diffusion(net, pt, mt, wt, ct, cfg["ddim_steps"], cfg["ddim_samples"])
    return pred.detach().cpu().numpy()

def evaluate(net, device, arrays, indexes, cfg, scales, method, seed):
    """Evaluate one method for every mask scenario and weather regime."""
    rng = np.random.default_rng(cfg["mask_seed"] + int(seed)); starts = indexes["test"]; L = cfg["window_length"]; grouped = []
    for mask_type, value in tqdm(scenarios(cfg), desc=f"evaluate {method} seed={seed}", unit="scenario"):
        regime = {(field, c): {"y": [], "p": [], "windows": set()} for field in POWER for c in range(3)}
        batch_size = min(cfg["batch_size"], 256)
        for offset in tqdm(range(0, len(starts), batch_size), desc=f"mask {mask_type}:{value}", leave=False, unit="batch"):
            ss = starts[offset:offset + batch_size]
            p = np.stack([arrays["test"]["power"][s:s + L] for s in ss])
            w = np.stack([arrays["test"]["weather"][s:s + L] for s in ss])
            window_clusters = np.asarray([arrays["test"]["cluster"][int(s) + L - 1] for s in ss], dtype=np.int64)
            masks = np.stack([mask_window(L, mask_type, value, rng) for _ in ss])
            pred = reconstruct_batch(method, net, device, p, w, masks, cfg, window_clusters)
            pred = np.where(masks > 0, p, pred)
            for j, s in enumerate(ss):
                for ch, field in enumerate(POWER):
                    miss = masks[j, :, ch] == 0
                    if not miss.any(): continue
                    y = inverse(p[j, miss, ch], scales["power"][field]); q = inverse(pred[j, miss, ch], scales["power"][field])
                    clusters = arrays["test"]["cluster"][s + np.flatnonzero(miss)]
                    for c in np.unique(clusters):
                        take = clusters == c; bucket = regime[(field, int(c))]; bucket["y"].extend(y[take].tolist()); bucket["p"].extend(q[take].tolist()); bucket["windows"].add(int(s))
        for field in POWER:
            # Grouping stratifies test errors for every method. It does not
            # change how the LSTM, interpolation, or all-weather models train.
            for c in range(len(REGIMES)):
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
    """Place all four methods side by side for every matched test condition."""
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

    expected = {f"{method}_{metric}" for method in EVAL_METHODS for metric in GROUPED_METRICS}
    output = []
    for key, entry in combined.items():
        missing = expected.difference(entry)
        if missing:
            raise ValueError(f"Grouped comparison is missing methods/metrics for {key}: {sorted(missing)}")
        output.append(entry)
    return sorted(output, key=lambda row: tuple(str(row.get(field, "")) for field in GROUPED_KEYS[:4]))

def run(config_path: Path, quick=False):
    """Train and evaluate all methods, exporting only the grouped comparison."""
    root = config_path.parent; cfg = json.loads(config_path.read_text(encoding="utf-8"));
    if quick: cfg.update({"train_max_epochs": 1, "early_stopping_patience": 1, "batch_size": 16, "ddim_steps": 3, "ddim_samples": 1})
    work, results = root / "work_v2", (root / "results_v2" / "quick" if quick else root / "results"); data = (root / cfg["data_path"]).resolve(); arrays, indexes, scales, audit = build_dataset(data, work, cfg); log = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "audit": audit, "runs": []}; grouped = []
    total_runs = (len(TRAIN_METHODS) + 1) * len(cfg["train_seeds"])
    run_number = 0
    for method in TRAIN_METHODS:
        for seed in cfg["train_seeds"]:
            run_number += 1
            print(f"\n[{run_number}/{total_runs}] Starting {method}, seed={seed}", flush=True)
            try:
                net, device, info = train_one(arrays, indexes, cfg, work, method, seed)
                grouped.extend(evaluate(net, device, arrays, indexes, cfg, scales, method, seed))
                del net
                log["runs"].append({"method": method, "seed": seed, "status": "completed", **info})
                print(f"Completed {method}, seed={seed}", flush=True)
            except Exception as e: log["runs"].append({"method": method, "seed": seed, "status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
    # Reuse each neural-model seed so all methods see identical test masks.
    print("\nEvaluating linear interpolation baseline", flush=True)
    for seed in cfg["train_seeds"]:
        run_number += 1
        print(f"[{run_number}/{total_runs}] Evaluating linear interpolation, mask seed={seed}", flush=True)
        try:
            method_grouped = evaluate(None, torch.device("cpu"), arrays, indexes, cfg, scales, "linear_interpolation", seed)
            grouped.extend(method_grouped)
            log["runs"].append({"method": "linear_interpolation", "seed": seed, "status": "completed", "mask_seed": cfg["mask_seed"] + int(seed)})
        except Exception as e:
            log["runs"].append({"method": "linear_interpolation", "seed": seed, "status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
    results.mkdir(parents=True, exist_ok=True)
    grouped_wide = pivot_grouped_rows(grouped)
    expected_rows = len(POWER) * len(scenarios(cfg)) * len(REGIMES) * len(cfg["train_seeds"])
    if len(grouped_wide) != expected_rows:
        raise ValueError(f"Expected {expected_rows} complete grouped rows, found {len(grouped_wide)}")
    write_csv(results / "reconstruction_grouped_comparison.csv", grouped_wide)
    # Remove the previous table set so results/ contains only the requested comparison.
    for name in ("reconstruction_comparison.csv", "forecast_support_comparison.csv", "ablation_comparison.csv"):
        stale_result = results / name
        if stale_result.exists():
            stale_result.unlink()
    (work / "logs").mkdir(parents=True, exist_ok=True)
    log["grouped_comparison"] = {"status": "completed", "rows": len(grouped_wide), "methods": EVAL_METHODS}
    (work / "logs" / "run_summary.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    exported = (results / "reconstruction_grouped_comparison.csv").is_file()
    print(f"\nExported grouped comparison to: {results}; rows={len(grouped_wide)}; complete={exported}", flush=True)
    return log



