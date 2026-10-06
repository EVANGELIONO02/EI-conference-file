from __future__ import annotations
import csv, json, math, random, time, traceback
from pathlib import Path
import numpy as np, torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from data_pipeline import POWER, build_dataset
from diffusion_model import ConditionalDiffusion

REGIMES = ["low_wind", "medium_wind", "high_wind"]
METHODS = ["diffusion_no_weather", "proposed_weather_diffusion"]

def seed_all(seed):
    """Fix Python, NumPy, and PyTorch randomness for reproducible seeds."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def mask_window(L, rate, block, rng):
    """Create a four-channel mixed mask: one block plus random points."""
    m = int(math.floor(L * rate + .5)); out = np.ones((L, 4), dtype=np.float32)
    if block >= m: return None
    for ch in range(4):
        start = int(rng.integers(0, L - block + 1)); out[start:start + block, ch] = 0
        candidates = np.flatnonzero(out[:, ch]); candidates = candidates[(candidates < start - 1) | (candidates > start + block)]
        if m - block > len(candidates): return None
        out[rng.choice(candidates, m - block, replace=False), ch] = 0
    return out

def scenarios(cfg):
    """Return all feasible (missing_rate, block_length) scenarios."""
    result = []
    for rate in cfg["missing_rates"]:
        m = int(math.floor(cfg["window_length"] * rate + .5))
        for block in cfg["block_lengths"]:
            if block < m: result.append((rate, block))
    return result

class Windows(Dataset):
    """Expose continuous arrays as PyTorch windows."""
    def __init__(self, arrays, starts, L): self.a, self.starts, self.L = arrays, starts, L
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = int(self.starts[i]); e = s + self.L
        return torch.from_numpy(self.a["power"][s:e]), torch.from_numpy(self.a["weather"][s:e]), torch.tensor(int(self.a["cluster"][e - 1]))

def model(cfg, weather): return ConditionalDiffusion(cfg["model_dim"], cfg["transformer_layers"], cfg["attention_heads"], cfg["diffusion_train_steps"], weather, cfg["window_length"])

def train_one(arrays, indexes, cfg, work, method, seed):
    """Train one diffusion method and save its best validation checkpoint."""
    seed_all(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); weather = method != "diffusion_no_weather"
    train = DataLoader(Windows(arrays["train"], indexes["train"], cfg["window_length"]), cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val = DataLoader(Windows(arrays["val"], indexes["val"], cfg["window_length"]), cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    net = model(cfg, weather).to(device); opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]); best = float("inf"); state = None; stale = 0; logs = []
    rng = np.random.default_rng(seed + 9001); counts = np.bincount(arrays["train"]["cluster"], minlength=3).astype(float); weights = torch.tensor((1 / np.maximum(counts, 1)) / np.mean(1 / np.maximum(counts, 1)), device=device, dtype=torch.float32)
    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"train {method} seed={seed}", unit="epoch"):
        net.train(); total = 0
        batches = tqdm(train, desc=f"epoch {epoch} batches", leave=False, unit="batch")
        for p, w, c in batches:
            p, w, c = p.to(device), w.to(device), c.to(device); masks = []
            for _ in range(len(p)):
                rate, block = scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))]; masks.append(mask_window(cfg["window_length"], rate, block, rng))
            mask = torch.from_numpy(np.stack(masks)).to(device); obs = p * mask; loss = (net.loss_per_sample(p, obs, mask, w if weather else None) * weights[c]).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1); opt.step(); total += float(loss.detach())
        val_score = validate(net, val, cfg, device, weather, seed + epoch); logs.append({"epoch": epoch, "train_loss": total / max(len(train), 1), "val_rmse": val_score})
        tqdm.write(f"{method} seed={seed} epoch={epoch}: val_rmse={val_score:.6f}")
        if val_score < best: best, state, stale = val_score, {k: v.detach().cpu() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]: break
    net.load_state_dict(state); net.eval(); path = work / "checkpoints" / f"{method}_seed{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": net.state_dict(), "method": method, "seed": seed}, path); write_csv(work / "logs" / f"train_{method}_seed{seed}.csv", logs); return net, device, {"best_val_rmse": best, "epochs_run": len(logs), "checkpoint": str(path)}

@torch.no_grad()
def validate(net, loader, cfg, device, weather, seed):
    """Evaluate normalized missing-position RMSE for early stopping."""
    rng = np.random.default_rng(seed); sq = n = 0
    for p, w, _ in tqdm(loader, desc="validation", leave=False, unit="batch"):
        p, w = p.to(device), w.to(device); pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(len(p))]; masks = torch.from_numpy(np.stack([mask_window(cfg["window_length"], a, b, rng) for a, b in pairs])).to(device); pred = net.sample(p * masks, masks, w if weather else None, cfg["validation_ddim_steps"], 1); sq += float((((pred - p) ** 2) * (1 - masks)).sum()); n += int((1 - masks).sum())
    return math.sqrt(sq / max(n, 1))

def inverse(x, scale): return np.asarray(x) * (scale["max"] - scale["min"] or 1) + scale["min"]
def metrics(y, p, scale):
    """Return absolute-error statistics on the original power scale."""
    y, p = np.asarray(y), np.asarray(p); abs_error = np.abs(p - y)
    if not len(y):
        return {"abs_error_max": "", "abs_error_min": "", "MAE": "", "n_points": 0}
    mean_error = float(np.mean(abs_error))
    return {"abs_error_max": float(np.max(abs_error)), "abs_error_min": float(np.min(abs_error)), "MAE": mean_error, "n_points": int(len(y))}

def evaluate(net, device, arrays, indexes, cfg, scales, method, seed):
    """Evaluate overall and weather-regime reconstruction with fixed masks."""
    rng = np.random.default_rng(cfg["mask_seed"]); starts = indexes["test"]; L = cfg["window_length"]; all_rows = []; grouped_bins = {}; pooled = {x: [] for x in POWER}; scenarios_data = scenarios(cfg)
    for rate, block in tqdm(scenarios_data, desc=f"evaluate {method} seed={seed}", unit="scenario"):
        vals = {x: {"y": [], "p": [], "obs": [], "clusters": []} for x in POWER}
        offsets = range(0, len(starts), min(cfg["batch_size"], 256))
        for offset in tqdm(offsets, desc=f"mask {rate:.2f}/{block}h", leave=False, unit="batch"):
            ss = starts[offset:offset + min(cfg["batch_size"], 256)]; p = np.stack([arrays["test"]["power"][s:s + L] for s in ss]); w = np.stack([arrays["test"]["weather"][s:s + L] for s in ss]); masks = np.stack([mask_window(L, rate, block, rng) for _ in ss]); pt = torch.tensor(p, device=device); wt = torch.tensor(w, device=device); mt = torch.tensor(masks, device=device); pred = net.sample(pt * mt, mt, wt, cfg["ddim_steps"], cfg["ddim_samples"]).cpu().numpy();
            for j, s in enumerate(ss):
                for ch, field in enumerate(POWER):
                    miss = masks[j, :, ch] == 0; obs = ~miss; y = inverse(p[j, miss, ch], scales["power"][field]); q = inverse(pred[j, miss, ch], scales["power"][field]); point_clusters = arrays["test"]["cluster"][s + np.flatnonzero(miss)]; vals[field]["y"].extend(y); vals[field]["p"].extend(q); vals[field]["obs"].append(float(np.max(np.abs(pred[j, obs, ch] - p[j, obs, ch]))) if obs.any() else 0); vals[field]["clusters"].extend(point_clusters.tolist()); pooled[field].append((y, q))
                    for cluster in np.unique(point_clusters):
                        take = point_clusters == cluster; key = (field, int(cluster)); bucket = grouped_bins.setdefault(key, {"y": [], "p": [], "windows": set()}); bucket["y"].extend(y[take]); bucket["p"].extend(q[take]); bucket["windows"].add(int(s))
        for field in POWER:
            scale = scales["power"][field]; v = vals[field]; mm = metrics(v["y"], v["p"], scale); all_rows.append({"method_id": method, "target_field": field, "mask_mode": "mixed", "missing_rate": rate, "block_length": block, "train_seed": seed, "row_type": "run", "n_windows": len(starts), "n_missing_points": mm.pop("n_points"), **mm, "observed_max_abs_error": max(v["obs"], default=0)})
    grouped = []
    for (field, cluster), bucket in grouped_bins.items():
        mm = metrics(bucket["y"], bucket["p"], scales["power"][field]); grouped.append({"method_id": method, "target_field": field, "weather_cluster": cluster, "weather_regime": REGIMES[cluster], "mask_mode": "mixed", "train_seed": seed, "row_type": "run", "n_windows": len(bucket["windows"]), "n_missing_points": mm.pop("n_points"), **mm})
    overall = []
    for field in POWER:
        y = np.concatenate([x[0] for x in pooled[field]]); p = np.concatenate([x[1] for x in pooled[field]]); mm = metrics(y, p, scales["power"][field]); overall.append({"method_id": method, "target_field": field, "train_seed": seed, "row_type": "run", "n_windows": len(starts), "n_missing_points": mm.pop("n_points"), **mm, "observed_max_abs_error": 0})
    return overall, grouped, all_rows

def write_csv(path, rows):
    """Write result rows using the union of their CSV fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    # Keep result tables focused on requested metrics and grouping fields.
    excluded = {"abs_error_mean", "observed_max_abs_error", "weather_cluster", "mask_mode", "row_type"}
    rows = [{key: value for key, value in row.items() if key not in excluded} for row in rows]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f: w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)

class ForecastNet(nn.Module):
    """Small feed-forward predictor used for forecast-support evaluation."""
    def __init__(self, dim, hidden):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)

def forecast_support(arrays, indexes, cfg, scales, models):
    """Compare complete, masked, and reconstructed histories for four targets."""
    rows = []; L = cfg["window_length"]; rng = np.random.default_rng(cfg["mask_seed"] + 700)
    for target in tqdm(POWER, desc="forecast support", unit="target"):
        ci = POWER.index(target)
        def select(split):
            data = arrays[split]; hs = []; ws = []; ys = []
            for s in indexes[split]:
                s = int(s); h = data["power"][s:s + L].copy(); h[-1, ci] = 0; hs.append(h); ws.append(data["weather"][s + L - 1]); ys.append(data["power"][s + L - 1, ci])
            return np.asarray(hs, dtype=np.float32), np.asarray(ws, dtype=np.float32), np.asarray(ys, dtype=np.float32)
        th, tw, ty = select("train"); vh, vw, vy = select("val"); eh, ew, ey = select("test")
        X = np.concatenate([th.reshape(len(th), -1), tw], 1); VX = np.concatenate([vh.reshape(len(vh), -1), vw], 1); EX = np.concatenate([eh.reshape(len(eh), -1), ew], 1)
        mu, sd = ty.mean(), ty.std() or 1; net = ForecastNet(X.shape[1], cfg["forecast_hidden_dim"]).to(next(iter(models.values()))[1] if models else torch.device("cuda" if torch.cuda.is_available() else "cpu")); device = next(net.parameters()).device; opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        tx, vx, yy, vy_t = [torch.tensor(z, device=device, dtype=torch.float32) for z in (X, VX, (ty - mu) / sd, (vy - mu) / sd)]
        for _ in range(cfg["forecast_epochs"]):
            opt.zero_grad(); loss = nn.functional.mse_loss(net(tx), yy[:, None]); loss.backward(); opt.step()
        histories = {"complete": eh.copy(), "masked": [], "linear_interpolation": []}
        preds = {key: [] for key in ["complete", "masked", "linear_interpolation", *METHODS]}
        for h in eh:
            mask = mask_window(L, .3, 3, rng); masked = h.copy(); masked[mask == 0] = .5; masked[-1, ci] = 0; linear = masked.copy()
            for ch in range(4):
                known = np.flatnonzero(mask[:, ch]);
                for pos in np.flatnonzero(mask[:, ch] == 0): linear[pos, ch] = h[known[-1], ch] if pos > known[-1] else h[known[0], ch]
            histories["masked"].append(masked); histories["linear_interpolation"].append(linear)
        for key, hs in histories.items():
            if len(hs) != len(ey): raise RuntimeError(f"forecast history length mismatch: {key}={len(hs)} test={len(ey)}")
            fx = torch.tensor(np.concatenate([np.asarray(hs).reshape(len(hs), -1), ew], 1), device=device, dtype=torch.float32)
            preds[key] = (net(fx).detach().cpu().numpy().ravel() * sd + mu)
        for (method, _seed), pair in models.items():
            net_diff, d = pair; batch = torch.tensor(eh, device=d); weather = torch.tensor(np.repeat(ew[:, None, :], L, 1), device=d); masks = torch.ones_like(batch); masks[:, -1, ci] = 0; out = net_diff.sample(batch * masks, masks, weather if method != "diffusion_no_weather" else None, cfg["ddim_steps"], cfg["ddim_samples"]); fx = torch.tensor(np.concatenate([out.cpu().numpy().reshape(len(eh), -1), ew], 1), device=device, dtype=torch.float32); preds[method] = (net(fx).detach().cpu().numpy().ravel() * sd + mu)
        y = inverse(ey, scales["power"][target])
        for key, values in preds.items():
            mm = metrics(y, inverse(values, scales["power"][target]), scales["power"][target]); rows.append({"target_field": target, "history_setting": "complete" if key == "complete" else ("masked" if key == "masked" else "reconstructed"), "reconstruction_method": key if key not in {"complete", "masked"} else "none", "n_windows": len(y), **mm, "status": "completed"})
    return rows

def run(config_path: Path, quick=False):
    """Run data preparation, six training jobs, evaluation, and CSV export."""
    root = config_path.parent; cfg = json.loads(config_path.read_text(encoding="utf-8"));
    if quick: cfg.update({"train_max_epochs": 1, "early_stopping_patience": 1, "batch_size": 16, "ddim_steps": 3, "ddim_samples": 1, "forecast_epochs": 1})
    work, results = root / "work_v2", (root / "results_v2" / "quick" if quick else root / "results"); data = (root / cfg["data_path"]).resolve(); arrays, indexes, scales, audit = build_dataset(data, work, cfg); log = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "audit": audit, "runs": []}; overall, grouped, scenarios_rows = [], [], []; models = {}
    total_runs = len(METHODS) * len(cfg["train_seeds"])
    run_number = 0
    for method in METHODS:
        for seed in cfg["train_seeds"]:
            run_number += 1
            print(f"\n[{run_number}/{total_runs}] Starting {method}, seed={seed}", flush=True)
            try:
                net, device, info = train_one(arrays, indexes, cfg, work, method, seed); a, b, c = evaluate(net, device, arrays, indexes, cfg, scales, method, seed); overall += a; grouped += b; scenarios_rows += c; models[(method, seed)] = (net, device); log["runs"].append({"method": method, "seed": seed, "status": "completed", **info}); print(f"Completed {method}, seed={seed}", flush=True)
            except Exception as e: log["runs"].append({"method": method, "seed": seed, "status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
    # The no-weather model is the sole ablation and reuses its already evaluated rows.
    ablation = [{"variant_id": "remove_weather", "removed_component": "weather_condition", **r} for r in overall if r["method_id"] == "diffusion_no_weather"]
    support_models = {k: v for k, v in models.items() if k[1] == cfg["train_seeds"][0]}
    print("\nStarting forecast-support evaluation", flush=True)
    support = forecast_support(arrays, indexes, cfg, scales, support_models)
    write_csv(results / "reconstruction_comparison.csv", overall); write_csv(results / "reconstruction_grouped_comparison.csv", grouped); write_csv(results / "forecast_support_comparison.csv", support); write_csv(results / "ablation_comparison.csv", ablation); (work / "logs").mkdir(parents=True, exist_ok=True); (work / "logs" / "run_summary.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"); print(f"\nExported four result CSV files to: {results}", flush=True); return log
