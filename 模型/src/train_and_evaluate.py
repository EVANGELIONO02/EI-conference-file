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
TRAIN_METHODS = ["gan_imputer", "diffusion_no_weather", "proposed_weather_diffusion"]
EVAL_METHODS = ["linear_interpolation", *TRAIN_METHODS]
MASK_TYPES = ["random_point", "random_continuous"]

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

def model(cfg, weather): return ConditionalDiffusion(cfg["model_dim"], cfg["transformer_layers"], cfg["attention_heads"], cfg["diffusion_train_steps"], weather, cfg["window_length"])

class GANImputer(nn.Module):
    """Conditional GAN baseline: generator fills missing values and discriminator scores realism."""
    def __init__(self, hidden=128):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(21, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))
        self.discriminator = nn.Sequential(nn.Linear(21, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def generate(self, observed, mask, weather):
        generated = self.generator(torch.cat([observed, mask, weather], -1))
        return torch.where(mask.bool(), observed, generated)

def train_gan(arrays, indexes, cfg, work, seed):
    """Train the GAN imputer using reconstruction loss on missing positions."""
    seed_all(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(Windows(arrays["train"], indexes["train"], cfg["window_length"]), cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    net = GANImputer(cfg["model_dim"]).to(device); opt_g = torch.optim.AdamW(net.generator.parameters(), lr=cfg["learning_rate"]); opt_d = torch.optim.AdamW(net.discriminator.parameters(), lr=cfg["learning_rate"]); rng = np.random.default_rng(seed + 7000); logs = []
    for epoch in tqdm(range(1, cfg["train_max_epochs"] + 1), desc=f"train gan_imputer seed={seed}", unit="epoch"):
        total = 0.0
        for p, w, _ in tqdm(loader, desc=f"gan epoch {epoch}", leave=False, unit="batch"):
            p, w = p.to(device), w.to(device); pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(len(p))]; mask = torch.from_numpy(np.stack([mask_window(cfg["window_length"], a, b, rng) for a, b in pairs])).to(device); obs = p * mask
            fake = net.generate(obs, mask, w); real_score = net.discriminator(torch.cat([p, mask, w], -1)); fake_score = net.discriminator(torch.cat([fake.detach(), mask, w], -1)); d_loss = nn.functional.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score)) + nn.functional.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score)); opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            fake = net.generate(obs, mask, w); score = net.discriminator(torch.cat([fake, mask, w], -1)); miss = 1 - mask; rec = ((fake - p).abs() * miss).sum() / miss.sum().clamp_min(1); g_loss = rec + 0.01 * nn.functional.binary_cross_entropy_with_logits(score, torch.ones_like(score)); opt_g.zero_grad(); g_loss.backward(); opt_g.step(); total += float(g_loss.detach())
        logs.append({"epoch": epoch, "train_loss": total / max(len(loader), 1)}); tqdm.write(f"gan_imputer seed={seed} epoch={epoch}: loss={logs[-1]['train_loss']:.6f}")
    path = work / "checkpoints" / f"gan_imputer_seed{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": net.state_dict(), "method": "gan_imputer", "seed": seed}, path); write_csv(work / "logs" / f"train_gan_imputer_seed{seed}.csv", logs); return net, device, {"epochs_run": len(logs), "checkpoint": str(path)}

def train_one(arrays, indexes, cfg, work, method, seed):
    """Train one diffusion method and save its best validation checkpoint."""
    if method == "gan_imputer":
        return train_gan(arrays, indexes, cfg, work, seed)
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
                mask_type, value = scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))]; masks.append(mask_window(cfg["window_length"], mask_type, value, rng))
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
        p, w = p.to(device), w.to(device)
        pairs = [scenarios(cfg)[int(rng.integers(len(scenarios(cfg))))] for _ in range(len(p))]
        masks = torch.from_numpy(np.stack([mask_window(cfg["window_length"], a, b, rng) for a, b in pairs])).to(device)
        pred = net.sample(p * masks, masks, w if weather else None, cfg["validation_ddim_steps"], 1)
        sq += float((((pred - p) ** 2) * (1 - masks)).sum()); n += int((1 - masks).sum())
    return math.sqrt(sq / max(n, 1))

def inverse(x, scale): return np.asarray(x) * (scale["max"] - scale["min"] or 1) + scale["min"]
def metrics(y, p, scale):
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
    if method == "gan_imputer":
        pred = net.generate(pt, mt, wt)
    else:
        pred = net.sample(pt, mt, wt if method != "diffusion_no_weather" else None, cfg["ddim_steps"], cfg["ddim_samples"])
    return pred.detach().cpu().numpy()

def evaluate(net, device, arrays, indexes, cfg, scales, method, seed):
    """Evaluate one method for every mask scenario and weather regime."""
    rng = np.random.default_rng(cfg["mask_seed"] + seed); starts = indexes["test"]; L = cfg["window_length"]; all_rows = []; grouped = []
    for mask_type, value in tqdm(scenarios(cfg), desc=f"evaluate {method} seed={seed}", unit="scenario"):
        vals = {x: {"y": [], "p": [], "clusters": [], "windows": set()} for x in POWER}
        regime = {(field, c): {"y": [], "p": [], "windows": set()} for field in POWER for c in range(3)}
        batch_size = min(cfg["batch_size"], 256)
        for offset in tqdm(range(0, len(starts), batch_size), desc=f"mask {mask_type}:{value}", leave=False, unit="batch"):
            ss = starts[offset:offset + batch_size]
            p = np.stack([arrays["test"]["power"][s:s + L] for s in ss])
            w = np.stack([arrays["test"]["weather"][s:s + L] for s in ss])
            masks = np.stack([mask_window(L, mask_type, value, rng) for _ in ss])
            pred = reconstruct_batch(method, net, device, p, w, masks, cfg)
            pred = np.where(masks > 0, p, pred)
            for j, s in enumerate(ss):
                for ch, field in enumerate(POWER):
                    miss = masks[j, :, ch] == 0
                    if not miss.any(): continue
                    y = inverse(p[j, miss, ch], scales["power"][field]); q = inverse(pred[j, miss, ch], scales["power"][field])
                    clusters = arrays["test"]["cluster"][s + np.flatnonzero(miss)]
                    vals[field]["y"].extend(y.tolist()); vals[field]["p"].extend(q.tolist()); vals[field]["windows"].add(int(s))
                    for c in np.unique(clusters):
                        take = clusters == c; bucket = regime[(field, int(c))]; bucket["y"].extend(y[take].tolist()); bucket["p"].extend(q[take].tolist()); bucket["windows"].add(int(s))
        for field in POWER:
            mm = metrics(vals[field]["y"], vals[field]["p"], scales["power"][field]); row = {"method_id": method, "target_field": field, "mask_type": mask_type, "train_seed": seed, "n_windows": len(vals[field]["windows"]), "n_missing_points": mm.pop("n_points"), **mm}
            if mask_type == "random_point": row["missing_rate"] = value
            else: row["missing_length"] = value
            all_rows.append(row)
            for c in range(3):
                b = regime[(field, c)]; gm = metrics(b["y"], b["p"], scales["power"][field]); grow = {"method_id": method, "target_field": field, "mask_type": mask_type, "weather_regime": REGIMES[c], "train_seed": seed, "n_windows": len(b["windows"]), "n_missing_points": gm.pop("n_points"), **gm}
                if mask_type == "random_point": grow["missing_rate"] = value
                else: grow["missing_length"] = value
                grouped.append(grow)
    return all_rows, grouped, all_rows
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
    """Compare complete, masked, and reconstructed histories per mask condition."""
    rows = []; L = cfg["window_length"]; rng = np.random.default_rng(cfg["mask_seed"] + 700); device = next(iter(models.values()))[1] if models else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for target in tqdm(POWER, desc="forecast support", unit="target"):
        ci = POWER.index(target)
        def select(split):
            data = arrays[split]; hs=[]; ws=[]; ys=[]
            for s in indexes[split]:
                s=int(s); hs.append(data["power"][s:s+L].copy()); ws.append(data["weather"][s+L-1]); ys.append(data["power"][s+L-1,ci])
            return np.asarray(hs,np.float32), np.asarray(ws,np.float32), np.asarray(ys,np.float32)
        th,tw,ty=select("train"); vh,vw,vy=select("val"); eh,ew,ey=select("test")
        X=np.concatenate([th.reshape(len(th),-1),tw],1); VX=np.concatenate([vh.reshape(len(vh),-1),vw],1); mu,sd=ty.mean(),ty.std() or 1
        predictor=ForecastNet(X.shape[1],cfg["forecast_hidden_dim"]).to(device); opt=torch.optim.AdamW(predictor.parameters(),lr=1e-3,weight_decay=1e-4)
        tx=torch.tensor(X,device=device); yy=torch.tensor((ty-mu)/sd,device=device,dtype=torch.float32)[:,None]
        for _ in range(cfg["forecast_epochs"]): opt.zero_grad(); loss=nn.functional.mse_loss(predictor(tx),yy); loss.backward(); opt.step()
        for mask_type,value in scenarios(cfg):
            masks=np.stack([mask_window(L,mask_type,value,rng) for _ in eh]); masked=eh*masks; linear=np.stack([linear_fill(masked[i],masks[i]) for i in range(len(eh))])
            variants={"complete":eh,"masked":masked,"linear_interpolation":linear}
            for (method,seed),(net,d) in models.items():
                w=torch.tensor(np.repeat(ew[:,None,:],L,1),device=d); out=reconstruct_batch(method,net,d,eh, np.repeat(ew[:,None,:],L,1), masks, cfg); variants[f"{method}_{seed}"]=out
            for setting,hist in variants.items():
                fx=torch.tensor(np.concatenate([hist.reshape(len(hist),-1),ew],1),device=device,dtype=torch.float32); pred=predictor(fx).detach().cpu().numpy().ravel()*sd+mu; mm=metrics(inverse(ey,scales["power"][target]),inverse(pred,scales["power"][target]),scales["power"][target]); row={"target_field":target,"mask_type":mask_type,"history_setting":"complete" if setting=="complete" else ("masked" if setting=="masked" else "reconstructed"),"reconstruction_method":"none" if setting in {"complete","masked"} else setting,"n_windows":len(ey),**mm,"status":"completed"}; row["missing_rate" if mask_type=="random_point" else "missing_length"]=value; rows.append(row)
    return rows
def run(config_path: Path, quick=False):
    """Run data preparation, six training jobs, evaluation, and CSV export."""
    root = config_path.parent; cfg = json.loads(config_path.read_text(encoding="utf-8"));
    if quick: cfg.update({"train_max_epochs": 1, "early_stopping_patience": 1, "batch_size": 16, "ddim_steps": 3, "ddim_samples": 1, "forecast_epochs": 1})
    work, results = root / "work_v2", (root / "results_v2" / "quick" if quick else root / "results"); data = (root / cfg["data_path"]).resolve(); arrays, indexes, scales, audit = build_dataset(data, work, cfg); log = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "audit": audit, "runs": []}; overall, grouped, scenarios_rows = [], [], []; models = {}
    total_runs = len(TRAIN_METHODS) * len(cfg["train_seeds"])
    run_number = 0
    for method in TRAIN_METHODS:
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



