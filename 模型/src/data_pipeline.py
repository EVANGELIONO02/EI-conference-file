from __future__ import annotations

import csv, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

POWER = ["Power_0", "Power_15", "Power_30", "Power_45"]
# Thirteen weather features: wind speed, sine/cosine direction components,
# temperature, pressure, and dew point.
WEATHER = ["WS10_A", "WDSIN10_A", "WDCOS10_A", "WS100_A", "WDSIN100_A", "WDCOS100_A", "WS200_A", "WDSIN200_A", "WDCOS200_A", "Sea_Surface_Temperature_K", "Mean_Sea_Level_Pressure_Pa", "Temperature_2m_K", "Dew_Point_2m_K"]
TIME = "Valid_Time"

def _dt(x):
    """Parse a timezone-aware Valid_Time value."""
    value = datetime.fromisoformat(x.strip().replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Valid_Time must include timezone")
    return value

def _csv(path, rows, fields):
    """Write an internal trace CSV using UTF-8 with BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def _fit_kmeans(x, k, seed=11, steps=80):
    """Fit the fixed K-Means weather clustering on training features only."""
    rng = np.random.default_rng(seed); centers = x[rng.choice(len(x), k, replace=False)].copy(); labels = None
    for _ in range(steps):
        new = ((x[:, None] - centers[None]) ** 2).sum(2).argmin(1)
        next_centers = centers.copy()
        for j in range(k):
            if np.any(new == j): next_centers[j] = x[new == j].mean(0)
        if labels is not None and np.array_equal(labels, new): break
        centers, labels = next_centers, new
    return centers, labels

def build_dataset(data_path: Path, work: Path, cfg):
    """Clean data, split time ranges, normalize features, cluster weather,
    and create windows that never cross time gaps."""
    (work / "data" / "clustered").mkdir(parents=True, exist_ok=True)
    (work / "windows").mkdir(parents=True, exist_ok=True)
    with data_path.open("r", encoding="utf-8-sig", newline="") as f: raw = list(csv.DictReader(f))
    required = [TIME, *POWER, *WEATHER]
    if not raw or any(k not in raw[0] for k in required): raise ValueError("CSV missing required fields")
    # Keep the last row for a duplicate timestamp and record the count.
    latest, duplicates = {}, 0
    for row in raw:
        t = _dt(row[TIME]); duplicates += t in latest; row["_dt"] = t; latest[t] = row
    ordered = sorted(latest.values(), key=lambda r: r["_dt"])
    clean, invalid = [], 0
    for row in ordered:
        vals = {}
        try:
            for key in [*POWER, *WEATHER]:
                vals[key] = float(row[key])
                if not np.isfinite(vals[key]): raise ValueError
        except (ValueError, TypeError):
            invalid += 1; continue
        row.update(vals); clean.append(row)
    # Split chronologically before making windows to prevent leakage.
    n = len(clean); a = int(n * cfg["split_ratio"][0]); b = int(n * sum(cfg["split_ratio"][:2]))
    splits = {"train": clean[:a], "val": clean[a:b], "test": clean[b:]}
    scalers = {"power": {}, "weather": {}}
    # Fit every scaler on training data only.
    for key in POWER:
        x = np.array([r[key] for r in splits["train"]]); scalers["power"][key] = {"min": float(x.min()), "max": float(x.max())}
    for key in WEATHER:
        x = np.array([r[key] for r in splits["train"]]); scalers["weather"][key] = {"mean": float(x.mean()), "std": float(x.std() or 1)}
    out_arrays, indexes, meta = {}, {}, {}
    train_weather = np.array([[(r[k] - scalers["weather"][k]["mean"]) / scalers["weather"][k]["std"] for k in WEATHER] for r in splits["train"]], dtype=np.float32)
    # Use K=3 and rename clusters by ascending mean WS100_A.
    centers, train_labels = _fit_kmeans(train_weather, 3)
    means = [np.mean([r["WS100_A"] for r, y in zip(splits["train"], train_labels) if y == j]) if np.any(train_labels == j) else np.inf for j in range(3)]
    remap = {int(old): new for new, old in enumerate(np.argsort(means))}
    regimes = ["low_wind", "medium_wind", "high_wind"]
    for split, rows in splits.items():
        p = np.array([[(r[k] - scalers["power"][k]["min"]) / (scalers["power"][k]["max"] - scalers["power"][k]["min"] or 1) for k in POWER] for r in rows], dtype=np.float32)
        w = np.array([[(r[k] - scalers["weather"][k]["mean"]) / scalers["weather"][k]["std"] for k in WEATHER] for r in rows], dtype=np.float32)
        labels = train_labels if split == "train" else ((w[:, None] - centers[None]) ** 2).sum(2).argmin(1)
        labels = np.array([remap[int(x)] for x in labels], dtype=np.int64)
        # A non-hourly gap starts a new segment; windows stay inside segments.
        segment = np.zeros(len(rows), dtype=np.int32)
        for i in range(1, len(rows)):
            segment[i] = segment[i - 1] + int(rows[i]["_dt"] - rows[i - 1]["_dt"] != timedelta(hours=1))
        starts = []
        for sid in np.unique(segment):
            pos = np.flatnonzero(segment == sid)
            if len(pos) >= cfg["window_length"]: starts.extend(range(int(pos[0]), int(pos[-1]) - cfg["window_length"] + 2, cfg["window_stride"]))
        out_arrays[split] = {"power": p, "weather": w, "cluster": labels, "segment": segment}
        indexes[split] = np.asarray(starts, dtype=np.int32)
        meta[split] = {"timestamps": np.array([r[TIME] for r in rows]), "window_end": np.asarray([s + cfg["window_length"] - 1 for s in starts], dtype=np.int32)}
        base = [{TIME: r[TIME], **{k: r[k] for k in POWER}, **{k: r[k] for k in WEATHER}, "weather_cluster": int(labels[i]), "segment_id": int(segment[i])} for i, r in enumerate(rows)]
        _csv(work / "data" / f"{split}.csv", base, [TIME, *POWER, *WEATHER, "weather_cluster", "segment_id"])
        for j, regime in enumerate(regimes): _csv(work / "data" / "clustered" / f"{split}_{regime}.csv", [x for x in base if x["weather_cluster"] == j], [TIME, *POWER, *WEATHER, "weather_cluster", "segment_id"])
        np.savez_compressed(work / "windows" / f"{split}.npz", power=p, weather=w, cluster=labels, segment=segment, starts=indexes[split], window_end=meta[split]["window_end"], timestamps=meta[split]["timestamps"])
    (work / "data").mkdir(parents=True, exist_ok=True); (work / "data" / "scalers.json").write_text(json.dumps(scalers, indent=2), encoding="utf-8")
    audit = {"rows_raw": len(raw), "rows_clean": n, "duplicates_replaced": duplicates, "invalid_removed": invalid, "split_rows": {k: len(v) for k, v in splits.items()}, "window_counts": {k: len(v) for k, v in indexes.items()}, "segment_counts": {k: int(np.unique(v["segment"]).size) for k, v in out_arrays.items()}, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}
    (work / "data" / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); return out_arrays, indexes, scalers, audit
