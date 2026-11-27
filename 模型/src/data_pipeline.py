from __future__ import annotations

import csv, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

POWER = ("actual_power",)
WEATHER = (
    "air_temperature", "momentum_flux", "wind_dir_30m", "wind_speed_170m", "wind_speed_100m",
    "wind_speed_30m", "wind_speed_10m", "wind_speed_0m", "wind_dir_170m", "wind_dir_100m",
    "wind_dir_10m", "wind_dir_0m", "sea_level_pressure", "cloud_cover", "sensible_heat_flux",
    "latent_heat_flux", "shortwave_radiation", "longwave_radiation", "surface_pressure",
    "unknown_feature", "large_scale_precipitation", "convective_precipitation", "temperature_2m", "humidity",
)
WIND_REGIME_KEY = "wind_speed_100m"
WIND_REGIME_THRESHOLDS = (3.0, 11.0, 20.0)
REGIME_NAMES = ["below_cut_in", "below_rated", "rated_to_high", "above_high_wind"]
TIME = "timestamp"
TIME_FORMAT = "%Y/%m/%d %H:%M"
SAMPLE_INTERVAL_MINUTES = 15
SAMPLE_COLUMN_MAP = {
    "实际功率": POWER[0], "气温": "air_temperature", "动量通量": "momentum_flux", "30米风向": "wind_dir_30m",
    "170米风速": "wind_speed_170m", "100米风速": "wind_speed_100m", "30米风速": "wind_speed_30m",
    "10米风速": "wind_speed_10m", "0米风速": "wind_speed_0m", "170米风向": "wind_dir_170m",
    "100米风向": "wind_dir_100m", "10米风向": "wind_dir_10m", "0米风向": "wind_dir_0m",
    "海平面气压": "sea_level_pressure", "云量": "cloud_cover", "感热通量": "sensible_heat_flux",
    "潜热通量": "latent_heat_flux", "短波辐射": "shortwave_radiation", "长波辐射": "longwave_radiation",
    "表面气压": "surface_pressure", "未知特征": "unknown_feature", "大尺度降水": "large_scale_precipitation",
    "对流降水": "convective_precipitation", "2米气温": "temperature_2m", "湿度": "humidity"
}

def _csv(path, rows, fields):
    """Write an internal trace CSV using UTF-8 with BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def wind_regime_labels(wind_speed, thresholds=WIND_REGIME_THRESHOLDS):
    """Map wind speed in m/s to fixed operating-regime labels."""
    values = np.asarray(wind_speed, dtype=np.float32)
    thresholds = tuple(float(value) for value in thresholds)
    if not np.all(np.isfinite(values)):
        raise ValueError("wind speed values must be finite")
    if len(thresholds) != 3 or not all(np.isfinite(thresholds)) or sorted(thresholds) != list(thresholds):
        raise ValueError("exactly three increasing finite wind regime thresholds are required")
    return np.digitize(values, thresholds).astype(np.int64)

def _split_chronologically(rows, names, ratios, key):
    if (len(ratios) != len(names) or any(not np.isfinite(x) or x <= 0 for x in ratios)
            or not np.isclose(sum(ratios), 1.0)):
        raise ValueError(f"{key} must contain {len(names)} positive shares summing to 1")
    boundaries = [0, *(int(len(rows) * sum(ratios[:i])) for i in range(1, len(names))), len(rows)]
    splits = {name: rows[boundaries[i]:boundaries[i + 1]] for i, name in enumerate(names)}
    for name, split_rows in splits.items():
        if not split_rows:
            raise ValueError(f"{key}: {name} split contains no valid records")
    return splits


def build_dataset(data_path: Path, work: Path, cfg):
    """Split one turbine's clean timeline into disjoint Diffusion and LSTM stages."""
    with data_path.open("r", encoding="utf-8-sig", newline="") as f: raw = list(csv.DictReader(f))
    required_source = ["date", *SAMPLE_COLUMN_MAP]
    if (len(SAMPLE_COLUMN_MAP) != len(POWER) + len(WEATHER)
            or set(SAMPLE_COLUMN_MAP.values()) != set(POWER + WEATHER)
            or len(set(SAMPLE_COLUMN_MAP.values())) != len(SAMPLE_COLUMN_MAP)
            or len(set(SAMPLE_COLUMN_MAP)) != len(SAMPLE_COLUMN_MAP)):
        raise ValueError("source column map must cover each model feature exactly once")
    if not raw or any(key not in raw[0] for key in required_source):
        missing = [key for key in required_source if not raw or key not in raw[0]]
        raise ValueError(f"CSV missing required fields: {missing}")
    latest = {}
    for source_row in raw:
        timestamp = datetime.strptime(source_row["date"].strip(), TIME_FORMAT)
        latest[timestamp] = source_row
    duplicates = len(raw) - len(latest)
    ordered = [
        {TIME: row["date"], "_dt": timestamp,
         **{target: row[source] for source, target in SAMPLE_COLUMN_MAP.items()}}
        for timestamp, row in sorted(latest.items())
    ]
    regime_key = cfg.get("wind_regime_key", WIND_REGIME_KEY)
    if regime_key not in WEATHER:
        raise ValueError(f"wind_regime_key must name a weather feature: {regime_key}")
    if cfg.get("window_length", 0) < 1 or cfg.get("window_stride", 0) < 1:
        raise ValueError("window_length and window_stride must be positive")
    if cfg.get("forecast_input_steps", 0) < 1 or cfg.get("forecast_steps", 0) < 1:
        raise ValueError("forecast_input_steps and forecast_steps must be positive")
    if cfg["forecast_input_steps"] + cfg["forecast_steps"] > cfg["window_length"]:
        raise ValueError("forecast input and target must fit inside window_length")
    clean, invalid, stopped_removed = [], 0, 0
    for row in ordered:
        vals = {}
        try:
            for key in [*POWER, *WEATHER]:
                vals[key] = float(row[key])
                if not np.isfinite(vals[key]): raise ValueError
        except (ValueError, TypeError):
            invalid += 1; continue
        row.update(vals)
        if cfg.get("low_wind_filter_enabled", False) and vals[regime_key] < float(cfg.get("low_wind_cut_in_m_s", 3.0)) and vals[POWER[0]] == 0.0:
            stopped_removed += 1
            continue
        clean.append(row)
    n = len(clean)
    midpoint = n // 2
    diffusion_splits = _split_chronologically(clean[:midpoint], ("train", "val"),
                                               cfg["diffusion_split_ratio"], "diffusion_split_ratio")
    forecast_splits = _split_chronologically(clean[midpoint:], ("train", "val", "test"),
                                              cfg["forecast_split_ratio"], "forecast_split_ratio")
    # Fit shared normalization before either validation split or the forecast half.
    scalers = _fit_scalers(diffusion_splits["train"], cfg)
    audit = {"data_path": str(data_path), "rows_raw": len(raw), "rows_clean": n,
             "duplicates_replaced": duplicates, "invalid_removed": invalid,
             "low_wind_filter_enabled": bool(cfg.get("low_wind_filter_enabled", False)),
             "low_wind_stopped_removed": stopped_removed,
             "sample_interval_minutes": SAMPLE_INTERVAL_MINUTES,
             "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
             "scaler_fit_split": "diffusion/train"}
    diffusion_arrays, diffusion_indexes, diffusion_audit = _prepare_splits(diffusion_splits, work / "diffusion", cfg, scalers, regime_key)
    arrays, indexes, forecast_audit = _prepare_splits(forecast_splits, work / "forecast", cfg, scalers, regime_key)
    audit.update(diffusion=diffusion_audit, forecast=forecast_audit)
    return diffusion_arrays, diffusion_indexes, arrays, indexes, scalers, audit


def _fit_scalers(rows, cfg):
    scalers = {"power": {}, "weather": {}}
    power_min = cfg.get("power_scaler_min")
    power_max = cfg.get("power_scaler_max")
    if (power_min is None) != (power_max is None):
        raise ValueError("power_scaler_min and power_scaler_max must be set together")
    if power_min is not None and float(power_max) <= float(power_min):
        raise ValueError("power_scaler_max must exceed power_scaler_min")
    for key in POWER:
        x = np.array([r[key] for r in rows])
        scalers["power"][key] = ({"min": float(power_min), "max": float(power_max)} if power_min is not None
                                  else {"min": float(x.min()), "max": float(x.max())})
    for key in WEATHER:
        x = np.array([r[key] for r in rows])
        scalers["weather"][key] = {"mean": float(x.mean()), "std": float(x.std() or 1)}
    return scalers


def _prepare_splits(splits, work, cfg, scalers, regime_key):
    """Normalize each split and create windows that never cross time gaps or split boundaries."""
    (work / "windows").mkdir(parents=True, exist_ok=True)
    thresholds = tuple(cfg.get("wind_regime_thresholds_m_s", WIND_REGIME_THRESHOLDS))
    regimes = list(cfg.get("wind_regime_names", REGIME_NAMES))
    if len(regimes) != len(thresholds) + 1:
        raise ValueError("wind_regime_names must contain one more entry than thresholds")
    if regimes != REGIME_NAMES:
        raise ValueError(f"wind_regime_names must match the model's fixed order: {REGIME_NAMES}")
    out_arrays, indexes, meta = {}, {}, {}
    for split, rows in splits.items():
        p = np.array([[(r[k] - scalers["power"][k]["min"]) / (scalers["power"][k]["max"] - scalers["power"][k]["min"] or 1) for k in POWER] for r in rows], dtype=np.float32)
        p = np.clip(p, 0.0, 1.0)
        w = np.array([[(r[k] - scalers["weather"][k]["mean"]) / scalers["weather"][k]["std"] for k in WEATHER] for r in rows], dtype=np.float32)
        wind = np.array([r[regime_key] for r in rows], dtype=np.float32)
        labels = wind_regime_labels(wind, thresholds)
        # Any gap after filtering starts a new segment; windows stay within observed cadence.
        segment = np.zeros(len(rows), dtype=np.int32)
        sample_delta = timedelta(minutes=SAMPLE_INTERVAL_MINUTES)
        for i in range(1, len(rows)):
            segment[i] = segment[i - 1] + int(rows[i]["_dt"] - rows[i - 1]["_dt"] != sample_delta)
        starts = []
        for sid in np.unique(segment):
            pos = np.flatnonzero(segment == sid)
            if len(pos) >= cfg["window_length"]: starts.extend(range(int(pos[0]), int(pos[-1]) - cfg["window_length"] + 2, cfg["window_stride"]))
        out_arrays[split] = {"power": p, "weather": w, "cluster": labels, "segment": segment}
        indexes[split] = np.asarray(starts, dtype=np.int32)
        meta[split] = {"timestamps": np.array([r[TIME] for r in rows]), "window_end": np.asarray([s + cfg["window_length"] - 1 for s in starts], dtype=np.int32)}
        base = [{TIME: r[TIME], **{k: r[k] for k in POWER}, **{k: r[k] for k in WEATHER}, "wind_regime": int(labels[i]), "wind_regime_name": regimes[int(labels[i])], "segment_id": int(segment[i])} for i, r in enumerate(rows)]
        _csv(work / "data" / f"{split}.csv", base, [TIME, *POWER, *WEATHER, "wind_regime", "wind_regime_name", "segment_id"])
        for j, regime in enumerate(regimes): _csv(work / "data" / "clustered" / f"{split}_{regime}.csv", [x for x in base if x["wind_regime"] == j], [TIME, *POWER, *WEATHER, "wind_regime", "wind_regime_name", "segment_id"])
        np.savez_compressed(work / "windows" / f"{split}.npz", power=p, weather=w, cluster=labels, segment=segment, starts=indexes[split], window_end=meta[split]["window_end"], timestamps=meta[split]["timestamps"])
    (work / "data").mkdir(parents=True, exist_ok=True); (work / "data" / "scalers.json").write_text(json.dumps(scalers, indent=2), encoding="utf-8")
    audit = {"split_rows": {k: len(v) for k, v in splits.items()},
             "window_counts": {k: len(v) for k, v in indexes.items()},
             "segment_counts": {k: int(np.unique(v["segment"]).size) for k, v in out_arrays.items()},
             "time_ranges": {k: {"start": v[0]["_dt"].isoformat(), "end": v[-1]["_dt"].isoformat()}
                             for k, v in splits.items()}}
    (work / "data" / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out_arrays, indexes, audit
