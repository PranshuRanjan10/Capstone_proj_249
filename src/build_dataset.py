"""
Assemble the multi-season master dataset.

Data lineage (in priority order):
  1. NASA POWER daily agroclimatology  →  observed weather + ET0
  2. Open-Meteo historical-forecast archive  →  genuine forecast rain (preferred)
  3. Synthetic noise-degraded rain  →  forecast rain fallback

Run this first: python src/build_dataset.py
"""
from __future__ import annotations
import sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import weather as W
from water_balance import simulate_season, add_sensor_noise
from features import build_features


def main(cfg=None):
    cfg = cfg or C.load()
    root = cfg["_root"]
    rng = np.random.default_rng(cfg["sensors"]["seed"])

    # ----- Observed weather source -----
    src = cfg["weather"]["source"]
    power = None
    if src == "power":
        p = root / cfg["weather"]["power_csv"]
        if not p.exists():
            print(f"[warn] {p} missing → falling back to synthetic weather")
            src = "synthetic"
        else:
            power = W.load_power_csv(p, cfg)
            print(f"[info] NASA POWER loaded: {len(power)} days "
                  f"({power.date.min().date()} → {power.date.max().date()})")

    # ----- Forecast source: Open-Meteo archive if available -----
    openmeteo = None
    om_path = root / cfg["weather"].get("openmeteo_csv", "data/raw/openmeteo_forecast.csv")
    if om_path.exists():
        openmeteo = pd.read_csv(om_path, parse_dates=["date"])
        print(f"[info] Open-Meteo forecast archive loaded: {len(openmeteo)} days")
    else:
        print(f"[info] Open-Meteo archive not found ({om_path}); "
              f"using synthetic forecast degradation. "
              f"Run src/fetch_openmeteo.py to upgrade.")

    frames = []
    skipped = 0
    for spec in cfg["seasons"]:
        for year in spec["years"]:
            start = dt.date(year, spec["transplant_month"], spec["transplant_day"])
            n = spec["season_length_days"]

            if src == "power" and power is not None:
                end = start + dt.timedelta(days=n - 1)
                wx = power[
                    (power.date.dt.date >= start) & (power.date.dt.date <= end)
                ].copy()
                if len(wx) < n:
                    print(f"[warn] {spec['name']} {year}: {len(wx)}/{n} POWER days; skipped")
                    skipped += 1
                    continue
                wx = wx.reset_index(drop=True)
            else:
                wx = W.synthetic_weather(cfg, year, start, n, rng)

            wx["year"] = year
            wx["season_name"] = spec["name"]
            wx["season_id"] = f"{spec['name']}-{year}"

            # Attach forecast columns (Open-Meteo archive or synthetic degradation)
            wx = W.make_forecast(wx, cfg, rng, openmeteo=openmeteo)

            sim = simulate_season(wx, cfg, rng, gdd_total=spec["gdd_total"])
            frames.append(add_sensor_noise(sim, cfg, rng))

    if not frames:
        raise RuntimeError("No seasons produced data — check weather source and date ranges.")

    full = pd.concat(frames, ignore_index=True)
    feat = build_features(full, cfg)

    outdir = root / "data" / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    full.to_csv(outdir / "field_state.csv", index=False)
    feat.to_parquet(outdir / "features.parquet", index=False)

    n_seasons = full.season_id.nunique()
    wx_source = full.weather_source.iloc[0]
    fc_source = full.weather_fc_source.iloc[0] if "weather_fc_source" in full.columns else "synthetic_degraded"

    print(f"\nDataset built: {n_seasons} seasons, {len(full)} rows")
    print(f"  Weather source  : {wx_source}")
    print(f"  Forecast source : {fc_source}")
    if skipped:
        print(f"  Seasons skipped : {skipped} (missing POWER days)")
    print("\nSeason summary:")
    print(full.groupby("season_name", observed=True)[["rain_mm", "irrigation_mm", "wl_true_mm"]]
              .agg({"rain_mm": "mean", "irrigation_mm": "sum", "wl_true_mm": ["mean", "min"]})
              .round(1))
    return feat


if __name__ == "__main__":
    main()
