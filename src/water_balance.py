"""
Paddy water balance -- the physical model that generates field hydrology.

    WL(t+1) = WL(t) + rain + irrigation - ET - percolation,  capped by bund overflow

This is the crux of the project's honesty story. The ML layer NEVER sees these
equations; it sees only weather forcing and lagged state, and must learn to forecast
a nonlinear, weather-driven dynamical system. That is a real forecasting task with a
real target, which is exactly what the original "predict anaerobic risk" design lacked.

WL is millimetres relative to the soil surface:
    WL > 0  -> ponded water of that depth
    WL < 0  -> water table that far below the surface
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from staging import cumulative_gdd, stage_from_gdd


def _moisture_from_wl(wl, wl_floor):
    """Volumetric soil moisture proxy (%). Saturated when ponded, declining below."""
    sat, dry = 45.0, 24.0
    x = np.clip(wl / wl_floor, 0.0, 1.0)   # 0 at surface, 1 at floor
    return np.where(wl >= 0, sat, sat - (sat - dry) * x)


def _farmer_policy(wl, stage, cfg, rng):
    """Reference irrigation behaviour used only to GENERATE plausible training data.
    Deliberately imperfect: a real farmer reacts late and sometimes skips."""
    floor = cfg["awd"]["stage_floor_mm"].get(stage, cfg["awd"]["safe_threshold_mm"])
    # Real farmers react late, judge by eye, and sometimes skip a day. Modelling that
    # imperfection matters: a perfectly-managed field never needs a DSS, so a tidy
    # farmer policy would make the whole system look unnecessary by construction.
    trigger = floor + rng.normal(-5.0, 22.0)
    if wl <= trigger and rng.random() < 0.6:
        return max(cfg["hydrology"]["irrigation_target_mm"] - wl + rng.normal(0, 12), 0.0)
    return 0.0


def simulate_season(weather: pd.DataFrame, cfg, rng, gdd_total=None) -> pd.DataFrame:
    h = cfg["hydrology"]
    tmax = weather["tmax_c"].to_numpy()
    tmin = weather["tmin_c"].to_numpy()
    rain = weather["rain_mm"].to_numpy()
    et0 = weather["et0_mm"].to_numpy()

    cgdd = cumulative_gdd(tmax, tmin, cfg)
    stage, stage_i, kc, gfrac = stage_from_gdd(cgdd, cfg, gdd_total)

    n = len(weather)
    wl = np.zeros(n); irr = np.zeros(n); et = np.zeros(n)
    perc = np.zeros(n); overflow = np.zeros(n)

    level = 40.0  # transplanting: shallow ponding
    for t in range(n):
        wl[t] = level
        irr[t] = _farmer_policy(level, stage[t], cfg, rng)
        level += rain[t] + irr[t]

        if level > h["bund_height_mm"]:
            overflow[t] = level - h["bund_height_mm"]
            level = h["bund_height_mm"]

        # ET: full crop demand when ponded, reduced under water stress
        stress = 1.0 if level >= 0 else float(np.clip(1.0 + level / 250.0, 0.25, 1.0))
        et[t] = et0[t] * kc[t] * stress
        perc[t] = h["percolation_ponded_mm_day"] if level > 0 else h["percolation_unsat_mm_day"]

        level = max(level - et[t] - perc[t], h["wl_floor_mm"])

    df = weather.copy()
    df["cum_gdd"] = np.round(cgdd, 1)
    df["gdd_fraction"] = np.round(gfrac, 4)
    df["stage"] = stage
    df["stage_idx"] = stage_i
    df["kc"] = kc
    df["wl_true_mm"] = np.round(wl, 2)
    df["irrigation_mm"] = np.round(irr, 2)
    df["et_mm"] = np.round(et, 3)
    df["percolation_mm"] = perc
    df["overflow_mm"] = np.round(overflow, 2)
    df["sm_true_pct"] = np.round(_moisture_from_wl(wl, h["wl_floor_mm"]), 2)
    df["das"] = np.arange(n)   # days after transplanting
    return df


def add_sensor_noise(df: pd.DataFrame, cfg, rng) -> pd.DataFrame:
    s = cfg["sensors"]
    out = df.copy()
    out["wl_obs_mm"] = np.round(out["wl_true_mm"] + rng.normal(0, s["wl_noise_mm"], len(out)), 2)
    out["sm_obs_pct"] = np.round(
        np.clip(out["sm_true_pct"] + rng.normal(0, s["sm_noise_pct"], len(out)), 0, 60), 2)
    return out
