"""
Feature engineering.

Leakage rules enforced here:
  * no feature may use a FUTURE observed value (only forecast rainfall, which is
    itself a degraded transform and is what deployment actually provides)
  * lags and rolling windows are strictly backward-looking
  * seasons never straddle train and test (handled by leave-one-season-out in train.py)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_PREFIX = "y_wl_t"


def build_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("season_id", sort=False)

    for lag in (1, 2, 3, 7):
        out[f"wl_lag{lag}"] = g["wl_obs_mm"].shift(lag)
        out[f"sm_lag{lag}"] = g["sm_obs_pct"].shift(lag)
    out["wl_delta1"] = out["wl_obs_mm"] - out["wl_lag1"]
    out["wl_delta3"] = out["wl_obs_mm"] - out["wl_lag3"]
    out["sm_delta3"] = out["sm_obs_pct"] - out["sm_lag3"]

    for w in (3, 7):
        out[f"rain_sum{w}"] = g["rain_mm"].transform(lambda s: s.rolling(w, min_periods=1).sum())
        out[f"et0_mean{w}"] = g["et0_mm"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    out["dry_days"] = g["rain_mm"].transform(
        lambda s: s.eq(0).groupby(s.ne(0).cumsum()).cumsum())

    out["et_demand"] = out["et0_mm"] * out["kc"]
    out["headroom_mm"] = cfg["hydrology"]["bund_height_mm"] - out["wl_obs_mm"]

    for h in cfg["model"]["horizons"]:
        out[f"{TARGET_PREFIX}{h}"] = g["wl_true_mm"].shift(-h)

    return out


FEATURES = [
    "wl_obs_mm", "sm_obs_pct",
    "wl_lag1", "wl_lag2", "wl_lag3", "wl_lag7",
    "sm_lag1", "sm_lag3",
    "wl_delta1", "wl_delta3", "sm_delta3",
    "rain_mm", "rain_sum3", "rain_sum7", "dry_days",
    "rain_fc_t1_mm", "rain_fc_t2_mm", "rain_fc_t3_mm", "rain_fc_sum3_mm",
    "et0_mm", "et0_mean3", "et0_mean7", "et_demand", "kc",
    "cum_gdd", "gdd_fraction", "stage_idx", "das", "headroom_mm",
]
