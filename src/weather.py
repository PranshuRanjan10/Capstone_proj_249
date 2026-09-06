"""
Weather layer.

Two sources:
  1. NASA POWER daily agroclimatology  (real, preferred)  -> fetch_power.py writes the CSV
  2. Synthetic stochastic climatology   (fallback)

The synthetic generator exists so the ENTIRE pipeline runs tonight without network
access. It is NOT a claim about real weather. Every artefact produced from it is
tagged weather_source='synthetic' and the dashboard says so on screen.

Forecast rainfall is generated as a NOISY transform of observed rainfall. This is
deliberate: at deployment the model only ever sees a forecast. Training on observed
rain and deploying on forecast rain is a silent leakage bug that makes validation
look good and field performance collapse.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GSC = 0.0820  # MJ m-2 min-1


def extraterrestrial_radiation(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    """Ra in mm/day water-equivalent (FAO-56 eq. 21, converted by 0.408)."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1, 1)
    ws = np.arccos(x)
    ra = (24 * 60 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws)
    )
    return ra * 0.408


def et0_hargreaves(tmax, tmin, lat_deg, doy):
    """Hargreaves-Samani ET0 (mm/day). Temperature-only: robust when radiation
    and wind are missing or unreliable. FAO-56 eq. 52."""
    tmean = (tmax + tmin) / 2.0
    ra = extraterrestrial_radiation(lat_deg, doy)
    trange = np.maximum(tmax - tmin, 0.1)
    return np.maximum(0.0023 * ra * (tmean + 17.8) * np.sqrt(trange), 0.0)


def _season_shape(doy):
    """NE-monsoon weighting for the Cauvery delta: peaks around 1 November."""
    return np.exp(-0.5 * ((((doy - 305) + 182) % 365 - 182) / 38.0) ** 2)


def synthetic_weather(cfg, year: int, start, ndays: int, rng) -> pd.DataFrame:
    dates = pd.date_range(start, periods=ndays, freq="D")
    doy = dates.dayofyear.to_numpy().astype(float)
    shape = _season_shape(doy)

    # Rainfall: two-state Markov occurrence + gamma depths on wet days.
    p_ww, p_dw = 0.55 + 0.20 * shape, 0.10 + 0.38 * shape
    wet = np.zeros(ndays, dtype=bool)
    wet[0] = rng.random() < p_dw[0]
    for i in range(1, ndays):
        wet[i] = rng.random() < (p_ww[i] if wet[i - 1] else p_dw[i])
    scale = 6.0 + 12.0 * shape
    depth = rng.gamma(shape=1.15, scale=scale, size=ndays)
    rain = np.where(wet, depth, 0.0)

    # Temperature: seasonal cycle, cooled on wet days.
    tmax = 34.5 - 4.5 * shape + rng.normal(0, 1.1, ndays) - 2.2 * wet
    tmin = 24.5 - 3.0 * shape + rng.normal(0, 0.8, ndays) - 0.6 * wet
    tmin = np.minimum(tmin, tmax - 2.0)

    df = pd.DataFrame({
        "date": dates, "year": year, "doy": doy,
        "rain_mm": np.round(rain, 2),
        "tmax_c": np.round(tmax, 2),
        "tmin_c": np.round(tmin, 2),
    })
    df["et0_mm"] = np.round(
        et0_hargreaves(df.tmax_c.to_numpy(), df.tmin_c.to_numpy(), cfg["site"]["lat"], doy), 3)
    df["weather_source"] = "synthetic"
    return df


def make_forecast(df: pd.DataFrame, cfg, rng,
                  openmeteo: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Attach forecast rainfall columns to df.

    Priority:
      1. Open-Meteo historical-forecast archive (genuine forecast, no leakage).
      2. Synthetic noise-degraded observed rain (fallback when archive unavailable).

    The fallback is explicitly labelled weather_fc_source='synthetic_degraded' in the
    output so downstream code and the dashboard can distinguish the two.

    Using forecast rain (not observed) in training is critical: the model is deployed
    with forecasts, not observations. Training on observed rain looks excellent in
    validation and collapses in the field — a silent leakage bug.
    """
    out = df.copy()
    horizons = cfg["model"]["horizons"]

    if openmeteo is not None and not openmeteo.empty:
        # Merge on date — use genuine forecast archive where available
        fc_cols = [f"rain_fc_t{h}_mm" for h in horizons] + ["rain_fc_sum3_mm"]
        merged = out.merge(
            openmeteo[["date"] + fc_cols],
            on="date", how="left", suffixes=("", "_om"),
        )
        # Fill any gaps (dates before archive starts) with synthetic degradation
        missing_mask = merged[f"rain_fc_t{horizons[0]}_mm"].isna()
        if missing_mask.any():
            _fill_synthetic(merged, cfg, rng, mask=missing_mask, horizons=horizons)
        # Flag the source
        merged["weather_fc_source"] = np.where(
            missing_mask, "synthetic_degraded", "openmeteo_archive"
        )
        return merged

    # Pure synthetic fallback
    _fill_synthetic(out, cfg, rng, mask=None, horizons=horizons)
    out["weather_fc_source"] = "synthetic_degraded"
    return out


def _fill_synthetic(df: pd.DataFrame, cfg, rng, mask, horizons) -> None:
    """In-place synthetic forecast fill (lognormal noise + false alarms)."""
    fs = cfg["weather"]["forecast_skill"]
    idx = df.index if mask is None else df.index[mask]
    for h in horizons:
        actual = df["rain_mm"].shift(-h).fillna(0.0).to_numpy()
        noise = rng.lognormal(0.0, fs["multiplicative_sd"], len(df))
        fa = np.where(
            rng.random(len(df)) < 0.12,
            rng.exponential(fs["false_alarm_mm"], len(df)),
            0.0,
        )
        synthetic = np.round(np.maximum(actual * noise + fa, 0.0), 2)
        col = f"rain_fc_t{h}_mm"
        if col not in df.columns:
            df[col] = np.nan
        if mask is None:
            df[col] = synthetic
        else:
            df.loc[mask, col] = synthetic[mask]
    sum_col = "rain_fc_sum3_mm"
    fc_cols = [f"rain_fc_t{h}_mm" for h in horizons]
    df[sum_col] = df[fc_cols].sum(axis=1)


def load_power_csv(path, cfg) -> pd.DataFrame:
    """Read the CSV produced by fetch_power.py (NASA POWER daily point API)."""
    df = pd.read_csv(path, parse_dates=["date"])
    ren = {"PRECTOTCORR": "rain_mm", "T2M_MAX": "tmax_c", "T2M_MIN": "tmin_c"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    if "et0_mm" not in df:
        df["et0_mm"] = et0_hargreaves(
            df.tmax_c.to_numpy(), df.tmin_c.to_numpy(),
            cfg["site"]["lat"], df.doy.to_numpy())
    df["weather_source"] = "nasa_power"
    return df
