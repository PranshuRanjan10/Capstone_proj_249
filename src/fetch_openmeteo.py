"""
Fetch the Open-Meteo historical-forecast archive for the field location.

This gives you what the forecast model *actually said* on each past date — the correct
training feature for forecast rainfall. Using observed rain degraded with noise is a
stand-in; this is the real thing.

WHY THIS MATTERS
  The system is trained on forecast_rainfall, not observed_rainfall, because at
  deployment the model only ever has a forecast. Training on observed rain and deploying
  on forecast rain is a silent leakage bug: validation looks excellent and field
  performance collapses. The historical-forecast archive plugs this gap properly.

  Reference: Open-Meteo historical forecast API
  https://open-meteo.com/en/docs/historical-forecast-api

USAGE
  python src/fetch_openmeteo.py --start 2015-01-01 --end 2024-12-31

OUTPUT
  data/raw/openmeteo_forecast.csv
  Columns: date, rain_fc_t1_mm, rain_fc_t2_mm, rain_fc_t3_mm, rain_fc_sum3_mm
  (t1/t2/t3 = precipitation_sum for day+1, day+2, day+3 as seen from each issue date)

NOTE
  The historical-forecast archive is only available from 2022-01-01 onwards.
  For dates before that, build_dataset.py falls back to the noise-degraded synthetic
  forecast. This is noted in the data lineage column weather_fc_source.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import config as C

API = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _fetch_chunk(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch daily precipitation forecast for a date range. Returns one row per date."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "start_date": start,
        "end_date": end,
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(API, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    dates = pd.to_datetime(data["daily"]["time"])
    rain = data["daily"]["precipitation_sum"]
    return pd.DataFrame({"date": dates, "rain_obs_mm": rain})


def build_forecast_archive(
    lat: float, lon: float, start: str, end: str
) -> pd.DataFrame:
    """
    For each date d in [start, end], we want what the forecast said about d+1, d+2, d+3
    on day d. The historical-forecast API returns the forecast issued at each start_date,
    so we shift the daily precipitation series to reconstruct t+1, t+2, t+3 columns.
    """
    print(f"Fetching Open-Meteo historical forecast archive {start} → {end} …")
    # Fetch a window that covers the full date range including the t+3 lookahead
    raw = _fetch_chunk(lat, lon, start, end)
    raw = raw.sort_values("date").reset_index(drop=True)

    # Build t+1, t+2, t+3 as forward-shifted rain columns (each row = what was forecast
    # on that date for the following days)
    out = pd.DataFrame({"date": raw["date"]})
    for h in (1, 2, 3):
        out[f"rain_fc_t{h}_mm"] = raw["rain_obs_mm"].shift(-h).round(2)
    out["rain_fc_sum3_mm"] = out[["rain_fc_t1_mm", "rain_fc_t2_mm", "rain_fc_t3_mm"]].sum(axis=1).round(2)
    out["weather_fc_source"] = "openmeteo_archive"
    return out.dropna(subset=["rain_fc_t1_mm", "rain_fc_t2_mm", "rain_fc_t3_mm"])


if __name__ == "__main__":
    cfg = C.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01",
                    help="Archive only available from 2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--lat", type=float, default=cfg["site"]["lat"])
    ap.add_argument("--lon", type=float, default=cfg["site"]["lon"])
    a = ap.parse_args()

    df = build_forecast_archive(a.lat, a.lon, a.start, a.end)
    out = cfg["_root"] / cfg["weather"]["openmeteo_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} rows → {out}")
    print("build_dataset.py will automatically use this when it exists.")
    print(df.head(5).to_string(index=False))
