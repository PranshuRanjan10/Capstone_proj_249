"""
Fetch daily weather from NASA POWER. RUN THIS ON YOUR OWN MACHINE.

power.larc.nasa.gov is outside the sandbox allowlist, so this script is written for
you to execute locally. It writes data/raw/power_daily.csv, after which you set
weather.source: "power" in config.yaml and re-run build_dataset.py. Nothing else changes.

  python src/fetch_power.py --start 2015-01-01 --end 2024-12-31

Free, no API key, no registration. Note in your limitations slide that POWER is
satellite/reanalysis-derived at roughly 0.5 degree resolution, so it is regionally
accurate but not point-accurate -- rainfall carries the largest local error.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd, requests

sys.path.insert(0, str(Path(__file__).parent))
import config as C

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"]


def fetch(lat, lon, start, end):
    r = requests.get(URL, params={
        "parameters": ",".join(PARAMS), "community": "AG",
        "latitude": lat, "longitude": lon,
        "start": start.replace("-", ""), "end": end.replace("-", ""),
        "format": "JSON"}, timeout=120)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    df = pd.DataFrame(p)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.reset_index(names="date").replace(-999.0, pd.NA)
    return df


if __name__ == "__main__":
    cfg = C.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--lat", type=float, default=cfg["site"]["lat"])
    ap.add_argument("--lon", type=float, default=cfg["site"]["lon"])
    a = ap.parse_args()

    df = fetch(a.lat, a.lon, a.start, a.end)
    out = cfg["_root"] / "data" / "raw" / "power_daily.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} days -> {out}")
    print("Now set  weather.source: \"power\"  in config/config.yaml and re-run build_dataset.py")
