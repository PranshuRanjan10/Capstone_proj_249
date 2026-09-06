"""
Virtual ESP32 publisher.

Writes to exactly the same interface the real hardware will use, so swapping the
simulator for the board changes nothing downstream. Supports injected faults --
this is what you break live in front of the panel.

  python iot/sensor_sim.py --interval 2 --fault stuck
"""
from __future__ import annotations
import argparse, json, sqlite3, sys, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C

DB = ROOT / "data" / "field.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts TEXT PRIMARY KEY,
    device_id TEXT,
    water_level_mm REAL,
    soil_moisture_pct REAL,
    air_temp_c REAL,
    humidity_pct REAL,
    battery_v REAL,
    fault_flag TEXT
);
"""
# Firebase Realtime Database equivalent of the same schema:
#   /fields/{field_id}/readings/{iso_ts} = { water_level_mm, soil_moisture_pct,
#       air_temp_c, humidity_pct, battery_v, fault_flag }
#   /fields/{field_id}/latest          = <same object>
#   /fields/{field_id}/decisions/{ts}  = { action, reason, gated, interval_width_mm }


def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute(SCHEMA)
    return con


def publish(con, rec: dict):
    con.execute("INSERT OR REPLACE INTO readings VALUES (:ts,:device_id,:water_level_mm,"
                ":soil_moisture_pct,:air_temp_c,:humidity_pct,:battery_v,:fault_flag)", rec)
    con.commit()


def replay(season_id=None, interval=2.0, fault="none", device="esp32-paddy-01"):
    cfg = C.load()
    df = pd.read_csv(ROOT / "data" / "processed" / "field_state.csv")
    season_id = season_id or df.season_id.iloc[-1]
    S = df[df.season_id == season_id].reset_index(drop=True)
    con, rng = connect(), np.random.default_rng(7)
    print(f"replaying {season_id} ({len(S)} days), fault={fault} -> {DB}")

    for i, r in S.iterrows():
        wl, sm, flag = r.wl_obs_mm, r.sm_obs_pct, "ok"
        if fault == "stuck" and i > len(S) * 0.6:
            wl, sm, flag = S.wl_obs_mm[int(len(S) * 0.6)], S.sm_obs_pct[int(len(S) * 0.6)], "stuck"
        elif fault == "drift" and i > len(S) * 0.5:
            wl, flag = wl - 1.8 * (i - len(S) * 0.5), "drift"
        elif fault == "noise" and i > len(S) * 0.5:
            wl, flag = wl + rng.normal(0, 45), "noisy"

        rec = {"ts": pd.Timestamp(r.date).isoformat(), "device_id": device,
               "water_level_mm": round(float(wl), 2),
               "soil_moisture_pct": round(float(sm), 2),
               "air_temp_c": round(float((r.tmax_c + r.tmin_c) / 2), 2),
               "humidity_pct": round(float(70 + 15 * (r.rain_mm > 0)), 1),
               "battery_v": round(float(4.05 - 0.0008 * i), 3),
               "fault_flag": flag}
        publish(con, rec)
        print(json.dumps(rec))
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--fault", choices=["none", "stuck", "drift", "noise"], default="none")
    a = ap.parse_args()
    replay(a.season, a.interval, a.fault)
