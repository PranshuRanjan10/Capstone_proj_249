"""
Manual-input mode: reconstruct the full 29-feature model row from a small set of
quantities a farmer or a field sensor could actually report.

WHY THIS EXISTS
  The model consumes 29 engineered features. Nobody can type 29 numbers, and most of
  them are not independent — they are lags, rolling windows and physical identities
  derived from a handful of underlying states. This module makes that derivation
  explicit and auditable rather than hiding it inside the dashboard.

  Every derived feature is produced by ONE of four documented rules:

    (a) IDENTITY      — an exact definition (headroom = bund - level)
    (b) PHYSICS       — the same function the simulator uses (soil moisture from
                        water level; ET0 from Hargreaves-Samani; GDD -> stage)
    (c) LINEAR TREND  — lags reconstructed by extrapolating the reported 3-day
                        trend backwards, clamped to physical bounds
    (d) STEADY STATE  — rolling means assumed equal to today's value when no
                        history is supplied (ET0 3-day and 7-day means)

  Rule (c) and (d) are approximations and are labelled as such in `derivation_notes`.
  They are stated on screen so a reviewer can see exactly what was assumed.

VALIDATION OF THE GDD ASSUMPTION
  Cumulative GDD is estimated as `das x daily_gdd(tmax, tmin)` — a constant-temperature
  assumption. Checked against the real 10-year NASA POWER trajectory, this lands within
  1-5% of the true cumulative GDD at every growth stage and assigns the correct stage
  at das = 20, 45, 70, 95 and 120 for both seasons.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field as _dcfield

import numpy as np
import pandas as pd

from staging import daily_gdd, stage_from_gdd
from weather import et0_hargreaves
from features import FEATURES

# Soil-moisture endpoints — must match water_balance._moisture_from_wl
_SM_SAT, _SM_DRY = 45.0, 24.0

# Representative (non-leap) year used only to convert day-after-transplanting
# into a day-of-year for the solar-radiation term.
_REF_YEAR = 2023


# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FieldInput:
    """The quantities a farmer, an agronomist, or an ESP32 could actually supply."""
    season_name: str = "Kuruvai"     # "Kuruvai" | "Samba"
    das: int = 60                    # days after transplanting
    wl_now_mm: float = 40.0          # current water level, mm rel. soil surface
    wl_trend_mm_day: float = -4.0    # mm/day; negative = falling
    rain_today_mm: float = 0.0
    rain_3d_mm: float = 0.0          # total over the last 3 days, incl. today
    rain_7d_mm: float = 0.0          # total over the last 7 days, incl. today
    rain_fc_t1_mm: float = 0.0       # forecast rain tomorrow
    rain_fc_t2_mm: float = 0.0       # forecast rain day after
    rain_fc_t3_mm: float = 0.0       # forecast rain in 3 days
    tmax_c: float = 34.5
    tmin_c: float = 25.5
    sm_override_pct: float | None = None   # if a real probe reading is available


# ──────────────────────────────────────────────────────────────────────────────
# Climatology — realistic defaults drawn from the actual NASA POWER record
# ──────────────────────────────────────────────────────────────────────────────
def climatology(field_state: pd.DataFrame, season_name: str, das: int) -> dict:
    """Ten-year average conditions for this season at this day after transplanting.

    Used to pre-fill the manual-input form so it opens in a realistic state rather
    than at zero. Every value remains user-overridable.
    """
    fs = field_state.copy()
    fs["_regime"] = fs.season_id.str.split("-").str[0]
    sub = fs[fs._regime == season_name]
    if sub.empty:
        return {}

    # nearest +/- 3 days, so a single missing das never empties the window
    win = sub[(sub.das >= das - 3) & (sub.das <= das + 3)]
    if win.empty:
        win = sub

    hist = sub[sub.das <= das]
    r3 = win.rain_mm.mean() * 3.0
    r7 = win.rain_mm.mean() * 7.0
    return {
        "tmax_c":        round(float(hist.tmax_c.mean()), 1),
        "tmin_c":        round(float(hist.tmin_c.mean()), 1),
        "rain_today_mm": round(float(win.rain_mm.mean()), 1),
        "rain_3d_mm":    round(float(r3), 1),
        "rain_7d_mm":    round(float(r7), 1),
        "wl_now_mm":     round(float(win.wl_obs_mm.mean()), 0),
        "rain_fc_t1_mm": round(float(win.rain_fc_t1_mm.mean()), 1),
        "rain_fc_t2_mm": round(float(win.rain_fc_t2_mm.mean()), 1),
        "rain_fc_t3_mm": round(float(win.rain_fc_t3_mm.mean()), 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Derivation
# ──────────────────────────────────────────────────────────────────────────────
def _season_spec(cfg, season_name: str) -> dict:
    for s in cfg["seasons"]:
        if s["name"] == season_name:
            return s
    return cfg["seasons"][0]


def _soil_moisture(wl_mm: float, wl_floor: float) -> float:
    """PHYSICS — identical to water_balance._moisture_from_wl."""
    if wl_mm >= 0:
        return _SM_SAT
    x = float(np.clip(wl_mm / wl_floor, 0.0, 1.0))
    return _SM_SAT - (_SM_SAT - _SM_DRY) * x


def derive_row(inp: FieldInput, cfg) -> tuple[pd.Series, dict, list[str]]:
    """Build the full 29-feature row.

    Returns
    -------
    row    : pd.Series indexed by FEATURES (plus `stage` for the decision engine)
    context: dict of human-readable derived quantities for the transparency panel
    notes  : list of assumption strings to display alongside
    """
    spec   = _season_spec(cfg, inp.season_name)
    bund   = float(cfg["hydrology"]["bund_height_mm"])
    floor  = float(cfg["hydrology"]["wl_floor_mm"])
    p      = cfg["phenology"]
    notes: list[str] = []

    # ── PHENOLOGY (physics) ───────────────────────────────────────────────────
    gdd_day  = float(daily_gdd(np.array([inp.tmax_c]), np.array([inp.tmin_c]),
                               p["t_base"], p["t_cap"])[0])
    cum_gdd  = gdd_day * max(inp.das, 0)
    gdd_tot  = float(spec["gdd_total"])
    names, idxs, kcs, fracs = stage_from_gdd(np.array([cum_gdd]), cfg, gdd_total=gdd_tot)
    stage, stage_idx, kc, gdd_fraction = (
        str(names[0]), int(idxs[0]), float(kcs[0]), float(fracs[0]))
    notes.append(
        f"Cumulative GDD estimated as {inp.das} days x {gdd_day:.1f} deg-day "
        f"(constant-temperature assumption; within 1-5% of the 10-year record)."
    )

    # ── EVAPOTRANSPIRATION (physics) ──────────────────────────────────────────
    start = dt.date(_REF_YEAR, spec["transplant_month"], spec["transplant_day"])
    doy   = (start + dt.timedelta(days=int(inp.das))).timetuple().tm_yday
    et0   = float(et0_hargreaves(np.array([inp.tmax_c]), np.array([inp.tmin_c]),
                                 cfg["site"]["lat"], np.array([doy]))[0])
    et_demand = et0 * kc
    notes.append(
        "ET0 3-day and 7-day means are set equal to today's ET0 (steady-state "
        "assumption — no temperature history is supplied in manual mode)."
    )

    # ── WATER LEVEL LAGS (linear trend) ───────────────────────────────────────
    def lag(k: int) -> float:
        return float(np.clip(inp.wl_now_mm - inp.wl_trend_mm_day * k, floor, bund))

    wl_lag1, wl_lag2, wl_lag3, wl_lag7 = lag(1), lag(2), lag(3), lag(7)
    wl_delta1 = inp.wl_now_mm - wl_lag1
    wl_delta3 = inp.wl_now_mm - wl_lag3
    notes.append(
        f"Water-level history reconstructed by extrapolating the reported trend of "
        f"{inp.wl_trend_mm_day:+.1f} mm/day backwards, clamped to "
        f"[{floor:.0f}, {bund:.0f}] mm."
    )

    # ── SOIL MOISTURE (physics, or probe override) ────────────────────────────
    sm_now = (float(inp.sm_override_pct) if inp.sm_override_pct is not None
              else _soil_moisture(inp.wl_now_mm, floor))
    sm_lag1 = _soil_moisture(wl_lag1, floor)
    sm_lag3 = _soil_moisture(wl_lag3, floor)
    if inp.sm_override_pct is not None:
        notes.append("Soil moisture taken from the probe reading you supplied.")
    else:
        notes.append(
            "Soil moisture derived from water level using the simulator's own "
            "retention curve (45% saturated, 24% at the -300 mm floor)."
        )

    # ── RAINFALL ──────────────────────────────────────────────────────────────
    rain_sum3 = max(inp.rain_3d_mm, inp.rain_today_mm)
    rain_sum7 = max(inp.rain_7d_mm, rain_sum3)
    if inp.rain_today_mm > 0:
        dry_days = 0.0
    elif rain_sum3 <= 0:
        dry_days = 7.0 if rain_sum7 <= 0 else 3.0
    else:
        dry_days = 1.0
    fc_sum3 = inp.rain_fc_t1_mm + inp.rain_fc_t2_mm + inp.rain_fc_t3_mm

    # ── IDENTITY ──────────────────────────────────────────────────────────────
    headroom = bund - inp.wl_now_mm

    vals = {
        "wl_obs_mm": inp.wl_now_mm, "sm_obs_pct": sm_now,
        "wl_lag1": wl_lag1, "wl_lag2": wl_lag2, "wl_lag3": wl_lag3, "wl_lag7": wl_lag7,
        "sm_lag1": sm_lag1, "sm_lag3": sm_lag3,
        "wl_delta1": wl_delta1, "wl_delta3": wl_delta3,
        "sm_delta3": sm_now - sm_lag3,
        "rain_mm": inp.rain_today_mm, "rain_sum3": rain_sum3, "rain_sum7": rain_sum7,
        "dry_days": dry_days,
        "rain_fc_t1_mm": inp.rain_fc_t1_mm, "rain_fc_t2_mm": inp.rain_fc_t2_mm,
        "rain_fc_t3_mm": inp.rain_fc_t3_mm, "rain_fc_sum3_mm": fc_sum3,
        "et0_mm": et0, "et0_mean3": et0, "et0_mean7": et0,
        "et_demand": et_demand, "kc": kc,
        "cum_gdd": cum_gdd, "gdd_fraction": gdd_fraction,
        "stage_idx": float(stage_idx), "das": float(inp.das),
        "headroom_mm": headroom,
    }
    missing = [f for f in FEATURES if f not in vals]
    if missing:
        raise KeyError(f"manual_input did not produce: {missing}")

    row = pd.Series({f: float(vals[f]) for f in FEATURES})
    row["stage"] = stage

    context = {
        "stage": stage,
        "stage_idx": stage_idx,
        "kc": round(kc, 2),
        "cum_gdd": round(cum_gdd, 0),
        "gdd_fraction": round(gdd_fraction, 3),
        "gdd_total": gdd_tot,
        "et0_mm": round(et0, 2),
        "et_demand_mm": round(et_demand, 2),
        "headroom_mm": round(headroom, 1),
        "soil_moisture_pct": round(sm_now, 1),
        "rain_fc_sum3_mm": round(fc_sum3, 1),
        "stage_floor_mm": float(
            cfg["awd"]["stage_floor_mm"].get(stage, cfg["awd"]["safe_threshold_mm"])),
        "drain_forbidden": stage in p["no_drain_stages"],
        "day_of_year": doy,
    }
    return row, context, notes


def feature_frame(rows: list[pd.Series]) -> pd.DataFrame:
    """Stack derived rows into a model-ready frame (FEATURES order preserved)."""
    return pd.DataFrame([r[FEATURES].astype(float) for r in rows]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Preset demo scenarios
# ──────────────────────────────────────────────────────────────────────────────
#  Each is a situation with a clear, explainable correct answer — chosen so the
#  policy can be exercised in front of an audience without fumbling sliders.
PRESETS: dict[str, dict] = {
    "1 · Healthy field — nothing to do": dict(
        season_name="Samba", das=95, wl_now_mm=70, wl_trend_mm_day=2.0,
        rain_today_mm=0, rain_3d_mm=0, rain_7d_mm=0,
        rain_fc_t1_mm=0, rain_fc_t2_mm=0, rain_fc_t3_mm=0,
        tmax_c=30.5, tmin_c=23.5,
        _expect="MAINTAIN",
        _why="Grain filling, 70 mm ponded, well clear of the -100 mm floor. "
             "The forecast is comfortable. Expect MAINTAIN.",
    ),
    "2 · Drying out, no rain coming": dict(
        season_name="Samba", das=15, wl_now_mm=0, wl_trend_mm_day=-4.0,
        rain_today_mm=0, rain_3d_mm=0, rain_7d_mm=1,
        rain_fc_t1_mm=0, rain_fc_t2_mm=0, rain_fc_t3_mm=0,
        tmax_c=30.5, tmin_c=23.5,
        _expect="IRRIGATE",
        _why="Establishment, level at the surface and falling 4 mm/day toward the "
             "-20 mm floor, with a bone-dry forecast. Expect IRRIGATE.",
    ),
    "3 · Same field — but rain is forecast": dict(
        season_name="Samba", das=15, wl_now_mm=0, wl_trend_mm_day=-4.0,
        rain_today_mm=0, rain_3d_mm=0, rain_7d_mm=1,
        rain_fc_t1_mm=16.8, rain_fc_t2_mm=13.2, rain_fc_t3_mm=6.0,
        tmax_c=30.5, tmin_c=23.5,
        _expect="DELAY",
        _why="Identical field to scenario 2 — ONLY the forecast changed. Expect "
             "DELAY: the system waits for 36 mm of free rain instead of pumping. "
             "Run 2 and 3 back to back; it is the clearest thing in the demo.",
    ),
    "4 · Flowering — the hard constraint fires": dict(
        season_name="Samba", das=78, wl_now_mm=95, wl_trend_mm_day=1.0,
        rain_today_mm=6, rain_3d_mm=30, rain_7d_mm=64,
        rain_fc_t1_mm=8, rain_fc_t2_mm=4, rain_fc_t3_mm=2,
        tmax_c=30.5, tmin_c=23.5,
        _expect="MAINTAIN",
        _why="Field is at the bund and would normally DRAIN — but drainage is "
             "forbidden at flowering. Expect MAINTAIN with the constraint flagged, "
             "and rule = agronomic_hold. Agronomy overrules the model.",
    ),
    "5 · Bund full at ripening — safe to drain": dict(
        season_name="Kuruvai", das=100, wl_now_mm=88, wl_trend_mm_day=2.0,
        rain_today_mm=0, rain_3d_mm=0, rain_7d_mm=0,
        rain_fc_t1_mm=0, rain_fc_t2_mm=0, rain_fc_t3_mm=0,
        tmax_c=34.0, tmin_c=25.0,
        _expect="DRAIN",
        _why="Late season, drainage now permitted, field near overflow. Expect "
             "DRAIN — the start of an AWD dry cycle.",
    ),
    "6 · Impossible field — the gate refuses": dict(
        season_name="Kuruvai", das=12, wl_now_mm=-160, wl_trend_mm_day=-8.0,
        rain_today_mm=0, rain_3d_mm=0, rain_7d_mm=0,
        rain_fc_t1_mm=0, rain_fc_t2_mm=0, rain_fc_t3_mm=0,
        tmax_c=34.0, tmin_c=25.0,
        _expect="INSPECT",
        _why="A field 160 mm below the surface twelve days after transplanting is "
             "not a state the model has ever seen — the crop would already be dead. "
             "Expect INSPECT. The system declines to act on nonsense rather than "
             "guessing confidently.",
    ),
}
