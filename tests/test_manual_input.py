"""Unit tests for manual-input feature derivation.

These guard the property that matters for the live demo: whatever a user types,
the derived row must be a complete, physically coherent, in-range 29-feature
vector that the trained models can consume.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C                                    # noqa: E402
from features import FEATURES                         # noqa: E402
from manual_input import (                            # noqa: E402
    FieldInput, derive_row, feature_frame, PRESETS, _soil_moisture,
)

CFG = C.load()


# ── completeness ──────────────────────────────────────────────────────────────
def test_produces_every_feature():
    row, _, _ = derive_row(FieldInput(), CFG)
    for f in FEATURES:
        assert f in row.index, f"missing feature {f}"
    assert len(FEATURES) == 29


def test_no_nan_or_inf():
    row, _, _ = derive_row(FieldInput(), CFG)
    vals = row[FEATURES].astype(float).to_numpy()
    assert np.isfinite(vals).all()


def test_stage_is_present_for_decision_engine():
    row, ctx, _ = derive_row(FieldInput(), CFG)
    assert isinstance(row["stage"], str)
    assert row["stage"] in [s["name"] for s in CFG["phenology"]["stages"]]
    assert ctx["stage"] == row["stage"]


# ── identities ────────────────────────────────────────────────────────────────
def test_headroom_identity():
    inp = FieldInput(wl_now_mm=37.0)
    row, _, _ = derive_row(inp, CFG)
    bund = CFG["hydrology"]["bund_height_mm"]
    assert row["headroom_mm"] == pytest.approx(bund - 37.0)


def test_forecast_sum_identity():
    inp = FieldInput(rain_fc_t1_mm=5, rain_fc_t2_mm=3, rain_fc_t3_mm=2)
    row, _, _ = derive_row(inp, CFG)
    assert row["rain_fc_sum3_mm"] == pytest.approx(10.0)


def test_et_demand_is_et0_times_kc():
    row, ctx, _ = derive_row(FieldInput(), CFG)
    assert row["et_demand"] == pytest.approx(row["et0_mm"] * row["kc"], rel=1e-6)
    assert ctx["kc"] == pytest.approx(round(row["kc"], 2))


# ── lag reconstruction ────────────────────────────────────────────────────────
def test_falling_trend_makes_past_levels_higher():
    row, _, _ = derive_row(FieldInput(wl_now_mm=0.0, wl_trend_mm_day=-5.0), CFG)
    assert row["wl_lag1"] > row["wl_obs_mm"]
    assert row["wl_lag3"] > row["wl_lag1"]
    assert row["wl_delta1"] < 0 and row["wl_delta3"] < 0


def test_rising_trend_makes_past_levels_lower():
    row, _, _ = derive_row(FieldInput(wl_now_mm=0.0, wl_trend_mm_day=5.0), CFG)
    assert row["wl_lag1"] < row["wl_obs_mm"]
    assert row["wl_delta1"] > 0


def test_lags_clamped_to_physical_bounds():
    bund  = CFG["hydrology"]["bund_height_mm"]
    floor = CFG["hydrology"]["wl_floor_mm"]
    for trend in (-400.0, 400.0):
        row, _, _ = derive_row(
            FieldInput(wl_now_mm=0.0, wl_trend_mm_day=trend), CFG)
        for lg in ("wl_lag1", "wl_lag2", "wl_lag3", "wl_lag7"):
            assert floor <= row[lg] <= bund, f"{lg} out of bounds at trend {trend}"


# ── soil moisture physics ─────────────────────────────────────────────────────
def test_ponded_field_is_saturated():
    assert _soil_moisture(50.0, -300.0) == pytest.approx(45.0)
    assert _soil_moisture(0.0, -300.0) == pytest.approx(45.0)


def test_moisture_declines_below_surface():
    assert _soil_moisture(-300.0, -300.0) == pytest.approx(24.0)
    assert 24.0 < _soil_moisture(-150.0, -300.0) < 45.0


def test_probe_override_is_respected():
    row, _, _ = derive_row(FieldInput(wl_now_mm=-200, sm_override_pct=38.5), CFG)
    assert row["sm_obs_pct"] == pytest.approx(38.5)


# ── rainfall consistency ──────────────────────────────────────────────────────
def test_rolling_rain_is_monotone_even_if_user_is_inconsistent():
    # user types a 3-day total smaller than today's rain, and a 7-day smaller still
    row, _, _ = derive_row(
        FieldInput(rain_today_mm=30, rain_3d_mm=5, rain_7d_mm=1), CFG)
    assert row["rain_sum3"] >= row["rain_mm"]
    assert row["rain_sum7"] >= row["rain_sum3"]


def test_dry_days_zero_when_raining():
    row, _, _ = derive_row(FieldInput(rain_today_mm=12), CFG)
    assert row["dry_days"] == 0


# ── phenology ─────────────────────────────────────────────────────────────────
def test_stage_advances_with_days_after_transplanting():
    seen = []
    for das in (5, 30, 55, 75, 100):
        _, ctx, _ = derive_row(FieldInput(season_name="Kuruvai", das=das), CFG)
        seen.append(ctx["stage_idx"])
    assert seen == sorted(seen), f"stage went backwards: {seen}"
    assert seen[0] < seen[-1]


def test_flowering_is_flagged_no_drain():
    for das in range(40, 110):
        _, ctx, _ = derive_row(FieldInput(season_name="Samba", das=das), CFG)
        if ctx["stage"] in CFG["phenology"]["no_drain_stages"]:
            assert ctx["drain_forbidden"] is True
            assert ctx["stage_floor_mm"] == 0.0
            break
    else:
        pytest.fail("no no-drain stage reached between das 40 and 110")


def test_gdd_fraction_within_unit_range():
    for das in (0, 60, 200):
        row, ctx, _ = derive_row(FieldInput(das=das), CFG)
        assert 0.0 <= ctx["gdd_fraction"] <= 3.0
        assert row["cum_gdd"] >= 0


# ── presets ───────────────────────────────────────────────────────────────────
def test_every_preset_derives_cleanly():
    for name, kw in PRESETS.items():
        kw = {k: v for k, v in kw.items() if not k.startswith("_")}
        row, ctx, notes = derive_row(FieldInput(**kw), CFG)
        vals = row[FEATURES].astype(float).to_numpy()
        assert np.isfinite(vals).all(), f"{name} produced non-finite features"
        assert notes, f"{name} produced no derivation notes"


def test_feature_frame_shape_and_order():
    rows = [derive_row(FieldInput(das=d), CFG)[0] for d in (20, 50, 80)]
    X = feature_frame(rows)
    assert X.shape == (3, 29)
    assert list(X.columns) == FEATURES


# ── robustness against extreme user entry ─────────────────────────────────────
@pytest.mark.parametrize("wl", [-300.0, -150.0, 0.0, 50.0, 100.0])
@pytest.mark.parametrize("trend", [-30.0, 0.0, 30.0])
def test_extreme_inputs_stay_finite(wl, trend):
    row, _, _ = derive_row(
        FieldInput(wl_now_mm=wl, wl_trend_mm_day=trend, das=1), CFG)
    assert np.isfinite(row[FEATURES].astype(float).to_numpy()).all()
