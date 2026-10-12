"""Unit tests for the Field 3D scene-state contract.

The 3D component renders only what build_scene_state() hands it, so these guard
that the dict is complete, JSON-safe (no numpy scalars leak through Streamlit's
serialiser) and consistent with the decision the rest of the dashboard shows.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

import config as C                                    # noqa: E402
from decision import decide                           # noqa: E402
from manual_input import FieldInput, derive_row, PRESETS  # noqa: E402
from field3d.state import SCENE_KEYS, build_scene_state  # noqa: E402

CFG = C.load()


def _forecast(wl):
    # numpy scalars on purpose — exactly what XGBRegressor.predict returns
    return {h: {"q05": np.float32(wl - 20 * h), "q50": np.float32(wl - 5 * h),
                "q95": np.float32(wl + 15 * h)} for h in (1, 2, 3)}


def _state(row, fault="none", wl_pre=None):
    fc = _forecast(float(row["wl_obs_mm"]))
    d = decide(float(row["wl_obs_mm"]), str(row["stage"]), fc,
               float(row["rain_fc_sum3_mm"]), CFG, gate_mm=90.0)
    return build_scene_state(
        row, wl_pre_fault=row["wl_obs_mm"] if wl_pre is None else wl_pre,
        forecast=fc, decision=d, gate_mm=np.float64(90.0), cfg=CFG,
        season="Kuruvai (manual)", fault=fault), d


@pytest.mark.parametrize("preset", list(PRESETS))
def test_contract_is_complete_and_json_safe(preset):
    pre = {k: v for k, v in PRESETS[preset].items() if not k.startswith("_")}
    row, _, _ = derive_row(FieldInput(**pre), CFG)
    s, d = _state(row)
    assert set(s) == set(SCENE_KEYS)
    json.dumps(s)                                      # raises on numpy types
    assert s["action"] == d.action
    assert s["gated"] == d.gated
    assert s["constraint_applied"] == d.constraint_applied


def test_stage_index_matches_config_order():
    names = [st["name"] for st in CFG["phenology"]["stages"]]
    for das in (5, 30, 60, 75, 90, 110):
        row, _, _ = derive_row(FieldInput(season_name="Kuruvai", das=das), CFG)
        s, _ = _state(row)
        assert names[s["stage_idx"]] == s["stage"]
        assert 0.0 <= s["gdd_fraction"] <= 1.0
        assert s["no_drain"] == (s["stage"] in CFG["phenology"]["no_drain_stages"])


def test_fault_keeps_sensor_and_actual_apart():
    row, _, _ = derive_row(FieldInput(wl_now_mm=30), CFG)
    row = row.copy()
    row["wl_obs_mm"] -= 80                             # drift, as app.py injects it
    s, _ = _state(row, fault="drifting sensor", wl_pre=30.0)
    assert s["wl_actual_mm"] == pytest.approx(30.0)
    assert s["wl_sensor_mm"] == pytest.approx(-50.0)


def test_forecast_keys_are_strings_in_horizon_order():
    row, _, _ = derive_row(FieldInput(), CFG)
    s, _ = _state(row)
    assert list(s["forecast"]) == ["1", "2", "3"]
    for v in s["forecast"].values():
        assert v["q05"] <= v["q50"] <= v["q95"]
