"""Unit tests for the decision engine. Run: python -m pytest tests -q"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
from decision import decide, stage_floor

CFG = C.load()


def fc(q50, width=10.0):
    return {h: {"q05": q50 - width / 2, "q50": q50, "q95": q50 + width / 2} for h in (1, 2, 3)}


def test_no_drain_during_flowering():
    d = decide(wl_now=95, stage="flowering", forecast=fc(90), rain_fc_sum3_mm=0, cfg=CFG)
    assert d.action != "DRAIN"
    assert d.constraint_applied


def test_drain_allowed_during_tillering():
    d = decide(wl_now=95, stage="tillering", forecast=fc(80), rain_fc_sum3_mm=0, cfg=CFG)
    assert d.action == "DRAIN"


def test_gate_blocks_actuation_when_uncertain():
    wide = CFG["model"]["gate_interval_width_mm"] + 50
    d = decide(wl_now=-140, stage="tillering", forecast=fc(-200, width=wide),
               rain_fc_sum3_mm=0, cfg=CFG)
    assert d.action == "INSPECT" and d.gated


def test_irrigate_when_floor_breached():
    d = decide(wl_now=-140, stage="tillering", forecast=fc(-190), rain_fc_sum3_mm=0, cfg=CFG)
    assert d.action == "IRRIGATE"


def test_delay_when_rain_will_refill():
    d = decide(wl_now=-140, stage="tillering", forecast=fc(-165), rain_fc_sum3_mm=60, cfg=CFG)
    assert d.action == "DELAY"


def test_reproductive_floors_are_zero():
    assert stage_floor("panicle_initiation", CFG) == 0.0
    assert stage_floor("flowering", CFG) == 0.0


def test_gate_can_be_disabled_for_ablation():
    wide = CFG["model"]["gate_interval_width_mm"] + 50
    d = decide(wl_now=-140, stage="tillering", forecast=fc(-200, width=wide),
               rain_fc_sum3_mm=0, cfg=CFG, apply_gate=False)
    assert d.action == "IRRIGATE" and not d.gated


def test_conformal_gate_uses_stage_threshold():
    """When conformal_thresholds are provided, the gate should use per-stage Q̂."""
    # Create a forecast with moderate raw interval (width=60mm, below the 90mm gate)
    # but the conformal correction for this stage is large enough to push it over
    conformal_thresholds = {
        "_global": 5.0,
        "tillering": 20.0,  # Q̂ = 20mm → adjusted width = 60 + 2*20 = 100mm > 90mm gate
    }
    d = decide(wl_now=0, stage="tillering", forecast=fc(0, width=60),
               rain_fc_sum3_mm=0, cfg=CFG,
               conformal_thresholds=conformal_thresholds)
    assert d.action == "INSPECT" and d.gated
    assert d.rule == "conformal_gate"


def test_conformal_gate_disabled_falls_back_to_fixed():
    """Without conformal_thresholds, the gate uses the fixed gate_mm threshold."""
    # Same forecast as above but without conformal thresholds
    # Width=60mm < 90mm gate → should NOT fire INSPECT
    d = decide(wl_now=0, stage="tillering", forecast=fc(0, width=60),
               rain_fc_sum3_mm=0, cfg=CFG,
               conformal_thresholds=None)
    assert d.action != "INSPECT"
    assert not d.gated

