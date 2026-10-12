"""
Scene state for the Field 3D component.

A pure function: it takes what app.py has already computed (the feature row,
the forecast, the decision) and flattens it into a small JSON-safe dict. The
3D scene renders this dict and nothing else, so it can never show a number
the rest of the dashboard does not.
"""
from __future__ import annotations

from typing import Any, Mapping

SCENE_KEYS = (
    "wl_sensor_mm", "wl_actual_mm", "stage", "stage_idx", "gdd_fraction", "das",
    "season", "stages", "forecast", "action", "gated", "constraint_applied",
    "rule", "interval_width_mm", "gate_mm", "bund_mm", "floor_mm", "no_drain",
    "rain_today_mm", "rain_fc_sum3_mm", "sm_pct", "fault",
)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if v != v else v                   # NaN → default


def build_scene_state(row: Mapping, *, wl_pre_fault: float,
                      forecast: Mapping[int, Mapping[str, float]], decision,
                      gate_mm: float, cfg: Mapping, season: str,
                      fault: str) -> dict:
    """Flatten the dashboard's current state into the component's data contract.

    `row` is the feature row *after* fault injection (what the sensor reports);
    `wl_pre_fault` is the water level before it (what is really in the field).
    """
    stage = str(row["stage"])
    stages = [s["name"] for s in cfg["phenology"]["stages"]]
    floor = cfg["awd"]["stage_floor_mm"].get(stage, cfg["awd"]["safe_threshold_mm"])

    return {
        "wl_sensor_mm":      _f(row["wl_obs_mm"]),
        "wl_actual_mm":      _f(wl_pre_fault),
        "stage":             stage,
        "stage_idx":         stages.index(stage) if stage in stages else 0,
        "gdd_fraction":      min(max(_f(row.get("gdd_fraction")), 0.0), 1.0),
        "das":               int(_f(row.get("das"))),
        "season":            str(season),
        "stages":            [{"name": s["name"], "start": _f(s["f_start"]),
                               "end": min(_f(s["f_end"]), 1.0)}
                              for s in cfg["phenology"]["stages"]],
        "forecast":          {str(int(h)): {q: _f(v[q]) for q in ("q05", "q50", "q95")}
                              for h, v in sorted(forecast.items())},
        "action":            str(decision.action),
        "gated":             bool(decision.gated),
        "constraint_applied": bool(decision.constraint_applied),
        "rule":              str(decision.rule),
        "interval_width_mm": _f(decision.interval_width_mm),
        "gate_mm":           _f(gate_mm),
        "bund_mm":           _f(cfg["hydrology"]["bund_height_mm"]),
        "floor_mm":          _f(floor),
        "no_drain":          stage in cfg["phenology"]["no_drain_stages"],
        "rain_today_mm":     _f(row.get("rain_mm")),
        "rain_fc_sum3_mm":   _f(row.get("rain_fc_sum3_mm")),
        "sm_pct":            _f(row.get("sm_obs_pct")),
        "fault":             str(fault),
    }
