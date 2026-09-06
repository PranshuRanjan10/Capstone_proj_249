"""
Decision layer: a transparent, NON-LEARNED policy function.

This is deliberately not a model. The ML forecasts water level; this maps a forecast
plus a crop stage onto an action. Keeping it a pure function means it is auditable,
unit-testable, and impossible to accuse of circularity.

Evaluation order is fixed and matters — the order is enforced by the structure of the
code, not by comments alone:

  1. HARD AGRONOMIC CONSTRAINT  (drainage forbidden PI → flowering, NO exceptions)
  2. UNCERTAINTY GATE           (interval too wide → INSPECT, refuse to actuate)
  3. THRESHOLDS                 (safe-AWD drawdown limits, stage-dependent)
  4. BUND-FULL DRAIN            (field near overflow and forecast stays high)
  5. Otherwise MAINTAIN

BUG NOTE (fixed here): the original code checked the uncertainty gate BEFORE the hard
agronomic constraint. That means a perfectly-certain forecast during flowering could
issue DRAIN. The constraint must be evaluated first, unconditionally.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field

ACTIONS = ("MAINTAIN", "IRRIGATE", "DELAY", "DRAIN", "INSPECT")
ACTUATIONS = ("IRRIGATE", "DRAIN")

# Actions that must never fire during no-drain stages
_DRAIN_ACTIONS = {"DRAIN"}


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    gated: bool = False
    constraint_applied: bool = False
    interval_width_mm: float = 0.0
    # which rule fired (for audit log / SHAP integration)
    rule: str = ""

    def as_dict(self):
        return asdict(self)


def stage_floor(stage: str, cfg) -> float:
    """Return the AWD lower water-level limit for this stage (mm, rel. surface)."""
    return cfg["awd"]["stage_floor_mm"].get(stage, cfg["awd"]["safe_threshold_mm"])


def decide(
    wl_now: float,
    stage: str,
    forecast: dict,
    rain_fc_sum3_mm: float,
    cfg,
    gate_mm: float | None = None,
    apply_gate: bool = True,
    conformal_thresholds: dict | None = None,
) -> Decision:
    """
    Parameters
    ----------
    wl_now          : current observed water level (mm, relative to soil surface)
    stage           : crop stage name string (must match config phenology.stages names)
    forecast        : {horizon: {"q05": .., "q50": .., "q95": ..}}  — mm rel. surface
    rain_fc_sum3_mm : 3-day ahead forecast rainfall sum (mm) — used for DELAY logic
    cfg             : loaded config dict
    gate_mm         : override gate width; None → use config value
    apply_gate      : set False for ablation / reference-policy computation
    conformal_thresholds : dict mapping stage_name → Q̂ (conformal correction, mm).
                     When provided, the uncertainty gate uses per-stage CQR-adjusted
                     interval widths instead of the fixed gate_mm threshold.
                     Keys: stage names + "_global" for fallback.

    Returns
    -------
    Decision dataclass (immutable, JSON-serialisable via .as_dict())
    """
    gate_mm = float(cfg["model"]["gate_interval_width_mm"]) if gate_mm is None else float(gate_mm)
    bund = float(cfg["hydrology"]["bund_height_mm"])
    floor = stage_floor(stage, cfg)
    margin = float(cfg["awd"].get("warning_margin_mm", 15.0))

    # Derived quantities from the forecast bundle
    interval_width = max(f["q95"] - f["q05"] for f in forecast.values())
    worst_median   = min(f["q50"] for f in forecast.values())   # driest median point
    worst_lo       = min(f["q05"] for f in forecast.values())   # pessimistic bound
    best_hi        = max(f["q95"] for f in forecast.values())   # optimistic bound

    drain_forbidden = stage in cfg["phenology"]["no_drain_stages"]

    # -----------------------------------------------------------------------
    # STEP 1 — HARD AGRONOMIC CONSTRAINT
    # Must be evaluated first, unconditionally. No model output can override agronomy.
    # -----------------------------------------------------------------------
    if drain_forbidden:
        # We may still need to irrigate even during no-drain stages if the field is
        # drying out — the constraint only forbids draining, not irrigating.
        if worst_median <= floor + margin or worst_lo <= floor:
            deficit = floor - worst_median
            if rain_fc_sum3_mm >= deficit + 10 and worst_lo > floor - 40:
                return Decision(
                    "DELAY",
                    f"Level forecast to reach {worst_median:.0f} mm (stage floor {floor:.0f} mm) "
                    f"but {rain_fc_sum3_mm:.0f} mm rain is forecast. Hold irrigation.",
                    constraint_applied=True,
                    interval_width_mm=interval_width,
                    rule="agronomic_delay",
                )
            return Decision(
                "IRRIGATE",
                f"Forecast level {worst_median:.0f} mm breaches the {stage} floor "
                f"({floor:.0f} mm). Stage is water-stress critical — irrigation required.",
                constraint_applied=True,
                interval_width_mm=interval_width,
                rule="agronomic_irrigate",
            )
        # Field is fine: honour no-drain even if we are near bund height
        if wl_now >= 0.85 * bund:
            return Decision(
                "MAINTAIN",
                f"Field is near bund height but drainage is forbidden during {stage} "
                f"(panicle initiation → flowering). Holding water.",
                constraint_applied=True,
                interval_width_mm=interval_width,
                rule="agronomic_hold",
            )

    # -----------------------------------------------------------------------
    # STEP 2 — UNCERTAINTY GATE
    # Sensor drift, stuck readings, and unusual weather all widen the interval.
    # One gate catches failure modes you would otherwise enumerate individually.
    # Only applies when apply_gate=True (set False for reference/ablation policy).
    #
    # NOVELTY (CQR): When conformal_thresholds are provided, the gate uses
    # per-stage conformal-adjusted interval width. The conformal correction Q̂
    # widens the raw quantile interval to guarantee (1-α) coverage, and the
    # per-stage Q̂_s automatically tightens during yield-critical stages.
    # -----------------------------------------------------------------------
    if apply_gate:
        if conformal_thresholds is not None:
            # CQR mode: compute conformal-adjusted interval width
            q_hat = conformal_thresholds.get(
                stage, conformal_thresholds.get("_global", 0.0)
            )
            # The conformal-adjusted interval is [q05 - Q̂, q95 + Q̂]
            # so the adjusted width = raw_width + 2 * Q̂
            cqr_width = interval_width + 2 * q_hat
            if cqr_width > gate_mm:
                return Decision(
                    "INSPECT",
                    f"Conformal-adjusted prediction uncertainty exceeds the safety "
                    f"threshold ({cqr_width:.0f} mm > {gate_mm:.0f} mm, "
                    f"stage Q̂={q_hat:.1f} mm). Automation blocked — "
                    f"manual inspection required.",
                    gated=True,
                    interval_width_mm=cqr_width,
                    rule="conformal_gate",
                )
        elif interval_width > gate_mm:
            return Decision(
                "INSPECT",
                f"Prediction uncertainty exceeds the safety threshold "
                f"({interval_width:.0f} mm > {gate_mm:.0f} mm). Automation blocked — "
                f"manual inspection required.",
                gated=True,
                interval_width_mm=interval_width,
                rule="uncertainty_gate",
            )

    # -----------------------------------------------------------------------
    # STEP 3 — THRESHOLD: approaching or past the stage floor
    # Act on *approach*, not after breach (crop is already stressed by then).
    # -----------------------------------------------------------------------
    if worst_median <= floor + margin or worst_lo <= floor:
        deficit = floor - worst_median
        if rain_fc_sum3_mm >= deficit + 10 and worst_lo > floor - 40:
            return Decision(
                "DELAY",
                f"Level would reach {worst_median:.0f} mm (stage floor {floor:.0f} mm), "
                f"but {rain_fc_sum3_mm:.0f} mm rain is forecast. Hold irrigation.",
                interval_width_mm=interval_width,
                rule="threshold_delay",
            )
        return Decision(
            "IRRIGATE",
            f"Forecast level {worst_median:.0f} mm breaches the {stage} floor "
            f"of {floor:.0f} mm within 3 days.",
            interval_width_mm=interval_width,
            rule="threshold_irrigate",
        )

    # -----------------------------------------------------------------------
    # STEP 4 — BUND FULL: field near overflow, forecast stays well above floor
    # Only drain when drain is not agronomically forbidden (checked above).
    # -----------------------------------------------------------------------
    if wl_now >= 0.85 * bund and worst_median > floor + 40:
        return Decision(
            "DRAIN",
            f"Level {wl_now:.0f} mm near bund height ({bund:.0f} mm) and forecast "
            f"stays above floor. Starting an AWD dry cycle.",
            interval_width_mm=interval_width,
            rule="bund_drain",
        )

    # -----------------------------------------------------------------------
    # STEP 5 — MAINTAIN
    # -----------------------------------------------------------------------
    return Decision(
        "MAINTAIN",
        f"Forecast ({worst_median:.0f}–{best_hi:.0f} mm) stays between the "
        f"{stage} floor ({floor:.0f} mm) and bund height. No action needed.",
        interval_width_mm=interval_width,
        rule="maintain",
    )


def reference_action(
    wl_now: float,
    stage: str,
    observed_future: dict,
    rain_actual_sum3_mm: float,
    cfg,
) -> str:
    """
    The same policy applied to the OBSERVED future — perfect foresight, no gate.
    This is the yardstick that makes 'false actuation rate' a real, measurable quantity
    rather than an assertion.

    observed_future : {horizon: water_level_mm}  — single point values, no uncertainty
    """
    fc = {h: {"q05": v, "q50": v, "q95": v} for h, v in observed_future.items()}
    return decide(wl_now, stage, fc, rain_actual_sum3_mm, cfg, apply_gate=False).action
