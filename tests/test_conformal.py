"""Unit tests for the CQR conformal prediction module."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conformal import (
    cqr_nonconformity_scores,
    conformal_quantile,
    per_stage_conformal,
    adjust_interval,
    conformal_interval_width,
    ConformalCalibrator,
)


# ---- Score computation ----

def test_cqr_scores_negative_when_inside():
    """If y is inside [q_lo, q_hi], the score should be negative."""
    y     = np.array([10.0, 20.0, 30.0])
    q_lo  = np.array([ 5.0, 15.0, 25.0])
    q_hi  = np.array([15.0, 25.0, 35.0])
    scores = cqr_nonconformity_scores(y, q_lo, q_hi)
    assert all(s < 0 for s in scores), f"Scores should be negative: {scores}"


def test_cqr_scores_positive_when_outside():
    """If y is outside [q_lo, q_hi], the score should be positive."""
    y     = np.array([ 0.0, 30.0])   # below q_lo and above q_hi
    q_lo  = np.array([ 5.0, 15.0])
    q_hi  = np.array([15.0, 25.0])
    scores = cqr_nonconformity_scores(y, q_lo, q_hi)
    assert all(s > 0 for s in scores), f"Scores should be positive: {scores}"


def test_cqr_scores_zero_on_boundary():
    """If y is exactly at the boundary, score should be zero."""
    y     = np.array([5.0, 15.0])
    q_lo  = np.array([5.0, 10.0])
    q_hi  = np.array([10.0, 15.0])
    scores = cqr_nonconformity_scores(y, q_lo, q_hi)
    np.testing.assert_array_equal(scores, [0.0, 0.0])


# ---- Conformal quantile ----

def test_conformal_quantile_monotone():
    """Tighter alpha (lower miscoverage) should give a larger or equal Q̂."""
    rng = np.random.default_rng(42)
    scores = rng.normal(0, 10, 200)
    q_90 = conformal_quantile(scores, alpha=0.10)   # 90% coverage
    q_95 = conformal_quantile(scores, alpha=0.05)   # 95% coverage
    assert q_95 >= q_90, f"95% Q̂ ({q_95}) should be >= 90% Q̂ ({q_90})"


def test_conformal_quantile_empty_returns_inf():
    """Empty calibration set should return infinity (maximally conservative)."""
    q = conformal_quantile(np.array([]), alpha=0.10)
    assert q == float("inf")


# ---- Per-stage conformal ----

def test_per_stage_fallback():
    """Stages with fewer than min_stage_n points should fall back to global Q̂."""
    scores = np.random.default_rng(42).normal(0, 10, 100)
    stages = np.array(["A"] * 80 + ["B"] * 20)  # B has only 20 points
    thresholds = per_stage_conformal(scores, stages, alpha=0.10, min_stage_n=30)

    assert "A" in thresholds
    assert "B" in thresholds
    assert "_global" in thresholds
    # B should fall back to global
    assert thresholds["B"] == thresholds["_global"]
    # A should have its own threshold (may differ from global)
    assert "A" in thresholds


def test_per_stage_distinct_thresholds():
    """Stages with distinct score distributions should get distinct Q̂ values.
    We use scores that are all positive (true value outside the interval)
    so the interpretation is straightforward: larger score = worse coverage."""
    rng = np.random.default_rng(42)
    # Stage A: small positive scores (model almost covers)
    scores_a = np.abs(rng.normal(0, 2, 100))
    # Stage B: large positive scores (model badly under-covers)
    scores_b = np.abs(rng.normal(0, 20, 100))
    scores = np.concatenate([scores_a, scores_b])
    stages = np.array(["A"] * 100 + ["B"] * 100)
    thresholds = per_stage_conformal(scores, stages, alpha=0.10, min_stage_n=30)
    assert thresholds["B"] > thresholds["A"], \
        f"Noisy stage B ({thresholds['B']:.1f}) should have wider Q̂ than A ({thresholds['A']:.1f})"


# ---- Interval adjustment ----

def test_adjusted_interval_wider_with_positive_qhat():
    """Positive Q̂ should widen the interval."""
    lo, hi = adjust_interval(10.0, 20.0, q_hat=5.0)
    assert lo == 5.0 and hi == 25.0


def test_adjusted_interval_narrower_with_negative_qhat():
    """Negative Q̂ (over-covering model) should narrow the interval."""
    lo, hi = adjust_interval(10.0, 20.0, q_hat=-2.0)
    assert lo == 12.0 and hi == 18.0


def test_conformal_interval_width_correct():
    """Width should be original width + 2 * Q̂."""
    w = conformal_interval_width(10.0, 20.0, q_hat=5.0)
    assert w == 20.0  # (20+5) - (10-5) = 25-5 = 20


# ---- ConformalCalibrator class ----

def test_calibrator_fit_transform():
    """End-to-end fit+transform should run and produce adjusted intervals."""
    rng = np.random.default_rng(42)
    n = 200
    # True values have std=15, but quantile predictions are much too narrow (±3).
    # This guarantees under-coverage → positive Q̂.
    y = rng.normal(50, 15, n)
    q_lo = np.full(n, 47.0)    # constant narrow band: [47, 53]
    q_hi = np.full(n, 53.0)    # only covers y within ±3 of 50
    stages = np.array(["A"] * 100 + ["B"] * 100)

    cal = ConformalCalibrator(alpha=0.10, min_stage_n=30)
    cal.fit(y, q_lo, q_hi, stages)

    # Q̂ should be positive because the model badly under-covers
    assert cal.get_q_hat("A") > 0, \
        f"Q̂ should be positive for under-covering model, got {cal.get_q_hat('A'):.2f}"

    # Transform some test data
    q_lo_test = np.array([40.0, 45.0])
    q_hi_test = np.array([60.0, 55.0])
    stages_test = np.array(["A", "B"])
    lo_adj, hi_adj = cal.transform(q_lo_test, q_hi_test, stages_test)

    # With positive Q̂, adjusted intervals should be wider
    assert all(hi_adj - lo_adj >= q_hi_test - q_lo_test - 1e-10)


def test_coverage_guarantee():
    """On held-out data, conformal coverage should be >= (1-alpha).

    This is the key theoretical property of conformal prediction:
    the coverage guarantee holds for any exchangeable data, regardless
    of the underlying distribution.
    """
    rng = np.random.default_rng(42)
    n_cal, n_test = 500, 1000
    alpha = 0.10

    # Generate exchangeable data from the same distribution
    x_all = rng.uniform(0, 10, n_cal + n_test)
    noise_all = rng.normal(0, 1 + x_all)  # heteroscedastic noise
    y_all = 3 * x_all + noise_all

    # Simulated quantile predictions (deliberately imperfect / under-covering)
    q_lo_all = 3 * x_all - 1.0 * (1 + x_all)
    q_hi_all = 3 * x_all + 1.0 * (1 + x_all)

    # Split into calibration and test
    x_cal, x_test = x_all[:n_cal], x_all[n_cal:]
    y_cal, y_test = y_all[:n_cal], y_all[n_cal:]
    q_lo_cal, q_lo_test = q_lo_all[:n_cal], q_lo_all[n_cal:]
    q_hi_cal, q_hi_test = q_hi_all[:n_cal], q_hi_all[n_cal:]

    stages_cal = np.full(n_cal, "A")
    stages_test = np.full(n_test, "A")

    cal = ConformalCalibrator(alpha=alpha, min_stage_n=30)
    cal.fit(y_cal, q_lo_cal, q_hi_cal, stages_cal)
    lo_adj, hi_adj = cal.transform(q_lo_test, q_hi_test, stages_test)

    coverage = np.mean((y_test >= lo_adj) & (y_test <= hi_adj))
    assert coverage >= (1 - alpha) - 0.03, \
        f"Coverage {coverage:.3f} should be >= {1-alpha-0.03:.3f}"


def test_thresholds_df():
    """The thresholds DataFrame should have the right shape and columns."""
    cal = ConformalCalibrator(alpha=0.10)
    cal.thresholds = {"_global": 5.0, "A": 3.0, "B": 7.0}
    df = cal.thresholds_df()
    assert set(df.columns) == {"stage", "q_hat_mm"}
    assert len(df) == 3
