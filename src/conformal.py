"""
Conformalized Quantile Regression (CQR) with per-stage adaptive calibration.

NOVELTY CONTRIBUTION: This module is the first application of split conformal
prediction with stage-conditional calibration in a paddy irrigation DSS.

WHAT IT DOES
  Standard quantile regression (q05/q95) gives prediction intervals with
  *approximate* 90% coverage. CQR adds a conformal correction that gives
  *distribution-free, finite-sample-valid* coverage — meaning the 90% guarantee
  holds for any data distribution, not just the one the model was trained on.

  Per-stage calibration goes further: instead of one global correction, we
  compute a separate correction per crop growth stage. This means the gate
  automatically tightens during yield-critical stages (panicle initiation,
  flowering) where the model is less accurate.

ALGORITHM (Romano, Patterson & Candès, NeurIPS 2019)
  1. Train quantile models q̂_α/2 and q̂_{1-α/2} on training data.
  2. On held-out calibration data, compute nonconformity scores:
       s_i = max(q̂_α/2(x_i) - y_i,  y_i - q̂_{1-α/2}(x_i))
     Positive score = y fell outside the interval.
     Negative score = y was comfortably inside.
  3. Compute Q̂ = ⌈(1-α)(1 + 1/n_cal)⌉-th quantile of {s_i}.
  4. For a new point x:
       C(x) = [q̂_α/2(x) - Q̂,  q̂_{1-α/2}(x) + Q̂]
     This interval has guaranteed ≥ (1-α) coverage.

PER-STAGE EXTENSION
  Compute Q̂_s separately for each stage s, using only calibration points
  from that stage. Fall back to global Q̂ when n_cal < min_stage_n (default 30).

REFERENCES
  Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile
    Regression. NeurIPS 2019. arXiv:1905.03222.
  Angelopoulos, A. N., & Bates, S. (2023). Conformal Prediction: A Gentle
    Introduction. Foundations and Trends in ML, 16(4), 494–591.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def cqr_nonconformity_scores(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> np.ndarray:
    """Compute CQR nonconformity scores.

    s_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))

    Positive → true value fell outside the quantile interval.
    Negative → true value was inside the interval (by that many mm).
    """
    return np.maximum(q_lo - y_true, y_true - q_hi)


def conformal_quantile(
    scores: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Compute the conformal correction Q̂.

    Q̂ = the ⌈(1-α)(1 + 1/n)⌉-th smallest value of scores.
    This is the key quantity: add Q̂ to q_hi and subtract from q_lo
    to get the conformally-adjusted interval.

    Parameters
    ----------
    scores : 1-D array of nonconformity scores from calibration set.
    alpha  : miscoverage level (0.10 → 90% target coverage).

    Returns
    -------
    Q̂ : float. The conformal correction in the same units as the scores (mm).
    """
    n = len(scores)
    if n == 0:
        return float("inf")  # no calibration data → infinitely wide (safe)
    # Romano et al. (2019): take the ⌈(n+1)(1-α)⌉-th smallest score.
    # np.quantile with level = ⌈(n+1)(1-α)⌉ / n implements this.
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(rank, n)  # clamp to n (can't exceed the number of scores)
    sorted_scores = np.sort(scores)
    return float(sorted_scores[rank - 1])  # -1 for 0-based indexing


def per_stage_conformal(
    scores: np.ndarray,
    stages: np.ndarray,
    alpha: float = 0.10,
    min_stage_n: int = 30,
) -> dict[str, float]:
    """Compute per-stage conformal corrections Q̂_s.

    For stages with fewer than `min_stage_n` calibration points, fall back
    to the global Q̂ to avoid instability from small-sample quantile estimation.

    Parameters
    ----------
    scores      : 1-D array of CQR nonconformity scores.
    stages      : 1-D array of stage names (same length as scores).
    alpha       : miscoverage level.
    min_stage_n : minimum calibration points per stage.

    Returns
    -------
    dict mapping stage_name → Q̂_s (float, in mm).
    Includes a "_global" key for the global Q̂.
    """
    global_q = conformal_quantile(scores, alpha)
    thresholds = {"_global": global_q}

    unique_stages = np.unique(stages)
    for stage in unique_stages:
        mask = stages == stage
        stage_scores = scores[mask]
        if len(stage_scores) >= min_stage_n:
            thresholds[str(stage)] = conformal_quantile(stage_scores, alpha)
        else:
            thresholds[str(stage)] = global_q  # fallback

    return thresholds


def adjust_interval(
    q_lo: np.ndarray | float,
    q_hi: np.ndarray | float,
    q_hat: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Apply conformal correction to widen (or narrow) the prediction interval.

    Adjusted interval: [q_lo - Q̂, q_hi + Q̂]

    If Q̂ is negative (rare — means the quantile model already over-covers),
    the interval actually narrows, which is correct and desirable.
    """
    return q_lo - q_hat, q_hi + q_hat


def conformal_interval_width(
    q_lo: float | np.ndarray,
    q_hi: float | np.ndarray,
    q_hat: float,
) -> float | np.ndarray:
    """Compute the width of the conformally-adjusted interval."""
    lo_adj, hi_adj = adjust_interval(q_lo, q_hi, q_hat)
    return hi_adj - lo_adj


class ConformalCalibrator:
    """Wraps CQR calibration into a fit/transform interface.

    Usage in the LOSO loop:
        cal = ConformalCalibrator(alpha=0.10, min_stage_n=30)
        cal.fit(y_cal, q05_cal, q95_cal, stages_cal)
        q05_adj, q95_adj = cal.transform(q05_test, q95_test, stages_test)
    """

    def __init__(self, alpha: float = 0.10, min_stage_n: int = 30):
        self.alpha = alpha
        self.min_stage_n = min_stage_n
        self.thresholds: dict[str, float] = {}
        self._scores: np.ndarray | None = None

    def fit(
        self,
        y_cal: np.ndarray,
        q_lo_cal: np.ndarray,
        q_hi_cal: np.ndarray,
        stages_cal: np.ndarray,
    ) -> "ConformalCalibrator":
        """Compute per-stage conformal corrections from calibration data."""
        self._scores = cqr_nonconformity_scores(y_cal, q_lo_cal, q_hi_cal)
        self.thresholds = per_stage_conformal(
            self._scores, stages_cal, self.alpha, self.min_stage_n
        )
        return self

    def get_q_hat(self, stage: str) -> float:
        """Get the conformal correction for a given stage."""
        return self.thresholds.get(stage, self.thresholds.get("_global", 0.0))

    def transform(
        self,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        stages: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply per-stage conformal correction to prediction intervals."""
        q_lo_adj = np.copy(q_lo).astype(float)
        q_hi_adj = np.copy(q_hi).astype(float)

        for stage in np.unique(stages):
            mask = stages == stage
            q_hat = self.get_q_hat(str(stage))
            q_lo_adj[mask] -= q_hat
            q_hi_adj[mask] += q_hat

        return q_lo_adj, q_hi_adj

    def transform_single(
        self, q_lo: float, q_hi: float, stage: str
    ) -> tuple[float, float]:
        """Apply conformal correction for a single prediction."""
        q_hat = self.get_q_hat(stage)
        return q_lo - q_hat, q_hi + q_hat

    def thresholds_df(self) -> pd.DataFrame:
        """Return the per-stage thresholds as a DataFrame for persistence."""
        rows = []
        for stage, q_hat in self.thresholds.items():
            rows.append({"stage": stage, "q_hat_mm": round(q_hat, 2)})
        return pd.DataFrame(rows)
