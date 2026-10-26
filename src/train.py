"""
Forecast layer: predict field water level at t+1/t+2/t+3 with prediction intervals.

VALIDATION: LEAVE-ONE-SEASON-OUT
  A random split would leak tomorrow into training (daily field state is strongly
  autocorrelated) and inflate every metric substantially. Note that Wu et al. (2023)
  used a random 7:3 split — a limitation worth naming in the deck.

MODELS
  persistence  — tomorrow = today. Baseline you must beat to justify the ML.
  RandomForest — tabular workhorse.
  AdaBoost     — Ma et al. (2025) ranked this first (R²=best) for paddy gas flux.
  XGBoost      — median + 5th/95th quantiles for the uncertainty gate.
                 Wu et al. (2023): XGBoost best for CH₄ (R²=0.754) and N₂O (R²=0.762).

WHY NO MLP
  Sample size and tabular structure. Ma et al. ranked BPNN 4th/6th. Wu et al. attribute
  neural-net weakness to dataset size. We have less data than either study.

SCALER
  StandardScaler fit on TRAINING fold only, applied to test fold. Prevents data leakage
  from the test season into the feature normalisation.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from features import FEATURES, TARGET_PREFIX
from conformal import ConformalCalibrator


def _scores(y, p, label: str, horizon: int, season: str) -> dict:
    return {
        "model": label, "horizon": horizon, "season": season,
        "MAE":  float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2":   float(r2_score(y, p)),
        "n":    int(len(y)),
    }


def _xgb_quantile(cfg, alpha: float, fast: bool = False) -> XGBRegressor:
    p = dict(cfg["model"]["xgb"])
    if fast:
        p["n_estimators"] = min(p.get("n_estimators", 300), 200)
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        random_state=0,
        n_jobs=-1,
        **p,
    )


def calibration_seasons(train_seasons, frac: float, seed: int) -> list[str]:
    """Hold out whole seasons for conformal calibration, balanced across regimes.

    Calibrating on whole seasons (rather than the tail of every season) gives the
    calibration set every growth stage, and keeps it as independent of the
    training seasons as the test season is — the split-conformal requirement.
    """
    rng = np.random.default_rng(seed)
    by_regime: dict[str, list[str]] = {}
    for s in sorted(train_seasons):
        by_regime.setdefault(s.split("-")[0], []).append(s)
    n_total = max(2, int(round(frac * len(train_seasons))))
    per = max(1, n_total // len(by_regime))
    picked = []
    for regime in sorted(by_regime):
        pool = by_regime[regime]
        picked += list(rng.choice(pool, size=min(per, len(pool) - 1), replace=False))
    return sorted(picked)


def run(cfg=None):
    cfg = cfg or C.load()
    root = cfg["_root"]
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    horizons, quants = cfg["model"]["horizons"], cfg["model"]["quantiles"]
    seasons = sorted(df.season_id.unique())

    rows, pred_frames = [], []
    conformal_cfg = cfg.get("conformal", {})
    cal_frac = conformal_cfg.get("calibration_fraction", 0.20)
    cqr_alpha = conformal_cfg.get("alpha", 0.10)
    cqr_min_n = conformal_cfg.get("min_stage_n", 30)
    if not conformal_cfg.get("per_stage", True):
        cqr_min_n = 10**9                      # every stage falls back to the global Q̂
    q_lo, q_hi = min(quants), max(quants)

    for h in horizons:
        tgt = f"{TARGET_PREFIX}{h}"
        d = df.dropna(subset=FEATURES + [tgt]).copy()

        for k, held in enumerate(seasons):
            tr = d[d.season_id != held]
            te = d[d.season_id == held]

            # --- Split-conformal: hold out whole calibration seasons ---
            # The interval models are fit on the remaining (proper) seasons, then
            # calibrated on the held-out seasons, and the SAME models produce the
            # test-season interval. The held-out test season is never touched.
            cal_ids = calibration_seasons(tr.season_id.unique(), cal_frac, seed=k)
            tr_proper = tr[~tr.season_id.isin(cal_ids)]
            cal_set = tr[tr.season_id.isin(cal_ids)]

            # Fit scaler on training fold only — never on the held-out season
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(tr[FEATURES].to_numpy(dtype=float))
            Xte = scaler.transform(te[FEATURES].to_numpy(dtype=float))
            ytr = tr[tgt].to_numpy()
            yte = te[tgt].to_numpy()

            out = {
                "season_id": held,
                "horizon":   h,
                "date":      te["date"].to_numpy(),
                "stage":     te["stage"].to_numpy(),
                "y_true":    yte,
                "wl_now":    te["wl_obs_mm"].to_numpy(),
            }

            # --- persistence (does not use ML at all, no scaler needed) ---
            p_pers = te["wl_obs_mm"].to_numpy()
            rows.append(_scores(yte, p_pers, "persistence", h, held))
            out["pred_persistence"] = p_pers

            # --- RandomForest ---
            rf = RandomForestRegressor(
                n_estimators=150, min_samples_leaf=4,
                max_features=0.7, random_state=0, n_jobs=-1,
            ).fit(Xtr, ytr)
            p_rf = rf.predict(Xte)
            rows.append(_scores(yte, p_rf, "random_forest", h, held))
            out["pred_rf"] = p_rf

            # --- AdaBoost (Ma et al. 2025: ranked first on paddy data) ---
            ada = AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=4),
                n_estimators=100, learning_rate=0.08,
                loss="square", random_state=0,
            ).fit(Xtr, ytr)
            p_ada = ada.predict(Xte)
            rows.append(_scores(yte, p_ada, "adaboost", h, held))
            out["pred_adaboost"] = p_ada

            # --- XGBoost median: the point forecast, trained on every training season ---
            med = _xgb_quantile(cfg, 0.50, fast=True).fit(tr[FEATURES], tr[tgt])
            out["pred_q50"] = med.predict(te[FEATURES])

            # --- XGBoost interval (q05, q95): fit on proper seasons only ---
            lo = _xgb_quantile(cfg, q_lo, fast=True).fit(tr_proper[FEATURES], tr_proper[tgt])
            hi = _xgb_quantile(cfg, q_hi, fast=True).fit(tr_proper[FEATURES], tr_proper[tgt])
            out["pred_q05"] = lo.predict(te[FEATURES])
            out["pred_q95"] = hi.predict(te[FEATURES])

            # --- CQR: calibrate those same interval models on the held-out seasons ---
            calibrator = ConformalCalibrator(alpha=cqr_alpha, min_stage_n=cqr_min_n)
            calibrator.fit(cal_set[tgt].to_numpy(), lo.predict(cal_set[FEATURES]),
                           hi.predict(cal_set[FEATURES]), cal_set["stage"].to_numpy())
            stages_te = te["stage"].to_numpy()
            cqr_lo, cqr_hi = calibrator.transform(out["pred_q05"], out["pred_q95"], stages_te)
            out["pred_cqr_lo"] = cqr_lo
            out["pred_cqr_hi"] = cqr_hi
            out["cqr_interval_width"] = cqr_hi - cqr_lo
            out["cqr_qhat"] = np.array([calibrator.get_q_hat(str(s)) for s in stages_te])
            out["cqr_stage_specific"] = np.array(
                [str(s) in calibrator.thresholds and
                 calibrator.thresholds[str(s)] != calibrator.thresholds["_global"]
                 for s in stages_te])

            rows.append(_scores(yte, out["pred_q50"], "xgboost_q50", h, held))
            pred_frames.append(pd.DataFrame(out))

    metrics = pd.DataFrame(rows)
    P = pd.concat(pred_frames, ignore_index=True)

    # --- Prediction-interval calibration: raw quantile interval vs CQR ---
    P["covered"]        = (P.y_true >= P.pred_q05) & (P.y_true <= P.pred_q95)
    P["interval_width"] = P.pred_q95 - P.pred_q05
    P["cqr_covered"]    = (P.y_true >= P.pred_cqr_lo) & (P.y_true <= P.pred_cqr_hi)
    cov = (P.groupby("horizon")
             .agg(coverage_90=("covered", "mean"),
                  mean_interval_mm=("interval_width", "mean"),
                  cqr_coverage=("cqr_covered", "mean"),
                  cqr_mean_interval_mm=("cqr_interval_width", "mean"))
             .reset_index())
    cov_stage = (P.groupby(["stage", "horizon"])
                   .agg(quantile_coverage=("covered", "mean"),
                        cqr_coverage=("cqr_covered", "mean"),
                        quantile_width=("interval_width", "mean"),
                        cqr_width=("cqr_interval_width", "mean"),
                        mean_qhat=("cqr_qhat", "mean"),
                        stage_specific_share=("cqr_stage_specific", "mean"),
                        n=("y_true", "count"))
                   .reset_index())

    summary = (
        metrics.groupby(["model", "horizon"])[["MAE", "RMSE", "R2"]]
               .mean().round(3).reset_index()
               .sort_values(["horizon", "MAE"])
    )

    # --- Final models on ALL seasons, for the live dashboard ---
    # The median uses every season. The interval models are fit on all but a fixed,
    # regime-balanced set of calibration seasons, and those same models are
    # conformally calibrated on them — so the deployed interval is the calibrated one.
    mo = root / "outputs" / "models"
    mo.mkdir(parents=True, exist_ok=True)
    final_cal = calibration_seasons(seasons, cal_frac, seed=len(seasons))
    thresh_rows = []
    for h in horizons:
        tgt = f"{TARGET_PREFIX}{h}"
        d = df.dropna(subset=FEATURES + [tgt])
        prop, cal = d[~d.season_id.isin(final_cal)], d[d.season_id.isin(final_cal)]
        _xgb_quantile(cfg, 0.50).fit(d[FEATURES], d[tgt]).save_model(mo / f"xgb_h{h}_q50.json")
        lo = _xgb_quantile(cfg, q_lo).fit(prop[FEATURES], prop[tgt])
        hi = _xgb_quantile(cfg, q_hi).fit(prop[FEATURES], prop[tgt])
        lo.save_model(mo / f"xgb_h{h}_q{int(q_lo*100):02d}.json")
        hi.save_model(mo / f"xgb_h{h}_q{int(q_hi*100):02d}.json")
        calib = ConformalCalibrator(alpha=cqr_alpha, min_stage_n=cqr_min_n).fit(
            cal[tgt].to_numpy(), lo.predict(cal[FEATURES]), hi.predict(cal[FEATURES]),
            cal["stage"].to_numpy())
        n_by_stage = cal.stage.value_counts().to_dict()
        for stage, qh in calib.thresholds.items():
            thresh_rows.append({"stage": stage, "horizon": h, "q_hat_mm": round(qh, 2),
                                "n_cal": int(n_by_stage.get(stage, len(cal)))})
    thresh_df = pd.DataFrame(thresh_rows)
    (mo / "features.json").write_text(json.dumps(FEATURES, indent=2))
    (mo / "calibration_seasons.json").write_text(json.dumps(final_cal, indent=2))
    thresh_df.to_csv(mo / "conformal_thresholds.csv", index=False)

    # --- Persist outputs ---
    md = root / "outputs" / "metrics"
    md.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(md / "loso_by_season.csv", index=False)
    summary.to_csv(md / "model_comparison.csv", index=False)
    cov.to_csv(md / "interval_coverage.csv", index=False)
    cov_stage.to_csv(md / "conformal_coverage_by_stage.csv", index=False)
    thresh_df.to_csv(md / "conformal_thresholds.csv", index=False)
    P.to_parquet(root / "outputs" / "predictions.parquet", index=False)

    print("\n=== Leave-one-season-out (mean over held-out seasons) ===")
    print(summary.to_string(index=False))
    print("\n=== Prediction-interval calibration (nominal 90%) ===")
    print(cov.round(3).to_string(index=False))
    print(f"\n=== Deployed CQR thresholds (Q̂, mm), calibrated on {final_cal} ===")
    print(thresh_df.pivot(index="stage", columns="horizon", values="q_hat_mm").to_string())

    # --- Rank check (important for defending model choice) ---
    h1 = summary[summary.horizon == 1].set_index("model")["MAE"]
    if "xgboost_q50" in h1 and "persistence" in h1:
        beats = h1["xgboost_q50"] < h1["persistence"]
        print(f"\nXGBoost beats persistence at h=1: {beats}")
        if not beats:
            print("  NOTE: persistence wins at t+1 — report this honestly. "
                  "XGBoost should win at t+2 and t+3 where lead time matters.")

    return summary, cov


if __name__ == "__main__":
    run()
