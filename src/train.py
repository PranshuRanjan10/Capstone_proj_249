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
from conformal import ConformalCalibrator, cqr_nonconformity_scores


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

    # Collect conformal scores across all held-out seasons for global calibration
    all_cqr_scores, all_cqr_stages = [], []

    for h in horizons:
        tgt = f"{TARGET_PREFIX}{h}"
        d = df.dropna(subset=FEATURES + [tgt]).copy()

        for held in seasons:
            tr = d[d.season_id != held]
            te = d[d.season_id == held]

            # --- Split training into train_proper + calibration for CQR ---
            # Take the last cal_frac of each training season as calibration
            # (respects temporal order within each season)
            tr_seasons = sorted(tr.season_id.unique())
            tr_proper_idx, cal_idx = [], []
            for s in tr_seasons:
                s_idx = tr.index[tr.season_id == s].tolist()
                split_pt = int(len(s_idx) * (1 - cal_frac))
                tr_proper_idx.extend(s_idx[:split_pt])
                cal_idx.extend(s_idx[split_pt:])
            tr_proper = tr.loc[tr_proper_idx]
            cal_set = tr.loc[cal_idx]

            # Fit scaler on training fold only — never on the held-out season
            scaler = StandardScaler()
            Xtr_raw = tr[FEATURES].to_numpy(dtype=float)
            Xte_raw = te[FEATURES].to_numpy(dtype=float)
            Xtr = scaler.fit_transform(Xtr_raw)
            Xte = scaler.transform(Xte_raw)
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

            # --- XGBoost quantiles (q05, q50, q95) ---
            # Train on full training set (not tr_proper) for maximum accuracy
            for q in quants:
                m = _xgb_quantile(cfg, q, fast=True).fit(
                    tr[FEATURES], tr[tgt]
                )
                out[f"pred_q{int(q*100):02d}"] = m.predict(te[FEATURES])

            # --- CQR: compute conformal scores on calibration set ---
            # Use the same quantile models (trained on full tr) to predict on cal_set
            q05_cal = _xgb_quantile(cfg, 0.05, fast=True).fit(
                tr_proper[FEATURES], tr_proper[tgt]
            ).predict(cal_set[FEATURES])
            q95_cal = _xgb_quantile(cfg, 0.95, fast=True).fit(
                tr_proper[FEATURES], tr_proper[tgt]
            ).predict(cal_set[FEATURES])
            y_cal = cal_set[tgt].to_numpy()
            stages_cal = cal_set["stage"].to_numpy()

            scores = cqr_nonconformity_scores(y_cal, q05_cal, q95_cal)
            all_cqr_scores.append(scores)
            all_cqr_stages.append(stages_cal)

            # Apply per-stage conformal correction to test predictions
            calibrator = ConformalCalibrator(alpha=cqr_alpha, min_stage_n=cqr_min_n)
            calibrator.fit(y_cal, q05_cal, q95_cal, stages_cal)

            cqr_lo, cqr_hi = calibrator.transform(
                out["pred_q05"], out["pred_q95"], te["stage"].to_numpy()
            )
            out["pred_cqr_lo"] = cqr_lo
            out["pred_cqr_hi"] = cqr_hi
            out["cqr_interval_width"] = cqr_hi - cqr_lo

            p_med = out["pred_q50"]
            rows.append(_scores(yte, p_med, "xgboost_q50", h, held))
            pred_frames.append(pd.DataFrame(out))

    metrics = pd.DataFrame(rows)
    P = pd.concat(pred_frames, ignore_index=True)

    # --- Prediction-interval calibration (quantile) ---
    P["covered"]        = (P.y_true >= P.pred_q05) & (P.y_true <= P.pred_q95)
    P["interval_width"] = P.pred_q95 - P.pred_q05
    cov = (
        P.groupby("horizon")
         .agg(
             coverage_90=("covered", "mean"),
             mean_interval_mm=("interval_width", "mean"),
         )
         .reset_index()
    )

    # --- CQR conformal calibration (novelty) ---
    P["cqr_covered"]        = (P.y_true >= P.pred_cqr_lo) & (P.y_true <= P.pred_cqr_hi)
    cov_cqr = (
        P.groupby("horizon")
         .agg(
             cqr_coverage=("cqr_covered", "mean"),
             cqr_mean_interval_mm=("cqr_interval_width", "mean"),
         )
         .reset_index()
    )
    cov = cov.merge(cov_cqr, on="horizon")

    # Per-stage conformal coverage
    cov_stage = (
        P.groupby(["stage", "horizon"])
         .agg(
             quantile_coverage=("covered", "mean"),
             cqr_coverage=("cqr_covered", "mean"),
             quantile_width=("interval_width", "mean"),
             cqr_width=("cqr_interval_width", "mean"),
             n=("y_true", "count"),
         )
         .reset_index()
    )

    # Global conformal thresholds (fit on all collected scores)
    all_scores = np.concatenate(all_cqr_scores)
    all_stages = np.concatenate(all_cqr_stages)
    global_calibrator = ConformalCalibrator(alpha=cqr_alpha, min_stage_n=cqr_min_n)
    global_calibrator.fit(
        np.zeros_like(all_scores),  # dummy y (we already have scores)
        -all_scores,  # reconstruct: q_lo - y = -score when score = max(q_lo-y, y-q_hi)
        all_scores,   # q_hi - y = score
        all_stages,
    )
    # Re-compute thresholds directly from collected scores
    from conformal import per_stage_conformal
    global_thresholds = per_stage_conformal(all_scores, all_stages, cqr_alpha, cqr_min_n)
    thresh_df = pd.DataFrame([
        {"stage": k, "q_hat_mm": round(v, 2)}
        for k, v in global_thresholds.items()
    ])

    summary = (
        metrics.groupby(["model", "horizon"])[["MAE", "RMSE", "R2"]]
               .mean().round(3).reset_index()
               .sort_values(["horizon", "MAE"])
    )

    # --- Persist outputs ---
    md = root / "outputs" / "metrics"
    md.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(md / "loso_by_season.csv", index=False)
    summary.to_csv(md / "model_comparison.csv", index=False)
    cov.to_csv(md / "interval_coverage.csv", index=False)
    cov_stage.to_csv(md / "conformal_coverage_by_stage.csv", index=False)
    thresh_df.to_csv(md / "conformal_thresholds.csv", index=False)
    P.to_parquet(root / "outputs" / "predictions.parquet", index=False)

    # --- Final models on ALL seasons, for the live dashboard ---
    mo = root / "outputs" / "models"
    mo.mkdir(parents=True, exist_ok=True)
    for h in horizons:
        tgt = f"{TARGET_PREFIX}{h}"
        d = df.dropna(subset=FEATURES + [tgt])
        for q in quants:
            m = _xgb_quantile(cfg, q).fit(d[FEATURES], d[tgt])
            m.save_model(mo / f"xgb_h{h}_q{int(q*100):02d}.json")
    (mo / "features.json").write_text(json.dumps(FEATURES, indent=2))
    # Save conformal thresholds for the dashboard
    thresh_df.to_csv(mo / "conformal_thresholds.csv", index=False)

    print("\n=== Leave-one-season-out (mean over held-out seasons) ===")
    print(summary.to_string(index=False))
    print("\n=== Prediction-interval calibration (nominal 90%) ===")
    print(cov.round(3).to_string(index=False))
    print("\n=== CQR per-stage conformal thresholds (Q̂, mm) ===")
    print(thresh_df.to_string(index=False))

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
