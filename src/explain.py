"""
Explainability: TreeSHAP on the median forecast model.

SHAP is FEATURE ATTRIBUTION, not causal inference. It answers "which inputs moved
this particular forecast", not "what causes water level". Say that on the slide.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from features import FEATURES, TARGET_PREFIX


def load_model(cfg, horizon=3, q=50):
    m = XGBRegressor()
    m.load_model(cfg["_root"] / "outputs" / "models" / f"xgb_h{horizon}_q{q:02d}.json")
    return m


def main(cfg=None, horizon=3):
    cfg = cfg or C.load()
    root = cfg["_root"]
    figd = root / "outputs" / "figures"; figd.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    d = df.dropna(subset=FEATURES + [f"{TARGET_PREFIX}{horizon}"])
    X = d[FEATURES]

    model = load_model(cfg, horizon, 50)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)

    plt.figure(figsize=(6.5, 5))
    shap.summary_plot(sv, X, show=False, max_display=14)
    plt.title(f"Feature attribution, water-level forecast t+{horizon}", fontsize=10)
    plt.tight_layout(); plt.savefig(figd / "shap_beeswarm.png", dpi=130); plt.close()

    # one worked example: the driest day in the record
    i = int(np.argmin(d["wl_obs_mm"].to_numpy()))
    plt.figure(figsize=(6.5, 4.2))
    shap.plots._waterfall.waterfall_legacy(
        expl.expected_value, sv[i], feature_names=FEATURES, max_display=10, show=False)
    plt.title(f"Why this forecast: {d.iloc[i].season_id}, {pd.Timestamp(d.iloc[i].date).date()}, "
              f"stage={d.iloc[i].stage}", fontsize=9)
    plt.tight_layout(); plt.savefig(figd / "shap_waterfall.png", dpi=130); plt.close()

    imp = (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(0)})
             .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(root / "outputs" / "metrics" / "shap_importance.csv", index=False)
    print(imp.head(12).to_string(index=False))
    return imp


if __name__ == "__main__":
    main()
