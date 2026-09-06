"""
Evaluation: false-actuation rate, the uncertainty-gating curve, and validation figures.

FALSE ACTUATION RATE (properly defined):
  reference action = policy applied to OBSERVED future state (perfect foresight, no gate)
  system action    = policy applied to FORECAST state
  false actuation  = fraction of system-issued IRRIGATE/DRAIN commands the reference
                     policy would NOT have issued
  → This is a confusion matrix on a real comparison, not an assertion.

THE GATING CURVE is the single most important figure:
  False-actuation rate vs fraction of decisions deferred to INSPECT.
  If gating works, the curve falls. If it doesn't, we say so.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from decision import decide, reference_action, ACTUATIONS
from features import FEATURES

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

STAGE_ORDER = [
    "establishment", "tillering", "stem_elongation",
    "panicle_initiation", "flowering", "grain_filling", "ripening",
]


def wide_frame(cfg):
    """One row per (season, date) with all three horizons of forecast + observed truth."""
    root = cfg["_root"]
    P = pd.read_parquet(root / "outputs" / "predictions.parquet")
    F = pd.read_parquet(root / "data" / "processed" / "features.parquet")

    keep = ["pred_q05", "pred_q50", "pred_q95", "y_true"]
    piv = P.pivot_table(index=["season_id", "date"], columns="horizon", values=keep)
    piv.columns = [f"{a}_h{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    meta = F[["season_id", "date", "stage", "wl_obs_mm",
              "rain_fc_sum3_mm", "rain_sum3", "season_name"]]
    return piv.merge(meta, on=["season_id", "date"], how="left").dropna()


def decisions_df(W, cfg, gate_mm=None, apply_gate=True, conformal_thresholds=None):
    """Apply the decision policy to every row of the wide frame."""
    sys_a, ref_a, gated, width, rule = [], [], [], [], []
    for r in W.itertuples():
        fc = {h: {
            "q05": getattr(r, f"pred_q05_h{h}"),
            "q50": getattr(r, f"pred_q50_h{h}"),
            "q95": getattr(r, f"pred_q95_h{h}"),
        } for h in (1, 2, 3)}
        obs = {h: getattr(r, f"y_true_h{h}") for h in (1, 2, 3)}
        d = decide(r.wl_obs_mm, r.stage, fc, r.rain_fc_sum3_mm, cfg,
                   gate_mm=gate_mm, apply_gate=apply_gate,
                   conformal_thresholds=conformal_thresholds)
        sys_a.append(d.action)
        gated.append(d.gated)
        width.append(d.interval_width_mm)
        rule.append(d.rule)
        ref_a.append(reference_action(r.wl_obs_mm, r.stage, obs, r.rain_sum3, cfg))

    out = W.copy()
    out["system_action"]    = sys_a
    out["reference_action"] = ref_a
    out["gated"]            = gated
    out["interval_width"]   = width
    out["rule"]             = rule
    return out


def false_actuation_rate(D):
    act = D[D.system_action.isin(ACTUATIONS)]
    if len(act) == 0:
        return np.nan, 0
    wrong = (act.system_action != act.reference_action).mean()
    return float(wrong), len(act)


def main(cfg=None):
    cfg = cfg or C.load()
    root = cfg["_root"]
    figd = root / "outputs" / "figures"; figd.mkdir(parents=True, exist_ok=True)
    metd = root / "outputs" / "metrics";  metd.mkdir(parents=True, exist_ok=True)

    W = wide_frame(cfg)

    # =====================================================================
    # 1. GATING CURVE — the most important figure
    # =====================================================================
    rows = []
    for g in np.arange(5, 200, 2.5):
        D = decisions_df(W, cfg, gate_mm=float(g))
        far, n = false_actuation_rate(D)
        rows.append({
            "gate_mm":            g,
            "deferral_fraction":  float((D.system_action == "INSPECT").mean()),
            "false_actuation_rate": far,
            "n_actuations":       n,
        })
    curve = pd.DataFrame(rows).dropna()
    curve.to_csv(metd / "gating_curve.csv", index=False)

    ungated = decisions_df(W, cfg, apply_gate=False)
    far_ungated, n_ung = false_actuation_rate(ungated)

    D0 = decisions_df(W, cfg)
    far_gated, n_g = false_actuation_rate(D0)
    D0.to_csv(metd / "decisions.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(curve.deferral_fraction * 100, curve.false_actuation_rate * 100,
            "o-", ms=2.5, lw=1.8, color="#3b7dd8", label="XGBoost system")
    ax.axhline(far_ungated * 100, ls="--", c="crimson", lw=1.3,
               label=f"No gate: {far_ungated*100:.1f}%")
    op = curve.iloc[(curve.gate_mm - cfg["model"]["gate_interval_width_mm"]).abs().argmin()]
    ax.plot(op.deferral_fraction * 100, op.false_actuation_rate * 100,
            "*", ms=14, c="darkorange",
            label=f"Operating point ({cfg['model']['gate_interval_width_mm']:.0f} mm gate)")
    ax.set_xlabel("Decisions deferred to INSPECT (%)")
    ax.set_ylabel("False actuation rate (%)")
    ax.set_title("Uncertainty gating trades deferral for safety\n(Simulation result — not field-validated)")
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    fig.savefig(figd / "gating_curve.png")
    plt.close(fig)

    # =====================================================================
    # 2. MODEL COMPARISON
    # =====================================================================
    comp = pd.read_csv(metd / "model_comparison.csv")
    model_order = ["persistence", "random_forest", "adaboost", "xgboost_q50"]
    colours     = ["#aaaaaa", "#f4a261", "#e76f51", "#3b7dd8"]
    present = [m for m in model_order if m in comp.model.unique()]
    bar_w = 0.7 / max(len(present), 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric in zip(axes, ["MAE", "R2"]):
        for i, (m, c) in enumerate(zip(present, colours)):
            s = comp[comp.model == m].sort_values("horizon")
            offset = (i - (len(present) - 1) / 2) * bar_w
            ax.bar(s.horizon + offset, s[metric], width=bar_w, label=m, color=c)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_ylabel(metric + (" (mm)" if metric == "MAE" else ""))
        ax.set_title(f"Leave-one-season-out  —  {metric}\n(Simulation result)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figd / "model_comparison.png")
    plt.close(fig)

    # =====================================================================
    # 3. INTERVAL CALIBRATION
    # =====================================================================
    cov = pd.read_csv(metd / "interval_coverage.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    bars = ax.bar(cov.horizon, cov.coverage_90 * 100, width=0.5, color="#3b7dd8")
    ax.axhline(90, ls="--", c="k", lw=1.2, label="Nominal 90%")
    # Annotate with actual coverage values
    for bar, val in zip(bars, cov.coverage_90 * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Prediction-interval calibration\n(nominal 90%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figd / "interval_coverage.png")
    plt.close(fig)

    # =====================================================================
    # 4. DECISION TIMELINE — most recent season
    # =====================================================================
    yr = sorted(W.season_id.unique())[-1]
    S = D0[D0.season_id == yr].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(S.date, S.pred_q05_h3, S.pred_q95_h3,
                    alpha=0.22, color="#3b7dd8", label="90% interval (t+3)")
    ax.plot(S.date, S.y_true_h3, c="k", lw=1.3, label="Observed WL (t+3)")
    ax.plot(S.date, S.pred_q50_h3, c="#3b7dd8", lw=1.2, ls="--", label="Forecast median")
    ax.axhline(0, c="grey", lw=0.8, ls=":")
    colours_act = {
        "IRRIGATE": "tab:green", "DRAIN": "tab:red",
        "INSPECT":  "tab:orange", "DELAY":  "tab:purple",
    }
    for a, c in colours_act.items():
        sub = S[S.system_action == a]
        if len(sub):
            ax.scatter(sub.date, np.full(len(sub), S.pred_q05_h3.min() - 10),
                       s=18, c=c, label=a, marker="s", zorder=5)
    ax.set_ylabel("Water level (mm, relative to soil surface)")
    ax.set_title(f"Season {yr} — forecast, 90% interval, and issued decisions\n"
                 f"(Simulation result — not field-validated)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(figd / "decision_timeline.png")
    plt.close(fig)

    # =====================================================================
    # 5. UNCERTAINTY BY STAGE (boxplot)
    # =====================================================================
    present_stages = [s for s in STAGE_ORDER if s in D0.stage.unique()]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    data_by_stage = [D0.loc[D0.stage == s, "interval_width"].dropna() for s in present_stages]
    bp = ax.boxplot(data_by_stage,
                    tick_labels=[s.replace("_", "\n") for s in present_stages],
                    showfliers=False, patch_artist=True)
    gate = cfg["model"]["gate_interval_width_mm"]
    ax.axhline(gate, ls="--", c="crimson", lw=1.2, label=f"Gate ({gate:.0f} mm)")
    for patch in bp["boxes"]:
        patch.set_facecolor("#cce5ff")
    ax.set_ylabel("90% interval width (mm)")
    ax.set_title("Forecast uncertainty by crop stage")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figd / "uncertainty_by_stage.png")
    plt.close(fig)

    # =====================================================================
    # 6. PER-STAGE FALSE ACTUATION RATE TABLE
    # =====================================================================
    stage_far = []
    for st in present_stages:
        sub = D0[D0.stage == st]
        far_s, n_s = false_actuation_rate(sub)
        deferred = (sub.system_action == "INSPECT").mean()
        stage_far.append({
            "stage":               st,
            "n_days":              len(sub),
            "false_actuation_rate": round(far_s, 4) if not np.isnan(far_s) else "—",
            "deferral_rate":       round(deferred, 4),
        })
    stage_far_df = pd.DataFrame(stage_far)
    stage_far_df.to_csv(metd / "stage_false_actuation.csv", index=False)

    # =====================================================================
    # 7. SUMMARY
    # =====================================================================
    summary = {
        "false_actuation_rate_ungated": far_ungated,
        "false_actuation_rate_gated":   far_gated,
        "relative_reduction":           (far_ungated - far_gated) / far_ungated
                                        if far_ungated and not np.isnan(far_ungated) else np.nan,
        "deferral_fraction":  float((D0.system_action == "INSPECT").mean()),
        "n_actuations_ungated": n_ung,
        "n_actuations_gated":   n_g,
        "n_decisions":          len(D0),
        "weather_source":       W.get("weather_source", pd.Series(["unknown"])).iloc[0]
                                if "weather_source" in W.columns else "unknown",
    }
    pd.Series(summary).to_csv(metd / "gating_summary.csv")

    # =====================================================================
    # 8. CONFORMAL VS FIXED GATE COMPARISON (novelty evaluation)
    # =====================================================================
    thresh_path = root / "outputs" / "metrics" / "conformal_thresholds.csv"
    if thresh_path.exists():
        thresh_df = pd.read_csv(thresh_path)
        conformal_thresholds = dict(zip(thresh_df.stage, thresh_df.q_hat_mm))

        D_cqr = decisions_df(W, cfg, conformal_thresholds=conformal_thresholds)
        far_cqr, n_cqr = false_actuation_rate(D_cqr)
        defer_cqr = float((D_cqr.system_action == "INSPECT").mean())

        # Per-stage comparison
        stage_cmp = []
        for st in present_stages:
            sub_fixed = D0[D0.stage == st]
            sub_cqr   = D_cqr[D_cqr.stage == st]
            far_f, n_f = false_actuation_rate(sub_fixed)
            far_c, n_c = false_actuation_rate(sub_cqr)
            defer_f = (sub_fixed.system_action == "INSPECT").mean()
            defer_c = (sub_cqr.system_action == "INSPECT").mean()
            stage_cmp.append({
                "stage":                     st,
                "n_days":                    len(sub_fixed),
                "fixed_far":                 round(far_f, 4) if not np.isnan(far_f) else "—",
                "conformal_far":             round(far_c, 4) if not np.isnan(far_c) else "—",
                "fixed_deferral":            round(defer_f, 4),
                "conformal_deferral":        round(defer_c, 4),
            })
        stage_cmp_df = pd.DataFrame(stage_cmp)
        stage_cmp_df.to_csv(metd / "conformal_vs_fixed_stage.csv", index=False)

        # Conformal summary
        cqr_summary = {
            "fixed_far":      far_gated,
            "conformal_far":  far_cqr,
            "fixed_deferral": float((D0.system_action == "INSPECT").mean()),
            "conformal_deferral": defer_cqr,
            "fixed_n_actuations": n_g,
            "conformal_n_actuations": n_cqr,
        }
        pd.Series(cqr_summary).to_csv(metd / "conformal_summary.csv")

        # ---- Figure: conformal vs fixed gate per-stage FAR ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

        # Left: FAR comparison
        ax = axes[0]
        plot_stages = [s for s in stage_cmp_df.itertuples()
                       if s.fixed_far != "—" or s.conformal_far != "—"]
        if plot_stages:
            labels = [s.stage.replace("_", "\n") for s in plot_stages]
            fixed_vals = [float(s.fixed_far) * 100 if s.fixed_far != "—" else 0
                          for s in plot_stages]
            cqr_vals = [float(s.conformal_far) * 100 if s.conformal_far != "—" else 0
                        for s in plot_stages]
            x_pos = np.arange(len(labels))
            ax.bar(x_pos - 0.2, fixed_vals, 0.4, label="Fixed gate", color="#e76f51")
            ax.bar(x_pos + 0.2, cqr_vals, 0.4, label="Conformal (CQR)", color="#3b7dd8")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("False actuation rate (%)")
            ax.set_title("Per-stage false actuation:\nFixed vs Conformal gate")
            ax.legend(frameon=False, fontsize=8)

        # Right: Deferral comparison
        ax = axes[1]
        if stage_cmp_df is not None and not stage_cmp_df.empty:
            labels = [s.replace("_", "\n") for s in stage_cmp_df.stage]
            x_pos = np.arange(len(labels))
            ax.bar(x_pos - 0.2, stage_cmp_df.fixed_deferral * 100, 0.4,
                   label="Fixed gate", color="#e76f51")
            ax.bar(x_pos + 0.2, stage_cmp_df.conformal_deferral * 100, 0.4,
                   label="Conformal (CQR)", color="#3b7dd8")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("Deferral rate (%)")
            ax.set_title("Per-stage deferral:\nFixed vs Conformal gate")
            ax.legend(frameon=False, fontsize=8)

        fig.suptitle("Conformal (CQR) vs Fixed uncertainty gating — per-stage comparison\n"
                     "(Simulation result — not field-validated)", fontsize=9)
        fig.tight_layout()
        fig.savefig(figd / "conformal_vs_fixed_gate.png")
        plt.close(fig)

        # ---- Figure: conformal coverage by stage ----
        cov_path = metd / "conformal_coverage_by_stage.csv"
        if cov_path.exists():
            cov_s = pd.read_csv(cov_path)
            cov_h3 = cov_s[cov_s.horizon == 3]
            if not cov_h3.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ordered = [s for s in STAGE_ORDER if s in cov_h3.stage.values]
                sub = cov_h3.set_index("stage").loc[ordered]
                x_pos = np.arange(len(ordered))
                ax.bar(x_pos - 0.2, sub.quantile_coverage * 100, 0.4,
                       label="Quantile XGBoost", color="#f4a261")
                ax.bar(x_pos + 0.2, sub.cqr_coverage * 100, 0.4,
                       label="CQR-adjusted", color="#3b7dd8")
                ax.axhline(90, ls="--", c="k", lw=1.2, label="Nominal 90%")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([s.replace("_", "\n") for s in ordered], fontsize=7)
                ax.set_ylabel("Empirical coverage (%)")
                ax.set_title("Per-stage interval coverage: Quantile vs CQR (h=3)\n"
                             "CQR provides distribution-free ≥90% guarantee per stage")
                ax.legend(frameon=False, fontsize=8)
                ax.set_ylim(0, 105)
                fig.tight_layout()
                fig.savefig(figd / "conformal_coverage_by_stage.png")
                plt.close(fig)

        print("\n=== Conformal vs Fixed gate ===")
        for k, v in cqr_summary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("\nPer-stage conformal comparison:")
        print(stage_cmp_df.to_string(index=False))
    else:
        print("\n[info] No conformal_thresholds.csv found; run train.py first.")

    print("\n=== Gating summary ===")
    print(pd.Series({k: v for k, v in summary.items() if k != "weather_source"}).round(4).to_string())
    print("\nAction distribution:")
    print(D0.system_action.value_counts().to_string())
    print("\nPer-stage false actuation:")
    print(stage_far_df.to_string(index=False))

    return summary


if __name__ == "__main__":
    main()
