"""
CARE-Paddy — Streamlit dashboard.

Demo path for the review:
  1. Decision tab      → action card, plain-English reason, rule audit trail
  2. Forecast chart    → median + 90% interval against the stage floor and bund
  3. Gate meter        → interval width vs the automation threshold
  4. Break the sensor  → inject a fault, watch the interval widen, watch the gate fire
  5. Manual entry      → type a field state; every derived feature is shown
  6. Decision map      → the whole policy across water level x forecast rain
  7. Validation tab    → LOSO error, calibration, gating, sensitivity

Run:
  streamlit run dashboard/app.py

The scope strip is mandatory per the project honesty rules and appears on every
view. Do not remove it. Wording is specified in AGENT_SPEC.md section 8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

import config as C
from decision import decide, ACTIONS
from features import FEATURES
from manual_input import FieldInput, derive_row, feature_frame, climatology, PRESETS

import theme as T

st.set_page_config(page_title="CARE-Paddy", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")
T.inject()

CFG   = C.load()
FIGD  = ROOT / "outputs" / "figures"
METD  = ROOT / "outputs" / "metrics"
BUND  = float(CFG["hydrology"]["bund_height_mm"])


# ── loaders ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    return pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")


@st.cache_data
def load_field_state():
    return pd.read_csv(ROOT / "data" / "processed" / "field_state.csv")


@st.cache_resource
def load_models():
    m = {}
    for h in CFG["model"]["horizons"]:
        for q in CFG["model"]["quantiles"]:
            k = int(q * 100)
            mdl = XGBRegressor()
            mdl.load_model(ROOT / "outputs" / "models" / f"xgb_h{h}_q{k:02d}.json")
            m[(h, k)] = mdl
    return m


@st.cache_data
def load_conformal():
    p = ROOT / "outputs" / "models" / "conformal_thresholds.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    return dict(zip(t.stage, t.q_hat_mm))


df                   = load_features()
fs                   = load_field_state()
mdls                 = load_models()
conformal_thresholds = load_conformal()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — inputs
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Input mode")
    mode = st.radio(
        "Where do the field readings come from?",
        ["Historical replay", "Manual entry (live)"], label_visibility="collapsed",
        help="Historical replay steps through a simulated season. Manual entry lets "
             "you type the field state directly — the path a deployed ESP32 drives.",
    )
    manual_mode = mode.startswith("Manual")
    st.divider()

    if not manual_mode:
        st.header("Field selection")
        all_seasons = sorted(df.season_id.unique())
        season = st.selectbox("Season", all_seasons, index=len(all_seasons) - 1)
        S = df[df.season_id == season].dropna(subset=FEATURES).reset_index(drop=True)
        day = st.slider("Day after transplanting",
                        int(S.das.min()), int(S.das.max()), int(S.das.median()))
    else:
        st.header("Scenario")
        preset_name = st.selectbox(
            "Start from a preset", ["— custom —"] + list(PRESETS.keys()),
            help="Each preset is a situation with a clear, verified correct answer.")
        pre = {k: v for k, v in PRESETS.get(preset_name, {}).items()
               if not k.startswith("_")}
        why = PRESETS.get(preset_name, {}).get("_why", "")
        exp = PRESETS.get(preset_name, {}).get("_expect", "")
        if exp:
            st.markdown(
                f'<div class="cp-flag water" style="margin-top:6px"><div class="lbl">'
                f'Expected outcome &nbsp;·&nbsp; {exp}</div>{why}</div>',
                unsafe_allow_html=True)

        st.divider()
        st.header("Crop")
        season_name = st.selectbox(
            "Season / variety", ["Kuruvai", "Samba"],
            index=["Kuruvai", "Samba"].index(pre.get("season_name", "Kuruvai")),
            help="Kuruvai: 120-day, irrigation-fed. Samba: 150-day, rain-fed.")
        max_das = 120 if season_name == "Kuruvai" else 150
        day = st.number_input("Days after transplanting", 1, max_das,
                              int(pre.get("das", min(60, max_das))), 1,
                              help="Drives growth stage, crop coefficient, AWD floor.")

        clim = climatology(fs, season_name, int(day))
        if st.button("Reset to typical conditions", use_container_width=True):
            for k, v in clim.items():
                st.session_state[f"mi_{k}"] = v
            st.rerun()

        def _d(key, fallback):
            return float(pre.get(key, clim.get(key, fallback)))

        st.divider()
        st.header("Field reading")
        wl_now = st.slider(
            "Current water level (mm)", -300, 100, int(_d("wl_now_mm", 40)), 1,
            key="mi_wl_now_mm",
            help="Relative to the soil surface. Positive = ponded to that depth. "
                 "Negative = water table that far below the surface.")
        wl_trend = st.slider(
            "3-day trend (mm/day)", -30.0, 30.0,
            float(pre.get("wl_trend_mm_day", -4.0)), 0.5, key="mi_wl_trend",
            help="Negative = falling. Reconstructs the lagged history the model expects.")
        use_probe = st.checkbox("I have a soil-moisture probe reading")
        sm_override = st.slider("Soil moisture (% vol)", 15.0, 50.0, 45.0, 0.5,
                                disabled=not use_probe)

        st.divider()
        st.header("Rain")
        rain_today = st.number_input("Today (mm)", 0.0, 300.0,
                                     _d("rain_today_mm", 0.0), 0.5, key="mi_rain_today_mm")
        rain_3d = st.number_input("Last 3 days, total (mm)", 0.0, 500.0,
                                  _d("rain_3d_mm", 0.0), 0.5, key="mi_rain_3d_mm")
        rain_7d = st.number_input("Last 7 days, total (mm)", 0.0, 800.0,
                                  _d("rain_7d_mm", 0.0), 0.5, key="mi_rain_7d_mm")
        st.caption("Forecast for the next three days — what a weather app shows.")
        f1, f2, f3 = st.columns(3)
        rain_fc1 = f1.number_input("+1d", 0.0, 300.0, _d("rain_fc_t1_mm", 0.0), 0.5,
                                   key="mi_rain_fc_t1_mm")
        rain_fc2 = f2.number_input("+2d", 0.0, 300.0, _d("rain_fc_t2_mm", 0.0), 0.5,
                                   key="mi_rain_fc_t2_mm")
        rain_fc3 = f3.number_input("+3d", 0.0, 300.0, _d("rain_fc_t3_mm", 0.0), 0.5,
                                   key="mi_rain_fc_t3_mm")

        st.divider()
        st.header("Temperature")
        tmax = st.slider("Recent mean Tmax (°C)", 20.0, 45.0, _d("tmax_c", 34.5), 0.5,
                         key="mi_tmax_c",
                         help="Drives growing degree days and evapotranspiration.")
        tmin = st.slider("Recent mean Tmin (°C)", 10.0, 32.0, _d("tmin_c", 25.5), 0.5,
                         key="mi_tmin_c")

        manual_input = FieldInput(
            season_name=season_name, das=int(day),
            wl_now_mm=float(wl_now), wl_trend_mm_day=float(wl_trend),
            rain_today_mm=float(rain_today), rain_3d_mm=float(rain_3d),
            rain_7d_mm=float(rain_7d),
            rain_fc_t1_mm=float(rain_fc1), rain_fc_t2_mm=float(rain_fc2),
            rain_fc_t3_mm=float(rain_fc3),
            tmax_c=float(tmax), tmin_c=float(tmin),
            sm_override_pct=float(sm_override) if use_probe else None)
        season = f"{season_name} (manual)"
        S = df[df.season_id.str.startswith(season_name)].dropna(
            subset=FEATURES).reset_index(drop=True)

    st.divider()
    st.header("Sensor fault injection")
    fault = st.radio(
        "Inject fault",
        ["none", "stuck water-level sensor", "drifting sensor", "erratic readings"],
        label_visibility="collapsed",
        help="Break the sensor and watch the uncertainty gate refuse to actuate.")
    st.caption("Drift, stuck readings and sudden noise all widen the 90% interval — "
               "one threshold catches all three.")

    st.divider()
    st.header("Gate calibration")
    gate_mode = st.radio(
        "Gate mode", ["Conformal (CQR)", "Fixed threshold"],
        label_visibility="collapsed",
        help="Fixed: compares the raw 90% interval width against the threshold. "
             "Conformal: applies the CQR correction first, then compares.")
    gate = st.slider(
        "Gate threshold (mm)", 10, 200,
        int(CFG["model"]["gate_interval_width_mm"]), 5,
        help="The safety / automation dial, used in BOTH modes. Lower defers more to "
             "a human; higher automates more. 90 mm sits near the knee of the curve.")

    st.divider()
    st.caption(f"{CFG['site']['name']}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════════════════
T.masthead(CFG["site"]["name"].split("(")[0].strip(),
           CFG["site"]["lat"], CFG["site"]["lon"])

derived_ctx, derived_notes = None, []
if manual_mode:
    row, derived_ctx, derived_notes = derive_row(manual_input, CFG)
    wx_src = "operator-entered"
else:
    _sel = S[S.das == day]
    if _sel.empty:
        st.error("No data for this day. Try adjusting the slider.")
        st.stop()
    row = _sel.iloc[0].copy()
    wx_src = {"nasa_power": "NASA POWER"}.get(
        str(row.get("weather_source", "synthetic")),
        str(row.get("weather_source", "synthetic")).replace("_", " "))

T.scope_strip(wx_src)

# fault injection
rng = np.random.default_rng(int(day))
if fault.startswith("stuck"):
    ref_val = (float(row["wl_lag7"]) if manual_mode
               else float(S.wl_obs_mm.iloc[max(0, int(day) - 7)]))
    for c in ["wl_obs_mm", "wl_lag1", "wl_lag2", "wl_lag3", "wl_lag7"]:
        row[c] = ref_val
    row["wl_delta1"] = 0.0
    row["wl_delta3"] = 0.0
elif fault.startswith("drift"):
    row["wl_obs_mm"] -= 80
    row["wl_delta1"] -= 20
    row["wl_delta3"] -= 55
elif fault.startswith("erratic"):
    row["wl_obs_mm"] += float(rng.normal(0, 85))
    row["wl_delta1"] += float(rng.normal(0, 60))

X  = pd.DataFrame([row[FEATURES].astype(float)])
fc = {h: {f"q{k:02d}": float(mdls[(h, k)].predict(X)[0]) for k in (5, 50, 95)}
      for h in CFG["model"]["horizons"]}
fcd = {h: {"q05": v["q05"], "q50": v["q50"], "q95": v["q95"]} for h, v in fc.items()}

_ct = conformal_thresholds if gate_mode == "Conformal (CQR)" else None
d = decide(float(row.wl_obs_mm), str(row.stage), fcd,
           float(row.rain_fc_sum3_mm), CFG, gate_mm=float(gate),
           conformal_thresholds=_ct)

stage_txt = str(row.stage).replace("_", " ").title()
no_drain  = str(row.stage) in CFG["phenology"]["no_drain_stages"]
floor     = float(CFG["awd"]["stage_floor_mm"].get(
    str(row.stage), CFG["awd"]["safe_threshold_mm"]))
q_hat = (conformal_thresholds.get(str(row.stage),
                                  conformal_thresholds.get("_global", 0.0))
         if (conformal_thresholds and gate_mode.startswith("Conformal")) else None)

T.pills([
    ("STAGE", stage_txt, "on"),
    ("DAY", f"{int(row.das)}", ""),
    ("AWD FLOOR", f"{floor:.0f} mm", ""),
    ("DRAIN", "forbidden" if no_drain else "permitted", "hot" if no_drain else ""),
    ("SOURCE", "manual entry" if manual_mode else season, ""),
    ("GATE", f"{gate_mode.split()[0].lower()} @ {gate:.0f} mm", ""),
] + ([("FAULT", fault.split()[0], "hot")] if fault != "none" else [])
  + ([("STAGE Q̂", f"{q_hat:+.1f} mm", "")] if q_hat is not None else []))


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_dec, tab_season, tab_why, tab_val, tab_method = st.tabs(
    ["  Decision  ", "  Season context  ", "  Why this forecast  ",
     "  Validation  ", "  Method & scope  "])


# ─────────────────────────────────────────────────────── DECISION ────────────
with tab_dec:
    left, right = st.columns([1, 2.15], gap="medium")

    with left:
        T.action_card(d.action, d.gated, d.constraint_applied, d.rule)

    with right:
        T.reason(d.reason.replace("_", " "))
        if d.constraint_applied:
            T.flag("rust", "Hard agronomic constraint applied",
                   "Drainage is forbidden between panicle initiation and flowering, "
                   "regardless of the forecast. This constraint is evaluated "
                   "<b>before</b> any model output and cannot be overridden.")
        if d.gated:
            T.flag("amber", "Uncertainty gate fired",
                   f"The 90% prediction interval is <b>{d.interval_width_mm:.0f} mm</b>, "
                   f"beyond the {gate:.0f} mm safety threshold. The system refuses to "
                   f"actuate and requests manual inspection.")
        if not d.gated and not d.constraint_applied:
            T.flag("rice", f"Rule fired · {d.rule}",
                   "Normal operation — the forecast was confident enough to act on, "
                   "and no agronomic constraint applied at this stage.")

    st.write("")
    t1, t2, t3, t4, t5 = st.columns(5)
    trend = float(row.wl_delta1)
    T.tile(t1, "Water level", f"{row.wl_obs_mm:.0f}", "mm",
           f"{trend:+.1f} mm/day" if abs(trend) > .05 else "steady")
    T.tile(t2, "Headroom to bund", f"{BUND - float(row.wl_obs_mm):.0f}", "mm",
           f"bund at {BUND:.0f} mm")
    T.tile(t3, "Soil moisture", f"{row.sm_obs_pct:.1f}", "%", "volumetric")
    T.tile(t4, "Rain forecast", f"{row.rain_fc_sum3_mm:.0f}", "mm", "next 3 days")
    T.tile(t5, "90% interval", f"{d.interval_width_mm:.0f}", "mm",
           "exceeds gate" if d.gated else f"{gate - d.interval_width_mm:.0f} mm margin",
           "alert" if d.gated else "good")

    # ── forecast + gate ──────────────────────────────────────────────────────
    T.section("Water level forecast",
              "Median with the 90% prediction interval, against this stage's safe "
              "operating band.")
    fcol, gcol = st.columns([2.1, 1], gap="medium")

    with fcol:
        hs   = list(CFG["model"]["horizons"])
        xs   = [0] + hs
        med  = [float(row.wl_obs_mm)] + [fc[h]["q50"] for h in hs]
        lo   = [float(row.wl_obs_mm)] + [fc[h]["q05"] for h in hs]
        hi   = [float(row.wl_obs_mm)] + [fc[h]["q95"] for h in hs]

        y_lo = min(lo + [floor]) - 30
        y_hi = max(hi + [BUND]) + 62          # headroom so the legend never collides
        x_hi = max(hs) + .35

        fig, ax = plt.subplots(figsize=(7.4, 3.5))
        ax.axhspan(y_lo, floor, color=T.RUST, alpha=.07, zorder=0)
        ax.axhline(floor, color=T.RUST, lw=1.5, ls="--", zorder=2)
        ax.text(x_hi, floor, f"AWD floor {floor:.0f} mm ", color=T.RUST, fontsize=8.5,
                va="bottom", ha="right", zorder=3)
        ax.axhline(BUND, color=T.SKY, lw=1.4, ls="--", zorder=2)
        ax.text(x_hi, BUND, f"bund {BUND:.0f} mm ", color=T.SKY, fontsize=8.5,
                va="bottom", ha="right", zorder=3)
        ax.axhline(0, color=T.INK3, lw=1.0, ls=":", zorder=2)

        ax.fill_between(xs, lo, hi, color=T.WATER, alpha=.16, zorder=3,
                        label="90% interval")
        ax.plot(xs, med, color=T.WATER, lw=2.4, marker="o", ms=6, zorder=5,
                label="Median forecast")
        ax.plot([0], [float(row.wl_obs_mm)], marker="o", ms=10, mfc="white",
                mec=T.INK, mew=2, zorder=6, label="Today (sensor)")
        for h in hs:
            ax.annotate(f"{fc[h]['q50']:.0f}", (h, fc[h]["q50"]),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.5, fontweight="bold", color=T.INK)

        ax.set_xticks(xs)
        ax.set_xticklabels(["today"] + [f"+{h}d" for h in hs])
        ax.set_ylabel("Water level (mm rel. soil surface)")
        ax.set_xlim(-.18, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.legend(loc="upper left", ncol=3, frameon=True, facecolor=T.CARD,
                  edgecolor=T.RULE, framealpha=.96, borderpad=.5,
                  handlelength=1.5, columnspacing=1.1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Positive = ponded water of that depth. Negative = water table "
                   "below the surface. The shaded band is where the crop is at risk.")

    with gcol:
        _w, _lim = float(d.interval_width_mm), float(gate)
        figg, axg = plt.subplots(figsize=(4.0, 1.5))
        axg.barh([0], [_w], height=.42,
                 color=T.RUST if _w > _lim else T.RICE)
        axg.axvline(_lim, color=T.INK, lw=1.8, ls="--")
        axg.text(_lim, .32, f" gate {_lim:.0f}", color=T.INK, fontsize=8.5,
                 va="bottom")
        axg.text(_w, 0, f" {_w:.0f} mm ", va="center",
                 ha="left" if _w < _lim else "right", fontsize=10,
                 fontweight="bold", color="white" if _w >= _lim else T.INK)
        axg.set_xlim(0, max(_w * 1.3, _lim * 1.45))
        axg.set_ylim(-.4, .55)
        axg.set_yticks([])
        axg.grid(False)
        axg.set_xlabel("interval width (mm)", fontsize=8.5)
        for sp in ("top", "right", "left"):
            axg.spines[sp].set_visible(False)
        st.pyplot(figg, use_container_width=True)
        plt.close(figg)

        if d.gated:
            T.flag("rust", "Automation blocked",
                   "The forecast is too uncertain to act on. This refusal is the "
                   "system working as designed, not failing.")
        else:
            T.flag("rice", "Within threshold",
                   f"{_lim - _w:.0f} mm of margin before the gate would fire.")
        st.caption("A stuck sensor, calibration drift or unprecedented weather all "
                   "push the model into sparse feature space, where the q05 and q95 "
                   "models disagree more and the interval widens. One rule catches "
                   "all of them.")

    # ── manual-mode extras ───────────────────────────────────────────────────
    if manual_mode:
        with st.expander("How these numbers were derived — 12 inputs to 29 features"):
            st.markdown(
                "Nothing was invented. Every feature you did not type comes from one "
                "of four documented rules: **identity**, **the simulator's own "
                "physics**, **linear back-extrapolation** of the trend you reported, "
                "or **steady state** where no history was supplied.")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown("**Phenology & water demand** — physics")
                st.dataframe(pd.DataFrame({
                    "Quantity": ["Growth stage", "Crop coefficient Kc",
                                 "Cumulative GDD", "GDD fraction", "Season GDD total",
                                 "Reference ET₀", "Crop water demand", "Day of year"],
                    "Value": [derived_ctx["stage"].replace("_", " ").title(),
                              derived_ctx["kc"], f"{derived_ctx['cum_gdd']:.0f}",
                              derived_ctx["gdd_fraction"],
                              f"{derived_ctx['gdd_total']:.0f}",
                              f"{derived_ctx['et0_mm']:.2f} mm/day",
                              f"{derived_ctx['et_demand_mm']:.2f} mm/day",
                              derived_ctx["day_of_year"]],
                }), use_container_width=True, hide_index=True)
            with dc2:
                st.markdown("**Field state** — identity & reconstruction")
                st.dataframe(pd.DataFrame({
                    "Quantity": ["Headroom to bund", "Soil moisture", "AWD floor",
                                 "Drainage permitted", "Forecast rain 3-day",
                                 "Level −1 day", "Level −3 days", "Level −7 days"],
                    "Value": [f"{derived_ctx['headroom_mm']:.1f} mm",
                              f"{derived_ctx['soil_moisture_pct']:.1f} %",
                              f"{derived_ctx['stage_floor_mm']:.0f} mm",
                              "No — forbidden at this stage"
                              if derived_ctx["drain_forbidden"] else "Yes",
                              f"{derived_ctx['rain_fc_sum3_mm']:.1f} mm",
                              f"{row['wl_lag1']:.0f} mm", f"{row['wl_lag3']:.0f} mm",
                              f"{row['wl_lag7']:.0f} mm"],
                }), use_container_width=True, hide_index=True)

            st.markdown("**Assumptions made**")
            for n in derived_notes:
                st.markdown(f"- {n}")
            st.markdown("**Full 29-feature vector sent to the model**")
            st.dataframe(pd.DataFrame(
                {"feature": FEATURES,
                 "value": [round(float(row[f]), 3) for f in FEATURES]}),
                use_container_width=True, hide_index=True, height=230)

        T.section("Decision boundary — the whole policy at a glance",
                  "Every cell is a complete run: features derived, nine models "
                  "queried, decision engine evaluated. Everything except water level "
                  "and forecast rain is held at your current settings.")

        @st.cache_data(show_spinner="Sweeping the decision space…")
        def decision_map(season_name, das, wl_trend, rain_today, rain_3d, rain_7d,
                         tmax, tmin, gate_mm, use_conformal, sm_override, fault_key):
            wls = np.linspace(-250, 100, 30)
            fcs = np.linspace(0, 60, 20)
            rows_, coords = [], []
            for fcsum in fcs:
                per = fcsum / 3.0
                for wl in wls:
                    r_, _, _ = derive_row(FieldInput(
                        season_name=season_name, das=int(das), wl_now_mm=float(wl),
                        wl_trend_mm_day=wl_trend, rain_today_mm=rain_today,
                        rain_3d_mm=rain_3d, rain_7d_mm=rain_7d,
                        rain_fc_t1_mm=per, rain_fc_t2_mm=per, rain_fc_t3_mm=per,
                        tmax_c=tmax, tmin_c=tmin, sm_override_pct=sm_override), CFG)
                    rows_.append(r_)
                    coords.append((wl, fcsum))
            Xg = feature_frame(rows_)
            P = {(h, k): mdls[(h, k)].predict(Xg)
                 for h in CFG["model"]["horizons"] for k in (5, 50, 95)}
            ct = conformal_thresholds if use_conformal else None
            grid = np.zeros((len(fcs), len(wls)), dtype=int)
            for i, (r_, (wl, fcsum)) in enumerate(zip(rows_, coords)):
                fcd_ = {h: {"q05": float(P[(h, 5)][i]), "q50": float(P[(h, 50)][i]),
                            "q95": float(P[(h, 95)][i])}
                        for h in CFG["model"]["horizons"]}
                act = decide(float(wl), str(r_["stage"]), fcd_, float(fcsum), CFG,
                             gate_mm=float(gate_mm), conformal_thresholds=ct).action
                grid[i // len(wls), i % len(wls)] = ACTIONS.index(act)
            return wls, fcs, grid

        try:
            wls, fcs, grid = decision_map(
                manual_input.season_name, manual_input.das,
                manual_input.wl_trend_mm_day, manual_input.rain_today_mm,
                manual_input.rain_3d_mm, manual_input.rain_7d_mm,
                manual_input.tmax_c, manual_input.tmin_c,
                float(gate), gate_mode == "Conformal (CQR)",
                manual_input.sm_override_pct, fault)

            mcol, lcol = st.columns([3, 1], gap="medium")
            with mcol:
                cmap = ListedColormap([T.ACTION_COLOUR[a] for a in ACTIONS])
                figm, axm = plt.subplots(figsize=(8.4, 4.0))
                axm.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                           norm=BoundaryNorm(np.arange(-.5, len(ACTIONS)), len(ACTIONS)),
                           extent=[wls[0], wls[-1], fcs[0], fcs[-1]],
                           interpolation="nearest")
                axm.grid(False)
                axm.axvline(derived_ctx["stage_floor_mm"], color="white", lw=1.5, ls=":")
                axm.text(derived_ctx["stage_floor_mm"], fcs[-1] * .97,
                         f" AWD floor {derived_ctx['stage_floor_mm']:.0f} mm",
                         color="white", fontsize=8, va="top")
                for lw_, col_ in ((3.0, "white"), (1.4, T.INK)):
                    axm.scatter([manual_input.wl_now_mm],
                                [derived_ctx["rain_fc_sum3_mm"]], s=200, marker="o",
                                facecolor="none", edgecolor=col_, lw=lw_, zorder=6)
                axm.annotate("you are here",
                             (manual_input.wl_now_mm, derived_ctx["rain_fc_sum3_mm"]),
                             textcoords="offset points", xytext=(15, 12),
                             color="white", fontsize=8.5, fontweight="bold", zorder=7)
                axm.set_xlabel("Current water level (mm rel. soil surface)")
                axm.set_ylabel("Forecast rain, next 3 days (mm)")
                axm.set_title(
                    f"{derived_ctx['stage'].replace('_',' ').title()} · day "
                    f"{manual_input.das} · "
                    f"{'conformal' if gate_mode.startswith('Conformal') else 'fixed'} "
                    f"gate at {gate:.0f} mm")
                st.pyplot(figm, use_container_width=True)
                plt.close(figm)
            with lcol:
                counts = pd.Series([ACTIONS[i] for i in grid.ravel()]).value_counts()
                st.markdown("**Share of the mapped space**")
                for a in ACTIONS:
                    pct = 100 * counts.get(a, 0) / grid.size
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;'
                        f'margin-bottom:5px;font-size:.82rem">'
                        f'<span style="width:12px;height:12px;border-radius:3px;'
                        f'background:{T.ACTION_COLOUR[a]};display:inline-block"></span>'
                        f'<span style="flex:1">{a}</span>'
                        f'<b>{pct:.0f}%</b></div>', unsafe_allow_html=True)
                st.caption("Read it left to right: at low levels the system irrigates; "
                           "as the rain forecast grows, IRRIGATE gives way to DELAY — "
                           "waiting for free water instead of pumping. Near the bund it "
                           "drains, unless the crop stage forbids it.")
        except Exception as exc:                                # pragma: no cover
            st.warning(f"Decision map unavailable: {exc}")


# ──────────────────────────────────────────────── SEASON CONTEXT ─────────────
with tab_season:
    T.section(
        "Season trajectory" if not manual_mode
        else f"Typical {season.split()[0]} season",
        "How the field, the rain and the crop's water demand move across the season."
        if not manual_mode else
        "Averaged across every season on record — context for the day you selected, "
        "not your entered field state.")

    _h = S[["das", "wl_obs_mm", "rain_mm", "et_demand"]]
    hist = (_h.groupby("das").mean() if manual_mode else _h.set_index("das"))
    hist = hist.rename(columns={"wl_obs_mm": "Water level (mm)",
                                "rain_mm": "Rain (mm)",
                                "et_demand": "Crop water demand (mm/day)"})
    st.line_chart(hist, height=330)

    T.section("Stage timeline", "When each growth stage begins and ends, and the "
                                "crop coefficient that governs water demand.")
    sdf = S[["das", "stage", "kc"]].copy()
    if manual_mode:
        sdf = sdf.drop_duplicates(subset="das", keep="first")
    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1:
        span = (sdf.groupby("stage", observed=True)["das"].agg(["min", "max"])
                .rename(columns={"min": "First day", "max": "Last day"}))
        span["Days"] = span["Last day"] - span["First day"] + 1
        span["Kc"] = sdf.groupby("stage", observed=True)["kc"].first()
        span["AWD floor (mm)"] = [
            CFG["awd"]["stage_floor_mm"].get(s, CFG["awd"]["safe_threshold_mm"])
            for s in span.index]
        span["Drain"] = ["forbidden" if s in CFG["phenology"]["no_drain_stages"]
                         else "permitted" for s in span.index]
        span.index = [s.replace("_", " ").title() for s in span.index]
        st.dataframe(span.sort_values("First day"), use_container_width=True)
    with c2:
        st.line_chart(sdf.set_index("das")[["kc"]].rename(
            columns={"kc": "Crop coefficient Kc"}), height=250)
    st.caption("Kc is the FAO-56 crop coefficient: it multiplies reference ET₀ to give "
               "actual crop water demand. It peaks at 1.20 through panicle initiation "
               "and flowering — the stages where the crop drinks hardest and drainage "
               "is forbidden.")


# ─────────────────────────────────────────── WHY THIS FORECAST ───────────────
with tab_why:
    T.section("Feature attribution — TreeSHAP",
              "Which inputs moved the forecast, computed on the median model at the "
              "3-day horizon.")
    T.flag("amber", "Read this first",
           "SHAP is <b>feature attribution, not causal inference</b>. It shows which "
           "inputs moved this forecast relative to a baseline. It does not say that "
           "changing a feature <i>causes</i> the water level to change.")
    st.write("")
    a, b = st.columns(2, gap="medium")
    for col, fname, cap in [
        (a, "shap_beeswarm.png", "Global importance across all seasons. Each dot is "
                                 "one day; position is the SHAP value, colour is the "
                                 "feature's magnitude."),
        (b, "shap_waterfall.png", "One single decision broken into per-feature "
                                  "contributions, from the baseline to the forecast."),
    ]:
        p = FIGD / fname
        with col:
            if p.exists():
                st.image(str(p), use_container_width=True)
                st.caption(cap)
            else:
                st.info(f"Run `python src/explain.py` to generate {fname}.")

    p = METD / "shap_importance.csv"
    if p.exists():
        T.section("Full ranking", "All 29 features by mean absolute SHAP value (mm).")
        st.dataframe(pd.read_csv(p), use_container_width=True, height=300,
                     hide_index=True)
    T.flag("water", "What the ranking says",
           "Current water level and headroom together carry roughly 58% of the total "
           "attribution — the model learned the physical water balance rather than a "
           "spurious correlation. Crop stage enters through <b>Kc</b> and "
           "<b>GDD fraction</b>, not through the stage index, so the model is using "
           "the continuous physiological signal rather than memorising a category.")


# ────────────────────────────────────────────────── VALIDATION ───────────────
with tab_val:
    T.section("Leave-one-season-out validation",
              "Train on 18 seasons, test on the held-out 19th, rotate. Never a random "
              "split — daily field state is autocorrelated, so a random split would "
              "leak tomorrow into training.")

    k1, k2, k3, k4 = st.columns(4)
    T.tile(k1, "MAE, next day", "5.02", "mm", "XGBoost q50 · vs 7.39 persistence")
    T.tile(k2, "R², next day", "0.930", "", "0.873 at 3 days ahead")
    T.tile(k3, "False actuation", "8.07", "%", "8.51% without the gate")
    T.tile(k4, "Deferred to human", "5.4", "%", "gate at 90 mm")

    st.write("")
    vt = st.tabs(["Model comparison", "Interval calibration", "Uncertainty gating",
                  "Per-stage", "Conformal (CQR)", "Sensitivity"])

    with vt[0]:
        p = METD / "model_comparison.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        f = FIGD / "model_comparison.png"
        if f.exists():
            st.image(str(f), use_container_width=True)
        T.flag("water", "How to read this",
               "Persistence — <i>tomorrow equals today</i> — is the bar the ML must "
               "clear. Its R² collapses from 0.886 to 0.681 as the horizon grows, "
               "while XGBoost holds at 0.873. The value grows with lead time, which is "
               "exactly where AWD needs notice. Random Forest matches XGBoost on "
               "accuracy; XGBoost is primary because it produces the calibrated "
               "quantiles the safety gate requires.")

    with vt[1]:
        p = METD / "interval_coverage.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        f = FIGD / "interval_coverage.png"
        if f.exists():
            st.image(str(f), use_container_width=True)
        st.caption("A nominal 90% interval should contain the true value 90% of the "
                   "time. The raw quantile models over-cover at 92–97%, which is safe "
                   "but wasteful — wider intervals mean the gate defers more often "
                   "than necessary.")

    with vt[2]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            f = FIGD / "gating_curve.png"
            if f.exists():
                st.image(str(f), use_container_width=True)
            st.caption("As the threshold falls, more decisions are deferred and false "
                       "actuation drops. 90 mm sits near the knee.")
        with c2:
            f = FIGD / "uncertainty_by_stage.png"
            if f.exists():
                st.image(str(f), use_container_width=True)
            st.caption("Where the model is uncertain. Stem elongation stands out — the "
                       "transition from ponded to the first AWD cycle.")
        p = METD / "gating_summary.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p, index_col=0).T, use_container_width=True)

    with vt[3]:
        p = METD / "stage_false_actuation.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        T.flag("rust", "Our weakest result — stated plainly",
               "Panicle initiation shows <b>44.2% false actuation</b>, at the most "
               "yield-critical stage. Every one of those is a false IRRIGATE, never a "
               "false DRAIN — draining there is structurally impossible because the "
               "hard constraint fires before any model output is consulted. The "
               "failure mode is wasted water, not a lost harvest. The cause is that "
               "the PI floor is 0 mm, the strictest of any stage, so the forecast must "
               "be near-perfect around zero.")

    with vt[4]:
        p = METD / "conformal_coverage_by_stage.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2, gap="medium")
        for col, fn in [(c1, "conformal_coverage_by_stage.png"),
                        (c2, "conformal_vs_fixed_gate.png")]:
            f = FIGD / fn
            if f.exists():
                with col:
                    st.image(str(f), use_container_width=True)
        p = METD / "conformal_thresholds.csv"
        if p.exists():
            T.section("Per-stage conformal thresholds (Q̂) — known limitation")
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        T.flag("amber", "Honest limitation",
               "Global CQR works as reported: it corrected the raw model's "
               "over-coverage onto the 90% target (90.1–90.5%) and cut deferral from "
               "5.38% to 4.87%. <b>Per-stage calibration is implemented and "
               "unit-tested, but not yet effective.</b> The calibration split takes "
               "the last 20% of each training season — and a rice season runs "
               "chronologically to harvest, so that tail is entirely ripening "
               "(510 rows, none from any other stage). Every other stage falls back to "
               "the global Q̂, and stem elongation reaches only 77–82% coverage. "
               "The fix is a stage-stratified split.")

    with vt[5]:
        T.flag("rice", "Conclusions survive a ±30% shake",
               "Percolation and Kc are sourced, not measured at this field. Across a "
               "5×5 grid perturbing both by ±30% — 225 evaluations — XGBoost beat "
               "persistence in <b>75 of 75</b> cells and the model ranking never "
               "inverted. The findings are not artefacts of our parameter choices.")
        st.write("")
        for fn in ["sensitivity_xgboost.png", "sensitivity_persistence.png"]:
            f = FIGD / fn
            if f.exists():
                st.image(str(f), use_container_width=True)
        p = METD / "sensitivity_analysis.csv"
        if p.exists():
            with st.expander("Full 225-row sensitivity grid"):
                st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────── METHOD & SCOPE ─────────────
with tab_method:
    T.section("Architecture", "Six layers. Only one contains machine learning.")
    layers = [
        ("SENSING", "ESP32 or virtual sensor",
         "Water level, soil moisture, air temperature and humidity", False),
        ("CONTEXT", "Weather and soil",
         "NASA POWER · Open-Meteo forecast archive · SoilGrids 250m", False),
        ("PHENOLOGY", "GDD → crop stage",
         "Seven stages from accumulated heat. Rule-based, deterministic", False),
        ("FORECAST", "Quantile XGBoost",
         "29 features → water level at t+1/2/3, each as q05 / q50 / q95", True),
        ("DECISION", "Pure function + gate",
         "Five steps, fixed order. Hard agronomic constraint evaluated first", False),
        ("ACTUATION", "Recommendation only",
         "Pump/valve relay interface exists; not autonomously wired", False),
    ]
    for tag, name, desc, is_ml in layers:
        bg  = T.WATER_L if is_ml else T.CARD
        bc  = T.WATER if is_ml else T.RULE
        badge = (f'<span style="background:{T.WATER};color:#fff;font-size:.6rem;'
                 f'padding:3px 9px;border-radius:20px;font-weight:600;'
                 f'letter-spacing:.06em">THE ML</span>') if is_ml else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:16px;background:{bg};'
            f'border:1px solid {bc};border-left:4px solid '
            f'{T.WATER if is_ml else T.INK3};border-radius:9px;padding:11px 15px;'
            f'margin-bottom:7px">'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:.63rem;'
            f'letter-spacing:.1em;color:{T.WATER if is_ml else T.INK3};width:95px;'
            f'flex-shrink:0">{tag}</span>'
            f'<span style="font-weight:650;width:200px;flex-shrink:0;'
            f'font-size:.92rem">{name}</span>'
            f'<span style="color:{T.INK2};font-size:.85rem;flex:1">{desc}</span>'
            f'{badge}</div>', unsafe_allow_html=True)

    T.flag("amber", "Why so little machine learning?",
           "Crop staging is standard agronomy, and the decision policy must be "
           "auditable by an agronomist. Making either one a model would add error, "
           "remove accountability, and re-introduce the circularity we removed from an "
           "earlier design that predicted rule-generated risk labels from the same "
           "features used to generate them. Only the forecasting layer is learned.")

    T.section("Data provenance", "Three real external sources. One simulated field.")
    st.dataframe(pd.DataFrame({
        "Source": ["NASA POWER", "Open-Meteo forecast archive", "SoilGrids 250m",
                   "Water-balance simulator"],
        "Records": ["3,653 days", "1,093 days", "1 point query", "2,550 days"],
        "Coverage": ["2015-01-01 → 2024-12-31", "2022-01-01 → 2024-12-31",
                     "11.00°N, 79.48°E, 0–5 cm", "19 seasons"],
        "Role": ["Observed weather driving the simulator; ET₀",
                 "Genuine forecast rain — the feature deployment actually has",
                 "Soil texture → percolation 1.0 mm/day (Saxton-Rawls)",
                 "Generates the water level the ML learns to forecast"],
        "Real?": ["Real", "Real", "Real", "Simulated"],
    }), use_container_width=True, hide_index=True)
    st.caption("Only 3 of the 7 NASA POWER variables are used — rainfall and the two "
               "temperature extremes. Hargreaves-Samani ET₀ is temperature-only by "
               "design, precisely because humidity, wind and radiation are the least "
               "reliable variables at 0.5° satellite resolution. Of 19 seasons, 6 "
               "contain genuine forecast-archive rain; the other 13 use a "
               "noise-degraded synthetic forecast, tracked per row.")

    T.section("Scope of claims", "What this project does and does not assert.")
    T.flag("rust", "Not claimed",
           "<b>No field-measured water level. No methane measured or modelled. No "
           "deployed hardware.</b> The weather is real, the soil is real, the field is "
           "simulated. Think of a flight simulator: not a real aircraft, but built on "
           "real physics — and you can tell whether the student crashed.")
    st.write("")
    T.flag("rice", "Claimed",
           "A non-circular forecasting formulation; a quantile model that beats the "
           "naive baseline at every horizon and by more as lead time grows; "
           "distribution-free interval calibration; a safety architecture in which a "
           "hard agronomic rule cannot be overridden by any model output; and a "
           "measured false-actuation rate computed against perfect foresight.")

st.markdown(
    f'<div class="cp-foot">CARE-Paddy · B.Tech capstone · '
    f'{CFG["site"]["name"]} · weather source: {wx_src} · '
    f'All figures are simulation results. '
    f'References: Wu et al. (2023), Ma et al. (2025), Ouyang et al. (2023), '
    f'Romano et al. (2019), FAO-56 Allen et al. (1998), Saxton &amp; Rawls (2006), '
    f'Hengl et al. (2017).</div>', unsafe_allow_html=True)
