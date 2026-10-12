"""Visual layer for the CARE-Paddy dashboard.

Kept separate from app.py so the presentation concerns (palette, CSS, the small
HTML components, matplotlib defaults) do not clutter the decision logic.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

# ── palette (matches the review deck) ────────────────────────────────────────
INK      = "#16211F"
INK2     = "#45544E"
INK3     = "#7B8A84"
PAPER    = "#FAFAF7"
CARD     = "#FFFFFF"
RULE     = "#DFE4DA"
WATER    = "#1F6F6B"
WATER_L  = "#E3F0EE"
RICE     = "#3F6B34"
RICE_L   = "#E8F0E4"
RUST     = "#A33B2A"
RUST_L   = "#F8E9E5"
AMBER    = "#8A6614"
AMBER_L  = "#F7EFDC"
SKY      = "#2B6CA3"
SKY_L    = "#E2EDF6"

ACTION_COLOUR = {
    "MAINTAIN": RICE,     # nothing to do — settled green
    "IRRIGATE": SKY,      # add water — blue
    "DELAY":    AMBER,    # wait — amber
    "DRAIN":    WATER,    # release water — deep teal
    "INSPECT":  RUST,     # stop, human needed — the alarm colour
}
ACTION_TINT = {
    "MAINTAIN": RICE_L, "IRRIGATE": SKY_L, "DELAY": AMBER_L,
    "DRAIN": WATER_L,   "INSPECT": RUST_L,
}
ACTION_ICON = {
    "MAINTAIN": "✓", "IRRIGATE": "💧", "DELAY": "⏳",
    "DRAIN": "🚰", "INSPECT": "⚠",
}
ACTION_GLOSS = {
    "MAINTAIN": "No action needed",
    "IRRIGATE": "Add water now",
    "DELAY":    "Hold — rain is coming",
    "DRAIN":    "Start an AWD dry cycle",
    "INSPECT":  "Automation blocked",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, select, textarea {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
/* Never restyle icon glyphs — they are ligature fonts and the rule above would
   make them render their own name as text. */
span[data-testid="stIconMaterial"], .material-symbols-rounded,
[class*="material-symbols"], [data-testid*="Icon"] i {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }}
footer {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* The toolbar CONTAINS the sidebar-expand control, so it must never be hidden —
   doing so strands the user with no way to bring a collapsed sidebar back.
   Hide only the Streamlit chrome inside it. */
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
#MainMenu {{ display: none !important; }}

/* Both sidebar controls stay permanently visible and clickable, so collapsing
   is always reversible and obviously so. */
[data-testid="stExpandSidebarButton"] {{
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    pointer-events: auto !important;
    background: {CARD} !important; border: 1px solid {WATER} !important;
    color: {WATER} !important; border-radius: 7px !important;
    box-shadow: 0 2px 8px rgba(20,40,36,.12) !important;
}}
[data-testid="stExpandSidebarButton"]:hover {{
    background: {WATER_L} !important;
}}
[data-testid="stSidebarCollapseButton"] {{
    visibility: visible !important; opacity: .75 !important;
}}
[data-testid="stSidebarCollapseButton"]:hover {{ opacity: 1 !important; }}

h1, h2, h3 {{ letter-spacing: -0.021em; color: {INK}; }}
h2 {{ font-size: 1.32rem !important; font-weight: 700 !important;
      margin: 0 0 .15rem 0 !important; padding-top: .3rem !important; }}
h3 {{ font-size: 1.04rem !important; font-weight: 650 !important; }}
hr {{ margin: 1.1rem 0 !important; border-color: {RULE} !important; }}

/* ── masthead ── */
.cp-head {{
    display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
    border-bottom:2px solid {INK}; padding-bottom:11px; margin-bottom:9px;
}}
.cp-head .name {{ font-size:1.62rem; font-weight:800; letter-spacing:-.032em; color:{INK}; }}
.cp-head .tag  {{ font-size:.9rem; color:{INK2}; font-weight:400; }}
.cp-head .site {{ margin-left:auto; font-family:'IBM Plex Mono',monospace;
                  font-size:.68rem; color:{INK3}; letter-spacing:.04em; }}

/* ── scope strip ── */
.cp-scope {{
    background:{RUST_L}; border-left:3px solid {RUST}; border-radius:5px;
    padding:8px 13px; margin-bottom:14px; font-size:.79rem; color:{INK}; line-height:1.45;
}}
.cp-scope b {{ color:{RUST}; font-weight:700; }}

/* ── pills ── */
.cp-pills {{ display:flex; gap:7px; flex-wrap:wrap; margin-bottom:14px; }}
.cp-pill {{
    font-family:'IBM Plex Mono',monospace; font-size:.68rem; font-weight:500;
    letter-spacing:.045em; padding:4px 11px; border-radius:20px;
    border:1px solid {RULE}; background:{CARD}; color:{INK2}; white-space:nowrap;
}}
.cp-pill b {{ color:{INK}; font-weight:600; }}
.cp-pill.hot {{ background:{RUST_L}; border-color:{RUST}; color:{RUST}; }}
.cp-pill.on  {{ background:{WATER_L}; border-color:{WATER}; color:{WATER}; }}

/* ── action card ── */
.cp-action {{
    border-radius:12px; padding:22px 20px 18px; text-align:center; color:#fff;
    box-shadow:0 1px 2px rgba(20,40,36,.07), 0 10px 26px rgba(20,40,36,.09);
}}
.cp-action .ico  {{ font-size:2.35rem; line-height:1; }}
.cp-action .verb {{ font-size:2.1rem; font-weight:800; letter-spacing:-.035em; margin:7px 0 2px; }}
.cp-action .glo  {{ font-size:.88rem; opacity:.93; font-weight:500; }}
.cp-action .rule {{
    font-family:'IBM Plex Mono',monospace; font-size:.64rem; letter-spacing:.07em;
    margin-top:13px; padding-top:10px; border-top:1px solid rgba(255,255,255,.28);
    opacity:.88;
}}

/* ── stat tiles ── */
.cp-tile {{
    background:{CARD}; border:1px solid {RULE}; border-radius:9px;
    padding:11px 13px; height:100%;
}}
.cp-tile .k {{
    font-family:'IBM Plex Mono',monospace; font-size:.62rem; letter-spacing:.09em;
    text-transform:uppercase; color:{INK3}; margin-bottom:5px;
}}
.cp-tile .v {{ font-size:1.32rem; font-weight:700; color:{INK}; line-height:1.15;
               letter-spacing:-.02em; }}
.cp-tile .v small {{ font-size:.62em; font-weight:600; color:{INK2}; }}
.cp-tile .d {{ font-size:.72rem; color:{INK3}; margin-top:3px; }}
.cp-tile.alert {{ border-color:{RUST}; background:{RUST_L}; }}
.cp-tile.alert .v, .cp-tile.alert .d {{ color:{RUST}; }}
.cp-tile.good  {{ border-color:{RICE}; background:{RICE_L}; }}

/* ── reason block ── */
.cp-reason {{
    background:{CARD}; border:1px solid {RULE}; border-left:3px solid {WATER};
    border-radius:8px; padding:15px 17px; font-size:1.0rem; line-height:1.5; color:{INK};
}}
.cp-reason .lbl {{
    font-family:'IBM Plex Mono',monospace; font-size:.63rem; letter-spacing:.1em;
    text-transform:uppercase; color:{WATER}; margin-bottom:7px; font-weight:600;
}}
.cp-flag {{
    border-radius:8px; padding:12px 15px; margin-top:9px; font-size:.86rem; line-height:1.45;
}}
.cp-flag .lbl {{
    font-family:'IBM Plex Mono',monospace; font-size:.62rem; letter-spacing:.1em;
    text-transform:uppercase; font-weight:700; margin-bottom:5px;
}}
.cp-flag.rust  {{ background:{RUST_L};  border-left:3px solid {RUST};  color:{INK}; }}
.cp-flag.rust .lbl  {{ color:{RUST}; }}
.cp-flag.amber {{ background:{AMBER_L}; border-left:3px solid {AMBER}; color:{INK}; }}
.cp-flag.amber .lbl {{ color:{AMBER}; }}
.cp-flag.rice  {{ background:{RICE_L};  border-left:3px solid {RICE};  color:{INK}; }}
.cp-flag.rice .lbl  {{ color:{RICE}; }}
.cp-flag.water {{ background:{WATER_L}; border-left:3px solid {WATER}; color:{INK}; }}
.cp-flag.water .lbl {{ color:{WATER}; }}

/* ── section heading ── */
.cp-sec {{ margin:22px 0 10px; }}
.cp-sec .t {{ font-size:1.12rem; font-weight:700; letter-spacing:-.02em; color:{INK}; }}
.cp-sec .s {{ font-size:.83rem; color:{INK3}; margin-top:1px; }}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{
    height:40px; padding:0 17px; font-size:.9rem; font-weight:550; color:{INK2};
    background:transparent; border-radius:7px 7px 0 0;
}}
.stTabs [aria-selected="true"] {{ color:{WATER} !important; background:{WATER_L} !important; }}

/* ── sidebar ── */
section[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {RULE}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top:1.1rem; }}
section[data-testid="stSidebar"] h2 {{
    font-size:.7rem !important; font-weight:700 !important; letter-spacing:.1em !important;
    text-transform:uppercase; color:{WATER} !important; margin:.1rem 0 .45rem !important;
}}
section[data-testid="stSidebar"] hr {{ margin:.85rem 0 !important; }}
section[data-testid="stSidebar"] label {{ font-size:.82rem !important; }}

.stDataFrame {{ border:1px solid {RULE}; border-radius:7px; }}
div[data-testid="stExpander"] details {{
    border:1px solid {RULE} !important; border-radius:8px !important; background:{CARD};
}}
.cp-foot {{
    margin-top:26px; padding-top:13px; border-top:1px solid {RULE};
    font-size:.71rem; color:{INK3}; line-height:1.55;
}}

/* ════════════════ presentation layer ════════════════ */
@keyframes cpRise {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:none; }} }}
@keyframes cpPulse {{ 0%,100% {{ box-shadow:0 0 0 0 rgba(163,59,42,.45); }} 50% {{ box-shadow:0 0 0 12px rgba(163,59,42,0); }} }}
@keyframes cpShimmer {{ from {{ background-position:0 0; }} to {{ background-position:240px 0; }} }}

.stApp {{ background:
    radial-gradient(1200px 500px at 85% -120px, rgba(31,111,107,.07), transparent 70%),
    {PAPER}; }}

/* hero */
.cp-hero {{
    position:relative; overflow:hidden; border-radius:16px; color:#fff;
    padding:26px 30px 0; margin:0 0 14px;
    background:linear-gradient(120deg, #0E2522 0%, #15403B 48%, #1F6F6B 100%);
    box-shadow:0 18px 40px rgba(14,37,34,.22); animation:cpRise .5s ease both;
}}
.cp-hero::before {{               /* transplanted rows, receding */
    content:""; position:absolute; inset:0; opacity:.13; pointer-events:none;
    background-image:repeating-linear-gradient(100deg, transparent 0 22px, #9fd99a 22px 24px);
    -webkit-mask-image:linear-gradient(90deg, transparent 30%, #000 100%);
            mask-image:linear-gradient(90deg, transparent 30%, #000 100%);
    animation:cpShimmer 18s linear infinite;
}}
.cp-hero .top {{ position:relative; display:flex; align-items:flex-start; gap:18px; flex-wrap:wrap; }}
.cp-hero .logo {{
    width:52px; height:52px; border-radius:14px; flex-shrink:0; font-size:1.7rem;
    display:flex; align-items:center; justify-content:center;
    background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2);
}}
.cp-hero .name {{ font-size:2.05rem; font-weight:800; letter-spacing:-.04em; line-height:1.05; }}
.cp-hero .tag {{ font-size:.98rem; opacity:.82; margin-top:5px; max-width:640px; line-height:1.4; }}
.cp-hero .chips {{ margin-left:auto; display:flex; flex-direction:column; gap:6px; align-items:flex-end; }}
.cp-hero .chip {{
    font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.06em;
    padding:4px 10px; border-radius:20px; background:rgba(255,255,255,.1);
    border:1px solid rgba(255,255,255,.18); white-space:nowrap;
}}
.cp-hero .chip i {{ font-style:normal; display:inline-block; width:7px; height:7px; border-radius:50%;
                    background:#7ee08a; margin-right:6px; box-shadow:0 0 8px #7ee08a; }}
.cp-kpis {{
    position:relative; display:grid; grid-template-columns:repeat(4, 1fr);
    margin:22px -30px 0; border-top:1px solid rgba(255,255,255,.14);
    background:rgba(0,0,0,.14);
}}
.cp-kpi {{ padding:13px 30px 14px; border-right:1px solid rgba(255,255,255,.1); }}
.cp-kpi:last-child {{ border-right:none; }}
.cp-kpi .v {{ font-size:1.45rem; font-weight:750; letter-spacing:-.03em; }}
.cp-kpi .v small {{ font-size:.6em; opacity:.75; font-weight:600; }}
.cp-kpi .k {{ font-size:.72rem; opacity:.72; margin-top:1px; }}
@media (max-width: 900px) {{ .cp-kpis {{ grid-template-columns:repeat(2, 1fr); }}
                             .cp-hero .chips {{ margin-left:0; align-items:flex-start; }} }}

/* entrance + hover */
.cp-action, .cp-tile, .cp-reason, .cp-flag, .cp-trail {{ animation:cpRise .45s ease both; }}
.cp-tile {{ transition:transform .18s ease, box-shadow .18s ease; }}
.cp-tile:hover {{ transform:translateY(-2px); box-shadow:0 8px 20px rgba(20,40,36,.08); }}
.cp-action {{ position:relative; overflow:hidden; }}
.cp-action::after {{
    content:""; position:absolute; inset:0; pointer-events:none;
    background:radial-gradient(400px 160px at 50% -40px, rgba(255,255,255,.22), transparent 70%);
}}
.cp-action.inspect {{ animation:cpRise .45s ease both, cpPulse 1.8s ease-in-out infinite; }}

/* pill tabs (Streamlit ≥1.5x renders react-aria tabs: role=tablist / stTab) */
.stTabs [role="tablist"] {{
    gap:4px !important; border-bottom:none !important; background:{CARD};
    border:1px solid {RULE}; border-radius:12px; padding:5px; width:fit-content;
    max-width:100%; flex-wrap:wrap; box-shadow:0 1px 3px rgba(20,40,36,.05);
}}
.stTabs [data-testid="stTab"] {{
    height:36px !important; border-radius:8px !important; padding:0 16px !important;
    font-weight:600 !important; color:{INK2} !important; transition:background .15s ease;
}}
.stTabs [data-testid="stTab"]:hover {{ background:{WATER_L} !important; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{
    color:#fff !important; background:{WATER} !important;
    box-shadow:0 2px 8px rgba(31,111,107,.3);
}}
.stTabs [data-testid="stTab"][aria-selected="true"] * {{ color:#fff !important; }}
.stTabs [role="tablist"] + div, .stTabs [role="tablist"] ~ [aria-hidden] {{ }}

/* decision audit trail */
.cp-trail {{ display:flex; align-items:stretch; gap:0; margin:6px 0 4px; }}
.cp-step {{
    flex:1; position:relative; background:{CARD}; border:1px solid {RULE};
    padding:11px 14px 11px 22px; font-size:.78rem; color:{INK3};
    clip-path:polygon(0 0, calc(100% - 12px) 0, 100% 50%, calc(100% - 12px) 100%, 0 100%, 12px 50%);
    margin-left:-6px;
}}
.cp-step:first-child {{ clip-path:polygon(0 0, calc(100% - 12px) 0, 100% 50%, calc(100% - 12px) 100%, 0 100%);
                        margin-left:0; padding-left:14px; border-radius:8px 0 0 8px; }}
.cp-step .n {{ font-family:'IBM Plex Mono',monospace; font-size:.6rem; letter-spacing:.1em; opacity:.8; }}
.cp-step .t {{ font-weight:650; font-size:.84rem; color:{INK2}; margin-top:2px; }}
.cp-step .s {{ font-size:.7rem; margin-top:2px; }}
.cp-step.pass {{ background:{RICE_L}; }}
.cp-step.pass .t {{ color:{RICE}; }}
.cp-step.fire {{ color:#fff; }}
.cp-step.fire .t {{ color:#fff; }}
.cp-step.skip {{ opacity:.55; }}

/* architecture flow */
.cp-flow {{ display:flex; flex-wrap:wrap; gap:10px; align-items:stretch; }}
.cp-layer {{
    flex:1 1 150px; background:{CARD}; border:1px solid {RULE}; border-radius:12px;
    padding:13px 14px; position:relative; animation:cpRise .45s ease both;
}}
.cp-layer .n {{ font-family:'IBM Plex Mono',monospace; font-size:.6rem; letter-spacing:.12em; color:{INK3}; }}
.cp-layer .t {{ font-weight:700; font-size:.92rem; margin:4px 0 4px; color:{INK}; }}
.cp-layer .d {{ font-size:.76rem; color:{INK2}; line-height:1.4; }}
.cp-layer.ml {{ background:linear-gradient(160deg, {WATER} 0%, #15403B 100%); border-color:{WATER}; }}
.cp-layer.ml .n, .cp-layer.ml .t, .cp-layer.ml .d {{ color:#fff; }}
.cp-layer:not(:last-child)::after {{
    content:"→"; position:absolute; right:-9px; top:50%; transform:translateY(-50%);
    color:{INK3}; font-size:.9rem; background:{PAPER}; line-height:1; z-index:2;
}}

/* sidebar brand */
.cp-brand {{ display:flex; align-items:center; gap:10px; padding:2px 0 12px; margin-bottom:6px;
             border-bottom:1px solid {RULE}; }}
.cp-brand .b {{ width:34px; height:34px; border-radius:9px; display:flex; align-items:center;
                justify-content:center; background:linear-gradient(135deg, #15403B, {WATER});
                color:#fff; font-size:1.05rem; }}
.cp-brand .n {{ font-weight:800; letter-spacing:-.03em; font-size:1.02rem; color:{INK}; }}
.cp-brand .s {{ font-size:.68rem; color:{INK3}; }}

/* framed charts / tables */
[data-testid="stVegaLiteChart"], [data-testid="stImage"] img, .stPlotlyChart {{
    border-radius:10px; }}
[data-testid="stVegaLiteChart"] {{ background:{CARD}; border:1px solid {RULE}; padding:10px 8px 4px; }}
</style>
"""


def inject():
    """Apply the stylesheet and matplotlib defaults. Call once, early."""
    st.markdown(CSS, unsafe_allow_html=True)
    plt.rcParams.update({
        "figure.facecolor":  CARD,
        "axes.facecolor":    CARD,
        "axes.edgecolor":    RULE,
        "axes.labelcolor":   INK2,
        "axes.titlecolor":   INK,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        RULE,
        "grid.linewidth":    0.7,
        "xtick.color":       INK3,
        "ytick.color":       INK3,
        "text.color":        INK,
        "font.size":         9.5,
        "axes.titlesize":    10.5,
        "axes.titleweight":  "bold",
        "axes.labelsize":    9.5,
        "legend.frameon":    False,
        "legend.fontsize":   8.8,
        "figure.autolayout": True,
    })


# ── small HTML components ────────────────────────────────────────────────────
def hero(site: str, lat, lon, wx_src: str, kpis):
    """Opening banner. kpis: list of (value, unit, caption) read from outputs/metrics."""
    k = "".join(f'<div class="cp-kpi"><div class="v">{v}<small> {u}</small></div>'
                f'<div class="k">{c}</div></div>' for v, u, c in kpis)
    st.markdown(
        f'<div class="cp-hero"><div class="top">'
        f'<div class="logo">🌾</div>'
        f'<div><div class="name">CARE-Paddy</div>'
        f'<div class="tag">Stage-aware, uncertainty-gated irrigation decision support '
        f'for Alternate Wetting &amp; Drying in the Cauvery delta — forecasts the field '
        f'1–3 days ahead and refuses to act when it cannot trust itself.</div></div>'
        f'<div class="chips">'
        f'<span class="chip"><i></i>{site}</span>'
        f'<span class="chip">{lat}°N · {lon}°E</span>'
        f'<span class="chip">weather · {wx_src}</span></div>'
        f'</div><div class="cp-kpis">{k}</div></div>',
        unsafe_allow_html=True)


def brand():
    st.markdown(
        '<div class="cp-brand"><div class="b">🌾</div><div><div class="n">CARE-Paddy</div>'
        '<div class="s">Irrigation decision support</div></div></div>',
        unsafe_allow_html=True)


# The decision engine's fixed evaluation order (src/decision.py). Each rule name
# maps to the step that returned it, so the trail shows exactly where evaluation
# stopped and which checks were passed on the way.
TRAIL = [
    ("Agronomic constraint", "PI → flowering: no drain", ("agronomic_",)),
    ("Uncertainty gate", "interval vs CQR threshold", ("conformal_gate", "uncertainty_gate")),
    ("AWD thresholds", "stage floor + rain", ("threshold_",)),
    ("Bund-full drain", "near overflow", ("bund_drain",)),
    ("Maintain", "nothing to do", ("maintain",)),
]


def trail(rule: str, action: str):
    fired = next((i for i, (_, _, keys) in enumerate(TRAIL)
                  if any(rule.startswith(k) for k in keys)), len(TRAIL) - 1)
    col = ACTION_COLOUR.get(action, INK2)
    steps = []
    for i, (t, s, _) in enumerate(TRAIL):
        if i < fired:
            cls, note, style = "pass", "✓ passed", ""
        elif i == fired:
            cls, note, style = "fire", f"→ {action}", f' style="background:{col}"'
        else:
            cls, note, style = "skip", "not reached", ""
        steps.append(f'<div class="cp-step {cls}"{style}><div class="n">STEP {i + 1}</div>'
                     f'<div class="t">{t}</div><div class="s">{s} · {note}</div></div>')
    st.markdown(f'<div class="cp-trail">{"".join(steps)}</div>', unsafe_allow_html=True)


def flow(layers):
    """layers: list of (tag, name, desc, is_ml)."""
    html = "".join(
        f'<div class="cp-layer{" ml" if ml else ""}" style="animation-delay:{i * .06:.2f}s">'
        f'<div class="n">{tag}{" · THE ML" if ml else ""}</div>'
        f'<div class="t">{name}</div><div class="d">{desc}</div></div>'
        for i, (tag, name, desc, ml) in enumerate(layers))
    st.markdown(f'<div class="cp-flow">{html}</div>', unsafe_allow_html=True)


def altair_theme():
    """Register and enable a vega-lite theme matching the dashboard palette."""
    import altair as alt

    @alt.theme.register("care_paddy", enable=True)
    def _t():
        return alt.theme.ThemeConfig({"config": {
            "background": CARD,
            "font": "Inter",
            "view": {"stroke": None},
            "axis": {"labelColor": INK3, "titleColor": INK2, "gridColor": RULE,
                     "domainColor": RULE, "tickColor": RULE, "labelFontSize": 11,
                     "titleFontSize": 11.5, "titleFontWeight": 600},
            "legend": {"labelColor": INK2, "titleColor": INK2, "orient": "top",
                       "labelFontSize": 11.5, "titleFontSize": 11},
            "title": {"color": INK, "fontSize": 13, "anchor": "start"},
        }})


def scope_strip(src: str):
    st.markdown(
        f'<div class="cp-scope"><b>Simulation result — not field-validated.</b> '
        f'Field hydrology is generated by a physically-based water-balance model driven '
        f'by {src} weather. The ML layer is a forecasting surrogate validated against '
        f'that simulator, not against field observations. The ESP32 replaces the '
        f'simulator without any change to the pipeline interface. '
        f'No methane has been measured.</div>',
        unsafe_allow_html=True)


def pills(items):
    """items: list of (label, value, kind) — kind in {'', 'hot', 'on'}"""
    html = "".join(
        f'<span class="cp-pill {k}">{lab}&nbsp; <b>{val}</b></span>'
        for lab, val, k in items)
    st.markdown(f'<div class="cp-pills">{html}</div>', unsafe_allow_html=True)


def action_card(action: str, gated: bool, constrained: bool, rule: str):
    note = ("UNCERTAINTY GATED" if gated
            else "AGRONOMIC CONSTRAINT" if constrained
            else f"RULE · {rule}")
    st.markdown(
        f'<div class="cp-action{" inspect" if gated else ""}" '
        f'style="background:{ACTION_COLOUR.get(action,INK2)}">'
        f'<div class="ico">{ACTION_ICON.get(action,"")}</div>'
        f'<div class="verb">{action}</div>'
        f'<div class="glo">{ACTION_GLOSS.get(action,"")}</div>'
        f'<div class="rule">{note}</div></div>',
        unsafe_allow_html=True)


def tile(col, key, value, unit="", sub="", kind=""):
    u = f' <small>{unit}</small>' if unit else ""
    s = f'<div class="d">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="cp-tile {kind}"><div class="k">{key}</div>'
        f'<div class="v">{value}{u}</div>{s}</div>',
        unsafe_allow_html=True)


def reason(text: str):
    st.markdown(
        f'<div class="cp-reason"><div class="lbl">Why this recommendation</div>'
        f'{text}</div>', unsafe_allow_html=True)


def flag(kind: str, label: str, text: str):
    st.markdown(
        f'<div class="cp-flag {kind}"><div class="lbl">{label}</div>{text}</div>',
        unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    s = f'<div class="s">{sub}</div>' if sub else ""
    st.markdown(f'<div class="cp-sec"><div class="t">{title}</div>{s}</div>',
                unsafe_allow_html=True)
