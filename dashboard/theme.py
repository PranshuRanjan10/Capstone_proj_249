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
def masthead(site: str, lat, lon):
    st.markdown(
        f'<div class="cp-head"><span class="name">CARE-Paddy</span>'
        f'<span class="tag">Stage-aware, uncertainty-gated irrigation decision support</span>'
        f'<span class="site">{site} &nbsp;·&nbsp; {lat}°N {lon}°E</span></div>',
        unsafe_allow_html=True)


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
        f'<div class="cp-action" style="background:{ACTION_COLOUR.get(action,INK2)}">'
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
