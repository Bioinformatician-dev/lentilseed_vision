"""
Visual system.

The app is a seed tray on graph paper. That is what the scale step asks people
to photograph, and it is where every colour and shape here comes from:

  * the sheet is the matte olive-grey of a tray, not paper-white and not cream
  * the ink is black lentil, the working accent is green lentil, and red lentil
    is reserved for the two things that need a warning colour — flags and the
    Defective class
  * the eight workflow stages keep the order of the published figure (blue,
    green, amber, leaf, violet, magenta, teal, orange) but are retuned so they
    sit in the same earthy family as everything else
  * the masthead sits on a faint 10 mm grid; that grid is the one decorative
    element and appears nowhere else

Type is one family, Bricolage Grotesque, which has an optical-size axis: wide
and characterful at display size, plain at body size. Every number is set with
tabular figures so a column can be read without re-aligning the eye per row.

Two surfaces
------------
The palette is resolved once per rerun and pushed into the modules that
captured it, because `ui/panels.py` and `ui/components.py` do
`from .theme import INK, LINE, MUTED, CARD` at import time and feed those
values to Plotly, which needs concrete colours rather than CSS variables:

    mode = theme.active_mode()
    st.markdown(theme.css(mode), unsafe_allow_html=True)
    theme.bind(mode, ui.panels, ui.components)

`STAGE_COLORS` is mutated in place rather than rebound, so every module holding
a reference to that dict sees the change without being listed in `bind`.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------

_SURFACE = {
    "light": {
        "paper": "#F2F4EC",
        "card": "#FFFFFF",
        "ink": "#1B2116",
        "line": "#D3D9C6",
        "muted": "#5F6A55",
        "sunk": "#E9EDE1",          # inset wells: dropzone, seed crops, notes
        "accent": "#5B7128",        # green lentil — buttons, active tab, ring
        "accent_ink": "#FFFFFF",
        "accent_hover": "#4A5D20",
        "flag": "#B8432A",          # red lentil — warnings only
        "grid": "rgba(91,113,40,.16)",
        "ring": "#5B7128",
        "swatch_ring": "rgba(27,33,22,.16)",
    },
    "dark": {
        "paper": "#161A12",
        "card": "#1F2419",
        "ink": "#E9ECDF",
        "line": "#333B2B",
        "muted": "#98A28B",
        "sunk": "#10130C",
        "accent": "#A5C25A",
        "accent_ink": "#141709",
        "accent_hover": "#B8D46E",
        "flag": "#E0765A",
        "grid": "rgba(165,194,90,.14)",
        "ring": "#A5C25A",
        "swatch_ring": "rgba(233,236,223,.28)",
    },
}

# One hue per workflow panel, in the order of the published figure. Every hue
# is used as type (a stage number, a lit spine seed), so each is held to at
# least 4.5:1 against its own card.
_STAGES = {
    "light": {
        "acquire": "#33648F",
        "detect": "#3F7A48",
        "segment": "#96651F",
        "phenotype": "#5B7128",
        "physical": "#6B5A9E",
        "reliability": "#A64C7B",
        "validation": "#2F7A88",
        "outputs": "#B35A22",
    },
    "dark": {
        "acquire": "#86B1DC",
        "detect": "#83C48F",
        "segment": "#E0AC5A",
        "phenotype": "#A5C25A",
        "physical": "#B4A2E0",
        "reliability": "#E390BD",
        "validation": "#74BFD0",
        "outputs": "#F0A365",
    },
}

# ---------------------------------------------------------------------------
# module-level constants — the light values are the import-time default, and
# `bind()` overwrites them (here and in the modules that captured them).
# ---------------------------------------------------------------------------

PAPER = _SURFACE["light"]["paper"]
CARD = _SURFACE["light"]["card"]
INK = _SURFACE["light"]["ink"]
LINE = _SURFACE["light"]["line"]
MUTED = _SURFACE["light"]["muted"]

STAGE_COLORS = dict(_STAGES["light"])
ACCENT = _SURFACE["light"]["accent"]
FLAG = _SURFACE["light"]["flag"]

# Detector classes. These are drawn onto the tray photographs themselves, so
# they are fixed: a seed's coat colour does not depend on the reader's theme.
# Warm browns for the coat types, red only for defects.
CLASS_COLORS = {
    "Black seeds": "#2B2B2B",
    "Defective": "#B8432A",
    "Dotted seeds": "#8E5B3A",
    "Marbled seeds": "#B08968",
    "Spotted seeds": "#D4A24C",
    "Unspotted seeds": "#E0C894",
}
DEFAULT_CLASS_COLOR = "#6E7F76"

PATTERN_COLORS = {
    "plain": "#E0C894",
    "dotted": "#8E5B3A",
    "spotted": "#D4A24C",
    "marbled": "#B08968",
    "black": "#2B2B2B",
}


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def class_colors_bgr() -> dict[str, tuple[int, int, int]]:
    return {k: hex_to_bgr(v) for k, v in CLASS_COLORS.items()}


def class_color(name) -> str:
    """Case-insensitive lookup, with a neutral for classes we don't publish."""
    key = str(name).strip().lower()
    for k, v in CLASS_COLORS.items():
        if k.lower() == key:
            return v
    return DEFAULT_CLASS_COLOR


# ---------------------------------------------------------------------------
# resolving and applying the palette
# ---------------------------------------------------------------------------

def active_mode() -> str:
    """
    Whichever surface the reader is actually on.

    Streamlit reports the resolved theme on `st.context.theme` from 1.46; older
    builds only know what the config file asked for.
    """
    try:
        kind = st.context.theme.type
        if kind in ("light", "dark"):
            return kind
    except Exception:                                  # noqa: BLE001
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:                                  # noqa: BLE001
        pass
    return "light"


def palette(mode: str = "light") -> dict[str, str]:
    surface = dict(_SURFACE.get(mode, _SURFACE["light"]))
    surface.update(_STAGES.get(mode, _STAGES["light"]))
    return surface


def bind(mode: str, *modules) -> dict[str, str]:
    """
    Point the Python-side colour constants at `mode` and return the palette.

    Pass every module that did `from .theme import INK, ...` — their globals are
    rebound in place. Anything reading `theme.STAGE_COLORS` needs no listing:
    that dict is updated, never replaced.
    """
    pal = palette(mode)
    named = {"PAPER": "paper", "CARD": "card", "INK": "ink",
             "LINE": "line", "MUTED": "muted", "ACCENT": "accent", "FLAG": "flag"}

    for namespace in (globals(), *(vars(m) for m in modules)):
        for const, key in named.items():
            if const in namespace:
                namespace[const] = pal[key]

    STAGE_COLORS.update(_STAGES.get(mode, _STAGES["light"]))
    return pal


# ---------------------------------------------------------------------------
# stylesheet
# ---------------------------------------------------------------------------

def css(mode: str = "light") -> str:
    p = palette(mode)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,500;12..96,600;12..96,700&display=swap');

:root {{
  --ink:{p['ink']}; --paper:{p['paper']}; --card:{p['card']};
  --line:{p['line']}; --muted:{p['muted']}; --sunk:{p['sunk']};
  --accent:{p['accent']}; --accent-ink:{p['accent_ink']}; --accent-hover:{p['accent_hover']};
  --flag:{p['flag']}; --grid:{p['grid']}; --ring:{p['ring']};
  --swatch-ring:{p['swatch_ring']};
  --s1:{p['acquire']}; --s2:{p['detect']}; --s3:{p['segment']}; --s4:{p['phenotype']};
  --s5:{p['physical']}; --s6:{p['reliability']}; --s7:{p['validation']}; --out:{p['outputs']};
  --font:'Bricolage Grotesque', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --r-card:10px; --r-ctl:6px;
}}

html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stButton button,
[data-testid="stSidebar"], .stTabs, .stSelectbox, .stTextInput, .stNumberInput,
.stRadio, .stSlider, .stCheckbox, .stExpander, [data-testid="stFileUploader"] {{
  font-family:var(--font);
}}
.stMarkdown, .stRadio label, .stCheckbox label {{ font-variant-numeric:tabular-nums; }}
[data-testid="stAppViewContainer"], .stApp {{ background:var(--paper); color:var(--ink); }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.6rem; padding-bottom:4rem; max-width:1180px; }}

:focus-visible {{ outline:2px solid var(--ring); outline-offset:2px; border-radius:4px; }}

/* masthead: a specimen card on graph paper ------------------------------- */
.mast {{ display:flex; align-items:center; gap:1.25rem; flex-wrap:wrap;
  position:relative; overflow:hidden; margin:0 0 1.2rem 0;
  padding:1.35rem 1.5rem; background:var(--card); border:1px solid var(--line);
  border-radius:var(--r-card); }}
.mast::before {{ content:""; position:absolute; inset:0; pointer-events:none;
  background-image:linear-gradient(var(--grid) 1px, transparent 1px),
                   linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size:10px 10px; background-position:-1px -1px;
  -webkit-mask-image:linear-gradient(90deg, rgba(0,0,0,.9), rgba(0,0,0,0) 62%);
          mask-image:linear-gradient(90deg, rgba(0,0,0,.9), rgba(0,0,0,0) 62%); }}
.mast > * {{ position:relative; }}
.mast .mark {{ width:52px; height:52px; flex:none; color:var(--ink); }}
.mast .mark svg, .mast .mark img {{ width:52px; height:52px; display:block; }}
.mast h1 {{ font-size:2.05rem; font-weight:600; letter-spacing:-.025em;
  font-variation-settings:'opsz' 96; color:var(--ink); margin:0; line-height:1.05; }}
.mast p {{ margin:.35rem 0 0 0; color:var(--muted); font-size:.95rem; line-height:1.45;
  max-width:58ch; }}
.mast .grow {{ flex:1; }}

/* status pills ------------------------------------------------------------ */
.chipline {{ display:flex; gap:.4rem; flex-wrap:wrap; align-items:center;
  font-size:.8rem; margin-bottom:.4rem; }}
.chip {{ border:1px solid var(--line); border-radius:999px; padding:.22rem .7rem;
  background:var(--card); color:var(--muted); white-space:nowrap; }}
.chip b {{ color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums; }}
.chip.live {{ border-color:var(--accent); color:var(--accent); }}
.chip.live b {{ color:var(--accent); }}
.chip.warn {{ border-color:var(--flag); color:var(--flag); }}
.chip.warn b {{ color:var(--flag); }}
.chip.off {{ border-style:dashed; }}

/* the three steps --------------------------------------------------------- */
.steps {{ display:flex; gap:.5rem; margin:.9rem 0 1.5rem 0; }}
.steps .s {{ flex:1; display:flex; align-items:center; gap:.6rem; padding:.55rem .85rem;
  border:1px solid var(--line); border-radius:var(--r-ctl); background:transparent;
  color:var(--muted); }}
.steps .s.done {{ border-color:var(--accent); background:var(--accent); color:var(--accent-ink); }}
.steps .s.now {{ border-color:var(--accent); border-width:2px; background:var(--card);
  color:var(--ink); }}
.steps .n {{ width:1.4rem; height:1.4rem; border-radius:50%; flex:none;
  display:grid; place-items:center; font-size:.78rem; font-weight:600;
  font-variant-numeric:tabular-nums; border:1.5px solid currentColor; }}
.steps .s.done .n {{ background:var(--accent-ink); color:var(--accent); border-color:transparent; }}
.steps .s.now .n {{ background:var(--accent); color:var(--accent-ink); border-color:transparent; }}
.steps .t {{ font-size:.9rem; line-height:1.25; }}
.steps .s.now .t {{ font-weight:600; }}
.steps .s.done .t {{ font-weight:500; }}

/* the seven-stage spine: seeds on a line ---------------------------------- */
.spine {{ display:flex; gap:0; margin:1.1rem 0 1.4rem 0; position:relative; }}
.spine::before {{ content:""; position:absolute; left:8px; right:8px; top:8px; height:2px;
  background:var(--line); }}
.spine .p {{ flex:1; position:relative; min-width:0; padding:1.35rem .6rem 0 0; }}
.spine .p::before {{ content:""; position:absolute; top:2px; left:0; width:14px; height:14px;
  border-radius:50%; background:var(--paper); border:2px solid var(--line);
  box-sizing:border-box; }}
.spine .p.done::before {{ background:var(--c); border-color:var(--c); }}
.spine .p.now::before {{ border-color:var(--c); border-width:3px; }}
.spine .num {{ display:none; }}
.spine .lab {{ font-size:.82rem; color:var(--muted); line-height:1.3; }}
.spine .p.done .lab {{ color:var(--ink); }}
.spine .p.now .lab {{ color:var(--c); font-weight:600; }}

/* readings ---------------------------------------------------------------- */
.readings {{ display:flex; flex-wrap:wrap; gap:1.2rem 2.2rem; margin:.2rem 0 1.3rem 0; }}
.reading {{ flex:0 1 auto; min-width:7.5rem; }}
.reading .val {{ font-size:2rem; font-weight:600; letter-spacing:-.02em;
  font-variation-settings:'opsz' 72; color:var(--ink); font-variant-numeric:tabular-nums;
  line-height:1.1; }}
.reading .lab {{ font-size:.82rem; color:var(--muted); margin-top:.15rem; }}

/* panel heading ----------------------------------------------------------- */
.phead {{ display:flex; align-items:center; gap:.65rem; margin:.4rem 0 .15rem 0; }}
.phead .n {{ width:1.55rem; height:1.55rem; border-radius:50%; display:grid; place-items:center;
  font-size:.8rem; font-weight:600; color:#fff; background:var(--c);
  font-variant-numeric:tabular-nums; flex:none; }}
.phead h3 {{ font-size:1.18rem; font-weight:600; letter-spacing:-.01em; color:var(--ink);
  margin:0; font-variation-settings:'opsz' 32; }}
.psub {{ color:var(--muted); font-size:.9rem; margin:.25rem 0 1rem 0; max-width:70ch;
  line-height:1.45; }}

/* note -------------------------------------------------------------------- */
.note {{ background:var(--sunk); border-radius:var(--r-ctl); padding:.6rem .85rem .6rem 2rem;
  position:relative; color:var(--muted); font-size:.88rem; line-height:1.45;
  margin:.4rem 0 1rem 0; max-width:78ch; }}
.note::before {{ content:""; position:absolute; left:.85rem; top:.95rem; width:8px; height:8px;
  border-radius:50%; background:var(--accent); }}
.note b, .note code {{ color:var(--ink); font-weight:600; }}
.note.flag::before {{ background:var(--flag); }}
.note.flag {{ color:var(--ink); }}

/* verdict rows ------------------------------------------------------------ */
.verdict {{ display:flex; gap:.7rem; align-items:flex-start; padding:.6rem .85rem;
  background:var(--card); border:1px solid var(--line); border-radius:var(--r-ctl);
  margin-bottom:.5rem; font-size:.88rem; color:var(--ink); line-height:1.45; }}
.verdict::before {{ content:""; width:9px; height:9px; border-radius:50%; flex:none;
  background:var(--c); margin-top:.38rem; }}
.verdict .k {{ color:var(--muted); min-width:9.5rem; font-weight:500; }}

/* tray caption ------------------------------------------------------------ */
.cap {{ display:flex; justify-content:space-between; gap:.5rem;
  padding:.35rem .1rem .9rem .1rem; font-size:.8rem; color:var(--muted); }}
.cap b {{ color:var(--ink); font-weight:600; }}

/* photographs waiting to be measured -------------------------------------- */
.tray-grid {{ display:grid; gap:12px; margin:.9rem 0 1.2rem 0;
  grid-template-columns:repeat(auto-fill, minmax(190px, 1fr)); }}
.tray {{ border:1px solid var(--line); background:var(--card); border-radius:var(--r-card);
  overflow:hidden; }}
.tray img {{ width:100%; display:block; aspect-ratio:4/3; object-fit:cover;
  background:var(--sunk); }}
.tray .f {{ padding:.45rem .7rem; font-size:.78rem; color:var(--muted);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

/* class composition ------------------------------------------------------- */
.class-grid {{ display:grid; gap:10px; margin:.5rem 0 1.4rem 0;
  grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); }}
.class-tile {{ padding:.75rem .9rem .8rem .9rem; min-width:0; background:var(--card);
  border:1px solid var(--line); border-radius:var(--r-card); }}
.class-tile .name {{ display:flex; align-items:center; gap:.45rem; font-size:.82rem;
  color:var(--muted); line-height:1.35; }}
.class-tile .swatch {{ width:11px; height:11px; flex:none; border-radius:50%;
  box-shadow:inset 0 0 0 1px var(--swatch-ring); }}
.class-tile .count {{ font-size:1.7rem; font-weight:600; letter-spacing:-.02em;
  color:var(--ink); font-variant-numeric:tabular-nums; line-height:1.2; margin-top:.2rem; }}
.class-tile .pct {{ font-size:.78rem; color:var(--muted); font-variant-numeric:tabular-nums; }}
.class-tile .bar {{ height:5px; border-radius:999px; background:var(--sunk); margin-top:.55rem;
  overflow:hidden; }}
.class-tile .bar i {{ display:block; height:5px; border-radius:999px; background:var(--c); }}

/* seed contact sheet: each crop is a seed, so it is round ----------------- */
.seed-grid {{ display:grid; gap:10px; margin:.5rem 0 1rem 0;
  grid-template-columns:repeat(auto-fill, minmax(88px, 1fr)); }}
.seed {{ aspect-ratio:1/1; border-radius:50%; background:var(--sunk);
  border:3px solid var(--c); padding:6px; overflow:hidden; }}
.seed img {{ width:100%; height:100%; object-fit:cover; display:block; border-radius:50%; }}

/* Streamlit chrome -------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {{ background:var(--sunk);
  border:1.5px dashed var(--line); border-radius:var(--r-card); }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:var(--accent); }}
[data-testid="stSidebar"] {{ border-right:1px solid var(--line); }}
[data-testid="stMetricValue"], .stDataFrame {{ font-variant-numeric:tabular-nums; }}
[data-testid="stExpander"] details {{ border:1px solid var(--line);
  border-radius:var(--r-card); background:var(--card); }}
[data-testid="stSegmentedControl"] button {{ border-radius:999px; }}

.stTabs [data-baseweb="tab-list"] {{ gap:.1rem; border-bottom:1px solid var(--line);
  overflow-x:auto; overflow-y:hidden; flex-wrap:nowrap;
  scrollbar-width:none; -ms-overflow-style:none; }}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display:none; }}
.stTabs [data-baseweb="tab"] {{ font-size:.92rem; font-weight:500; padding:.5rem .9rem;
  white-space:nowrap; color:var(--muted); }}
.stTabs [aria-selected="true"] {{ color:var(--ink); }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--accent); height:2px; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}

.stButton button {{ border-radius:999px; font-weight:600; padding:.45rem 1.1rem; }}
.stButton button[kind="primary"] {{ background:var(--accent); border-color:var(--accent);
  color:var(--accent-ink); }}
.stButton button[kind="primary"]:hover {{ background:var(--accent-hover);
  border-color:var(--accent-hover); }}
.stButton button[kind="secondary"] {{ background:var(--card); border-color:var(--line);
  color:var(--ink); }}
.stButton button[kind="secondary"]:hover {{ border-color:var(--accent); color:var(--accent); }}
.stDownloadButton button {{ border-radius:999px; font-weight:600; }}

/* narrow screens ---------------------------------------------------------- */
@media (max-width: 720px) {{
  .block-container {{ padding-top:1rem; padding-left:1rem; padding-right:1rem; }}
  .mast {{ padding:1rem 1.1rem; gap:.9rem; }}
  .mast .mark, .mast .mark svg, .mast .mark img {{ width:40px; height:40px; }}
  .mast h1 {{ font-size:1.5rem; }}
  .mast p {{ font-size:.88rem; }}

  .steps {{ gap:.35rem; }}
  .steps .s {{ padding:.45rem .5rem; gap:.4rem; }}
  .steps .t {{ font-size:.78rem; }}

  /* Seven seeds on a line is still legible on a phone; the labels are not,
     so only the current stage keeps its label. */
  .spine .p {{ padding-right:.2rem; }}
  .spine .lab {{ display:none; }}
  .spine .p.now .lab {{ display:block; font-size:.78rem; }}

  .readings {{ gap:.9rem 1.4rem; }}
  .reading .val {{ font-size:1.5rem; }}
  .tray-grid {{ grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:8px; }}
  .seed-grid {{ grid-template-columns:repeat(auto-fill, minmax(64px, 1fr)); gap:7px; }}
  .class-grid {{ grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:8px; }}
  .verdict {{ flex-wrap:wrap; }}
  .verdict .k {{ min-width:0; }}
}}

@media (prefers-reduced-motion: reduce) {{
  * {{ animation:none !important; transition:none !important; }}
}}
</style>
"""


# Kept so anything still doing `from .theme import CSS` keeps working; new code
# should call `css(mode)`.
CSS = css("light")
