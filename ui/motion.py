"""
Motion.

Animation earns its place here in three ways and no others:

  * it shows the app doing its job — the hero traces a seed and measures it,
    the loader drops seeds into a tray and scans them,
  * it shows progress — the stage spine advances, the progress bar sweeps,
  * it shows what changed — results rise in, headline numbers count up, a tray
    photograph wipes in behind a scanning line.

Nothing loops forever except two slow, quiet things: the mark re-measures
itself every seven seconds, and the accent dot breathes. Everything else runs
once and stops. Every animation is disabled under `prefers-reduced-motion`.

The counters use CSS `@property`, so a browser without it simply shows the
final number immediately, which is the right fallback.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st

from .components import asset_path

MOTION_CSS = """
<style>
/* ---- headline counters ------------------------------------------------- */
@property --tick { syntax: '<integer>'; initial-value: 0; inherits: false; }

.count { counter-reset: tick var(--tick); animation: tickup 1.1s .15s cubic-bezier(.2,.8,.3,1) forwards; }
.count::after { content: counter(tick); }
.count.pct::after { content: counter(tick) '%'; }
@keyframes tickup { from { --tick: 0; } to { --tick: var(--target); } }

/* ---- entrances --------------------------------------------------------- */
@keyframes rise   { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
@keyframes wipe   { from { clip-path:inset(0 100% 0 0); } to { clip-path:inset(0 0 0 0); } }
@keyframes grow   { from { transform:scaleY(0); } to { transform:scaleY(1); } }
@keyframes shimmer{ from { background-position:-220% 0; } to { background-position:220% 0; } }
@keyframes breathe{ 0%,100% { box-shadow:0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent); }
                    50%      { box-shadow:0 0 0 7px color-mix(in srgb, var(--accent) 5%, transparent); } }
@keyframes sweepline { 0% { left:0; opacity:0; } 8% { opacity:1; }
                       92% { opacity:1; } 100% { left:100%; opacity:0; } }

/* readings: staggered rise */
.readings .reading { animation: rise .5s ease-out both; }
.readings .reading:nth-child(1) { animation-delay:.04s; }
.readings .reading:nth-child(2) { animation-delay:.11s; }
.readings .reading:nth-child(3) { animation-delay:.18s; }
.readings .reading:nth-child(4) { animation-delay:.25s; }
.readings .reading:nth-child(5) { animation-delay:.32s; }

/* a tray photograph wipes in behind a scanning line */
[data-testid="stImage"] { position:relative; }
[data-testid="stImage"] img { animation: wipe .7s cubic-bezier(.4,0,.2,1) both; }
[data-testid="stImage"]::after {
  content:""; position:absolute; top:0; bottom:0; width:2px; background:var(--accent);
  box-shadow:0 0 12px 2px color-mix(in srgb, var(--accent) 45%, transparent);
  animation: sweepline .75s cubic-bezier(.4,0,.2,1) 1 forwards; }

/* the run's progress bar reads as a measurement in progress, not a download */
[data-testid="stProgress"] div[role="progressbar"] > div {
  background-image:linear-gradient(90deg, var(--accent) 0%, var(--accent-hover) 45%, var(--accent) 90%);
  background-size:220% 100%; animation: shimmer 1.4s linear infinite; }

/* the active seed on the spine pulses while its stage runs; a finished one pops full */
@keyframes seedpulse { 0%,100% { transform:scale(1); } 50% { transform:scale(1.35); } }
@keyframes seedpop   { from { transform:scale(.4); } to { transform:scale(1); } }
.spine .p.now::before  { animation: seedpulse 1.2s ease-in-out infinite; }
.spine .p.done::before { animation: seedpop .35s cubic-bezier(.2,1.4,.4,1) both; }

/* panels and verdicts arrive rather than appear */
.phead, .psub { animation: rise .45s ease-out both; }
.verdict { animation: rise .4s ease-out both; }

/* ---- hero -------------------------------------------------------------- */
.hero { display:flex; align-items:center; gap:1.4rem; flex-wrap:wrap;
  border:1px solid var(--line); border-radius:var(--r-card); background:var(--card);
  padding:1.3rem 1.5rem; margin-bottom:1rem; position:relative; overflow:hidden; }
.hero .mark { width:78px; height:78px; flex:none; }
.hero .mark img, .hero .mark svg { width:78px; height:78px; display:block; }
.hero .copy { flex:1 1 340px; }
.hero h1 { font-size:2.05rem; font-weight:600; letter-spacing:-.025em; color:var(--ink);
  font-variation-settings:'opsz' 96;
  margin:0; line-height:1.1; animation: rise .5s .1s ease-out both; }
.hero p { margin:.3rem 0 0 0; color:var(--muted); font-size:.92rem; max-width:60ch;
  animation: rise .5s .2s ease-out both; }
.hero .dots { display:flex; gap:.42rem; margin-top:.7rem; }
.hero .dots i { width:9px; height:9px; border-radius:50%; display:block;
  animation: rise .35s ease-out both; }

/* ---- loader ------------------------------------------------------------ */
.loader { display:flex; align-items:center; gap:1rem; padding:.7rem 0 .2rem 0; }
.loader img, .loader svg { width:230px; height:86px; flex:none; }
.loader .say { font-size:.9rem; color:var(--ink); }
.loader .say b { font-weight:600; }
.loader .say span { display:block; color:var(--muted); font-size:.82rem; margin-top:.15rem;
  font-variant-numeric:tabular-nums; }

/* ---- live status dot ---------------------------------------------------- */
.chip.live::before { content:""; display:inline-block; width:7px; height:7px; border-radius:50%;
  background:var(--accent); margin-right:.4rem; vertical-align:middle;
  animation: breathe 2.8s ease-in-out infinite; }

@media (prefers-reduced-motion: reduce) {
  *, .hero h1, .hero p, .readings .reading, .chip.live::before,
  .spine .p::before { animation:none !important; }
  [data-testid="stImage"] img { clip-path:none; }
  [data-testid="stImage"]::after { display:none; }
  .count::after { content: var(--fallback, attr(data-value)); }
}
</style>
"""


def inject() -> None:
    """Add the motion layer. Call once, after the theme CSS."""
    st.markdown(MOTION_CSS, unsafe_allow_html=True)


def _art(name: str, width: int, height: int, css_class: str = "") -> str:
    """
    An SVG asset as a sized `<img>`, with a PNG fallback.

    The markup is not inlined. Streamlit sanitises the HTML it renders, and a
    `<style>` block inside an inlined SVG is one of the things that can be
    dropped — which takes the animation with it and leaves a shape with no
    fills. An SVG loaded through `<img>` is a self-contained document: its own
    stylesheet and keyframes run normally, and nothing can reach in and strip
    them. Explicit width and height keep a missing file from collapsing the
    layout around it.
    """
    for candidate, mime in ((name, "image/svg+xml"),
                            (name.replace(".svg", "_96.png"), "image/png")):
        p: Path | None = asset_path(candidate)
        if not p:
            continue
        data = base64.b64encode(p.read_bytes()).decode()
        cls = f' class="{css_class}"' if css_class else ""
        return (f'<img{cls} src="data:{mime};base64,{data}" alt="" '
                f'width="{width}" height="{height}" '
                f'style="width:{width}px;height:{height}px;display:block">')
    return ""


# ---------------------------------------------------------------------------
# hero
# ---------------------------------------------------------------------------

_STAGE_DOTS = ["#33648F", "#3F7A48", "#96651F", "#5B7128", "#6B5A9E", "#A64C7B", "#2F7A88"]


def hero(title: str, subtitle: str) -> None:
    """
    The masthead: the mark traces a seed, measures it, and the seven panels
    light along the bottom in the order the run will follow.
    """
    mark = _art("logo.svg", 78, 78)
    dots = "".join(
        f'<i style="background:{c};animation-delay:{0.35 + i * 0.07:.2f}s"></i>'
        for i, c in enumerate(_STAGE_DOTS)
    )
    st.markdown(
        '<div class="hero">'
        + (f'<span class="mark">{mark}</span>' if mark else "")
        + f'<div class="copy"><h1>{html.escape(title)}</h1>'
          f'<p>{subtitle}</p><div class="dots">{dots}</div></div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

def loader(headline: str, detail: str = "") -> None:
    """Seeds falling into a tray under a scan line, in place of a spinner."""
    art = _art("loader_seeds.svg", 230, 86)
    st.markdown(
        f'<div class="loader">{art}<div class="say"><b>{html.escape(headline)}</b>'
        + (f"<span>{html.escape(detail)}</span>" if detail else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# counting readings
# ---------------------------------------------------------------------------

def _countable(value: str) -> tuple[int, str, str] | None:
    """
    Decide whether a headline value can count up.

    Only plain integers and whole percentages do. A number with a decimal point
    or a unit reads worse mid-count than it does arriving whole, so those are
    left alone.
    """
    v = value.strip()
    if "," in v:
        # A thousands separator would be lost by counter(), and 1204 reads worse
        # than 1,204 for the one number people quote from this row.
        return None
    if v.endswith("%") and v[:-1].isdigit():
        return int(v[:-1]), "pct", value
    if v.isdigit():
        return int(v), "", value
    return None


def readings(items: list[tuple[str, str]], stage: str = "outputs") -> None:
    """
    The headline row, with whole numbers counting up to their value.

    Same markup as `components.readings`, so the theme's styling applies; this
    version just swaps countable values for an animated counter.
    """
    from .theme import STAGE_COLORS

    cells = []
    for value, label in items:
        counted = _countable(value)
        if counted:
            target, extra, original = counted
            inner = (f'<span class="count {extra}" style="--target:{target}" '
                     f'data-value="{html.escape(original)}"></span>')
        else:
            inner = html.escape(value)
        cells.append(
            f'<div class="reading"><div class="val">{inner}</div>'
            f'<div class="lab">{html.escape(label)}</div></div>'
        )
    colour = STAGE_COLORS.get(stage, "#12211D")
    st.markdown(f'<div class="readings" style="--c:{colour}">{"".join(cells)}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# animated shape atlas
# ---------------------------------------------------------------------------

def atlas_animation(fig, n_curves: int, frame_ms: int = 55):
    """
    Add Play/Pause to a shape-atlas figure whose traces are already attached.

    The outlines stack on one at a time, so a bundle that looks solid when
    finished shows you how tight it actually is while it builds.
    """
    from .theme import CARD, INK, LINE

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.06, x=0.01, xanchor="left",
            bgcolor=CARD, bordercolor=LINE, font=dict(color=INK, size=11),
            buttons=[
                dict(label="Draw the atlas", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                      transition=dict(duration=0), fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False))]),
            ],
        )],
    )
    return fig
