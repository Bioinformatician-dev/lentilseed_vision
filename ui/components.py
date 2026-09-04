"""
Small render helpers shared by every panel.

Each one writes markup and returns nothing; the panels stay readable because
none of them contain a wall of f-strings.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from seedvision.runner import STAGES

from .theme import CARD, INK, LINE, MUTED, STAGE_COLORS

ASSET_DIRS: tuple[Path, ...] = ()


def set_asset_dirs(*dirs: Path) -> None:
    global ASSET_DIRS
    ASSET_DIRS = tuple(dirs)


def asset_path(name: str) -> Path | None:
    for d in ASSET_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def asset_image(name: str):
    p = asset_path(name)
    return Image.open(p) if p else None


def _inline_svg(name: str) -> str:
    p = asset_path(name)
    return p.read_text() if p else ""


def _data_uri(name: str) -> str:
    p = asset_path(name)
    if not p:
        return ""
    return f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"


# ---------------------------------------------------------------------------
# masthead & status
# ---------------------------------------------------------------------------

def masthead(title: str, subtitle: str) -> None:
    mark = _inline_svg("logo.svg")
    if not mark:
        uri = _data_uri("logo_96.png")
        mark = f'<img src="{uri}" alt="">' if uri else ""
    st.markdown(
        '<div class="mast">'
        + (f'<span class="mark">{mark}</span>' if mark else "")
        + f"<div><h1>{title}</h1><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def chipline(chips: list[tuple[str, str, str]]) -> None:
    """Each chip is (label, value, kind) where kind is '', 'live', 'warn' or 'off'."""
    html = "".join(
        f'<span class="chip {kind}">{label} <b>{value}</b></span>'
        for label, value, kind in chips
    )
    st.markdown(f'<div class="chipline">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# the seven-panel spine
# ---------------------------------------------------------------------------

def spine(active: str | None, done: set[str]) -> str:
    cells = []
    for i, (key, label) in enumerate(STAGES, start=1):
        cls = "p"
        if key in done:
            cls += " done"
        if key == active:
            cls += " now"
        colour = STAGE_COLORS.get(key, INK)
        cells.append(
            f'<div class="{cls}" style="--c:{colour}">'
            f'<div class="num">{i}</div><div class="lab">{label}</div></div>'
        )
    return f'<div class="spine">{"".join(cells)}</div>'


def show_spine(active: str | None, done: set[str], slot=None) -> None:
    target = slot or st
    target.markdown(spine(active, done), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# headings, notes, readings
# ---------------------------------------------------------------------------

def panel_head(number: int, title: str, subtitle: str, stage: str) -> None:
    colour = STAGE_COLORS.get(stage, INK)
    st.markdown(
        f'<div class="phead" style="--c:{colour}"><span class="n">{number}</span>'
        f"<h3>{title}</h3></div><p class='psub'>{subtitle}</p>",
        unsafe_allow_html=True,
    )


def note(text: str, flag: bool = False) -> None:
    st.markdown(f'<div class="note{" flag" if flag else ""}">{text}</div>',
                unsafe_allow_html=True)


def readings(items: list[tuple[str, str]], stage: str = "segment") -> None:
    colour = STAGE_COLORS.get(stage, INK)
    body = "".join(
        f'<div class="reading"><div class="val">{v}</div><div class="lab">{lab}</div></div>'
        for v, lab in items
    )
    st.markdown(f'<div class="readings" style="--c:{colour}">{body}</div>',
                unsafe_allow_html=True)


def verdict(key: str, text: str, stage: str) -> None:
    colour = STAGE_COLORS.get(stage, INK)
    st.markdown(
        f'<div class="verdict" style="--c:{colour}"><span class="k">{key}</span>'
        f"<span>{text}</span></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tray_card(name: str, vis: np.ndarray, caption: str) -> None:
    st.image(bgr_to_rgb(vis), width="stretch")
    st.markdown(f'<div class="cap"><span>{name}</span><span>{caption}</span></div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def style_fig(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, title=title, margin=dict(t=46 if title else 16, r=12, b=12, l=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12, family="Bricolage Grotesque, sans-serif"),
        title_font=dict(size=13.5, color=INK),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        legend=dict(bgcolor=CARD, bordercolor=LINE, borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(color=INK)),
    )
    return fig


def empty_note(message: str) -> None:
    st.markdown(f'<div class="note">{message}</div>', unsafe_allow_html=True)


def hline() -> None:
    st.markdown(f'<hr style="border:none;border-top:1px solid {LINE};margin:1.1rem 0">',
                unsafe_allow_html=True)


def muted_caption(text: str) -> None:
    st.markdown(f'<div style="color:{MUTED};font-size:.8rem">{text}</div>',
                unsafe_allow_html=True)
