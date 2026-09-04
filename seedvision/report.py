"""
The run as a single HTML file.

One file, no folder of assets beside it: the annotated trays are embedded as
data URIs and the charts are drawn as inline SVG, so the report can be emailed,
attached to a supervisor's message, or opened from a USB stick years later and
still look the same. It prints to PDF cleanly from any browser.

Charts are drawn here rather than by a plotting library, because a report that
depends on Plotly's CDN is a report that goes blank the first time it is opened
without a network.

    Path("report.html").write_bytes(build_html(result, validation))
"""
from __future__ import annotations

import base64
import html
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd

from .config import COLOUR_TRAITS, PATTERN_TRAITS
from .export import clean, data_dictionary_frame

INK = "#12211D"
MUTED = "#6F7B76"
LINE = "#E2E6E3"
PAPER = "#FAFBFA"
STAGE = {
    "acquire": "#2E6FA8", "detect": "#3C8B57", "segment": "#C08430",
    "phenotype": "#4E9A51", "physical": "#7659A6", "reliability": "#B24C87",
    "validation": "#2E86A0", "outputs": "#D2762C",
}


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------

def _esc(x) -> str:
    return html.escape(str(x))


def _jpeg_uri(img: np.ndarray, max_width: int = 1400, quality: int = 82) -> str:
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


def _table(df: pd.DataFrame, digits: int = 3, max_rows: int | None = 60) -> str:
    if df is None or df.empty:
        return '<p class="muted">Nothing to show here.</p>'
    d = df.copy()
    truncated = max_rows is not None and len(d) > max_rows
    if truncated:
        d = d.head(max_rows)
    d = d.round(digits)

    head = "".join(f"<th>{_esc(c)}</th>" for c in d.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        for row in d.itertuples(index=False)
    )
    note = (f'<p class="muted">First {max_rows} of {len(df):,} rows. '
            "The full table is in the CSV export.</p>") if truncated else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>{note}"


def _histogram(values: pd.Series, label: str, colour: str,
               width: int = 620, height: int = 190, bins: int = 30) -> str:
    """A histogram as inline SVG — no library, no network."""
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 3:
        return ""
    counts, edges = np.histogram(v, bins=bins)
    if counts.max() == 0:
        return ""

    pad_l, pad_b, pad_t = 42, 26, 10
    plot_w, plot_h = width - pad_l - 12, height - pad_b - pad_t
    bar_w = plot_w / len(counts)

    bars = "".join(
        f'<rect x="{pad_l + i * bar_w:.1f}" y="{pad_t + plot_h - c / counts.max() * plot_h:.1f}" '
        f'width="{max(1.0, bar_w - 1):.1f}" height="{c / counts.max() * plot_h:.1f}" '
        f'fill="{colour}" opacity="0.85"/>'
        for i, c in enumerate(counts)
    )
    mean_x = pad_l + (v.mean() - edges[0]) / (edges[-1] - edges[0]) * plot_w
    ticks = "".join(
        f'<text x="{pad_l + f * plot_w:.0f}" y="{height - 8}" text-anchor="middle" '
        f'class="tick">{edges[0] + f * (edges[-1] - edges[0]):.2f}</text>'
        for f in (0, 0.5, 1.0)
    )
    return f"""
<figure>
  <svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{_esc(label)}">
    <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - 12}" y2="{pad_t + plot_h}"
          stroke="{LINE}"/>
    {bars}
    <line x1="{mean_x:.1f}" y1="{pad_t}" x2="{mean_x:.1f}" y2="{pad_t + plot_h}"
          stroke="{INK}" stroke-dasharray="3 3"/>
    <text x="{mean_x:.1f}" y="{pad_t - 1}" text-anchor="middle" class="tick">mean</text>
    <text x="4" y="{pad_t + 10}" class="tick">{counts.max()}</text>
    <text x="4" y="{pad_t + plot_h}" class="tick">0</text>
    {ticks}
  </svg>
  <figcaption>{_esc(label)} — n = {len(v):,}, mean {v.mean():.3f}, SD {v.std(ddof=1):.3f}</figcaption>
</figure>"""


def _bars(counts: pd.Series, label: str, colour: str, width: int = 620) -> str:
    if counts is None or counts.empty:
        return ""
    rows = []
    top = counts.max()
    for name, n in counts.items():
        pct = 100 * n / counts.sum()
        rows.append(
            f'<div class="bar"><span class="bl">{_esc(name)}</span>'
            f'<span class="bt"><i style="width:{100 * n / top:.1f}%;background:{colour}"></i></span>'
            f'<span class="bn">{n:,} · {pct:.0f}%</span></div>'
        )
    return f'<div class="bars" aria-label="{_esc(label)}">{"".join(rows)}</div>'


def _readings(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="reading"><div class="val">{_esc(v)}</div>'
        f'<div class="lab">{_esc(lab)}</div></div>' for v, lab in items
    )
    return f'<div class="readings">{cells}</div>'


def _section(number: int | None, title: str, stage: str, body: str,
             blurb: str = "") -> str:
    colour = STAGE.get(stage, INK)
    tag = f'<span class="n" style="background:{colour}">{number}</span>' if number else ""
    sub = f'<p class="sub">{blurb}</p>' if blurb else ""
    return (f'<section><h2>{tag}{_esc(title)}</h2>{sub}{body}</section>')


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

CSS = f"""
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --paper:{PAPER}; }}
* {{ box-sizing:border-box; }}
body {{ font-family:'IBM Plex Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  color:var(--ink); background:var(--paper); margin:0; padding:2.2rem 1.4rem 4rem;
  line-height:1.55; }}
main {{ max-width:900px; margin:0 auto; }}
h1 {{ font-size:1.7rem; font-weight:600; letter-spacing:-.02em; margin:0 0 .2rem; }}
h2 {{ font-size:1.05rem; font-weight:600; margin:0 0 .2rem; display:flex; align-items:center;
  gap:.55rem; }}
h2 .n {{ color:#fff; font-family:'IBM Plex Mono',monospace; font-size:.75rem;
  border-radius:2px; padding:.05rem .38rem; }}
.lede {{ color:var(--muted); margin:.1rem 0 1.4rem; }}
.sub {{ color:var(--muted); font-size:.9rem; margin:.1rem 0 .8rem; max-width:74ch; }}
section {{ background:#fff; border:1px solid var(--line); border-radius:4px;
  padding:1.1rem 1.2rem; margin:0 0 1rem; }}
.readings {{ display:flex; flex-wrap:wrap; margin:.2rem 0 1.2rem; }}
.reading {{ flex:1 1 140px; padding:.1rem 1rem; border-left:1px solid var(--line); }}
.reading:first-child {{ border-left:2px solid {STAGE['outputs']}; }}
.reading .val {{ font-family:'IBM Plex Mono',monospace; font-size:1.45rem; font-weight:500;
  font-variant-numeric:tabular-nums; }}
.reading .lab {{ font-size:.76rem; color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-size:.8rem;
  font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:.32rem .5rem; border-bottom:1px solid var(--line);
  white-space:nowrap; }}
th {{ color:var(--muted); font-weight:500; font-size:.74rem; }}
.scroll {{ overflow-x:auto; }}
.muted {{ color:var(--muted); font-size:.82rem; }}
figure {{ margin:.6rem 0 1rem; }}
figcaption {{ color:var(--muted); font-size:.78rem; margin-top:.2rem; }}
.tick {{ font-family:'IBM Plex Mono',monospace; font-size:10px; fill:{MUTED}; }}
.bars {{ margin:.4rem 0 .8rem; }}
.bar {{ display:flex; align-items:center; gap:.6rem; font-size:.82rem; margin-bottom:.28rem; }}
.bl {{ width:11rem; }}
.bt {{ flex:1; background:var(--paper); border:1px solid var(--line); height:12px; }}
.bt i {{ display:block; height:100%; }}
.bn {{ width:8rem; text-align:right; font-family:'IBM Plex Mono',monospace;
  color:var(--muted); font-size:.76rem; }}
.tray {{ margin:0 0 1.2rem; }}
.tray img {{ width:100%; border:1px solid var(--line); border-radius:3px; display:block; }}
.tray .cap {{ display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace;
  font-size:.75rem; color:var(--muted); padding-top:.3rem; }}
.note {{ border-left:2px solid var(--line); padding:.1rem 0 .1rem .75rem; color:var(--muted);
  font-size:.85rem; margin:.4rem 0; max-width:78ch; }}
.flag {{ border-left-color:{STAGE['outputs']}; }}
footer {{ color:var(--muted); font-size:.78rem; text-align:center; margin-top:1.4rem; }}
@media print {{
  body {{ background:#fff; padding:0; }}
  section {{ break-inside:avoid; border:none; border-top:1px solid var(--line);
    border-radius:0; padding:.8rem 0; }}
  .tray {{ break-inside:avoid; }}
}}
"""


def build_html(result, validation: pd.DataFrame | None = None,
               title: str = "Lentil seed phenotyping",
               include_images: bool = True, max_table_rows: int = 60) -> bytes:
    """Render the whole run as one standalone HTML document."""
    from . import lotstats

    seeds = clean(result.seeds)
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    lot = result.config.get("lot_id", "lot")
    units = result.units
    parts: list[str] = []

    # ---- headline ---------------------------------------------------------
    parts.append(_readings(lotstats.headline(seeds, units)))
    if units == "px":
        parts.append('<p class="note flag">This run was uncalibrated, so every size is in '
                     'pixels. No volume, mass or thousand-seed weight was derived.</p>')

    # ---- 1 photographs ----------------------------------------------------
    qc = result.qc
    cols = [c for c in ["image", "megapixels", "focus", "clip_frac", "illum_cv",
                        "seed_width_px", "n_seeds", "size_grade", "qc_pass", "qc_reason"]
            if c in qc.columns]
    body = _table(qc[cols], 3, None)
    failed = qc[~qc.qc_pass] if "qc_pass" in qc.columns else pd.DataFrame()
    if not failed.empty:
        body += "".join(
            f'<p class="note flag"><b>{_esc(r.image)}</b> — {_esc(r.qc_reason)}</p>'
            for r in failed.itertuples()
        )
    parts.append(_section(1, "Image acquisition & quality control", "acquire", body,
                          "Whether each photograph was fit to measure, judged before "
                          "anything was measured from it."))

    # ---- 2 detection ------------------------------------------------------
    body = _table(result.detection, 3, None)
    if "cls" in seeds.columns:
        body += _bars(seeds["cls"].value_counts(), "seeds per class", STAGE["detect"])
    if include_images and result.annotated:
        for name, img in result.annotated.items():
            n = int((seeds.image == name).sum())
            uri = _jpeg_uri(img)
            body += (f'<div class="tray"><img src="{uri}" alt="{_esc(name)}">'
                     f'<div class="cap"><span>{_esc(name)}</span>'
                     f"<span>{n} seeds</span></div></div>")
    parts.append(_section(2, "YOLO seed detection", "detect", body,
                          "Every visible seed found, with the duplicates that finding "
                          "everything costs removed afterwards."))

    # ---- 3 outlines -------------------------------------------------------
    s = result.outline_summary
    body = ""
    if "segment_method" in seeds.columns:
        body += _bars(seeds.segment_method.value_counts(), "threshold used",
                      STAGE["segment"])
    if s:
        body += (f'<p class="note">Median outline error '
                 f'{s.get("median_error_pct", float("nan")):.2f}%, '
                 f'{s.get("flagged", 0)} of {len(seeds):,} outlines flagged for '
                 "inspection. The comparison is the traced area against the area an "
                 "ellipse of the same length and width would have.</p>")
    if "outline_error" in seeds.columns:
        body += _histogram(100 * seeds.outline_error, "Outline error (%)", STAGE["segment"])
    parts.append(_section(3, "Segmentation & contour extraction", "segment", body,
                          "Each seed thresholded inside its own detection box, so uneven "
                          "light across the tray cannot bias the outline."))

    # ---- 4 phenotypes -----------------------------------------------------
    body = ""
    primary = "length_mm" if units == "mm" and "length_mm" in seeds.columns else "length_px"
    for trait, colour in ((primary, STAGE["phenotype"]),
                          ("roundness", STAGE["phenotype"])):
        if trait in seeds.columns:
            body += _histogram(seeds[trait], trait, colour)
    if "pattern_class" in seeds.columns:
        body += _bars(seeds.pattern_class.value_counts(), "coat pattern",
                      STAGE["phenotype"])
        cross = lotstats.agreement_table(seeds)
        if not cross.empty:
            body += ('<p class="sub">Detector class against coat pattern. These are two '
                     "independent readings of the same seeds — the pattern is derived from "
                     "pixels alone — so they will not match exactly.</p>"
                     f'<div class="scroll">{_table(cross.reset_index(), 0, None)}</div>')
    parts.append(_section(4, "Image-derived phenotypes", "phenotype", body,
                          "Shape and size from the outline, colour from the coat, and the "
                          "markings on it."))

    # ---- 5 physical -------------------------------------------------------
    body = "".join(f'<p class="note">{_esc(n)}</p>' for n in result.notes)
    if result.tsw:
        t = result.tsw
        body += _readings([
            (f"{t['tsw_g']:.2f} g", "thousand-seed weight"),
            (f"± {t.get('tsw_ci95_g', float('nan')):.2f} g", "95% interval"),
            (f"{t['mean_mass_mg']:.2f} mg", "mean seed mass"),
            (f"{t.get('cv_mass_pct', float('nan')):.1f}%", "mass CV"),
        ])
    parts.append(_section(5, "Derived physical properties", "physical", body,
                          "What the millimetre columns rest on, stated rather than assumed."))

    # ---- 6 reliability ----------------------------------------------------
    body = _table(result.checks, 3, None)
    body += ('<p class="note">Two boxes on one seed inflate the count; one box across two '
             "deflates it. Both survive a confidence threshold, which is why they are "
             "counted rather than assumed away.</p>")
    parts.append(_section(6, "Reliability & self-consistency", "reliability", body,
                          "Whether the pipeline agrees with itself. No ground truth needed."))

    # ---- 7 validation -----------------------------------------------------
    if validation is not None and not validation.empty:
        body = _table(validation, 4, None)
        body += "".join(
            f'<p class="note"><b>{_esc(r.trait)}</b> — {_esc(r.verdict)}</p>'
            for r in validation.itertuples() if hasattr(r, "verdict")
        )
    else:
        body = ('<p class="note flag">No caliper measurements were supplied. Everything '
                "above is the pipeline checking its own work; only a caliper comparison "
                "can show the numbers are true.</p>")
    parts.append(_section(7, "External validation", "validation", body,
                          "Agreement with digital calipers on a measured sub-sample."))

    # ---- outputs ----------------------------------------------------------
    body = f'<div class="scroll">{_table(result.lot_stats, 3, None)}</div>'
    if not result.composition.empty:
        body += (f'<h3 class="sub">Composition</h3>'
                 f'<div class="scroll">{_table(result.composition, 1, None)}</div>')
    colour_cols = [c for c in [*COLOUR_TRAITS, *PATTERN_TRAITS] if c in seeds.columns]
    show = [c for c in ["image", "seed_id", "cls", "pattern_class"] if c in seeds.columns]
    show += [c for c in seeds.columns if c.endswith(("_mm", "_mm2", "_mm3"))][:6]
    show += [c for c in ["roundness", "solidity"] if c in seeds.columns]
    show += colour_cols[:2]
    body += (f'<h3 class="sub">Per-seed measurements</h3>'
             f'<div class="scroll">{_table(seeds[show], 3, max_table_rows)}</div>')
    body += (f'<h3 class="sub">What every column means</h3>'
             f'<div class="scroll">{_table(data_dictionary_frame(), 0, None)}</div>')
    parts.append(_section(None, "Lot statistics and measurements", "outputs", body))

    # ---- provenance -------------------------------------------------------
    cfg_rows = pd.DataFrame(
        [{"setting": k, "value": str(v)} for k, v in result.config.items()]
    )
    parts.append(_section(None, "How this run was configured", "outputs",
                          f'<div class="scroll">{_table(cfg_rows, 3, None)}</div>',
                          "Every setting the pipeline used, so the run can be repeated."))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} — {_esc(lot)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><main>
<h1>{_esc(title)}</h1>
<p class="lede">{_esc(lot)} · {len(seeds):,} seeds across {seeds.image.nunique()} photographs ·
measured in {units} · {stamp}</p>
{"".join(parts)}
<footer>Generated by LentilSeedVision. The tables in this report are also in the CSV,
Excel and JSON exports; this file is the readable version of the same run.</footer>
</main></body></html>""".encode("utf-8")
