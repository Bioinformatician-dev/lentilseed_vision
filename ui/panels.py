"""
The seven workflow panels, plus the outputs section, as render functions.

Each panel answers one question and says what to do when the answer is bad.
They take the finished `RunResult` and, where a panel needs to re-run something
(the repeatability test, the per-seed inspector), the detector and the original
photographs alongside it.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from seedvision import shape as shp
from seedvision import stage1_acquisition as s1
from seedvision import stage3_segmentation as s3
from seedvision import stage6_reliability as s6
from seedvision import stage7_validation as s7
from seedvision import export, lotstats
from seedvision.report import build_html
from seedvision.config import (
    COLOUR_TRAITS, CORE_TRAITS_MM, CORE_TRAITS_PX, PATTERN_TRAITS, SHAPE_TRAITS,
)

from . import motion
from .components import (
    bgr_to_rgb, empty_note, hline, muted_caption, note, panel_head, readings,
    style_fig, tray_card, verdict,
)
from .theme import CLASS_COLORS, DEFAULT_CLASS_COLOR, INK, LINE, MUTED, PATTERN_COLORS, STAGE_COLORS


@st.cache_data(show_spinner=False)
def _report_bytes(run_token: str, seeds_n: int, has_validation: bool, _result,
                  _validation) -> bytes:
    """Building the report embeds every tray, so it is cached per run."""
    return build_html(_result, _validation)


def export_report(result, validation) -> bytes:
    token = f"{id(result)}-{len(result.seeds)}-{validation is not None}"
    return _report_bytes(token, len(result.seeds), validation is not None,
                         result, validation)


def _colours_for(values) -> list[str]:
    return [CLASS_COLORS.get(v, DEFAULT_CLASS_COLOR) for v in values]


def _fmt(x, digits=2) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return f"{float(x):,.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# 1 — image acquisition & quality control
# ---------------------------------------------------------------------------

def panel_photographs(result, images: dict, cfg) -> None:
    panel_head(1, "Image acquisition & quality control",
               "Whether each photograph was fit to measure, before anything was measured "
               "from it. Flagged images stay in the results — the flag tells you which "
               "numbers to distrust.", "acquire")

    qc = result.qc
    passed = int(qc.qc_pass.sum())
    readings([
        (f"{passed}/{len(qc)}", "photographs passed"),
        (f"{qc.focus.median():.0f}", "median focus score"),
        (f"{100 * qc.illum_cv.median():.1f}%", "lighting variation"),
        (s1.scale_summary(cfg.scale).split("·")[0].strip(), "scale"),
    ], "acquire")

    show = qc[[
        "image", "megapixels", "focus", "clip_frac", "illum_cv", "seed_width_px",
        "n_seeds", "size_grade", "qc_pass", "qc_reason",
    ]].rename(columns={
        "megapixels": "MP", "clip_frac": "clipped", "illum_cv": "light CV",
        "seed_width_px": "seed width px", "n_seeds": "seeds", "size_grade": "resolution",
        "qc_pass": "passed", "qc_reason": "reading",
    })
    st.dataframe(show, width="stretch", hide_index=True)

    failed = qc[~qc.qc_pass]
    if not failed.empty:
        for _, row in failed.iterrows():
            verdict(row.image, f"{row.qc_reason} — reshoot or treat its numbers as indicative.",
                    "acquire")

    hline()
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Lighting across the frame**")
        pick = st.selectbox("Photograph", list(images), key="qc_illum",
                            label_visibility="collapsed")
        field = s1.illumination_field(images[pick])
        fig = go.Figure(go.Heatmap(
            z=field, colorscale="RdYlBu_r", zmid=1.0,
            colorbar=dict(title="× mean"),
        ))
        fig.update_yaxes(autorange="reversed", visible=False)
        fig.update_xaxes(visible=False)
        st.plotly_chart(style_fig(fig, 320), width="stretch")
        note("Even lighting is a flat field. A bright corner or a dark edge here is the "
             "same effect that shows up as position drift in panel 6.")

    with right:
        st.markdown("**Focus and seed size against the limits**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=qc.image, y=qc.focus, name="focus",
            marker_color=[STAGE_COLORS["acquire"] if f else "#C86A4E" for f in ~qc.flag_focus],
        ))
        fig2.add_hline(y=cfg.qc.focus_min, line_dash="dot", line_color=MUTED,
                       annotation_text="minimum focus", annotation_position="top left")
        st.plotly_chart(style_fig(fig2, 320), width="stretch")
        note(f"Seeds should be at least <b>{cfg.qc.seed_min_px:.0f} px</b> across, and "
             f"<b>{cfg.qc.seed_good_px:.0f} px</b> before measurement error stops mattering. "
             "Move the camera closer rather than cropping afterwards.")


# ---------------------------------------------------------------------------
# 2 — YOLO seed detection
# ---------------------------------------------------------------------------

def panel_detection(result, images: dict, cfg) -> None:
    panel_head(2, "YOLO seed detection",
               "Find every visible seed, then remove the duplicates that finding "
               "everything costs. The count on the tray is only as good as this step.",
               "detect")

    det = result.detection
    seeds = result.seeds
    readings([
        (f"{len(seeds):,}", "seeds kept"),
        (f"{int(det.duplicates_removed.sum())}", "duplicates merged"),
        (f"{int(det.truncated.sum())}", "cut by the frame edge"),
        (f"{seeds.confidence.mean():.2f}", "mean confidence"),
    ], "detect")

    st.dataframe(
        det.rename(columns={
            "raw_detections": "raw", "kept": "kept", "duplicates_removed": "merged",
            "truncated": "edge", "strategy": "pass", "mean_confidence": "mean conf",
        }),
        width="stretch", hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        counts = seeds["cls"].value_counts()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                               marker_color=_colours_for(counts.index)))
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(style_fig(fig, 320, "Seeds per class"), width="stretch")
    with c2:
        fig = go.Figure(go.Histogram(x=seeds.confidence, nbinsx=30,
                                     marker_color=STAGE_COLORS["detect"]))
        fig.add_vline(x=cfg.detect.conf, line_dash="dot", line_color=MUTED,
                      annotation_text="threshold")
        st.plotly_chart(style_fig(fig, 320, "Detector confidence"), width="stretch")

    if seeds.confidence.min() > cfg.detect.conf * 1.5:
        note("Nothing was detected near the threshold, so raising it slightly would not "
             "lose seeds — but nor is it costing you anything where it is.")
    else:
        note("Detections sit close to the threshold. Lower it a little and re-run to see "
             "whether the count climbs; if it does, seeds are being missed.")

    hline()
    st.markdown("**Annotated trays**")
    cols = st.columns(min(3, max(1, len(result.annotated))))
    for i, (name, vis) in enumerate(result.annotated.items()):
        with cols[i % len(cols)]:
            n = int((seeds.image == name).sum())
            tray_card(name, vis, f"<b>{n}</b> seeds")


# ---------------------------------------------------------------------------
# 3 — segmentation & contour extraction
# ---------------------------------------------------------------------------

def panel_outlines(result, images: dict, cfg) -> None:
    panel_head(3, "Seed-wise segmentation & contour extraction",
               "Each seed is thresholded inside its own detection box, so uneven light "
               "across the tray cannot bias the outline. Every measurement in panel 4 "
               "comes from these outlines.", "segment")

    seeds = result.seeds
    summary = result.outline_summary
    by_method = seeds.segment_method.value_counts()

    readings([
        (f"{100 * (seeds.segment_ok.mean()):.0f}%", "clean first cut"),
        (f"{_fmt(summary.get('median_error_pct'), 1)}%", "median outline error"),
        (f"{summary.get('flagged', 0)}", "outlines to inspect"),
        (by_method.index[0] if len(by_method) else "—", "commonest method"),
    ], "segment")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Bar(x=by_method.index, y=by_method.values,
                               marker_color=STAGE_COLORS["segment"]))
        st.plotly_chart(style_fig(fig, 300, "Which threshold produced each outline"),
                        width="stretch")
        note("<b>otsu</b> is the intended path. A lot leaning on <b>adaptive</b> or the "
             "colour-channel fallbacks is telling you the coat and the tray are too close "
             "in tone — change the background, not the settings.")
    with c2:
        err = (100 * seeds.outline_error).replace([np.inf, -np.inf], np.nan).dropna()
        fig = go.Figure(go.Histogram(x=err, nbinsx=40, marker_color=STAGE_COLORS["segment"]))
        fig.add_vline(x=0, line_color=MUTED, line_dash="dot")
        fig.update_xaxes(title="traced area vs ellipse-predicted area (%)")
        st.plotly_chart(style_fig(fig, 300, "Outline consistency"), width="stretch")
        note("A seed is nearly an ellipse, so this should cluster near zero. The long tail "
             "is where the trace leaked into a shadow or swallowed a neighbour.")

    hline()
    st.markdown("**Inspect one seed**")
    muted_caption("Region of interest, threshold, mask and the outline that came out of it.")

    ic1, ic2 = st.columns([1, 3])
    with ic1:
        img_name = st.selectbox("Photograph", sorted(seeds.image.unique()), key="seg_img")
        subset = seeds[seeds.image == img_name]
        worst = subset.reindex(subset.outline_error.abs().sort_values(ascending=False).index)
        mode = st.radio("Pick", ["Most suspect", "By id"], key="seg_mode",
                        label_visibility="collapsed")
        if mode == "Most suspect" and not worst.empty:
            row = worst.iloc[0]
        else:
            sid = st.number_input("Seed id", 0, int(subset.seed_id.max()), 0, key="seg_id")
            match = subset[subset.seed_id == sid]
            row = match.iloc[0] if not match.empty else subset.iloc[0]
        st.markdown(
            f"<div class='chipline'><span class='chip'>id <b>{int(row.seed_id)}</b></span>"
            f"<span class='chip'>{row.cls}</span>"
            f"<span class='chip'>error <b>{100 * row.outline_error:+.1f}%</b></span>"
            f"<span class='chip'>method <b>{row.segment_method}</b></span></div>",
            unsafe_allow_html=True,
        )

    with ic2:
        img = images.get(img_name)
        if img is None:
            empty_note("That photograph is no longer in memory — re-run the batch.")
            return
        roi, ox, oy = s3.extract_roi(img, (row.x1, row.y1, row.x2, row.y2), cfg.segment.pad_px)
        seg = s3.segment_seed(roi, cfg.segment)
        if seg is None:
            empty_note("This seed could not be re-segmented at the current settings.")
            return
        outline = np.zeros_like(seg.mask)
        cv2.drawContours(outline, [seg.contour], -1, 255, 2)
        traced = roi.copy()
        cv2.drawContours(traced, [seg.contour], -1, (60, 132, 48), 2)

        g1, g2, g3, g4 = st.columns(4)
        for col, arr, label in (
            (g1, bgr_to_rgb(roi), "region of interest"),
            (g2, seg.mask, "mask after thresholding"),
            (g3, outline, "clean outline"),
            (g4, bgr_to_rgb(traced), "outline on the seed"),
        ):
            with col:
                st.image(arr, width="stretch")
                muted_caption(label)


# ---------------------------------------------------------------------------
# 4 — image-derived phenotypes
# ---------------------------------------------------------------------------

def panel_phenotypes(result, images: dict, cfg) -> None:
    panel_head(4, "Image-derived phenotypes",
               "Three readings per seed: shape and size from the outline, colour from the "
               "coat, and the markings on it. The coat pattern is read from pixels alone "
               "and never consults the detector's label, so the two can disagree.",
               "phenotype")

    seeds = result.seeds
    units = result.units
    tabs = st.tabs(["Morphology", "Colour", "Coat pattern"])

    with tabs[0]:
        options = [c for c in [*(CORE_TRAITS_MM if units == "mm" else CORE_TRAITS_PX),
                               *SHAPE_TRAITS] if c in seeds.columns and seeds[c].notna().any()]
        trait = st.selectbox("Trait", options, key="morph_trait")
        v = pd.to_numeric(seeds[trait], errors="coerce").dropna()

        readings([
            (_fmt(v.mean()), f"mean {trait}"),
            (_fmt(v.std(ddof=1)), "standard deviation"),
            (f"{100 * v.std(ddof=1) / v.mean():.1f}%" if v.mean() else "—", "CV"),
            (f"{_fmt(v.quantile(.05))}–{_fmt(v.quantile(.95))}", "5th–95th percentile"),
        ], "phenotype")

        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = go.Figure(go.Histogram(x=v, nbinsx=40, marker_color=STAGE_COLORS["phenotype"]))
            fig.add_vline(x=v.mean(), line_color=INK, line_dash="dot", annotation_text="mean")
            st.plotly_chart(style_fig(fig, 330, f"{trait} across the lot"), width="stretch")
        with c2:
            fig = go.Figure()
            for cls, g in seeds.groupby("cls"):
                fig.add_trace(go.Box(
                    y=pd.to_numeric(g[trait], errors="coerce"), name=cls, boxpoints=False,
                    marker_color=CLASS_COLORS.get(cls, DEFAULT_CLASS_COLOR),
                ))
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 330, "By detector class"), width="stretch")

        st.markdown("**Length against width**")
        fig = go.Figure()
        xcol = "length_mm" if units == "mm" else "length_px"
        ycol = "width_mm" if units == "mm" else "width_px"
        for cls, g in seeds.groupby("cls"):
            fig.add_trace(go.Scatter(
                x=g[xcol], y=g[ycol], mode="markers", name=cls,
                marker=dict(size=6, color=CLASS_COLORS.get(cls, DEFAULT_CLASS_COLOR),
                            line=dict(width=0)),
            ))
        fig.update_xaxes(title=xcol)
        fig.update_yaxes(title=ycol)
        st.plotly_chart(style_fig(fig, 380), width="stretch")
        note("Seeds of one variety fall on a line through the origin — a seed off that line "
             "is a different shape, not just a different size.")

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(
                x=seeds.mean_a, y=seeds.mean_b, mode="markers",
                marker=dict(size=6, color=seeds.mean_L, colorscale="YlOrBr",
                            colorbar=dict(title="L*"), showscale=True),
                text=seeds.cls, hovertemplate="a* %{x:.1f}<br>b* %{y:.1f}<br>%{text}<extra></extra>",
            ))
            fig.update_xaxes(title="a*  (green ← → red)")
            fig.update_yaxes(title="b*  (blue ← → yellow)")
            st.plotly_chart(style_fig(fig, 380, "Coat colour in CIE a*b*"), width="stretch")
        with c2:
            trait = st.selectbox("Colour trait", [c for c in COLOUR_TRAITS if c in seeds.columns],
                                 key="col_trait")
            fig = go.Figure()
            for cls, g in seeds.groupby("cls"):
                fig.add_trace(go.Violin(
                    y=pd.to_numeric(g[trait], errors="coerce"), name=cls, box_visible=True,
                    line_color=CLASS_COLORS.get(cls, DEFAULT_CLASS_COLOR), points=False,
                ))
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 380, f"{trait} by class"), width="stretch")

        if "hex" in seeds.columns:
            st.markdown("**Colour of every seed, sorted by lightness**")
            ordered = seeds.sort_values("mean_L")
            swatches = "".join(
                f'<span style="display:inline-block;width:11px;height:22px;background:{h}"></span>'
                for h in ordered.hex.head(400)
            )
            st.markdown(f"<div style='line-height:0'>{swatches}</div>", unsafe_allow_html=True)
            muted_caption("Each bar is one seed's mean coat colour. A clean gradient means one "
                          "population; a break means two.")

    with tabs[2]:
        if "pattern_class" not in seeds.columns:
            empty_note("No coat pattern was read for this lot.")
            return
        counts = seeds.pattern_class.value_counts()
        readings([
            (counts.index[0], "commonest pattern"),
            (f"{100 * counts.iloc[0] / len(seeds):.0f}%", "of the lot"),
            (f"{seeds.spot_count.mean():.1f}", "markings per seed"),
            (f"{100 * seeds.spot_coverage.mean():.1f}%", "of coat marked"),
        ], "phenotype")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=counts.index, y=counts.values,
                marker_color=[PATTERN_COLORS.get(p, DEFAULT_CLASS_COLOR) for p in counts.index],
            ))
            st.plotly_chart(style_fig(fig, 320, "Coat pattern, read from pixels"),
                            width="stretch")
        with c2:
            fig = go.Figure(go.Scatter(
                x=seeds.spot_count, y=seeds.spot_coverage, mode="markers",
                marker=dict(size=6, color=seeds.spot_contrast_dE, colorscale="Oranges",
                            colorbar=dict(title="ΔE"), showscale=True),
                text=seeds.pattern_class,
            ))
            fig.update_xaxes(title="markings per seed")
            fig.update_yaxes(title="share of coat marked")
            st.plotly_chart(style_fig(fig, 320, "How marked the coats are"), width="stretch")

        cross = lotstats.agreement_table(seeds)
        if not cross.empty:
            st.markdown("**Detector class against coat pattern**")
            st.dataframe(cross, width="stretch")
            off_diagonal = 1 - np.trace(cross.values) / cross.values.sum()
            note(f"These are two independent readings of the same seeds, so they will not "
                 f"match exactly. Here they differ on <b>{100 * off_diagonal:.0f}%</b> of "
                 "seeds — worth opening a few of those in panel 3 before trusting either.")


# ---------------------------------------------------------------------------
# 5 — derived physical properties
# ---------------------------------------------------------------------------

def panel_physical(result, images: dict, cfg) -> None:
    panel_head(5, "Derived physical properties",
               "Pixels become millimetres, and the engineering properties follow: mean "
               "diameters, sphericity, surface area, volume, mass and thousand-seed weight.",
               "physical")

    seeds = result.seeds
    if result.units != "mm":
        empty_note("This run is uncalibrated, so there are no physical properties yet.")
        note("Photograph a ruler beside the tray, then set the scale in the sidebar under "
             "<b>2 · Scale</b>. Every column below appears once it is set: mm sizes, "
             "volume, mass and thousand-seed weight.")
        return

    for n in result.notes:
        verdict("assumption", n, "physical")

    tsw = result.tsw
    readings([
        (f"{seeds.length_mm.mean():.2f}", "mean length (mm)"),
        (f"{seeds.gmd_mm.mean():.2f}", "geometric mean diameter (mm)"),
        (f"{seeds.volume_mm3.mean():.1f}", "mean volume (mm³)"),
        (f"{tsw['tsw_g']:.1f} ± {tsw.get('tsw_ci95_g', float('nan')):.1f}" if tsw else "—",
         "thousand-seed weight (g)"),
    ], "physical")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for col, colour in (("length_mm", "#7659A6"), ("width_mm", "#A48CCB"),
                            ("thickness_mm", "#CFC1E4")):
            if col in seeds.columns:
                fig.add_trace(go.Histogram(x=seeds[col], name=col, nbinsx=35,
                                           marker_color=colour, opacity=0.75))
        fig.update_layout(barmode="overlay", xaxis_title="mm")
        st.plotly_chart(style_fig(fig, 340, "The three axes"), width="stretch")
        note("Thickness is modelled from width, so its distribution is width's distribution "
             "rescaled. It is drawn here to show what the volume rests on, not as an "
             "independent measurement.")
    with c2:
        fig = go.Figure(go.Scatter(
            x=seeds.gmd_mm, y=seeds.sphericity, mode="markers",
            marker=dict(size=6, color=seeds.volume_mm3, colorscale="Purples",
                        colorbar=dict(title="mm³"), showscale=True),
        ))
        fig.update_xaxes(title="geometric mean diameter (mm)")
        fig.update_yaxes(title="sphericity")
        st.plotly_chart(style_fig(fig, 340, "Size against roundness of the solid"),
                        width="stretch")

    hline()
    st.markdown("**Thousand-seed weight**")
    if not tsw:
        note("Set a density in the sidebar under <b>5 · Physical properties</b> — either the "
             "literature figure for a first estimate, or your own, by weighing a counted "
             "sub-sample.")
    else:
        cols = st.columns(4)
        for col, (label, value) in zip(cols, [
            ("TSW", f"{tsw['tsw_g']:.2f} g"),
            ("95% interval", f"± {tsw.get('tsw_ci95_g', float('nan')):.2f} g"),
            ("mean seed mass", f"{tsw['mean_mass_mg']:.2f} mg"),
            ("mass CV", f"{tsw.get('cv_mass_pct', float('nan')):.1f}%"),
        ]):
            col.metric(label, value)
        note("TSW here is computed from measured volumes and a density, not from a balance. "
             "It follows the lot's real size distribution, which a scoop-and-weigh does not — "
             "but it is only as good as the density and the thickness ratio behind it.")

    st.markdown("**Physical traits, seed by seed**")
    show = [c for c in ["image", "seed_id", "cls", "length_mm", "width_mm", "thickness_mm",
                        "gmd_mm", "amd_mm", "sphericity", "surface_area_mm2",
                        "volume_mm3", "mass_mg"] if c in seeds.columns]
    st.dataframe(seeds[show].round(3), width="stretch", height=320, hide_index=True)


# ---------------------------------------------------------------------------
# 6 — reliability & self-consistency
# ---------------------------------------------------------------------------

def panel_reliability(result, images: dict, cfg, detector) -> None:
    panel_head(6, "Reliability & self-consistency checks",
               "No ground truth needed. These ask whether the pipeline agrees with itself: "
               "across duplicate detections, across transformations of the same tray, and "
               "across the frame.", "reliability")

    seeds = result.seeds
    checks = result.checks
    summary = result.outline_summary

    readings([
        (f"{int(checks.duplicate_pairs.sum())}", "duplicate detections"),
        (f"{int(checks.merged_suspects.sum())}", "merged-seed suspects"),
        (f"{summary.get('flagged', 0)}", "outlines flagged"),
        (f"{int(checks.edge_truncated.sum())}", "seeds cut by the frame"),
    ], "reliability")

    tabs = st.tabs([
        "Detection checks", "Outline consistency", "Repeatability",
        "Position drift", "Seed-to-seed matching",
    ])

    with tabs[0]:
        st.dataframe(checks, width="stretch", hide_index=True)
        note("Two boxes on one seed inflate the count; one box across two deflates it. "
             "Both survive a confidence threshold, which is why they are counted here "
             "rather than assumed away.")

    with tabs[1]:
        c1, c2 = st.columns([1, 1])
        with c1:
            for k, v in [
                ("median error", f"{_fmt(summary.get('median_error_pct'), 2)}%"),
                ("spread (IQR)", f"{_fmt(summary.get('iqr_error_pct'), 2)}%"),
                ("flagged", f"{summary.get('flagged', 0)} seeds "
                            f"({_fmt(summary.get('flagged_pct'), 1)}%)"),
            ]:
                verdict(k, v, "reliability")
            by_method = summary.get("by_method", {})
            if by_method:
                st.markdown("**Failure rate by threshold method**")
                st.dataframe(
                    pd.DataFrame(by_method.items(), columns=["method", "flagged %"]),
                    width="stretch", hide_index=True,
                )
        with c2:
            fig = go.Figure(go.Scatter(
                x=seeds.area_px2, y=seeds.ellipse_area_px2, mode="markers",
                marker=dict(size=5, color=np.where(seeds.outline_flag, "#B24C87", "#9AA8A2")),
            ))
            lim = float(np.nanpercentile(seeds.area_px2, 99)) * 1.1
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                     line=dict(color=INK, dash="dot"), showlegend=False))
            fig.update_xaxes(title="traced area (px²)")
            fig.update_yaxes(title="ellipse-predicted area (px²)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 340, "Traced against predicted"), width="stretch")

    with tabs[2]:
        note("The same tray, flipped, rotated and resized. Nothing about the seeds changed, "
             "so whatever the numbers do is the pipeline's own noise. A difference between "
             "two lots smaller than this CV is not a difference.")
        c1, c2 = st.columns([1, 2])
        with c1:
            name = st.selectbox("Photograph", list(images), key="rep_img")
            traits = st.multiselect(
                "Traits", result.trait_menu,
                default=result.trait_menu[:3], key="rep_traits",
            )
            run = st.button("Run the test-retest", width="stretch", key="rep_run")
        with c2:
            if run and detector is not None and traits:
                with st.spinner("Measuring the same tray four ways…"):
                    def measure(image, label):
                        from seedvision.runner import measure_image
                        return measure_image(image, label, detector, cfg)
                    per_rep, per_trait = s6.repeatability(
                        images[name], name, measure, cfg.reliability, traits
                    )
                st.session_state["repeat"] = (per_rep, per_trait)
            stored = st.session_state.get("repeat")
            if stored:
                per_rep, per_trait = stored
                st.dataframe(per_rep.round(3), width="stretch", hide_index=True)
                if not per_trait.empty:
                    st.dataframe(
                        per_trait[["trait", "original", "cv_pct", "max_shift_pct", "verdict"]]
                        .round(3).rename(columns={"cv_pct": "CV %", "max_shift_pct": "worst shift %"}),
                        width="stretch", hide_index=True,
                    )
                    worst = per_trait.sort_values("cv_pct").iloc[-1]
                    verdict("noise floor",
                            f"{worst.trait} moves by {worst.cv_pct:.2f}% across replicates — "
                            "treat that as the smallest difference this pipeline can resolve.",
                            "reliability")
            else:
                empty_note("Not run yet. It re-measures one tray four times, so it takes about "
                           "four times a single photograph.")

    with tabs[3]:
        drift, grid = s6.position_drift(seeds, result.trait_menu[:6], cfg.reliability)
        if drift.empty:
            empty_note("Not enough seeds to test for a position effect.")
        else:
            st.dataframe(
                drift[["trait", "slope_x_pct", "slope_y_pct", "r2", "n", "reading"]].round(3)
                .rename(columns={"slope_x_pct": "left→right %", "slope_y_pct": "top→bottom %"}),
                width="stretch", hide_index=True,
            )
            note("Seeds are laid out at random, so where a seed sits should tell you nothing "
                 "about its size. When it does, the cause is the camera: vignetting, a lamp "
                 "on one side, or the lens stretching the corners.")
            if not grid.empty:
                trait = grid.attrs.get("trait")
                pivot = grid.pivot(index="row", columns="col", values="pct_of_frame_mean")
                fig = go.Figure(go.Heatmap(z=pivot.values, colorscale="RdYlBu_r", zmid=100,
                                           colorbar=dict(title="% of mean")))
                fig.update_yaxes(autorange="reversed", visible=False)
                fig.update_xaxes(visible=False)
                st.plotly_chart(style_fig(fig, 300, f"{trait} by position in frame"),
                                width="stretch")

    with tabs[4]:
        note("Photograph one tray twice without disturbing it. The seeds are paired by "
             "position, and the difference between the two readings of each seed is the "
             "error bar you can put on a single measurement.")
        names = sorted(seeds.image.unique())
        if len(names) < 2:
            empty_note("Two photographs of the same tray are needed. Upload a second shot "
                       "and re-run.")
        else:
            c1, c2, c3 = st.columns(3)
            a = c1.selectbox("First photograph", names, key="match_a")
            b = c2.selectbox("Second photograph", names, index=1, key="match_b")
            trait = c3.selectbox("Trait", result.trait_menu, key="match_trait")
            matched = s6.match_seeds(seeds[seeds.image == a], seeds[seeds.image == b],
                                     cfg.reliability)
            if matched.empty or f"{trait}_diff" not in matched.columns:
                empty_note("No seeds paired up. If the tray moved between shots, the pairing "
                           "cannot be trusted and this check does not apply.")
            else:
                summ = s6.matching_summary(matched, trait)
                cols = st.columns(4)
                cols[0].metric("pairs", summ["pairs"])
                cols[1].metric("bias", f"{summ['bias']:+.3f}")
                cols[2].metric("SD of difference", f"{summ['sd_of_difference']:.3f}")
                cols[3].metric("repeatability CV", f"{summ['repeatability_cv_pct']:.2f}%")
                fig = go.Figure(go.Scatter(
                    x=matched[f"{trait}_a"], y=matched[f"{trait}_b"], mode="markers",
                    marker=dict(size=7, color=STAGE_COLORS["reliability"]),
                ))
                lo = float(min(matched[f"{trait}_a"].min(), matched[f"{trait}_b"].min()))
                hi = float(max(matched[f"{trait}_a"].max(), matched[f"{trait}_b"].max()))
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                         line=dict(color=INK, dash="dot"), showlegend=False))
                fig.update_xaxes(title=f"{trait} — first shot")
                fig.update_yaxes(title=f"{trait} — second shot")
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig, 380, "The same seeds, measured twice"),
                                width="stretch")


# ---------------------------------------------------------------------------
# 7 — external validation
# ---------------------------------------------------------------------------

def panel_validation(result, images: dict, cfg) -> None:
    panel_head(7, "External validation",
               "Caliper a sub-sample and compare. Self-consistency shows the pipeline "
               "repeats itself; only this shows it is right.", "validation")

    seeds = result.seeds
    if result.units != "mm":
        empty_note("Validation compares millimetres with millimetres. Set the scale in the "
                   "sidebar first.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**1 · Take a blank sheet**")
        n = st.number_input("Seeds to measure", 10, 200, 25, 5, key="val_n")
        template = s7.reference_template(seeds, int(n), cfg.validation.traits)
        st.download_button(
            "Download the caliper sheet", template.to_csv(index=False).encode("utf-8"),
            "caliper_sheet.csv", "text/csv", width="stretch",
        )
        muted_caption("A random sub-sample, so the check is honest at both ends of the size "
                      "distribution. Measure those seeds, type the numbers into the empty "
                      "columns, and upload it below.")
    with c2:
        st.markdown("**2 · Upload the measurements**")
        ref_file = st.file_uploader("Caliper CSV", type=["csv"], key="val_ref",
                                    label_visibility="collapsed")
        join = st.radio(
            "Pairing", ["seed_id", "rank"], horizontal=True, key="val_join",
            help="seed_id pairs by the ids on the sheet. rank pairs by size order within "
                 "each photograph — only right if the same seeds were measured.",
        )

    if ref_file is None:
        empty_note("No caliper measurements yet. Everything else in this app is the pipeline "
                   "checking its own work; this panel is the only one that can tell you the "
                   "numbers are true.")
        return

    try:
        df_ref = pd.read_csv(ref_file)
    except Exception as exc:                      # noqa: BLE001
        st.error(f"That CSV could not be read: {exc}")
        return

    vcfg = cfg.validation
    vcfg.join_on = join
    available = tuple(t for t in vcfg.traits if t in df_ref.columns)
    if not available:
        st.error(
            "None of the expected trait columns are in that file. It needs at least one of: "
            + ", ".join(vcfg.traits)
        )
        return
    vcfg.traits = available

    summary, paired = s7.validate(seeds, df_ref, vcfg)
    if summary.empty:
        st.error("No seeds paired up. Check that the image names match the ones in the run.")
        return

    st.session_state["validation_summary"] = summary
    st.dataframe(
        summary.round(4).rename(columns={
            "r2": "R²", "ccc": "concordance", "rmse": "RMSE", "mae": "MAE",
            "bias_pct": "bias %", "loa_low": "LoA low", "loa_high": "LoA high",
        }),
        width="stretch", hide_index=True,
    )
    for _, row in summary.iterrows():
        verdict(row.trait, row.verdict, "validation")

    trait = st.selectbox("Trait to plot", list(paired), key="val_trait")
    d = paired[trait]
    stats = summary[summary.trait == trait].iloc[0]

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(go.Scatter(
            x=d.reference, y=d.measured, mode="markers",
            marker=dict(size=7, color=STAGE_COLORS["validation"]),
        ))
        lo = float(min(d.reference.min(), d.measured.min()))
        hi = float(max(d.reference.max(), d.measured.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                                 line=dict(color=INK, dash="dot")))
        xs = np.linspace(lo, hi, 20)
        fig.add_trace(go.Scatter(x=xs, y=stats.slope * xs + stats.intercept, mode="lines",
                                 name="fit", line=dict(color="#C86A4E")))
        fig.update_xaxes(title="calipers (mm)")
        fig.update_yaxes(title="image (mm)")
        st.plotly_chart(style_fig(fig, 380, f"R² {stats.r2:.3f} · RMSE {stats.rmse:.3f} mm"),
                        width="stretch")
    with g2:
        fig = go.Figure(go.Scatter(
            x=d.mean_of_pair, y=d.difference, mode="markers",
            marker=dict(size=7, color=STAGE_COLORS["validation"]),
        ))
        for y, label, dash in (
            (stats.bias, "bias", "solid"),
            (stats.loa_low, "lower limit", "dash"),
            (stats.loa_high, "upper limit", "dash"),
        ):
            fig.add_hline(y=y, line_dash=dash, line_color=MUTED,
                          annotation_text=f"{label} {y:+.3f}", annotation_position="right")
        fig.update_xaxes(title="mean of the two measurements (mm)")
        fig.update_yaxes(title="image − calipers (mm)")
        st.plotly_chart(style_fig(fig, 380, "Bland–Altman"), width="stretch")

    note("The left plot shows whether the two move together; the right shows whether they "
         "agree. A pipeline that reads every seed 5% long sits perfectly on the left plot "
         "and well off zero on the right, which is the failure this panel exists to catch.")


# ---------------------------------------------------------------------------
# final outputs
# ---------------------------------------------------------------------------

def panel_outputs(result, images: dict, cfg) -> None:
    panel_head(8, "Final outputs",
               "The tables that leave the app: every seed, the lot summary, the shape "
               "atlas, and the files to hand to a collaborator.", "outputs")

    seeds = result.seeds
    tabs = st.tabs(["Lot statistics", "Shape atlas", "Shape PCA", "Per-seed table", "Download"])

    with tabs[0]:
        st.dataframe(result.lot_stats.round(3), width="stretch", hide_index=True)
        if not result.composition.empty:
            st.markdown("**Composition**")
            st.dataframe(result.composition.round(1), width="stretch", hide_index=True)
        if not result.per_image.empty:
            st.markdown("**By photograph**")
            st.dataframe(result.per_image.round(3), width="stretch", hide_index=True)
            note("Trays of one lot should agree. One tray sitting apart from the others is "
                 "usually a photography difference, and panel 1 will normally say which.")

    with tabs[1]:
        efds = {k: v for k, v in result.efd.items() if v is not None}
        if len(efds) < 3:
            empty_note("At least three traced seeds are needed to draw the atlas.")
        else:
            keys = list(efds)[:cfg.shape.atlas_max_outlines]
            curves = [shp.efd_reconstruct(efds[k]) for k in keys]
            mx, my = shp.efd_reconstruct(shp.mean_shape(efds))
            faint = dict(width=1, color="rgba(18,33,29,0.13)")
            average = go.Scatter(x=mx, y=my, mode="lines", name="lot average",
                                 line=dict(width=3, color=STAGE_COLORS["outputs"]))

            # Frames stack the outlines on a few at a time. A bundle that looks
            # solid when finished shows how tight it really is while it builds.
            step = max(1, len(curves) // 40)
            frames = []
            for i in range(step, len(curves) + step, step):
                i = min(i, len(curves))
                frames.append(go.Frame(name=str(i), data=[
                    *(go.Scatter(x=x, y=y, mode="lines", showlegend=False,
                                 line=faint, hoverinfo="skip") for x, y in curves[:i]),
                    average,
                ]))

            fig = go.Figure(
                data=[
                    *(go.Scatter(x=x, y=y, mode="lines", showlegend=False,
                                 line=faint, hoverinfo="skip") for x, y in curves),
                    average,
                ],
                frames=frames,
            )
            fig.update_xaxes(scaleanchor="y", visible=False)
            fig.update_yaxes(visible=False)
            style_fig(fig, 480, f"{len(keys)} outlines, size and rotation removed")
            st.plotly_chart(motion.atlas_animation(fig, len(curves)), width="stretch")
            note("Every outline normalised and stacked on the lot average. An outline far "
                 "from the bundle is usually a segmentation failure rather than an unusual "
                 "seed — panel 3 will show which.")

    with tabs[2]:
        model = result.pca_model
        if not model:
            empty_note("Not enough traced outlines for a shape PCA.")
        else:
            exp = model["explained"]
            c1, c2 = st.columns([1.3, 1])
            with c1:
                pcx = st.selectbox("Horizontal", range(1, model["n_components"] + 1),
                                   index=0, format_func=lambda i: f"PC{i}", key="pcx")
                pcy = st.selectbox("Vertical", range(1, model["n_components"] + 1),
                                   index=min(1, model["n_components"] - 1),
                                   format_func=lambda i: f"PC{i}", key="pcy")
                key_index = {k: i for i, k in enumerate(result.pca_keys)}
                idx = [key_index.get((r.image, r.seed_id)) for r in seeds.itertuples()]
                mask = [i is not None for i in idx]
                sub = seeds[mask]
                rows = [i for i in idx if i is not None]
                fig = go.Figure()
                for cls, g in sub.groupby("cls"):
                    sel = [key_index[(r.image, r.seed_id)] for r in g.itertuples()]
                    fig.add_trace(go.Scatter(
                        x=model["scores"][sel, pcx - 1], y=model["scores"][sel, pcy - 1],
                        mode="markers", name=cls,
                        marker=dict(size=6, color=CLASS_COLORS.get(cls, DEFAULT_CLASS_COLOR)),
                    ))
                fig.update_xaxes(title=f"PC{pcx} ({100 * exp[pcx - 1]:.1f}%)")
                fig.update_yaxes(title=f"PC{pcy} ({100 * exp[pcy - 1]:.1f}%)")
                st.plotly_chart(style_fig(fig, 400, "Outline shape, reduced to two axes"),
                                width="stretch")
                _ = rows
            with c2:
                st.markdown(f"**What PC{pcx} means**")
                extremes = shp.shape_along_pc(model, cfg.shape.harmonics, pcx - 1)
                if extremes:
                    fig = go.Figure()
                    for (x, y), label, colour in zip(
                        extremes, ("−2 SD", "+2 SD"), ("#9AA8A2", STAGE_COLORS["outputs"])
                    ):
                        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label,
                                                 line=dict(width=2.5, color=colour)))
                    fig.update_xaxes(scaleanchor="y", visible=False)
                    fig.update_yaxes(visible=False)
                    st.plotly_chart(style_fig(fig, 400), width="stretch")
                muted_caption("The two outlines a seed would have at either end of this axis. "
                              "The component scores are in the per-seed table as shape_pc1 "
                              "onwards, ready to use as traits.")

    with tabs[3]:
        cols = st.multiselect(
            "Columns", list(export.clean(seeds).columns),
            default=[c for c in ["image", "seed_id", "cls", "confidence", "pattern_class",
                                 *(CORE_TRAITS_MM if result.units == "mm" else CORE_TRAITS_PX)[:5],
                                 "roundness", "solidity"] if c in seeds.columns],
            key="table_cols",
        )
        st.dataframe(export.clean(seeds)[cols] if cols else export.clean(seeds),
                     width="stretch", height=460, hide_index=True)

    with tabs[4]:
        validation = st.session_state.get("validation_summary")

        st.markdown("**The whole run as one page**")
        r1, r2 = st.columns([1, 3])
        with r1:
            st.download_button(
                "HTML report", export_report(result, validation),
                f"{cfg.lot_id}_report.html", "text/html", width="stretch",
                type="primary",
            )
        with r2:
            note("One file, nothing beside it: the annotated trays are embedded and the "
                 "charts are drawn into the page, so it opens the same on a machine with "
                 "no network. Print it to PDF from the browser for a supervisor or a "
                 "supplementary file.")

        st.markdown("**Tables**")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("Per-seed CSV", export.to_csv_bytes(seeds),
                               "lentil_per_seed.csv", "text/csv", width="stretch")
        with d2:
            st.download_button(
                "Excel workbook", export.to_excel_bytes(result, validation),
                "lentil_phenotypes.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with d3:
            st.download_button("JSON", export.to_json_bytes(result, validation),
                               "lentil_phenotypes.json", "application/json", width="stretch")
        with d4:
            st.download_button(
                "Everything (ZIP)", export.to_zip_bytes(result, validation),
                "lentil_results.zip", "application/zip", width="stretch",
            )
        note("The ZIP carries the tables, the report, the annotated trays, the data "
             "dictionary and the exact settings this run used — enough for someone else "
             "to repeat it.")

        st.markdown("**Annotated photographs**")
        a1, a2, a3 = st.columns([1.4, 1, 1])
        with a1:
            pick = st.selectbox("Photograph", list(result.annotated), key="dl_img")
        with a2:
            fmt = st.radio("Format", ["png", "jpg"], horizontal=True, key="dl_fmt",
                           help="PNG keeps the drawn outlines crisp; JPEG is smaller.")
        with a3:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            st.download_button(
                "Download this tray",
                export.annotated_bytes(result.annotated[pick], fmt),
                export.annotated_name(pick, fmt), f"image/{fmt}", width="stretch",
            )
        b1, b2 = st.columns([1, 3])
        with b1:
            st.download_button(
                f"All {len(result.annotated)} trays (ZIP)",
                export.annotated_zip_bytes(result.annotated, fmt),
                f"{cfg.lot_id}_annotated_{fmt}.zip", "application/zip", width="stretch",
            )
        with b2:
            note("Boxes, traced outlines, per-seed class labels and the count badge, drawn "
                 "at the photograph's own resolution — figure-ready as they are.")

        hline()

        st.markdown("**GWAS-ready trait table**")
        level = st.radio("One row per", ["lot", "photograph"], horizontal=True, key="gwas_level")
        wide = lotstats.gwas_wide(seeds, result.trait_menu, cfg.lot_id,
                                  by="image" if level == "photograph" else "lot")
        st.dataframe(wide.round(4), width="stretch", hide_index=True)
        st.download_button("Download the wide table", wide.to_csv(index=False).encode("utf-8"),
                           "lentil_traits_wide.csv", "text/csv")

        st.markdown("**What every column means**")
        st.dataframe(export.data_dictionary_frame(), width="stretch", hide_index=True,
                     height=280)
