"""
The pipeline, end to end, with no user interface attached.

`run_batch` walks a set of photographs through panels 1 to 6 and returns a
`RunResult` holding every table the outputs section needs. Progress is
reported through a callback, so the Streamlit shell can animate the stage rail
and a CLI can print lines, without either of them owning the logic.

    result = run_batch(images, Detector("best.pt"), RunConfig())
    result.seeds.to_csv("per_seed.csv", index=False)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import lotstats, shape, stage1_acquisition as s1, stage3_segmentation as s3, \
    stage4_phenotypes as s4, stage5_physical as s5, stage6_reliability as s6
from .config import CORE_TRAITS_MM, CORE_TRAITS_PX, SHAPE_TRAITS, RunConfig
from .stage2_detection import Detector

# The seven panels of the workflow, in order, with the label the UI shows.
STAGES: list[tuple[str, str]] = [
    ("acquire", "Photograph"),
    ("detect", "Detection"),
    ("segment", "Outlines"),
    ("phenotype", "Phenotypes"),
    ("physical", "Physical"),
    ("reliability", "Reliability"),
    ("outputs", "Outputs"),
]

ProgressFn = Callable[[str, float, str], None]


def _noop(stage: str, fraction: float, message: str) -> None:
    return None


@dataclass
class RunResult:
    seeds: pd.DataFrame                       # one row per seed, everything joined
    qc: pd.DataFrame                          # one row per photograph
    detection: pd.DataFrame                   # panel 2 report per photograph
    checks: pd.DataFrame                      # panel 6 detection checks
    outline_summary: dict
    lot_stats: pd.DataFrame
    per_image: pd.DataFrame
    composition: pd.DataFrame
    tsw: dict
    annotated: dict[str, np.ndarray] = field(default_factory=dict)
    contours: dict[tuple, np.ndarray] = field(default_factory=dict)
    efd: dict[tuple, np.ndarray] = field(default_factory=dict)
    pca_model: dict = field(default_factory=dict)
    pca_keys: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timing: dict = field(default_factory=dict)

    @property
    def units(self) -> str:
        return "mm" if self.seeds.get("units", pd.Series(["px"])).iloc[0] == "mm" else "px"

    @property
    def trait_menu(self) -> list[str]:
        base = CORE_TRAITS_MM if self.units == "mm" else CORE_TRAITS_PX
        return [c for c in [*base, *SHAPE_TRAITS] if c in self.seeds.columns
                and self.seeds[c].notna().any()]


# ---------------------------------------------------------------------------
# one photograph
# ---------------------------------------------------------------------------

def measure_image(img: np.ndarray, name: str, detector: Detector,
                  cfg: RunConfig, collect: dict | None = None) -> pd.DataFrame:
    """
    Panels 2 to 4 for a single photograph.

    `collect`, when given, receives the contours and descriptors keyed by
    (image, seed_id) — the repeatability check calls this function without it,
    since it only needs the numbers.
    """
    h, w = img.shape[:2]
    boxes, report = detector.detect(img, cfg.detect)

    rows = []
    for i, box in enumerate(boxes):
        roi, ox, oy = s3.extract_roi(img, box.xyxy, cfg.segment.pad_px)
        seg = s3.segment_seed(roi, cfg.segment)
        if seg is None:
            continue
        pheno = s4.phenotype_seed(roi, seg.mask, seg.contour, cfg.pattern, cfg.colour)
        if pheno is None:
            continue

        rows.append({
            "image": name, "seed_id": i,
            "cls": box.cls_name, "confidence": box.conf,
            "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
            "cx": box.centre[0], "cy": box.centre[1],
            "img_w": w, "img_h": h,
            "truncated": box.truncated,
            "segment_method": seg.method,
            "segment_ok": seg.ok,
            "fragments": seg.fragments,
            **pheno,
        })

        if collect is not None:
            global_contour = s3.to_global(seg.contour, ox, oy)
            collect["contours"][(name, i)] = global_contour
            collect["efd"][(name, i)] = shape.elliptic_fourier(
                seg.contour, cfg.shape.harmonics, cfg.shape.normalize
            )

    df = pd.DataFrame(rows)
    if collect is not None:
        collect["reports"].append({"image": name, **report})
    return df


# ---------------------------------------------------------------------------
# annotation
# ---------------------------------------------------------------------------

def annotate(img: np.ndarray, rows: pd.DataFrame, contours: dict,
             class_colors: dict[str, tuple[int, int, int]],
             default_color=(110, 127, 118)) -> np.ndarray:
    """Draw the boxes, the traced outlines and a count badge onto the tray."""
    vis = img.copy()
    counts: dict[str, int] = {}

    for _, r in rows.iterrows():
        colour = class_colors.get(r["cls"], default_color)
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1
        cv2.rectangle(vis, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, 2)
        cnt = contours.get((r["image"], r["seed_id"]))
        if cnt is not None:
            cv2.drawContours(vis, [cnt], -1, colour, 1)
        label = f'{r["cls"]} {r["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        ly = max(th + 4, int(r.y1) - 3)
        cv2.rectangle(vis, (int(r.x1), ly - th - 4), (int(r.x1) + tw + 4, ly + 2), colour, -1)
        fg = (0, 0, 0) if sum(colour) > 380 else (255, 255, 255)
        cv2.putText(vis, label, (int(r.x1) + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, fg, 1, cv2.LINE_AA)

    badge = f"{len(rows)} seeds"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(vis, (8, 8), (18 + tw, 20 + th), (36, 42, 30), -1)
    cv2.putText(vis, badge, (14, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)

    y = 14
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        colour = class_colors.get(cls, default_color)
        chip = f"{cls}: {n}"
        (tw, th), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x2 = vis.shape[1] - 8
        cv2.rectangle(vis, (x2 - tw - 12, y - th - 3), (x2, y + 5), colour, -1)
        fg = (0, 0, 0) if sum(colour) > 380 else (255, 255, 255)
        cv2.putText(vis, chip, (x2 - tw - 8, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    fg, 1, cv2.LINE_AA)
        y += th + 10
    return vis


# ---------------------------------------------------------------------------
# whole batch
# ---------------------------------------------------------------------------

def run_batch(images: list[tuple[str, np.ndarray]], detector: Detector,
              cfg: RunConfig, progress: ProgressFn = _noop,
              class_colors: dict | None = None,
              run_repeatability: bool = False) -> RunResult:
    """Panels 1 to 6 over every photograph, then the lot-level roll-up."""
    t0 = time.time()
    class_colors = class_colors or {}
    collect = {"contours": {}, "efd": {}, "reports": []}

    qc_rows, seed_frames, annotated = [], [], {}
    n = max(1, len(images))

    for idx, (name, img) in enumerate(images):
        progress("acquire", idx / n, f"Checking {name}")
        qc_rows.append(s1.qc_metrics(img, name))

        progress("detect", (idx + 0.25) / n, f"Detecting seeds in {name}")
        df = measure_image(img, name, detector, cfg, collect)

        progress("phenotype", (idx + 0.75) / n, f"Measuring {len(df)} seeds in {name}")
        if not df.empty:
            seed_frames.append(df)
            annotated[name] = annotate(img, df, collect["contours"], class_colors)

    seeds = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    detection = pd.DataFrame(collect["reports"])

    if seeds.empty:
        qc = s1.apply_qc_rules(pd.DataFrame(qc_rows), cfg.qc)
        return RunResult(
            seeds=seeds, qc=qc, detection=detection, checks=pd.DataFrame(),
            outline_summary={}, lot_stats=pd.DataFrame(), per_image=pd.DataFrame(),
            composition=pd.DataFrame(), tsw={}, config=cfg.to_dict(),
            notes=["No seeds were measured — lower the confidence threshold and run again."],
            timing={"seconds": time.time() - t0},
        )

    # ---- panel 5: physical units -----------------------------------------
    progress("physical", 0.85, "Converting to physical units")
    seeds = s5.to_physical(seeds, cfg.scale, cfg.physical)

    # ---- panel 1 completion + panel 6 checks ------------------------------
    progress("reliability", 0.9, "Running consistency checks")
    qc = s1.apply_qc_rules(
        pd.DataFrame(qc_rows), cfg.qc,
        seed_width_px=seeds.groupby("image")["width_px"].median(),
        seed_counts=seeds.groupby("image").size(),
    )
    seeds, outline_summary = s6.outline_consistency(seeds, cfg.reliability)
    checks = s6.detection_checks(seeds, cfg.reliability)

    # ---- shape ------------------------------------------------------------
    X, keys = shape.descriptor_matrix(collect["efd"])
    pca_model = shape.pca(X, cfg.shape.pca_components) if len(keys) >= 4 else {}
    if pca_model:
        for i in range(pca_model["n_components"]):
            col = f"shape_pc{i + 1}"
            mapping = {k: pca_model["scores"][j, i] for j, k in enumerate(keys)}
            seeds[col] = [
                mapping.get((r.image, r.seed_id), np.nan)
                for r in seeds.itertuples()
            ]

    # ---- outputs ----------------------------------------------------------
    progress("outputs", 0.97, "Summarising the lot")
    units = "mm" if cfg.scale.calibrated else "px"
    traits = [c for c in [
        *(CORE_TRAITS_MM if units == "mm" else CORE_TRAITS_PX), *SHAPE_TRAITS
    ] if c in seeds.columns and seeds[c].notna().any()]

    result = RunResult(
        seeds=seeds,
        qc=qc,
        detection=detection,
        checks=checks,
        outline_summary=outline_summary,
        lot_stats=lotstats.lot_summary(seeds, traits),
        per_image=lotstats.per_image_summary(seeds, traits[:6]),
        composition=lotstats.composition(seeds),
        tsw=s5.thousand_seed_weight(seeds),
        annotated=annotated,
        contours=collect["contours"],
        efd=collect["efd"],
        pca_model=pca_model,
        pca_keys=keys,
        config=cfg.to_dict(),
        notes=s5.assumption_notes(cfg.scale, cfg.physical),
        timing={"seconds": time.time() - t0, "images": len(images), "seeds": len(seeds)},
    )
    progress("outputs", 1.0, "Done")
    return result


def repeatability_for(images: list[tuple[str, np.ndarray]], detector: Detector,
                      cfg: RunConfig, image_name: str, traits: list[str]):
    """Run panel 6's test-retest on one photograph from the batch."""
    img = dict(images).get(image_name)
    if img is None:
        return pd.DataFrame(), pd.DataFrame()

    def measure(image, label):
        return measure_image(image, label, detector, cfg)

    return s6.repeatability(img, image_name, measure, cfg.reliability, traits)
