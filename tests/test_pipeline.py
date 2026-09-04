"""
Smoke test for the library, with no model and no photographs.

A synthetic tray of ellipse "seeds" is drawn, a stub detector returns their
boxes, and the whole workflow runs over them. It checks that the tables come
out the right shape and that the millimetre block is only populated when a
scale is set — the two failures that would quietly corrupt a real run.

    python -m tests.test_pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedvision import RunConfig, run_batch                      # noqa: E402
from seedvision.stage2_detection import Box                      # noqa: E402
from seedvision import stage6_reliability as s6                  # noqa: E402
from seedvision import stage7_validation as s7                   # noqa: E402
from seedvision import export, lotstats, shape                   # noqa: E402


# ---------------------------------------------------------------------------
# synthetic tray
# ---------------------------------------------------------------------------

def make_tray(seed: int = 0, n_rows: int = 5, n_cols: int = 7, spacing: int = 90):
    rng = np.random.default_rng(seed)
    h, w = n_rows * spacing + 60, n_cols * spacing + 60
    img = np.full((h, w, 3), 238, np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)

    boxes = []
    for r in range(n_rows):
        for c in range(n_cols):
            cx = 40 + c * spacing + int(rng.normal(0, 4))
            cy = 40 + r * spacing + int(rng.normal(0, 4))
            a = int(rng.normal(30, 2.5))
            b = int(a * rng.normal(0.86, 0.03))
            angle = int(rng.uniform(0, 180))
            base = (
                int(rng.uniform(60, 110)),
                int(rng.uniform(110, 150)),
                int(rng.uniform(150, 190)),
            )
            cv2.ellipse(img, (cx, cy), (a, b), angle, 0, 360, base, -1)
            for _ in range(rng.integers(0, 14)):     # coat markings
                sx = cx + int(rng.normal(0, a / 3))
                sy = cy + int(rng.normal(0, b / 3))
                cv2.circle(img, (sx, sy), int(rng.integers(1, 3)), (40, 55, 70), -1)
            boxes.append(Box(cx - a - 3, cy - b - 3, cx + a + 3, cy + b + 3,
                             float(rng.uniform(0.6, 0.95)), "Spotted seeds"))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, boxes


class StubDetector:
    """Stands in for YOLO: returns the boxes the tray was drawn from."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.names = {0: "Spotted seeds"}
        self.weights_path = "stub"

    def detect(self, img, cfg):
        key = (img.shape[0], img.shape[1])
        boxes = self.mapping.get(key)
        if boxes is None:      # a transformed copy — re-detect by thresholding
            boxes = _boxes_by_threshold(img)
        return list(boxes), dict(raw_detections=len(boxes), kept=len(boxes),
                                 duplicates_removed=0, truncated=0,
                                 strategy="stub", mean_confidence=0.8)


def _boxes_by_threshold(img):
    """Find the synthetic seeds directly, so transformed trays still work."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 150:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append(Box(x - 2, y - 2, x + w + 2, y + h + 2, 0.8, "Spotted seeds"))
    return out


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def main() -> int:
    img_a, boxes_a = make_tray(0)
    img_b, boxes_b = make_tray(1, spacing=90)
    detector = StubDetector({
        (img_a.shape[0], img_a.shape[1]): boxes_a,
    })
    images = [("tray_a.jpg", img_a), ("tray_b.jpg", img_b)]

    failures = []

    def check(label, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(label)

    # --- uncalibrated run --------------------------------------------------
    cfg = RunConfig()
    res = run_batch(images, detector, cfg)
    check("seeds measured", len(res.seeds) > 50, f"{len(res.seeds)} seeds")
    check("units are pixels", res.units == "px")
    check("mm columns empty without a scale", res.seeds.length_mm.isna().all())
    check("QC table has a row per photograph", len(res.qc) == 2)
    check("QC reasons written", res.qc.qc_reason.notna().all())
    check("outline consistency ran", "outline_error" in res.seeds.columns)
    check("detection checks ran", len(res.checks) == 2)
    check("lot statistics produced", len(res.lot_stats) > 4)
    check("shape PCA scored every seed",
          "shape_pc1" in res.seeds.columns and res.seeds.shape_pc1.notna().mean() > 0.9,
          f"{100 * res.seeds.shape_pc1.notna().mean():.0f}% scored")
    check("annotated trays returned", len(res.annotated) == 2)

    # --- calibrated run ----------------------------------------------------
    cfg2 = RunConfig()
    cfg2.scale.px_per_mm = 12.0
    cfg2.scale.source = "manual"
    cfg2.physical.density_source = "literature"
    res2 = run_batch(images, detector, cfg2)
    check("units are mm", res2.units == "mm")
    check("length in mm is sensible", 3 < res2.seeds.length_mm.mean() < 8,
          f"{res2.seeds.length_mm.mean():.2f} mm")
    check("volume computed", res2.seeds.volume_mm3.notna().all())
    check("mass computed from density", res2.seeds.mass_mg.notna().all())
    check("TSW reported", "tsw_g" in res2.tsw, f"{res2.tsw.get('tsw_g', float('nan')):.1f} g")

    # --- panel 6: repeatability -------------------------------------------
    def measure(image, label):
        from seedvision.runner import measure_image
        return measure_image(image, label, detector, cfg2)

    per_rep, per_trait = s6.repeatability(
        img_a, "tray_a.jpg", measure, cfg2.reliability, ["length_px", "area_px2", "roundness"]
    )
    check("repeatability ran four replicates", len(per_rep) == 4)
    check("length is stable under transformation",
          float(per_trait.loc[per_trait.trait == "length_px", "cv_pct"].iloc[0]) < 5,
          f"CV {per_trait.loc[per_trait.trait == 'length_px', 'cv_pct'].iloc[0]:.2f}%")

    # --- panel 6: position drift ------------------------------------------
    drift, grid = s6.position_drift(res2.seeds, ["length_px", "brightness"], cfg2.reliability)
    check("position drift table built", len(drift) == 2)
    check("no false drift on random layout",
          bool((drift.r2 < 0.25).all()), f"max R2 {drift.r2.max():.3f}")

    # --- panel 6: seed matching -------------------------------------------
    a = res2.seeds[res2.seeds.image == "tray_a.jpg"]
    matched = s6.match_seeds(a, a.copy(), cfg2.reliability)
    check("a tray matches itself", len(matched) == len(a), f"{len(matched)}/{len(a)} pairs")
    summ = s6.matching_summary(matched, "length_px")
    check("self-match has zero bias", abs(summ["bias"]) < 1e-9)

    # --- panel 7: validation ----------------------------------------------
    rng = np.random.default_rng(3)
    ref = res2.seeds[["image", "seed_id", "length_mm", "width_mm"]].copy()
    ref["length_mm"] = ref.length_mm * 1.02 + rng.normal(0, 0.05, len(ref))
    ref["width_mm"] = ref.width_mm + rng.normal(0, 0.05, len(ref))
    summary, paired = s7.validate(res2.seeds, ref, cfg2.validation)
    check("validation summary built", len(summary) == 2)
    check("R2 recovers the planted relationship",
          bool((summary.r2 > 0.9).all()), f"min R2 {summary.r2.min():.3f}")
    check("bias detected on length",
          summary.loc[summary.trait == "length_mm", "bias"].iloc[0] < 0,
          "pipeline reads under the inflated reference, as planted")
    check("Bland-Altman frame built", "difference" in paired["length_mm"].columns)
    check("caliper template offered",
          len(s7.reference_template(res2.seeds, 20)) == 20)

    # --- outputs -----------------------------------------------------------
    check("GWAS-ready wide table",
          len(lotstats.gwas_wide(res2.seeds, ["length_mm", "width_mm"], "lot-1")) == 1)
    check("shape atlas has a mean outline", shape.mean_shape(res2.efd) is not None)
    check("PC extremes drawable",
          shape.shape_along_pc(res2.pca_model, cfg2.shape.harmonics, 0) is not None)
    check("CSV export", len(export.to_csv_bytes(res2.seeds)) > 1000)
    check("Excel export", export.to_excel_bytes(res2, summary)[:2] == b"PK")
    check("JSON export", b'"units": "mm"' in export.to_json_bytes(res2, summary))
    zip_bytes = export.to_zip_bytes(res2, summary)
    check("ZIP bundle with images", len(zip_bytes) > 20000, f"{len(zip_bytes) / 1e3:.0f} kB")

    # --- edge cases --------------------------------------------------------
    blank = np.full((300, 300, 3), 240, np.uint8)
    res3 = run_batch([("blank.jpg", blank)], StubDetector({}), RunConfig())
    check("blank photograph handled", res3.seeds.empty and res3.notes)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    raise SystemExit(main())
