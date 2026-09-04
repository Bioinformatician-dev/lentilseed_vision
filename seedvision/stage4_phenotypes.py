"""
Panel 4 — image-derived phenotypes.

Three readings per seed, all taken from the mask and contour that panel 3
produced, all still in pixels:

  A. morphology — length, width, area, perimeter, Feret, ellipse fit, roundness
  B. colour     — mean RGB, CIE-Lab, hue, saturation, texture
  C. coat pattern — spot count, coverage, largest spot, contrast, pattern class

The pattern class is derived from the pixels alone and never consults the
detector's class label, so the two are independent readings of the same seed
and disagreement between them is a signal worth inspecting.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .config import ColourConfig, PatternRules


# ---------------------------------------------------------------------------
# A. morphology
# ---------------------------------------------------------------------------

def morphology(contour: np.ndarray, mask: np.ndarray) -> dict | None:
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, True))
    if area < 20 or perim <= 0:
        return None

    (_, _), (w_r, h_r), rect_angle = cv2.minAreaRect(contour)
    length, width = float(max(w_r, h_r)), float(min(w_r, h_r))
    if width <= 0:
        return None

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perim = float(cv2.arcLength(hull, True))

    # Maximum Feret diameter: the longest distance between any two hull points.
    hp = hull.reshape(-1, 2).astype(float)
    feret_max = float(np.max(np.linalg.norm(hp[:, None, :] - hp[None, :, :], axis=-1)))

    if len(contour) >= 5:
        (_, _), (e1, e2), ell_angle = cv2.fitEllipse(contour)
        ell_major, ell_minor = float(max(e1, e2)), float(min(e1, e2))
    else:
        ell_major, ell_minor, ell_angle = length, width, float("nan")

    ecc = (
        math.sqrt(max(0.0, 1 - (ell_minor / ell_major) ** 2))
        if ell_major > 0 else float("nan")
    )
    ell_area_pred = math.pi / 4 * length * width
    bbox_area = length * width

    return {
        "area_px2": area,
        "perimeter_px": perim,
        "length_px": length,
        "width_px": width,
        "feret_max_px": feret_max,
        "feret_min_px": width,
        "ellipse_major_px": ell_major,
        "ellipse_minor_px": ell_minor,
        "ellipse_angle": float(ell_angle),
        "rect_angle": float(rect_angle),
        "radius_px": float(cv2.minEnclosingCircle(contour)[1]),
        "eq_radius_px": math.sqrt(area / math.pi),
        "eq_diameter_px": 2 * math.sqrt(area / math.pi),
        "roundness": 4 * math.pi * area / perim ** 2,
        "aspect_ratio": length / width,
        "elongation": 1 - width / length,
        "solidity": area / hull_area if hull_area else float("nan"),
        "convexity": hull_perim / perim if perim else float("nan"),
        "extent": area / bbox_area if bbox_area else float("nan"),
        "eccentricity": ecc,
        "area_ratio": area / ell_area_pred if ell_area_pred else float("nan"),
        "mask_px": int((mask > 0).sum()),
    }


# ---------------------------------------------------------------------------
# B. colour
# ---------------------------------------------------------------------------

def colour(roi_bgr: np.ndarray, mask: np.ndarray, cfg: ColourConfig) -> dict:
    """Sample colour a step inside the rim, where the edge pixels are not mixed."""
    core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=cfg.erode_iter)
    m = (core if (core > 0).sum() >= 10 else mask).astype(bool)
    if m.sum() < 10:
        return {}

    bgr = roi_bgr[m]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)[m]
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[m]
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(float)[m]

    L = lab[:, 0] * 100.0 / 255.0
    A = lab[:, 1] - 128.0
    B = lab[:, 2] - 128.0

    return {
        "mean_R": float(bgr[:, 2].mean()),
        "mean_G": float(bgr[:, 1].mean()),
        "mean_B": float(bgr[:, 0].mean()),
        "mean_L": float(L.mean()),
        "mean_a": float(A.mean()),
        "mean_b": float(B.mean()),
        "chroma": float(np.hypot(A, B).mean()),
        "hue_angle": float(math.degrees(math.atan2(B.mean(), A.mean())) % 360),
        "mean_hue": float(hsv[:, 0].mean()),
        "saturation": float(hsv[:, 1].mean()),
        "brightness": float(gray.mean()),
        "texture_std": float(gray.std()),
        "hex": "#%02X%02X%02X" % (
            int(bgr[:, 2].mean()), int(bgr[:, 1].mean()), int(bgr[:, 0].mean())
        ),
    }


# ---------------------------------------------------------------------------
# C. coat pattern
# ---------------------------------------------------------------------------

def classify_pattern(f: dict, rules: PatternRules) -> str:
    if f.get("mean_L", 100) < rules.black_L_max:
        return "black"
    if f.get("largest_spot", 0) > rules.marbled_largest:
        return "marbled"
    if f.get("spot_count", 0) == 0 or f.get("spot_coverage", 0) < rules.plain_coverage_max:
        return "plain"
    if f["spot_count"] >= rules.dotted_count_min:
        return "dotted"
    return "spotted"


def coat_pattern(roi_bgr: np.ndarray, mask: np.ndarray, rules: PatternRules,
                 colour_stats: dict) -> dict:
    """Split the coat into base and markings, then describe the markings."""
    m = mask.astype(bool)
    if m.sum() < 10:
        return {}

    gray_full = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lab_full = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(float)
    gray = gray_full[m]

    # Otsu again, this time *within* the seed, to separate markings from coat.
    thr, _ = cv2.threshold(gray.astype(np.uint8).reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark, light = gray[gray <= thr], gray[gray > thr]
    if len(dark) == 0 or len(light) == 0 or \
            (light.mean() - dark.mean()) < rules.spot_min_separation:
        thr = -1     # one population: the coat is plain, there is nothing to split

    spot_mask = np.zeros_like(gray_full, dtype=np.uint8)
    spot_mask[m] = (gray <= thr).astype(np.uint8) * 255
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    seed_area = float(m.sum())
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats((spot_mask > 0).astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_lab > 1 else np.array([])
    areas = areas[areas >= rules.spot_min_area_px]

    # If "markings" cover almost the whole seed, the split found shading, not spots.
    if seed_area and areas.sum() / seed_area > 0.90:
        areas = np.array([])
        spot_mask[:] = 0

    spot_px = spot_mask.astype(bool) & m
    base_px = m & ~spot_px
    if spot_px.sum() >= 5 and base_px.sum() >= 5:
        s, b = lab_full[spot_px], lab_full[base_px]
        d = np.array([
            (s[:, 0].mean() - b[:, 0].mean()) * 100.0 / 255.0,
            s[:, 1].mean() - b[:, 1].mean(),
            s[:, 2].mean() - b[:, 2].mean(),
        ])
        delta_e = float(np.sqrt((d ** 2).sum()))
    else:
        delta_e = 0.0

    out = {
        "spot_count": int(len(areas)),
        "spot_coverage": float(areas.sum() / seed_area) if seed_area else float("nan"),
        "largest_spot": float(areas.max() / seed_area) if len(areas) and seed_area else 0.0,
        "mean_spot_px": float(areas.mean()) if len(areas) else 0.0,
        "spot_contrast_dE": delta_e,
        "dark_fraction": float((gray < gray.mean() - gray.std()).mean()),
    }
    out["pattern_class"] = classify_pattern({**colour_stats, **out}, rules)
    return out


# ---------------------------------------------------------------------------
# one call per seed
# ---------------------------------------------------------------------------

def phenotype_seed(roi_bgr, mask, contour, pattern_rules: PatternRules,
                   colour_cfg: ColourConfig) -> dict | None:
    morph = morphology(contour, mask)
    if morph is None:
        return None
    col = colour(roi_bgr, mask, colour_cfg)
    pat = coat_pattern(roi_bgr, mask, pattern_rules, col)
    return {**morph, **col, **pat}
