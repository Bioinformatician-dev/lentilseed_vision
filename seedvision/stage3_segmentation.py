"""
Panel 3 — seed-wise segmentation & contour extraction.

The point of the panel: threshold *inside* each detection box, not across the
tray. A local threshold sees one seed against one patch of background, so a
lamp that is brighter at the left of the tray stops mattering.

Four candidate cuts are available. `auto` tries them in order and takes the
first that produces a plausible seed — one central blob, not touching the ROI
border, filling a sensible share of the box. The method that won is recorded
per seed, because a lot where half the seeds needed the fallback is telling you
something about the photography.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SegmentConfig


@dataclass
class SegmentResult:
    mask: np.ndarray               # uint8, ROI-sized, 255 on the seed
    contour: np.ndarray            # cv2 contour, ROI coordinates
    method: str
    fill: float                    # mask area / ROI area
    border_touch: float            # share of the ROI border the mask sits on
    fragments: int                 # other blobs big enough to be a second seed
    ok: bool


def extract_roi(img: np.ndarray, box, pad: int) -> tuple[np.ndarray, int, int]:
    """Crop a padded box and report where the crop starts, for mapping back."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    cx2, cy2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
    return img[cy1:cy2, cx1:cx2], cx1, cy1


# ---------------------------------------------------------------------------
# candidate thresholds
# ---------------------------------------------------------------------------

def _otsu(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def _adaptive(gray: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    block = int(max(gray.shape) * cfg.adaptive_block_frac) | 1
    block = max(11, min(block, 255))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, cfg.adaptive_c,
    )


def _otsu_channel(roi_bgr: np.ndarray, code: int, index: int) -> np.ndarray:
    """Otsu on one channel of another colour space — rescues low-contrast coats."""
    chan = cv2.cvtColor(roi_bgr, code)[..., index]
    return _otsu(chan)


def _orient(th: np.ndarray) -> np.ndarray:
    """Make foreground mean seed: if the border is mostly lit, invert."""
    border = np.concatenate([th[0, :], th[-1, :], th[:, 0], th[:, -1]])
    return cv2.bitwise_not(th) if (border > 0).mean() > 0.5 else th


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

def _clean(th: np.ndarray, cfg: SegmentConfig) -> tuple[np.ndarray, int] | None:
    """Close gaps, keep the central blob, fill holes. Returns mask and fragments."""
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=cfg.close_iter)
    if cfg.open_iter:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=cfg.open_iter)

    n, lab, stats, _ = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8), 8)
    if n <= 1:
        return None

    h, w = th.shape
    cid = int(lab[h // 2, w // 2])
    if cid == 0:   # the box centre landed on background — fall back to the biggest blob
        cid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    main_area = stats[cid, cv2.CC_STAT_AREA]
    fragments = int((stats[1:, cv2.CC_STAT_AREA] > 0.25 * main_area).sum()) - 1

    mask = ((lab == cid).astype(np.uint8)) * 255
    # Fill interior holes — dark markings on a pale coat otherwise punch holes
    # in the seed and shrink its area.
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return mask, fragments


def _outline(mask: np.ndarray, cfg: SegmentConfig) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < cfg.min_area_px:
        return None
    if cfg.smooth_epsilon > 0 and len(cnt) > 12:
        eps = cfg.smooth_epsilon * cv2.arcLength(cnt, True)
        smoothed = cv2.approxPolyDP(cnt, eps, True)
        if len(smoothed) >= 8:     # keep enough vertices for Fourier descriptors
            cnt = smoothed
    return cnt


def _border_touch(mask: np.ndarray) -> float:
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float((border > 0).mean())


def _plausible(mask: np.ndarray, cfg: SegmentConfig) -> tuple[bool, float, float]:
    fill = float((mask > 0).mean())
    touch = _border_touch(mask)
    ok = cfg.min_fill <= fill <= cfg.max_fill and touch <= cfg.max_border_touch
    return ok, fill, touch


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

_METHOD_ORDER = ("otsu", "adaptive", "otsu_lab", "otsu_sat")


def segment_seed(roi_bgr: np.ndarray, cfg: SegmentConfig) -> SegmentResult | None:
    """Cut one seed out of its ROI. Returns None if no cut was usable."""
    if roi_bgr is None or roi_bgr.size == 0 or min(roi_bgr.shape[:2]) < 6:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    order = _METHOD_ORDER if cfg.method == "auto" else (cfg.method,)

    best: SegmentResult | None = None
    for method in order:
        if method == "otsu":
            th = _otsu(gray)
        elif method == "adaptive":
            th = _adaptive(gray, cfg)
        elif method == "otsu_lab":
            th = _otsu_channel(roi_bgr, cv2.COLOR_BGR2LAB, 0)
        elif method == "otsu_sat":
            th = _otsu_channel(roi_bgr, cv2.COLOR_BGR2HSV, 1)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

        cleaned = _clean(_orient(th), cfg)
        if cleaned is None:
            continue
        mask, fragments = cleaned
        cnt = _outline(mask, cfg)
        if cnt is None:
            continue

        ok, fill, touch = _plausible(mask, cfg)
        result = SegmentResult(mask, cnt, method, fill, touch, fragments, ok)
        if ok:
            return result
        # Hold on to the first attempt so a marginal cut is still returned,
        # flagged, rather than the seed vanishing from the table entirely.
        best = best or result

    return best


def to_global(contour: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """ROI contour to whole-image coordinates, for drawing on the tray."""
    return contour + np.array([ox, oy])
