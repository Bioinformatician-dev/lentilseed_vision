"""
Panel 1 — image acquisition & quality control.

Everything here runs before the detector, on the photograph alone: focus,
illumination, clipping, resolution, and the pixels-per-millimetre scale.
Seed size is the one check that needs detections, so `apply_qc_rules` takes it
as an argument and is called once the run has boxes.

A failing photograph is flagged and carried through, never silently dropped —
the operator decides whether to reshoot.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .config import QCConfig, ScaleConfig


# ---------------------------------------------------------------------------
# quality metrics
# ---------------------------------------------------------------------------

def qc_metrics(img_bgr: np.ndarray, name: str) -> dict:
    """Measure one photograph. No thresholds applied — that is `apply_qc_rules`."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    clip_low = float((gray <= 2).mean())
    clip_high = float((gray >= 253).mean())

    # Coarse 8x8 background map: blur away the seeds, then look at what light
    # is left. A vignette or a side lamp shows up as a high CV here.
    coarse = cv2.resize(
        cv2.GaussianBlur(gray, (0, 0), max(w, h) / 25.0),
        (8, 8), interpolation=cv2.INTER_AREA,
    ).astype(float)
    illum_cv = float(coarse.std() / coarse.mean()) if coarse.mean() else np.nan
    illum_range = float((coarse.max() - coarse.min()) / 255.0)

    # Colour cast: a neutral tray under neutral light sits near a* = b* = 0.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    cast = float(np.hypot(lab[..., 1].mean() - 128.0, lab[..., 2].mean() - 128.0))

    return dict(
        image=name,
        width=w,
        height=h,
        megapixels=round(w * h / 1e6, 2),
        focus=focus,
        clip_low=clip_low,
        clip_high=clip_high,
        clip_frac=clip_low + clip_high,
        illum_cv=illum_cv,
        illum_range=illum_range,
        colour_cast=cast,
        mean_intensity=float(gray.mean()),
    )


def illumination_field(img_bgr: np.ndarray, grid: int = 24) -> np.ndarray:
    """Background light as a small grid, for the uniformity heat map in the UI."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    field = cv2.resize(
        cv2.GaussianBlur(gray, (0, 0), max(w, h) / 30.0),
        (grid, grid), interpolation=cv2.INTER_AREA,
    ).astype(float)
    return field / max(field.mean(), 1e-6)


# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------

_REASONS = {
    "flag_focus": "out of focus",
    "flag_clip": "blown highlights or crushed shadows",
    "flag_illum": "uneven lighting",
    "flag_size": "seeds too small in frame",
    "flag_sparse": "too few seeds to summarise",
}


def apply_qc_rules(
    df_qc: pd.DataFrame,
    cfg: QCConfig,
    seed_width_px: pd.Series | None = None,
    seed_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Turn metrics into pass/fail flags plus a plain-language reason."""
    df = df_qc.copy()
    df["seed_width_px"] = (
        df["image"].map(seed_width_px) if seed_width_px is not None else np.nan
    )
    df["n_seeds"] = (
        df["image"].map(seed_counts).fillna(0).astype(int)
        if seed_counts is not None else 0
    )

    df["flag_focus"] = df["focus"] < cfg.focus_min
    df["flag_clip"] = df["clip_frac"] > cfg.clip_max
    df["flag_illum"] = df["illum_cv"] > cfg.illum_cv_max
    df["flag_size"] = df["seed_width_px"].fillna(np.inf) < cfg.seed_min_px
    df["flag_sparse"] = df["n_seeds"] < cfg.min_seeds

    flags = list(_REASONS)
    df["qc_pass"] = ~df[flags].any(axis=1)
    df["qc_reason"] = [
        "; ".join(_REASONS[f] for f in flags if row[f]) or "passes every check"
        for _, row in df.iterrows()
    ]
    # A photograph can pass and still be marginal; say so rather than implying
    # every passing image is equally good.
    df["size_grade"] = np.where(
        df["seed_width_px"] >= cfg.seed_good_px, "good",
        np.where(df["seed_width_px"] >= cfg.seed_min_px, "workable", "too small"),
    )
    return df


# ---------------------------------------------------------------------------
# scale calibration
# ---------------------------------------------------------------------------

def scale_from_reference(pixel_length: float, real_mm: float) -> ScaleConfig:
    """Calibrate from any object of known size: a ruler span, a coin, a card."""
    if pixel_length <= 0 or real_mm <= 0:
        raise ValueError("Both the pixel length and the real length must be positive.")
    return ScaleConfig(
        px_per_mm=pixel_length / real_mm,
        source="reference",
        reference_mm=real_mm,
        note=f"{pixel_length:.0f} px measured across {real_mm:g} mm",
    )


def scale_from_ruler_strip(strip_bgr: np.ndarray, mm_per_tick: float = 1.0) -> ScaleConfig:
    """
    Read tick spacing straight off a cropped ruler strip.

    The strip is collapsed along its short axis into a one-dimensional profile;
    evenly spaced ticks then show up as a single dominant frequency, which the
    autocorrelation of that profile recovers. Works on a plain millimetre rule
    photographed square-on, which is the case the workflow assumes.
    """
    if strip_bgr.size == 0:
        raise ValueError("The ruler crop is empty.")
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    if gray.shape[0] > gray.shape[1]:      # ticks should run left to right
        gray = gray.T

    profile = gray.mean(axis=0)
    profile -= cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 12).ravel()
    profile -= profile.mean()
    if profile.std() < 1e-6:
        raise ValueError("No tick contrast in that crop — include the marks themselves.")

    ac = np.correlate(profile, profile, mode="full")[len(profile) - 1:]
    ac /= ac[0]
    lo = max(3, int(0.004 * len(profile)))
    hi = max(lo + 2, int(0.25 * len(profile)))
    window = ac[lo:hi]
    peak = int(np.argmax(window)) + lo
    if ac[peak] < 0.20:
        raise ValueError("Tick spacing was not clear enough to trust — calibrate manually.")

    px_per_tick = float(peak)
    return ScaleConfig(
        px_per_mm=px_per_tick / mm_per_tick,
        source="ruler",
        reference_mm=mm_per_tick,
        note=f"{px_per_tick:.1f} px per {mm_per_tick:g} mm tick "
             f"(autocorrelation {ac[peak]:.2f})",
    )


def scale_summary(scale: ScaleConfig) -> str:
    if not scale.calibrated:
        return "Not calibrated — sizes stay in pixels."
    return f"{scale.px_per_mm:.2f} px/mm · {scale.source}" + (
        f" · {scale.note}" if scale.note else ""
    )
