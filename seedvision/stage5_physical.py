"""
Panel 5 — derived physical properties.

Once the scale is known, pixel measurements become millimetres and the
engineering properties that seed literature reports follow: geometric and
arithmetic mean diameter, sphericity, surface area, volume, mass, and
thousand-seed weight.

Two honesty rules run through this module:

  * Thickness cannot be seen from overhead. It is width x a ratio, and the
    ratio is a literature figure until someone calipers a sub-sample. Every
    quantity that depends on it — GMD, sphericity, volume, mass, TSW — inherits
    that assumption, and `assumption_notes()` says so in the export.
  * Without a scale, the millimetre columns are absent rather than filled with
    numbers that look like measurements.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import PhysicalConfig, ScaleConfig

MM_COLUMNS = [
    "length_mm", "width_mm", "thickness_mm", "area_mm2", "perimeter_mm",
    "feret_max_mm", "eq_diameter_mm", "gmd_mm", "amd_mm", "sphericity",
    "surface_area_mm2", "volume_mm3", "mass_mg",
]


def to_physical(df: pd.DataFrame, scale: ScaleConfig, cfg: PhysicalConfig) -> pd.DataFrame:
    """Add the millimetre block. Columns are NaN when the run is uncalibrated."""
    d = df.copy()
    if not scale.calibrated:
        for c in MM_COLUMNS:
            d[c] = np.nan
        d["units"] = "px"
        return d

    s = float(scale.px_per_mm)
    d["length_mm"] = d["length_px"] / s
    d["width_mm"] = d["width_px"] / s
    d["perimeter_mm"] = d["perimeter_px"] / s
    d["area_mm2"] = d["area_px2"] / s ** 2
    d["feret_max_mm"] = d["feret_max_px"] / s
    d["eq_diameter_mm"] = d["eq_diameter_px"] / s
    d["thickness_mm"] = d["width_mm"] * cfg.thickness_from_width

    L, W, T = d["length_mm"], d["width_mm"], d["thickness_mm"]
    d["gmd_mm"] = (L * W * T) ** (1 / 3)          # geometric mean diameter
    d["amd_mm"] = (L + W + T) / 3                 # arithmetic mean diameter
    d["sphericity"] = d["gmd_mm"] / L
    d["surface_area_mm2"] = math.pi * d["gmd_mm"] ** 2
    d["volume_mm3"] = (math.pi / 6) * L * W * T   # tri-axial ellipsoid

    density = cfg.density_g_cm3 or (
        cfg.literature_density if cfg.density_source != "unset" else None
    )
    if density:
        # g/cm3 -> mg/mm3 is a factor of one, since 1 g/cm3 = 1 mg/mm3.
        d["mass_mg"] = d["volume_mm3"] * density
    else:
        d["mass_mg"] = np.nan

    d["units"] = "mm"
    return d


def calibrate_density(volumes_mm3, sample_mass_g: float, n_seeds: int) -> float:
    """
    Fit density by weighing a counted sub-sample.

    Weigh n seeds on a balance, count them, and the density that reconciles
    their measured volumes with that mass is the one to use for the whole lot.
    Returns g/cm3.
    """
    v = np.asarray(volumes_mm3, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or n_seeds <= 0 or sample_mass_g <= 0:
        raise ValueError("Need seed volumes, a positive mass and a positive count.")
    mean_volume_mm3 = float(v.mean())
    mean_mass_mg = sample_mass_g * 1000.0 / n_seeds
    return mean_mass_mg / mean_volume_mm3


def thousand_seed_weight(df: pd.DataFrame) -> dict:
    """TSW in grams, plus the interval that the seed-to-seed spread implies."""
    m = df.get("mass_mg")
    if m is None:
        return {}
    m = pd.to_numeric(m, errors="coerce").dropna()
    if m.empty:
        return {}
    n = len(m)
    tsw = float(m.mean())          # mg per seed x 1000 seeds = mg x 1000 = g
    se = float(m.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "tsw_g": tsw,
        "tsw_se_g": se,
        "tsw_ci95_g": 1.96 * se if n > 1 else float("nan"),
        "n_seeds": int(n),
        "mean_mass_mg": float(m.mean()),
        "cv_mass_pct": float(100 * m.std(ddof=1) / m.mean()) if n > 1 and m.mean() else float("nan"),
    }


def assumption_notes(scale: ScaleConfig, cfg: PhysicalConfig) -> list[str]:
    """Plain sentences describing what the physical block rests on."""
    notes = []
    if not scale.calibrated:
        notes.append(
            "No scale was set, so sizes are in pixels and no physical property "
            "was derived."
        )
        return notes

    notes.append(
        f"Scale: {scale.px_per_mm:.2f} px/mm, set by {scale.source}."
    )
    notes.append(
        f"Thickness is width x {cfg.thickness_from_width:g}, a geometry ratio rather "
        "than a measurement. GMD, sphericity, volume, mass and TSW all inherit it."
    )
    if cfg.density_source == "weighed" and cfg.density_g_cm3:
        notes.append(
            f"Density {cfg.density_g_cm3:.3f} g/cm3, fitted from a weighed sub-sample."
        )
    elif cfg.density_source == "literature":
        notes.append(
            f"Density {cfg.literature_density:.2f} g/cm3 from literature; weigh a "
            "counted sub-sample to replace it with your lot's own figure."
        )
    else:
        notes.append("No density set, so mass and TSW were not computed.")
    return notes
