"""
Compatibility shim for the previous single-file pipeline.

The measurement code now lives in `seedvision/`, one module per workflow panel.
Anything that imported `pipeline` — a notebook, a script, an older branch —
keeps working through the wrappers below, but new code should import from
`seedvision` directly, where the configuration is typed and every panel has its
own module:

    old:  import pipeline as pl; pl.measure_seed(crop)
    new:  from seedvision import stage3_segmentation as s3, stage4_phenotypes as s4
          seg = s3.segment_seed(crop, SegmentConfig())
          s4.morphology(seg.contour, seg.mask)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from seedvision import shape as _shape
from seedvision import stage1_acquisition as _s1
from seedvision import stage3_segmentation as _s3
from seedvision import stage4_phenotypes as _s4
from seedvision import stage5_physical as _s5
from seedvision import stage6_reliability as _s6
from seedvision.config import (
    ColourConfig, PatternRules, PhysicalConfig, QCConfig, ReliabilityConfig,
    ScaleConfig, SegmentConfig,
)

_WARNED = False


def _deprecated(new_home: str) -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            f"`pipeline` is a compatibility shim; this function now lives in {new_home}.",
            DeprecationWarning, stacklevel=3,
        )
        _WARNED = True


DEFAULT_QC = dict(
    focus_min=QCConfig.focus_min, clip_max=QCConfig.clip_max,
    illum_cv_max=QCConfig.illum_cv_max, seed_min_px=QCConfig.seed_min_px,
)
DEFAULT_PATTERN_RULES = dict(
    black_L_max=PatternRules.black_L_max,
    plain_coverage_max=PatternRules.plain_coverage_max,
    dotted_count_min=PatternRules.dotted_count_min,
    marbled_largest=PatternRules.marbled_largest,
    spot_min_separation=PatternRules.spot_min_separation,
)
DEFAULT_THICKNESS_FROM_WIDTH = PhysicalConfig.thickness_from_width
EFD_HARMONICS = 10


def _rules(d: dict | PatternRules) -> PatternRules:
    return d if isinstance(d, PatternRules) else PatternRules(**{
        k: v for k, v in d.items() if k in PatternRules.__dataclass_fields__
    })


def qc_static(img, name, px_per_mm=None) -> dict:
    _deprecated("seedvision.stage1_acquisition.qc_metrics")
    out = _s1.qc_metrics(img, name)
    out["px_per_mm"] = px_per_mm if px_per_mm else np.nan
    return out


def qc_flag(df_qc: pd.DataFrame, qc_cfg: dict, seed_px=None) -> pd.DataFrame:
    _deprecated("seedvision.stage1_acquisition.apply_qc_rules")
    cfg = QCConfig(**{k: v for k, v in qc_cfg.items() if k in QCConfig.__dataclass_fields__})
    cfg.min_seeds = 0
    return _s1.apply_qc_rules(df_qc, cfg, seed_width_px=seed_px)


def measure_seed(crop_bgr) -> dict | None:
    _deprecated("seedvision.stage3_segmentation + stage4_phenotypes")
    seg = _s3.segment_seed(crop_bgr, SegmentConfig())
    if seg is None:
        return None
    morph = _s4.morphology(seg.contour, seg.mask)
    if morph is None:
        return None
    morph["neighbour_frags"] = seg.fragments
    morph["_mask"] = seg.mask
    morph["_contour"] = seg.contour
    return morph


def classify_pattern(f: dict, rules) -> str:
    _deprecated("seedvision.stage4_phenotypes.classify_pattern")
    return _s4.classify_pattern(f, _rules(rules))


def colour_and_pattern(crop_bgr, mask, rules) -> dict:
    _deprecated("seedvision.stage4_phenotypes.phenotype_seed")
    r = _rules(rules)
    col = _s4.colour(crop_bgr, mask, ColourConfig())
    if not col:
        return {}
    return {**col, **_s4.coat_pattern(crop_bgr, mask, r, col)}


def elliptic_fourier(contour, harmonics: int = EFD_HARMONICS, normalize: bool = True):
    _deprecated("seedvision.shape.elliptic_fourier")
    return _shape.elliptic_fourier(contour, harmonics, normalize)


def efd_reconstruct(coeffs, n_points: int = 160):
    _deprecated("seedvision.shape.efd_reconstruct")
    return _shape.efd_reconstruct(coeffs, n_points)


def to_physical(df: pd.DataFrame, px_per_mm, thickness_from_width) -> pd.DataFrame:
    _deprecated("seedvision.stage5_physical.to_physical")
    scale = ScaleConfig(px_per_mm=px_per_mm, source="manual" if px_per_mm else "unset")
    cfg = PhysicalConfig(thickness_from_width=thickness_from_width)
    return _s5.to_physical(df, scale, cfg)


def detection_checks(df: pd.DataFrame, iou_dup: float = 0.30) -> pd.DataFrame:
    _deprecated("seedvision.stage6_reliability.detection_checks")
    d = df.copy()
    if "neighbour_frags" in d.columns and "fragments" not in d.columns:
        d["fragments"] = d["neighbour_frags"]
    if "confidence" not in d.columns:
        d["confidence"] = 1.0
    return _s6.detection_checks(d, ReliabilityConfig(dup_iou=iou_dup))
