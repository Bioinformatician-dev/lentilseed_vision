"""
Panel 6 — reliability & self-consistency checks.

These checks need no ground truth. They ask whether the pipeline agrees with
itself, which is the cheapest way to find out that a number is not measuring
what it claims to:

  * detection checks   — duplicate boxes, merged seeds, seeds cut by the frame
  * outline consistency — traced area against the area an ellipse of the same
                          length and width would have
  * image quality      — panel 1's metrics, gathered per photograph
  * repeatability      — the same tray flipped, rotated and resized; whatever
                         the numbers do is pipeline noise, not biology
  * position drift     — trait against position in frame; a gradient means the
                         lens or the lamp, not the seeds
  * seed-to-seed match  — the same seeds across two photographs, paired, giving
                         a per-seed error bar

A check that fails is not a rejection. It tells the operator which number to
distrust and, usually, what to change about the photograph.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np
import pandas as pd

from .config import ReliabilityConfig

# How a trait scales when the photograph is resized: lengths by s, areas by s^2,
# ratios not at all. Used to make the resize replicate comparable.
_TRAIT_POWER = {
    "length_px": 1, "width_px": 1, "perimeter_px": 1, "feret_max_px": 1,
    "feret_min_px": 1, "eq_radius_px": 1, "eq_diameter_px": 1, "radius_px": 1,
    "ellipse_major_px": 1, "ellipse_minor_px": 1,
    "area_px2": 2, "mask_px": 2, "mean_spot_px": 2,
}


# ---------------------------------------------------------------------------
# detection checks
# ---------------------------------------------------------------------------

def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def detection_checks(df: pd.DataFrame, cfg: ReliabilityConfig) -> pd.DataFrame:
    """Per photograph: how many detections look like counting errors."""
    rows = []
    for name, g in df.groupby("image"):
        boxes = g[["x1", "y1", "x2", "y2"]].values
        dups = sum(
            1 for i, j in itertools.combinations(range(len(boxes)), 2)
            if _iou_xyxy(boxes[i], boxes[j]) > cfg.dup_iou
        )
        med = g["area_px2"].median()
        merged = int((
            (g["area_px2"] > cfg.merged_area_factor * med) |
            (g.get("fragments", pd.Series(0, index=g.index)) > 0)
        ).sum())
        truncated = int(g.get("truncated", pd.Series(False, index=g.index)).sum())
        low_conf = int((g["confidence"] < g["confidence"].median() * 0.5).sum())

        rows.append(dict(
            image=name,
            detected=len(g),
            duplicate_pairs=dups,
            merged_suspects=merged,
            edge_truncated=truncated,
            weak_detections=low_conf,
            count_confidence=(
                "counts look clean" if dups + merged == 0
                else f"{dups + merged} of {len(g)} detections need a look"
            ),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# outline consistency
# ---------------------------------------------------------------------------

def outline_consistency(df: pd.DataFrame, cfg: ReliabilityConfig) -> tuple[pd.DataFrame, dict]:
    """
    Compare each traced area with pi/4 x length x width.

    A seed is close to an ellipse, so the two should agree within a few per
    cent. They disagree when the trace leaked into a shadow, clipped the coat,
    or swallowed a neighbour.
    """
    d = df.copy()
    d["ellipse_area_px2"] = np.pi / 4 * d["length_px"] * d["width_px"]
    d["outline_error"] = d["area_px2"] / d["ellipse_area_px2"] - 1
    d["outline_flag"] = d["outline_error"].abs() > cfg.outline_tol

    ok = d["outline_error"].replace([np.inf, -np.inf], np.nan).dropna()
    summary = {
        "n": int(len(ok)),
        "median_error_pct": float(100 * ok.median()) if len(ok) else float("nan"),
        "iqr_error_pct": float(100 * (ok.quantile(0.75) - ok.quantile(0.25))) if len(ok) else float("nan"),
        "flagged": int(d["outline_flag"].sum()),
        "flagged_pct": float(100 * d["outline_flag"].mean()) if len(d) else float("nan"),
        "by_method": (
            d.groupby("segment_method")["outline_flag"].mean().mul(100).round(1).to_dict()
            if "segment_method" in d.columns else {}
        ),
    }
    return d, summary


# ---------------------------------------------------------------------------
# repeatability — the same tray, transformed
# ---------------------------------------------------------------------------

def transform_image(img: np.ndarray, kind: str) -> tuple[np.ndarray, float]:
    """Return the transformed image and the linear scale factor it applied."""
    if kind == "flip_h":
        return cv2.flip(img, 1), 1.0
    if kind == "flip_v":
        return cv2.flip(img, 0), 1.0
    if kind == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1.0
    if kind == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180), 1.0
    if kind.startswith("scale_"):
        pct = float(kind.split("_")[1]) / 100.0
        h, w = img.shape[:2]
        resized = cv2.resize(img, (max(1, int(w * pct)), max(1, int(h * pct))),
                             interpolation=cv2.INTER_AREA)
        return resized, pct
    raise ValueError(f"Unknown transform: {kind}")


def repeatability(img: np.ndarray, name: str, measure_fn, cfg: ReliabilityConfig,
                  traits: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Measure one tray four ways and compare the lot means.

    `measure_fn(image, label) -> DataFrame` is the per-image half of the run,
    handed in so this module never has to know about the detector. Linear
    traits are rescaled back before comparison, so the resize replicate is
    testing the pipeline rather than arithmetic.

    Returns (per-replicate table, per-trait summary). The summary's CV is the
    pipeline's own noise floor: a difference between two lots smaller than this
    is not a difference.
    """
    replicates = [("original", img, 1.0)]
    for kind in cfg.repeat_transforms:
        timg, factor = transform_image(img, kind)
        replicates.append((kind, timg, factor))

    rows = []
    for label, rimg, factor in replicates:
        df = measure_fn(rimg, f"{name}::{label}")
        rec = {"replicate": label, "n_seeds": int(len(df))}
        for t in traits:
            if t not in df.columns:
                continue
            power = _TRAIT_POWER.get(t, 0)
            corrected = df[t] / (factor ** power) if power else df[t]
            rec[t] = float(pd.to_numeric(corrected, errors="coerce").mean())
        rows.append(rec)

    per_rep = pd.DataFrame(rows)

    summary = []
    for t in traits:
        if t not in per_rep.columns:
            continue
        vals = pd.to_numeric(per_rep[t], errors="coerce").dropna()
        if len(vals) < 2 or not np.isfinite(vals.mean()) or vals.mean() == 0:
            continue
        base = float(per_rep.loc[per_rep.replicate == "original", t].iloc[0])
        summary.append(dict(
            trait=t,
            original=base,
            mean=float(vals.mean()),
            sd=float(vals.std(ddof=1)),
            cv_pct=float(100 * vals.std(ddof=1) / abs(vals.mean())),
            max_shift_pct=float(100 * (vals - base).abs().max() / abs(base)) if base else np.nan,
        ))
    per_trait = pd.DataFrame(summary)
    if not per_trait.empty:
        per_trait["verdict"] = np.where(
            per_trait.cv_pct < 1, "stable",
            np.where(per_trait.cv_pct < 3, "acceptable", "noisy"),
        )
    count_cv = (
        float(100 * per_rep.n_seeds.std(ddof=1) / per_rep.n_seeds.mean())
        if per_rep.n_seeds.mean() else float("nan")
    )
    per_rep.attrs["count_cv_pct"] = count_cv
    return per_rep, per_trait


# ---------------------------------------------------------------------------
# position drift across the frame
# ---------------------------------------------------------------------------

def position_drift(df: pd.DataFrame, traits: list[str],
                   cfg: ReliabilityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Regress each trait on where the seed sat in the frame.

    Seeds are laid out at random, so position should explain nothing. When it
    explains something, the cause is on the camera side: vignetting, a lamp on
    one side, or barrel distortion stretching the corners.

    Returns (per-trait regression table, grid cell means for the heat map).
    """
    d = df.copy()
    d["fx"] = d["cx"] / d["img_w"]
    d["fy"] = d["cy"] / d["img_h"]

    stats = []
    for t in traits:
        if t not in d.columns:
            continue
        sub = d[["fx", "fy", t]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 12:
            continue
        y = sub[t].values.astype(float)
        X = np.column_stack([np.ones(len(sub)), sub.fx.values, sub.fy.values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        mean = float(y.mean())
        stats.append(dict(
            trait=t,
            slope_x_pct=100 * beta[1] / mean if mean else np.nan,   # % change across the frame
            slope_y_pct=100 * beta[2] / mean if mean else np.nan,
            r2=r2,
            n=len(sub),
            flag=bool(np.isfinite(r2) and r2 > cfg.drift_r2_flag),
        ))
    drift = pd.DataFrame(stats)
    if not drift.empty:
        drift["reading"] = np.where(
            drift.flag,
            "position explains part of this trait — check lighting and lens",
            "no position effect",
        )

    bins = max(2, cfg.drift_bins)
    grid_trait = next((t for t in traits if t in d.columns), None)
    if grid_trait is None:
        return drift, pd.DataFrame()
    d["col"] = np.clip((d.fx * bins).astype(int), 0, bins - 1)
    d["row"] = np.clip((d.fy * bins).astype(int), 0, bins - 1)
    grid = (
        d.groupby(["row", "col"])[grid_trait]
        .agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": grid_trait})
    )
    overall = d[grid_trait].mean()
    grid["pct_of_frame_mean"] = 100 * grid[grid_trait] / overall if overall else np.nan
    grid.attrs["trait"] = grid_trait
    return drift, grid


# ---------------------------------------------------------------------------
# seed-to-seed matching across two photographs
# ---------------------------------------------------------------------------

def match_seeds(df_a: pd.DataFrame, df_b: pd.DataFrame,
                cfg: ReliabilityConfig) -> pd.DataFrame:
    """
    Pair the same physical seeds across two photographs of one tray.

    Positions are normalised to the frame, then paired by mutual nearest
    neighbour with a distance ceiling, so a seed that moved between shots
    simply goes unmatched rather than pairing with its neighbour.
    """
    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    a = np.column_stack([df_a.cx / df_a.img_w, df_a.cy / df_a.img_h])
    b = np.column_stack([df_b.cx / df_b.img_w, df_b.cy / df_b.img_h])
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    nearest_b = dist.argmin(axis=1)
    nearest_a = dist.argmin(axis=0)
    pairs = [
        (i, j) for i, j in enumerate(nearest_b)
        if nearest_a[j] == i and dist[i, j] <= cfg.match_max_dist
    ]
    if not pairs:
        return pd.DataFrame()

    ia = [p[0] for p in pairs]
    ib = [p[1] for p in pairs]
    out = pd.DataFrame({
        "image_a": df_a.image.iloc[ia].values,
        "image_b": df_b.image.iloc[ib].values,
        "seed_a": df_a.seed_id.iloc[ia].values,
        "seed_b": df_b.seed_id.iloc[ib].values,
        "distance": dist[ia, ib],
    })
    for t in ("length_px", "width_px", "area_px2", "roundness", "length_mm", "width_mm"):
        if t in df_a.columns and t in df_b.columns:
            va = pd.to_numeric(df_a[t].iloc[ia], errors="coerce").values
            vb = pd.to_numeric(df_b[t].iloc[ib], errors="coerce").values
            out[f"{t}_a"] = va
            out[f"{t}_b"] = vb
            out[f"{t}_diff"] = vb - va
    return out


def matching_summary(matched: pd.DataFrame, trait: str) -> dict:
    """Repeatability of one trait across the matched pairs."""
    col = f"{trait}_diff"
    if matched.empty or col not in matched.columns:
        return {}
    diff = pd.to_numeric(matched[col], errors="coerce").dropna()
    mean_level = pd.concat([
        matched[f"{trait}_a"], matched[f"{trait}_b"]
    ]).astype(float).mean()
    if diff.empty or not mean_level:
        return {}
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    return {
        "trait": trait,
        "pairs": int(len(diff)),
        "bias": float(diff.mean()),
        "sd_of_difference": sd,
        "repeatability_cv_pct": 100 * sd / mean_level if mean_level else float("nan"),
        # The 1.96 x SD band inside which two measurements of one seed should sit.
        "loa_low": float(diff.mean() - 1.96 * sd),
        "loa_high": float(diff.mean() + 1.96 * sd),
    }
