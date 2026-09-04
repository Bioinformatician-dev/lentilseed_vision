"""
Panel 7 — external validation.

Self-consistency says the pipeline repeats itself. It does not say the pipeline
is right. For that, a sub-sample gets measured with digital calipers and the
two sets of numbers are compared: regression, R2, RMSE, MAE, bias, Lin's
concordance, and Bland-Altman limits of agreement.

Correlation alone is not agreement. A pipeline that reads every seed 8% long
correlates almost perfectly with the calipers and is still wrong by 8%, which
is why bias and the limits of agreement are reported next to R2 rather than
underneath it.

Reference file: a CSV with one row per measured seed and columns
`image`, `seed_id`, and one column per caliper trait (`length_mm`,
`width_mm`, ...). If seed ids were not recorded, join on rank instead and the
seeds are paired by size order within each photograph.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ValidationConfig


# ---------------------------------------------------------------------------
# joining
# ---------------------------------------------------------------------------

def join_reference(df_seeds: pd.DataFrame, df_ref: pd.DataFrame,
                   trait: str, how: str = "seed_id") -> pd.DataFrame:
    """Pair pipeline rows with caliper rows for one trait."""
    if trait not in df_seeds.columns or trait not in df_ref.columns:
        return pd.DataFrame()

    left = df_seeds[["image", "seed_id", trait]].rename(columns={trait: "measured"})
    right = df_ref[["image", "seed_id", trait]].rename(columns={trait: "reference"}) \
        if "seed_id" in df_ref.columns else df_ref[["image", trait]].rename(columns={trait: "reference"})

    if how == "seed_id" and "seed_id" in right.columns:
        out = left.merge(right, on=["image", "seed_id"], how="inner")
    else:
        # Rank join: sort both by the trait within each photograph and pair in
        # order. Right for a sub-sample measured without recording which seed
        # was which, wrong if the two sets are not the same seeds.
        left = left.sort_values(["image", "measured"]).copy()
        right = right.sort_values(["image", "reference"]).copy()
        left["rank"] = left.groupby("image").cumcount()
        right["rank"] = right.groupby("image").cumcount()
        out = left.merge(right[["image", "rank", "reference"]], on=["image", "rank"], how="inner")

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["measured", "reference"])
    return out


# ---------------------------------------------------------------------------
# agreement statistics
# ---------------------------------------------------------------------------

def agreement(measured, reference, loa_z: float = 1.96) -> dict:
    m = np.asarray(measured, dtype=float)
    r = np.asarray(reference, dtype=float)
    keep = np.isfinite(m) & np.isfinite(r)
    m, r = m[keep], r[keep]
    n = len(m)
    if n < 3:
        return {"n": n, "note": "at least three paired seeds are needed"}

    diff = m - r
    bias = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))

    slope, intercept = np.polyfit(r, m, 1)
    pred = slope * r + intercept
    ss_res = float(((m - pred) ** 2).sum())
    ss_tot = float(((m - m.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    pearson = float(np.corrcoef(m, r)[0, 1])

    # Lin's concordance: how far the points fall from the 1:1 line, not from
    # the best-fit line. This is the number that catches a constant offset.
    var_m, var_r = m.var(ddof=1), r.var(ddof=1)
    cov = float(np.cov(m, r, ddof=1)[0, 1])
    denom = var_m + var_r + (m.mean() - r.mean()) ** 2
    ccc = float(2 * cov / denom) if denom else float("nan")

    return {
        "n": n,
        "r2": float(r2),
        "pearson_r": pearson,
        "ccc": ccc,
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "mae": float(np.abs(diff).mean()),
        "bias": bias,
        "bias_pct": float(100 * bias / r.mean()) if r.mean() else float("nan"),
        "sd_of_difference": sd_diff,
        "loa_low": bias - loa_z * sd_diff,
        "loa_high": bias + loa_z * sd_diff,
        "slope": float(slope),
        "intercept": float(intercept),
        "reference_mean": float(r.mean()),
        "measured_mean": float(m.mean()),
    }


def bland_altman(paired: pd.DataFrame) -> pd.DataFrame:
    """Per-pair points for the Bland-Altman plot."""
    d = paired.copy()
    d["mean_of_pair"] = (d.measured + d.reference) / 2
    d["difference"] = d.measured - d.reference
    return d


def validate(df_seeds: pd.DataFrame, df_ref: pd.DataFrame,
             cfg: ValidationConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run every requested trait. Returns a summary table and the paired frames."""
    rows, paired_frames = [], {}
    for trait in cfg.traits:
        paired = join_reference(df_seeds, df_ref, trait, how=cfg.join_on)
        if paired.empty:
            continue
        stats = agreement(paired.measured, paired.reference, cfg.loa_z)
        stats["trait"] = trait
        rows.append(stats)
        paired_frames[trait] = bland_altman(paired)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        cols = ["trait", "n", "r2", "ccc", "rmse", "mae", "bias", "bias_pct",
                "loa_low", "loa_high", "slope", "intercept"]
        summary = summary[[c for c in cols if c in summary.columns]]
        summary["verdict"] = [interpret(r) for _, r in summary.iterrows()]
    return summary, paired_frames


def interpret(stats) -> str:
    """One sentence a reviewer can read without opening the plot."""
    r2 = stats.get("r2", np.nan)
    ccc = stats.get("ccc", np.nan)
    bias_pct = stats.get("bias_pct", np.nan)

    if not np.isfinite(r2):
        return "not enough paired seeds"
    if abs(bias_pct) > 5:
        direction = "over" if bias_pct > 0 else "under"
        return f"tracks the calipers but reads {abs(bias_pct):.1f}% {direction} — correct the scale"
    if ccc >= 0.95 and r2 >= 0.95:
        return "agrees with the calipers"
    if r2 >= 0.90 > ccc:
        return "correlated but offset — the bias, not the spread, is the problem"
    if r2 >= 0.80:
        return "usable, with more scatter than a caliper study should show"
    return "does not agree — check the scale and the segmentation before using these numbers"


def reference_template(df_seeds: pd.DataFrame, n: int = 20,
                       traits: tuple[str, ...] = ("length_mm", "width_mm")) -> pd.DataFrame:
    """
    A blank caliper sheet for a random sub-sample.

    Print it, measure those seeds, type the numbers in, upload. Sampling at
    random rather than by size keeps the validation honest at both ends of the
    distribution.
    """
    if df_seeds.empty:
        return pd.DataFrame(columns=["image", "seed_id", *traits])
    take = min(n, len(df_seeds))
    sample = df_seeds.sample(take, random_state=0).sort_values(["image", "seed_id"])
    out = sample[["image", "seed_id"]].copy()
    for t in traits:
        out[t] = ""
    return out.reset_index(drop=True)
