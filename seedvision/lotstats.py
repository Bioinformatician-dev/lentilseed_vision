"""
Final outputs — lot-level statistics.

The per-seed table is the raw material; what a breeder reports is the lot.
This module produces the three shapes that get used downstream: a trait
summary with spread and percentiles, the class and pattern composition, and a
one-row-per-lot wide table with the column names a GWAS pipeline expects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(df: pd.DataFrame, trait: str) -> pd.Series:
    return pd.to_numeric(df[trait], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def lot_summary(df: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    """Mean, spread and percentiles per trait, with the CV that reports use."""
    rows = []
    for t in traits:
        if t not in df.columns:
            continue
        v = _numeric(df, t)
        if v.empty:
            continue
        mean = float(v.mean())
        sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        rows.append({
            "trait": t,
            "n": int(len(v)),
            "mean": mean,
            "sd": sd,
            "cv_pct": 100 * sd / mean if mean else float("nan"),
            "se": sd / np.sqrt(len(v)) if len(v) > 1 else float("nan"),
            "min": float(v.min()),
            "p5": float(v.quantile(0.05)),
            "p25": float(v.quantile(0.25)),
            "median": float(v.median()),
            "p75": float(v.quantile(0.75)),
            "p95": float(v.quantile(0.95)),
            "max": float(v.max()),
            "skew": float(v.skew()) if len(v) > 2 else float("nan"),
        })
    return pd.DataFrame(rows)


def per_image_summary(df: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    """The same summary, one row per photograph — for spotting an odd tray."""
    keep = [t for t in traits if t in df.columns]
    if not keep:
        return pd.DataFrame()
    g = df.groupby("image")
    out = g[keep].agg(["count", "mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.insert(0, "n_seeds", g.size())
    return out.reset_index()


def composition(df: pd.DataFrame) -> pd.DataFrame:
    """Detector class against pixel-derived coat pattern, as counts and shares."""
    if "cls" not in df.columns:
        return pd.DataFrame()
    counts = df["cls"].value_counts().rename_axis("class").reset_index(name="seeds")
    counts["share_pct"] = 100 * counts.seeds / counts.seeds.sum()
    if "pattern_class" in df.columns:
        pattern = (
            df.groupby("cls")["pattern_class"]
            .agg(lambda s: s.value_counts().idxmax() if len(s) else "")
            .rename("commonest_coat_pattern")
        )
        counts = counts.merge(pattern, left_on="class", right_index=True, how="left")
    return counts


def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Where the detector's label and the coat pattern read differently."""
    if "cls" not in df.columns or "pattern_class" not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df["cls"], df["pattern_class"])


def gwas_wide(df: pd.DataFrame, traits: list[str], lot_id: str,
              by: str = "lot") -> pd.DataFrame:
    """
    One row per lot (or per photograph), one column per trait.

    This is the shape association pipelines want: an identifier column followed
    by trait means, with `_sd` and `_n` alongside so a downstream model can
    weight by how well each mean was estimated.
    """
    keep = [t for t in traits if t in df.columns]
    if not keep:
        return pd.DataFrame()

    def _row(sub: pd.DataFrame, ident: str) -> dict:
        rec = {"id": ident, "n_seeds": int(len(sub))}
        for t in keep:
            v = _numeric(sub, t)
            rec[t] = float(v.mean()) if len(v) else np.nan
            rec[f"{t}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        return rec

    if by == "image":
        return pd.DataFrame([_row(g, name) for name, g in df.groupby("image")])
    return pd.DataFrame([_row(df, lot_id)])


def headline(df: pd.DataFrame, units: str) -> list[tuple[str, str]]:
    """The four or five numbers that belong at the top of the results page."""
    out: list[tuple[str, str]] = [
        (f"{len(df):,}", "seeds measured"),
        (f"{df.image.nunique()}", "photographs"),
    ]
    length_col = "length_mm" if units == "mm" and "length_mm" in df.columns else "length_px"
    v = _numeric(df, length_col) if length_col in df.columns else pd.Series(dtype=float)
    if not v.empty:
        suffix = "mm" if length_col.endswith("mm") else "px"
        out.append((f"{v.mean():.2f} {suffix}", "mean length"))
        out.append((f"{100 * v.std(ddof=1) / v.mean():.1f}%", "length CV"))
    if "pattern_class" in df.columns and len(df):
        top = df.pattern_class.value_counts()
        out.append((f"{top.index[0]}", f"commonest coat ({100 * top.iloc[0] / len(df):.0f}%)"))
    return out
