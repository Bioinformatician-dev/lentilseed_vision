"""
Final outputs — export.

Four shapes leave the app: the per-seed CSV, an Excel workbook with a sheet per
table, a JSON record, and a ZIP that also carries the annotated photographs.
Every one of them travels with the run configuration and a data dictionary, so
a table found on a shared drive in two years still explains itself.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import cv2
import pandas as pd

# Columns that only matter inside a run.
_INTERNAL = ("_contour", "_mask")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith(_INTERNAL)], errors="ignore")


DATA_DICTIONARY: list[tuple[str, str, str]] = [
    # column, units, meaning
    ("image", "", "file name of the photograph"),
    ("seed_id", "", "index of the seed within that photograph"),
    ("cls", "", "class predicted by the detector"),
    ("confidence", "0-1", "detector confidence for that class"),
    ("cx, cy", "px", "seed centre in the frame, used for the position-drift check"),
    ("length_px, width_px", "px", "long and short side of the minimum-area rectangle"),
    ("area_px2", "px^2", "area enclosed by the traced outline"),
    ("perimeter_px", "px", "length of the traced outline"),
    ("feret_max_px", "px", "longest distance between any two points on the outline"),
    ("roundness", "0-1", "4*pi*area / perimeter^2; 1 is a circle"),
    ("aspect_ratio", "", "length / width"),
    ("solidity", "0-1", "area / convex hull area; low means a dent or a chip"),
    ("convexity", "0-1", "hull perimeter / outline perimeter; low means a rough edge"),
    ("extent", "0-1", "area / bounding-box area"),
    ("eccentricity", "0-1", "how far the fitted ellipse is from circular"),
    ("area_ratio", "", "traced area / ellipse-predicted area; the outline-consistency check"),
    ("mean_L, mean_a, mean_b", "CIE-Lab", "coat colour, sampled inside the rim"),
    ("chroma, hue_angle", "CIE-Lab", "colourfulness and hue derived from a* and b*"),
    ("saturation, brightness", "0-255", "HSV saturation and grey level"),
    ("texture_std", "0-255", "grey-level standard deviation across the coat"),
    ("spot_count", "", "distinct markings on the coat"),
    ("spot_coverage", "0-1", "share of the coat covered by markings"),
    ("largest_spot", "0-1", "largest single marking as a share of the coat"),
    ("spot_contrast_dE", "dE", "colour distance between markings and base coat"),
    ("pattern_class", "", "plain / dotted / spotted / marbled / black, from pixels alone"),
    ("length_mm, width_mm", "mm", "size after scale calibration"),
    ("thickness_mm", "mm", "width x the thickness ratio — modelled, not measured"),
    ("gmd_mm, amd_mm", "mm", "geometric and arithmetic mean diameter"),
    ("sphericity", "0-1", "GMD / length"),
    ("surface_area_mm2", "mm^2", "sphere-equivalent surface area from GMD"),
    ("volume_mm3", "mm^3", "tri-axial ellipsoid volume"),
    ("mass_mg", "mg", "volume x density; needs a density to be set"),
    ("shape_pc1..", "", "principal component scores of the Fourier outline descriptors"),
    ("segment_method", "", "which threshold produced the outline"),
    ("outline_error", "", "area_ratio - 1; large values mean a suspect trace"),
    ("truncated", "", "the seed runs off the edge of the frame"),
]


def data_dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(DATA_DICTIONARY, columns=["column", "units", "meaning"])


def _tables(result, validation: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    tables = {
        "per_seed": clean(result.seeds),
        "lot_statistics": result.lot_stats,
        "per_image": result.per_image,
        "composition": result.composition,
        "quality_control": result.qc,
        "detection_report": result.detection,
        "detection_checks": result.checks,
        "data_dictionary": data_dictionary_frame(),
    }
    if validation is not None and not validation.empty:
        tables["external_validation"] = validation
    return {k: v for k, v in tables.items() if v is not None and not v.empty}


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return clean(df).to_csv(index=False).encode("utf-8")


def to_excel_bytes(result, validation: pd.DataFrame | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for sheet, df in _tables(result, validation).items():
            df.to_excel(xl, sheet_name=sheet[:31], index=False)
        pd.DataFrame(
            [{"setting": k, "value": json.dumps(v)} for k, v in result.config.items()]
        ).to_excel(xl, sheet_name="run_config", index=False)
    return buf.getvalue()


def to_json_bytes(result, validation: pd.DataFrame | None = None) -> bytes:
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "units": result.units,
        "n_images": int(result.seeds.image.nunique()) if not result.seeds.empty else 0,
        "n_seeds": int(len(result.seeds)),
        "thousand_seed_weight": result.tsw,
        "assumptions": result.notes,
        "config": result.config,
        "lot_statistics": json.loads(result.lot_stats.to_json(orient="records"))
        if not result.lot_stats.empty else [],
        "seeds": json.loads(clean(result.seeds).to_json(orient="records")),
    }
    if validation is not None and not validation.empty:
        payload["external_validation"] = json.loads(validation.to_json(orient="records"))
    return json.dumps(payload, indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
# annotated photographs
# ---------------------------------------------------------------------------

def annotated_bytes(img, fmt: str = "png", quality: int = 92) -> bytes:
    """One annotated tray as file bytes. PNG keeps the drawn outlines crisp."""
    if fmt == "png":
        ok, enc = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return enc.tobytes() if ok else b""


def annotated_name(image_name: str, fmt: str = "png") -> str:
    return f"{image_name.rsplit('.', 1)[0]}_annotated.{fmt}"


def annotated_zip_bytes(annotated: dict, fmt: str = "png") -> bytes:
    """Every annotated tray in one archive, for dropping into a manuscript."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, img in annotated.items():
            data = annotated_bytes(img, fmt)
            if data:
                z.writestr(annotated_name(name, fmt), data)
    return buf.getvalue()


def to_zip_bytes(result, validation: pd.DataFrame | None = None,
                 include_images: bool = True, include_report: bool = True) -> bytes:
    """Everything at once: tables, annotated trays, the report, config, README."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in _tables(result, validation).items():
            z.writestr(f"tables/{name}.csv", df.to_csv(index=False))
        z.writestr("run_config.json", json.dumps(result.config, indent=2))
        z.writestr("README.txt", _bundle_readme(result))
        if include_report:
            from .report import build_html
            z.writestr("report.html", build_html(result, validation))
        if include_images:
            for name, img in result.annotated.items():
                data = annotated_bytes(img, "jpg", 88)
                if data:
                    z.writestr(f"annotated/{annotated_name(name, 'jpg')}", data)
    return buf.getvalue()


def _bundle_readme(result) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "Lentil seed phenotyping — results bundle",
        f"Written {stamp}",
        "",
        f"{len(result.seeds)} seeds across {result.seeds.image.nunique()} photographs, "
        f"measured in {result.units}.",
        "",
        "tables/per_seed.csv        one row per seed, every trait",
        "tables/lot_statistics.csv  mean, SD, CV and percentiles per trait",
        "tables/per_image.csv       the same, split by photograph",
        "tables/quality_control.csv what each photograph passed or failed",
        "tables/detection_checks.csv duplicate, merged and edge detections",
        "tables/data_dictionary.csv what every column means",
        "annotated/                 each tray with its boxes and outlines drawn",
        "report.html                the whole run as one readable page — open it in a browser",
        "run_config.json            every setting used, for reproducing this run",
        "",
        "Assumptions behind the physical columns:",
    ]
    lines += [f"  - {n}" for n in result.notes]
    return "\n".join(lines)
