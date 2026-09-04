"""
Batch phenotyping from the command line.

Same pipeline as the app, no browser. Useful when a season's photographs arrive
as a folder and the results need to land in a directory rather than a download.

    python cli.py photos/ --out results/ --px-per-mm 20.5 --density 1.30
    python cli.py photos/ --out results/ --tiled --conf 0.08

Writes per_seed.csv, lot_statistics.csv, quality_control.csv, the annotated
trays, and run_config.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from seedvision import Detector, RunConfig, run_batch
from seedvision import export

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_images(path: Path) -> list[tuple[str, "cv2.Mat"]]:
    files = (
        [path] if path.is_file()
        else sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    )
    out = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  skipped (unreadable): {f.name}", file=sys.stderr)
            continue
        out.append((f.name, img))
    return out


def build_config(args) -> RunConfig:
    cfg = RunConfig()
    cfg.lot_id = args.lot
    cfg.detect.conf = args.conf
    cfg.detect.iou = args.iou
    cfg.detect.tiled = args.tiled
    cfg.segment.method = args.segment
    if args.px_per_mm:
        cfg.scale.px_per_mm = args.px_per_mm
        cfg.scale.source = "manual"
    if args.density:
        cfg.physical.density_g_cm3 = args.density
        cfg.physical.density_source = "weighed"
    cfg.physical.thickness_from_width = args.thickness_ratio
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lentil seed phenotyping, batch mode.")
    ap.add_argument("images", type=Path, help="a photograph or a folder of them")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--weights", type=Path, default=Path(__file__).parent / "best.pt")
    ap.add_argument("--lot", default="lot-1")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--tiled", action="store_true",
                    help="slice large photographs, for seeds under about 40 px")
    ap.add_argument("--segment", default="auto",
                    choices=["auto", "otsu", "adaptive", "otsu_lab", "otsu_sat"])
    ap.add_argument("--px-per-mm", type=float, default=None)
    ap.add_argument("--density", type=float, default=None, help="g/cm3, for mass and TSW")
    ap.add_argument("--thickness-ratio", type=float, default=0.62)
    ap.add_argument("--no-images", action="store_true", help="skip writing annotated trays")
    args = ap.parse_args(argv)

    if not args.weights.exists():
        print(f"No weights at {args.weights}", file=sys.stderr)
        return 2

    images = load_images(args.images)
    if not images:
        print(f"No readable images under {args.images}", file=sys.stderr)
        return 2
    print(f"{len(images)} photograph(s) to measure")

    detector = Detector(str(args.weights))
    cfg = build_config(args)

    last = {"stage": None}

    def progress(stage, fraction, message):
        if stage != last["stage"]:
            print(f"  [{stage}] {message}")
            last["stage"] = stage

    result = run_batch(images, detector, cfg, progress)
    if result.seeds.empty:
        print("No seeds measured. Try a lower --conf.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    export.clean(result.seeds).to_csv(args.out / "per_seed.csv", index=False)
    result.lot_stats.to_csv(args.out / "lot_statistics.csv", index=False)
    result.qc.to_csv(args.out / "quality_control.csv", index=False)
    result.checks.to_csv(args.out / "detection_checks.csv", index=False)
    export.data_dictionary_frame().to_csv(args.out / "data_dictionary.csv", index=False)
    (args.out / "run_config.json").write_text(json.dumps(result.config, indent=2))

    if not args.no_images:
        shots = args.out / "annotated"
        shots.mkdir(exist_ok=True)
        for name, img in result.annotated.items():
            cv2.imwrite(str(shots / f"{Path(name).stem}_annotated.jpg"), img)

    print(f"\n{len(result.seeds)} seeds across {result.seeds.image.nunique()} photographs "
          f"in {result.timing['seconds']:.1f}s, measured in {result.units}")
    if result.tsw:
        print(f"Thousand-seed weight: {result.tsw['tsw_g']:.2f} g")
    for n in result.notes:
        print(f"  note: {n}")
    print(f"Written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
