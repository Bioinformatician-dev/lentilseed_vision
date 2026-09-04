"""
Panel 2 — YOLO seed detection.

The panel's three goals drive this module: high recall, minimal duplicates,
every visible seed found. Recall comes from a low confidence threshold and,
for large trays, from slicing the photograph into overlapping tiles so a 40 px
seed is not shrunk to 12 px by the 640 px letterbox. Slicing creates duplicates
along the seams, so a containment-aware merge runs afterwards.

Ultralytics is imported lazily. The rest of the library, and its tests, run
without torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectConfig


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls_name: str
    truncated: bool = False        # runs off the edge of the frame

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _intersection(a: Box, b: Box) -> float:
    iw = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    ih = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return float(iw * ih)


def iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def containment(a: Box, b: Box) -> float:
    """How much of the smaller box sits inside the larger one."""
    inter = _intersection(a, b)
    smaller = min(a.area, b.area)
    return inter / smaller if smaller else 0.0


def deduplicate(boxes: list[Box], cfg: DetectConfig) -> tuple[list[Box], int]:
    """
    Keep the most confident box of each cluster.

    IoU alone misses the tile-seam case, where a seed is caught whole in one
    tile and clipped in the next: the boxes overlap almost completely from the
    clipped box's point of view but score a middling IoU. Containment catches it.
    """
    kept: list[Box] = []
    removed = 0
    for box in sorted(boxes, key=lambda b: b.conf, reverse=True):
        duplicate = any(
            iou(box, k) > cfg.dedup_iou or containment(box, k) > cfg.containment
            for k in kept
        )
        if duplicate:
            removed += 1
        else:
            kept.append(box)
    return kept, removed


def _tiles(w: int, h: int, size: int, overlap: float):
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(1, w - size + step), step))
    ys = list(range(0, max(1, h - size + step), step))
    for y in ys:
        for x in xs:
            yield x, y, min(x + size, w), min(y + size, h)


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------

class Detector:
    """Thin wrapper over an Ultralytics YOLO model."""

    def __init__(self, weights_path: str):
        from ultralytics import YOLO   # imported here so the library stays light

        self.model = YOLO(weights_path)
        self.names = self.model.names
        self.weights_path = weights_path

    def _predict(self, img: np.ndarray, cfg: DetectConfig, offset=(0, 0)) -> list[Box]:
        res = self.model.predict(
            img, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
            agnostic_nms=cfg.agnostic_nms, max_det=int(cfg.max_det), verbose=False,
        )[0]
        ox, oy = offset
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append(Box(
                x1=int(x1 + ox), y1=int(y1 + oy), x2=int(x2 + ox), y2=int(y2 + oy),
                conf=float(b.conf[0]), cls_name=self.names[int(b.cls[0])],
            ))
        return out

    def detect(self, img: np.ndarray, cfg: DetectConfig) -> tuple[list[Box], dict]:
        """Return the surviving boxes plus a small report on how they were found."""
        h, w = img.shape[:2]

        if cfg.tiled and max(h, w) > cfg.tile_size:
            raw: list[Box] = []
            n_tiles = 0
            for x1, y1, x2, y2 in _tiles(w, h, cfg.tile_size, cfg.tile_overlap):
                n_tiles += 1
                raw += self._predict(img[y1:y2, x1:x2], cfg, offset=(x1, y1))
            pass_label = f"{n_tiles} tiles of {cfg.tile_size} px"
        else:
            raw = self._predict(img, cfg)
            pass_label = f"whole frame at {cfg.imgsz} px"

        boxes, removed = deduplicate(raw, cfg)

        m = cfg.edge_margin_px
        for b in boxes:
            b.truncated = (
                b.x1 <= m or b.y1 <= m or b.x2 >= w - m or b.y2 >= h - m
            )

        report = dict(
            raw_detections=len(raw),
            kept=len(boxes),
            duplicates_removed=removed,
            truncated=sum(b.truncated for b in boxes),
            strategy=pass_label,
            mean_confidence=float(np.mean([b.conf for b in boxes])) if boxes else float("nan"),
        )
        return boxes, report


def boxes_to_records(boxes: list[Box]) -> list[dict]:
    return [
        dict(
            seed_id=i, cls=b.cls_name, confidence=b.conf,
            x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
            cx=(b.x1 + b.x2) / 2.0, cy=(b.y1 + b.y2) / 2.0,
            truncated=b.truncated,
        )
        for i, b in enumerate(boxes)
    ]
