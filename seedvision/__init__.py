"""
seedvision — image-based lentil seed phenotyping.

The library follows the published workflow one module per panel:

    1  stage1_acquisition   image quality control, scale calibration
    2  stage2_detection     YOLO detection, tiling, duplicate suppression
    3  stage3_segmentation  per-seed thresholding and contour extraction
    4  stage4_phenotypes    morphology, colour, coat pattern
    5  stage5_physical      millimetres, volume, mass, thousand-seed weight
    6  stage6_reliability   self-consistency and repeatability checks
    7  stage7_validation    agreement with caliper measurements

`report.py` renders any finished run as a single self-contained HTML page.

Nothing here imports Streamlit, so the same code runs in the app, in `cli.py`
and in a notebook:

    from seedvision import RunConfig, Detector, run_batch
    result = run_batch(images, Detector("best.pt"), RunConfig())
"""
from .config import (
    ColourConfig, DetectConfig, PatternRules, PhysicalConfig, QCConfig,
    ReliabilityConfig, RunConfig, ScaleConfig, SegmentConfig, ShapeConfig,
    ValidationConfig,
)
from .report import build_html
from .runner import STAGES, RunResult, measure_image, run_batch, repeatability_for
from .stage2_detection import Detector

__all__ = [
    "ColourConfig", "DetectConfig", "PatternRules", "PhysicalConfig", "QCConfig",
    "ReliabilityConfig", "RunConfig", "ScaleConfig", "SegmentConfig", "ShapeConfig",
    "ValidationConfig", "Detector", "RunResult", "STAGES",
    "run_batch", "measure_image", "repeatability_for", "build_html",
]

__version__ = "2.0.0"
