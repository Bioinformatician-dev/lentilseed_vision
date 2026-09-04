"""
Run configuration.

One dataclass per workflow panel, gathered into `RunConfig`. Every number the
pipeline uses lives here, so a run can be reproduced from the JSON that
`RunConfig.to_dict()` writes into the export bundle.

Nothing in this module imports Streamlit: the same configuration drives the
app, the CLI and a notebook.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


# ---------------------------------------------------------------------------
# 1 — image acquisition & quality control
# ---------------------------------------------------------------------------

@dataclass
class QCConfig:
    """Limits that decide whether a photograph is fit to measure."""

    focus_min: float = 60.0        # variance of the Laplacian; lower = blurrier
    clip_max: float = 0.01         # fraction of pixels pinned at 0 or 255
    illum_cv_max: float = 0.12     # coefficient of variation of background light
    seed_min_px: float = 35.0      # median seed width in pixels
    seed_good_px: float = 60.0     # width at which measurement error stops mattering
    min_seeds: int = 5             # below this a lot statistic means little


@dataclass
class ScaleConfig:
    """Pixels to millimetres. Left unset, the whole run stays in pixels."""

    px_per_mm: float | None = None
    source: str = "unset"          # unset | manual | reference | ruler
    reference_mm: float = 0.0      # long edge of the object used to calibrate
    note: str = ""

    @property
    def calibrated(self) -> bool:
        return bool(self.px_per_mm and self.px_per_mm > 0)


# ---------------------------------------------------------------------------
# 2 — YOLO seed detection
# ---------------------------------------------------------------------------

@dataclass
class DetectConfig:
    """Recall first, then remove the duplicates that recall buys."""

    conf: float = 0.10
    iou: float = 0.50
    max_det: int = 600
    imgsz: int = 640
    agnostic_nms: bool = True
    tiled: bool = False            # slice big trays so small seeds survive resizing
    tile_size: int = 1024
    tile_overlap: float = 0.20
    dedup_iou: float = 0.55        # merge boxes across tile seams
    containment: float = 0.80      # a box mostly inside another is the same seed
    edge_margin_px: int = 2        # boxes this close to the frame are truncated


# ---------------------------------------------------------------------------
# 3 — seed-wise segmentation & contour extraction
# ---------------------------------------------------------------------------

@dataclass
class SegmentConfig:
    """Thresholding happens inside each detection box, never tray-wide."""

    pad_px: int = 6                # breathing room so the coat edge is not clipped
    method: str = "auto"           # auto | otsu | adaptive | otsu_lab | otsu_sat
    adaptive_block_frac: float = 0.35   # block size as a fraction of ROI width
    adaptive_c: float = 4.0
    close_iter: int = 2
    open_iter: int = 1
    smooth_epsilon: float = 0.004  # contour simplification, as a fraction of perimeter
    min_area_px: float = 20.0
    min_fill: float = 0.10         # mask area / ROI area below this is a failed cut
    max_fill: float = 0.95         # above this the threshold caught the background
    max_border_touch: float = 0.35 # fraction of the ROI border the mask may occupy


# ---------------------------------------------------------------------------
# 4 — image-derived phenotypes
# ---------------------------------------------------------------------------

@dataclass
class PatternRules:
    """Coat pattern read from pixels alone, independent of the detector's class."""

    black_L_max: float = 35.0
    plain_coverage_max: float = 0.02
    dotted_count_min: int = 12
    marbled_largest: float = 0.15
    spot_min_separation: float = 8.0   # grey levels between coat and marking
    spot_min_area_px: int = 3


@dataclass
class ColourConfig:
    erode_iter: int = 1            # step in from the rim before sampling colour
    swatch_bins: int = 5           # colours in the per-seed swatch


@dataclass
class ShapeConfig:
    harmonics: int = 10            # elliptic Fourier descriptors per outline
    normalize: bool = True
    pca_components: int = 4
    atlas_max_outlines: int = 250


# ---------------------------------------------------------------------------
# 5 — derived physical properties
# ---------------------------------------------------------------------------

@dataclass
class PhysicalConfig:
    """Thickness is invisible from overhead, so it is modelled, not measured."""

    thickness_from_width: float = 0.62   # published lentil geometry, not your lot
    density_g_cm3: float | None = None   # set by weighing a counted sub-sample
    density_source: str = "unset"        # unset | literature | weighed
    literature_density: float = 1.30     # g/cm3, dry lentil, for a rough mass


# ---------------------------------------------------------------------------
# 6 — reliability & self-consistency
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityConfig:
    dup_iou: float = 0.30
    merged_area_factor: float = 1.8      # this much bigger than the median = two seeds
    outline_tol: float = 0.15            # |contour area / ellipse area - 1|
    drift_bins: int = 3                  # grid across the frame for position drift
    drift_r2_flag: float = 0.10          # position explaining this much variance is a lens/light effect
    match_max_dist: float = 0.03         # normalised centroid distance for seed-to-seed matching
    repeat_transforms: tuple[str, ...] = ("flip_h", "rot90", "scale_80")


# ---------------------------------------------------------------------------
# 7 — external validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    join_on: str = "seed_id"       # seed_id | rank  (rank pairs by size order)
    traits: tuple[str, ...] = ("length_mm", "width_mm")
    loa_z: float = 1.96            # limits of agreement width


# ---------------------------------------------------------------------------
# whole run
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    qc: QCConfig = field(default_factory=QCConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    pattern: PatternRules = field(default_factory=PatternRules)
    colour: ColourConfig = field(default_factory=ColourConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    lot_id: str = "lot-1"
    operator: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Traits offered everywhere a trait has to be picked, in the order a seed
# scientist reads them: size, then shape, then colour, then markings.
CORE_TRAITS_MM = [
    "length_mm", "width_mm", "thickness_mm", "area_mm2", "perimeter_mm",
    "feret_max_mm", "gmd_mm", "amd_mm", "surface_area_mm2", "volume_mm3", "mass_mg",
]
CORE_TRAITS_PX = [
    "length_px", "width_px", "area_px2", "perimeter_px", "feret_max_px", "eq_radius_px",
]
SHAPE_TRAITS = [
    "roundness", "aspect_ratio", "solidity", "convexity", "extent",
    "eccentricity", "sphericity", "area_ratio",
]
COLOUR_TRAITS = [
    "mean_L", "mean_a", "mean_b", "mean_hue", "saturation", "brightness", "texture_std",
]
PATTERN_TRAITS = [
    "spot_count", "spot_coverage", "largest_spot", "spot_contrast_dE", "dark_fraction",
]
