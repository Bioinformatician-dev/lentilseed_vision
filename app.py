"""
Lentil seed phenotyping — Streamlit app
=======================================
Tray photograph in, phenotype table out.

The app is a three-step instrument rather than a page that does everything at
once:

    1  Photographs   choose the trays, see what will be measured
    2  Settings      set the scale, choose how hard to look for seeds
    3  Results       the seven-panel run, then the numbers

Behind step 3 the published workflow runs in order — quality control,
detection, per-seed segmentation, phenotypes, physical properties, reliability
checks, validation against calipers — and an eighth panel sorts the seeds into
the six recognised classes (Black, Defective, Dotted, Marbled, Spotted,
Unspotted).

This file is the shell only. Every measurement lives in `seedvision/`, which
imports nothing from Streamlit, so the same pipeline runs from `cli.py` or a
notebook. Every pixel of styling lives in `ui/theme.py`, which this file
injects — there is no CSS here.

    streamlit run app.py
"""
from __future__ import annotations

import base64
import html
import io
import sys
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def _bootstrap_path() -> Path:
    """
    Put the directory holding `seedvision/` on the import path.

    Normally that is this file's own directory and Python has already added it.
    Deployments move things around, though — the app can end up a level below
    the packages, or beside a folder whose name picked up a stray character
    during an upload — so the likely places are checked in order and the first
    that actually holds the package wins.
    """
    candidates = [APP_DIR, APP_DIR.parent]
    candidates += sorted(p for p in APP_DIR.parent.iterdir() if p.is_dir())

    for root in candidates:
        if (root / "seedvision" / "__init__.py").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root

    looked = "\n".join(f"  {p}" for p in candidates)
    raise ModuleNotFoundError(
        "The `seedvision` package could not be found. It should sit next to "
        f"app.py, at {APP_DIR / 'seedvision'}.\n\nLooked in:\n{looked}\n\n"
        "If the folder is in the repository but not at that path, move it there "
        "— an upload with a stray space in the destination path is the usual "
        "cause, and the two folder names look identical in a file listing."
    )


PACKAGE_ROOT = _bootstrap_path()

import cv2                                             # noqa: E402
import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
import streamlit as st                                 # noqa: E402
from PIL import Image                                  # noqa: E402

import model_loader as ml                              # noqa: E402
from seedvision import Detector, RunConfig, run_batch  # noqa: E402
from seedvision.runner import STAGES                   # noqa: E402
from ui import components as C                         # noqa: E402
from ui import motion, panels, theme                   # noqa: E402

C.set_asset_dirs(APP_DIR / "assets", APP_DIR, PACKAGE_ROOT / "assets")

st.set_page_config(
    page_title="LentilSeedVision",
    page_icon=C.asset_image("favicon.png") or "\U0001FAD8",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# The palette is resolved per rerun and pushed into the two modules that
# captured colour constants at import time, so Plotly and the stylesheet always
# agree about which surface we are on.
MODE = theme.active_mode()
st.markdown(theme.css(MODE), unsafe_allow_html=True)
theme.bind(MODE, panels, C)
motion.inject()


# ---------------------------------------------------------------------------
# fixed pipeline defaults — anything not on the Settings step lives here
# ---------------------------------------------------------------------------

SENSITIVITY = {
    "Conservative": (0.25, "Only confident seeds. Use when a tray is busy and "
                           "you would rather miss a seed than measure a shadow."),
    "Balanced": (0.10, "The default. Finds nearly every seed on an evenly lit tray."),
    "Thorough": (0.05, "Finds faint and partly hidden seeds, and more false ones. "
                       "Check the detection panel afterwards."),
}

SEGMENT_METHODS = ["auto", "otsu", "adaptive", "colour", "grabcut"]

CLASS_NAMES = [
    "Black seeds", "Defective", "Dotted seeds",
    "Marbled seeds", "Spotted seeds", "Unspotted seeds",
]


def build_config() -> RunConfig:
    """Turn the Settings step into a RunConfig."""
    s = st.session_state
    cfg = RunConfig()
    cfg.lot_id = s.get("lot_id") or datetime.now().strftime("run-%Y%m%d-%H%M%S")

    cfg.detect.conf = SENSITIVITY[s.get("sensitivity", "Balanced")][0]
    cfg.detect.iou = s.get("iou", 0.50)
    cfg.detect.max_det = s.get("max_det", 600)
    cfg.detect.tiled = s.get("tiled", False)

    cfg.segment.method = s.get("segment_method", "auto")

    cfg.reliability.outline_tol = 0.20
    cfg.reliability.dup_iou = 0.40
    cfg.reliability.drift_bins = 3

    px_per_mm = s.get("px_per_mm")
    if px_per_mm:
        # Without this the whole run comes back in pixels, which leaves area,
        # volume, mass and thousand-seed weight unusable. Guarded because older
        # RunConfig revisions spell the scale block differently.
        try:
            cfg.scale.px_per_mm = float(px_per_mm)
            cfg.scale.source = s.get("scale_source", "manual")
        except AttributeError:
            st.warning(
                "This build of `seedvision` does not expose `cfg.scale.px_per_mm`, "
                "so the run will report pixels. Update the package, or set the "
                "scale directly in `build_config()`.",
                icon="\u26A0\uFE0F",
            )
    return cfg


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------

DEFAULT_STATE = {
    "step": 1,                 # 1 photographs, 2 settings, 3 results
    "trays": [],               # [(filename, raw bytes)] — read once, kept across reruns
    "result": None,
    "images": {},
    "detector": None,
    "px_per_mm": None,
    "scale_source": "manual",
    "sensitivity": "Balanced",
    "iou": 0.50,
    "max_det": 600,
    "tiled": False,
    "segment_method": "auto",
    "lot_id": "",
    "scale_mode": "Measure from a reference object",
    "ref_mm": 10.0,
    "ref_px": 200.0,
}

for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


def goto(step: int) -> None:
    st.session_state["step"] = step
    st.rerun()


def clear_run() -> None:
    for key in ("result", "images", "repeat", "validation_summary"):
        st.session_state.pop(key, None)
    st.session_state["result"] = None
    st.session_state["images"] = {}


def subnav(key: str, options: list[str]) -> str:
    """
    One row of sub-sections inside a tab.

    `st.segmented_control` arrived in Streamlit 1.40 and is the right control
    here; a horizontal radio is the fallback and behaves identically. The
    current choice is seeded into session state rather than passed as a
    default, because a widget given both a key and a default warns on rerun.
    """
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control("Section", options, key=key,
                                      label_visibility="collapsed")
        return choice or options[0]
    return st.radio("Section", options, horizontal=True, key=key,
                    label_visibility="collapsed")


def heading(title: str, subtitle: str, stage: str) -> None:
    """
    A section head for the setup steps.

    `C.panel_head` numbers its heading, and those numbers mean the workflow
    panels 1-8. Settings are not part of that sequence, so they get the same
    typography without a number.
    """
    colour = theme.STAGE_COLORS.get(stage, theme.INK)
    st.markdown(
        f'<div class="phead" style="--c:{colour}"><h3>{html.escape(title)}</h3></div>'
        f'<p class="psub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_detector(weights_path: str) -> Detector:
    return Detector(weights_path)


weights_path, weights_error, weights_warning = None, None, None
try:
    weights_path = ml.resolve_weights("")
    if Path(weights_path).stat().st_size < 1_000_000:
        weights_warning = (
            "The weights file is under a megabyte, so it is a Git LFS pointer rather "
            "than a model. Download the real `best.pt`, or set `weights_url` in "
            "`.streamlit/secrets.toml`."
        )
except ml.WeightsError as e:
    weights_error = str(e)
except Exception as e:                                 # noqa: BLE001
    weights_error = f"The model weights could not be resolved: {e}"


# ---------------------------------------------------------------------------
# masthead
# ---------------------------------------------------------------------------

C.masthead(
    "LentilSeedVision",
    "Photograph a tray of seeds. Each one is found, traced, measured and "
    "sorted by coat type, then checked, and the results come out as a table "
    "you can publish from.",
)

px_per_mm = st.session_state["px_per_mm"]
C.chipline([
    ("model", "ready" if not weights_error else "missing",
     "live" if not weights_error else "warn"),
    ("scale", f"{px_per_mm:.2f} px/mm" if px_per_mm else "uncalibrated",
     "" if px_per_mm else "off"),
    ("sensitivity", st.session_state["sensitivity"].lower(), ""),
    ("classes", str(len(CLASS_NAMES)), ""),
])

if weights_error:
    st.error(weights_error)
if weights_warning:
    st.warning(weights_warning, icon="\u26A0\uFE0F")


# ---------------------------------------------------------------------------
# the three-step rule
# ---------------------------------------------------------------------------

STEP_LABELS = [(1, "Photographs"), (2, "Settings"), (3, "Results")]


def step_rule(current: int) -> None:
    cells = []
    for number, label in STEP_LABELS:
        state = "done" if number < current else ("now" if number == current else "")
        cells.append(
            f'<div class="s {state}"><div class="n">{number}</div>'
            f'<div class="t">{label}</div></div>'
        )
    st.markdown(f'<div class="steps">{"".join(cells)}</div>', unsafe_allow_html=True)


step = st.session_state["step"]
step_rule(step)


# ---------------------------------------------------------------------------
# step 1 — photographs
# ---------------------------------------------------------------------------

def tray_preview(trays: list[tuple[str, bytes]]) -> None:
    cards = []
    for name, raw in trays:
        try:
            img = Image.open(io.BytesIO(raw))
            img.thumbnail((420, 420))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=82)
            src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:                              # noqa: BLE001
            continue
        cards.append(f'<div class="tray"><img src="{src}" alt="">'
                     f'<div class="f">{html.escape(name)}</div></div>')
    if cards:
        st.markdown(f'<div class="tray-grid">{"".join(cards)}</div>',
                    unsafe_allow_html=True)


def step_photographs() -> None:
    heading("Choose the tray photographs",
            "One photograph or a whole batch. Everything you add here runs "
            "through the same seven panels.", "acquire")

    uploads = st.file_uploader(
        "Tray photographs", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    # Read the bytes now. The uploader widget is not on screen at step 2, so
    # its file handles do not survive the move; the bytes do.
    if uploads:
        st.session_state["trays"] = [(f.name, f.getvalue()) for f in uploads]

    trays = st.session_state["trays"]

    if not trays:
        C.note("Spread the seeds so they do not touch. Even light, matte "
               "background, and at least <b>60 pixels across each seed</b>. "
               "Include a ruler or a coin in frame if you want millimetres.")
        return

    tray_preview(trays)
    C.note(f"<b>{len(trays)}</b> photograph{'s' if len(trays) != 1 else ''} ready.")

    left, right = st.columns([1, 3])
    if left.button("Continue", type="primary", width="stretch"):
        clear_run()
        goto(2)
    if right.button("Clear photographs", width="stretch"):
        st.session_state["trays"] = []
        clear_run()
        st.rerun()


# ---------------------------------------------------------------------------
# step 2 — settings
# ---------------------------------------------------------------------------

def scale_controls() -> None:
    heading("Set the scale",
            "Millimetres, volume, mass and thousand-seed weight all come from "
            "one number: how many pixels cover a millimetre. Without it the run "
            "still works, but every size is reported in pixels.", "segment")

    modes = [
        "Measure from a reference object",
        "Enter pixels per millimetre",
        "Leave uncalibrated (report pixels)",
    ]
    if st.session_state.get("scale_mode") not in modes:
        st.session_state["scale_mode"] = modes[0]
    mode = st.radio("How do you want to set the scale?", modes, key="scale_mode")

    if mode == modes[0]:
        a, b = st.columns(2)
        real_mm = a.number_input(
            "Reference width (mm)", min_value=0.1, max_value=500.0,
            value=float(st.session_state.get("ref_mm", 10.0)), step=0.1,
            help="A graph-paper square is usually 10.0 mm and is the easiest "
                 "reference to photograph flat next to the seeds. A Pakistani "
                 "1-rupee coin is 20.0 mm; a US quarter is 24.26 mm.",
        )
        ref_px = b.number_input(
            "Its width in the photograph (px)", min_value=1.0, max_value=20000.0,
            value=float(st.session_state.get("ref_px", 200.0)), step=1.0,
            help="Open the photograph, measure the same object across, and put "
                 "that pixel count here.",
        )
        st.session_state["ref_mm"] = real_mm
        st.session_state["ref_px"] = ref_px
        st.session_state["px_per_mm"] = ref_px / real_mm if real_mm else None
        st.session_state["scale_source"] = "reference object"
        C.note(f"That is <b>{st.session_state['px_per_mm']:.2f} px/mm</b>. "
               f"A seed 5 mm long will measure about "
               f"{st.session_state['px_per_mm'] * 5:.0f} px across.")

    elif mode == modes[1]:
        value = st.number_input(
            "Pixels per millimetre", min_value=0.1, max_value=2000.0,
            value=float(st.session_state.get("px_per_mm") or 20.0), step=0.1,
        )
        st.session_state["px_per_mm"] = value
        st.session_state["scale_source"] = "manual"

    else:
        st.session_state["px_per_mm"] = None
        C.note("Sizes will be reported in pixels. Area, volume, mass and "
               "thousand-seed weight are not comparable between photographs "
               "taken at different distances.", flag=True)


def detection_controls() -> None:
    heading("Choose how hard to look",
            "How willing the detector should be to call something a seed.",
            "detect")

    names = list(SENSITIVITY)
    choice = st.select_slider("Detection sensitivity", names, key="sensitivity")
    C.note(SENSITIVITY[choice][1])

    with st.expander("Advanced"):
        st.session_state["segment_method"] = st.selectbox(
            "Outline method", SEGMENT_METHODS,
            index=SEGMENT_METHODS.index(st.session_state.get("segment_method", "auto")),
            help="`auto` tries four thresholds in order and keeps whichever "
                 "produces a plausible outline.",
        )
        a, b = st.columns(2)
        st.session_state["iou"] = a.slider(
            "Overlap before two boxes are merged", 0.10, 0.90,
            float(st.session_state.get("iou", 0.50)), 0.05,
        )
        st.session_state["max_det"] = b.number_input(
            "Most seeds per photograph", 50, 5000,
            int(st.session_state.get("max_det", 600)), 50,
        )
        st.session_state["tiled"] = st.checkbox(
            "Split the photograph into tiles before detecting",
            value=bool(st.session_state.get("tiled", False)),
            help="Slower, and worth it when seeds are small in a wide frame.",
        )
        st.session_state["lot_id"] = st.text_input(
            "Lot name", value=st.session_state.get("lot_id", ""),
            placeholder="Left blank, the run is timestamped",
        )


def step_settings() -> None:
    scale_controls()
    C.hline()
    detection_controls()
    C.hline()

    left, right = st.columns([1, 3])
    if left.button("Measure seeds", type="primary", width="stretch",
                   disabled=bool(weights_error)):
        clear_run()
        goto(3)
    if right.button("Back to photographs", width="stretch"):
        goto(1)


# ---------------------------------------------------------------------------
# step 3 — the run
# ---------------------------------------------------------------------------

def decode(raw: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def do_run(cfg: RunConfig):
    if weights_error:
        st.error(weights_error)
        return None

    trays = st.session_state["trays"]
    if not trays:
        st.error("There are no photographs to measure. Go back to step 1 and add some.")
        return None

    slot = st.empty()
    with slot.container():
        motion.loader("Loading the model", Path(weights_path).name)
    try:
        detector = load_detector(weights_path)
    except Exception as e:                             # noqa: BLE001
        slot.empty()
        st.error(f"The model file could not be opened: {e}")
        return None
    slot.empty()

    images: list[tuple[str, np.ndarray]] = []
    for name, raw in trays:
        img = decode(raw)
        if img is not None:
            images.append((name, img))
    if not images:
        st.error("None of those files could be read as an image. JPEG and PNG only.")
        return None

    spine_slot = st.empty()
    working = st.empty()
    with working.container():
        motion.loader("Measuring", f"{len(images)} photograph"
                                   f"{'s' if len(images) != 1 else ''}")
    bar = st.progress(0.0, text="Starting")
    done: set[str] = set()
    order = [k for k, _ in STAGES]

    def on_progress(stage: str, fraction: float, message: str) -> None:
        for k in order[:order.index(stage)]:
            done.add(k)
        C.show_spine(stage, done, spine_slot)
        bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    result = run_batch(images, detector, cfg, on_progress,
                       class_colors=theme.class_colors_bgr())

    bar.empty()
    spine_slot.empty()
    working.empty()

    st.session_state["result"] = result
    st.session_state["images"] = dict(images)
    st.session_state["detector"] = detector
    st.session_state.pop("repeat", None)
    st.session_state.pop("validation_summary", None)
    return result


# ---------------------------------------------------------------------------
# step 3 — the classes panel
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


CLASS_COLUMNS = ["class_name", "class", "label", "category",
                 "predicted_class", "seed_class"]
IMAGE_COLUMNS = ["image", "image_name", "source_image", "file", "filename"]


def _thumbnail(img_bgr: np.ndarray, x1, y1, x2, y2, size: int = 140) -> str | None:
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop)
    pil.thumbnail((size, size))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def panel_classes(result, images: dict) -> None:
    seeds = result.seeds
    class_col = _find_col(seeds, CLASS_COLUMNS)

    C.panel_head(8, "Seed classes",
                 "Every measured seed sorted into the six recognised coat types.",
                 "outputs")

    if class_col is None:
        C.empty_note(
            "The seeds table has no class column, so this panel has nothing to "
            "sort. Looked for <code>class_name</code>, <code>class</code>, "
            "<code>label</code>, <code>category</code>, <code>predicted_class</code> "
            "and <code>seed_class</code> — rename the classifier's output column "
            "to one of those."
        )
        return

    counts = seeds[class_col].value_counts()
    total = int(counts.sum())
    extra = [c for c in counts.index if str(c) not in CLASS_NAMES]
    ordered = CLASS_NAMES + extra

    tiles = []
    for name in ordered:
        n = int(counts.get(name, 0))
        pct = (n / total * 100) if total else 0.0
        colour = theme.class_color(name)
        tiles.append(
            f'<div class="class-tile" style="--c:{colour}">'
            f'<div class="name"><span class="swatch" style="background:{colour}"></span>'
            f'{html.escape(str(name))}</div>'
            f'<div class="count">{n}</div>'
            f'<div class="pct">{pct:.1f}%</div>'
            f'<div class="bar"><i style="width:{pct:.1f}%"></i></div></div>'
        )
    st.markdown(f'<div class="class-grid">{"".join(tiles)}</div>',
                unsafe_allow_html=True)

    if extra:
        C.note(f"{len(extra)} class{'es' if len(extra) != 1 else ''} outside the "
               f"published six: {', '.join(html.escape(str(e)) for e in extra)}.",
               flag=True)

    x1c, y1c = _find_col(seeds, ["x1", "bbox_x1", "xmin"]), _find_col(seeds, ["y1", "bbox_y1", "ymin"])
    x2c, y2c = _find_col(seeds, ["x2", "bbox_x2", "xmax"]), _find_col(seeds, ["y2", "bbox_y2", "ymax"])
    img_col = _find_col(seeds, IMAGE_COLUMNS)

    if not all([img_col, x1c, y1c, x2c, y2c]):
        C.muted_caption(
            "Per-seed crops need an image-name column plus x1/y1/x2/y2, which "
            "this run did not produce, so only the counts are shown."
        )
        return

    C.hline()
    present = [n for n in ordered if int(counts.get(n, 0))]
    if not present:
        return
    chosen = subnav("class_sheet", present)

    rows = seeds[seeds[class_col] == chosen].head(24)
    colour = theme.class_color(chosen)
    cards = []
    for _, r in rows.iterrows():
        img_bgr = images.get(r[img_col])
        if img_bgr is None:
            continue
        b64 = _thumbnail(img_bgr, r[x1c], r[y1c], r[x2c], r[y2c])
        if b64:
            cards.append(f'<div class="seed" style="--c:{colour}">'
                         f'<img src="data:image/png;base64,{b64}" alt=""></div>')
    if cards:
        st.markdown(f'<div class="seed-grid">{"".join(cards)}</div>',
                    unsafe_allow_html=True)
        C.muted_caption(f"{len(cards)} of {int(counts.get(chosen, 0))} "
                        f"{chosen.lower()}, in the order they were measured.")
    else:
        C.empty_note("The crops for this class could not be cut from the "
                     "photographs in this run.")


# ---------------------------------------------------------------------------
# step 3 — results
# ---------------------------------------------------------------------------

def headline(result, cfg) -> None:
    seeds = result.seeds
    class_col = _find_col(seeds, CLASS_COLUMNS)
    n_seeds = len(seeds)
    n_images = int(result.timing.get("images", len(st.session_state["images"])))

    items = [(f"{n_seeds:,}", "seeds measured"),
             (str(n_images), "photographs")]

    if class_col:
        share = float(seeds[class_col].value_counts(normalize=True).iloc[0] * 100)
        top = str(seeds[class_col].value_counts().index[0])
        items.append((f"{share:.0f}%", f"{top.lower()}"))

    length_col = _find_col(seeds, ["length_mm", "length", "major_axis_mm", "major_axis"])
    if length_col is not None:
        unit = "mm" if result.units == "mm" else "px"
        items.append((f"{seeds[length_col].mean():.2f} {unit}", "mean length"))

    items.append((f"{result.timing['seconds']:.1f} s", "run time"))
    motion.readings(items, stage="outputs")


def show_results(result, cfg) -> None:
    images = st.session_state["images"]
    detector = st.session_state["detector"]

    C.show_spine("outputs", {k for k, _ in STAGES[:-1]})

    if result.seeds.empty:
        st.warning(
            "No seeds were measured. Nothing cleared the detection threshold — "
            "try a sharper, more evenly lit photograph, or set sensitivity to "
            "Thorough on step 2.",
            icon="\U0001F50D",
        )
        st.dataframe(result.qc, width="stretch", hide_index=True)
        return

    headline(result, cfg)

    if result.units == "px":
        C.note("Sizes are in pixels because no scale was set. Go back to "
               "<b>Settings</b> and calibrate to get millimetres, volume, mass "
               "and thousand-seed weight.", flag=True)

    tabs = st.tabs(["Overview", "Measurements", "Classes", "Quality", "Export"])

    with tabs[0]:
        which = subnav("nav_overview", ["Photographs", "Detection"])
        if which == "Photographs":
            panels.panel_photographs(result, images, cfg)
        else:
            panels.panel_detection(result, images, cfg)

    with tabs[1]:
        which = subnav("nav_measure", ["Outlines", "Phenotypes", "Physical"])
        if which == "Outlines":
            panels.panel_outlines(result, images, cfg)
        elif which == "Phenotypes":
            panels.panel_phenotypes(result, images, cfg)
        else:
            panels.panel_physical(result, images, cfg)

    with tabs[2]:
        panel_classes(result, images)

    with tabs[3]:
        which = subnav("nav_quality", ["Reliability", "Validation"])
        if which == "Reliability":
            panels.panel_reliability(result, images, cfg, detector)
        else:
            panels.panel_validation(result, images, cfg)

    with tabs[4]:
        panels.panel_outputs(result, images, cfg)

    C.hline()
    left, right = st.columns([1, 3])
    if left.button("Measure another batch", type="primary", width="stretch"):
        st.session_state["trays"] = []
        clear_run()
        goto(1)
    if right.button("Change settings and re-run", width="stretch"):
        clear_run()
        goto(2)

    C.chipline([
        ("lot", cfg.lot_id, ""),
        ("seeds", str(result.timing["seeds"]), ""),
        ("photographs", str(result.timing["images"]), ""),
        ("elapsed", f"{result.timing['seconds']:.1f}s", ""),
        ("units", result.units, "" if result.units == "mm" else "off"),
    ])


def step_results() -> None:
    cfg = build_config()
    result = st.session_state["result"]
    if result is None:
        result = do_run(cfg)
        if result is None:
            if st.button("Back to settings"):
                goto(2)
            return
    show_results(result, cfg)


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

if step == 1:
    step_photographs()
elif step == 2:
    step_settings()
else:
    step_results()
