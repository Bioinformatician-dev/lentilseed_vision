"""
Rebuild the raster branding from the vector geometry.

`assets/logo.svg` and `assets/banner_title.svg` are the live, animated marks the
app uses. GitHub and image embeds want flat PNGs, so this script draws the same
shapes with Pillow and writes them out. No SVG renderer is needed, which keeps
the dependency list to what the app already installs.

    python generate_branding.py

Writes into assets/ (and copies beside app.py, where the app also looks):
    favicon.png            browser tab icon, also the page icon
    logo_96.png            the mark alone
    logo_256.png           the mark at social-card size
    banner_title.png       full title banner for the README
    banner_title_app.png   the same on a transparent background

Keep the palette below in sync with ui/theme.py and .streamlit/config.toml.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
ASSETS = HERE / "assets"

INK = (27, 33, 22, 255)
PAPER = (242, 244, 236, 255)
CARD = (255, 255, 255, 255)
LINE = (211, 217, 198, 255)
MUTED = (95, 106, 85, 255)
ACCENT = (91, 113, 40, 255)
COAT = (230, 213, 184, 255)
SPOT = (107, 83, 52, 255)
STAGE = ["#33648F", "#3F7A48", "#96651F", "#5B7128", "#6B5A9E", "#A64C7B", "#2F7A88"]

FONT_CANDIDATES = {
    "sans": [
        "BricolageGrotesque-Regular.ttf", "IBMPlexSans-Regular.ttf", "Poppins-Regular.ttf", "LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf", "Helvetica.ttc",
    ],
    "sans_bold": [
        "IBMPlexSans-SemiBold.ttf", "Poppins-SemiBold.ttf", "Poppins-Medium.ttf",
        "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf",
    ],
    "mono": [
        "IBMPlexMono-Regular.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf",
    ],
}
SEARCH_DIRS = [
    Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
    Path("/Library/Fonts"), Path("/System/Library/Fonts"), Path("C:/Windows/Fonts"),
    ASSETS / "fonts",
]


def _hex(value: str) -> tuple[int, int, int, int]:
    v = value.lstrip("#")
    return (*(int(v[i:i + 2], 16) for i in (0, 2, 4)), 255)


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for name in FONT_CANDIDATES[kind]:
        for root in SEARCH_DIRS:
            if not root.exists():
                continue
            hit = next(root.rglob(name), None)
            if hit:
                return ImageFont.truetype(str(hit), size)
    return ImageFont.load_default(size)


# ---------------------------------------------------------------------------
# the mark
# ---------------------------------------------------------------------------

def draw_mark(draw: ImageDraw.ImageDraw, cx: float, cy: float, rx: float,
              stroke: float, caliper: bool = True) -> None:
    """One seed: filled coat, traced outline, four markings, a caliper under it."""
    ry = rx * 0.79
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=COAT, outline=INK,
                 width=max(1, int(stroke)))

    for dx, dy, r in ((-0.33, -0.30, 0.12), (0.24, -0.18, 0.095),
                      (-0.10, 0.28, 0.082), (0.38, 0.22, 0.068)):
        sx, sy, sr = cx + dx * rx, cy + dy * ry, r * rx
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=SPOT)

    if caliper:
        y = cy + ry + rx * 0.42
        w = max(1, int(stroke * 0.75))
        draw.line([cx - rx, y, cx + rx, y], fill=ACCENT, width=w)
        for x in (cx - rx, cx + rx):
            draw.line([x, y - rx * 0.16, x, y + rx * 0.16], fill=ACCENT, width=w)


def render_mark(size: int, background=None, caliper: bool = True) -> Image.Image:
    """Supersampled so the ellipse edge stays clean at favicon sizes."""
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), background or (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_mark(d, s * 0.5, s * 0.42, s * 0.36, stroke=s * 0.022, caliper=caliper)
    return img.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# the banner
# ---------------------------------------------------------------------------

def render_banner(transparent: bool = False, scale: int = 2) -> Image.Image:
    w, h = 900 * scale, 220 * scale
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0) if transparent else PAPER)
    d = ImageDraw.Draw(img)

    if not transparent:
        d.rectangle([0, 0, w - 1, h - 1], outline=LINE, width=scale)

    mark = render_mark(168 * scale, caliper=True)
    img.alpha_composite(mark, (40 * scale, 26 * scale))

    f_title = font("sans_bold", 44 * scale)
    f_tag = font("sans", 17 * scale)
    f_mono = font("mono", 12 * scale)

    x = 214 * scale
    d.text((x, 52 * scale), "LentilSeedVision", font=f_title, fill=INK)
    d.text((x + 2 * scale, 106 * scale),
           "Image-based lentil seed phenotyping — tray photograph to phenotype data",
           font=f_tag, fill=MUTED)

    # the seven panels along a rule, in workflow order
    rail_y = 152 * scale
    d.line([x, rail_y, x + 470 * scale, rail_y], fill=LINE, width=2 * scale)
    for i, colour in enumerate(STAGE):
        cx = x + (8 + i * 74) * scale
        r = 7 * scale
        d.ellipse([cx - r, rail_y - r, cx + r, rail_y + r], fill=_hex(colour))
        label = str(i + 1)
        tw = d.textlength(label, font=f_mono)
        d.text((cx - tw / 2, rail_y + 12 * scale), label, font=f_mono, fill=MUTED)

    # a millimetre rule along the foot, because that is what the app is for
    if not transparent:
        base = h - 16 * scale
        d.line([44 * scale, base, w - 44 * scale, base], fill=LINE, width=scale)
        step = 18 * scale
        for i, tx in enumerate(range(44 * scale, w - 44 * scale, step)):
            length = 7 * scale if i % 5 == 0 else 4 * scale
            d.line([tx, base - length, tx, base], fill=LINE, width=scale)

    return img


# ---------------------------------------------------------------------------

def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    written: list[Path] = []

    for name, size in (("favicon.png", 64), ("logo_96.png", 96), ("logo_256.png", 256)):
        out = ASSETS / name
        render_mark(size, caliper=size >= 96).save(out)
        written.append(out)

    (ASSETS / "banner_title.png").write_bytes(b"")     # placeholder, replaced below
    render_banner().convert("RGB").save(ASSETS / "banner_title.png")
    written.append(ASSETS / "banner_title.png")
    render_banner(transparent=True).save(ASSETS / "banner_title_app.png")
    written.append(ASSETS / "banner_title_app.png")

    # The app looks in assets/ first and then beside app.py, so a copy at the
    # top level keeps a flat checkout working too.
    for f in written:
        shutil.copy(f, HERE / f.name)

    for f in written:
        print(f"wrote {f.relative_to(HERE)}  ({f.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
