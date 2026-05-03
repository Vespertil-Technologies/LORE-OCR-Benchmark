"""
dataset/renderer.py

Renders serialized text (from dataset/serializer.py) onto a plain document-like
PIL image. This is the input to dataset/image_noise.py and ultimately to the
real OCR engine in runners/ocr_engine.py.

The rendering is intentionally minimal: a white background, a black monospace
font, fixed margin. We are not trying to fool a human; we are trying to give
Tesseract a realistic enough document image to make OCR errors plausible.

This module is an optional component used only by the real_ocr track. It is
imported lazily, so the rest of the pipeline still works without Pillow
installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Pillow is required for the vision track. "
        "Install it with: pip install -e .[vision]"
    ) from e

log = logging.getLogger(__name__)


# ── Font resolution ──────────────────────────────────────────────────────────

# Ordered list of TrueType fonts to look for. First one that exists wins.
# We prefer monospace because it OCRs cleanly and it visually reads as
# "form output" rather than typeset prose.
_FONT_CANDIDATES: tuple[str, ...] = (
    # Linux (Debian/Ubuntu, common on CI runners)
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    # macOS
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Courier New.ttf",
    # Windows
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _resolve_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the first available TrueType font, falling back to PIL default.

    The return type covers both branches: TrueType fonts come back as
    FreeTypeFont, the bitmap fallback comes back as the base ImageFont.
    Both implement the same drawing protocol that ImageDraw.text expects.
    """
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    log.warning(
        "No TrueType font found among %d candidates; falling back to PIL default. "
        "OCR quality will be very low.", len(_FONT_CANDIDATES),
    )
    return ImageFont.load_default()


# ── Public API ───────────────────────────────────────────────────────────────

DEFAULT_FONT_SIZE = 18
DEFAULT_LINE_SPACING = 1.4
DEFAULT_MARGIN = 40
DEFAULT_BG = (255, 255, 255)
DEFAULT_FG = (0, 0, 0)


def render_text(
    text:          str,
    font_size:     int                = DEFAULT_FONT_SIZE,
    line_spacing:  float              = DEFAULT_LINE_SPACING,
    margin:        int                = DEFAULT_MARGIN,
    bg:            tuple[int, int, int] = DEFAULT_BG,
    fg:            tuple[int, int, int] = DEFAULT_FG,
) -> Image.Image:
    """Render multiline text onto a white image sized to fit the content.

    Args:
        text:         The serialized form text. Newlines split lines.
        font_size:    Point size of the rendered text.
        line_spacing: Line height multiplier.
        margin:       Pixels of whitespace on every side.
        bg:           RGB background color.
        fg:           RGB text color.

    Returns:
        A PIL Image in RGB mode.
    """
    font = _resolve_font(font_size)
    lines = text.split("\n")

    # Measure widest line and total height using a temp draw context.
    # Using a 1x1 placeholder image just for the .textbbox() call.
    # Pillow's textbbox returns floats in newer typeshed stubs, so we keep
    # the running maximum as a float and cast once when sizing the image.
    tmp = Image.new("RGB", (1, 1), bg)
    draw = ImageDraw.Draw(tmp)
    line_h = int(font_size * line_spacing)
    max_w: float = 0.0
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        max_w = max(max_w, bbox[2] - bbox[0])

    width  = int(max_w) + 2 * margin
    height = max(line_h * len(lines), line_h) + 2 * margin

    image = Image.new("RGB", (width, height), bg)
    draw  = ImageDraw.Draw(image)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=fg, font=font)
        y += line_h

    return image


def save_image(image: Image.Image, path: Path) -> None:
    """Write image to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
