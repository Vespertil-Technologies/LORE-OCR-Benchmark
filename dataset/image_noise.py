"""
dataset/image_noise.py

Image-level corruptions applied to a rendered document image. The output is
fed to the OCR engine, so the corruptions here are what cause the OCR errors
that the benchmark then asks the LLM to recover from.

Each noise function is deterministic given the rng. The four difficulty tiers
map to progressively heavier corruption; see VISION_DIFFICULTY_PRESETS.

This module imports Pillow lazily; the rest of the pipeline still works
without Pillow installed.
"""

from __future__ import annotations

import io
import random

try:
    from PIL import Image, ImageFilter
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Pillow is required for the vision track. "
        "Install it with: pip install -e .[vision]"
    ) from e


# ── Individual noise functions ───────────────────────────────────────────────

def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """Gaussian blur. Radius 0.5 is barely visible, 2.5 makes text fuzzy."""
    if radius <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_rotation(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate by N degrees. Tesseract handles small skew but degrades past 5 degrees."""
    if degrees == 0:
        return image
    return image.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))


def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG at the given quality (1-95). Lower quality = more compression artifacts."""
    if quality >= 95:
        return image
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_speckle(image: Image.Image, density: float, rng: random.Random) -> Image.Image:
    """Sprinkle dark pixels randomly. Density is fraction of pixels to flip (0.0 to 0.05)."""
    if density <= 0:
        return image
    px = image.load()
    assert px is not None, "PIL.Image.load() returned None on a valid image"
    w, h = image.size
    n = int(w * h * density)
    for _ in range(n):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        px[x, y] = (0, 0, 0)
    return image


# ── Difficulty presets ───────────────────────────────────────────────────────

# Progressive corruption: easy is near-clean, extreme is heavily degraded.
# Tuned so a competent OCR can still produce mostly-readable output at
# extreme; the benchmark measures recovery, not OCR catastrophic failure.

VISION_DIFFICULTY_PRESETS: dict[str, dict[str, float]] = {
    "easy":    {"blur": 0.4, "rotation": 0.0, "jpeg": 95, "speckle": 0.000},
    "medium":  {"blur": 0.8, "rotation": 1.0, "jpeg": 80, "speckle": 0.001},
    "hard":    {"blur": 1.4, "rotation": 2.0, "jpeg": 65, "speckle": 0.003},
    "extreme": {"blur": 2.0, "rotation": 3.5, "jpeg": 50, "speckle": 0.006},
}


def apply_difficulty(
    image:       Image.Image,
    difficulty:  str,
    rng:         random.Random,
    overrides:   dict[str, float] | None = None,
) -> tuple[Image.Image, dict]:
    """Apply the bundle of noise functions associated with a difficulty tier.

    Args:
        image:       The clean rendered document image.
        difficulty:  One of easy / medium / hard / extreme.
        rng:         Seeded RNG. Only used by speckle; other functions are deterministic.
        overrides:   Optional dict to override individual parameters.

    Returns:
        (degraded_image, applied_params) tuple. The applied_params dict is
        written into the sample's metadata for traceability.
    """
    if difficulty not in VISION_DIFFICULTY_PRESETS:
        raise ValueError(
            f"Unknown vision difficulty '{difficulty}'. "
            f"Known: {sorted(VISION_DIFFICULTY_PRESETS)}"
        )
    params = dict(VISION_DIFFICULTY_PRESETS[difficulty])
    if overrides:
        params.update(overrides)

    out = image
    out = apply_blur(out, params["blur"])
    out = apply_rotation(out, params["rotation"])
    out = apply_jpeg(out, int(params["jpeg"]))
    out = apply_speckle(out, params["speckle"], rng)
    return out, params
