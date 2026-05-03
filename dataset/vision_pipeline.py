"""
dataset/vision_pipeline.py

Orchestrates the three vision-track steps for a single sample:

    serialized_text  -render->  image  -image_noise->  noisy image  -OCR->  ocr_text

The synthetic noise generator (dataset/noise_generator.py) is bypassed for
real_ocr samples; the OCR errors come from an actual OCR engine acting on a
rendered, degraded document image.

The output of render_and_ocr() is shape-compatible with what the synthetic
track would have produced: the (ocr_text, metadata) pair is consumed by
sample_builder and stored on the sample. Downstream consumers (multi_run,
prompt_formatter, evaluators) need no changes.
"""

from __future__ import annotations

import logging
import random

from dataset.image_noise import apply_difficulty
from dataset.renderer import render_text
from runners.ocr_engine import ocr_image

log = logging.getLogger(__name__)


def render_and_ocr(
    serialized_text:  str,
    difficulty:       str,
    rng:              random.Random,
    font_size:        int = 18,
    save_image_to:    str | None = None,
) -> tuple[str, dict]:
    """Render text, apply image-level noise, and OCR the result.

    Args:
        serialized_text: The clean text (output of dataset/serializer.serialize).
        difficulty:      One of easy / medium / hard / extreme.
        rng:             Seeded RNG, used by the speckle noise function.
        font_size:       Point size for the rendered text.
        save_image_to:   Optional file path. If given, the noisy image is
                         saved here (for debugging or human inspection).

    Returns:
        (ocr_text, metadata) where metadata is a dict with the applied
        noise parameters and image dimensions.
    """
    image = render_text(serialized_text, font_size=font_size)
    noisy, applied_params = apply_difficulty(image, difficulty, rng)

    if save_image_to:
        from pathlib import Path

        from dataset.renderer import save_image
        save_image(noisy, Path(save_image_to))

    ocr_text = ocr_image(noisy)

    metadata = {
        "image_size":   list(noisy.size),
        "noise_params": applied_params,
        "image_path":   save_image_to,
    }
    return ocr_text, metadata
