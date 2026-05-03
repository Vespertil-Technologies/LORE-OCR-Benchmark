"""
runners/ocr_engine.py

Thin wrapper around an OCR engine that turns a PIL image into a text string.
Default and only engine right now is Tesseract via pytesseract.

The wrapper is intentionally narrow: it does not own any image rendering or
noise injection (that is dataset/renderer and dataset/image_noise) and it
does not own any prediction logic (that is the LLM under test). It is a one
function shim.

Tesseract requires a system binary in addition to the pytesseract Python
package. If the binary is missing, the call raises FileNotFoundError with a
clear install hint per platform.
"""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)


# ── Availability ─────────────────────────────────────────────────────────────

def is_available() -> tuple[bool, str]:
    """Return (True, version_string) if Tesseract is callable, else (False, hint)."""
    try:
        import pytesseract
    except ImportError:
        return False, (
            "pytesseract not installed. Run: pip install -e \".[vision]\""
        )
    if shutil.which("tesseract") is None:
        return False, (
            "tesseract binary not found on PATH. Install it with:\n"
            "  Linux:   apt install tesseract-ocr tesseract-ocr-eng\n"
            "  macOS:   brew install tesseract\n"
            "  Windows: choco install tesseract  (or grab the installer from UB-Mannheim)"
        )
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as e:  # noqa: BLE001 - pytesseract surfaces several error types
        return False, f"tesseract installed but not callable: {e}"
    return True, version


# ── OCR call ─────────────────────────────────────────────────────────────────

def ocr_image(
    image:    Image.Image,
    lang:     str = "eng",
    psm:      int = 6,
    timeout:  float = 30.0,
) -> str:
    """Run Tesseract on the given PIL image and return the recognized text.

    Args:
        image:   PIL Image (RGB).
        lang:    Tesseract language pack code. Defaults to 'eng'.
        psm:     Page Segmentation Mode. 6 = "Assume a single uniform block of text".
                 This matches our renderer output and gives the best results
                 for form-style documents.
        timeout: Seconds before Tesseract is killed.

    Returns:
        The OCR text. Empty string if Tesseract returned nothing.

    Raises:
        FileNotFoundError: If the Tesseract binary is not installed.
        ImportError:       If pytesseract is not installed.
        RuntimeError:      For any other Tesseract failure.
    """
    ok, msg = is_available()
    if not ok:
        raise FileNotFoundError(msg)

    import pytesseract  # safe now that is_available passed

    config = f"--psm {psm}"
    try:
        text = pytesseract.image_to_string(
            image, lang=lang, config=config, timeout=timeout,
        )
    except pytesseract.TesseractError as e:
        raise RuntimeError(f"Tesseract failed: {e}") from e

    return text.strip("\n")
