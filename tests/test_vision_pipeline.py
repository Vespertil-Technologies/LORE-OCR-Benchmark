"""
Tests for dataset/vision_pipeline.py.

The full render -> noise -> OCR pipeline requires a system Tesseract binary.
These tests skip cleanly when Tesseract is not installed; CI installs it on
the Ubuntu runner so coverage is exercised there.
"""

import random

import pytest

pytest.importorskip("PIL")

from runners.ocr_engine import is_available  # noqa: E402

_OCR_OK, _OCR_MSG = is_available()
needs_tesseract = pytest.mark.skipif(not _OCR_OK, reason=f"Tesseract not available: {_OCR_MSG.splitlines()[0]}")


@needs_tesseract
class TestRenderAndOcr:
    def test_round_trip_preserves_short_text(self):
        from dataset.vision_pipeline import render_and_ocr

        text = "Vendor: SwiggyMart\nDate: 2024-03-14\nTotal: 368.16"
        ocr_text, meta = render_and_ocr(text, difficulty="easy", rng=random.Random(0))
        # Exact reproduction is unrealistic; require that at least one
        # distinctive token survives to confirm the OCR engine actually ran.
        assert any(needle in ocr_text for needle in ("Vendor", "SwiggyMart", "Total"))
        assert "noise_params" in meta
        assert "image_size" in meta
        assert isinstance(meta["image_size"], list)

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "extreme"])
    def test_runs_for_each_difficulty(self, difficulty):
        from dataset.vision_pipeline import render_and_ocr

        text = "Sample line one\nSample line two\nSample line three"
        ocr_text, meta = render_and_ocr(text, difficulty=difficulty, rng=random.Random(0))
        # OCR may produce empty string at extreme degradation, so we only
        # require the metadata to be well formed.
        assert isinstance(ocr_text, str)
        assert meta["noise_params"]
        assert meta["image_size"] and meta["image_size"][0] > 0


@needs_tesseract
def test_save_image_path_is_recorded(tmp_path):
    from dataset.vision_pipeline import render_and_ocr

    out = tmp_path / "rendered.png"
    _, meta = render_and_ocr(
        "Hello world",
        difficulty="easy",
        rng=random.Random(0),
        save_image_to=str(out),
    )
    assert meta["image_path"] == str(out)
    assert out.exists()
