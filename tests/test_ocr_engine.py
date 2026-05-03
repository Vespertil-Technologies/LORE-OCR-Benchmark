"""
Tests for runners/ocr_engine.py.

The availability probe and the error-message contract are covered here
without needing Tesseract installed. The actual OCR call is exercised in
tests/test_vision_pipeline.py and skipped when Tesseract is not present.
"""

import pytest

from runners.ocr_engine import is_available


class TestIsAvailable:
    def test_returns_a_pair(self):
        ok, msg = is_available()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)
        assert msg  # non-empty either way

    def test_message_includes_install_hint_when_missing(self):
        ok, msg = is_available()
        if ok:
            pytest.skip("Tesseract is installed; cannot test the missing-hint message")
        # When unavailable, the message should mention how to install
        assert any(kw in msg.lower() for kw in ("install", "apt", "brew", "choco", "pytesseract"))


class TestOcrImageNotInstalled:
    def test_raises_filenotfounderror_with_hint_when_missing(self):
        ok, _ = is_available()
        if ok:
            pytest.skip("Tesseract is installed; cannot test missing-binary path")

        from PIL import Image

        from runners.ocr_engine import ocr_image

        img = Image.new("RGB", (50, 50), (255, 255, 255))
        with pytest.raises(FileNotFoundError) as excinfo:
            ocr_image(img)
        msg = str(excinfo.value).lower()
        assert "tesseract" in msg
