"""
Tests for dataset/renderer.py.

The renderer is a thin wrapper over Pillow. We pin the basics: it produces
an RGB image, sized to fit the content with the requested margin, and
remains stable across calls with the same input.
"""

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from dataset.renderer import (  # noqa: E402
    render_text,
    save_image,
)


class TestRenderText:
    def test_returns_pil_image(self):
        img = render_text("hello")
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_image_is_at_least_two_margins_wide(self):
        img = render_text("hi", font_size=18, margin=20)
        assert img.size[0] >= 2 * 20

    def test_taller_for_more_lines(self):
        one_line = render_text("a")
        five_lines = render_text("a\nb\nc\nd\ne")
        assert five_lines.size[1] > one_line.size[1]

    def test_wider_for_longer_lines(self):
        short = render_text("abc")
        long  = render_text("a" * 80)
        assert long.size[0] > short.size[0]

    def test_deterministic(self):
        a = render_text("Vendor: SwiggyMart\nDate: 2024-03-14")
        b = render_text("Vendor: SwiggyMart\nDate: 2024-03-14")
        assert a.tobytes() == b.tobytes()

    def test_white_background_default(self):
        img = render_text("x", margin=10)
        # Sample a corner pixel; should be background (white).
        assert img.getpixel((0, 0)) == (255, 255, 255)

    def test_custom_bg_fg(self):
        img = render_text("x", bg=(0, 0, 0), fg=(255, 255, 255), margin=10)
        assert img.getpixel((0, 0)) == (0, 0, 0)


class TestSaveImage:
    def test_writes_png(self, tmp_path):
        img = render_text("hello")
        out = tmp_path / "sub" / "out.png"
        save_image(img, out)
        assert out.exists()
        # Round-trip to confirm valid PNG. Use a context manager so the
        # underlying file handle closes promptly.
        with Image.open(out) as round_trip:
            round_trip.load()
            assert round_trip.size == img.size
