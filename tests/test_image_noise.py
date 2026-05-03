"""
Tests for dataset/image_noise.py.

Verifies the per-tier difficulty presets produce monotonically heavier
corruption (larger blur, more rotation, lower JPEG quality) and that
individual noise functions are deterministic given their seed.
"""

import random

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from dataset.image_noise import (  # noqa: E402
    VISION_DIFFICULTY_PRESETS,
    apply_blur,
    apply_difficulty,
    apply_jpeg,
    apply_rotation,
    apply_speckle,
)


def _solid_image(color: tuple[int, int, int] = (200, 200, 200), size: tuple[int, int] = (200, 80)) -> Image.Image:
    return Image.new("RGB", size, color)


class TestPresets:
    def test_all_four_tiers_present(self):
        assert sorted(VISION_DIFFICULTY_PRESETS) == ["easy", "extreme", "hard", "medium"]

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard", "extreme"])
    def test_preset_keys_complete(self, tier):
        keys = set(VISION_DIFFICULTY_PRESETS[tier])
        assert {"blur", "rotation", "jpeg", "speckle"} <= keys

    def test_blur_is_monotonic(self):
        seq = [VISION_DIFFICULTY_PRESETS[t]["blur"] for t in ("easy", "medium", "hard", "extreme")]
        assert seq == sorted(seq)

    def test_jpeg_quality_decreases_with_difficulty(self):
        seq = [VISION_DIFFICULTY_PRESETS[t]["jpeg"] for t in ("easy", "medium", "hard", "extreme")]
        assert seq == sorted(seq, reverse=True)


class TestIndividualFunctions:
    def test_apply_blur_zero_is_noop(self):
        img = _solid_image()
        out = apply_blur(img, 0)
        assert out.tobytes() == img.tobytes()

    def test_apply_blur_changes_pixels(self):
        img = Image.new("RGB", (40, 40), (255, 255, 255))
        # Stamp a single black pixel so blur has something to spread.
        img.putpixel((20, 20), (0, 0, 0))
        out = apply_blur(img, radius=2.0)
        assert out.tobytes() != img.tobytes()

    def test_apply_rotation_zero_is_noop(self):
        img = _solid_image()
        out = apply_rotation(img, 0)
        assert out.tobytes() == img.tobytes()

    def test_apply_jpeg_high_quality_is_noop(self):
        img = _solid_image()
        out = apply_jpeg(img, quality=95)
        # 95 returns the original, no JPEG round-trip.
        assert out.size == img.size

    def test_apply_jpeg_low_quality_returns_rgb(self):
        img = _solid_image()
        out = apply_jpeg(img, quality=10)
        assert out.mode == "RGB"
        assert out.size == img.size

    def test_apply_speckle_zero_is_noop(self):
        img = _solid_image()
        out = apply_speckle(img, density=0.0, rng=random.Random(0))
        assert out.tobytes() == img.tobytes()

    def test_apply_speckle_deterministic_under_seed(self):
        img = _solid_image()
        a = apply_speckle(img.copy(), density=0.05, rng=random.Random(42))
        b = apply_speckle(img.copy(), density=0.05, rng=random.Random(42))
        assert a.tobytes() == b.tobytes()


class TestApplyDifficulty:
    @pytest.mark.parametrize("tier", ["easy", "medium", "hard", "extreme"])
    def test_returns_image_and_params(self, tier):
        img = _solid_image()
        out, params = apply_difficulty(img, tier, random.Random(0))
        assert isinstance(out, Image.Image)
        assert {"blur", "rotation", "jpeg", "speckle"} <= set(params)

    def test_unknown_difficulty_raises(self):
        img = _solid_image()
        with pytest.raises(ValueError):
            apply_difficulty(img, "trivial", random.Random(0))

    def test_overrides_take_precedence(self):
        img = _solid_image()
        _, params = apply_difficulty(img, "easy", random.Random(0), overrides={"blur": 10.0})
        assert params["blur"] == 10.0

    def test_extreme_more_corrupted_than_easy(self):
        """At least one of the noise dimensions must be heavier at extreme than at easy."""
        easy_params = VISION_DIFFICULTY_PRESETS["easy"]
        extreme_params = VISION_DIFFICULTY_PRESETS["extreme"]
        assert (
            extreme_params["blur"]    > easy_params["blur"]
            or extreme_params["rotation"] > easy_params["rotation"]
            or extreme_params["jpeg"]     < easy_params["jpeg"]
            or extreme_params["speckle"]  > easy_params["speckle"]
        )
