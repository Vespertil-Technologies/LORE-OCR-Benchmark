"""
Tests for dataset/loader.py.

Locks in the track parameter contract: synthetic loads from data_dir directly,
real_ocr loads from data_dir/vision/, unknown tracks raise.

Sample loading itself is exercised indirectly by the rest of the suite (every
end-to-end test goes through the loader); these tests just pin the new
track-routing behaviour.
"""

import json
from pathlib import Path

import pytest

from dataset.loader import (
    VALID_TRACKS,
    _resolve_track_dir,
    iter_samples,
    load_samples,
)


class TestResolveTrackDir:
    def test_synthetic_returns_root(self, tmp_path):
        assert _resolve_track_dir(tmp_path, "synthetic") == tmp_path

    def test_real_ocr_returns_vision_subdir(self, tmp_path):
        assert _resolve_track_dir(tmp_path, "real_ocr") == tmp_path / "vision"

    def test_unknown_track_raises(self, tmp_path):
        with pytest.raises(ValueError):
            _resolve_track_dir(tmp_path, "made_up")

    def test_known_tracks_set_intact(self):
        assert "synthetic" in VALID_TRACKS
        assert "real_ocr" in VALID_TRACKS


def _write_sample(path: Path, sample: dict) -> None:
    """Write one JSONL sample to path, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")


class TestLoaderRoutesByTrack:
    def test_synthetic_track_finds_root_files(self, tmp_path):
        _write_sample(
            tmp_path / "receipts" / "receipts_easy_train.jsonl",
            {"id": "rec_synth_1", "domain": "receipts", "task": "extraction",
             "difficulty": "easy", "ocr_text": "x", "gt_struct": {}},
        )
        out = load_samples(tmp_path, track="synthetic")
        assert len(out) == 1
        assert out[0]["id"] == "rec_synth_1"

    def test_real_ocr_track_finds_vision_subdir_files(self, tmp_path):
        _write_sample(
            tmp_path / "vision" / "receipts" / "receipts_easy_train.jsonl",
            {"id": "rec_vision_1", "domain": "receipts", "task": "extraction",
             "difficulty": "easy", "ocr_text": "x", "gt_struct": {}},
        )
        out = load_samples(tmp_path, track="real_ocr")
        assert len(out) == 1
        assert out[0]["id"] == "rec_vision_1"

    def test_synthetic_does_not_pick_up_vision_files(self, tmp_path):
        _write_sample(
            tmp_path / "vision" / "receipts" / "receipts_easy_train.jsonl",
            {"id": "rec_vision_1", "domain": "receipts", "task": "extraction",
             "difficulty": "easy", "ocr_text": "x", "gt_struct": {}},
        )
        # No synthetic samples present, so synthetic load should be empty.
        out = load_samples(tmp_path, track="synthetic")
        assert out == []

    def test_real_ocr_does_not_pick_up_synthetic_files(self, tmp_path):
        _write_sample(
            tmp_path / "receipts" / "receipts_easy_train.jsonl",
            {"id": "rec_synth_1", "domain": "receipts", "task": "extraction",
             "difficulty": "easy", "ocr_text": "x", "gt_struct": {}},
        )
        # vision subdir does not exist
        with pytest.raises(FileNotFoundError):
            load_samples(tmp_path, track="real_ocr")


class TestIterSamplesTrackParameter:
    def test_iter_samples_track_parameter_is_passed_through(self, tmp_path):
        _write_sample(
            tmp_path / "vision" / "receipts" / "receipts_easy_dev.jsonl",
            {"id": "rec_vision_2", "domain": "receipts", "task": "extraction",
             "difficulty": "easy", "ocr_text": "x", "gt_struct": {}},
        )
        out = list(iter_samples(tmp_path, track="real_ocr"))
        assert [s["id"] for s in out] == ["rec_vision_2"]

    def test_default_track_is_synthetic(self, tmp_path):
        _write_sample(
            tmp_path / "receipts" / "receipts_easy_train.jsonl",
            {"id": "rec_default_1", "domain": "receipts", "task": "extraction",
             "difficulty": "easy", "ocr_text": "x", "gt_struct": {}},
        )
        # Calling without track= should match synthetic.
        out = list(iter_samples(tmp_path))
        assert [s["id"] for s in out] == ["rec_default_1"]
