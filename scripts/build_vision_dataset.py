"""
scripts/build_vision_dataset.py

Generate the vision-track companion to the synthetic dataset.

Each sample is rendered as a document image, degraded with image-level noise
keyed to its difficulty tier, and OCR'd through Tesseract. The resulting
ocr_text is whatever Tesseract produced, mirroring how a real production
pipeline would consume scanned documents.

Output paths come from config/generation_config.json -> vision_track:
    image_output_dir : where rendered noisy PNGs are saved
    jsonl_output_dir : where the JSONL files land (parallel to data/)

Usage:
    python scripts/build_vision_dataset.py

Requires the 'vision' extra (pillow, pytesseract) and a system Tesseract
binary on PATH. Run python -m runners.ocr_engine_check or just import
runners.ocr_engine and call is_available() to verify your install before
launching a long generation.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.gt_generator import generate_batch  # noqa: E402
from dataset.sample_builder import build_batch, write_dataset  # noqa: E402
from runners.ocr_engine import is_available  # noqa: E402

CONFIG_PATH = ROOT / "config" / "generation_config.json"


def main() -> int:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    vision_cfg = cfg.get("vision_track", {})

    base_seed       = cfg["base_seed"]
    domains         = cfg["domains"]
    difficulties    = cfg["difficulties"]
    samples_per_cell = cfg["sample_counts"][cfg["target"]]["per_difficulty_per_domain"]
    split_ratios    = cfg["split_ratios"]

    image_dir = ROOT / vision_cfg.get("image_output_dir", "data/vision_images/")
    jsonl_dir = ROOT / vision_cfg.get("jsonl_output_dir", "data/vision/")

    ok, msg = is_available()
    if not ok:
        print("[FAIL] Tesseract not available:")
        for line in msg.splitlines():
            print(f"   {line}")
        return 2
    print(f"[OK] Tesseract: {msg}")
    print(f"     Output: JSONL -> {jsonl_dir}")
    print(f"             Images -> {image_dir}")
    print(f"     Generating {len(domains)} x {len(difficulties)} x {samples_per_cell} samples")
    print()

    global_index = 0
    for domain in domains:
        all_gts = generate_batch(domain, n=samples_per_cell * len(difficulties), seed=base_seed)
        for diff_idx, difficulty in enumerate(difficulties):
            cell_gts = all_gts[diff_idx * samples_per_cell:(diff_idx + 1) * samples_per_cell]
            samples = build_batch(
                gt_structs=cell_gts,
                domain=domain,
                difficulty=difficulty,
                start_index=global_index,
                base_seed=base_seed,
                track="real_ocr",
                image_dir=image_dir,
            )
            write_dataset(
                samples=samples,
                domain=domain,
                difficulty=difficulty,
                output_dir=jsonl_dir,
                split_ratios=split_ratios,
                rng=random.Random(base_seed + global_index),
            )
            global_index += samples_per_cell

    print()
    print("[OK] Vision-track generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
