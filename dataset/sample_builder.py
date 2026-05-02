"""
dataset/sample_builder.py

Orchestrates the full pipeline from gt_struct → one complete JSONL sample dict.

Pipeline:
    gt_struct
        ↓  serializer.py
    raw_text
        ↓  noise_generator.py
    (corrupted_text, noise_tags)
        ↓  assemble
    sample dict  (ready to write as one JSONL line)

Responsibilities:
    - Derive per-sample seed from base_seed + sample_index
    - Call serializer → raw_text
    - Call noise_generator → (corrupted_text, noise_tags)
    - Assemble the complete sample dict
    - Generate unique sample IDs
    - Look up required_fields from domains.json
    - Does NOT write to disk - that is loader/writer's job
    - Does NOT know about splits - the dataset writer assigns those
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.noise_generator import generate_noise
from dataset.serializer import serialize

# ── Config ─────────────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_json(filename: str) -> dict:
    with open(_CONFIG_DIR / filename, encoding="utf-8") as f:
        return json.load(f)

DOMAINS     = _load_json("domains.json")
GEN_CONFIG  = _load_json("generation_config.json")


# ── ID generation ──────────────────────────────────────────────────────────────

def _make_id(domain: str, index: int) -> str:
    """
    Generate a unique sample ID.
    Format: {prefix}_{index:06d}
    Example: ins_000042, hosp_000001, rec_000123
    """
    prefix = GEN_CONFIG["id_prefixes"][domain]
    return f"{prefix}_{index:06d}"


# ── Required fields lookup ─────────────────────────────────────────────────────

def _get_required_fields(domain: str) -> list[str]:
    """Return the list of required field paths for a domain from domains.json."""
    return DOMAINS[domain]["required_fields"]


# ── Task assignment ────────────────────────────────────────────────────────────

def _assign_task(difficulty: str, rng: random.Random) -> str:
    """
    Randomly assign a task label based on the difficulty-weighted distribution
    defined in generation_config.json.
    """
    dist = GEN_CONFIG["task_distribution"][difficulty]
    tasks = [t for t in dist if not t.startswith("_")]
    weights = [dist[t] for t in tasks]
    return rng.choices(tasks, weights=weights, k=1)[0]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_sample(
    gt_struct: dict,
    domain: str,
    difficulty: str,
    sample_index: int,
    base_seed: int,
    notes: str = "",
) -> dict[str, Any]:
    """
    Build one complete sample dict from a clean gt_struct.

    Args:
        gt_struct:      Clean ground-truth dict matching the domain schema.
        domain:         'receipts', 'insurance', or 'hospital'.
        difficulty:     'easy', 'medium', 'hard', or 'extreme'.
        sample_index:   Globally unique integer index for this sample.
                        Used to derive a per-sample seed.
        base_seed:      Base random seed from generation_config.json.
        notes:          Optional annotator notes or ambiguity flags.

    Returns:
        A dict with all required JSONL fields, ready to be json.dumps()'d.
    """
    # Derive a reproducible per-sample seed
    per_sample_seed = base_seed + sample_index

    # Two separate RNGs - one for serializer, one for noise
    # This ensures serializer randomness doesn't bleed into noise randomness
    serialize_rng = random.Random(per_sample_seed)
    noise_rng     = random.Random(per_sample_seed + 1_000_000)
    task_rng      = random.Random(per_sample_seed + 2_000_000)

    # Step 1 - Serialize gt_struct → raw text
    raw_text = serialize(gt_struct, domain, serialize_rng)

    # Step 2 - Inject noise → corrupted OCR text + applied tags
    corrupted_text, noise_tags = generate_noise(
        raw_text=raw_text,
        difficulty=difficulty,
        domain=domain,
        rng=noise_rng,
        gt_struct=gt_struct,
    )

    # Step 3 - Assign task label
    task = _assign_task(difficulty, task_rng)

    # Step 4 - Assemble complete sample dict
    sample = {
        "id":              _make_id(domain, sample_index),
        "domain":          domain,
        "task":            task,
        "difficulty":      difficulty,
        "source_type":     GEN_CONFIG["source_type"],  # "synthetic"
        "ocr_text":        corrupted_text,
        "gt_struct":       gt_struct,
        "noise_tags":      noise_tags,
        "required_fields": _get_required_fields(domain),
        "generation_meta": {
            "base_seed":        base_seed,
            "sample_index":     sample_index,
            "per_sample_seed":  per_sample_seed,
            "raw_text":         raw_text,          # stored for correction_metrics
        },
        "notes": notes,
    }

    return sample


# ══════════════════════════════════════════════════════════════════════════════
# BATCH BUILDER - generates N samples for one (domain, difficulty) combination
# ══════════════════════════════════════════════════════════════════════════════

def build_batch(
    gt_structs: list[dict],
    domain: str,
    difficulty: str,
    start_index: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """
    Build multiple samples from a list of gt_structs.

    Args:
        gt_structs:   List of clean ground-truth dicts.
        domain:       Domain name.
        difficulty:   Difficulty level.
        start_index:  Sample index to start from (for globally unique IDs).
        base_seed:    Base random seed.

    Returns:
        List of complete sample dicts.
    """
    samples = []
    for i, gt_struct in enumerate(gt_structs):
        sample = build_sample(
            gt_struct=gt_struct,
            domain=domain,
            difficulty=difficulty,
            sample_index=start_index + i,
            base_seed=base_seed,
        )
        samples.append(sample)
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# DATASET WRITER - splits samples and writes JSONL files to disk
# ══════════════════════════════════════════════════════════════════════════════

def write_dataset(
    samples: list[dict],
    domain: str,
    difficulty: str,
    output_dir: Path,
    split_ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, Path]:
    """
    Split samples into train/dev/test and write one JSONL file per split.

    File naming: {output_dir}/{domain}/{domain}_{difficulty}_{split}.jsonl

    Args:
        samples:       List of complete sample dicts from build_batch().
        domain:        Domain name.
        difficulty:    Difficulty level.
        output_dir:    Root output directory (e.g. Path("data/")).
        split_ratios:  Dict like {"train": 0.6, "dev": 0.2, "test": 0.2}.
        rng:           Seeded RNG for reproducible shuffling.

    Returns:
        Dict mapping split name → path of written file.
    """
    # Shuffle before splitting - deterministic given the seed
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    splits = list(split_ratios.keys())
    ratios = list(split_ratios.values())

    # Compute split boundaries
    boundaries = [0]
    cumulative = 0
    for ratio in ratios[:-1]:
        cumulative += ratio
        boundaries.append(int(n * cumulative))
    boundaries.append(n)

    # Write each split
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    written_paths = {}
    for i, split_name in enumerate(splits):
        split_samples = shuffled[boundaries[i]:boundaries[i+1]]
        filepath = domain_dir / f"{domain}_{difficulty}_{split_name}.jsonl"

        with open(filepath, "w", encoding="utf-8") as f:
            for sample in split_samples:
                # Add split label into the sample
                sample_with_split = {**sample, "split": split_name}
                f.write(json.dumps(sample_with_split, ensure_ascii=False) + "\n")

        written_paths[split_name] = filepath
        print(f"  Wrote {len(split_samples):>4} samples → {filepath}")

    return written_paths


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run this to build the full benchmark dataset.
    Writes 36 JSONL files to data/ (3 domains × 4 difficulties × 3 splits).
    1,200 samples total - each with a distinct gt_struct from gt_generator.py.

    Usage:
        python dataset/sample_builder.py
    """
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.path.insert(0, str(Path(__file__).parent))
    from gt_generator import generate_batch

    BASE_SEED  = GEN_CONFIG["base_seed"]   # 42
    OUTPUT_DIR = Path(__file__).parent.parent / "data"
    ALL_DOMAINS  = ["receipts", "insurance", "hospital"]
    DIFFICULTIES = ["easy", "medium", "hard", "extreme"]

    # 100 samples per (domain × difficulty) cell = 1,200 total
    SAMPLES_PER_CELL = 100

    print("Building full benchmark dataset...")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Samples: {len(ALL_DOMAINS)} domains × {len(DIFFICULTIES)} difficulties "
          f"× {SAMPLES_PER_CELL} = {len(ALL_DOMAINS)*len(DIFFICULTIES)*SAMPLES_PER_CELL} total\n")

    global_index = 0

    for domain in ALL_DOMAINS:
        # Generate 400 distinct gt_structs for this domain up front
        # (100 per difficulty cell, no overlap)
        all_gts = generate_batch(domain, n=SAMPLES_PER_CELL * len(DIFFICULTIES), seed=BASE_SEED)

        for diff_idx, difficulty in enumerate(DIFFICULTIES):
            cell_gts = all_gts[diff_idx * SAMPLES_PER_CELL : (diff_idx + 1) * SAMPLES_PER_CELL]

            samples = build_batch(
                gt_structs=cell_gts,
                domain=domain,
                difficulty=difficulty,
                start_index=global_index,
                base_seed=BASE_SEED,
            )

            written = write_dataset(
                samples=samples,
                domain=domain,
                difficulty=difficulty,
                output_dir=OUTPUT_DIR,
                split_ratios=GEN_CONFIG["split_ratios"],
                rng=random.Random(BASE_SEED + global_index),
            )

            global_index += SAMPLES_PER_CELL
            splits_written = {sp: p.name for sp, p in written.items()}
            print(f"  {domain:<12} {difficulty:<8}  {SAMPLES_PER_CELL} samples  → {splits_written}")

    # ── Verify final manifest ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Manifest")
    print("=" * 60)

    from loader import print_manifest
    print_manifest(OUTPUT_DIR)
