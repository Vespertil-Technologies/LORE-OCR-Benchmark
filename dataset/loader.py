"""
dataset/loader.py

Loads JSONL dataset files and serves samples for evaluation.

Responsibilities:
    - Discover JSONL files matching the {domain}_{difficulty}_{split}.jsonl pattern
    - Filter by any combination of domain / difficulty / split / task / noise_tag
    - Validate every line is parseable JSON on load
    - Support sampling N random samples (for quick dev runs)
    - Return samples as plain dicts - no framework dependencies
    - Generator mode for large files (memory efficient)

Does NOT:
    - Write anything to disk
    - Know about model runners or evaluation
    - Modify sample contents in any way
"""

import json
import random
from collections.abc import Generator
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

VALID_DOMAINS     = {"receipts", "insurance", "hospital"}
VALID_DIFFICULTIES = {"easy", "medium", "hard", "extreme"}
VALID_SPLITS      = {"train", "dev", "test"}
VALID_TASKS       = {"extraction", "normalization", "correction", "hallucination", "schema"}
VALID_TRACKS      = {"synthetic", "real_ocr"}

# Track-to-subdirectory mapping. The synthetic track lives at the data root;
# real_ocr lives under data_dir/vision so the two tracks can coexist.
_TRACK_SUBDIR: dict[str, str] = {
    "synthetic": "",
    "real_ocr":  "vision",
}


def _resolve_track_dir(data_dir: Path, track: str) -> Path:
    """Return the subdirectory of data_dir that holds samples for the given track."""
    if track not in VALID_TRACKS:
        raise ValueError(f"Invalid track '{track}'. Must be one of: {sorted(VALID_TRACKS)}")
    sub = _TRACK_SUBDIR[track]
    return data_dir / sub if sub else data_dir


# ══════════════════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_files(
    data_dir: Path,
    domain: str | None = None,
    difficulty: str | None = None,
    split: str | None = None,
) -> list[Path]:
    """
    Find all JSONL files under data_dir matching the naming pattern
    {domain}_{difficulty}_{split}.jsonl, filtered by any provided arguments.

    Args:
        data_dir:   Root data directory (e.g. Path("data/")).
        domain:     Optional filter - 'receipts', 'insurance', or 'hospital'.
        difficulty: Optional filter - 'easy', 'medium', 'hard', or 'extreme'.
        split:      Optional filter - 'train', 'dev', or 'test'.

    Returns:
        Sorted list of matching Path objects.

    Raises:
        ValueError: If a provided filter value is not a valid option.
        FileNotFoundError: If data_dir does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if domain and domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of: {VALID_DOMAINS}")
    if difficulty and difficulty not in VALID_DIFFICULTIES:
        raise ValueError(f"Invalid difficulty '{difficulty}'. Must be one of: {VALID_DIFFICULTIES}")
    if split and split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {VALID_SPLITS}")

    # Build glob pattern - use wildcards for unspecified dimensions
    d  = domain     or "*"
    di = difficulty or "*"
    s  = split      or "*"

    pattern = f"{d}/{d}_{di}_{s}.jsonl"
    files = sorted(data_dir.glob(pattern))

    # If domain specified, also try without subdirectory (flat layout)
    if not files:
        pattern_flat = f"{d}_{di}_{s}.jsonl"
        files = sorted(data_dir.glob(pattern_flat))

    return files


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def _passes_filters(
    sample: dict,
    task: str | None,
    noise_tags: list[str] | None,
) -> bool:
    """Return True if a sample passes all active filters."""
    if task and sample.get("task") != task:
        return False
    if noise_tags:
        sample_tags = set(sample.get("noise_tags", []))
        if not all(tag in sample_tags for tag in noise_tags):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def iter_samples(
    data_dir: Path,
    domain: str | None = None,
    difficulty: str | None = None,
    split: str | None = None,
    task: str | None = None,
    noise_tags: list[str] | None = None,
    track: str = "synthetic",
) -> Generator[dict, None, None]:
    """
    Generator that yields samples one at a time. Memory-efficient for large files.

    Args:
        data_dir:   Root data directory.
        domain:     Optional domain filter.
        difficulty: Optional difficulty filter.
        split:      Optional split filter.
        task:       Optional task filter.
        noise_tags: Optional list of noise tags - yields only samples containing
                    ALL of the listed tags.
        track:      Which corruption track to load. 'synthetic' (default) reads
                    from data_dir directly. 'real_ocr' reads from data_dir/vision/
                    where scripts/build_vision_dataset.py writes its output.

    Yields:
        Sample dicts, one at a time.

    Raises:
        ValueError:  On invalid filter values or malformed JSON lines.
    """
    if task and task not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Must be one of: {VALID_TASKS}")

    track_dir = _resolve_track_dir(data_dir, track)
    files = discover_files(track_dir, domain, difficulty, split)

    if not files:
        return  # No files matched - yield nothing

    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Malformed JSON at {filepath}:{line_num} - {e}"
                    ) from e
                if _passes_filters(sample, task, noise_tags):
                    yield sample


def load_samples(
    data_dir: Path,
    domain: str | None = None,
    difficulty: str | None = None,
    split: str | None = None,
    task: str | None = None,
    noise_tags: list[str] | None = None,
    n: int | None = None,
    seed: int = 42,
    track: str = "synthetic",
) -> list[dict]:
    """
    Load all matching samples into memory as a list.

    Args:
        data_dir:   Root data directory.
        domain:     Optional domain filter.
        difficulty: Optional difficulty filter.
        split:      Optional split filter.
        task:       Optional task filter.
        noise_tags: Optional list of noise tags (ALL must be present).
        n:          If set, return a random sample of N items instead of all.
        seed:       Random seed for reproducible sampling when n is set.
        track:      'synthetic' (default) or 'real_ocr'. See iter_samples.

    Returns:
        List of sample dicts.
    """
    samples = list(iter_samples(data_dir, domain, difficulty, split, task, noise_tags, track))

    if n is not None:
        if n > len(samples):
            raise ValueError(
                f"Requested n={n} but only {len(samples)} samples matched the filters."
            )
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST & STATS
# ══════════════════════════════════════════════════════════════════════════════

def dataset_manifest(data_dir: Path) -> dict:
    """
    Scan the data directory and return a summary of what's present.

    Returns a nested dict:
        {domain: {difficulty: {split: count}}}
    plus top-level 'total' and 'files_found'.
    """
    files = discover_files(data_dir)
    manifest: dict = {"files_found": len(files), "total": 0, "domains": {}}

    for filepath in files:
        # Parse filename: {domain}_{difficulty}_{split}.jsonl
        stem = filepath.stem  # e.g. "insurance_hard_test"
        parts = stem.split("_")
        if len(parts) < 3:
            continue

        split      = parts[-1]
        difficulty = parts[-2]
        domain     = "_".join(parts[:-2])

        # Count lines in file
        with open(filepath, encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())

        manifest["domains"] \
            .setdefault(domain, {}) \
            .setdefault(difficulty, {})[split] = count

        manifest["total"] += count

    return manifest


def print_manifest(data_dir: Path) -> None:
    """Pretty-print the dataset manifest to stdout."""
    m = dataset_manifest(data_dir)
    print(f"Data directory : {data_dir}")
    print(f"Files found    : {m['files_found']}")
    print(f"Total samples  : {m['total']}")
    print()

    for domain, difficulties in sorted(m["domains"].items()):
        print(f"  {domain}/")
        for difficulty, splits in sorted(difficulties.items()):
            row = "  ".join(f"{s}: {c:>4}" for s, c in sorted(splits.items()))
            domain_total = sum(splits.values())
            print(f"    {difficulty:<10}  {row}   (total: {domain_total})")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    DATA_DIR = Path(__file__).parent.parent / "data"

    # ── Part 1: Manifest - what's in the dataset ───────────────────────────
    print("=" * 60)
    print("PART 1 - Dataset manifest")
    print("=" * 60)
    print_manifest(DATA_DIR)

    # ── Part 2: Load all test samples for one domain ───────────────────────
    print("=" * 60)
    print("PART 2 - Load all insurance test samples")
    print("=" * 60)
    samples = load_samples(DATA_DIR, domain="insurance", split="test")
    print(f"Loaded {len(samples)} samples")
    difficulties_seen = {s["difficulty"] for s in samples}
    tasks_seen        = {s["task"] for s in samples}
    print(f"Difficulties : {sorted(difficulties_seen)}")
    print(f"Tasks        : {sorted(tasks_seen)}")

    # ── Part 3: Filter by difficulty + task ────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 3 - Filter: hospital / extreme / hallucination task")
    print("=" * 60)
    filtered = load_samples(
        DATA_DIR,
        domain="hospital",
        difficulty="extreme",
        task="hallucination",
    )
    print(f"Matched {len(filtered)} sample(s)")
    for s in filtered:
        print(f"  {s['id']}  noise_tags: {s['noise_tags']}")

    # ── Part 4: Filter by noise tag ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 4 - Filter by noise tag: only samples with 'conflicting_field'")
    print("=" * 60)
    tagged = load_samples(DATA_DIR, noise_tags=["conflicting_field"])
    print(f"Matched {len(tagged)} sample(s) containing 'conflicting_field'")
    for s in tagged:
        print(f"  {s['id']}  domain: {s['domain']}  difficulty: {s['difficulty']}")

    # ── Part 5: Random subsample (n=3) for quick dev run ──────────────────
    print("\n" + "=" * 60)
    print("PART 5 - Random subsample: n=3 from all test data")
    print("=" * 60)
    subsample = load_samples(DATA_DIR, split="test", n=3, seed=42)
    print(f"Sampled {len(subsample)} items:")
    for s in subsample:
        print(f"  {s['id']}  {s['domain']:<12} {s['difficulty']:<8} task: {s['task']}")

    # ── Part 6: Generator mode ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 6 - Generator mode (memory efficient iteration)")
    print("=" * 60)
    total = sum(1 for _ in iter_samples(DATA_DIR, domain="receipts"))
    print(f"Iterated {total} receipt samples via generator")
