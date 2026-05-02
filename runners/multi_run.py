"""
runners/multi_run.py

Orchestrates a full evaluation run - iterates samples, formats prompts,
calls the LLM, and writes results to disk incrementally.

Responsibilities:
    - Load samples from data/ using loader.py filters
    - Build prompts via prompt_formatter.py
    - Call the LLM via llm_adapter.py
    - Write predictions.jsonl incrementally (crash-safe)
    - Write run_config.json alongside predictions for full reproducibility
    - Support dry_run mode (skips actual API calls, writes mock output)
    - Support resume (skip already-completed sample IDs)

Output structure:
    runs/{model_name}_{timestamp}/
        predictions.jsonl   - one line per sample, written as it completes
        run_config.json     - frozen config snapshot for this run
        call_log.jsonl      - one CallRecord per sample
"""

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.loader       import load_samples, dataset_manifest
from runners.prompt_formatter import build_prompt
from runners.llm_adapter  import call, build_model_cfg, CallRecord, EVAL_CONFIG

# ── Config ─────────────────────────────────────────────────────────────────────

_BASE_DIR  = Path(__file__).parent.parent
_DATA_DIR  = _BASE_DIR / "data"
_RUNS_DIR  = _BASE_DIR / "runs"


# ══════════════════════════════════════════════════════════════════════════════
# RUN DIRECTORY SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _make_run_dir(model_name: str) -> Path:
    """Create and return a timestamped run directory."""
    safe_name = model_name.replace("/", "-").replace(":", "-")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir   = _RUNS_DIR / f"{safe_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_run_config(
    run_dir:    Path,
    run_id:     str,
    model_cfg:  dict,
    filters:    dict,
    n_samples:  int,
    dry_run:    bool,
) -> None:
    """Write a frozen snapshot of this run's configuration."""
    config = {
        "run_id":          run_id,
        "model":           model_cfg.get("name"),
        "backend":         model_cfg.get("backend"),
        "temperature":     model_cfg.get("temperature"),
        "max_tokens":      model_cfg.get("max_tokens"),
        "filters":         filters,
        "n_samples":       n_samples,
        "dry_run":         dry_run,
        "eval_config_version":     EVAL_CONFIG["_version"],
        "timestamp_utc":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# RESUME SUPPORT
# ══════════════════════════════════════════════════════════════════════════════

def _load_completed_ids(predictions_path: Path) -> set[str]:
    """Read a partially written predictions.jsonl and return all sample IDs already done."""
    if not predictions_path.exists():
        return set()
    completed = set()
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    completed.add(record["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION RECORD ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def _make_prediction_record(
    sample:       dict,
    model_output: str,
    prompt_hash:  str,
    call_record:  CallRecord,
) -> dict:
    """
    Assemble the prediction record written to predictions.jsonl.
    Contains everything needed by the evaluator - no re-loading required.
    """
    return {
        # Sample identity
        "id":          sample["id"],
        "domain":      sample["domain"],
        "task":        sample["task"],
        "difficulty":  sample["difficulty"],
        "split":       sample.get("split", "unknown"),

        # Prompt traceability
        "prompt_hash":  prompt_hash,
        "model":        call_record.model,
        "backend":      call_record.backend,
        "latency_ms":   call_record.latency_ms,
        "success":      call_record.success,
        "error":        call_record.error,

        # Raw model output (unparsed - evaluator handles parsing)
        "model_output": model_output,

        # Ground truth and OCR text (evaluator needs these)
        "ocr_text":        sample["ocr_text"],
        "gt_struct":       sample["gt_struct"],
        "noise_tags":      sample["noise_tags"],
        "required_fields": sample["required_fields"],

        # Raw text before noise (for correction_metrics)
        "raw_text": sample.get("generation_meta", {}).get("raw_text", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DRY RUN MOCK OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _mock_model_output(sample: dict) -> str:
    """
    In dry_run mode, return a plausible-looking (but fake) JSON string.
    Used to test the full pipeline without API calls.
    """
    domain = sample["domain"]
    if domain == "receipts":
        return json.dumps({
            "vendor_name":  "DryRun Mart",
            "date":         "2024-03-14",
            "total_amount": 368.16,
            "currency":     "INR",
        })
    elif domain == "insurance":
        return json.dumps({
            "policyholder": {"name": "Dry Run", "dob": "2002-08-12"},
            "policy":       {"policy_number": "DR-000001"},
            "premium":      {"amount": 5000, "currency": "INR"},
        })
    else:  # hospital
        return json.dumps({
            "patient": {"name": "Dry Patient", "dob": "1995-11-07"},
            "visit":   {"date": "2024-03-14", "reason_for_visit": "Dry run complaint"},
        })


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run(
    run_id:     str,
    split:      str = "test",
    domain:     str | None = None,
    difficulty: str | None = None,
    task:       str | None = None,
    n:          int | None = None,
    dry_run:    bool = False,
    resume:     bool = False,
    run_dir:    Path | None = None,
) -> Path:
    """
    Execute a full evaluation run for a given model configuration.

    Args:
        run_id:     Model run ID from eval_config.json (e.g. 'R03', 'A01').
        split:      Which data split to evaluate on. Default 'test'.
        domain:     Optional domain filter.
        difficulty: Optional difficulty filter.
        task:       Optional task filter.
        n:          Optional - evaluate only N samples (for quick checks).
        dry_run:    If True, skip actual API calls and use mock outputs.
        resume:     If True, skip samples already in predictions.jsonl.
        run_dir:    Optional - provide an existing run dir to resume into.

    Returns:
        Path to the run directory containing predictions.jsonl.
    """
    # Force UTF-8 stdout so progress lines with non-ASCII characters do not
    # crash on Windows consoles (default cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_cfg = build_model_cfg(run_id)

    # ── Setup run directory ────────────────────────────────────────────────
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if run_dir is None:
        # Auto-resume: look for an existing partial run for this model
        # Match any run folder for this model that has partial predictions
        # Use a loose prefix match - folder name uses whatever sanitization _make_run_dir applied
        model_prefix = re.sub(r"[^\w]", "_", model_cfg["name"])  # same as _make_run_dir
        existing = sorted(
            [d for d in _RUNS_DIR.iterdir()
             if d.is_dir() and re.sub(r"[^\w]", "_", d.name.split("_2026")[0]) == model_prefix
             and (d / "predictions.jsonl").exists()
             and not (d / "evaluated_predictions.jsonl").exists()],
            reverse=True  # most recent first
        )
        if existing:
            run_dir = existing[0]
            resume = True
            print(f"Auto-resuming existing run: {run_dir.name}")
        else:
            run_dir = _make_run_dir(model_cfg["name"])

    predictions_path = run_dir / "predictions.jsonl"
    call_log_path    = run_dir / "call_log.jsonl"

    # ── Load samples ───────────────────────────────────────────────────────
    samples = load_samples(
        _DATA_DIR,
        domain=domain,
        difficulty=difficulty,
        split=split,
        task=task,
        n=n,
        seed=EVAL_CONFIG.get("base_seed", 42),
    )

    filters = {
        "split": split, "domain": domain,
        "difficulty": difficulty, "task": task, "n": n,
    }

    _write_run_config(run_dir, run_id, model_cfg, filters, len(samples), dry_run)

    # ── Resume: skip already-done samples ─────────────────────────────────
    completed_ids: set[str] = set()
    if resume:
        completed_ids = _load_completed_ids(predictions_path)
        skipped = len([s for s in samples if s["id"] in completed_ids])
        if skipped:
            print(f"Resuming - skipping {skipped} already-completed samples.")

    todo = [s for s in samples if s["id"] not in completed_ids]

    # ── Run ────────────────────────────────────────────────────────────────
    print(f"\nRun ID    : {run_id}")
    print(f"Model     : {model_cfg['name']} ({model_cfg['backend']})")
    print(f"Split     : {split}")
    print(f"Samples   : {len(todo)} to process  ({len(completed_ids)} already done)")
    print(f"Output    : {run_dir}")
    print(f"Dry run   : {dry_run}")
    print()

    with (
        open(predictions_path, "a", encoding="utf-8") as pred_f,
        open(call_log_path,    "a", encoding="utf-8") as log_f,
    ):
        for i, sample in enumerate(todo, start=1):
            sample_id = sample["id"]

            # Build prompt
            prompt, prompt_hash = build_prompt(sample)

            # Get model output
            if dry_run:
                model_output = _mock_model_output(sample)
                call_record  = CallRecord(
                    sample_id       = sample_id,
                    model           = model_cfg["name"],
                    backend         = "dry_run",
                    prompt_hash     = prompt_hash,
                    timestamp_utc   = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    latency_ms      = 0,
                    response_length = len(model_output),
                    success         = True,
                    error           = None,
                )
            else:
                model_output, call_record = call(
                    prompt=prompt,
                    prompt_hash=prompt_hash,
                    sample_id=sample_id,
                    model_cfg=model_cfg,
                )

            # Check for rate limit - stop immediately, don't save as failure
            if not call_record.success and call_record.error:
                err = call_record.error.lower()
                is_rate_limit = any(kw in err for kw in ["rate limit", "429", "too many requests", "quota", "tokens per day", "tokens per minute"])
                if is_rate_limit:
                    print(f"\n  Rate limit hit on sample {i}/{len(todo)} - stopping to avoid saving failed records.")
                    print(f"  Re-run the same command tomorrow to resume from sample {i}.")
                    print(f"  ({i-1} samples saved successfully)")
                    break

            # Assemble and write prediction record immediately
            pred_record = _make_prediction_record(
                sample, model_output, prompt_hash, call_record
            )
            pred_f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
            pred_f.flush()  # Ensure it hits disk - crash-safe

            # Write call log
            log_f.write(json.dumps(call_record.to_dict()) + "\n")
            log_f.flush()

            # Progress
            status = "OK" if call_record.success else "FAIL"
            print(f"  [{i:>4}/{len(todo)}]  {sample_id:<16}  {status}  {call_record.latency_ms}ms")

    print(f"\nDone. {len(todo)} samples written to {predictions_path}")
    return run_dir


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS READER - for evaluator use
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(run_dir: Path) -> list[dict]:
    """
    Load all prediction records from a completed run directory.

    Args:
        run_dir: Path to a run directory (contains predictions.jsonl).

    Returns:
        List of prediction record dicts.
    """
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions.jsonl found in {run_dir}")

    records = []
    with open(pred_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at predictions.jsonl:{line_num} - {e}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("PART 1 - Dry run: all test samples, all domains")
    print("=" * 60)

    run_dir = run(
        run_id   = "R03",   # GPT-4o (uses dry_run so no real API call)
        split    = "test",
        dry_run  = True,
    )

    print()
    print("=" * 60)
    print("PART 2 - Load and inspect predictions")
    print("=" * 60)

    predictions = load_predictions(run_dir)
    print(f"Total predictions loaded: {len(predictions)}")
    print()

    # Show breakdown by domain + difficulty
    from collections import Counter
    breakdown = Counter(
        (p["domain"], p["difficulty"]) for p in predictions
    )
    print("Domain × Difficulty breakdown:")
    for (domain, diff), count in sorted(breakdown.items()):
        print(f"  {domain:<12}  {diff:<8}  {count:>3} samples")

    print()
    print("First prediction record (keys):")
    first = predictions[0]
    for key in first:
        val = first[key]
        preview = str(val)[:60] + "..." if len(str(val)) > 60 else str(val)
        print(f"  {key:<20} : {preview}")

    print()
    print("=" * 60)
    print("PART 3 - Dry run: single domain, single difficulty")
    print("=" * 60)

    run(
        run_id     = "R04",     # Claude (dry_run)
        split      = "test",
        domain     = "hospital",
        difficulty = "extreme",
        dry_run    = True,
    )

    print()
    print("=" * 60)
    print("PART 4 - run_config.json contents")
    print("=" * 60)

    with open(run_dir / "run_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    print(json.dumps(cfg, indent=2))