"""
report/generator.py

The final stage. Takes a run directory from multi_run.py, runs all
evaluator modules on every prediction record, aggregates results,
and renders a self-contained HTML report.

Pipeline:
    runs/{model}_{ts}/predictions.jsonl
        ↓  evaluate()     - runs all 5 evaluator modules per record
        ↓  aggregate()    - groups stats by domain/difficulty
        ↓  bootstrap_ci() - CIs per metric
        ↓  render()       - writes self-contained HTML report

Output:
    runs/{model}_{ts}/report.html
    runs/{model}_{ts}/evaluated_predictions.jsonl  (predictions + metrics)
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Imports ────────────────────────────────────────────────────────────────────

from parsers.json_coercion   import coerce
from parsers.normalizers     import normalize_struct

from evaluator.field_metrics         import compute_field_metrics
from evaluator.normalization_metrics import compute_normalization_metrics
from evaluator.correction_metrics    import compute_correction_metrics
from evaluator.hallucination_detector import detect_hallucinations
from evaluator.schema_validator      import validate_schema

from stats.aggregator import aggregate, print_summary
from stats.bootstrap  import bootstrap_ci
from stats.visuals    import build_all_charts, ascii_bar_chart

from runners.multi_run import load_predictions

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

DOMAINS   = _load_json(_CONFIG_DIR / "domains.json")
EVAL_CFG  = _load_json(_CONFIG_DIR / "eval_config.json")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 - EVALUATE ONE PREDICTION RECORD
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_record(record: dict) -> dict:
    """
    Run all 5 evaluator modules on one prediction record.
    Returns the record enriched with a "metrics" key.
    """
    domain   = record["domain"]
    ocr_text = record["ocr_text"]
    raw_text = record.get("raw_text", "")
    gt_struct = record["gt_struct"]
    required  = record["required_fields"]

    # ── Parse raw model output ────────────────────────────────────────────
    pred_struct, parse_status = coerce(record.get("model_output", ""))
    record["parse_status"] = parse_status

    # ── Normalize both pred and GT ────────────────────────────────────────
    try:
        pred_norm = normalize_struct(pred_struct, domain, DOMAINS) if pred_struct else {}
        gt_norm   = normalize_struct(gt_struct,   domain, DOMAINS)
    except Exception:
        pred_norm = pred_struct or {}
        gt_norm   = gt_struct

    # ── Schema validation (on raw pred, before normalization) ─────────────
    schema_result = validate_schema(pred_struct, domain, parse_status)

    # ── Field metrics ─────────────────────────────────────────────────────
    field_result = compute_field_metrics(pred_norm, gt_norm, required, domain)

    # ── Normalization metrics ─────────────────────────────────────────────
    norm_result = compute_normalization_metrics(pred_norm, gt_norm, domain)

    # ── Correction metrics ────────────────────────────────────────────────
    corr_result = compute_correction_metrics(
        ocr_text, raw_text, pred_norm, gt_norm, domain, DOMAINS
    )

    # ── Hallucination detection ───────────────────────────────────────────
    hall_result = detect_hallucinations(pred_norm, gt_norm, ocr_text)

    # ── Attach all metrics ────────────────────────────────────────────────
    record["metrics"] = {
        "schema":                schema_result,
        "field_metrics":         field_result,
        "normalization_metrics": norm_result,
        "correction_metrics":    corr_result,
        "hallucination":         hall_result,
    }

    # ── Failure mode classification ───────────────────────────────────────
    record["failure_modes"] = _classify_failure_modes(record)

    return record


def _classify_failure_modes(record: dict) -> list[str]:
    """Return list of failure mode labels for this record."""
    modes = []
    ps    = record.get("parse_status", "failure")
    m     = record.get("metrics", {})

    if ps == "failure":
        modes.append("parse_failure")
    elif ps == "partial":
        modes.append("partial_parse")

    if not m.get("schema", {}).get("schema_valid", True):
        modes.append("schema_mismatch")

    fm = m.get("field_metrics", {})
    if fm.get("required_f1", 1.0) < 1.0:
        modes.append("missing_required")
    if fm.get("exact_match_rate", 1.0) < 1.0 and fm.get("field_f1", 0) > 0:
        modes.append("wrong_value")

    if m.get("hallucination", {}).get("n_hallucinated", 0) > 0:
        modes.append("hallucination")

    gain = m.get("correction_metrics", {}).get("mean_correction_gain")
    if gain is not None and gain < 0:
        modes.append("correction_regression")

    return modes


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 - EVALUATE ALL PREDICTIONS IN A RUN
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_run(run_dir: Path, verbose: bool = True) -> list[dict]:
    """
    Load predictions from a run directory, evaluate every record,
    and write evaluated_predictions.jsonl back to the same directory.

    Returns the list of evaluated records.
    """
    predictions = load_predictions(run_dir)
    evaluated   = []

    if verbose:
        print(f"Evaluating {len(predictions)} predictions from {run_dir.name}...")

    for i, record in enumerate(predictions, start=1):
        ev = evaluate_record(record)
        evaluated.append(ev)
        if verbose and i % 10 == 0:
            print(f"  {i}/{len(predictions)}")

    # Write evaluated predictions
    out_path = run_dir / "evaluated_predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in evaluated:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    if verbose:
        print(f"  Wrote {len(evaluated)} evaluated records → {out_path.name}")

    return evaluated


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 - RENDER HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
body { font-family: 'Segoe UI', system-ui, monospace; margin: 0; background: #f5f5f5; color: #222; }
.wrap { max-width: 980px; margin: 0 auto; padding: 24px; }
h1 { font-size: 1.6em; color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }
h2 { font-size: 1.2em; color: #283593; margin-top: 32px; border-bottom: 1px solid #c5cae9; padding-bottom: 4px; }
h3 { font-size: 1.0em; color: #37474f; margin-top: 20px; }
.meta { background: #e8eaf6; border-radius: 6px; padding: 12px 16px; font-size: 0.88em; line-height: 1.8; }
.meta strong { color: #1a237e; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 16px 0; }
.card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
.card h3 { margin: 0 0 10px; font-size: 0.95em; color: #555; }
.stat-big { font-size: 2.2em; font-weight: bold; color: #1a237e; line-height: 1; }
.stat-label { font-size: 0.78em; color: #888; margin-top: 2px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 0.75em; font-weight: bold; margin: 2px; }
.badge-green  { background: #c8e6c9; color: #1b5e20; }
.badge-yellow { background: #fff9c4; color: #f57f17; }
.badge-red    { background: #ffcdd2; color: #b71c1c; }
.badge-blue   { background: #bbdefb; color: #0d47a1; }
table.scores { width: 100%; border-collapse: collapse; font-size: 0.88em; margin: 12px 0; }
table.scores th { background: #e8eaf6; padding: 7px 10px; text-align: left; border-bottom: 2px solid #9fa8da; }
table.scores td { padding: 6px 10px; border-bottom: 1px solid #e0e0e0; }
table.scores tr:hover td { background: #f5f5f5; }
.good  { color: #2e7d32; font-weight: bold; }
.ok    { color: #f57c00; }
.bad   { color: #c62828; font-weight: bold; }
.failure-mode { display: inline-block; background: #ffecb3; border-radius: 4px;
                padding: 2px 7px; margin: 2px; font-size: 0.8em; }
pre.ocr { background: #263238; color: #cfd8dc; padding: 12px; border-radius: 6px;
          font-size: 0.82em; overflow-x: auto; white-space: pre-wrap; max-height: 200px; }
.section-chart { background: #fff; border-radius: 8px; padding: 16px;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin: 12px 0; }
"""

def _score_class(v: float | None, higher_better: bool = True) -> str:
    if v is None:
        return ""
    if higher_better:
        return "good" if v >= 0.85 else ("ok" if v >= 0.65 else "bad")
    else:
        return "good" if v <= 0.05 else ("ok" if v <= 0.15 else "bad")


def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def render_report(
    evaluated:    list[dict],
    run_dir:      Path,
    run_config:   dict,
    comparison:   dict | None = None,
) -> Path:
    """
    Render a self-contained HTML report and write it to run_dir/report.html.

    Args:
        evaluated:  List of evaluated prediction records.
        run_dir:    Run directory path.
        run_config: Contents of run_config.json.
        comparison: Optional full_comparison() result for a second model.

    Returns:
        Path to the written report.
    """
    # ── Aggregate ─────────────────────────────────────────────────────────
    agg    = aggregate(evaluated)
    charts = build_all_charts(agg, comparison)

    # ── Key headline numbers ──────────────────────────────────────────────
    overall      = agg["overall"]
    field_f1     = overall.get("field_f1", {})
    req_f1       = overall.get("required_f1", {})
    hall_rate    = overall.get("hallucination_rate", {})
    schema_valid = overall.get("schema_valid", {})
    parse_ok     = overall.get("parse_success", {})
    corr_gain    = overall.get("mean_correction_gain", {})
    exact_match  = overall.get("exact_match_rate", {})

    n_total      = agg["n_records"]
    model_name   = run_config.get("model", "unknown")
    ts           = run_config.get("timestamp_utc", "")
    split        = run_config.get("filters", {}).get("split", "test")
    n_fail       = agg["failure_modes"].get("parse_failure", 0)

    # ── Build per-sample table (up to 50 rows) ────────────────────────────
    sample_rows = []
    for rec in sorted(evaluated, key=lambda r: r.get("id", ""))[:50]:
        fm   = rec.get("metrics", {}).get("field_metrics", {})
        hall = rec.get("metrics", {}).get("hallucination", {})
        sch  = rec.get("metrics", {}).get("schema", {})
        f1   = fm.get("field_f1")
        hr   = hall.get("hallucination_rate")
        sv   = sch.get("schema_valid", False)
        ps   = rec.get("parse_status", "?")
        modes = rec.get("failure_modes", [])

        mode_badges = "".join(
            f'<span class="failure-mode">{m}</span>' for m in modes
        ) if modes else '<span style="color:#999">-</span>'

        sample_rows.append(
            f'<tr>'
            f'<td>{rec.get("id","")}</td>'
            f'<td><span class="badge badge-blue">{rec.get("domain","")}</span></td>'
            f'<td>{rec.get("difficulty","")}</td>'
            f'<td>{rec.get("task","")}</td>'
            f'<td class="{_score_class(f1)}">{_fmt(f1)}</td>'
            f'<td class="{_score_class(hr, higher_better=False)}">{_fmt(hr)}</td>'
            f'<td>{"yes" if sv else "no"}</td>'
            f'<td>{ps}</td>'
            f'<td>{mode_badges}</td>'
            f'</tr>'
        )

    # ── Build domain × difficulty table ───────────────────────────────────
    cross_rows = []
    for cross_key in sorted(agg["by_domain_x_difficulty"]):
        stats  = agg["by_domain_x_difficulty"][cross_key]
        domain, diff = cross_key.split("__")
        f1m    = stats.get("field_f1", {}).get("mean")
        rm     = stats.get("required_f1", {}).get("mean")
        nm     = stats.get("mean_ned", {}).get("mean")
        hm     = stats.get("hallucination_rate", {}).get("mean")
        n      = stats.get("field_f1", {}).get("n", 0)

        cross_rows.append(
            f'<tr>'
            f'<td>{domain}</td>'
            f'<td>{diff}</td>'
            f'<td>{n}</td>'
            f'<td class="{_score_class(f1m)}">{_fmt(f1m)}</td>'
            f'<td class="{_score_class(rm)}">{_fmt(rm)}</td>'
            f'<td class="{_score_class(nm, higher_better=False)}">{_fmt(nm)}</td>'
            f'<td class="{_score_class(hm, higher_better=False)}">{_fmt(hm)}</td>'
            f'</tr>'
        )

    # ── Failure mode table ────────────────────────────────────────────────
    fm_rows = "".join(
        f'<tr><td>{mode}</td><td>{count}</td>'
        f'<td>{100*count/n_total:.1f}%</td></tr>'
        for mode, count in sorted(
            agg["failure_modes"].items(), key=lambda x: -x[1]
        )
    )

    # ── Bootstrap CI table ────────────────────────────────────────────────
    ci_metrics = [
        "field_f1", "required_f1", "exact_match_rate",
        "mean_ned", "hallucination_rate", "mean_correction_gain",
        "schema_valid", "parse_success"
    ]
    ci_rows = []
    for metric in ci_metrics:
        vals = [
            v for rec in evaluated
            for v in [_extract_scalar(rec, metric)]
            if v is not None
        ]
        if not vals:
            continue
        ci = bootstrap_ci(vals)
        higher_better = metric not in ("mean_ned", "hallucination_rate")
        cls = _score_class(ci["mean"], higher_better)
        ci_rows.append(
            f'<tr>'
            f'<td>{metric}</td>'
            f'<td class="{cls}">{_fmt(ci["mean"])}</td>'
            f'<td>[{_fmt(ci["ci_lower"])}, {_fmt(ci["ci_upper"])}]</td>'
            f'<td>{_fmt(ci["ci_width"], 4)}</td>'
            f'<td>{ci["n"]}</td>'
            f'</tr>'
        )

    # ── Comparison section ────────────────────────────────────────────────
    comparison_html = ""
    if comparison:
        comparison_html = f"""
        <h2>Model Comparison</h2>
        <div class="section-chart">{charts.get("html_comparison","")}</div>
        <table class="scores">
          <tr><th>Metric</th><th>{comparison['model_a']}</th>
              <th>{comparison['model_b']}</th>
              <th>Δ mean</th><th>p-value</th><th>Significant</th><th>Winner</th></tr>
        """
        for metric, result in comparison["metrics"].items():
            if "error" in result:
                continue
            ca = result["ci_a"]
            cb = result["ci_b"]
            w  = result.get("wilcoxon", {})
            sig = w.get("significant", False)
            comparison_html += (
                f'<tr>'
                f'<td>{metric}</td>'
                f'<td>{_fmt(ca["mean"])} ±{_fmt(ca["ci_width"],4)}</td>'
                f'<td>{_fmt(cb["mean"])} ±{_fmt(cb["ci_width"],4)}</td>'
                f'<td>{_fmt(result["mean_diff"],4)}</td>'
                f'<td>{_fmt(w.get("p_value"),5)}</td>'
                f'<td>{"yes" if sig else "no"}</td>'
                f'<td><strong>{result["winner"]}</strong></td>'
                f'</tr>'
            )
        comparison_html += "</table>"

    # ── Assemble full HTML ────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCR Benchmark - {model_name}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<h1>OCR Benchmark Report</h1>

<div class="meta">
  <strong>Model:</strong> {model_name} &nbsp;|&nbsp;
  <strong>Split:</strong> {split} &nbsp;|&nbsp;
  <strong>Samples:</strong> {n_total} &nbsp;|&nbsp;
  <strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp;
  <strong>Run:</strong> {run_dir.name}
</div>

<h2>Headlines</h2>
<div class="grid">
  <div class="card">
    <h3>Exact Match Rate</h3>
    <div class="stat-big {_score_class(exact_match.get('mean'))}">{_fmt(exact_match.get('mean'))}</div>
    <div class="stat-label">Fraction of fields with correct value</div>
  </div>
  <div class="card">
    <h3>Field F1 (overall)</h3>
    <div class="stat-big {_score_class(field_f1.get('mean'))}">{_fmt(field_f1.get('mean'))}</div>
    <div class="stat-label">Field presence (required F1: {_fmt(req_f1.get('mean'))})</div>
  </div>
  <div class="card">
    <h3>Hallucination Rate</h3>
    <div class="stat-big {_score_class(hall_rate.get('mean'), higher_better=False)}">{_fmt(hall_rate.get('mean'))}</div>
    <div class="stat-label">Fraction of invented fields (lower=better)</div>
  </div>
  <div class="card">
    <h3>Schema Validity</h3>
    <div class="stat-big {_score_class(schema_valid.get('mean'))}">{_fmt(schema_valid.get('mean'))}</div>
    <div class="stat-label">Parse failures: {n_fail}/{n_total}</div>
  </div>
  <div class="card">
    <h3>Correction Gain</h3>
    <div class="stat-big {_score_class(corr_gain.get('mean'))}">{_fmt(corr_gain.get('mean'))}</div>
    <div class="stat-label">OCR→LLM improvement (1.0=perfect, &lt;0=regression)</div>
  </div>
  <div class="card">
    <h3>Mean NED (strings)</h3>
    <div class="stat-big {_score_class(overall.get('mean_ned',{}).get('mean'), higher_better=False)}">{_fmt(overall.get('mean_ned',{}).get('mean'))}</div>
    <div class="stat-label">Normalized edit distance (lower=better)</div>
  </div>
</div>

<h2>Mean NED by Difficulty (lower = better)</h2>
<div class="section-chart">{charts.get("html_f1_by_difficulty","")}</div>

<h2>Exact Match Rate by Domain</h2>
<div class="section-chart">{charts.get("html_f1_by_domain","")}</div>

<h2>Domain × Difficulty Heatmap</h2>
<div class="section-chart">{charts.get("html_heatmap","")}</div>

<h2>Confidence Intervals (95% Bootstrap)</h2>
<table class="scores">
  <tr><th>Metric</th><th>Mean</th><th>95% CI</th><th>CI width</th><th>n</th></tr>
  {"".join(ci_rows)}
</table>

<h2>Domain × Difficulty Breakdown</h2>
<table class="scores">
  <tr><th>Domain</th><th>Difficulty</th><th>n</th>
      <th>Field F1</th><th>Required F1</th><th>Mean NED</th><th>Hall. Rate</th></tr>
  {"".join(cross_rows)}
</table>

<h2>Failure Mode Breakdown</h2>
<table class="scores">
  <tr><th>Failure Mode</th><th>Count</th><th>Rate</th></tr>
  {fm_rows}
</table>

{comparison_html}

<h2>Per-Sample Results (first 50)</h2>
<table class="scores">
  <tr><th>ID</th><th>Domain</th><th>Difficulty</th><th>Task</th>
      <th>Field F1</th><th>Hall. Rate</th><th>Schema</th>
      <th>Parse</th><th>Failure Modes</th></tr>
  {"".join(sample_rows)}
</table>

<p style="color:#aaa;font-size:0.8em;margin-top:32px;border-top:1px solid #eee;padding-top:8px">
  OCR Benchmark v{EVAL_CFG.get('_version','1.0.0')} - Generated {time.strftime('%Y-%m-%d %H:%M UTC')}
</p>

</div>
</body>
</html>"""

    out_path = run_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _extract_scalar(record: dict, metric: str) -> float | None:
    """Pull one scalar metric value from an evaluated record."""
    m    = record.get("metrics", {})
    fm   = m.get("field_metrics", {})
    nm   = m.get("normalization_metrics", {})
    cm   = m.get("correction_metrics", {})
    hall = m.get("hallucination", {})
    sch  = m.get("schema", {})
    dispatch = {
        "field_f1":               fm.get("field_f1"),
        "required_f1":            fm.get("required_f1"),
        "optional_f1":            fm.get("optional_f1"),
        "exact_match_rate":       fm.get("exact_match_rate"),
        "mean_ned":               nm.get("mean_ned"),
        "numeric_tolerance_rate": nm.get("numeric_tolerance_rate"),
        "mean_correction_gain":   cm.get("mean_correction_gain"),
        "hallucination_rate":     hall.get("hallucination_rate"),
        "schema_valid":           float(sch.get("schema_valid", False)) if "schema_valid" in sch else None,
        "parse_success":          float(record.get("parse_status") == "success"),
    }
    return dispatch.get(metric)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    run_dir:    Path,
    comparison_run_dir: Path | None = None,
    verbose:    bool = True,
) -> Path:
    """
    Full pipeline: load → evaluate → aggregate → render HTML report.

    Args:
        run_dir:              Path to a completed run directory.
        comparison_run_dir:   Optional second run to compare against.
        verbose:              Print progress.

    Returns:
        Path to the generated report.html.
    """
    # Load run config
    config_path = run_dir / "run_config.json"
    run_config  = json.loads(config_path.read_text()) if config_path.exists() else {}

    # Evaluate predictions
    evaluated = evaluate_run(run_dir, verbose=verbose)

    # Optional: compare two models
    comparison = None
    if comparison_run_dir:
        from stats.bootstrap import full_comparison
        comp_eval   = evaluate_run(comparison_run_dir, verbose=False)
        comp_config = json.loads((comparison_run_dir / "run_config.json").read_text()) \
                      if (comparison_run_dir / "run_config.json").exists() else {}
        comparison = full_comparison(
            evaluated, comp_eval,
            run_config.get("model", "Model A"),
            comp_config.get("model", "Model B"),
        )

    # Render
    report_path = render_report(evaluated, run_dir, run_config, comparison)

    if verbose:
        print(f"\nReport written → {report_path}")
        _print_terminal_summary(evaluated)

    return report_path


def _print_terminal_summary(evaluated: list[dict]) -> None:
    """Print a concise summary table to the terminal."""
    agg = aggregate(evaluated)
    print()
    print_summary(agg, "exact_match_rate")

    print("\nFailure modes:")
    for mode, count in sorted(agg["failure_modes"].items(), key=lambda x: -x[1]):
        pct = 100 * count / agg["n_records"]
        print(f"  {mode:<30} {count:>4}  ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Usage:
        # Generate report for your most recent run:
        python report/generator.py

        # Or specify a run directory:
        python report/generator.py runs/llama3.2_20260228_123817

        # Compare two runs:
        python report/generator.py runs/llama3.2_20260228_123817 runs/mistral_20260228_150000
    """
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    runs_root = Path(__file__).parent.parent / "runs"

    # Collect all run dirs, sorted newest first
    if not runs_root.exists():
        print("No runs/ directory found yet.")
        print("Run a model first: python -c \"from runners.multi_run import run; run(run_id='R06', split='test')\"")
        sys.exit(1)

    all_runs = sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and (d / "predictions.jsonl").exists()],
        reverse=True,
    )

    if not all_runs:
        print("No completed runs found in runs/ directory.")
        print("Run a model first: python -c \"from runners.multi_run import run; run(run_id='R06', split='test')\"")
        sys.exit(1)

    # Parse command-line arguments
    if len(sys.argv) >= 2:
        run_dir_a = Path(sys.argv[1])
    else:
        run_dir_a = all_runs[0]
        print(f"No run specified - using most recent: {run_dir_a.name}")

    comparison_dir = None
    if len(sys.argv) >= 3:
        comparison_dir = Path(sys.argv[2])

    print(f"\nGenerating report for: {run_dir_a.name}")
    if comparison_dir:
        print(f"Comparing against:     {comparison_dir.name}")

    report_path = generate_report(
        run_dir=run_dir_a,
        comparison_run_dir=comparison_dir,
        verbose=True,
    )

    print(f"\nReport size: {report_path.stat().st_size:,} bytes")
    print(f"Open in browser: {report_path}")