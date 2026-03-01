"""
stats/visuals.py

Generates charts from aggregated benchmark results.
No matplotlib or external plotting dependencies — produces:
    1. ASCII bar charts for terminal output
    2. HTML/SVG charts embedded in the report

Charts produced:
    - F1 by difficulty (bar chart per model)
    - F1 by domain (grouped bar)
    - Metric radar/heatmap (domain × difficulty grid)
    - Hallucination rate by difficulty
    - Correction gain distribution
    - Model comparison bars with CI error bars (HTML only)
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# ASCII CHARTS — for terminal/log output
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    """Render a horizontal bar for a value in [0, 1]."""
    filled = int(round(value * width))
    return fill * filled + empty * (width - filled)


def ascii_bar_chart(
    data:   dict[str, float],
    title:  str = "",
    width:  int = 30,
    fmt:    str = ".4f",
) -> str:
    """
    Render a simple horizontal bar chart to a string.

    Args:
        data:  Dict mapping label → value (values should be in [0, 1]).
        title: Chart title.
        width: Bar width in characters.
        fmt:   Format string for values.
    """
    lines = []
    if title:
        lines.append(title)
        lines.append("─" * (width + 30))

    max_label = max((len(k) for k in data), default=10)
    for label, value in data.items():
        if value is None:
            bar = "─" * width + "  N/A"
        else:
            bar = _bar(value, width) + f"  {value:{fmt}}"
        lines.append(f"  {label:<{max_label}}  {bar}")

    return "\n".join(lines)


def ascii_heatmap(
    data:        dict[str, dict[str, float]],
    row_label:   str = "domain",
    col_label:   str = "difficulty",
    metric:      str = "field_f1",
    cols:        list[str] | None = None,
) -> str:
    """
    Render a domain × difficulty heatmap using block characters.
    ▓ = high, ▒ = mid, ░ = low, · = missing.
    """
    if cols is None:
        cols = ["easy", "medium", "hard", "extreme"]

    def _block(v: float | None) -> str:
        if v is None: return "·   "
        if v >= 0.85: return "▓▓▓ "
        if v >= 0.70: return "▒▒▒ "
        if v >= 0.50: return "░░░ "
        return "··· "

    lines = [f"  {metric}  ({row_label} × {col_label})"]
    col_header = "".join(f"{c[:6]:<7}" for c in cols)
    lines.append(f"  {'':15}  {col_header}")
    lines.append("  " + "─" * (16 + 7 * len(cols)))

    for row_key, col_data in sorted(data.items()):
        cells = "".join(_block(col_data.get(c)) for c in cols)
        # Add numeric values for readability
        vals = "  ".join(
            f"{col_data.get(c):.2f}" if col_data.get(c) is not None else " N/A"
            for c in cols
        )
        lines.append(f"  {row_key:<15}  {cells}  [{vals}]")

    lines.append(f"\n  Legend: ▓▓▓=≥0.85  ▒▒▒=≥0.70  ░░░=≥0.50  ···=<0.50")
    return "\n".join(lines)


def ascii_comparison(
    comparison: dict,
    metric:     str = "field_f1",
) -> str:
    """Render a two-model comparison for one metric as ASCII bars."""
    a_name = comparison["model_a"]
    b_name = comparison["model_b"]
    result = comparison["metrics"].get(metric, {})
    if "error" in result:
        return f"  {metric}: ERROR — {result['error']}"

    ca = result["ci_a"]
    cb = result["ci_b"]
    lines = [
        f"  {metric}  (n={comparison['n_paired']} paired samples)",
        f"  {a_name:<20}  {_bar(ca['mean'], 25)}  {ca['mean']} [{ca['ci_lower']}, {ca['ci_upper']}]",
        f"  {b_name:<20}  {_bar(cb['mean'], 25)}  {cb['mean']} [{cb['ci_lower']}, {cb['ci_upper']}]",
        f"  Winner: {result['winner']}",
    ]
    w = result.get("wilcoxon", {})
    if w.get("p_value") is not None:
        lines.append(f"  Wilcoxon p={w['p_value']}  sig={'✓' if w['significant'] else '✗'}  "
                     f"effect_size={w['effect_size']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# HTML/SVG CHART GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "easy":     "#4CAF50",
    "medium":   "#FFC107",
    "hard":     "#FF5722",
    "extreme":  "#9C27B0",
    "receipts": "#2196F3",
    "insurance":"#00BCD4",
    "hospital": "#E91E63",
    "model_a":  "#1565C0",
    "model_b":  "#C62828",
}

def _color_for(label: str) -> str:
    return _COLORS.get(label.lower(), "#607D8B")


def html_bar_chart(
    data:        dict[str, float],
    title:       str  = "",
    ci_data:     dict[str, dict] | None = None,
    width_px:    int  = 480,
    bar_height:  int  = 32,
    x_max:       float = 1.0,
) -> str:
    """
    Generate an inline SVG horizontal bar chart.

    Args:
        data:       dict label → value
        title:      Chart title
        ci_data:    Optional dict label → {ci_lower, ci_upper} for error bars
        width_px:   Total SVG width
        bar_height: Height of each bar + gap
        x_max:      Maximum x-axis value (default 1.0)
    """
    label_w  = 140
    bar_area = width_px - label_w - 60
    n        = len(data)
    svg_h    = n * bar_height + 60
    items    = list(data.items())

    svg = [f'<svg width="{width_px}" height="{svg_h}" '
           f'xmlns="http://www.w3.org/2000/svg" '
           f'style="font-family:monospace;font-size:12px;background:#fafafa;'
           f'border-radius:6px;padding:4px">']

    if title:
        svg.append(f'<text x="{width_px//2}" y="20" text-anchor="middle" '
                   f'font-size="13" font-weight="bold" fill="#333">{title}</text>')

    for i, (label, value) in enumerate(items):
        y = 35 + i * bar_height
        bar_w = int((value / x_max) * bar_area) if value is not None else 0
        color = _color_for(label)

        # Label
        svg.append(f'<text x="{label_w - 6}" y="{y + 16}" '
                   f'text-anchor="end" fill="#444">{label}</text>')
        # Background track
        svg.append(f'<rect x="{label_w}" y="{y + 4}" '
                   f'width="{bar_area}" height="{bar_height - 10}" '
                   f'fill="#e8e8e8" rx="3"/>')
        # Bar
        if bar_w > 0:
            svg.append(f'<rect x="{label_w}" y="{y + 4}" '
                       f'width="{bar_w}" height="{bar_height - 10}" '
                       f'fill="{color}" rx="3" opacity="0.85"/>')
        # Value label
        val_str = f"{value:.3f}" if value is not None else "N/A"
        svg.append(f'<text x="{label_w + bar_w + 6}" y="{y + 16}" '
                   f'fill="#333">{val_str}</text>')

        # CI error bars
        if ci_data and label in ci_data:
            ci = ci_data[label]
            lo_x = label_w + int((ci["ci_lower"] / x_max) * bar_area)
            hi_x = label_w + int((ci["ci_upper"] / x_max) * bar_area)
            mid_y = y + (bar_height - 10) // 2 + 4
            svg.append(f'<line x1="{lo_x}" y1="{mid_y - 5}" '
                       f'x2="{lo_x}" y2="{mid_y + 5}" stroke="#333" stroke-width="1.5"/>')
            svg.append(f'<line x1="{hi_x}" y1="{mid_y - 5}" '
                       f'x2="{hi_x}" y2="{mid_y + 5}" stroke="#333" stroke-width="1.5"/>')
            svg.append(f'<line x1="{lo_x}" y1="{mid_y}" '
                       f'x2="{hi_x}" y2="{mid_y}" stroke="#333" stroke-width="1"/>')

    svg.append('</svg>')
    return "\n".join(svg)


def html_heatmap_table(
    data:          dict[str, dict[str, float | None]],
    title:         str = "",
    cols:          list[str] | None = None,
    metric:        str = "mean_ned",
    higher_better: bool = False,
) -> str:
    """
    Generate an HTML table heatmap (domain × difficulty).
    Cells are colored green→red based on value.
    higher_better=True  → high values are green (e.g. F1)
    higher_better=False → low values are green (e.g. NED)
    """
    if cols is None:
        cols = ["easy", "medium", "hard", "extreme"]

    def _cell_color(v: float | None) -> str:
        if v is None: return "#e0e0e0"
        # Clamp to [0, 1] range
        v = max(0.0, min(1.0, v))
        score = v if higher_better else (1 - v)
        r = int(255 * (1 - score))
        g = int(200 * score)
        b = 60
        return f"rgb({r},{g},{b})"

    rows = [
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:12px">',
        f'<caption style="font-weight:bold;padding:6px">{title} — {metric}</caption>',
        '<tr><th style="padding:6px;border:1px solid #ccc">Domain</th>',
    ]
    for col in cols:
        rows.append(f'<th style="padding:6px;border:1px solid #ccc">{col}</th>')
    rows.append('</tr>')

    for domain, col_data in sorted(data.items()):
        rows.append(f'<tr><td style="padding:6px;border:1px solid #ccc;'
                    f'font-weight:bold">{domain}</td>')
        for col in cols:
            v    = col_data.get(col)
            bg   = _cell_color(v)
            txt  = f"{v:.3f}" if v is not None else "—"
            rows.append(
                f'<td style="padding:6px;border:1px solid #ccc;'
                f'background:{bg};text-align:center;color:#111">{txt}</td>'
            )
        rows.append('</tr>')

    rows.append('</table>')
    return "\n".join(rows)


def html_comparison_chart(comparison: dict, metrics: list[str] | None = None) -> str:
    """
    Generate an SVG grouped bar chart comparing two models across metrics.
    """
    if metrics is None:
        metrics = ["field_f1", "required_f1", "exact_match_rate",
                   "mean_correction_gain", "schema_valid"]

    a_name = comparison["model_a"]
    b_name = comparison["model_b"]
    n_metrics = len(metrics)

    width_px   = 620
    left_pad   = 180
    bar_area   = width_px - left_pad - 40
    row_h      = 48
    svg_h      = n_metrics * row_h + 80

    svg = [
        f'<svg width="{width_px}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:monospace;font-size:11px;background:#fafafa;border-radius:6px">',
        f'<text x="{width_px//2}" y="18" text-anchor="middle" font-size="13" '
        f'font-weight="bold" fill="#333">{a_name} vs {b_name}</text>',
        # Legend
        f'<rect x="{left_pad}" y="28" width="12" height="12" fill="{_COLORS["model_a"]}" opacity="0.85"/>',
        f'<text x="{left_pad+16}" y="39" fill="#333">{a_name}</text>',
        f'<rect x="{left_pad+120}" y="28" width="12" height="12" fill="{_COLORS["model_b"]}" opacity="0.85"/>',
        f'<text x="{left_pad+136}" y="39" fill="#333">{b_name}</text>',
    ]

    bar_h = 14
    gap   = 4

    for i, metric in enumerate(metrics):
        result = comparison["metrics"].get(metric, {})
        if "error" in result or not result:
            continue

        y_base = 55 + i * row_h
        ca = result["ci_a"]
        cb = result["ci_b"]

        # Metric label
        svg.append(f'<text x="{left_pad - 6}" y="{y_base + bar_h + gap}" '
                   f'text-anchor="end" fill="#444">{metric}</text>')

        for ci, color, y_off in [
            (ca, _COLORS["model_a"], 0),
            (cb, _COLORS["model_b"], bar_h + gap),
        ]:
            mean = ci.get("mean") or 0
            lo   = ci.get("ci_lower") or 0
            hi   = ci.get("ci_upper") or 0
            bar_w = int(mean * bar_area)
            lo_x  = left_pad + int(lo * bar_area)
            hi_x  = left_pad + int(hi * bar_area)
            mid_y = y_base + y_off + bar_h // 2

            # Background
            svg.append(f'<rect x="{left_pad}" y="{y_base + y_off}" '
                       f'width="{bar_area}" height="{bar_h}" fill="#e8e8e8" rx="2"/>')
            # Bar
            svg.append(f'<rect x="{left_pad}" y="{y_base + y_off}" '
                       f'width="{bar_w}" height="{bar_h}" fill="{color}" opacity="0.8" rx="2"/>')
            # CI error bar
            svg.append(f'<line x1="{lo_x}" y1="{mid_y-4}" x2="{lo_x}" y2="{mid_y+4}" '
                       f'stroke="#333" stroke-width="1.5"/>')
            svg.append(f'<line x1="{hi_x}" y1="{mid_y-4}" x2="{hi_x}" y2="{mid_y+4}" '
                       f'stroke="#333" stroke-width="1.5"/>')
            svg.append(f'<line x1="{lo_x}" y1="{mid_y}" x2="{hi_x}" y2="{mid_y}" '
                       f'stroke="#333" stroke-width="1"/>')
            # Value
            svg.append(f'<text x="{left_pad + bar_w + 4}" y="{y_base + y_off + bar_h - 2}" '
                       f'fill="#333">{mean:.3f}</text>')

        # Winner badge
        winner = result.get("winner", "")
        sig    = result.get("wilcoxon", {}).get("significant", False)
        badge_color = "#4CAF50" if sig else "#9E9E9E"
        svg.append(f'<text x="{width_px - 6}" y="{y_base + bar_h}" '
                   f'text-anchor="end" fill="{badge_color}" font-size="10">'
                   f'{"★" if sig else "≈"}</text>')

    svg.append('</svg>')
    return "\n".join(svg)


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE — build all charts for one aggregate result
# ══════════════════════════════════════════════════════════════════════════════

def build_all_charts(agg: dict, comparison: dict | None = None) -> dict[str, str]:
    """
    Build all standard charts from an aggregate result.

    Returns dict mapping chart name → HTML/SVG string or ASCII string.
    """
    charts = {}

    # F1 by difficulty
    diff_ned = {
        diff: agg["by_difficulty"].get(diff, {}).get("mean_ned", {}).get("mean")
        for diff in ["easy", "medium", "hard", "extreme"]
    }
    charts["ascii_f1_by_difficulty"] = ascii_bar_chart(
        diff_ned, title="Mean NED by Difficulty (lower=better)"
    )
    charts["html_f1_by_difficulty"] = html_bar_chart(
        diff_ned, title="Mean NED by Difficulty (lower=better)"
    )

    # F1 by domain
    domain_exact = {
        domain: agg["by_domain"].get(domain, {}).get("exact_match_rate", {}).get("mean")
        for domain in ["receipts", "insurance", "hospital"]
    }
    charts["ascii_f1_by_domain"] = ascii_bar_chart(
        domain_exact, title="Exact Match Rate by Domain"
    )
    charts["html_f1_by_domain"] = html_bar_chart(
        domain_exact, title="Exact Match Rate by Domain"
    )

    # Hallucination by difficulty
    hall_by_diff = {
        diff: agg["by_difficulty"].get(diff, {}).get("hallucination_rate", {}).get("mean")
        for diff in ["easy", "medium", "hard", "extreme"]
    }
    charts["ascii_hallucination"] = ascii_bar_chart(
        hall_by_diff, title="Hallucination Rate by Difficulty"
    )

    # Domain × difficulty heatmap (field_f1)
    heatmap_data: dict[str, dict[str, float | None]] = {}
    for cross_key, stats in agg["by_domain_x_difficulty"].items():
        domain, diff = cross_key.split("__")
        heatmap_data.setdefault(domain, {})[diff] = stats.get("mean_ned", {}).get("mean")

    charts["ascii_heatmap"] = ascii_heatmap(
        heatmap_data, metric="mean_ned"
    )
    charts["html_heatmap"] = html_heatmap_table(
        heatmap_data, title="Mean NED (lower=better)", metric="mean_ned", higher_better=False
    )

    # Model comparison (if provided)
    if comparison:
        charts["ascii_comparison"] = ascii_comparison(comparison, "field_f1")
        charts["html_comparison"]  = html_comparison_chart(comparison)

    return charts


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, random as _rnd
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from stats.aggregator import aggregate
    from stats.bootstrap  import full_comparison

    # Simulate evaluated records
    rng = _rnd.Random(42)
    domains      = ["receipts", "insurance", "hospital"]
    difficulties = ["easy", "medium", "hard", "extreme"]
    dp           = {"easy": 0.0, "medium": 0.08, "hard": 0.18, "extreme": 0.32}

    def _sim(n, base, noise, seed):
        r = _rnd.Random(seed)
        records = []
        for i in range(n):
            d  = r.choice(difficulties)
            f1 = max(0, min(1, base - dp[d] + r.gauss(0, noise)))
            hr = max(0, r.uniform(0, 0.08) + dp[d] * 0.25)
            ok = r.random() > dp[d] * 0.25
            records.append({
                "id": f"s{i:04d}", "domain": r.choice(domains),
                "difficulty": d, "task": "extraction",
                "parse_status": "success" if ok else "failure",
                "metrics": {
                    "field_metrics":        {"field_f1": round(f1,4), "required_f1": round(min(1,f1+.05),4), "optional_f1": round(max(0,f1-.1),4), "exact_match_rate": round(f1*.9,4)},
                    "normalization_metrics":{"mean_ned": round(max(0,.2-f1*.15),4), "numeric_tolerance_rate": round(f1*.95,4)},
                    "correction_metrics":   {"mean_correction_gain": round(f1-.1,4)},
                    "hallucination":        {"hallucination_rate": round(hr,4), "n_hallucinated": 0},
                    "schema":               {"schema_valid": ok},
                },
            })
        return records

    recs_a = _sim(80, 0.87, 0.07, 1)
    recs_b = _sim(80, 0.73, 0.09, 2)
    agg_a  = aggregate(recs_a)
    comp   = full_comparison(recs_a, recs_b, "GPT-4o", "Llama-8B")
    charts = build_all_charts(agg_a, comp)

    print("=" * 60)
    print("PART 1 — ASCII: F1 by difficulty")
    print("=" * 60)
    print(charts["ascii_f1_by_difficulty"])

    print("\n" + "=" * 60)
    print("PART 2 — ASCII: Domain × Difficulty heatmap")
    print("=" * 60)
    print(charts["ascii_heatmap"])

    print("\n" + "=" * 60)
    print("PART 3 — ASCII: Model comparison")
    print("=" * 60)
    print(charts["ascii_comparison"])

    print("\n" + "=" * 60)
    print("PART 4 — HTML chart sizes (bytes)")
    print("=" * 60)
    for name, html in charts.items():
        if name.startswith("html"):
            print(f"  {name:<30} {len(html):>6} bytes")

    # Write HTML preview
    out_path = Path("/mnt/user-data/outputs/charts_preview.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_body = "\n<br><br>\n".join(
        f"<h3>{name}</h3>\n{html}"
        for name, html in charts.items()
        if name.startswith("html")
    )
    out_path.write_text(
        f"<!DOCTYPE html><html><body style='font-family:monospace;padding:20px'>"
        f"<h2>OCR Benchmark — Chart Preview</h2>{html_body}</body></html>"
    )
    print(f"\n  HTML preview written → {out_path}")