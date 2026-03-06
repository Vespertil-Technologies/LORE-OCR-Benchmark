[![DOI](https://zenodo.org/badge/1169776567.svg)](https://doi.org/10.5281/zenodo.18888390)

# LORE — LLM OCR Robustness Evaluation

A benchmark for evaluating how well large language models extract and normalize structured data from corrupted OCR text. Tests models across three document domains, four difficulty tiers, and five evaluation dimensions — with no third-party ML dependencies.

---

## What it measures

Real OCR pipelines produce noisy text — characters substituted, lines merged, values truncated, dates in wrong formats. This benchmark asks: given that corrupted text, can an LLM reconstruct the original structured record?

This is harder than it looks. A model that just echoes the schema back scores high on field presence (F1) but low on value correctness (exact match). The benchmark is designed so that **exact match rate** and **mean normalized edit distance** are the primary metrics — not field F1.

---

## Domains

| Domain | Document type | Fields |
|--------|--------------|--------|
| `receipts` | Retail / food receipts | vendor, date, total, tax, payment method, line items |
| `insurance` | Insurance policy documents | policyholder details, policy number, premium, agent |
| `hospital` | Hospital visit records | patient details, vitals, visit reason, physician, insurance |

---

## Difficulty tiers

Each sample is assigned one of four difficulty levels, controlled by how many and which noise functions are applied:

| Tier | Noise applied | What breaks |
|------|--------------|-------------|
| `easy` | 0–1 character-level errors | Minor substitutions (O→0, l→1) |
| `medium` | 2–3 character + structure | Merged lines, missing colons, date format changes |
| `hard` | 4–5 character + structure + numeric | Decimal shifts, partial dates, value truncation |
| `extreme` | 7–9 all tiers + semantic traps | Value swaps between fields, section erasure, ghost values, conflicting duplicates |

---

## Metrics

| Metric | What it measures | Primary? |
|--------|-----------------|----------|
| `exact_match_rate` | Fraction of fields with exactly correct value | ✓ Primary |
| `mean_ned` | Normalized edit distance on string fields | ✓ Primary |
| `field_f1` | Field presence F1 (precision × recall) | Secondary |
| `hallucination_rate` | Fraction of invented fields not derivable from OCR | Secondary |
| `schema_valid` | Fraction of outputs with correct nested structure | Secondary |
| `correction_gain` | Improvement over raw OCR text (negative = regression) | Secondary |
| `parse_success` | Fraction of outputs parseable as valid JSON | Secondary |

ID-type fields (`policy_number`, `receipt_number`, `attending_physician.id`, `agent.agent_id`) require case-sensitive exact matches — a model cannot score partial credit by returning a plausible-looking identifier.

---

## Results — Llama 3.2 (2B, local, dev split)

| Difficulty | Exact Match | Mean NED |
|------------|-------------|----------|
| easy | 0.731 | 0.208 |
| medium | 0.518 | 0.294 |
| hard | 0.333 | 0.377 |
| extreme | 0.243 | 0.428 |

**Overall:** exact match 0.456 · NED 0.327 · hallucination rate 0.143

---

## Project structure

```
llm_ocr_transformation_benchmark/
├── config/
│   ├── domains.json              # Field schemas and required fields per domain
│   ├── eval_config.json          # Frozen evaluation settings (metrics, thresholds, models)
│   ├── generation_config.json    # Dataset generation settings
│   └── noise_weights.json        # Noise function weights and difficulty ranges
│
├── prompts/
│   └── templates/
│       ├── receipts.txt          # Domain-specific prompt template
│       ├── insurance.txt
│       ├── hospital.txt
│       └── task_instructions.json  # Per-task instruction variants
│
├── dataset/
│   ├── gt_generator.py           # Generates diverse ground-truth records
│   ├── serializer.py             # gt_struct → clean text representation
│   ├── noise_generator.py        # Applies noise tiers to clean text
│   ├── sample_builder.py         # Assembles samples and writes JSONL files
│   └── loader.py                 # Loads and filters dataset files
│
├── runners/
│   ├── prompt_formatter.py       # Injects sample into prompt template
│   ├── llm_adapter.py            # Unified API caller (OpenAI / Anthropic / Ollama)
│   └── multi_run.py              # Crash-safe orchestrator with resume support
│
├── parsers/
│   ├── json_coercion.py          # 5-attempt cascade JSON parser
│   └── normalizers.py            # Date / time / number / phone / string normalizers
│
├── evaluator/
│   ├── field_metrics.py          # Precision, recall, F1, exact match
│   ├── normalization_metrics.py  # Levenshtein NED, numeric tolerance
│   ├── correction_metrics.py     # OCR→LLM correction gain
│   ├── hallucination_detector.py # Fuzzy substring hallucination check
│   └── schema_validator.py       # Structural and type validation
│
├── stats/
│   ├── aggregator.py             # Grouped statistics (overall / domain / difficulty)
│   ├── bootstrap.py              # Bootstrap CI + Wilcoxon signed-rank test
│   └── visuals.py                # ASCII and HTML/SVG chart generators
│
├── report/
│   └── generator.py              # Full pipeline → self-contained HTML report
│
├── data/                         # Generated by sample_builder.py — not committed
└── runs/                         # Generated by multi_run.py — not committed
```

---

## Setup

**Requirements:** Python 3.10+. No ML libraries required — the entire pipeline runs on the standard library.

```bash
git clone https://github.com/ashwin549/lore-benchmark
cd llm-ocr-benchmark
pip install -r requirements.txt
```

**Generate the dataset** (one time):

```bash
python dataset/sample_builder.py
```

This writes 36 JSONL files to `data/` — 1,200 samples total (3 domains × 4 difficulties × 100 samples, split 60/20/20 train/dev/test).

---

## Running benchmarks

### Free — local models via Ollama

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2

python -c "
from runners.multi_run import run
run(run_id='R06', split='dev')   # 60 samples, ~30-60 min on CPU
"

python report/generator.py       # auto-picks most recent run
```

### Paid — OpenAI or Anthropic

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

python -c "
from runners.multi_run import run
run(run_id='R03', split='test')  # GPT-4o, 240 test samples
"
```

### Comparing two models

```bash
python report/generator.py runs/llama3.2_20260301_104150 runs/gpt4o_20260301_120000
```

The report includes a statistical comparison section with 95% bootstrap CIs and Wilcoxon signed-rank p-values.

---

## Adding a model

Add an entry to `config/eval_config.json` under `supported_models`:

```json
"R06": {
    "name":            "llama3.2",
    "backend":         "ollama",
    "temperature":     0,
    "max_tokens":      1024,
    "api_key_env_var": null
}
```

Supported backends: `openai`, `anthropic`, `ollama`, `llama_cpp`.

For Groq (free tier, OpenAI-compatible):

```json
"R07": {
    "name":            "llama-3.1-70b-versatile",
    "backend":         "openai",
    "base_url":        "https://api.groq.com/openai/v1",
    "api_key_env_var": "GROQ_API_KEY",
    "temperature":     0,
    "max_tokens":      1024
}
```

---

## Design decisions

**Why synthetic data?** Real OCR documents contain PII and are hard to license. Synthetic generation lets us control difficulty precisely, guarantee ground truth, and regenerate the dataset at any time.

**Why exact match over F1?** Field F1 measures whether the model returns the right field names — easy for any model that reads the prompt schema. Exact match measures whether the model correctly extracted the actual value from corrupted text, which is the hard part.

**Why no ML dependencies?** Levenshtein distance, bootstrap CI, and Wilcoxon signed-rank are all implemented in pure Python. This keeps the evaluation pipeline auditable, dependency-free, and runnable anywhere.

**Why Indian-context data?** The pools (names, cities, insurers, hospital chains, UPI payments) reflect the document types the noise functions were designed around. The benchmark is domain-specific by design — it tests extraction quality, not world knowledge.

**Seeded reproducibility:** every sample is generated from `base_seed + sample_index`. Running `sample_builder.py` twice with the same seed produces identical datasets.

---

## Noise functions

### Tier 1 — Character level
`char_O0` `char_l1I` `char_B8` `char_S5` `char_sub` `char_del` `char_transpose`

### Tier 2 — Structure level
`missing_colon` `merged_lines` `split_line` `key_abbrev` `value_truncated` `delimiter_swap`

### Tier 3 — Numeric / date
`num_comma_drop` `num_decimal_shift` `date_format_vary` `date_partial`

### Tier 4 — Semantic traps (extreme only)
`extraneous_field` `ghost_value` `conflicting_field` `ambiguous_key` `value_swap` `section_erase`

---

## Citation

If you use LORE in your work, please cite:

```
@misc{lore2026,
  title   = {LORE: LLM OCR Robustness Evaluation},
  year    = {2026},
  url     = {https://github.com/ashwin549/lore-benchmark}
}
```

---

## License

MIT
