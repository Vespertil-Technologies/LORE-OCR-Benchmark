[![CI](https://github.com/Vespertil-Technologies/LORE-OCR-Benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Vespertil-Technologies/LORE-OCR-Benchmark/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1169776567.svg)](https://doi.org/10.5281/zenodo.18888390)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# LORE: LLM OCR Robustness Evaluation

A benchmark for evaluating how well large language models extract and normalize structured data from corrupted OCR text. Tests models across three document domains, four difficulty tiers, and seven evaluation metrics, with no third-party ML dependencies in the core pipeline.

## What it measures

Real OCR pipelines produce noisy text: characters substituted, lines merged, values truncated, dates in wrong formats. This benchmark asks the question "given that corrupted text, can an LLM reconstruct the original structured record?".

This is harder than it looks. A model that just echoes the schema back scores high on field presence (F1) but low on value correctness (exact match). The benchmark is designed so that **exact match rate** and **mean normalized edit distance** are the primary metrics, not field F1.

## Domains

| Domain | Document type | Fields |
|--------|--------------|--------|
| `receipts` | Retail and food receipts | vendor, date, total, tax, payment method, line items |
| `insurance` | Insurance policy documents | policyholder details, policy number, premium, agent |
| `hospital` | Hospital visit records | patient details, vitals, visit reason, physician, insurance |

## Difficulty tiers

Each sample is assigned one of four difficulty levels, controlled by how many and which noise functions are applied:

| Tier | Noise applied | What breaks |
|------|--------------|-------------|
| `easy` | 0 to 1 character-level errors | Minor substitutions (O to 0, l to 1) |
| `medium` | 2 to 3 character + structure | Merged lines, missing colons, date format changes |
| `hard` | 4 to 5 character + structure + numeric | Decimal shifts, partial dates, value truncation |
| `extreme` | 7 to 9 across all tiers + semantic traps | Value swaps between fields, section erasure, ghost values, conflicting duplicates |

## Metrics

| Metric | What it measures | Primary |
|--------|-----------------|---------|
| `exact_match_rate` | Fraction of fields with exactly correct value | yes |
| `mean_ned` | Normalized edit distance on string fields | yes |
| `field_f1` | Field presence F1 across all fields (required and optional) | secondary |
| `required_f1` | Field presence F1 computed only over required fields | secondary |
| `hallucination_rate` | Fraction of invented fields not derivable from OCR | secondary |
| `schema_valid` | Fraction of outputs with correct nested structure | secondary |
| `correction_gain` | Improvement over raw OCR text (negative means regression) | secondary |
| `parse_success` | Fraction of outputs parseable as valid JSON | secondary |

ID-type fields (`policy_number`, `receipt_number`, `attending_physician.id`, `agent.agent_id`) require case-sensitive exact matches. A model cannot score partial credit by returning a plausible-looking identifier.

## Results

Verifiable baseline runs are committed under [`results/`](results/). Open the `report.html` files directly in a browser, no run required.

| Model | Backend | Split | Exact match | Mean NED | Required F1 | Hallucination |
|-------|---------|-------|-------------|----------|-------------|---------------|
| `regex_rules` | hardcoded | dev | 0.3629 | 0.4393 | 0.8085 | 0.0425 |
| `always_null` | hardcoded | dev | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

The full leaderboard (auto-generated from `results/<model>/summary.json`) lives at [`results/leaderboard.md`](results/leaderboard.md). To add your own model, see [Submitting a run](#submitting-a-run).

### Indicative LLM result

A previous Llama 3.2 (3B, local, dev split) run produced these numbers. The artifacts are not committed; reproduce locally with `run_id='R06'`:

| Difficulty | Exact match | Mean NED |
|------------|-------------|----------|
| easy | 0.731 | 0.208 |
| medium | 0.518 | 0.294 |
| hard | 0.333 | 0.377 |
| extreme | 0.243 | 0.428 |

Overall: exact match 0.456, NED 0.327, hallucination rate 0.143.

## Project structure

```
LORE-OCR-Benchmark/
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
│       └── task_instructions.json
│
├── dataset/
│   ├── gt_generator.py           # Generates diverse ground-truth records
│   ├── serializer.py             # gt_struct to clean text representation
│   ├── noise_generator.py        # Applies synthetic noise tiers to clean text
│   ├── renderer.py               # (vision track) text to PIL document image
│   ├── image_noise.py            # (vision track) blur / rotation / JPEG / speckle
│   ├── vision_pipeline.py        # (vision track) render + noise + OCR orchestration
│   ├── sample_builder.py         # Assembles samples and writes JSONL files
│   └── loader.py                 # Loads and filters dataset files
│
├── runners/
│   ├── prompt_formatter.py       # Injects sample into prompt template
│   ├── llm_adapter.py            # Unified API caller (OpenAI / Anthropic / Ollama / llama_cpp)
│   ├── ocr_engine.py             # (vision track) Tesseract wrapper
│   ├── baselines.py              # Non-LLM baselines (always_null, regex_rules)
│   └── multi_run.py              # Crash-safe orchestrator with resume support
│
├── parsers/
│   ├── json_coercion.py          # 5-attempt cascade JSON parser
│   └── normalizers.py            # Date / time / number / phone / string normalizers
│
├── evaluator/
│   ├── field_metrics.py          # Precision, recall, F1, exact match
│   ├── normalization_metrics.py  # Levenshtein NED, numeric tolerance
│   ├── correction_metrics.py     # OCR vs LLM correction gain
│   ├── hallucination_detector.py # Fuzzy substring hallucination check
│   └── schema_validator.py       # Structural and type validation
│
├── stats/
│   ├── aggregator.py             # Grouped statistics (overall / domain / difficulty)
│   ├── bootstrap.py              # Bootstrap CI + Wilcoxon signed-rank test
│   └── visuals.py                # ASCII and HTML/SVG chart generators
│
├── report/
│   └── generator.py              # Full pipeline to self-contained HTML report + summary.json
│
├── tests/                        # pytest suite (173 tests, pure Python, no model deps)
│
├── scripts/
│   ├── snapshot_run.py           # Copies a run's artifacts into results/<model>/
│   ├── build_leaderboard.py      # Aggregates results/*/summary.json into results/leaderboard.md
│   └── build_vision_dataset.py   # Generates the optional real-OCR companion dataset
│
├── results/                      # Committed leaderboard receipts (one folder per submitted model)
├── data/                         # Generated by sample_builder.py, not committed
└── runs/                         # Generated by multi_run.py, not committed
```

## Quickstart

Requirements: Python 3.10 or newer. The core pipeline runs on the standard library; the LLM backends are optional extras.

```bash
git clone https://github.com/Vespertil-Technologies/LORE-OCR-Benchmark.git
cd LORE-OCR-Benchmark
pip install -e .[dev]            # core + pytest + ruff + mypy
```

Optional backend extras:

```bash
pip install -e .[openai]         # OpenAI / OpenAI-compatible (Groq)
pip install -e .[anthropic]      # Anthropic Claude
pip install -e .[vision]         # Real-OCR track (Pillow + pytesseract; needs Tesseract binary)
pip install -e .[all]            # all of the above
```

Generate the dataset (one time):

```bash
python dataset/sample_builder.py
```

This writes 36 JSONL files to `data/` (1,200 samples total: 3 domains x 4 difficulties x 100 samples, split 60/20/20 train/dev/test).

Run a baseline end-to-end (no API needed):

```bash
python -c "from runners.multi_run import run; run(run_id='R01', split='dev')"
python report/generator.py        # auto-picks the most recent run
```

## Running benchmarks

### Baselines (no API, instant)

| Run ID | Name | What it does |
|--------|------|--------------|
| `R00` | `always_null` | Returns `{}` for every sample. The minimum-effort floor. |
| `R01` | `regex_rules` | Extracts `key: value` pairs from OCR text, mapping labels to canonical fields via `domains.json`. |

```bash
python -c "from runners.multi_run import run; run(run_id='R00', split='dev')"
python -c "from runners.multi_run import run; run(run_id='R01', split='dev')"
```

Every model the benchmark evaluates should beat both baselines. If it does not, something is wrong with the prompt, the parser, or the model.

### Free: local models via Ollama

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2

python -c "from runners.multi_run import run; run(run_id='R06', split='dev')"
python report/generator.py
```

### Paid: OpenAI or Anthropic

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

python -c "from runners.multi_run import run; run(run_id='R03', split='test')"
```

### Comparing two models

```bash
python report/generator.py runs/<model_a>_<timestamp> runs/<model_b>_<timestamp>
```

The report includes a statistical comparison section with 95% bootstrap CIs and Wilcoxon signed-rank p-values.

## Vision track (optional, real OCR)

The default benchmark applies *synthetic* noise functions to clean serialized text. There is also an optional vision track that swaps the synthetic noise step for real OCR: clean text gets rendered as a document image, the image is degraded (blur, rotation, JPEG compression, speckle), and Tesseract is run on the result. The OCR output is what the LLM under test sees.

The synthetic track is the canonical benchmark (reproducible from a seed, no system dependencies, four difficulty tiers tied to specific noise functions). The vision track is additive: it tests the same models against real OCR errors instead of simulated ones.

### Requirements

```bash
pip install -e .[vision]                                # Pillow + pytesseract
sudo apt install tesseract-ocr tesseract-ocr-eng        # Linux
brew install tesseract                                  # macOS
choco install tesseract                                 # Windows (or grab the UB-Mannheim installer)
```

### Generate the vision-track dataset

```bash
python scripts/build_vision_dataset.py
```

This renders 1,200 PNGs into `data/vision_images/<domain>/<difficulty>/` and writes parallel JSONL files to `data/vision/`. The synthetic dataset under `data/` is left untouched so the two tracks can co-exist.

### Difficulty mapping

The four tiers map to image-noise intensity rather than to synthetic noise functions:

| Tier | Blur radius | Rotation | JPEG quality | Speckle density |
|------|-------------|----------|--------------|-----------------|
| `easy`    | 0.4 | 0.0 deg | 95 | 0.000 |
| `medium`  | 0.8 | 1.0 deg | 80 | 0.001 |
| `hard`    | 1.4 | 2.0 deg | 65 | 0.003 |
| `extreme` | 2.0 | 3.5 deg | 50 | 0.006 |

### Run a model against the vision dataset

The same `runners.multi_run.run` entry point works; point its loader at `data/vision/` instead of `data/`. See `dataset/vision_pipeline.py` and `scripts/build_vision_dataset.py` for the exact wiring.

## Submitting a run

After running a model, snapshot the artifacts into `results/` and refresh the leaderboard:

```bash
python report/generator.py runs/<model>_<timestamp>      # writes report.html + summary.json
python scripts/snapshot_run.py runs/<model>_<timestamp>  # copies into results/<model>/
python scripts/build_leaderboard.py                      # rewrites results/leaderboard.md
```

Then open a pull request adding the new entries under `results/`. Anyone reading the repo can verify the claim by opening the committed `report.html` in a browser.

## Adding a model

Add an entry to `config/eval_config.json` under `supported_models`. Pick an unused run ID (R00 through R10 are taken in the shipped config):

```json
"R11": {
    "name":            "llama3.2",
    "backend":         "ollama",
    "temperature":     0,
    "max_tokens":      1024,
    "api_key_env_var": null
}
```

Supported backends: `openai`, `anthropic`, `ollama`, `llama_cpp`, `hardcoded` (for non-LLM baselines).

For Groq (free tier, OpenAI-compatible):

```json
"R12": {
    "name":            "llama-3.1-70b-versatile",
    "backend":         "openai",
    "base_url":        "https://api.groq.com/openai/v1",
    "api_key_env_var": "GROQ_API_KEY",
    "temperature":     0,
    "max_tokens":      1024
}
```

## Design decisions

**Why synthetic data?** Real OCR documents contain PII and are hard to license. Synthetic generation lets us control difficulty precisely, guarantee ground truth, and regenerate the dataset at any time.

**Why exact match over F1?** Field F1 measures whether the model returns the right field names, which is easy for any model that reads the prompt schema. Exact match measures whether the model correctly extracted the actual value from corrupted text, which is the hard part.

**Why no ML dependencies?** Levenshtein distance, bootstrap CI, and Wilcoxon signed-rank are all implemented in pure Python. This keeps the evaluation pipeline auditable, dependency-free, and runnable anywhere. The LLM SDKs (`openai`, `anthropic`) are optional extras only used when calling those backends.

**Why Indian-context data?** The pools (names, cities, insurers, hospital chains, UPI payments) reflect the document types the noise functions were designed around. The benchmark is domain-specific by design; it tests extraction quality, not world knowledge.

**Seeded reproducibility.** Every sample is generated from `base_seed + sample_index`. Running `sample_builder.py` twice with the same seed produces identical datasets. Every benchmark run writes a frozen `run_config.json` so a result can be replayed exactly.

## Noise functions

### Tier 1: character level
`char_O0`, `char_l1I`, `char_B8`, `char_S5`, `char_sub`, `char_del`, `char_transpose`

### Tier 2: structure level
`missing_colon`, `merged_lines`, `split_line`, `key_abbrev`, `value_truncated`, `delimiter_swap`

### Tier 3: numeric and date
`num_comma_drop`, `num_decimal_shift`, `date_format_vary`, `date_partial`

### Tier 4: semantic traps (extreme only)
`extraneous_field`, `ghost_value`, `conflicting_field`, `ambiguous_key`, `value_swap`, `section_erase`

## Development

```bash
pip install -e .[dev]            # installs pytest, ruff, mypy, pillow, pytesseract
python validate_configs.py       # cross-validates the four config files
ruff check .                     # lint
mypy .                           # type check
pytest                           # 205 tests (vision-pipeline tests skip if Tesseract is absent)
```

CI runs all of the above on Python 3.10, 3.11, and 3.12 for every push to `main` and every pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution guide and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines.

## Citation

If you use LORE in your work, please cite:

```
@misc{lore2026,
  title   = {LORE: LLM OCR Robustness Evaluation},
  year    = {2026},
  url     = {https://github.com/Vespertil-Technologies/LORE-OCR-Benchmark}
}
```

## License

MIT. See [LICENSE](LICENSE).
