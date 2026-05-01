# Contributing to LORE

Thanks for your interest in improving LORE. This document covers how to set up your environment, what to test before sending a change, and how to file good issues and pull requests.

## Development setup

Requirements: Python 3.10 or newer. The core pipeline runs on the standard library, so no ML dependencies are required to develop or run tests.

```bash
git clone https://github.com/Vespertil-Technologies/LORE-OCR-Benchmark.git
cd LORE-OCR-Benchmark
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt # only needed if you plan to run paid backends
```

To exercise the full pipeline locally:

```bash
python validate_configs.py            # sanity-check the four config files
python dataset/sample_builder.py      # generate the dataset under data/
python -c "from runners.multi_run import run; run(run_id='R00', split='dev', n=10)"
python report/generator.py            # auto-picks the most recent run
```

## Before opening a pull request

- Run `python validate_configs.py` and make sure it exits clean.
- If you changed anything under `evaluator/`, `parsers/`, `stats/`, or `dataset/`, run the affected module's `__main__` block (every module has one) and confirm the printed test cases still pass.
- Keep changes focused. One PR per logical change is easier to review than a mixed bag.
- Update or add docstrings if you changed a public function's signature or behaviour.

## Coding style

- Python 3.10+. Use type hints on new public functions.
- Keep modules importable with no side effects at import time. Anything interactive belongs under `if __name__ == "__main__":`.
- Prefer the standard library. Adding a third-party dependency needs an explicit justification.
- Comments explain *why*, not *what*. If a variable name is enough, don't add a comment.

## Filing issues

Use the issue templates. For bug reports, include:

- The exact command or code that triggered the bug.
- The full traceback if there was one.
- Your Python version and OS.
- The relevant `run_config.json` if the bug surfaced during evaluation.

For feature requests, describe the use case before the proposed solution. The framework is intentionally minimal, so additions need to earn their place.

## Reproducibility

LORE is a reproducible benchmark. Anything that affects sample generation, prompt construction, or scoring must be:

- Driven by a config file under `config/`, not by code constants.
- Seeded against `base_seed` from `config/generation_config.json`.
- Snapshotted into `run_config.json` so a run can be replayed.

If your change breaks reproducibility for an existing run, that's a breaking change and must be called out in the PR description.

## License

By contributing, you agree that your contribution is licensed under the MIT License (see [LICENSE](LICENSE)).
