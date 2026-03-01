"""
runners/llm_adapter.py

Unified adapter for calling different LLM backends.
Abstracts away all API differences behind a single .call() method.

Supported backends:
    - openai    : OpenAI API (GPT-4o, etc.)
    - anthropic : Anthropic API (Claude)
    - ollama    : Local models via Ollama HTTP endpoint
    - llama_cpp : Local models via llama-cpp-python HTTP server
    - hardcoded : For baselines (always_null, regex_rules) — no API call

Responsibilities:
    - Send a prompt, receive raw string response
    - Handle rate limit retries with exponential backoff
    - Log every call: timestamp, model, prompt hash, latency, response length
    - Read temperature + max_tokens strictly from eval_config.json
    - Never expose API keys — always read from environment variables

Does NOT:
    - Parse model output
    - Know about evaluation metrics
    - Format prompts (that is prompt_formatter.py's job)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_eval_config() -> dict:
    with open(_CONFIG_DIR / "eval_config.json", encoding="utf-8") as f:
        return json.load(f)

EVAL_CONFIG = _load_eval_config()


# ══════════════════════════════════════════════════════════════════════════════
# CALL RECORD — what gets logged for every API call
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CallRecord:
    sample_id:       str
    model:           str
    backend:         str
    prompt_hash:     str
    timestamp_utc:   str
    latency_ms:      int
    response_length: int
    success:         bool
    error:           str | None = None

    def to_dict(self) -> dict:
        return {
            "sample_id":       self.sample_id,
            "model":           self.model,
            "backend":         self.backend,
            "prompt_hash":     self.prompt_hash,
            "timestamp_utc":   self.timestamp_utc,
            "latency_ms":      self.latency_ms,
            "response_length": self.response_length,
            "success":         self.success,
            "error":           self.error,
        }


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _call_openai(prompt: str, model_cfg: dict) -> str:
    """Call OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.environ.get(model_cfg["api_key_env_var"])
    if not api_key:
        raise EnvironmentError(
            f"API key not found in environment variable '{model_cfg['api_key_env_var']}'"
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_cfg["name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )
    return response.choices[0].message.content or ""


def _call_anthropic(prompt: str, model_cfg: dict) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get(model_cfg["api_key_env_var"])
    if not api_key:
        raise EnvironmentError(
            f"API key not found in environment variable '{model_cfg['api_key_env_var']}'"
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model_cfg["name"],
        max_tokens=model_cfg["max_tokens"],
        temperature=model_cfg["temperature"],
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text if message.content else ""


def _call_ollama(prompt: str, model_cfg: dict) -> str:
    """Call a local model via Ollama HTTP endpoint."""
    try:
        import urllib.request
    except ImportError:
        raise ImportError("urllib is part of the standard library — this should not happen.")

    base_url = model_cfg.get("base_url", "http://localhost:11434")
    payload = json.dumps({
        "model":  model_cfg["name"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": model_cfg["temperature"],
            "num_predict": model_cfg["max_tokens"],
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def _call_llama_cpp(prompt: str, model_cfg: dict) -> str:
    """Call a local model via llama-cpp-python HTTP server."""
    try:
        import urllib.request
    except ImportError:
        raise ImportError("urllib is part of the standard library.")

    base_url = model_cfg.get("base_url", "http://localhost:8080")
    payload = json.dumps({
        "prompt":      prompt,
        "temperature": model_cfg["temperature"],
        "n_predict":   model_cfg["max_tokens"],
        "stop":        ["\n\n\n"],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("content", "")


# Dispatch table
_BACKEND_DISPATCH = {
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
    "ollama":    _call_ollama,
    "llama_cpp": _call_llama_cpp,
}


# ══════════════════════════════════════════════════════════════════════════════
# RETRY LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def _call_with_retry(
    fn,
    prompt: str,
    model_cfg: dict,
    max_retries: int,
    backoff_base: float,
) -> str:
    """
    Call fn(prompt, model_cfg) with exponential backoff on failure.
    Retries on rate limit errors and transient network errors.
    Raises on the final attempt if still failing.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn(prompt, model_cfg)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Detect rate limit errors across different clients
            is_rate_limit = any(
                kw in error_str
                for kw in ["rate limit", "429", "too many requests", "quota"]
            )
            is_transient = any(
                kw in error_str
                for kw in ["timeout", "connection", "503", "502", "500"]
            )

            if attempt < max_retries and (is_rate_limit or is_transient):
                wait = backoff_base ** (attempt + 1)
                log.warning(f"Attempt {attempt + 1} failed ({type(e).__name__}). Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                break

    raise last_error


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def call(
    prompt: str,
    prompt_hash: str,
    sample_id: str,
    model_cfg: dict | None = None,
) -> tuple[str, CallRecord]:
    """
    Send a prompt to the configured LLM and return the raw response.

    Args:
        prompt:       The fully formatted prompt string.
        prompt_hash:  8-char hash of the prompt (from prompt_formatter.py).
        sample_id:    Sample ID for logging.
        model_cfg:    Model configuration dict. If None, reads from eval_config.json.

    Returns:
        (response_text, call_record)
        - response_text: Raw string from the model (unparsed).
        - call_record:   Structured log entry for this call.

    Raises:
        ValueError:      If the backend is not supported.
        EnvironmentError: If the API key env var is not set.
        Exception:       If all retries are exhausted.
    """
    if model_cfg is None:
        model_cfg = EVAL_CONFIG["model"]

    backend     = model_cfg.get("backend", "openai")
    retry_cfg   = EVAL_CONFIG["retry_policy"]
    max_retries = retry_cfg["max_retries"]
    backoff     = retry_cfg["backoff_base_seconds"]

    if backend == "hardcoded":
        raise ValueError(
            "Backend 'hardcoded' is for baseline models only. "
            "Use baseline_runner.py for always_null and regex_rules."
        )

    if backend not in _BACKEND_DISPATCH:
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Must be one of: {list(_BACKEND_DISPATCH.keys())}"
        )

    backend_fn = _BACKEND_DISPATCH[backend]
    timestamp  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start      = time.perf_counter()
    error_msg  = None
    response   = ""

    try:
        response = _call_with_retry(backend_fn, prompt, model_cfg, max_retries, backoff)
        success  = True
        log.info(f"[{sample_id}] {model_cfg['name']} → {len(response)} chars  ({int((time.perf_counter()-start)*1000)}ms)")
    except Exception as e:
        success   = False
        error_msg = f"{type(e).__name__}: {e}"
        log.error(f"[{sample_id}] FAILED after {max_retries} retries — {error_msg}")

    latency_ms = int((time.perf_counter() - start) * 1000)

    record = CallRecord(
        sample_id       = sample_id,
        model           = model_cfg["name"],
        backend         = backend,
        prompt_hash     = prompt_hash,
        timestamp_utc   = timestamp,
        latency_ms      = latency_ms,
        response_length = len(response),
        success         = success,
        error           = error_msg,
    )

    return response, record


def build_model_cfg(run_id: str) -> dict:
    """
    Build a model config dict for a specific run ID from eval_config.json.
    Merges run-specific settings with the base model config.

    Args:
        run_id: e.g. 'R03', 'R04', 'A01' (from Phase 4 experiment matrix)

    Returns:
        model_cfg dict ready to pass to call().
    """
    supported = EVAL_CONFIG["supported_models"]
    if run_id not in supported:
        raise ValueError(f"Unknown run_id '{run_id}'. Must be one of: {list(supported.keys())}")

    base = dict(EVAL_CONFIG["model"])  # copy base defaults
    base.update(supported[run_id])     # overlay run-specific values
    return base


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Dry-run demo — does NOT make real API calls.
    Shows how call() would be invoked and what a CallRecord looks like.
    Replace the mock with a real call() invocation once API keys are set.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset.loader import load_samples
    from runners.prompt_formatter import build_prompt

    DATA_DIR = Path(__file__).parent.parent / "data"

    # Load one sample
    samples = load_samples(DATA_DIR, domain="insurance", difficulty="hard", split="test")
    sample  = samples[0]

    # Build prompt
    prompt, prompt_hash = build_prompt(sample)

    print("=" * 60)
    print("DRY RUN — llm_adapter.call() demo")
    print("=" * 60)
    print(f"Sample ID   : {sample['id']}")
    print(f"Prompt hash : {prompt_hash}")
    print(f"Prompt len  : {len(prompt)} chars")
    print()

    # Show what a real call looks like (without hitting the API)
    print("To make a real call, set your API key and run:")
    print()
    print("  import os")
    print("  os.environ['OPENAI_API_KEY'] = 'sk-...'")
    print()
    print("  from runners.llm_adapter import call")
    print("  response, record = call(prompt, prompt_hash, sample['id'])")
    print("  print(response)")
    print("  print(record.to_dict())")
    print()

    # Show a mock CallRecord to illustrate the structure
    mock_record = CallRecord(
        sample_id       = sample["id"],
        model           = "gpt-4o-2024-08-06",
        backend         = "openai",
        prompt_hash     = prompt_hash,
        timestamp_utc   = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        latency_ms      = 1842,
        response_length = 312,
        success         = True,
        error           = None,
    )

    print("Example CallRecord structure:")
    print(json.dumps(mock_record.to_dict(), indent=2))

    print()
    print("=" * 60)
    print("Supported run IDs (from eval_config.json):")
    print("=" * 60)
    for run_id, run_val in EVAL_CONFIG["supported_models"].items():
        if run_id.startswith("_"):
            continue
        cfg = build_model_cfg(run_id)
        print(f"  {run_id}  →  {cfg.get('name'):<40}  backend: {cfg.get('backend')}")