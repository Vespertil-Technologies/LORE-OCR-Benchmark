"""
dataset/noise_generator.py

The synthetic noise injection engine. Takes clean serialized text from
serializer.py and corrupts it to simulate real OCR errors.

Architecture (4 layers):
    1. Noise Selector  - picks which noise functions to apply based on difficulty
    2. Noise Applicator - applies Tier 1-2 sequentially, Tier 3-4 independently
    3. Noise Functions  - one function per noise tag, all self-contained
    4. Validator        - ensures the result meets difficulty requirements

Noise tiers:
    Tier 1: Character-level  (sequential)
    Tier 2: Structure-level  (sequential)
    Tier 3: Numeric/Date     (independent - applied to original text, merged in)
    Tier 4: Hallucination    (independent - appended/injected at line level)
"""

import json
import random
import re
from collections.abc import Callable
from pathlib import Path

# ── Config loading ─────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_json(filename: str) -> dict:
    with open(_CONFIG_DIR / filename, encoding="utf-8") as f:
        return json.load(f)

NOISE_WEIGHTS = _load_json("noise_weights.json")
DOMAINS       = _load_json("domains.json")
GEN_CONFIG    = _load_json("generation_config.json")


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 - Character-level noise functions
# Each takes (text, rng, rate_cfg) and returns (corrupted_text, tag)
# Applied SEQUENTIALLY - output of one feeds the next
# ══════════════════════════════════════════════════════════════════════════════

def _apply_char_O0(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Replace O↔0 in value portions of lines."""
    rate = rng.uniform(rate_cfg["min"], rate_cfg["max"])
    result = []
    for char in text:
        if char == "O" and rng.random() < rate:
            result.append("0")
        elif char == "0" and rng.random() < rate:
            result.append("O")
        else:
            result.append(char)
    return "".join(result), "char_O0"


def _apply_char_l1I(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Confuse l / 1 / I characters."""
    rate = rng.uniform(rate_cfg["min"], rate_cfg["max"])
    confusion_map = {"l": ["1", "I"], "1": ["l", "I"], "I": ["l", "1"]}
    result = []
    for char in text:
        if char in confusion_map and rng.random() < rate:
            result.append(rng.choice(confusion_map[char]))
        else:
            result.append(char)
    return "".join(result), "char_l1I"


def _apply_char_B8(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Confuse B↔8 - common in alphanumeric IDs and names."""
    rate = rng.uniform(rate_cfg["min"], rate_cfg["max"])
    result = []
    for char in text:
        if char == "B" and rng.random() < rate:
            result.append("8")
        elif char == "8" and rng.random() < rate:
            result.append("B")
        else:
            result.append(char)
    return "".join(result), "char_B8"


def _apply_char_S5(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Confuse S↔5."""
    rate = rng.uniform(rate_cfg["min"], rate_cfg["max"])
    result = []
    for char in text:
        if char == "S" and rng.random() < rate:
            result.append("5")
        elif char == "5" and rng.random() < rate:
            result.append("S")
        else:
            result.append(char)
    return "".join(result), "char_S5"


def _apply_char_sub(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Randomly substitute 1-2 alphabetic characters per line with a nearby key."""
    # Nearby keys on a QWERTY layout - simulates OCR misread of similar shapes
    nearby = {
        "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr",
        "f": "dg", "g": "fh", "h": "gj", "i": "uo", "j": "hk",
        "k": "jl", "l": "k",  "m": "n",  "n": "mb", "o": "ip",
        "p": "o",  "q": "wa", "r": "et", "s": "ad", "t": "ry",
        "u": "yi", "v": "cb", "w": "qe", "x": "cz", "y": "tu",
        "z": "x"
    }
    lines = text.split("\n")
    count = rng.randint(rate_cfg["chars_per_line"][0], rate_cfg["chars_per_line"][1])
    result_lines = []
    for line in lines:
        chars = list(line)
        # Only touch value side - skip the key label (before first colon/separator)
        sep_idx = _find_separator_idx(line)
        eligible = [
            i for i, c in enumerate(chars)
            if i > sep_idx and c.isalpha() and c.lower() in nearby
        ]
        for _ in range(min(count, len(eligible))):
            idx = rng.choice(eligible)
            c = chars[idx]
            replacement = rng.choice(nearby[c.lower()])
            chars[idx] = replacement.upper() if c.isupper() else replacement
            eligible.remove(idx)
        result_lines.append("".join(chars))
    return "\n".join(result_lines), "char_sub"


def _apply_char_del(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Delete one character per affected line (not first/last of a word)."""
    lines = text.split("\n")
    result_lines = []
    for line in lines:
        sep_idx = _find_separator_idx(line)
        # Find eligible positions: alphabetic, not at word boundaries, in value portion
        eligible = [
            i for i in range(sep_idx + 1, len(line) - 1)
            if line[i].isalpha() and line[i-1] != " " and line[i+1] != " "
        ]
        if eligible:
            idx = rng.choice(eligible)
            line = line[:idx] + line[idx+1:]
        result_lines.append(line)
    return "\n".join(result_lines), "char_del"


def _apply_char_transpose(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Swap two adjacent characters inside a word, in the value portion."""
    lines = text.split("\n")
    result_lines = []
    for line in lines:
        sep_idx = _find_separator_idx(line)
        chars = list(line)
        eligible = [
            i for i in range(sep_idx + 1, len(chars) - 1)
            if chars[i].isalpha() and chars[i+1].isalpha()
        ]
        if eligible:
            idx = rng.choice(eligible)
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        result_lines.append("".join(chars))
    return "\n".join(result_lines), "char_transpose"


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 - Structure-level noise functions
# Applied SEQUENTIALLY after Tier 1
# ══════════════════════════════════════════════════════════════════════════════

def _apply_missing_colon(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Remove the colon (or separator) between key and value on some lines."""
    rate = rng.uniform(rate_cfg["min"], rate_cfg["max"])
    lines = text.split("\n")
    result_lines = []
    for line in lines:
        if line and rng.random() < rate:
            # Remove colon/separator - leave a space between key and value
            line = re.sub(r"\s*[:]\s*", " ", line, count=1)
        result_lines.append(line)
    return "\n".join(result_lines), "missing_colon"


def _apply_merged_lines(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Join two consecutive non-empty lines into one."""
    n_merges = rng.randint(
        rate_cfg["merges_per_sample"][0],
        rate_cfg["merges_per_sample"][1]
    )
    lines = text.split("\n")
    for _ in range(n_merges):
        # Find pairs of consecutive non-empty lines
        eligible = [i for i in range(len(lines) - 1) if lines[i].strip() and lines[i+1].strip()]
        if not eligible:
            break
        idx = rng.choice(eligible)
        lines[idx] = lines[idx] + " " + lines[idx+1]
        lines.pop(idx+1)
    return "\n".join(lines), "merged_lines"


def _apply_split_line(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Break one line into two at a random word boundary in the value portion."""
    lines = text.split("\n")
    # Find lines with a value containing at least 2 words after the separator
    eligible = []
    for i, line in enumerate(lines):
        sep_idx = _find_separator_idx(line)
        value_part = line[sep_idx+1:].strip() if sep_idx >= 0 else ""
        if len(value_part.split()) >= 2:
            eligible.append(i)
    if eligible:
        idx = rng.choice(eligible)
        line = lines[idx]
        sep_idx = _find_separator_idx(line)
        key_part = line[:sep_idx+1]
        value_part = line[sep_idx+1:].strip()
        words = value_part.split()
        split_point = rng.randint(1, len(words) - 1)
        lines[idx] = key_part
        lines.insert(idx + 1, " ".join(words[split_point:]))
        lines[idx] = key_part + " " + " ".join(words[:split_point])
    return "\n".join(lines), "split_line"


def _apply_key_abbrev(text: str, rng: random.Random, rate_cfg: dict, domain: str) -> tuple[str, str]:
    """Replace key labels with known abbreviations from the domain's key_label_variants."""
    rate = rng.uniform(rate_cfg["min"], rate_cfg["max"])
    # Build a flat map: full_label -> list of short variants
    abbrev_map: dict[str, list[str]] = {}
    for _field, variants in DOMAINS[domain]["key_label_variants"].items():
        if len(variants) > 1:
            # First variant is usually the "full" form - rest are abbreviations
            full = variants[0]
            abbrevs = [v for v in variants[1:] if len(v) <= len(full)]
            if abbrevs:
                abbrev_map[full.lower()] = abbrevs

    lines = text.split("\n")
    result_lines = []
    for line in lines:
        if line and rng.random() < rate:
            sep_idx = _find_separator_idx(line)
            key_part = line[:sep_idx].strip() if sep_idx >= 0 else line
            value_part = line[sep_idx:] if sep_idx >= 0 else ""
            replacement = abbrev_map.get(key_part.lower())
            if replacement:
                key_part = rng.choice(replacement)
                line = key_part + value_part
        result_lines.append(line)
    return "\n".join(result_lines), "key_abbrev"


def _apply_value_truncated(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Cut a string value off near the end - simulates line-end OCR truncation."""
    lines = text.split("\n")
    eligible = []
    for i, line in enumerate(lines):
        sep_idx = _find_separator_idx(line)
        value = line[sep_idx+1:].strip() if sep_idx >= 0 else line
        if len(value) > 5:
            eligible.append(i)
    if eligible:
        idx = rng.choice(eligible)
        line = lines[idx]
        sep_idx = _find_separator_idx(line)
        key_part = line[:sep_idx+1] if sep_idx >= 0 else ""
        value_part = line[sep_idx+1:].strip() if sep_idx >= 0 else line
        # Cut off between 20-50% from the end
        cut = rng.randint(int(len(value_part) * 0.5), int(len(value_part) * 0.8))
        lines[idx] = key_part + " " + value_part[:cut]
    return "\n".join(lines), "value_truncated"


def _apply_delimiter_swap(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Replace colon separator with a dash, slash, or space on some lines."""
    n = rng.randint(rate_cfg["lines_per_sample"][0], rate_cfg["lines_per_sample"][1])
    delimiters = NOISE_WEIGHTS["delimiter_swap_chars"]
    lines = text.split("\n")
    eligible = [i for i, ln in enumerate(lines) if ":" in ln and ln.strip()]
    rng.shuffle(eligible)
    for idx in eligible[:n]:
        new_delim = rng.choice(delimiters)
        lines[idx] = re.sub(r"\s*:\s*", new_delim, lines[idx], count=1)
    return "\n".join(lines), "delimiter_swap"


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 - Numeric / Date noise functions
# Applied INDEPENDENTLY on the original text, results merged into Tier 1-2 output
# ══════════════════════════════════════════════════════════════════════════════

def _apply_num_comma_drop(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Remove thousand-separator commas from numeric values."""
    # Match numbers with commas like 5,000 or 1,50,000
    result = re.sub(r"(\d{1,3}(?:,\d{2,3})+)", lambda m: m.group().replace(",", ""), text)
    return result, "num_comma_drop"


def _apply_num_decimal_shift(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Move the decimal point one position left or right in a numeric value."""
    def shift_decimal(m: re.Match) -> str:
        num_str = m.group()
        try:
            val = float(num_str)
            direction = rng.choice([-1, 1])
            shifted = val * (10 ** direction)
            # Preserve decimal representation
            if "." in num_str:
                decimals = len(num_str.split(".")[1])
                return f"{shifted:.{decimals}f}"
            return str(int(shifted))
        except ValueError:
            return num_str

    result = re.sub(r"\b\d+\.\d+\b", shift_decimal, text)
    return result, "num_decimal_shift"


def _apply_date_format_vary(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """
    Reformat dates already in the text to a different non-ISO variant.
    Detects common date patterns and reformats them.
    """
    from datetime import datetime

    # Patterns to detect dates already serialized by serializer.py
    date_patterns = [
        (r"\b(\d{2})/(\d{2})/(\d{4})\b", "%d/%m/%Y"),
        (r"\b(\d{2})-(\d{2})-(\d{4})\b", "%d-%m-%Y"),
        (r"\b(\d{2}) ([A-Za-z]{3}) (\d{4})\b", "%d %b %Y"),
        (r"\b(\d{2})/(\d{2})/(\d{2})\b", "%d/%m/%y"),
    ]

    target_formats = [f for f in NOISE_WEIGHTS["date_format_variants"]]

    def reformat(m: re.Match, src_fmt: str) -> str:
        try:
            dt = datetime.strptime(m.group(), src_fmt)
            fmt_str = rng.choice(target_formats)
            # Convert format string token to strftime
            fmt_map = {
                "DD": "%d", "MM": "%m", "YYYY": "%Y", "YY": "%y", "Mon": "%b"
            }
            strftime_fmt = fmt_str
            for token, code in sorted(fmt_map.items(), key=lambda x: -len(x[0])):
                strftime_fmt = strftime_fmt.replace(token, code)
            return dt.strftime(strftime_fmt)
        except ValueError:
            return m.group()

    result = text
    for pattern, src_fmt in date_patterns:
        result = re.sub(pattern, lambda m, f=src_fmt: reformat(m, f), result)
    return result, "date_format_vary"


def _apply_date_partial(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Corrupt one character inside a date using a char confusion (O/0, B/8)."""
    # Find a date-like pattern
    date_pattern = re.compile(r"\b(\d{2}[/\-]\d{2}[/\-]\d{2,4})\b")
    matches = list(date_pattern.finditer(text))
    if not matches:
        return text, "date_partial"

    match = rng.choice(matches)
    date_str = list(match.group())
    # Corrupt one digit
    digit_indices = [i for i, c in enumerate(date_str) if c.isdigit()]
    if digit_indices:
        idx = rng.choice(digit_indices)
        confusions = {"0": "O", "8": "B", "1": "l", "5": "S"}
        if date_str[idx] in confusions:
            date_str[idx] = confusions[date_str[idx]]
    corrupted = "".join(date_str)
    result = text[:match.start()] + corrupted + text[match.end():]
    return result, "date_partial"


# ══════════════════════════════════════════════════════════════════════════════
# TIER 4 - Hallucination trap functions (extreme difficulty only)
# Applied INDEPENDENTLY - operate at line level, append/inject into text
# ══════════════════════════════════════════════════════════════════════════════

def _apply_extraneous_field(text: str, rng: random.Random, rate_cfg: dict, domain: str) -> tuple[str, str]:
    """Append 1-2 realistic-looking fields that are NOT in the schema."""
    pool = NOISE_WEIGHTS["extraneous_field_pool"][domain]
    n = rng.randint(rate_cfg["lines_per_sample"][0], rate_cfg["lines_per_sample"][1])
    chosen = rng.sample(pool, min(n, len(pool)))
    result = text + "\n" + "\n".join(chosen)
    return result, "extraneous_field"


def _apply_ghost_value(
    text: str,
    rng: random.Random,
    rate_cfg: dict,
    domain: str,
    gt_struct: dict
) -> tuple[str, str]:
    """
    Replace a real field's value in the OCR text with a plausible but wrong value.
    The gt_struct remains correct - the model must not copy the ghost value.
    """
    lines = text.split("\n")
    # Find lines that have a separator (key: value structure)
    eligible = [i for i, ln in enumerate(lines) if _find_separator_idx(ln) >= 0 and ln.strip()]
    if not eligible:
        return text, "ghost_value"

    idx = rng.choice(eligible)
    line = lines[idx]
    sep_idx = _find_separator_idx(line)
    key_part = line[:sep_idx+1]

    # Generate a plausible ghost value - scramble the real value
    value_part = line[sep_idx+1:].strip()
    ghost = _scramble_value(value_part, rng)
    lines[idx] = key_part + " " + ghost
    return "\n".join(lines), "ghost_value"


def _apply_conflicting_field(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Duplicate one key-value line near the end with a different value."""
    lines = text.split("\n")
    eligible = [i for i, ln in enumerate(lines) if _find_separator_idx(ln) >= 0 and ln.strip()]
    if not eligible:
        return text, "conflicting_field"

    idx = rng.choice(eligible)
    line = lines[idx]
    sep_idx = _find_separator_idx(line)
    key_part = line[:sep_idx+1]
    value_part = line[sep_idx+1:].strip()
    conflict_value = _scramble_value(value_part, rng)
    conflict_line = key_part + " " + conflict_value
    # Insert the conflicting line somewhere after the original
    insert_pos = rng.randint(idx + 1, len(lines))
    lines.insert(insert_pos, conflict_line)
    return "\n".join(lines), "conflicting_field"


def _apply_ambiguous_key(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """Replace a key label with a vague label that could map to multiple fields."""
    ambiguous_labels = ["Ref No", "Date", "Number", "ID", "Code", "No", "Details"]
    lines = text.split("\n")
    eligible = [i for i, ln in enumerate(lines) if _find_separator_idx(ln) >= 0 and ln.strip()]
    if not eligible:
        return text, "ambiguous_key"

    idx = rng.choice(eligible)
    line = lines[idx]
    sep_idx = _find_separator_idx(line)
    value_part = line[sep_idx:]
    new_key = rng.choice(ambiguous_labels)
    lines[idx] = new_key + value_part
    return "\n".join(lines), "ambiguous_key"



def _apply_value_swap(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """
    Swap the values of two different key-value lines.
    E.g. patient name gets the date value and vice versa.
    Forces the model to correctly identify which value belongs to which key,
    rather than relying on position or label heuristics.
    """
    lines = text.split("\n")
    eligible = [i for i, ln in enumerate(lines) if _find_separator_idx(ln) >= 0 and ln.strip()]
    if len(eligible) < 2:
        return text, "value_swap"

    i, j = rng.sample(eligible, 2)
    sep_i = _find_separator_idx(lines[i])
    sep_j = _find_separator_idx(lines[j])
    key_i = lines[i][:sep_i+1]
    key_j = lines[j][:sep_j+1]
    val_i = lines[i][sep_i+1:]
    val_j = lines[j][sep_j+1:]
    lines[i] = key_i + val_j
    lines[j] = key_j + val_i
    return "\n".join(lines), "value_swap"


def _apply_section_erase(text: str, rng: random.Random, rate_cfg: dict) -> tuple[str, str]:
    """
    Wipe out all values in one contiguous block of lines (simulates ink dropout
    or scanner shadow covering part of a document).
    The keys remain visible but all values become garbled or empty.
    """
    lines = text.split("\n")
    eligible = [i for i, ln in enumerate(lines) if _find_separator_idx(ln) >= 0 and ln.strip()]
    if len(eligible) < 2:
        return text, "section_erase"

    # Pick a random contiguous window of 2-4 lines to erase
    window = rng.randint(2, min(4, len(eligible)))
    start_pos = rng.randint(0, len(eligible) - window)
    erase_indices = set(eligible[start_pos : start_pos + window])

    corruptions = ["", "???", "----", "XXXXX", "......"]
    for idx in erase_indices:
        sep_idx = _find_separator_idx(lines[idx])
        if sep_idx >= 0:
            key_part = lines[idx][:sep_idx+1]
            lines[idx] = key_part + " " + rng.choice(corruptions)

    return "\n".join(lines), "section_erase"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _find_separator_idx(line: str) -> int:
    """Return the index of the first colon in a line, or -1 if none found."""
    return line.find(":")


def _scramble_value(value: str, rng: random.Random) -> str:
    """
    Produce a plausible but wrong version of a value.
    - Numbers: add/subtract a random offset
    - Dates: change one digit
    - Strings: shuffle some characters
    """
    # Numeric
    num_match = re.match(r"^[\d,\.]+$", value.replace(" ", ""))
    if num_match:
        try:
            n = float(value.replace(",", ""))
            offset = rng.choice([-1000, -500, 500, 1000, -100, 100])
            return str(int(n + offset))
        except ValueError:
            pass

    # Date-like
    if re.match(r"\d{2}[/\-]\d{2}[/\-]\d{2,4}", value):
        chars = list(value)
        digit_idxs = [i for i, c in enumerate(chars) if c.isdigit()]
        if digit_idxs:
            idx = rng.choice(digit_idxs)
            chars[idx] = str((int(chars[idx]) + rng.randint(1, 5)) % 10)
        return "".join(chars)

    # String - shuffle middle characters
    if len(value) > 4:
        mid = list(value[1:-1])
        rng.shuffle(mid)
        return value[0] + "".join(mid) + value[-1]

    return value + "X"  # fallback


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 - Noise Selector
# ══════════════════════════════════════════════════════════════════════════════

def _select_noise_functions(
    difficulty: str,
    domain: str,
    rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    """
    Select which noise functions to apply for this sample.
    Returns (tier_1_2_tags, tier_3_tags, tier_4_tags).
    """
    ranges = NOISE_WEIGHTS["difficulty_ranges"][difficulty]

    def sample_tier(tier_key: str, count_key: str) -> list[str]:
        weights_dict = {
            k: v for k, v in NOISE_WEIGHTS[tier_key].items()
            if not k.startswith("_")
        }
        # Filter by domain applicability
        applicability = NOISE_WEIGHTS["domain_applicability"]
        eligible = {
            tag: w for tag, w in weights_dict.items()
            if applicability.get(tag) is None or domain in applicability[tag]
        }
        if not eligible:
            return []
        count_range = ranges[count_key]
        n = rng.randint(count_range[0], count_range[1])
        tags = list(eligible.keys())
        weights = list(eligible.values())
        # Sample without replacement (up to available tags)
        selected = []
        remaining_tags = tags[:]
        remaining_weights = weights[:]
        for _ in range(min(n, len(remaining_tags))):
            chosen = rng.choices(remaining_tags, weights=remaining_weights, k=1)[0]
            selected.append(chosen)
            idx = remaining_tags.index(chosen)
            remaining_tags.pop(idx)
            remaining_weights.pop(idx)
        return selected

    tier_1_weights = {k: v for k, v in NOISE_WEIGHTS["tier_1_weights"].items() if not k.startswith("_")}
    tier_2_weights = {k: v for k, v in NOISE_WEIGHTS["tier_2_weights"].items() if not k.startswith("_")}
    combined_1_2 = {**tier_1_weights, **tier_2_weights}

    # Tier 1+2 combined selection
    applicability = NOISE_WEIGHTS["domain_applicability"]
    eligible_1_2 = {
        tag: w for tag, w in combined_1_2.items()
        if applicability.get(tag) is None or domain in applicability[tag]
    }
    count_range = ranges["tier_1_2_count"]
    n = rng.randint(count_range[0], count_range[1])
    tags_1_2 = list(eligible_1_2.keys())
    weights_1_2 = list(eligible_1_2.values())
    selected_1_2 = []
    remaining_tags = tags_1_2[:]
    remaining_weights = weights_1_2[:]
    for _ in range(min(n, len(remaining_tags))):
        chosen = rng.choices(remaining_tags, weights=remaining_weights, k=1)[0]
        selected_1_2.append(chosen)
        idx = remaining_tags.index(chosen)
        remaining_tags.pop(idx)
        remaining_weights.pop(idx)

    # Tier 3 selection
    eligible_3 = {
        tag: w for tag, w in NOISE_WEIGHTS["tier_3_weights"].items()
        if not tag.startswith("_") and (
            applicability.get(tag) is None or domain in applicability[tag]
        )
    }
    count_range_3 = ranges["tier_3_count"]
    n3 = rng.randint(count_range_3[0], count_range_3[1])
    selected_3 = rng.choices(list(eligible_3.keys()), weights=list(eligible_3.values()), k=n3) if eligible_3 and n3 > 0 else []

    # Tier 4 selection (extreme only)
    eligible_4 = {
        tag: w for tag, w in NOISE_WEIGHTS["tier_4_weights"].items()
        if not tag.startswith("_") and (
            applicability.get(tag) is None or domain in applicability[tag]
        )
    }
    count_range_4 = ranges["tier_4_count"]
    n4 = rng.randint(count_range_4[0], count_range_4[1])
    selected_4 = rng.choices(list(eligible_4.keys()), weights=list(eligible_4.values()), k=n4) if eligible_4 and n4 > 0 else []

    return selected_1_2, selected_3, selected_4


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 - Noise Applicator
# ══════════════════════════════════════════════════════════════════════════════

# Dispatch table mapping tag names to functions
_TIER_1_2_FUNCTIONS: dict[str, Callable] = {
    "char_O0":         _apply_char_O0,
    "char_l1I":        _apply_char_l1I,
    "char_B8":         _apply_char_B8,
    "char_S5":         _apply_char_S5,
    "char_sub":        _apply_char_sub,
    "char_del":        _apply_char_del,
    "char_transpose":  _apply_char_transpose,
    "missing_colon":   _apply_missing_colon,
    "merged_lines":    _apply_merged_lines,
    "split_line":      _apply_split_line,
    "value_truncated": _apply_value_truncated,
    "delimiter_swap":  _apply_delimiter_swap,
}

_TIER_3_FUNCTIONS: dict[str, Callable] = {
    "num_comma_drop":    _apply_num_comma_drop,
    "num_decimal_shift": _apply_num_decimal_shift,
    "date_format_vary":  _apply_date_format_vary,
    "date_partial":      _apply_date_partial,
}

_TIER_4_FUNCTIONS: dict[str, Callable] = {
    "extraneous_field":  _apply_extraneous_field,
    "ghost_value":       _apply_ghost_value,
    "conflicting_field": _apply_conflicting_field,
    "ambiguous_key":     _apply_ambiguous_key,
    "value_swap":        _apply_value_swap,
    "section_erase":     _apply_section_erase,
}


def _apply_tier_1_2(
    text: str,
    tags: list[str],
    domain: str,
    rng: random.Random
) -> tuple[str, list[str]]:
    """Apply Tier 1-2 functions sequentially. Output of each feeds the next."""
    applied_tags = []
    current = text
    for tag in tags:
        fn = _TIER_1_2_FUNCTIONS.get(tag)
        if fn is None:
            continue
        rate_cfg = NOISE_WEIGHTS["noise_application_rates"][tag]
        if tag == "key_abbrev":
            current, applied = fn(current, rng, rate_cfg, domain)
        else:
            current, applied = fn(current, rng, rate_cfg)
        applied_tags.append(applied)
    return current, applied_tags


def _apply_tier_3(
    original_text: str,
    corrupted_text: str,
    tags: list[str],
    rng: random.Random
) -> tuple[str, list[str]]:
    """
    Apply Tier 3 functions independently on the original text.
    Merge changed tokens back into the already-corrupted text.
    In practice: apply on corrupted_text since token positions may have shifted.
    """
    applied_tags = []
    current = corrupted_text
    for tag in tags:
        fn = _TIER_3_FUNCTIONS.get(tag)
        if fn is None:
            continue
        rate_cfg = NOISE_WEIGHTS["noise_application_rates"][tag]
        current, applied = fn(current, rng, rate_cfg)
        applied_tags.append(applied)
    return current, applied_tags


def _apply_tier_4(
    text: str,
    tags: list[str],
    domain: str,
    rng: random.Random,
    gt_struct: dict
) -> tuple[str, list[str]]:
    """Apply Tier 4 hallucination traps independently."""
    applied_tags = []
    current = text
    for tag in tags:
        fn = _TIER_4_FUNCTIONS.get(tag)
        if fn is None:
            continue
        rate_cfg = NOISE_WEIGHTS["noise_application_rates"][tag]
        if tag == "extraneous_field":
            current, applied = fn(current, rng, rate_cfg, domain)
        elif tag == "ghost_value":
            current, applied = fn(current, rng, rate_cfg, domain, gt_struct)
        else:
            current, applied = fn(current, rng, rate_cfg)
        applied_tags.append(applied)
    return current, applied_tags


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 - Validator
# ══════════════════════════════════════════════════════════════════════════════

def _validate(
    original_text: str,
    corrupted_text: str,
    difficulty: str,
    domain: str,
    gt_struct: dict
) -> bool:
    """
    Check that the generated sample meets difficulty requirements.
    Returns True if valid, False if the engine should retry.
    """
    # Must have actually changed something
    if original_text == corrupted_text:
        if difficulty != "easy":
            return False

    val_settings = GEN_CONFIG["validator_settings"]

    if difficulty in ("easy", "medium") and val_settings["easy_medium_required_fields_must_survive"]:
        # Required fields must still be roughly detectable
        required_fields = DOMAINS[domain]["required_fields"]
        key_variants = DOMAINS[domain]["key_label_variants"]
        for field_path in required_fields:
            variants = key_variants.get(field_path, [field_path.split(".")[-1]])
            found = any(v.lower() in corrupted_text.lower() for v in variants)
            if not found:
                return False  # A required field label was completely destroyed

    if difficulty == "extreme":
        # At least one required field must be degraded or missing
        min_degraded = val_settings["extreme_min_required_fields_degraded"]
        required_fields = DOMAINS[domain]["required_fields"]
        key_variants = DOMAINS[domain]["key_label_variants"]
        degraded_count = 0
        for field_path in required_fields:
            variants = key_variants.get(field_path, [field_path.split(".")[-1]])
            found = any(v.lower() in corrupted_text.lower() for v in variants)
            if not found:
                degraded_count += 1
        if degraded_count < min_degraded:
            return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_noise(
    raw_text: str,
    difficulty: str,
    domain: str,
    rng: random.Random,
    gt_struct: dict
) -> tuple[str, list[str]]:
    """
    Main entry point. Apply noise to raw serialized text.

    Args:
        raw_text:   Clean text from serializer.py
        difficulty: 'easy', 'medium', 'hard', or 'extreme'
        domain:     'receipts', 'insurance', or 'hospital'
        rng:        Seeded random.Random for reproducibility
        gt_struct:  Original clean dict (needed for Tier 4 ghost_value)

    Returns:
        (corrupted_text, noise_tags) - the noisy OCR string and list of applied tags
    """
    max_retries = GEN_CONFIG["validator_settings"]["max_retries"]

    for _attempt in range(max_retries):
        # Use a fresh sub-rng per attempt so retries don't pollute the main sequence
        attempt_rng = random.Random(rng.random())

        # Select noise functions for this difficulty + domain
        tags_1_2, tags_3, tags_4 = _select_noise_functions(difficulty, domain, attempt_rng)

        # Apply Tier 1-2 sequentially
        text, applied_1_2 = _apply_tier_1_2(raw_text, tags_1_2, domain, attempt_rng)

        # Apply Tier 3 independently
        text, applied_3 = _apply_tier_3(raw_text, text, tags_3, attempt_rng)

        # Apply Tier 4 independently (extreme only - selector returns [] otherwise)
        text, applied_4 = _apply_tier_4(text, tags_4, domain, attempt_rng, gt_struct)

        all_tags = applied_1_2 + applied_3 + applied_4

        # Validate - retry if invalid
        if _validate(raw_text, text, difficulty, domain, gt_struct):
            return text, all_tags

    # After max retries, return best effort
    return text, all_tags


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset.serializer import serialize

    # ── Shared gt_structs ──────────────────────────────────────────────────

    insurance_gt = {
        "policyholder": {
            "name": "Ashwin Shetty", "dob": "2002-08-12",
            "gender": "Male", "contact_number": None, "address": None
        },
        "policy": {
            "policy_number": "P1-093482", "policy_type": "Term Life",
            "start_date": "2023-01-01", "end_date": None
        },
        "premium": {"amount": 5000, "currency": "INR", "payment_frequency": "yearly"},
        "agent": {"name": "Rahul Verma", "agent_id": "RV-221"}
    }

    hospital_gt = {
        "patient": {
            "name": "Priya Sharma", "dob": "1995-11-07", "gender": "Female",
            "blood_group": "B+", "contact_number": None, "address": None
        },
        "visit": {
            "date": "2024-03-14", "time": "09:15",
            "department": "Cardiology", "reason_for_visit": "Chest pain and shortness of breath"
        },
        "vitals": {"blood_pressure": "120/80", "pulse": "78", "temperature": "98.6F", "weight": None, "height": None},
        "insurance": {"provider": "StarHealth", "policy_number": "SH-4521"},
        "attending_physician": {"name": "Dr Mehta", "id": "MH-0042"}
    }

    DIVIDER = "=" * 60

    for domain, gt in [("insurance", insurance_gt), ("hospital", hospital_gt)]:
        for difficulty in ["easy", "medium", "hard", "extreme"]:
            rng = random.Random(42)
            raw = serialize(gt, domain, random.Random(42))
            corrupted, tags = generate_noise(raw, difficulty, domain, rng, gt)

            print(f"\n{DIVIDER}")
            print(f"DOMAIN: {domain.upper()}  |  DIFFICULTY: {difficulty.upper()}")
            print(f"NOISE TAGS: {tags}")
            print(DIVIDER)
            print(corrupted)
