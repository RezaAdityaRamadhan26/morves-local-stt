"""Semantic diagnostic metrics for numbers, dates, and currencies (STT-1C).

STRICT metrics (``morves_stt.metrics``) stay authoritative: they penalize any
transcript deviation, including spoken words transcribed as digits
("dua ratus lima puluh juta" -> "250 juta"). The metrics here are *separate
diagnostics* that recognize equivalent numeric meaning. They never replace
strict scores and never normalize entities: Medita != media, always.

Rules:
- Amounts: "dua ratus lima puluh juta" == "250 juta" == "250.000.000"? No -
  bare "250" is 250, only equal when the multiplier is present in both.
  "lima juta" != "lima miliar" (5e6 vs 5e9).
- Dates: "dua belas September" == "12 September"; "September" != "Oktober"
  (different month index); unrecognized month spellings do not match.
- Currencies: rupiah == Rp == IDR; dolar == dollar == USD; rupiah != dolar.
"""

from __future__ import annotations

import re
from collections import Counter

_UNITS = {
    "nol": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
}
_TEENS = {"sepuluh": 10, "sebelas": 11}
_MULTIPLIERS = {"ribu": 1e3, "juta": 1e6, "miliar": 1e9, "triliun": 1e12}
_SPECIAL_ATOMS = {"seratus": 100, "setengah": 0.5}
_SPECIAL_MULTIPLIER = {"seribu": 1e3}
_COMMA = "koma"

_MONTHS = {
    "januari": 1,
    "februari": 2,
    "maret": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "october": 10,
    "december": 12,
}
_MONTH_ALIASES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "agu": 8,
    "aug": 8,
    "sep": 9,
    "okt": 10,
    "oct": 10,
    "nov": 11,
    "des": 12,
    "dec": 12,
}

_CURRENCY_CLASSES: dict[str, str] = {}
for _tokens, _cls in (
    (("idr", "rupiah", "rp"), "IDR"),
    (("usd", "dolar", "dollar"), "USD"),
    (("eur", "euro"), "EUR"),
):
    for _t in _tokens:
        _CURRENCY_CLASSES[_t] = _cls

_DIGIT_RE = re.compile(r"^\d+(?:[.,]\d+)*$")
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+$")
# Words or intact digit runs: "1,5" / "250.000" / "13.500.000" stay one token
# (normalize.tokenize would split them into separate numbers).
_SEMANTIC_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")


def _semantic_tokens(text: str) -> list[str]:
    return _SEMANTIC_TOKEN_RE.findall(text.lower())


def _digit_value(tok: str) -> float | None:
    if not _DIGIT_RE.match(tok):
        return None
    try:
        if _THOUSANDS_RE.match(tok):  # 250.000 -> 250000 (Indonesian thousands)
            return float(tok.replace(".", ""))
        if "," in tok:  # 1,5 -> 1.5 (Indonesian decimal comma)
            return float(tok.replace(",", "."))
        return float(tok)
    except ValueError:  # e.g. "1.2.3" - separators, not a real number
        return None


def _word_atom(tokens: list[str], i: int) -> tuple[float, int] | None:
    """Parse ONE word-number group (e.g. 'dua ratus', 'lima puluh', 'sebelas')."""
    t = tokens[i]
    if t in _SPECIAL_ATOMS:
        return _SPECIAL_ATOMS[t], i + 1
    if t in _TEENS:
        return float(_TEENS[t]), i + 1
    if t not in _UNITS:
        return None
    val = _UNITS[t]
    j = i + 1
    if j < len(tokens) and tokens[j] == "ratus":
        val *= 100
        j += 1
    if j < len(tokens) and tokens[j] == "puluh":
        if val == 1:  # "satu puluh" never occurs; "sepuluh" handled above
            return None
        val *= 10
        j += 1
        if j < len(tokens) and tokens[j] in _UNITS:
            val += _UNITS[tokens[j]]
            j += 1
    elif j < len(tokens) and tokens[j] == "belas":
        val += 10
        j += 1
    return float(val), j


def parse_number_at(tokens: list[str], i: int) -> tuple[float, int] | None:
    """Parse the maximal number phrase starting at index i -> (value, next_i).

    Grammar: atoms (word numbers, digit tokens) accumulate into `current`;
    a multiplier (ribu/juta/miliar/triliun/seribu) commits `current` (1 when
    absent) scaled into `total`; 'koma' switches to fractional-digit mode.
    Returns None if no number starts at i.
    """
    total = 0.0
    current: float | None = None
    j = i
    consumed = False

    while j < len(tokens):
        tok = tokens[j]
        if tok in _MULTIPLIERS or tok in _SPECIAL_MULTIPLIER:
            mult = _MULTIPLIERS.get(tok) or _SPECIAL_MULTIPLIER[tok]
            total += (current if current is not None else 1.0) * mult
            current = None
            j += 1
            consumed = True
            continue
        if tok == _COMMA and current is not None:
            k = j + 1
            digits = ""
            while k < len(tokens) and tokens[k] in _UNITS:
                digits += str(_UNITS[tokens[k]])
                k += 1
            if digits:
                current = current + int(digits) / (10 ** len(digits))
                j = k
                continue
            break
        atom = _digit_value(tok)
        if atom is None:
            parsed = _word_atom(tokens, j)
            if parsed is None:
                break
            atom, j = parsed
            current = (current or 0.0) + atom
            consumed = True
            continue
        current = (current or 0.0) + atom
        j += 1
        consumed = True

    if not consumed:
        return None
    if current is not None:
        total += current
    return total, j


def extract_amounts(text: str) -> list[float]:
    """All numeric values in text (spoken words and digits, unified grammar)."""
    tokens = _semantic_tokens(text)
    values: list[float] = []
    i = 0
    while i < len(tokens):
        parsed = parse_number_at(tokens, i)
        if parsed is not None:
            value, nxt = parsed
            values.append(value)
            i = nxt
        else:
            i += 1
    return values


def _month_of(tok: str) -> int | None:
    return _MONTHS.get(tok) or _MONTH_ALIASES.get(tok)


def _year_value(tokens: list[str], i: int) -> tuple[int, int] | None:
    parsed = parse_number_at(tokens, i)
    if parsed is None:
        return None
    value, nxt = parsed
    return (int(value), nxt) if 1000 <= value <= 2100 else None


def extract_dates(text: str) -> list[tuple[int | None, int, int | None]]:
    """(day, month, year) tuples for each month mention in text.

    Day = number phrase ending immediately before the month (<= 31); year =
    number phrase starting right after the month in 1000..2100. Month
    spellings that are not real months do not match - semantic equivalence
    covers numeric notation only, not recognition errors.
    """
    tokens = _semantic_tokens(text)
    out: list[tuple[int | None, int, int | None]] = []
    i = 0
    while i < len(tokens):
        month = _month_of(tokens[i])
        if month is None:
            i += 1
            continue
        day: int | None = None
        for s in range(max(0, i - 3), i):
            parsed = parse_number_at(tokens, s)
            if parsed is not None and parsed[1] == i and parsed[0] <= 31:
                day = int(parsed[0])
                break
        year: int | None = None
        yr = _year_value(tokens, i + 1)
        if yr is not None:
            year = yr[0]
        out.append((day, month, year))
        i += 1
    return out


def extract_currencies(text: str) -> Counter[str]:
    """Counts per canonical currency class (IDR/USD/EUR) mentioned in text."""
    tokens = _semantic_tokens(text)
    counts: Counter[str] = Counter()
    for tok in tokens:
        cls = _CURRENCY_CLASSES.get(tok)
        if cls:
            counts[cls] += 1
    return counts


def _multiset_match_fraction(expected: list, predicted: list) -> float:
    """Fraction of expected items matched by predicted items (order-free)."""
    if not expected:
        return 1.0
    pool = list(predicted)
    matched = 0
    for item in expected:
        for k, cand in enumerate(pool):
            if _values_equal(item, cand):
                del pool[k]
                matched += 1
                break
    return matched / len(expected)


def _values_equal(a, b) -> bool:  # numeric or tuple compare
    if isinstance(a, tuple) and isinstance(b, tuple):
        return all(x == y for x, y in zip(a, b, strict=False))
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))


def amount_semantic_accuracy(reference: str, hypothesis: str) -> float | None:
    """None when the reference contains no numbers (utterance not applicable)."""
    exp = extract_amounts(reference)
    if not exp:
        return None
    return _multiset_match_fraction(exp, extract_amounts(hypothesis))


def date_semantic_accuracy(reference: str, hypothesis: str) -> float | None:
    exp = extract_dates(reference)
    if not exp:
        return None
    return _multiset_match_fraction(exp, extract_dates(hypothesis))


def currency_semantic_accuracy(reference: str, hypothesis: str) -> float | None:
    exp = extract_currencies(reference)
    if not exp:
        return None
    got = extract_currencies(hypothesis)
    total_exp = sum(exp.values())
    matched = sum((exp & got).values())
    return matched / total_exp


def semantic_scores(reference: str, hypothesis: str) -> dict[str, float | None]:
    """All semantic diagnostics for one utterance (None = not applicable)."""
    return {
        "amount_semantic": amount_semantic_accuracy(reference, hypothesis),
        "date_semantic": date_semantic_accuracy(reference, hypothesis),
        "currency_semantic": currency_semantic_accuracy(reference, hypothesis),
    }
