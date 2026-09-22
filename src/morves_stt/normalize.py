"""Text normalization rules for metric scoring.

IMPORTANT: Normalization for WER/CER must standardize purely typographic
differences (case, trailing punctuation, multiple spaces) WITHOUT erasing
meaningful financial differences:
  - DO NOT map "debit" -> "kredit"
  - DO NOT map "juta" -> "miliar"
  - DO NOT map "Medita" -> "media"
  - DO NOT convert spoken number words ("dua puluh") to digits ("20")
    because the ASR contract is verbatim transcript-level text.
"""

from __future__ import annotations

import re
import unicodedata

# Hyphens are DELETED (not spaced): "antar-perusahaan" unifies with the
# lexicon's canonical "antarperusahaan" - Indonesian hyphenation is orthographic
# variation, not a word-boundary/meaning difference in this domain.
_HYPHEN_REGEX = re.compile("-")
# All other punctuation/symbols are replaced with a space (token separator,
# never glued to neighbouring characters).
_PUNCT_REGEX = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_REGEX = re.compile(r"\s+")


def normalize_for_metric(text: str) -> str:
    """Normalize text for WER/CER and domain metric calculation.

    Rules:
    1. Unicode NFKC normalization (compose accents, normalize compatibility chars).
    2. Lowercase (Indonesian has no semantic case contrast in financial speech).
    3. Strip punctuation and separators; hyphens are removed so spelling
       variants unify (antar-perusahaan == antarperusahaan).
    4. Replace newlines/tabs with a single space.
    5. Collapse multiple whitespace into one space, strip leading/trailing.
    """
    if not text:
        return ""
    # Unicode NFKC
    t = unicodedata.normalize("NFKC", text)
    # Lowercase
    t = t.lower()
    # Hyphen variants deleted first (spelling-variant unification)
    t = _HYPHEN_REGEX.sub("", t)
    # Non-semantic punctuation becomes a token separator
    t = _PUNCT_REGEX.sub(" ", t)
    # Collapse whitespace
    t = _MULTI_SPACE_REGEX.sub(" ", t).strip()
    return t


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    norm = normalize_for_metric(text)
    return norm.split() if norm else []
