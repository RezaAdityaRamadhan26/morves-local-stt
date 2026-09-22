"""Domain-aware evaluation metrics for Morves Local STT.

Implements:
  - WER (word error rate) via token-level Levenshtein alignment
  - CER (character error rate) via character-level Levenshtein alignment
  - Entity Term Accuracy
  - Financial Term Accuracy
  - Amount Token Accuracy
  - Date Token Accuracy
  - Currency Token Accuracy
  - Domain Phrase Accuracy (exact normalized match rate per domain)
  - Critical confusion tracking (e.g. medita -> media, debit -> kredit)

All metrics use normalize.tokenize / normalize_for_metric from morves_stt.normalize.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from morves_stt.normalize import normalize_for_metric, tokenize

# ---------------------------------------------------------------------------
# Alignment core (Levenshtein with backtrace)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    """Token-level edit operations between reference and hypothesis."""

    substitutions: int
    deletions: int
    insertions: int
    ref_len: int
    hyp_len: int
    # (ref_token, hyp_token) pairs for every substitution
    substitution_pairs: tuple[tuple[str, str], ...] = ()

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Compute Levenshtein alignment ops between two token sequences."""
    ref = list(reference)
    hyp = list(hypothesis)
    n, m = len(ref), len(hyp)

    # DP table of (cost, op) where op: 'match'|'sub'|'del'|'ins'
    dp: list[list[tuple[int, str]]] = [[(0, "") for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, "del")
    for j in range(1, m + 1):
        dp[0][j] = (j, "ins")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = (dp[i - 1][j - 1][0], "match")
            else:
                best_cost, best_op = min(
                    (dp[i - 1][j - 1][0] + 1, "sub"),
                    (dp[i - 1][j][0] + 1, "del"),
                    (dp[i][j - 1][0] + 1, "ins"),
                )
                dp[i][j] = (best_cost, best_op)

    # Backtrace
    subs, dels, ins = 0, 0, 0
    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = dp[i][j][1]
        if op == "match":
            i, j = i - 1, j - 1
        elif op == "sub":
            subs += 1
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif op == "del":
            dels += 1
            i -= 1
        elif op == "ins":
            ins += 1
            j -= 1
        else:  # pragma: no cover - defensive
            break

    return Alignment(
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        ref_len=n,
        hyp_len=m,
        substitution_pairs=tuple(reversed(pairs)),
    )


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WERResult:
    wer: float
    alignment: Alignment

    @property
    def errors(self) -> int:
        return self.alignment.errors


def wer(reference: str, hypothesis: str) -> WERResult:
    """Word Error Rate = (S + D + I) / N over normalized tokens."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    a = align(ref_tokens, hyp_tokens)
    denom = max(a.ref_len, 1)
    return WERResult(wer=a.errors / denom, alignment=a)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate over normalized text with whitespace removed."""
    ref_chars = list(normalize_for_metric(reference).replace(" ", ""))
    hyp_chars = list(normalize_for_metric(hypothesis).replace(" ", ""))
    a = align(ref_chars, hyp_chars)
    return a.errors / max(a.ref_len, 1)


# ---------------------------------------------------------------------------
# Term occurrence helpers (multi-word aware)
# ---------------------------------------------------------------------------


def count_term_occurrences(tokens_: Sequence[str], term: str) -> int:
    """Count occurrences of a (possibly multi-word) term in a token sequence."""
    term_tokens = term.lower().split()
    if not term_tokens or not tokens_:
        return 0
    count = 0
    tl = len(term_tokens)
    for i in range(len(tokens_) - tl + 1):
        if list(tokens_[i : i + tl]) == term_tokens:
            count += 1
    return count


@dataclass
class TermCategoryScore:
    """Occurrence-based term accuracy for one term category in one utterance."""

    expected: int = 0
    predicted: int = 0
    correct: int = 0

    def accuracy(self) -> float | None:
        return self.correct / self.expected if self.expected else None


def score_terms(
    ref_tokens: Sequence[str], hyp_tokens: Sequence[str], terms: Iterable[str]
) -> TermCategoryScore:
    """Occurrence-based scoring: correct = min(expected, predicted) per term.

    This is a conservative, position-blind proxy suitable for STT-0; positional
    accuracy arrives with forced-alignment tooling in a later stage.
    """
    score = TermCategoryScore()
    for term in terms:
        term = term.lower().strip()
        if not term:
            continue
        expected = count_term_occurrences(ref_tokens, term)
        predicted = count_term_occurrences(hyp_tokens, term)
        score.expected += expected
        score.predicted += predicted
        score.correct += min(expected, predicted)
    return score


# ---------------------------------------------------------------------------
# Critical confusion model
# ---------------------------------------------------------------------------


@dataclass
class CriticalErrorEvent:
    reference_term: str
    hypothesis_term: str  # "" means deletion


def find_critical_errors(
    wer_result: WERResult,
    critical_pairs: dict[str, set[str]],
) -> list[CriticalErrorEvent]:
    """Detect critical domain errors from substitution pairs.

    critical_pairs maps normalized reference token -> set of forbidden
    hypothesis tokens (e.g. {"medita": {"media"}, "debit": {"kredit"}}).
    """
    events: list[CriticalErrorEvent] = []
    for ref_tok, hyp_tok in wer_result.alignment.substitution_pairs:
        forbidden = critical_pairs.get(ref_tok)
        if forbidden and hyp_tok in forbidden:
            events.append(CriticalErrorEvent(ref_tok, hyp_tok))
    return events


def find_critical_deletions(
    reference: str, hypothesis: str, critical_terms: Iterable[str]
) -> list[CriticalErrorEvent]:
    """A critical term present in the reference but wholly absent in the hypothesis."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    events: list[CriticalErrorEvent] = []
    for term in critical_terms:
        term = term.lower().strip()
        if not term:
            continue
        if count_term_occurrences(ref_tokens, term) > count_term_occurrences(hyp_tokens, term):
            events.append(CriticalErrorEvent(term, ""))
    return events


# ---------------------------------------------------------------------------
# Per-utterance evaluation
# ---------------------------------------------------------------------------


@dataclass
class TermProfile:
    entities: list[str] = field(default_factory=list)
    financial_terms: list[str] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)
    amount_tokens: list[str] = field(default_factory=list)
    date_tokens: list[str] = field(default_factory=list)
    critical_terms: list[str] = field(default_factory=list)
    critical_pairs: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class UtteranceMetrics:
    wer: float
    cer: float
    exact_match: bool
    entity_score: TermCategoryScore
    financial_score: TermCategoryScore
    currency_score: TermCategoryScore
    amount_score: TermCategoryScore
    date_score: TermCategoryScore
    critical_events: list[CriticalErrorEvent]


def evaluate_utterance(reference: str, hypothesis: str, profile: TermProfile) -> UtteranceMetrics:
    w = wer(reference, hypothesis)
    c = cer(reference, hypothesis)
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    events = find_critical_errors(w, profile.critical_pairs)
    events.extend(find_critical_deletions(reference, hypothesis, profile.critical_terms))

    return UtteranceMetrics(
        wer=w.wer,
        cer=c,
        exact_match=normalize_for_metric(reference) == normalize_for_metric(hypothesis),
        entity_score=score_terms(ref_tokens, hyp_tokens, profile.entities),
        financial_score=score_terms(ref_tokens, hyp_tokens, profile.financial_terms),
        currency_score=score_terms(ref_tokens, hyp_tokens, profile.currencies),
        amount_score=score_terms(ref_tokens, hyp_tokens, profile.amount_tokens),
        date_score=score_terms(ref_tokens, hyp_tokens, profile.date_tokens),
        critical_events=events,
    )


def load_term_profile(path: str) -> TermProfile:
    """Load a TermProfile from a YAML terms file (see evaluation/fixtures/terms.yaml)."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def _terms(key: str) -> list[str]:
        return [str(t).lower() for t in data.get(key, [])]

    critical_pairs: dict[str, set[str]] = {}
    for item in data.get("critical_confusions", []):
        ref = str(item.get("reference", "")).lower().strip()
        forbidden = {str(t).lower().strip() for t in item.get("forbidden", [])}
        if ref and forbidden:
            critical_pairs.setdefault(ref, set()).update(forbidden)

    return TermProfile(
        entities=_terms("entities"),
        financial_terms=_terms("financial_terms"),
        currencies=_terms("currencies"),
        amount_tokens=_terms("amount_tokens"),
        date_tokens=_terms("date_tokens"),
        critical_terms=sorted(set().union(critical_pairs.keys(), _terms("critical_terms"))),
        critical_pairs=critical_pairs,
    )
