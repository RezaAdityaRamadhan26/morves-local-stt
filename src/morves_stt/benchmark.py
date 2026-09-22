"""Benchmark runner: manifest -> transcription -> metrics -> JSON + Markdown reports."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from morves_stt.dataset import ManifestEntry
from morves_stt.inference import SpeechToTextModel, TranscriptionResult
from morves_stt.metrics import TermProfile, UtteranceMetrics, evaluate_utterance
from morves_stt.semantic import semantic_scores


@dataclass
class UtteranceRecord:
    speaker_id: str
    domain: str
    duration_sec: float
    latency_ms: float
    real_time_factor: float
    reference: str
    hypothesis: str
    metrics: UtteranceMetrics
    audio_path: str
    recording_profile: str | None = None
    semantic: dict[str, float | None] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    model_id: str
    records: list[UtteranceRecord] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)  # skipped utterances with reason
    started_at: str = ""
    finished_at: str = ""


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _aggregate_scores(records: list[UtteranceRecord], attr: str) -> dict[str, float | None]:
    expected = sum(getattr(r.metrics, attr).expected for r in records)
    predicted = sum(getattr(r.metrics, attr).predicted for r in records)
    correct = sum(getattr(r.metrics, attr).correct for r in records)
    return {
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "accuracy": (correct / expected) if expected else None,
    }


def summarize(result: BenchmarkResult) -> dict[str, Any]:
    """Aggregate utterance records into global / per-domain / per-speaker stats."""
    records = result.records
    wers = [r.metrics.wer for r in records]
    cers = [r.metrics.cer for r in records]

    latencies = sorted(r.latency_ms for r in records)
    rtfs = [r.real_time_factor for r in records]

    # Per-domain
    domains: dict[str, list[UtteranceRecord]] = {}
    for r in records:
        domains.setdefault(r.domain, []).append(r)

    per_domain = {
        d: {
            "utterances": len(rs),
            "mean_wer": mean([r.metrics.wer for r in rs]) if rs else None,
            "phrase_accuracy": (
                sum(1 for r in rs if r.metrics.exact_match) / len(rs) if rs else None
            ),
        }
        for d, rs in sorted(domains.items())
    }

    # Per-speaker (pseudonymous IDs only)
    speakers: dict[str, list[UtteranceRecord]] = {}
    for r in records:
        speakers.setdefault(r.speaker_id, []).append(r)
    per_speaker = {
        s: {
            "utterances": len(rs),
            "mean_wer": mean([r.metrics.wer for r in rs]) if rs else None,
        }
        for s, rs in sorted(speakers.items())
    }

    # Critical confusion counts (global, per speaker, per recording profile)
    confusions: dict[str, dict[str, int]] = {}
    by_speaker: dict[str, dict[str, dict[str, int]]] = {}
    by_profile: dict[str, dict[str, dict[str, int]]] = {}

    def _bump(target: dict, ref: str, hyp: str) -> None:
        target.setdefault(ref, {})
        target[ref][hyp] = target[ref].get(hyp, 0) + 1

    for r in records:
        for ev in r.metrics.critical_events:
            hyp_term = ev.hypothesis_term or "<deleted>"
            _bump(confusions, ev.reference_term, hyp_term)
            _bump(by_speaker.setdefault(r.speaker_id, {}), ev.reference_term, hyp_term)
            _bump(
                by_profile.setdefault(r.recording_profile or "(unset)", {}),
                ev.reference_term,
                hyp_term,
            )

    def _semantic_accuracy(attr: str) -> float | None:
        matched = 0.0
        counted = 0
        for r in records:
            score = r.semantic.get(attr)
            if score is not None:
                matched += score
                counted += 1
        return (matched / counted) if counted else None

    return {
        "model_id": result.model_id,
        "utterances": len(records),
        "skipped": len(result.errors),
        "wer_macro": mean(wers) if wers else None,
        "cer_macro": mean(cers) if cers else None,
        "phrase_accuracy_global": (
            sum(1 for r in records if r.metrics.exact_match) / len(records) if records else None
        ),
        "entity_term_accuracy": _aggregate_scores(records, "entity_score")["accuracy"],
        "financial_term_accuracy": _aggregate_scores(records, "financial_score")["accuracy"],
        "currency_term_accuracy": _aggregate_scores(records, "currency_score")["accuracy"],
        "amount_token_accuracy": _aggregate_scores(records, "amount_score")["accuracy"],
        "date_token_accuracy": _aggregate_scores(records, "date_score")["accuracy"],
        "amount_semantic_accuracy": _semantic_accuracy("amount_semantic"),
        "date_semantic_accuracy": _semantic_accuracy("date_semantic"),
        "currency_semantic_accuracy": _semantic_accuracy("currency_semantic"),
        "critical_term_errors": confusions,
        "critical_term_errors_by_speaker": by_speaker,
        "critical_term_errors_by_profile": by_profile,
        "latency_ms": {
            "mean": mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "real_time_factor_mean": mean(rtfs) if rtfs else None,
        "per_domain": per_domain,
        "per_speaker": per_speaker,
    }


def run_benchmark(
    model: SpeechToTextModel,
    entries: list[ManifestEntry],
    audio_root: str | Path,
    terms_profile: TermProfile,
    warmup: int = 0,
    limit: int | None = None,
) -> BenchmarkResult:
    """Transcribe every manifest entry and score it. Never fabricates numbers.

    Entries whose audio file is missing are recorded in .errors, not silently dropped.
    """
    result = BenchmarkResult(
        model_id=getattr(model, "model_id", "unknown"),
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    selected = entries[:limit] if limit else entries

    # Warmup: run on the first available utterance N times (untimed).
    if warmup > 0:
        for entry in selected:
            path = Path(audio_root) / entry.audio
            if not path.exists():
                continue
            for _ in range(warmup):
                try:
                    model.transcribe(path, language=entry.language)
                except Exception:
                    break
            break

    for entry in selected:
        path = Path(audio_root) / entry.audio
        if not path.exists():
            result.errors.append(
                {"audio": entry.audio, "reason": "file not found", "speaker_id": entry.speaker_id}
            )
            continue

        t0 = time.perf_counter()
        try:
            hyp: TranscriptionResult = model.transcribe(path, language=entry.language)
        except Exception as exc:
            result.errors.append(
                {
                    "audio": entry.audio,
                    "reason": f"transcription failed: {exc}",
                    "speaker_id": entry.speaker_id,
                }
            )
            continue
        latency_ms = (time.perf_counter() - t0) * 1000.0

        duration = hyp.duration_ms / 1000.0 if hyp.duration_ms else entry.duration_sec
        metrics = evaluate_utterance(entry.text, hyp.text, terms_profile)

        result.records.append(
            UtteranceRecord(
                speaker_id=entry.speaker_id,
                domain=entry.domain,
                duration_sec=duration,
                latency_ms=latency_ms,
                real_time_factor=(latency_ms / 1000.0 / duration) if duration > 0 else 0.0,
                reference=entry.text,
                hypothesis=hyp.text,
                metrics=metrics,
                audio_path=entry.audio,
                recording_profile=entry.recording_profile,
                semantic=semantic_scores(entry.text, hyp.text),
            )
        )

    result.finished_at = datetime.now(timezone.utc).isoformat()
    return result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_reports(
    result: BenchmarkResult, output_dir: str | Path, meta: dict[str, Any] | None = None
) -> tuple[Path, Path]:
    """Write machine-readable JSON and human-readable Markdown reports.

    Reports may contain transcripts of private audio; evaluation/reports/ is
    gitignored for this reason. Never commit them.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()

    summary = summarize(result)

    json_path = out_dir / f"{stamp}-baseline.json"
    payload = {
        "meta": meta or {},
        "summary": summary,
        "records": [
            {
                "speaker_id": r.speaker_id,
                "domain": r.domain,
                "audio": r.audio_path,
                "duration_sec": r.duration_sec,
                "latency_ms": r.latency_ms,
                "real_time_factor": r.real_time_factor,
                "reference": r.reference,
                "hypothesis": r.hypothesis,
                "wer": r.metrics.wer,
                "cer": r.metrics.cer,
                "exact_match": r.metrics.exact_match,
                "semantic": r.semantic,
                "recording_profile": r.recording_profile,
                "critical_events": [asdict(e) for e in r.metrics.critical_events],
            }
            for r in result.records
        ],
        "errors": result.errors,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / f"{stamp}-baseline.md"
    md: list[str] = [
        "# Morves Local STT — Baseline Benchmark Report",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Utterances: {summary['utterances']} (skipped: {summary['skipped']})",
        f"- WER (macro): {summary['wer_macro']}",
        f"- CER (macro): {summary['cer_macro']}",
        f"- Phrase accuracy: {summary['phrase_accuracy_global']}",
        f"- Entity accuracy: {summary['entity_term_accuracy']}",
        f"- Financial term accuracy: {summary['financial_term_accuracy']}",
        f"- Currency accuracy: {summary['currency_term_accuracy']}",
        f"- Amount accuracy (strict): {summary['amount_token_accuracy']}",
        f"- Date accuracy (strict): {summary['date_token_accuracy']}",
        f"- Amount semantic accuracy: {summary['amount_semantic_accuracy']}",
        f"- Date semantic accuracy: {summary['date_semantic_accuracy']}",
        f"- Currency semantic accuracy: {summary['currency_semantic_accuracy']}",
        f"- Latency ms (mean/p50/p95): {summary['latency_ms']}",
        f"- RTF (mean): {summary['real_time_factor_mean']}",
        "",
        "## Critical term errors",
        "",
    ]
    if summary["critical_term_errors"]:
        for ref, conf in summary["critical_term_errors"].items():
            for hyp_t, n in conf.items():
                md.append(f"- `{ref}` -> `{hyp_t}` x {n}")
    else:
        md.append("- none")

    md += [
        "",
        "## Per-domain",
        "",
        "| domain | utterances | mean WER | phrase acc |",
        "|---|---|---|---|",
    ]
    for d, stats in summary["per_domain"].items():
        md.append(
            f"| {d} | {stats['utterances']} | {stats['mean_wer']} | {stats['phrase_accuracy']} |"
        )

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path
