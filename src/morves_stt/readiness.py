"""Pilot dataset readiness evaluation (STT-1 gate for STT-2).

Single source of truth for the "PILOT DATASET READY FOR STT-2" verdict.
Both ``scripts/validate_dataset.py`` and ``scripts/check_pilot_readiness.py``
consume :func:`evaluate_readiness`; readiness rules are never duplicated.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from morves_stt.dataset import DOMAIN_TAXONOMY, ManifestEntry

# STT-1 pilot floors (see docs/STT-1-PILOT-DATASET-PLAN.md).
DEFAULT_MIN_SPEAKERS = 4
DEFAULT_MAX_SPEAKERS = 6
DEFAULT_MIN_TOTAL_HOURS = 1.0
DEFAULT_MIN_PER_SPEAKER_MINUTES = 10.0
READY_VERDICT = "PILOT DATASET READY FOR STT-2"
NOT_READY_VERDICT = "PILOT DATASET NOT READY"


@dataclass
class PilotDatasetStats:
    """Aggregated statistics over all loaded (train+validation+test) entries."""

    utterances: int = 0
    speaker_ids: list[str] = field(default_factory=list)
    total_hours: float = 0.0
    per_speaker_hours: dict[str, float] = field(default_factory=dict)
    per_speaker_utterances: dict[str, int] = field(default_factory=dict)
    domain_counts: dict[str, int] = field(default_factory=dict)
    recording_profiles: dict[str, int] = field(default_factory=dict)
    noise_profiles: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    synthetic_in_test: int = 0
    missing_audio: list[str] = field(default_factory=list)
    speaker_overlap: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)


@dataclass
class ReadinessResult:
    """Verdict plus the exact deficits that block STT-2."""

    ready: bool
    verdict: str
    deficits: list[str] = field(default_factory=list)


def collect_stats(
    entries: list[ManifestEntry],
    *,
    test_entries: list[ManifestEntry] | None = None,
    missing_audio: list[str] | None = None,
    speaker_overlap: list[str] | None = None,
    schema_errors: list[str] | None = None,
) -> PilotDatasetStats:
    """Aggregate pilot statistics from manifest entries.

    ``missing_audio``/``speaker_overlap``/``schema_errors`` are collected by the
    caller (filesystem / split checks) and passed through so the verdict sees
    everything in one place.
    """
    per_speaker_sec: dict[str, float] = defaultdict(float)
    per_speaker_n: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    rec_profiles: Counter[str] = Counter()
    noise_prof: Counter[str] = Counter()
    languages: Counter[str] = Counter()

    for e in entries:
        per_speaker_sec[e.speaker_id] += e.duration_sec
        per_speaker_n[e.speaker_id] += 1
        domains[e.domain] += 1
        languages[e.language] += 1
        rec_profiles[e.recording_profile or "(unset)"] += 1
        noise_prof[e.noise_profile or "(unset)"] += 1

    test_entries = test_entries or []
    synthetic_in_test = sum(1 for e in test_entries if e.source == "synthetic")

    return PilotDatasetStats(
        utterances=len(entries),
        speaker_ids=sorted(per_speaker_n),
        total_hours=sum(per_speaker_sec.values()) / 3600,
        per_speaker_hours={s: sec / 3600 for s, sec in sorted(per_speaker_sec.items())},
        per_speaker_utterances=dict(sorted(per_speaker_n.items())),
        domain_counts=dict(sorted(domains.items())),
        recording_profiles=dict(sorted(rec_profiles.items())),
        noise_profiles=dict(sorted(noise_prof.items())),
        language_counts=dict(sorted(languages.items())),
        synthetic_in_test=synthetic_in_test,
        missing_audio=sorted(missing_audio or []),
        speaker_overlap=sorted(speaker_overlap or []),
        schema_errors=sorted(schema_errors or []),
    )


def evaluate_readiness(
    stats: PilotDatasetStats,
    *,
    min_speakers: int = DEFAULT_MIN_SPEAKERS,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    min_total_hours: float = DEFAULT_MIN_TOTAL_HOURS,
    min_per_speaker_minutes: float = DEFAULT_MIN_PER_SPEAKER_MINUTES,
) -> ReadinessResult:
    """Apply STT-1 pilot readiness floors.

    Only real human recordings count: synthetic utterances never contribute to
    speaker counts or duration floors (callers must exclude them or accept the
    verdict's honest shortfall).
    """
    deficits: list[str] = []

    if stats.schema_errors:
        deficits.append(f"{len(stats.schema_errors)} schema error(s) in manifests")

    if stats.speaker_overlap:
        deficits.append(f"speaker leakage across splits: {', '.join(stats.speaker_overlap)}")

    if stats.missing_audio:
        deficits.append(
            f"{len(stats.missing_audio)} manifest entries reference missing audio files"
        )

    if stats.synthetic_in_test:
        deficits.append(
            f"test split contains {stats.synthetic_in_test} synthetic utterance(s) "
            "(real speech only)"
        )

    n_speakers = len(stats.speaker_ids)
    if n_speakers < min_speakers:
        deficits.append(f"speakers: {n_speakers} (need >= {min_speakers})")
        if n_speakers == 0:
            deficits.append("no human recordings loaded - WAITING FOR HUMAN RECORDINGS")
    elif n_speakers > max_speakers:
        deficits.append(f"speakers: {n_speakers} (pilot scope is {min_speakers}-{max_speakers})")

    if stats.total_hours < min_total_hours:
        deficits.append(
            f"total duration: {stats.total_hours:.2f} h (need >= {min_total_hours:.1f} h)"
        )
        if stats.total_hours == 0:
            deficits[-1] += " - WAITING FOR HUMAN RECORDINGS"

    thin = [
        f"{s} ({stats.per_speaker_hours.get(s, 0.0) * 60:.0f} min)"
        for s in stats.speaker_ids
        if stats.per_speaker_hours.get(s, 0.0) * 60 < min_per_speaker_minutes
    ]
    if thin:
        deficits.append(f"speakers under {min_per_speaker_minutes:.0f} min each: {', '.join(thin)}")

    missing_domains = [d for d in DOMAIN_TAXONOMY if d not in stats.domain_counts]
    if missing_domains:
        deficits.append(
            f"domains not covered ({len(missing_domains)}/{len(DOMAIN_TAXONOMY)}): "
            f"{', '.join(missing_domains)}"
        )

    unknown_domains = [d for d in stats.domain_counts if d not in DOMAIN_TAXONOMY]
    if unknown_domains:
        deficits.append(f"unknown domains outside taxonomy: {', '.join(unknown_domains)}")

    ready = not deficits
    return ReadinessResult(
        ready=ready,
        verdict=READY_VERDICT if ready else NOT_READY_VERDICT,
        deficits=deficits,
    )
