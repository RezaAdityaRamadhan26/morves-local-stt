"""Validate dataset manifests and print the STT-1 pilot readiness verdict.

Checks schema, speaker independence, test-split real-speech policy, unknown
domains, and missing audio files; then aggregates per-speaker / per-domain /
per-profile statistics and applies the pilot readiness floors from
``morves_stt.readiness`` (single source of truth).

Readiness command (from repo root):
  python scripts/validate_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from morves_stt.dataset import (
    ManifestEntry,
    ManifestValidationError,
    SpeakerLeakageError,
    assert_no_speaker_leakage,
    load_manifest,
)
from morves_stt.readiness import (
    DEFAULT_MAX_SPEAKERS,
    DEFAULT_MIN_PER_SPEAKER_MINUTES,
    DEFAULT_MIN_SPEAKERS,
    DEFAULT_MIN_TOTAL_HOURS,
    collect_stats,
    evaluate_readiness,
)

SPLITS = ("train", "validation", "test")


def find_missing_audio(entries: list[ManifestEntry], datasets_dir: Path) -> list[str]:
    """Return manifest audio paths (relative to datasets_dir) with no file."""
    missing: list[str] = []
    for e in entries:
        if not (datasets_dir / e.audio).exists():
            missing.append(e.audio)
    return missing


def fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in counts.items()) if counts else "(none)"


def fmt_hours(counts: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.2f}h" for k, v in counts.items()) if counts else "(none)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=str, default="datasets")
    parser.add_argument("--min-speakers", type=int, default=DEFAULT_MIN_SPEAKERS)
    parser.add_argument("--max-speakers", type=int, default=DEFAULT_MAX_SPEAKERS)
    parser.add_argument("--min-total-hours", type=float, default=DEFAULT_MIN_TOTAL_HOURS)
    parser.add_argument(
        "--min-per-speaker-minutes", type=float, default=DEFAULT_MIN_PER_SPEAKER_MINUTES
    )
    args = parser.parse_args()

    root = Path(args.datasets_dir)
    manifest_dir = root / "manifests"

    loaded: dict[str, list[ManifestEntry]] = {}
    schema_errors: list[str] = []

    for name in SPLITS:
        path = manifest_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found (skipping)")
            continue
        try:
            loaded[name] = load_manifest(path)
        except ManifestValidationError as exc:
            schema_errors.append(f"{path.name}: {exc}")

    if schema_errors:
        print("\nSCHEMA ERRORS:")
        for e in schema_errors:
            print(f"  - {e}")

    # Speaker leakage check (only across splits that exist).
    overlap_errors: list[str] = []
    try:
        assert_no_speaker_leakage(
            loaded.get("train", []),
            loaded.get("validation", []),
            loaded.get("test", []),
        )
        if loaded:
            print("speaker-independent split: OK (zero overlap)")
    except SpeakerLeakageError as exc:
        overlap_errors.append(str(exc))
        print(f"\nFAIL: {exc}")

    all_entries = [e for name in SPLITS for e in loaded.get(name, [])]
    missing = find_missing_audio(all_entries, root)

    stats = collect_stats(
        all_entries,
        test_entries=loaded.get("test", []),
        missing_audio=missing,
        speaker_overlap=overlap_errors,
        schema_errors=schema_errors,
    )

    if not all_entries and not schema_errors:
        print(
            "\nno manifest entries found - manifests are placeholders. "
            "Record speakers, run prepare_manifest.py, then rerun."
        )

    # Report.
    print(
        f"\nutterances: {stats.utterances} | speakers: {len(stats.speaker_ids)} "
        f"| total: {stats.total_hours:.2f} h"
    )
    print(f"per-speaker: {fmt_hours(stats.per_speaker_hours)}")
    print(f"utterances/speaker: {fmt_counts(stats.per_speaker_utterances)}")
    print(f"domain coverage: {fmt_counts(stats.domain_counts)}")
    print(f"recording profiles: {fmt_counts(stats.recording_profiles)}")
    print(f"noise profiles: {fmt_counts(stats.noise_profiles)}")
    print(f"languages: {fmt_counts(stats.language_counts)}")
    if stats.synthetic_in_test:
        print(
            f"POLICY VIOLATION: test split has {stats.synthetic_in_test} synthetic "
            "utterance(s) (real speech only)"
        )
    if stats.missing_audio:
        print(f"MISSING AUDIO: {len(stats.missing_audio)} file(s), e.g. {stats.missing_audio[:3]}")

    result = evaluate_readiness(
        stats,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        min_total_hours=args.min_total_hours,
        min_per_speaker_minutes=args.min_per_speaker_minutes,
    )
    print("\n" + result.verdict)
    for d in result.deficits:
        print(f"  - {d}")
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
