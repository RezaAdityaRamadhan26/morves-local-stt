"""Validate dataset manifests: schema, speaker independence, and sanity stats.

Usage:
  python scripts/validate_dataset.py --datasets-dir datasets
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from morves_stt.dataset import (
    DOMAIN_TAXONOMY,
    ManifestValidationError,
    SpeakerLeakageError,
    assert_no_speaker_leakage,
    load_manifest,
)

SPLITS = ("train", "validation", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=str, default="datasets")
    args = parser.parse_args()

    root = Path(args.datasets_dir)
    manifest_dir = root / "manifests"

    loaded: dict[str, list] = {}
    errors: list[str] = []

    for name in SPLITS:
        path = manifest_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found (skipping)")
            continue
        try:
            loaded[name] = load_manifest(path)
        except ManifestValidationError as exc:
            errors.append(f"{path}: {exc}")

    if errors:
        print("\nSCHEMA ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1

    if not loaded:
        print("no manifests to validate")
        return 0

    # Speaker leakage check (only across splits that exist).
    names = [n for n in SPLITS if n in loaded]
    try:
        assert_no_speaker_leakage(
            loaded.get("train", []),
            loaded.get("validation", []),
            loaded.get("test", []),
        )
    except SpeakerLeakageError as exc:
        print(f"\nFAIL: {exc}")
        return 1
    print("speaker-independent split: OK (zero overlap)")

    # Stats + policy checks.
    all_ok = True
    for name in names:
        entries = loaded[name]
        speakers = {e.speaker_id for e in entries}
        hours = sum(e.duration_sec for e in entries) / 3600
        domains = Counter(e.domain for e in entries)
        synth = sum(1 for e in entries if e.source == "synthetic")
        print(f"\n{name}: {len(entries)} utterances | {len(speakers)} speakers | {hours:.2f} h")
        print(f"  domains: {dict(sorted(domains.items()))}")
        if name == "test" and synth:
            print(
                f"  POLICY VIOLATION: test split contains {synth} "
                "synthetic utterances (must be real speech only)"
            )
            all_ok = False

    import itertools

    all_entries = list(itertools.chain.from_iterable(loaded.values()))
    unknown = {e.domain for e in all_entries if e.domain not in DOMAIN_TAXONOMY}
    if unknown:
        print(f"unknown domains outside taxonomy: {sorted(unknown)}")
        all_ok = False

    print("\nVALIDATION " + ("PASSED" if all_ok else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
