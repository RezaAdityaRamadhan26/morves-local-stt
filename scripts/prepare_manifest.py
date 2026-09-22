"""Build speaker-independent dataset manifests from a transcripts CSV.

CSV columns (header required):
  audio,text,speaker_id,language,duration_sec,domain,source[,recording_profile,noise_profile]

'audio' paths are relative to the datasets/ directory (e.g. processed/spk_001/000001.wav).
'duration_sec' may be left empty - it is then probed from the audio file.

Usage:
  python scripts/prepare_manifest.py --csv datasets/transcripts/records.csv --datasets-dir datasets
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from morves_stt.audio import AudioValidationError, inspect_audio
from morves_stt.dataset import (
    ManifestEntry,
    ManifestValidationError,
    parse_entry,
    save_manifest,
    split_by_speaker,
)

SPLITS = ("train", "validation", "test")


def load_csv(path: Path, datasets_dir: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"audio", "text", "speaker_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ManifestValidationError(f"CSV missing required columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            raw = {
                "audio": (row.get("audio") or "").strip(),
                "text": (row.get("text") or "").strip(),
                "speaker_id": (row.get("speaker_id") or "").strip(),
                "language": (row.get("language") or "id").strip(),
                "duration_sec": (row.get("duration_sec") or "").strip(),
                "domain": (row.get("domain") or "general").strip(),
                "source": (row.get("source") or "recorded").strip(),
                "recording_profile": (row.get("recording_profile") or "").strip() or None,
                "noise_profile": (row.get("noise_profile") or "").strip() or None,
            }
            if not raw["duration_sec"]:
                # Probe duration from the actual audio file.
                audio_rel = str(raw["audio"])
                audio_path = datasets_dir / audio_rel
                raw["duration_sec"] = str(inspect_audio(audio_path).duration_sec)
            entries.append(parse_entry(raw, index=i))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, required=True, help="Transcripts CSV path")
    parser.add_argument("--datasets-dir", type=str, default="datasets")
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    try:
        entries = load_csv(csv_path, datasets_dir)
    except (ManifestValidationError, AudioValidationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"loaded {len(entries)} entries from {csv_path}")
    hours = sum(e.duration_sec for e in entries) / 3600
    print(f"total duration: {hours:.2f} h; speakers: {len({e.speaker_id for e in entries})}")

    try:
        groups = split_by_speaker(entries, ratios=tuple(args.ratios), seed=args.seed)
    except ManifestValidationError as exc:
        print(f"ERROR: {exc}")
        return 1

    for name in SPLITS:
        # Private naming: generated manifests hold real transcript references
        # and are gitignored (datasets/manifests/*.private.*).
        out = datasets_dir / "manifests" / f"{name}.private.jsonl"
        save_manifest(groups[name], out)
        n = len(groups[name])
        h = sum(e.duration_sec for e in groups[name]) / 3600
        spk_n = len({e.speaker_id for e in groups[name]})
        print(f"{name}: {n} utterances ({h:.2f} h, {spk_n} speakers) -> {out}")

    print("speaker-independent split verified (no overlap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
