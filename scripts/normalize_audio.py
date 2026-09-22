"""Normalize dataset audio into the canonical format (WAV PCM mono 16 kHz 16-bit).

Actually decodes and converts; never renames compressed files to .wav.

Usage:
  python scripts/normalize_audio.py datasets/raw --out-dir datasets/processed [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from morves_stt.audio import AudioValidationError, inspect_audio, normalize_to_canonical_wav

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".aiff", ".webm"}


def iter_audio(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="File or directory of audio to normalize")
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Output directory (structure preserved)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be done")
    args = parser.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)

    files = [src] if src.is_file() else iter_audio(src)
    if not files:
        print(f"no audio files found under {src}")
        return 1

    ok, failed, skipped = 0, 0, 0
    for f in files:
        rel = f.relative_to(src) if src.is_dir() else Path(f.name)
        # Canonical layout: out_dir/<relative parents>/<stem>.wav
        dest = out_dir / rel.with_suffix(".wav")
        print(f"{f} -> {dest}")
        if args.dry_run:
            continue
        try:
            info = inspect_audio(f)
            normalize_to_canonical_wav(f, dest)
            print(
                f"  ok: {info.sample_rate}Hz ch={info.channels} {info.duration_sec:.2f}s "
                f"-> 16000Hz ch=1 PCM_16"
            )
            ok += 1
        except AudioValidationError as exc:
            print(f"  FAIL: {exc}")
            failed += 1
        except FileExistsError:
            skipped += 1

    print(f"\nsummary: ok={ok} failed={failed} skipped_existing={skipped} total={len(files)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
