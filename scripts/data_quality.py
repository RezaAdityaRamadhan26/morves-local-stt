"""Per-speaker data-quality analysis over private manifests (STT-1C section 19).

For every speaker reports: usable clips, duration, missing prompts vs a
reference pack, corrupt clips, very short/long clips, clipping-risk and
low-volume warnings (on the normalized WAVs), and mapping-confidence
distribution (from mapping.private.csv when present).

Output is a PRIVATE markdown report under evaluation/reports/ (gitignored).
Never commit it - it may contain private transcript references.

Usage:
  python scripts/data_quality.py --extra-manifest evaluation/private-manifest.jsonl \
      [--references datasets/recording-pack/sanity-prompts.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from morves_stt.audio import AudioValidationError, peak_dbfs, read_audio, rms_dbfs  # noqa: E402
from morves_stt.dataset import load_manifest  # noqa: E402

SHORT_SEC = 1.0
LONG_SEC = 30.0
CLIP_DBFS = -0.1  # normalized peak at/above this = clipping risk
QUIET_DBFS = -35.0  # RMS below this = low-volume warning


def mapping_confidences(speaker: str) -> Counter[str]:
    """Mapping confidence counts from any of the speaker's mapping CSVs.

    Ingest writes datasets/raw/<spk>/canonical/mapping.private.csv; the STT-1B
    sanity batch used datasets/raw/<spk>/sanity-001/mapping.private.csv.
    """
    counts: Counter[str] = Counter()
    raw = REPO_ROOT / "datasets" / "raw" / speaker
    for p in sorted(raw.glob("*/mapping.private.csv")):
        with p.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                counts[row.get("mapping_confidence", "?")] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-manifest", action="append", required=True)
    parser.add_argument(
        "--references", default=None, help="prompts CSV to compute missing prompts per speaker"
    )
    parser.add_argument("--out-dir", default="evaluation/reports")
    args = parser.parse_args()

    entries = []
    for m in args.extra_manifest:
        entries.extend(load_manifest(Path(m)))

    refs: dict[str, str] = {}
    if args.references:
        with open(args.references, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                pid = (row.get("prompt_id") or "").strip()
                if pid:
                    refs[pid] = (row.get("text") or "").strip()

    by_speaker: dict[str, list] = {}
    for e in entries:
        by_speaker.setdefault(e.speaker_id, []).append(e)

    lines = [
        "# STT-1C Data Quality Report (PRIVATE - never commit)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for spk in sorted(by_speaker):
        es = by_speaker[spk]
        total = sum(e.duration_sec for e in es)
        corrupt, very_short, very_long, clipping, quiet = [], [], [], [], []
        for e in es:
            path = REPO_ROOT / e.audio
            try:
                data, _ = read_audio(path)
            except (AudioValidationError, Exception):
                corrupt.append(e.audio)
                continue
            if e.duration_sec < SHORT_SEC:
                very_short.append(e.audio)
            if e.duration_sec > LONG_SEC:
                very_long.append(e.audio)
            pk, rm = peak_dbfs(data), rms_dbfs(data)
            if pk >= CLIP_DBFS:
                clipping.append(e.audio)
            if rm < QUIET_DBFS:
                quiet.append(e.audio)

        prompt_ids = {Path(e.audio).stem for e in es}
        missing = sorted(set(refs) - prompt_ids) if refs else []

        lines += [
            f"## {spk}",
            f"- usable clips: {len(es) - len(corrupt)} / {len(es)}",
            f"- duration: {total / 60:.1f} min",
            f"- corrupt: {len(corrupt)}{corrupt if corrupt else ''}",
            f"- very short (<{SHORT_SEC}s): {len(very_short)}{very_short if very_short else ''}",
            f"- very long (>{LONG_SEC}s): {len(very_long)}{very_long if very_long else ''}",
            f"- clipping risk (peak >= {CLIP_DBFS} dBFS): "
            f"{len(clipping)}{clipping if clipping else ''}",
            f"- low volume (RMS < {QUIET_DBFS} dBFS): {len(quiet)}{quiet if quiet else ''}",
            f"- mapping confidence: {dict(mapping_confidences(spk)) or '(no mapping CSV)'}",
            f"- missing prompts vs references: {len(missing)}" + (f" {missing}" if missing else ""),
            "",
        ]

    out = Path(args.out_dir) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-quality.private.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"quality report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
