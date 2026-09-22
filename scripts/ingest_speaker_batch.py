"""Ingest a raw speaker batch: transcribe -> map -> canonical copies -> normalize -> manifest.

The STT-1B workflow, productized for spk_002..spk_006 batches and prompt
recoveries. Dry-run by default: prints the mapping table only. ``--apply``
performs canonical copies, normalization, and private manifest updates.

Modes:
  mapping (default)  raw files have arbitrary names (e.g. Telegram ids);
                     locally transcribed with the baseline model, then mapped
                     onto the reference prompts with confidence + ambiguity
                     gating (morves_stt.mapping)
  --direct           files are already named <prompt_id>.<ext> (e.g. a
                     re-recorded p017.ogg dropped into a recovery folder);
                     mapping is skipped, validation still runs

Examples:
  python scripts/ingest_speaker_batch.py --speaker spk_002 \
      --input-dir datasets/raw/spk_002/incoming
  python scripts/ingest_speaker_batch.py --speaker spk_001 --direct \
      --input-dir datasets/raw/spk_001/recovery \
      --references datasets/recording-pack/sanity-prompts.csv --apply

All outputs (canonical copies, mapping CSV, processed WAVs, manifests) stay
in gitignored private areas. Nothing here is ever committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from morves_stt.adapters.faster_whisper import FasterWhisperModel  # noqa: E402
from morves_stt.audio import inspect_audio  # noqa: E402
from morves_stt.dataset import parse_entry  # noqa: E402
from morves_stt.mapping import map_batch  # noqa: E402

AUDIO_EXTENSIONS = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".opus", ".aac"}
ACCEPTED_CONFIDENCE = {"HIGH", "MEDIUM"}


def load_references(path: Path) -> dict[str, tuple[str, str]]:
    """prompt_id -> (text, domain) from a prompts CSV (prompt_id[,domain],text)."""
    refs: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("prompt_id") or "").strip()
            text = (row.get("text") or "").strip()
            if pid and text:
                refs[pid] = (text, (row.get("domain") or "general").strip())
    if not refs:
        raise SystemExit(f"no references found in {path}")
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True, help="spk_NNN (pseudonym, never a name)")
    parser.add_argument("--input-dir", required=True, help="folder of raw recordings")
    parser.add_argument(
        "--references",
        default="datasets/recording-pack/pilot-prompts.csv",
        help="prompts CSV with prompt_id,text[,domain] columns",
    )
    parser.add_argument("--model-id", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="id")
    parser.add_argument("--recording-profile", default=None)
    parser.add_argument("--noise-profile", default=None)
    parser.add_argument(
        "--direct", action="store_true", help="files already named <prompt_id>.<ext>; skip mapping"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually copy/normalize/write manifest (default: dry-run)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"input dir not found: {in_dir}")
        return 1
    files = sorted(
        p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        print(f"no audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) in {in_dir}")
        return 1
    refs = load_references(Path(args.references))
    print(f"{len(files)} raw file(s) | {len(refs)} reference prompts | model {args.model_id}")

    # -- transcription / mapping -------------------------------------------
    plans: list[dict] = []  # {file, prompt_id, confidence, transcript, duration}
    if args.direct:
        for f in files:
            pid = f.stem
            if pid not in refs:
                print(f"REJECTED {f.name}: stem is not a known prompt_id")
                return 1
            plans.append(
                {
                    "file": f,
                    "prompt_id": pid,
                    "confidence": "DIRECT",
                    "transcript": "",
                    "duration": inspect_audio(f).duration_sec,
                }
            )
    else:
        model = FasterWhisperModel(
            model_id=args.model_id,
            device=args.device,
            compute_type=args.compute_type,
            download_root=REPO_ROOT / "models",
            default_language=args.language,
        )
        hypotheses: dict[str, str] = {}
        durations: dict[str, float] = {}
        for f in files:
            info = inspect_audio(f)
            durations[f.name] = info.duration_sec
            hypotheses[f.name] = model.transcribe(f, language=args.language).text
            print(f"  {f.name} ({info.duration_sec:.1f}s): {hypotheses[f.name][:60]}")
        ref_texts = {pid: text for pid, (text, _) in refs.items()}
        decisions = map_batch(ref_texts, hypotheses)
        for d in decisions:
            plans.append(
                {
                    "file": in_dir / d.original_filename,
                    "prompt_id": d.prompt_id,
                    "confidence": d.confidence,
                    "transcript": hypotheses[d.original_filename],
                    "duration": durations[d.original_filename],
                    "reason": d.reason,
                }
            )

    print(f"\n{'file':<30}{'prompt':<9}{'conf':<10}{'dur':<6}reason")
    for p in plans:
        print(
            f"{p['file'].name:<30}{p['prompt_id']:<9}{p['confidence']:<10}"
            f"{p['duration']:.1f}s  {p.get('reason', '')}"
        )

    blocked = [p for p in plans if p["confidence"] not in ACCEPTED_CONFIDENCE | {"DIRECT"}]
    usable = [p for p in plans if p["confidence"] in ACCEPTED_CONFIDENCE | {"DIRECT"}]
    if blocked:
        print(f"\nNOT APPLIED (review required): {[p['file'].name for p in blocked]}")
    if not args.apply:
        print(f"\ndry-run: {len(usable)} would be copied+normalized; rerun with --apply")
        return 0 if not blocked else 1

    # -- apply: canonical copies, mapping CSV, normalization, manifest -------
    spk = args.speaker
    canonical_dir = REPO_ROOT / "datasets" / "raw" / spk / "canonical"
    processed_dir = REPO_ROOT / "datasets" / "processed" / spk / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    for p in usable:
        dst = canonical_dir / f"{p['prompt_id']}{p['file'].suffix.lower()}"
        if not dst.exists():
            shutil.copy2(p["file"], dst)

    with (canonical_dir / "mapping.private.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        w = csv.writer(csv_file)
        w.writerow(
            [
                "prompt_id",
                "canonical_filename",
                "original_filename",
                "reference_text",
                "baseline_transcript",
                "mapping_confidence",
                "duration_sec",
            ]
        )
        for p in sorted(plans, key=lambda x: x["prompt_id"]):
            ref_text = refs[p["prompt_id"]][0]
            w.writerow(
                [
                    p["prompt_id"],
                    f"{p['prompt_id']}{p['file'].suffix.lower()}",
                    p["file"].name,
                    ref_text,
                    p["transcript"],
                    p["confidence"],
                    round(p["duration"], 3),
                ]
            )

    for p in usable:
        src = canonical_dir / f"{p['prompt_id']}{p['file'].suffix.lower()}"
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "normalize_audio.py"),
                str(src),
                "--out-dir",
                str(processed_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            print(f"NORMALIZE FAILED {src.name}: {proc.stdout}{proc.stderr}")
            return 1

    manifest_path = REPO_ROOT / "datasets" / "manifests" / f"ingest.{spk}.private.jsonl"
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                existing[obj["audio"]] = obj
    for p in usable:
        wav = f"datasets/processed/{spk}/canonical/{p['prompt_id']}.wav"
        entry = parse_entry(
            {
                "audio": wav,
                "text": refs[p["prompt_id"]][0],
                "language": args.language,
                "speaker_id": spk,
                "duration_sec": round(p["duration"], 3),
                "domain": refs[p["prompt_id"]][1],
                "source": "recorded",
                "recording_profile": args.recording_profile or "",
                "noise_profile": args.noise_profile or "",
            }
        )
        existing[entry.audio] = entry.to_dict()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as mf:
        for audio in sorted(existing):
            mf.write(json.dumps(existing[audio], ensure_ascii=False) + "\n")

    print(f"\napplied: {len(usable)} canonical copies + normalized WAVs")
    print(f"mapping: {canonical_dir / 'mapping.private.csv'}")
    print(f"manifest: {manifest_path} ({len(existing)} entries)")
    return 0 if not blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
