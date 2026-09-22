"""Run the baseline ASR benchmark over a manifest and write JSON + Markdown reports.

Quality benchmarking requires REAL speech with references:
  - default manifest: evaluation/private-manifest.jsonl (gitignored, local only)
  - or pass --manifest explicitly

Pipeline smoke test without private audio:
  python scripts/benchmark_baseline.py --smoke
  (uses locally generated synthetic audio; latency-only, NOT a quality result)

No fake numbers are ever produced: without a manifest and without --smoke the
script exits BLOCKED with an explanation.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from morves_stt.adapters.faster_whisper import FasterWhisperModel
from morves_stt.benchmark import run_benchmark, write_reports
from morves_stt.dataset import ManifestEntry, load_manifest
from morves_stt.metrics import load_term_profile

DEFAULT_CONFIG = Path("configs/baseline.yaml")
DEFAULT_PRIVATE_MANIFEST = Path("evaluation/private-manifest.jsonl")


def load_config(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def smoke_entries(count: int, tmp: Path) -> list[ManifestEntry]:
    """Generate synthetic non-speech audio fixtures; references are placeholders.

    This exercises the pipeline and measures latency ONLY. Transcription quality
    of noise is meaningless and MUST NOT be reported as a benchmark.
    """
    rng = np.random.default_rng(42)
    entries: list[ManifestEntry] = []
    domains = ["general", "receivable", "transaction"]
    for i in range(count):
        seconds = float(rng.uniform(1.0, 3.0))
        sr = 16000
        t = np.arange(int(seconds * sr)) / sr
        tone = 0.3 * np.sin(2 * np.pi * (220 + 40 * i) * t)
        noise = 0.05 * rng.standard_normal(len(t)).astype(np.float32)
        wave = (tone + noise).astype(np.float32)

        import soundfile as sf

        path = tmp / f"smoke_{i:03d}.wav"
        sf.write(str(path), wave, sr, subtype="PCM_16")

        entries.append(
            ManifestEntry(
                audio=str(path),
                text="[synthetic-smoke-not-a-reference]",
                language="id",
                speaker_id="spk_000",
                duration_sec=seconds,
                domain=domains[i % len(domains)],
                source="synthetic",
            )
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest path")
    parser.add_argument("--model-id", type=str, default=None, help="Override model id/size")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="Synthetic-audio pipeline smoke test (latency only)"
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model_cfg = dict(cfg.get("model", {}))
    bench_cfg = dict(cfg.get("benchmark", {}))

    if args.model_id:
        model_cfg["model_id"] = args.model_id

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(bench_cfg.get("manifest", DEFAULT_PRIVATE_MANIFEST))
    )

    model = FasterWhisperModel(
        model_id=str(model_cfg.get("model_id", "small")),
        device=str(model_cfg.get("device", "auto")),
        compute_type=str(model_cfg.get("compute_type", "auto")),
        download_root=str(model_cfg.get("download_root", "models")),
        default_language=str(model_cfg.get("default_language", "id")),
    )

    terms_file = str(bench_cfg.get("terms_file", "evaluation/fixtures/terms.yaml"))
    terms_profile = load_term_profile(terms_file)

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="morves-smoke-") as td:
            entries = smoke_entries(int(args.limit or 3), Path(td))
            print(f"SMOKE MODE: {len(entries)} synthetic utterances (latency only, NOT quality)")
            try:
                model.load()
            except Exception as exc:
                print(f"BLOCKED: model could not load: {exc}")
                return 2
            result = run_benchmark(
                model,
                entries,
                audio_root=".",
                terms_profile=terms_profile,
                warmup=int(bench_cfg.get("warmup", 1)),
            )
            json_path, md_path = write_reports(
                result,
                bench_cfg.get("output_dir", "evaluation/reports"),
                meta={
                    "mode": "smoke",
                    "note": "synthetic audio; latency only; NOT a quality benchmark",
                },
            )
            print(f"reports: {json_path} , {md_path}")
            return 0

    if not manifest_path.exists():
        print(
            "BLOCKED: no benchmark manifest available.\n"
            f"  looked for: {manifest_path}\n"
            "  Provide real private speech via --manifest <path> or create\n"
            "  evaluation/private-manifest.jsonl (see evaluation/README.md).\n"
            "  Use --smoke for a pipeline-only run. Quality numbers require real audio."
        )
        return 2

    entries = load_manifest(manifest_path)
    print(f"manifest: {manifest_path} ({len(entries)} utterances)")

    try:
        model.load()
    except Exception as exc:
        print(f"BLOCKED: model could not load: {exc}")
        return 2

    t0 = time.perf_counter()
    result = run_benchmark(
        model,
        entries,
        audio_root=str(bench_cfg.get("audio_root", ".")),
        terms_profile=terms_profile,
        warmup=int(bench_cfg.get("warmup", 1)),
        limit=args.limit,
    )
    elapsed = time.perf_counter() - t0
    print(f"benchmark completed in {elapsed:.1f}s; skipped {len(result.errors)}")

    json_path, md_path = write_reports(
        result,
        bench_cfg.get("output_dir", "evaluation/reports"),
        meta={"mode": "quality", "manifest": str(manifest_path)},
    )
    print(f"reports: {json_path} , {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
