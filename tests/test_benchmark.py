"""Benchmark runner tests using the DeterministicModel stub (no downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from morves_stt.audio import inspect_audio
from morves_stt.benchmark import run_benchmark, summarize, write_reports
from morves_stt.dataset import parse_entry
from morves_stt.inference import DeterministicModel, TranscriptionResult
from morves_stt.metrics import TermProfile


def entry(i: int, speaker: str, text: str, path: str) -> dict:
    return {
        "audio": path,
        "text": text,
        "language": "id",
        "speaker_id": speaker,
        "duration_sec": 1.0,
        "domain": "receivable",
        "source": "recorded",
    }


def test_run_benchmark_scores_and_reports(tmp_path):
    # Two synthetic utterances; the stub returns the reference for one, garbage for the other.
    texts = ["tampilkan piutang medita", "tampilkan piutang media"]
    entries = []
    for i, text in enumerate(texts):
        wav = tmp_path / f"utt_{i}.wav"
        t = np.arange(16000) / 16000
        sf.write(
            str(wav),
            (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32),
            16000,
            subtype="PCM_16",
        )
        entries.append(parse_entry(entry(i, f"spk_{i:03d}", text, str(wav))))

    profile = TermProfile(
        entities=["medita"],
        critical_pairs={"medita": {"media"}},
        critical_terms=["medita"],
    )

    class HalfRightModel(DeterministicModel):
        def transcribe(self, audio_path, language="id"):  # type: ignore[override]
            info = inspect_audio(audio_path)
            idx = int(Path(audio_path).stem.split("_")[-1])
            return TranscriptionResult(
                text=texts[idx],
                language="id",
                duration_ms=int(info.duration_sec * 1000),
                model_id="half-right",
            )

    model = HalfRightModel(model_id="half-right")
    result = run_benchmark(model, entries, audio_root=".", terms_profile=profile, warmup=0)

    assert len(result.records) == 2
    assert result.errors == []

    summary = summarize(result)
    assert summary["utterances"] == 2
    assert summary["wer_macro"] == 0.0  # stub echoes the reference exactly
    assert summary["per_domain"]["receivable"]["utterances"] == 2
    assert summary["latency_ms"]["mean"] is not None

    json_path, md_path = write_reports(result, tmp_path / "reports")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["utterances"] == 2
    assert "WER" in md_path.read_text(encoding="utf-8")


def test_run_benchmark_missing_audio_recorded_as_error(tmp_path):
    d = entry(0, "spk_001", "teks", str(tmp_path / "missing.wav"))
    e = parse_entry(d)
    result = run_benchmark(
        DeterministicModel(), [e], audio_root=".", terms_profile=TermProfile(), warmup=0
    )
    assert result.records == []
    assert result.errors and result.errors[0]["reason"] == "file not found"
