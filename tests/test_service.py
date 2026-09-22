"""Service API tests using the DeterministicModel stub (no model download)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from service.main import create_app

from morves_stt.inference import DeterministicModel


def settings(**overrides) -> dict:
    svc = {
        "max_upload_mb": 5,
        "max_audio_duration_sec": 30,
        "allowed_extensions": [".wav"],
        "allowed_mime_prefixes": ["audio/", "application/octet-stream"],
        "log_transcripts": False,
    }
    svc.update(overrides)
    return {"service": svc, "model": {"adapter": "deterministic", "model_id": "stub"}}


def wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    wave = 0.4 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


@pytest.fixture()
def client():
    app = create_app(settings(), DeterministicModel(text="tampilkan piutang medita"))
    with TestClient(app) as tc:  # context manager runs lifespan (model load)
        yield tc


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_ok(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True


def test_ready_503_when_model_fails():
    class BrokenModel(DeterministicModel):
        def load(self) -> None:  # type: ignore[override]
            raise RuntimeError("weights unavailable")

    app = create_app(settings(), BrokenModel())
    with TestClient(app) as tc:
        r = tc.get("/ready")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False
        assert "weights unavailable" in body["reason"]
        # health stays 200 even when model cannot load - liveness != readiness
        assert tc.get("/health").status_code == 200


def test_model_info(client):
    r = client.get("/v1/model")
    assert r.status_code == 200
    assert r.json()["adapter"] == "deterministic"


def test_transcription_happy_path(client):
    r = client.post(
        "/v1/transcriptions",
        files={"file": ("utt.wav", wav_bytes(1.5), "audio/wav")},
        data={"language": "id"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "tampilkan piutang medita"
    assert body["language"] == "id"
    assert body["model"] == "deterministic-stub"
    assert 1200 <= body["durationMs"] <= 2000


def test_transcription_rejects_non_wav_extension(client):
    r = client.post(
        "/v1/transcriptions",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_format"


def test_transcription_rejects_bad_content_type(client):
    r = client.post(
        "/v1/transcriptions",
        files={"file": ("utt.wav", wav_bytes(0.5), "text/plain")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_transcription_rejects_empty_upload(client):
    r = client.post(
        "/v1/transcriptions",
        files={"file": ("utt.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_upload"


def test_transcription_rejects_corrupt_wav(client):
    r = client.post(
        "/v1/transcriptions",
        files={"file": ("utt.wav", b"garbage-not-a-wav", "audio/wav")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_audio"


def test_transcription_oversize_rejected():
    app = create_app(
        settings(max_upload_mb=0.001),  # ~1 KB limit
        DeterministicModel(text="x"),
    )
    with TestClient(app) as tc:
        r = tc.post(
            "/v1/transcriptions",
            files={"file": ("utt.wav", wav_bytes(1.0), "audio/wav")},  # ~32 KB
        )
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "payload_too_large"


def test_transcription_duration_limit():
    app = create_app(
        settings(max_audio_duration_sec=1.0),
        DeterministicModel(text="x"),
    )
    with TestClient(app) as tc:
        r = tc.post(
            "/v1/transcriptions",
            files={"file": ("utt.wav", wav_bytes(2.5), "audio/wav")},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "audio_too_long"


def test_temp_files_cleaned_up(tmp_path):
    # Point the service temp dir at tmp_path; nothing must survive the request.
    app = create_app(settings(temp_dir=str(tmp_path)), DeterministicModel(text="x"))
    with TestClient(app) as tc:
        r = tc.post(
            "/v1/transcriptions",
            files={"file": ("utt.wav", wav_bytes(0.5), "audio/wav")},
        )
    assert r.status_code == 200
    residue = list(tmp_path.iterdir())
    assert residue == [], f"temp files leaked: {residue}"
