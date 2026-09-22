"""Minimal local transcription service.

Contract:
  GET  /health            liveness only
  GET  /ready             model loaded and able to transcribe (503 otherwise)
  GET  /v1/model          model metadata
  POST /v1/transcriptions multipart/form-data WAV upload -> transcript JSON

Privacy:
  - never logs raw or base64 audio (hard rule, not configurable)
  - transcript logging disabled by default
  - uploads are written to a temp file that is always deleted (finally block)
  - bounded file size, duration, and MIME/extension allowlist
  - no URL fetching, no shell execution, no database access, no business logic
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse

from morves_stt.audio import (
    AudioValidationError,
    inspect_audio,
    normalize_to_canonical_wav,
)
from morves_stt.inference import SpeechToTextModel
from service.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthStatus,
    ModelInfo,
    ReadyStatus,
    TranscriptionResponse,
)

logger = logging.getLogger("morves_stt.service")

# Hard privacy constants (not configuration): audio payloads must never be logged.
_FORBIDDEN_LOG_KEYS = ("audio", "file", "content", "data")


@contextlib.contextmanager
def _temp_path(suffix: str, tmp_dir: str | None) -> Generator[Path, None, None]:
    """Context manager guaranteeing temp file cleanup."""
    fd, name = tempfile.mkstemp(suffix=suffix, dir=tmp_dir)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def _safe_log(payload: dict[str, Any]) -> None:
    """Log request metadata only; strips anything resembling audio content."""
    safe = {k: v for k, v in payload.items() if k not in _FORBIDDEN_LOG_KEYS}
    logger.info("request %s", json.dumps(safe, ensure_ascii=False))


def create_app(
    settings: dict[str, Any],
    model: SpeechToTextModel,
) -> FastAPI:
    """Build the FastAPI app.

    ``settings`` comes from configs/inference.yaml. Tests inject a
    DeterministicModel as ``model`` to avoid any real download.
    """
    svc = settings.get("service", {})
    model_cfg = settings.get("model", {})

    max_upload_bytes = int(float(svc.get("max_upload_mb", 25)) * 1024 * 1024)
    max_duration_sec = float(svc.get("max_audio_duration_sec", 120))
    allowed_ext = {
        e.lower() if e.startswith(".") else f".{e.lower()}"
        for e in svc.get("allowed_extensions", [".wav"])
    }
    allowed_mime = tuple(svc.get("allowed_mime_prefixes", ["audio/", "application/octet-stream"]))
    log_transcripts = bool(svc.get("log_transcripts", False))
    tmp_dir = svc.get("temp_dir")

    runtime: dict[str, Any] = {"ready": False, "model_error": None}

    def _load_model() -> bool:
        try:
            load = getattr(model, "load", None)
            if callable(load):
                load()
            runtime["ready"] = True
            runtime["model_error"] = None
            return True
        except Exception as exc:
            runtime["ready"] = False
            runtime["model_error"] = str(exc)
            return False

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        _load_model()
        yield

    app = FastAPI(
        title="Morves Local STT",
        version="0.1.0",
        description=(
            "Local Indonesian ASR. Audio in, transcript out. "
            "No chat, no finance logic, no database."
        ),
        lifespan=_lifespan,
    )
    app.state.model = model
    app.state.runtime = runtime

    def _err(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(),
        )

    @app.get("/health", response_model=HealthStatus)
    def health() -> HealthStatus:
        # Liveness: the process is up. Model state is intentionally NOT considered.
        return HealthStatus(status="ok")

    @app.get("/ready", response_model=ReadyStatus)
    def ready() -> Any:
        if not runtime["ready"] and not _load_model():
            return JSONResponse(
                status_code=503,
                content=ReadyStatus(ready=False, reason=runtime["model_error"]).model_dump(),
            )
        return ReadyStatus(ready=True, model=getattr(model, "model_id", None))

    @app.get("/v1/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        return ModelInfo(
            model_id=str(model_cfg.get("model_id", getattr(model, "model_id", "unknown"))),
            adapter=str(model_cfg.get("adapter", "faster_whisper")),
            device=str(model_cfg.get("device", "auto")),
            compute_type=str(model_cfg.get("compute_type", "auto")),
            default_language=str(model_cfg.get("default_language", "id")),
        )

    @app.post("/v1/transcriptions", response_model=TranscriptionResponse)
    def transcribe(file: UploadFile, language: str = Form(default="id")) -> Any:
        # --- validate extension / content type ---------------------------------
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in allowed_ext:
            return _err(
                415,
                "unsupported_format",
                f"Unsupported file extension '{ext}'. Allowed: {sorted(allowed_ext)}. "
                "Canonical input is WAV PCM mono 16 kHz 16-bit.",
            )
        ctype = (file.content_type or "").lower()
        if not ctype.startswith(tuple(p.lower() for p in allowed_mime)):
            return _err(
                415,
                "unsupported_media_type",
                f"Content-Type '{ctype}' is not an accepted audio type.",
            )

        # --- bounded read -------------------------------------------------------
        payload = file.file.read(max_upload_bytes + 1)
        if len(payload) > max_upload_bytes:
            return _err(
                413, "payload_too_large", f"Upload exceeds {svc.get('max_upload_mb', 25)} MB limit."
            )
        if not payload:
            return _err(400, "empty_upload", "Uploaded file is empty.")

        _safe_log({"filename": filename, "bytes": len(payload), "content_type": ctype})

        # --- canonicalize + transcribe via temp files (always cleaned up) -------
        with _temp_path(".in" + (ext or ".bin"), tmp_dir) as raw_in:
            raw_in.write_bytes(payload)
            try:
                info = inspect_audio(raw_in)
                if info.duration_sec > max_duration_sec:
                    return _err(
                        400,
                        "audio_too_long",
                        f"Audio duration {info.duration_sec:.1f}s exceeds "
                        f"limit {max_duration_sec:.0f}s.",
                    )
                with _temp_path(".canon.wav", tmp_dir) as canonical:
                    normalize_to_canonical_wav(raw_in, canonical)
                    result = model.transcribe(canonical, language=language or "id")
            except AudioValidationError as exc:
                return _err(400, "invalid_audio", str(exc))

        if log_transcripts:
            logger.info("transcript %s", json.dumps({"text": result.text}, ensure_ascii=False))

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            durationMs=result.duration_ms,
            model=result.model_id,
        )

    return app


def build_model_from_config(model_cfg: dict[str, Any]) -> SpeechToTextModel:
    """Construct the model adapter named in config. Deterministic stub available for dev."""
    adapter = model_cfg.get("adapter", "faster_whisper")
    if adapter == "deterministic":
        from morves_stt.inference import DeterministicModel

        return DeterministicModel(model_id=str(model_cfg.get("model_id", "deterministic-stub")))
    from morves_stt.adapters.faster_whisper import FasterWhisperModel

    return FasterWhisperModel(
        model_id=str(model_cfg.get("model_id", "base")),
        device=str(model_cfg.get("device", "auto")),
        compute_type=str(model_cfg.get("compute_type", "auto")),
        download_root=str(model_cfg.get("download_root", "models")),
        default_language=str(model_cfg.get("default_language", "id")),
    )


def load_settings(config_path: str | Path = "configs/inference.yaml") -> dict[str, Any]:
    """Load inference settings; empty dict if the file is absent."""
    import yaml

    p = Path(config_path)
    settings: dict[str, Any] = {}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
    return settings


def main() -> None:  # pragma: no cover - manual entry point
    """Run: python -m service.main"""
    import uvicorn

    settings = load_settings()
    model = build_model_from_config(settings.get("model", {}))
    app = create_app(settings, model)
    svc = settings.get("service", {})
    uvicorn.run(app, host=svc.get("host", "127.0.0.1"), port=int(svc.get("port", 8000)))


if __name__ == "__main__":  # pragma: no cover
    main()
