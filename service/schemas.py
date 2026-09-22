"""Pydantic schemas for the local STT service API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    """Liveness: the process is running. Says nothing about model state."""

    status: str = "ok"


class ReadyStatus(BaseModel):
    """Readiness: the ASR model is loaded and the service can transcribe."""

    ready: bool
    model: str | None = None
    reason: str | None = None


class ModelInfo(BaseModel):
    model_id: str
    adapter: str
    device: str
    compute_type: str
    default_language: str


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    durationMs: int = Field(ge=0, description="Audio duration in milliseconds")
    model: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ErrorDetail(BaseModel):
    code: str
    message: str
