"""SpeechToTextModel abstraction shared by adapters, the benchmark runner, and the service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from morves_stt.audio import AudioValidationError, inspect_audio


@dataclass(frozen=True)
class Segment:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    duration_ms: int
    model_id: str
    segments: tuple[Segment, ...] = field(default_factory=tuple)


@runtime_checkable
class SpeechToTextModel(Protocol):
    """Conceptual ASR contract: audio file in, transcript out. Nothing else."""

    model_id: str

    def transcribe(self, audio_path: str | Path, language: str = "id") -> TranscriptionResult: ...


class DeterministicModel:
    """Test double returning a fixed transcript; requires no model download.

    Used by unit tests and as a service fallback target in dev configurations.
    Never use for quality claims.
    """

    def __init__(self, text: str = "", model_id: str = "deterministic-stub") -> None:
        self.text = text
        self.model_id = model_id

    def transcribe(self, audio_path: str | Path, language: str = "id") -> TranscriptionResult:
        info = inspect_audio(audio_path)  # raises AudioValidationError on bad input
        return TranscriptionResult(
            text=self.text,
            language=language,
            duration_ms=int(info.duration_sec * 1000),
            model_id=self.model_id,
        )


__all__ = [
    "AudioValidationError",
    "DeterministicModel",
    "Segment",
    "SpeechToTextModel",
    "TranscriptionResult",
]
