"""faster-whisper (CTranslate2) adapter - the STT-0 baseline model wrapper.

Model ids: whisper size aliases (tiny, base, small, medium, large-v2, large-v3,
large-v3-turbo, distil-*) or a Hugging Face repo id of a CTranslate2-converted
model. Weights download into the project-local models/ directory; no
credentials are required for public models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from morves_stt.audio import (
    CANONICAL_SAMPLE_RATE,
    inspect_audio,
    prevent_clipping,
    read_audio,
    resample,
    to_mono,
)
from morves_stt.inference import Segment, TranscriptionResult


class ModelLoadError(RuntimeError):
    """Raised when the underlying ASR model cannot be loaded on this machine."""


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to 'cuda' when a CUDA-capable runtime is present, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import ctranslate2  # type: ignore[import-not-found]

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


class FasterWhisperModel:
    """Lazy-loading wrapper around faster_whisper.WhisperModel."""

    def __init__(
        self,
        model_id: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: str | Path = "models",
        default_language: str = "id",
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.download_root = str(download_root)
        self.default_language = default_language
        self._model = None  # lazy: loaded by load() or first transcribe()

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        """Load model weights; raise ModelLoadError with the exact reason on failure."""
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadError(f"faster-whisper is not installed: {exc}") from exc

        Path(self.download_root).mkdir(parents=True, exist_ok=True)
        try:
            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load faster-whisper model '{self.model_id}' "
                f"(device={self.device}, compute_type={self.compute_type}): {exc}"
            ) from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- inference ----------------------------------------------------------

    def _canonical_waveform(self, audio_path: str | Path) -> np.ndarray:
        """Decode input audio to canonical 16 kHz mono float32."""
        inspect_audio(audio_path)
        data, sr = read_audio(audio_path)
        mono = to_mono(np.asarray(data, dtype=np.float32))
        res = resample(mono, sr, CANONICAL_SAMPLE_RATE)
        return np.ascontiguousarray(prevent_clipping(res, peak_limit=0.95))

    def transcribe(
        self, audio_path: str | Path, language: str | None = None
    ) -> TranscriptionResult:
        lang = language or self.default_language
        self.load()
        assert self._model is not None  # narrowed by load()

        waveform = self._canonical_waveform(audio_path)
        info = inspect_audio(audio_path)

        segments_iter, gen_info = self._model.transcribe(
            waveform,
            language=lang,
            beam_size=5,
            vad_filter=True,
        )
        segments = [
            Segment(start_sec=float(s.start), end_sec=float(s.end), text=s.text.strip())
            for s in segments_iter
        ]
        text = " ".join(seg.text for seg in segments if seg.text).strip()

        duration_ms = int(float(getattr(gen_info, "duration", info.duration_sec)) * 1000)
        return TranscriptionResult(
            text=text,
            language=str(getattr(gen_info, "language", lang)),
            duration_ms=duration_ms,
            model_id=self.model_id,
            segments=tuple(segments),
        )
