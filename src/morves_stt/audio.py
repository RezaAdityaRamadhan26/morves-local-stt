"""Audio inspection, validation, and canonical normalization.

Canonical profile:
  - Container: WAV
  - Channels: 1 (mono)
  - Sample rate: 16,000 Hz
  - Encoding: PCM 16-bit little-endian
  - Safe peak scaling to avoid clipping
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

CANONICAL_SAMPLE_RATE = 16000
CANONICAL_CHANNELS = 1
CANONICAL_SUBTYPE = "PCM_16"
MAX_SAFE_DURATION_SEC = 600.0  # 10 minutes


class AudioValidationError(Exception):
    """Raised when audio fails inspection or cannot be read safely."""


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    duration_sec: float
    frames: int
    format: str
    subtype: str

    @property
    def is_canonical(self) -> bool:
        return (
            self.sample_rate == CANONICAL_SAMPLE_RATE
            and self.channels == CANONICAL_CHANNELS
            and self.subtype == CANONICAL_SUBTYPE
        )


def inspect_audio(path: str | Path) -> AudioInfo:
    """Inspect audio metadata without loading entire waveform into memory."""
    p = Path(path)
    if not p.exists():
        raise AudioValidationError(f"Audio file does not exist: {p}")
    if p.stat().st_size == 0:
        raise AudioValidationError(f"Audio file is empty (0 bytes): {p}")

    try:
        info = sf.info(str(p))
    except Exception as exc:
        raise AudioValidationError(f"Cannot read audio header from {p}: {exc}") from exc

    if info.duration <= 0:
        raise AudioValidationError(f"Invalid non-positive duration ({info.duration}s): {p}")
    if info.duration > MAX_SAFE_DURATION_SEC:
        raise AudioValidationError(
            f"Duration ({info.duration:.1f}s) exceeds maximum safe limit "
            f"({MAX_SAFE_DURATION_SEC}s): {p}"
        )

    return AudioInfo(
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_sec=float(info.duration),
        frames=info.frames,
        format=info.format,
        subtype=info.subtype,
    )


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read audio into a float32 numpy array normalized to [-1.0, 1.0]."""
    p = Path(path)
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioValidationError(f"Failed to decode audio {p}: {exc}") from exc
    return data, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono via averaging."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise AudioValidationError(f"Unsupported audio array dimension: {audio.ndim}")


def resample(
    audio: np.ndarray, source_sr: int, target_sr: int = CANONICAL_SAMPLE_RATE
) -> np.ndarray:
    """Resample 1D float32 audio to target sample rate using soxr."""
    if source_sr == target_sr:
        return audio
    resampled = soxr.resample(audio, source_sr, target_sr, quality="HQ")
    return resampled.astype(np.float32)


def prevent_clipping(audio: np.ndarray, peak_limit: float = 0.95) -> np.ndarray:
    """Scale audio down if peak exceeds peak_limit to avoid clipping."""
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > peak_limit:
        scale = peak_limit / peak
        return audio * scale
    return audio


def peak_dbfs(audio: np.ndarray) -> float:
    """Peak level in dBFS (-inf for digital silence)."""
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    return 20.0 * float(np.log10(peak)) if peak > 0 else float("-inf")


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS level in dBFS (-inf for digital silence)."""
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * float(np.log10(rms)) if rms > 0 else float("-inf")


def normalize_to_canonical_wav(
    input_path: str | Path,
    output_path: str | Path,
    overwrite: bool = True,
) -> AudioInfo:
    """Decode any supported audio and write canonical 16 kHz mono 16-bit PCM WAV.

    Never renames files blindly; decodes, resamples, averages channels, and writes
    clean linear PCM.
    """
    inp = Path(input_path)
    out = Path(output_path)

    inspect_audio(inp)  # pre-validate
    audio, sr = read_audio(inp)
    mono = to_mono(audio)
    resampled = resample(mono, sr, CANONICAL_SAMPLE_RATE)
    safe = prevent_clipping(resampled, peak_limit=0.95)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {out}")

    sf.write(str(out), safe, CANONICAL_SAMPLE_RATE, subtype=CANONICAL_SUBTYPE)
    return inspect_audio(out)
