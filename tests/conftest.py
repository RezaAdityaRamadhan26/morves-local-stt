"""Shared test fixtures. All audio is synthetic (sine/noise) - no private speech."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Ensure repo root is importable (service/ lives outside src/).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def make_sine(tmp_path: Path):
    """Factory producing synthetic WAV files."""

    def _make(
        name: str = "sine.wav",
        sample_rate: int = 44100,
        seconds: float = 1.0,
        channels: int = 2,
        freq: float = 440.0,
        amp: float = 0.5,
        subtype: str = "PCM_16",
    ) -> Path:
        t = np.arange(int(seconds * sample_rate)) / sample_rate
        wave = amp * np.sin(2 * np.pi * freq * t)
        data = np.stack([wave] * channels, axis=1) if channels > 1 else wave
        path = tmp_path / name
        sf.write(str(path), data, sample_rate, subtype=subtype)
        return path

    return _make


@pytest.fixture()
def terms_profile() -> object:
    from morves_stt.metrics import load_term_profile

    return load_term_profile(str(REPO_ROOT / "evaluation" / "fixtures" / "terms.yaml"))
