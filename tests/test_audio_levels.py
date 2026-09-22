"""Loudness helper tests (peak/RMS dBFS)."""

from __future__ import annotations

import numpy as np

from morves_stt.audio import peak_dbfs, rms_dbfs


def test_peak_and_rms_of_sine():
    t = np.linspace(0.0, 0.1, 1600, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # peak 0.5 -> ~-6.02 dBFS
    assert abs(peak_dbfs(tone) - (-6.0206)) < 0.01
    # RMS of a sine at amp 0.5 is 0.5/sqrt(2) ~ -9.03 dBFS
    assert abs(rms_dbfs(tone) - (-9.03)) < 0.05


def test_silence_is_neg_inf():
    silence = np.zeros(1000, dtype=np.float32)
    assert peak_dbfs(silence) == float("-inf")
    assert rms_dbfs(silence) == float("-inf")


def test_empty_is_neg_inf():
    assert peak_dbfs(np.zeros(0, dtype=np.float32)) == float("-inf")
    assert rms_dbfs(np.zeros(0, dtype=np.float32)) == float("-inf")
