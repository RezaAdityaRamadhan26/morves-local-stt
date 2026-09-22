"""Audio inspection and canonical normalization tests (synthetic signals only)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

from morves_stt.audio import (
    CANONICAL_SAMPLE_RATE,
    AudioValidationError,
    inspect_audio,
    normalize_to_canonical_wav,
    read_audio,
    to_mono,
)


def test_inspect_reports_metadata(make_sine):
    p = make_sine("stereo.wav", sample_rate=44100, channels=2)
    info = inspect_audio(p)
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert 0.9 < info.duration_sec < 1.1
    assert not info.is_canonical


def test_normalize_produces_canonical_wav(make_sine, tmp_path):
    src = make_sine("src.flac", sample_rate=44100, channels=2)
    dst = tmp_path / "out" / "src.wav"
    info = normalize_to_canonical_wav(src, dst)
    assert info.is_canonical
    assert info.sample_rate == CANONICAL_SAMPLE_RATE
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    # duration preserved within tolerance
    assert abs(info.duration_sec - 1.0) < 0.05


def test_normalize_mono_48k_passthrough_channels(make_sine, tmp_path):
    src = make_sine("mono48.wav", sample_rate=48000, channels=1)
    dst = tmp_path / "mono16.wav"
    info = normalize_to_canonical_wav(src, dst)
    assert info.sample_rate == 16000 and info.channels == 1
    data, sr = read_audio(dst)
    assert sr == 16000
    assert data.ndim == 1


def test_normalize_flac_input(make_sine, tmp_path):
    src = make_sine("speech.flac", sample_rate=44100, channels=1)
    dst = tmp_path / "speech.wav"
    info = normalize_to_canonical_wav(src, dst)
    assert info.format == "WAV"
    assert dst.exists()


def test_peak_guard_prevents_clipping(tmp_path):
    # Loud signal whose resampling ripple can exceed 1.0.
    sr = 44100
    t = np.arange(sr) / sr
    loud = (1.5 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    src = tmp_path / "loud.wav"
    sf.write(str(src), loud, sr, subtype="FLOAT")
    dst = tmp_path / "loud_canon.wav"
    normalize_to_canonical_wav(src, dst)
    data, _ = read_audio(dst)
    assert float(np.max(np.abs(data))) <= 1.0


def test_corrupt_audio_raises(tmp_path):
    bad = tmp_path / "corrupt.wav"
    bad.write_bytes(b"this is not audio data at all")
    with pytest.raises(AudioValidationError):
        inspect_audio(bad)
    with pytest.raises(AudioValidationError):
        normalize_to_canonical_wav(bad, tmp_path / "out.wav")


def test_missing_file_raises(tmp_path):
    with pytest.raises(AudioValidationError):
        inspect_audio(tmp_path / "does_not_exist.wav")


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioValidationError):
        inspect_audio(empty)


def test_oversized_duration_rejected(make_sine, tmp_path):
    # 601 s of silence would be huge; craft via header trick is overkill -
    # instead verify the guard using a monkeypatched limit.
    from morves_stt import audio as audio_mod

    p = make_sine("short.wav")
    orig = audio_mod.MAX_SAFE_DURATION_SEC
    audio_mod.MAX_SAFE_DURATION_SEC = 0.0001
    try:
        with pytest.raises(AudioValidationError):
            inspect_audio(p)
    finally:
        audio_mod.MAX_SAFE_DURATION_SEC = orig


def test_unsupported_format_clear_error(tmp_path):
    if shutil.which("ffmpeg"):
        pytest.skip("ffmpeg present; m4a may decode via fallback")
    fake = tmp_path / "clip.m4a"
    fake.write_bytes(b"\x00\x00\x00 ftypM4A garbage")
    with pytest.raises(AudioValidationError):
        normalize_to_canonical_wav(fake, tmp_path / "out.wav")


def test_to_mono_shapes():
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]])
    assert np.allclose(to_mono(stereo), [0.0, 0.5])
    mono = np.array([1.0, 2.0])
    assert np.allclose(to_mono(mono), mono)
