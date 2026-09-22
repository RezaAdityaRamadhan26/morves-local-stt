"""Adapter selection logic tests (no model download, pure functions only)."""

from __future__ import annotations

from morves_stt.adapters.faster_whisper import resolve_compute_type
from morves_stt.adapters.faster_whisper import resolve_device as rd


def test_resolve_device_explicit_passthrough(monkeypatch):
    assert rd("cpu") == "cpu"
    assert rd("cuda") == "cuda"


def test_resolve_device_auto_cpu_when_no_cuda(monkeypatch):
    import morves_stt.adapters.faster_whisper as mod

    class FakeCT2:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 0

    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", FakeCT2)
    assert mod.resolve_device("auto") == "cpu"


def test_resolve_device_auto_cuda_when_present(monkeypatch):
    import sys

    import morves_stt.adapters.faster_whisper as mod

    class FakeCT2:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 1

    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCT2)
    assert mod.resolve_device("auto") == "cuda"


def test_resolve_device_auto_survives_missing_ctranslate2(monkeypatch):
    import sys

    import morves_stt.adapters.faster_whisper as mod

    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    assert mod.resolve_device("auto") == "cpu"


def test_resolve_compute_type():
    assert resolve_compute_type("auto", "cpu") == "int8"
    assert resolve_compute_type("auto", "cuda") == "float16"
    assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"
