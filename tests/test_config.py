"""Config loader tests."""

from __future__ import annotations

from typing import Any

from morves_stt.config import load_yaml, merge_configs


def test_load_yaml_missing_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_ok(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("audio:\n  target_sample_rate: 16000\n", encoding="utf-8")
    assert load_yaml(p) == {"audio": {"target_sample_rate": 16000}}


def test_merge_configs_nested():
    base: dict[str, Any] = {"audio": {"sr": 16000, "ch": 1}, "model": {"id": "base"}}
    over = {"audio": {"sr": 8000}, "extra": True}
    merged = merge_configs(base, over)
    assert merged["audio"] == {"sr": 8000, "ch": 1}
    assert merged["extra"] is True
    assert base["audio"]["sr"] == 16000  # base not mutated
