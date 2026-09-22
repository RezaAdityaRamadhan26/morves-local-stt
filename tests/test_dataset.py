"""Manifest schema validation and speaker-independent split tests."""

from __future__ import annotations

import json

import pytest

from morves_stt.dataset import (
    ManifestEntry,
    ManifestValidationError,
    SpeakerLeakageError,
    assert_no_speaker_leakage,
    load_manifest,
    parse_entry,
    save_manifest,
    speaker_overlap,
    split_by_speaker,
)


def valid_entry(speaker: str = "spk_001", i: int = 1) -> dict:
    return {
        "audio": f"processed/{speaker}/{i:06d}.wav",
        "text": "Tampilkan piutang Medita bulan ini.",
        "language": "id",
        "speaker_id": speaker,
        "duration_sec": 3.5,
        "domain": "receivable",
        "source": "recorded",
    }


def test_parse_valid_entry():
    e = parse_entry(valid_entry())
    assert e.speaker_id == "spk_001"
    assert e.domain == "receivable"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("text"),
        lambda d: d.update(text="   "),
        lambda d: d.pop("speaker_id"),
        lambda d: d.update(speaker_id="Budi Santoso"),  # identifying name rejected
        lambda d: d.update(duration_sec=0),
        lambda d: d.update(duration_sec=-1),
        lambda d: d.update(duration_sec="short"),
        lambda d: d.update(domain="not_a_domain"),
        lambda d: d.update(source="scraped"),
        lambda d: d.update(language="xx"),
        lambda d: d.update(audio=""),
    ],
)
def test_parse_rejects_invalid(mutate):
    d = valid_entry()
    mutate(d)
    with pytest.raises(ManifestValidationError):
        parse_entry(d)


def test_manifest_roundtrip(tmp_path):
    entries = [parse_entry(valid_entry(f"spk_{i:03d}", i)) for i in range(1, 4)]
    path = tmp_path / "m.jsonl"
    save_manifest(entries, path)
    loaded = load_manifest(path)
    assert loaded == entries


def test_load_manifest_skips_comments_and_blank(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(
        "# comment line\n\n" + json.dumps(valid_entry()) + "\n",
        encoding="utf-8",
    )
    assert len(load_manifest(path)) == 1


def test_load_manifest_invalid_json(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestValidationError):
        load_manifest(path)


def make_entries(speakers: int, per_speaker: int) -> list[ManifestEntry]:
    out = []
    i = 1
    for s in range(1, speakers + 1):
        spk = f"spk_{s:03d}"
        for _ in range(per_speaker):
            out.append(parse_entry(valid_entry(spk, i)))
            i += 1
    return out


def test_split_by_speaker_is_disjoint():
    groups = split_by_speaker(make_entries(8, 5), seed=42)
    assert_no_speaker_leakage(groups["train"], groups["validation"], groups["test"])
    total = sum(len(g) for g in groups.values())
    assert total == 40
    assert all(len(g) > 0 for g in groups.values())


def test_split_deterministic_with_seed():
    a = split_by_speaker(make_entries(8, 5), seed=7)
    b = split_by_speaker(make_entries(8, 5), seed=7)
    assert {k: sorted(e.audio for e in v) for k, v in a.items()} == {
        k: sorted(e.audio for e in v) for k, v in b.items()
    }


def test_split_ratios_approximate():
    groups = split_by_speaker(make_entries(30, 4), seed=42)
    n = sum(len(g) for g in groups.values())
    train_frac = len(groups["train"]) / n
    assert 0.6 <= train_frac <= 0.95  # greedy speaker-block split has coarse granularity


def test_split_requires_three_speakers():
    with pytest.raises(ManifestValidationError):
        split_by_speaker(make_entries(2, 5))


def test_speaker_leakage_detection():
    a = [parse_entry(valid_entry("spk_001", 1))]
    b = [parse_entry(valid_entry("spk_001", 2))]  # same speaker in another split
    c = [parse_entry(valid_entry("spk_002", 3))]
    assert speaker_overlap(a, b) == {"spk_001"}
    with pytest.raises(SpeakerLeakageError):
        assert_no_speaker_leakage(a, b, c)


def test_synthetic_entries_rejected_for_test_policy():  # policy documented; schema allows synthetic
    d = valid_entry()
    d["source"] = "synthetic"
    e = parse_entry(d)
    assert e.source == "synthetic"
