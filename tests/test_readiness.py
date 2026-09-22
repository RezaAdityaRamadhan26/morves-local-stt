"""Pilot dataset readiness logic tests (pure functions, no filesystem)."""

from __future__ import annotations

import pytest

from morves_stt.dataset import DOMAIN_TAXONOMY, ManifestEntry
from morves_stt.readiness import (
    NOT_READY_VERDICT,
    READY_VERDICT,
    collect_stats,
    evaluate_readiness,
)


def make_entry(
    speaker: str = "spk_001",
    domain: str = "general",
    duration: float = 60.0,
    source: str = "recorded",
    language: str = "id",
    recording_profile: str | None = "laptop_quiet",
    noise_profile: str | None = "quiet",
) -> ManifestEntry:
    return ManifestEntry(
        audio=f"processed/{speaker}/000001.wav",
        text="contoh",
        language=language,
        speaker_id=speaker,
        duration_sec=duration,
        domain=domain,
        source=source,
        recording_profile=recording_profile,
        noise_profile=noise_profile,
    )


def build_full_pilot(offset: int = 0) -> list[ManifestEntry]:
    """4 speakers x 15 min, every domain covered, all real speech.

    ``offset`` shifts speaker numbering so multiple batches add new speakers.
    """
    entries: list[ManifestEntry] = []
    speakers = [f"spk_{i + offset:03d}" for i in (1, 2, 3, 4)]
    for i, spk in enumerate(speakers):
        # 15 utterances of 60 s = 15 min per speaker -> 1.0 h total.
        for j, domain in enumerate(DOMAIN_TAXONOMY):
            entries.append(
                make_entry(
                    speaker=spk,
                    domain=domain,
                    duration=60.0,
                    recording_profile="laptop_quiet" if i % 2 == 0 else "phone_quiet",
                    noise_profile="quiet" if j % 2 == 0 else "ambient",
                )
            )
    return entries


def test_collect_stats_aggregates_everything():
    entries = build_full_pilot()
    stats = collect_stats(entries, test_entries=entries[-15:])
    assert stats.utterances == 60
    assert stats.speaker_ids == ["spk_001", "spk_002", "spk_003", "spk_004"]
    assert stats.total_hours == pytest.approx(1.0)
    assert stats.per_speaker_hours["spk_001"] == pytest.approx(0.25)
    assert stats.per_speaker_utterances["spk_002"] == 15
    assert set(stats.domain_counts) == set(DOMAIN_TAXONOMY)
    assert stats.recording_profiles == {"laptop_quiet": 30, "phone_quiet": 30}
    # 15 domains -> odd split 8/7 per speaker -> 32/28 across 4 speakers.
    assert stats.noise_profiles == {"ambient": 28, "quiet": 32}
    assert stats.language_counts == {"id": 60}
    assert stats.synthetic_in_test == 0
    assert stats.missing_audio == []


def test_collect_stats_flags_synthetic_test_and_passthrough_lists():
    entries = build_full_pilot()
    synth = make_entry(source="synthetic")
    stats = collect_stats(
        entries,
        test_entries=[synth],
        missing_audio=["processed/spk_001/missing.wav"],
        speaker_overlap=["train and test share spk_001"],
        schema_errors=["train.jsonl: bad row"],
    )
    assert stats.synthetic_in_test == 1
    assert stats.missing_audio == ["processed/spk_001/missing.wav"]
    assert stats.speaker_overlap == ["train and test share spk_001"]
    assert stats.schema_errors == ["train.jsonl: bad row"]


def test_ready_dataset_passes():
    stats = collect_stats(build_full_pilot())
    result = evaluate_readiness(stats)
    assert result.ready is True
    assert result.verdict == READY_VERDICT
    assert result.deficits == []


def test_empty_dataset_reports_waiting_for_human_recordings():
    stats = collect_stats([])
    result = evaluate_readiness(stats)
    assert result.ready is False
    assert result.verdict == NOT_READY_VERDICT
    assert any("WAITING FOR HUMAN RECORDINGS" in d for d in result.deficits)
    assert any("speakers" in d for d in result.deficits)
    assert any("total duration" in d for d in result.deficits)


def test_missing_domain_is_a_deficit():
    entries = [e for e in build_full_pilot() if e.domain != "currency_fx"]
    result = evaluate_readiness(collect_stats(entries))
    assert result.ready is False
    assert any("currency_fx" in d for d in result.deficits)


def test_too_few_and_too_many_speakers():
    entries = [e for e in build_full_pilot() if e.speaker_id in ("spk_001", "spk_002")]
    result = evaluate_readiness(collect_stats(entries))
    assert result.ready is False
    assert any("speakers: 2" in d for d in result.deficits)

    many = build_full_pilot() + build_full_pilot(offset=4)
    stats = collect_stats(many)
    assert len(stats.speaker_ids) == 8
    result2 = evaluate_readiness(stats)
    assert result2.ready is False
    assert any("pilot scope" in d for d in result2.deficits)


def test_thin_speaker_is_a_deficit():
    entries = [*build_full_pilot(), make_entry(speaker="spk_005", duration=300.0)]
    result = evaluate_readiness(collect_stats(entries))
    assert result.ready is False
    assert any("spk_005" in d and "min" in d for d in result.deficits)


def test_synthetic_test_missing_audio_leakage_schema_block():
    entries = build_full_pilot()
    stats = collect_stats(
        entries,
        test_entries=[make_entry(source="synthetic")],
        missing_audio=["processed/spk_001/x.wav"],
        speaker_overlap=["leak"],
        schema_errors=["err"],
    )
    result = evaluate_readiness(stats)
    assert result.ready is False
    joined = " | ".join(result.deficits)
    assert "synthetic" in joined
    assert "missing audio" in joined
    assert "leakage" in joined
    assert "schema error" in joined


def test_floors_are_overridable():
    entries = [make_entry(duration=60.0)]  # 1 speaker, 1 min, general only
    result = evaluate_readiness(
        collect_stats(entries),
        min_speakers=1,
        max_speakers=1,
        min_total_hours=0.0,
        min_per_speaker_minutes=0.0,
    )
    # Only domain coverage is unmet now.
    assert result.ready is False
    assert len(result.deficits) == 1
    assert "domains not covered" in result.deficits[0]
