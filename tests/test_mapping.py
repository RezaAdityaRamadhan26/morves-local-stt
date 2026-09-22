"""Transcription-assisted mapping policy tests (pure functions)."""

from __future__ import annotations

from morves_stt.mapping import assign_prompts, build_cost_matrix, map_batch

REFS = {
    "p001": "Tampilkan piutang Medita bulan ini.",
    "p002": "Berapa total utang Morves yang jatuh tempo minggu ini?",
    "p003": "Coba lihat pendapatan Creatifyl bulan September.",
}


def test_perfect_transcript_maps_high():
    hyps = {"a.ogg": REFS["p001"], "b.ogg": REFS["p002"], "c.ogg": REFS["p003"]}
    decisions = {d.original_filename: d for d in map_batch(REFS, hyps)}
    assert all(d.confidence == "HIGH" for d in decisions.values())
    assert decisions["a.ogg"].prompt_id == "p001"
    assert decisions["b.ogg"].prompt_id == "p002"
    assert decisions["c.ogg"].prompt_id == "p003"


def test_casual_variation_still_maps():
    # User speech variation: 'Tampilin' instead of 'Tampilkan' (STT-1B section 8)
    hyps = {"a.ogg": "Tampilin piutang Medita bulan ini."}
    (d,) = map_batch(REFS, hyps)
    assert d.prompt_id == "p001"
    assert d.confidence in {"HIGH", "MEDIUM"}


def test_critical_terms_keep_distinct_prompts_separate():
    # piutang vs utang and Medita vs Morves differences must not collapse
    hyps = {
        "a.ogg": "Tampilkan piutang Medita bulan ini.",
        "b.ogg": "Berapa total utang Morves yang jatuh tempo minggu ini?",
    }
    decisions = {d.original_filename: d.prompt_id for d in map_batch(REFS, hyps)}
    assert decisions == {"a.ogg": "p001", "b.ogg": "p002"}


def test_unrelated_take_gets_low_confidence():
    hyps = {"test.ogg": "pastis pastis pastis"}
    (d,) = map_batch(REFS, hyps)
    assert d.confidence == "LOW"
    assert d.reason.startswith("assigned elsewhere") or d.cost > 0.5


def test_duplicate_take_one_wins_one_flags_low():
    # two near-identical takes of p001: one maps, the other must surface as LOW
    # ("assigned elsewhere") so ingest never canonically renames both.
    hyps = {
        "a.ogg": "Tampilkan piutang Medita bulan ini.",
        "a2.ogg": "Tampilkan piutang Medita bulan ini",
    }
    decisions = map_batch(REFS, hyps)
    accepted = [d for d in decisions if d.confidence in {"HIGH", "MEDIUM"}]
    flagged = [d for d in decisions if d.confidence == "LOW"]
    assert len(accepted) == 1
    assert accepted[0].prompt_id == "p001"
    assert len(flagged) == 1
    assert flagged[0].prompt_id == "p001"
    assert "assigned elsewhere" in flagged[0].reason


def test_assign_prompts_greedy_global():
    cost = {
        "f1": {"p1": 0.1, "p2": 0.2},
        "f2": {"p1": 0.05, "p2": 0.9},  # f2 wants p1 more, but global cost lower otherwise
    }
    out = assign_prompts(cost)
    assert out["f2"] == "p1"
    assert out["f1"] == "p2"


def test_build_cost_matrix_normalized():
    cost = build_cost_matrix(REFS, {"a.ogg": REFS["p001"].upper() + "  "})
    assert cost["a.ogg"]["p001"] == 0.0
