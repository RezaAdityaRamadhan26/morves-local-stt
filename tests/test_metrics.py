"""WER / CER / domain metric / critical-confusion tests."""

from __future__ import annotations

import pytest

from morves_stt.metrics import (
    TermProfile,
    align,
    cer,
    count_term_occurrences,
    evaluate_utterance,
    score_terms,
    wer,
)
from morves_stt.normalize import tokenize


def test_wer_identical():
    r = wer("Tampilkan piutang Medita bulan ini.", "tampilkan piutang medita bulan ini")
    assert r.wer == 0.0
    assert r.errors == 0


def test_wer_substitution_counts():
    r = wer("Tampilkan piutang Medita", "Tampilkan piutang media")
    assert r.alignment.substitutions == 1
    assert r.alignment.deletions == 0
    assert r.alignment.insertions == 0
    assert r.wer == pytest.approx(1 / 3)


def test_wer_insertion_and_deletion():
    r = wer("satu dua tiga", "satu dua dua tiga")  # insertion
    assert r.alignment.insertions == 1
    assert r.wer == pytest.approx(1 / 3)
    r2 = wer("satu dua tiga empat", "satu dua tiga")  # deletion
    assert r2.alignment.deletions == 1
    assert r2.wer == pytest.approx(1 / 4)


def test_cer_basic():
    assert cer("abcde", "abcde") == 0.0
    assert cer("abcde", "abxde") == pytest.approx(1 / 5)


def test_cer_ignores_punctuation_and_case():
    assert cer("Piutang!", "piutang") == 0.0


def test_align_pairs_exposed():
    a = align(["medita", "bulan"], ["media", "bulan"])
    assert a.substitution_pairs == (("medita", "media"),)


def test_count_term_occurrences_multiword():
    toks = tokenize("neraca saldo dan neraca saldo lagi")
    assert count_term_occurrences(toks, "neraca saldo") == 2
    assert count_term_occurrences(toks, "neraca") == 2
    assert count_term_occurrences(toks, "laba rugi") == 0


def test_score_terms_bounded():
    ref = tokenize("piutang Medita piutang")
    hyp = tokenize("piutang media piutang piutang")
    s = score_terms(ref, hyp, ["medita"])
    assert s.expected == 1
    assert s.predicted == 0
    assert s.correct == 0
    assert s.accuracy() == 0.0


def test_critical_confusion_medita_media(terms_profile):
    m = evaluate_utterance(
        "Tampilkan piutang Medita bulan ini.", "Tampilkan piutang media bulan ini.", terms_profile
    )
    events = [(e.reference_term, e.hypothesis_term) for e in m.critical_events]
    assert ("medita", "media") in events
    # WER is only 1/5 but entity accuracy is 0 - the whole point of domain metrics.
    assert m.wer == pytest.approx(1 / 5)
    assert m.entity_score.accuracy() == 0.0


def test_amount_magnitude_confusion(terms_profile):
    m = evaluate_utterance(
        "dua ratus lima puluh juta rupiah", "dua ratus lima puluh miliar rupiah", terms_profile
    )
    pairs = {(e.reference_term, e.hypothesis_term) for e in m.critical_events}
    assert ("juta", "miliar") in pairs


def test_debit_kredit_confusion(terms_profile):
    m = evaluate_utterance("jurnal debit kas", "jurnal kredit kas", terms_profile)
    pairs = {(e.reference_term, e.hypothesis_term) for e in m.critical_events}
    assert ("debit", "kredit") in pairs


def test_critical_deletion_detected(terms_profile):
    m = evaluate_utterance("piutang Medita bulan ini", "piutang bulan ini", terms_profile)
    pairs = {(e.reference_term, e.hypothesis_term) for e in m.critical_events}
    assert ("medita", "") in pairs


def test_exact_match_flag(terms_profile):
    m = evaluate_utterance("Berapa saldo kas?", "berapa saldo kas", terms_profile)
    assert m.exact_match is True
    m2 = evaluate_utterance("Berapa saldo kas?", "berapa saldo kas sekarang", terms_profile)
    assert m2.exact_match is False


def test_amount_marker_scoring(terms_profile):
    m = evaluate_utterance("seratus ribu rupiah", "seratus ribu dolar", terms_profile)
    # amount markers present: seratus, ribu (rupiah is a currency term, not amount)
    assert m.amount_score.expected == 2
    assert m.amount_score.correct == 2
    # currency confusion is critical
    pairs = {(e.reference_term, e.hypothesis_term) for e in m.critical_events}
    assert ("rupiah", "dolar") in pairs
    assert m.currency_score.correct == 0


def test_date_marker_scoring(terms_profile):
    m = evaluate_utterance(
        "tanggal dua belas September", "tanggal dua belas Oktober", terms_profile
    )
    # date markers present: tanggal, september (numbers belong to amount tokens)
    assert m.date_score.expected == 2
    assert m.date_score.correct == 1


def test_load_term_profile_from_fixture(terms_profile):
    assert "medita" in terms_profile.entities
    assert "piutang" in terms_profile.financial_terms
    assert "medita" in terms_profile.critical_pairs
    assert "media" in terms_profile.critical_pairs["medita"]


def test_empty_profile_defaults():
    p = TermProfile()
    m = evaluate_utterance("apa kabar", "apa kabar", p)
    assert m.wer == 0.0
