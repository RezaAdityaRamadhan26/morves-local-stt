"""Metric text normalization rule tests."""

from __future__ import annotations

from morves_stt.normalize import normalize_for_metric, tokenize


def test_lowercase_and_punctuation():
    assert (
        normalize_for_metric("Tampilkan piutang Medita, bulan ini!")
        == "tampilkan piutang medita bulan ini"
    )


def test_whitespace_collapse():
    assert normalize_for_metric("  piutang\t\n juta  ") == "piutang juta"


def test_hyphen_treated_as_word_boundary():
    # antar-perusahaan and antarperusahaan compare equal (spelling variant), not meaning change
    assert normalize_for_metric("antar-perusahaan") == normalize_for_metric("antarperusahaan")


def test_digits_not_converted_to_words():
    # The ASR contract is verbatim spoken form; "12" and "dua belas" must NOT unify.
    assert normalize_for_metric("tanggal 12 September") != normalize_for_metric(
        "tanggal dua belas September"
    )


def test_amount_words_not_unified():
    assert normalize_for_metric("juta") != normalize_for_metric("miliar")
    assert tokenize("seratus ribu") != tokenize("100 ribu")


def test_entity_not_unified_with_common_word():
    assert normalize_for_metric("Medita") != normalize_for_metric("media")
    assert normalize_for_metric("Morves") != normalize_for_metric("modus")


def test_debit_kredit_distinct():
    assert normalize_for_metric("debit") != normalize_for_metric("kredit")


def test_currency_presentation_separates():
    # Punctuation becomes a token separator (never glued): Rp250.000.000 -> rp250 000 000
    assert normalize_for_metric("Rp250.000.000") == "rp250 000 000"
    # and remains distinct from the spoken-word form (numbers stay numbers)
    assert normalize_for_metric("Rp250.000.000") != normalize_for_metric(
        "dua ratus lima puluh juta rupiah"
    )


def test_nfkc_composition():
    # decomposed (NFD) accent must unify with the composed form under NFKC
    decomposed = "cafe" + "́"
    assert normalize_for_metric("café") == normalize_for_metric(decomposed)


def test_tokenizes():
    assert tokenize("Neraca Saldo, Q1!") == ["neraca", "saldo", "q1"]


def test_empty():
    assert normalize_for_metric("") == ""
    assert tokenize("   ") == []
