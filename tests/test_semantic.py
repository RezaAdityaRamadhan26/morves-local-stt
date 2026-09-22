"""Semantic diagnostic metric tests (numbers, dates, currencies)."""

from __future__ import annotations

import pytest

from morves_stt.semantic import (
    amount_semantic_accuracy,
    currency_semantic_accuracy,
    date_semantic_accuracy,
    extract_amounts,
    extract_currencies,
    extract_dates,
    semantic_scores,
)


class TestAmounts:
    def test_spoken_word_phrases(self):
        assert extract_amounts("dua ratus lima puluh juta rupiah") == [250_000_000.0]

    def test_digit_phrases(self):
        assert extract_amounts("250 juta") == [250_000_000.0]
        assert extract_amounts("nilai 1,5 miliar bulan ini") == [1.5e9]

    def test_equivalence_spoken_vs_digits(self):
        ref = "sebesar dua ratus lima puluh juta rupiah"
        hyp = "sebesar 250 juta rupiah"
        assert amount_semantic_accuracy(ref, hyp) == 1.0

    def test_magnitude_mismatch_is_not_equivalent(self):
        assert amount_semantic_accuracy("lima juta", "lima miliar") == 0.0
        assert amount_semantic_accuracy("lima ratus ribu", "lima juta") == 0.0

    def test_bare_digit_without_multiplier_is_not_equal(self):
        # "250" alone is 250, not 250 million - multiplier must be present
        assert amount_semantic_accuracy("dua ratus lima puluh juta", "250") == 0.0

    def test_koma_decimal(self):
        assert extract_amounts("satu koma lima miliar rupiah") == [1.5e9]
        assert extract_amounts("nol koma nol lima") == [0.05]

    def test_multiple_values_multiset(self):
        ref = "lima juta dan tiga ratus ribu"
        assert amount_semantic_accuracy(ref, "5 juta dan 300 ribu") == 1.0
        assert amount_semantic_accuracy(ref, "5 juta dan 300 juta") == 0.5

    def test_thousands_dots(self):
        assert extract_amounts("total 250.000 sudah masuk") == [250_000.0]

    def test_year_word_number(self):
        assert extract_amounts("tahun dua ribu dua puluh enam") == [2026.0]

    def test_not_applicable_returns_none(self):
        assert amount_semantic_accuracy("tampilkan piutang", "tampilin piutang") is None


class TestDates:
    def test_spoken_vs_digit_day(self):
        ref = "transaksi tanggal dua belas September"
        hyp = "transaksi tanggal 12 September"
        assert date_semantic_accuracy(ref, hyp) == 1.0

    def test_month_mismatch_fails(self):
        assert date_semantic_accuracy("dua belas September", "dua belas Oktober") == 0.0

    def test_misspelled_month_fails(self):
        # recognition errors are never semantically normalized away
        assert date_semantic_accuracy("Januari", "jenuari") == 0.0

    def test_year_attached(self):
        ref = "satu Januari sampai tiga puluh satu Desember dua ribu dua puluh enam"
        hyp = "1 Januari sampai 31 Desember 2026"
        assert date_semantic_accuracy(ref, hyp) == 1.0

    def test_tuple_extraction(self):
        dates = extract_dates("tanggal dua belas September 2026")
        assert dates == [(12, 9, 2026)]

    def test_not_applicable(self):
        assert date_semantic_accuracy("piutang bulan ini", "piutang bulan ini") is None


class TestCurrencies:
    def test_equivalence_classes(self):
        assert extract_currencies("dalam rupiah dan dolar") == {"IDR": 1, "USD": 1}
        assert extract_currencies("IDR dan USD") == {"IDR": 1, "USD": 1}

    def test_mismatch(self):
        assert currency_semantic_accuracy("rupiah dan dolar", "rupiah dan rupiah") == 0.5

    def test_not_applicable(self):
        assert currency_semantic_accuracy("tampilkan piutang", "piutang") is None


def test_semantic_scores_bundle():
    scores = semantic_scores(
        "dua belas September lima juta",
        "12 September 5 juta",
    )
    assert scores["amount_semantic"] == 1.0
    assert scores["date_semantic"] == 1.0
    assert scores["currency_semantic"] is None
    # 'rupiah' on both sides is an applicable, matching currency mention
    assert semantic_scores("lima juta rupiah", "5 juta rupiah")["currency_semantic"] == 1.0


@pytest.mark.parametrize(
    "spoken,digit",
    [
        ("seratus ribu", "100 ribu"),
        ("seribu", "1000"),
        ("sebelas", "11"),
        ("dua belas", "12"),
        ("dua ratus lima puluh", "250"),
        ("tiga belas juta lima ratus ribu", "13.500.000"),
    ],
)
def test_spoken_digit_equivalences(spoken, digit):
    assert amount_semantic_accuracy(f"nilai {spoken} rupiah", f"nilai {digit} rupiah") == 1.0
