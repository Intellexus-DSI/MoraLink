import math

import pytest

from moralink.bm25 import rank_bm25, score_bm25, tokenize_bm25


def test_bm25_ranks_the_matching_document_first():
    rankings = rank_bm25(
        ["fox grapes"],
        ["a fox reaches for grapes", "a lion sleeps"],
        ["q1"],
        ["f1", "f2"],
    )

    assert rankings["q1"][0] == "f1"


def test_bm25_tokenization_is_lowercase_and_alphanumeric():
    assert tokenize_bm25("Fox's GRAPES, 2026!") == ["fox's", "grapes", "2026"]


def test_bm25_uses_the_paper_positive_robertson_idf():
    scores = score_bm25(["fox"], ["fox", "fox fox", "lion"])
    expected_idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected_first = expected_idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 1 / (4 / 3)))
    expected_second = (
        expected_idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / (4 / 3)))
    )

    assert scores[0, 0] == pytest.approx(expected_first)
    assert scores[0, 1] == pytest.approx(expected_second)
    assert scores[0, 2] == 0.0


def test_bm25_preserves_document_order_for_ties():
    rankings = rank_bm25(
        ["missing"],
        ["fox", "lion"],
        ["q1"],
        ["f1", "f2"],
    )

    assert rankings["q1"] == ["f1", "f2"]
