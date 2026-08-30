import numpy as np
import pytest

from moralink.retrieval import rank_embeddings


def test_rank_embeddings_uses_deterministic_descending_cosine_order():
    query_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    document_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    rankings = rank_embeddings(
        query_embeddings,
        document_embeddings,
        query_ids=["q1", "q2"],
        fable_ids=["f1", "f2", "f3"],
    )

    assert rankings == {
        "q1": ["f1", "f3", "f2"],
        "q2": ["f2", "f3", "f1"],
    }


def test_rank_embeddings_preserves_fable_order_for_ties():
    rankings = rank_embeddings(
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0], [0.0, 2.0]]),
        query_ids=["q1"],
        fable_ids=["f1", "f2"],
    )

    assert rankings["q1"] == ["f1", "f2"]


def test_rank_embeddings_rejects_wrong_id_counts():
    with pytest.raises(ValueError, match="query IDs"):
        rank_embeddings(
            np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0]]),
            query_ids=[],
            fable_ids=["f1"],
        )
