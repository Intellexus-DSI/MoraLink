"""BM25 lexical retrieval baseline."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize_bm25(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens used by the paper baseline."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


def rank_bm25(
    queries: list[str],
    documents: list[str],
    query_ids: list[str],
    fable_ids: list[str],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, list[str]]:
    """Rank all documents with the paper BM25 scores and stable tie ordering."""

    if len(queries) != len(query_ids):
        raise ValueError("The number of queries and query IDs must match")
    if len(documents) != len(fable_ids):
        raise ValueError("The number of documents and fable IDs must match")
    if not documents:
        raise ValueError("BM25 requires at least one document")

    score_matrix = score_bm25(queries, documents, k1=k1, b=b)
    orders = np.argsort(-score_matrix, axis=1, kind="stable")
    return {
        query_id: [fable_ids[position] for position in order]
        for query_id, order in zip(query_ids, orders, strict=True)
    }


def score_bm25(
    queries: list[str],
    documents: list[str],
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    """Return the exact positive-IDF BM25 score matrix used in the paper."""

    tokenized_documents = [tokenize_bm25(document) for document in documents]
    document_lengths = np.asarray(
        [len(tokens) for tokens in tokenized_documents],
        dtype=np.float64,
    )
    average_length = float(np.mean(document_lengths)) if documents else 0.0
    term_frequencies = [Counter(tokens) for tokens in tokenized_documents]
    document_frequencies: Counter[str] = Counter()
    for frequencies in term_frequencies:
        document_frequencies.update(frequencies.keys())

    document_count = len(documents)
    inverse_document_frequencies = {
        term: math.log(
            1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
        )
        for term, frequency in document_frequencies.items()
    }
    scores = np.zeros((len(queries), document_count), dtype=np.float64)
    for query_index, query in enumerate(queries):
        query_frequencies = Counter(tokenize_bm25(query))
        for term, query_count in query_frequencies.items():
            term_idf = inverse_document_frequencies.get(term)
            if term_idf is None:
                continue
            for document_index, frequencies in enumerate(term_frequencies):
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + k1 * (
                    1.0
                    - b
                    + b * document_lengths[document_index] / average_length
                )
                scores[query_index, document_index] += (
                    query_count
                    * term_idf
                    * (frequency * (k1 + 1.0))
                    / denominator
                )
    return scores
