"""Paper-compatible and standard MoraLink retrieval metrics."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

Qrels = dict[str, set[str] | frozenset[str]]
Rankings = dict[str, list[str]]


def paper_compatible_mrr(rankings: Rankings, qrels: Qrels) -> float:
    """Return uncapped MRR, matching the paper's reported MRR@10 column."""

    contributions = _reciprocal_ranks(rankings, qrels, k=None)
    return _mean(contributions.values())


def standard_mrr_at_k(rankings: Rankings, qrels: Qrels, k: int = 10) -> float:
    """Return standard MRR@k, with the first relevant result capped at rank k."""

    contributions = _reciprocal_ranks(rankings, qrels, k=k)
    return _mean(contributions.values())


def compute_paper_metrics(
    rankings: Rankings,
    qrels: Qrels,
    paper_reciprocal_ranks: dict[str, float] | None = None,
    k: int = 10,
) -> dict[str, float]:
    """Compute the paper metrics, whose MRR first-relevant rank is not capped."""

    _validate_inputs(rankings, qrels, k)
    if paper_reciprocal_ranks is None:
        unresolved = {
            query_id
            for query_id, relevant in qrels.items()
            if not any(fable_id in relevant for fable_id in rankings[query_id])
        }
        if unresolved:
            raise ValueError(
                "Compact rankings require released paper reciprocal-rank "
                "contributions for queries whose first relevant result is not saved"
            )
        resolved_mrr = paper_compatible_mrr(rankings, qrels)
    else:
        if set(paper_reciprocal_ranks) != set(qrels):
            raise ValueError(
                "Paper reciprocal-rank contributions do not match the query IDs"
            )
        resolved_mrr = _mean(paper_reciprocal_ranks.values())

    return {
        "MRR@10": resolved_mrr,
        "MAP@10": _map_at_k(rankings, qrels, k),
        "Recall@10": _recall_at_k(rankings, qrels, k),
    }


def compute_standard_metrics(
    rankings: Rankings,
    qrels: Qrels,
    k: int = 10,
) -> dict[str, float]:
    """Compute standard MRR, MAP, and Recall, all capped at rank k."""

    _validate_inputs(rankings, qrels, k)
    return {
        "MRR@10": standard_mrr_at_k(rankings, qrels, k),
        "MAP@10": _map_at_k(rankings, qrels, k),
        "Recall@10": _recall_at_k(rankings, qrels, k),
    }


def load_paper_contributions(path: Path, run_id: str) -> dict[str, float]:
    """Load uncapped per-query reciprocal ranks for one official run."""

    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        required = {"run_id", "query_id", "MRR"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Contribution file is missing columns: {sorted(required)}")
        contributions: dict[str, float] = {}
        for row in reader:
            if row["run_id"] != run_id:
                continue
            query_id = row["query_id"]
            if query_id in contributions:
                raise ValueError(f"Duplicate contribution for query ID: {query_id}")
            contributions[query_id] = float(row["MRR"])
    if not contributions:
        raise ValueError(f"No paper contributions found for run: {run_id}")
    return contributions


def _reciprocal_ranks(
    rankings: Rankings,
    qrels: Qrels,
    k: int | None,
) -> dict[str, float]:
    _validate_inputs(rankings, qrels, k)
    result: dict[str, float] = {}
    for query_id, ranking in rankings.items():
        candidates = ranking if k is None else ranking[:k]
        first_rank = next(
            (
                rank
                for rank, fable_id in enumerate(candidates, start=1)
                if fable_id in qrels[query_id]
            ),
            None,
        )
        result[query_id] = 0.0 if first_rank is None else 1.0 / first_rank
    return result


def _map_at_k(rankings: Rankings, qrels: Qrels, k: int) -> float:
    contributions: list[float] = []
    for query_id, ranking in rankings.items():
        relevant = qrels[query_id]
        hits = 0
        precision_sum = 0.0
        for rank, fable_id in enumerate(ranking[:k], start=1):
            if fable_id in relevant:
                hits += 1
                precision_sum += hits / rank
        contributions.append(precision_sum / min(len(relevant), k))
    return _mean(contributions)


def _recall_at_k(rankings: Rankings, qrels: Qrels, k: int) -> float:
    contributions = [
        sum(fable_id in qrels[query_id] for fable_id in ranking[:k])
        / len(qrels[query_id])
        for query_id, ranking in rankings.items()
    ]
    return _mean(contributions)


def _validate_inputs(rankings: Rankings, qrels: Qrels, k: int | None) -> None:
    if set(rankings) != set(qrels):
        raise ValueError("Ranking and relevance query IDs do not match")
    if not rankings:
        raise ValueError("Cannot evaluate an empty ranking set")
    if k is not None and k < 1:
        raise ValueError("k must be a positive integer")
    empty_qrels = [query_id for query_id, relevant in qrels.items() if not relevant]
    if empty_qrels:
        raise ValueError(f"Queries have no relevant fables: {sorted(empty_qrels)}")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("Cannot average an empty collection")
    return sum(items) / len(items)


def _open_csv(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")
