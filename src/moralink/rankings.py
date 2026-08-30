"""Load and validate MoraLink ranking artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from moralink.data import Benchmark


def load_rankings(
    path: Path,
    benchmark: Benchmark,
    *,
    expected_ranking_depth: int | None = None,
) -> dict[str, list[str]]:
    """Load rankings with exact query coverage and valid, unique fable IDs."""

    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported ranking schema in {path}")

    if payload.get("schema_version") == "ranking_v1":
        query_ids = payload.get("query_ids")
        ranked_fable_ids = payload.get("ranked_fable_ids")
    elif "queries" in payload:
        query_rows = payload["queries"]
        if not isinstance(query_rows, list):
            raise ValueError("Legacy ranking queries must be a list")
        query_ids = [row.get("query_id") for row in query_rows]
        ranked_fable_ids = [row.get("ranked_fable_ids") for row in query_rows]
    else:
        raise ValueError(f"Unsupported ranking schema in {path}")

    if not isinstance(query_ids, list) or not isinstance(ranked_fable_ids, list):
        raise TypeError("Ranking query IDs and ranked fable IDs must be lists")
    if len(query_ids) != len(ranked_fable_ids):
        raise ValueError("Query and ranking counts differ")
    if payload.get("n_queries") != len(query_ids):
        raise ValueError("n_queries metadata does not match the ranking payload")

    expected_query_ids = {query.doc_id for query in benchmark.queries}
    expected_fable_ids = {fable.doc_id for fable in benchmark.fables}
    expected_doc_count = len(benchmark.fables)
    if payload.get("n_docs") != expected_doc_count:
        raise ValueError("n_docs metadata does not match the benchmark")

    ranking_depth = expected_ranking_depth
    if ranking_depth is None:
        ranking_depth = payload.get("ranking_depth", expected_doc_count)
    if not isinstance(ranking_depth, int) or ranking_depth < 1:
        raise ValueError("ranking depth must be a positive integer")
    if "ranking_depth" in payload and payload["ranking_depth"] != ranking_depth:
        raise ValueError("ranking_depth metadata does not match the expected depth")

    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Ranking contains duplicate query IDs")
    unknown_queries = set(query_ids) - expected_query_ids
    if unknown_queries:
        raise ValueError(f"Ranking contains unknown query IDs: {sorted(unknown_queries)}")
    missing_queries = expected_query_ids - set(query_ids)
    if missing_queries:
        raise ValueError(f"Ranking is missing query IDs: {sorted(missing_queries)}")

    result: dict[str, list[str]] = {}
    for query_id, ranking in zip(query_ids, ranked_fable_ids, strict=True):
        if not isinstance(query_id, str) or not isinstance(ranking, list):
            raise TypeError("Ranking rows must contain string query IDs and lists")
        if len(ranking) != ranking_depth:
            raise ValueError(
                f"Ranking for {query_id} has {len(ranking)} documents; "
                f"expected {ranking_depth} documents"
            )
        if not all(isinstance(fable_id, str) for fable_id in ranking):
            raise ValueError(f"Ranking for {query_id} contains a non-string fable ID")
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"Ranking for {query_id} contains duplicate documents")
        unknown_fables = set(ranking) - expected_fable_ids
        if unknown_fables:
            raise ValueError(
                f"Ranking for {query_id} contains unknown fable IDs: "
                f"{sorted(unknown_fables)}"
            )
        result[query_id] = list(ranking)
    return result
