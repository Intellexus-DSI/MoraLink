"""Typed loading and validation for the released MoraLink benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMMARY_FILENAMES = {
    "gemini": "gemini-3-flash.json",
    "gemma4-E2B": "gemma-4-e2b.json",
    "gemma4-E4B": "gemma-4-e4b.json",
    "gemma4-31B": "gemma-4-31b.json",
}


@dataclass(frozen=True)
class Fable:
    doc_id: str
    alias: str
    text: str


@dataclass(frozen=True)
class Query:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Benchmark:
    fables: tuple[Fable, ...]
    queries: tuple[Query, ...]
    qrels: dict[str, frozenset[str]]
    summaries: dict[str, dict[str, str]]


def load_benchmark(root: Path, summary_source: str = "gemini") -> Benchmark:
    """Load the public MoraLink files and verify their cross-file references."""

    fable_rows = _load_json_list(root / "data" / "fables.json")
    query_rows = _load_json_list(root / "data" / "queries.json")
    qrel_rows = _load_json_list(root / "data" / "qrels.json")
    try:
        summary_filename = SUMMARY_FILENAMES[summary_source]
    except KeyError as error:
        raise ValueError(f"Unknown summary source: {summary_source}") from error
    summary_rows = _load_json_list(root / "summaries" / summary_filename)

    fables = tuple(
        Fable(
            doc_id=_string_field(row, "doc_id", "fable"),
            alias=_string_field(row, "alias", "fable"),
            text=_string_field(row, "text", "fable"),
        )
        for row in fable_rows
    )
    queries = tuple(
        Query(
            doc_id=_string_field(row, "doc_id", "query"),
            text=_string_field(row, "text", "query"),
        )
        for row in query_rows
    )

    _require_unique((fable.doc_id for fable in fables), "Duplicate fable ID")
    _require_unique((query.doc_id for query in queries), "Duplicate query ID")
    _require_unique((fable.alias for fable in fables), "Duplicate fable alias")

    fable_ids = {fable.doc_id for fable in fables}
    query_ids = {query.doc_id for query in queries}
    qrels: dict[str, set[str]] = {query_id: set() for query_id in query_ids}
    for row in qrel_rows:
        query_id = _string_field(row, "query_id", "qrel")
        fable_id = _string_field(row, "doc_id", "qrel")
        if query_id not in query_ids:
            raise ValueError(f"Unknown query ID in qrels: {query_id}")
        if fable_id not in fable_ids:
            raise ValueError(f"Unknown fable ID in qrels: {fable_id}")
        if row.get("relevance", 1) > 0:
            qrels[query_id].add(fable_id)

    queries_without_relevance = sorted(
        query_id for query_id, relevant in qrels.items() if not relevant
    )
    if queries_without_relevance:
        raise ValueError(
            "Queries without relevant fables: " + ", ".join(queries_without_relevance)
        )

    alias_to_id = {fable.alias: fable.doc_id for fable in fables}
    summaries: dict[str, dict[str, str]] = {}
    for row in summary_rows:
        alias = _string_field(row, "original_fable_id", "summary")
        if alias not in alias_to_id:
            raise ValueError(f"Unknown fable alias in summaries: {alias}")
        fable_id = alias_to_id[alias]
        if fable_id in summaries:
            raise ValueError(f"Duplicate summary for fable ID: {fable_id}")
        raw_summaries = row.get("summaries")
        if not isinstance(raw_summaries, dict):
            raise TypeError(f"Invalid summaries for fable ID: {fable_id}")
        summaries[fable_id] = _string_mapping(raw_summaries, fable_id)

    missing_summaries = sorted(fable_ids - summaries.keys())
    if missing_summaries:
        raise ValueError("Missing summaries for fables: " + ", ".join(missing_summaries))

    return Benchmark(
        fables=fables,
        queries=queries,
        qrels={
            query_id: frozenset(relevant) for query_id, relevant in qrels.items()
        },
        summaries=summaries,
    )


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"Expected a list of objects in {path}")
    return payload


def _string_field(row: dict[str, Any], field: str, kind: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {field!r} in {kind} record")
    return value


def _string_mapping(values: dict[Any, Any], fable_id: str) -> dict[str, str]:
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in values.items()
    ):
        raise ValueError(f"Invalid summary text for fable ID: {fable_id}")
    return dict(values)


def _require_unique(values: Any, message: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{message}: {value}")
        seen.add(value)
