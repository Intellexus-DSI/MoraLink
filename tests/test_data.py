import json
from pathlib import Path

import pytest

from moralink.data import load_benchmark


def _write_release(
    root: Path,
    *,
    fables: list[dict] | None = None,
    queries: list[dict] | None = None,
    qrels: list[dict] | None = None,
    summaries: list[dict] | None = None,
) -> None:
    (root / "data").mkdir()
    (root / "summaries").mkdir()
    payloads = {
        "data/fables.json": fables
        if fables is not None
        else [
            {"doc_id": "f1", "alias": "aesop_1", "text": "Fable one."},
            {"doc_id": "f2", "alias": "aesop_2", "text": "Fable two."},
        ],
        "data/queries.json": queries
        if queries is not None
        else [
            {"doc_id": "q1", "text": "Moral one."},
            {"doc_id": "q2", "text": "Moral two."},
        ],
        "data/qrels.json": qrels
        if qrels is not None
        else [
            {"query_id": "q1", "doc_id": "f1", "relevance": 1},
            {"query_id": "q2", "doc_id": "f2", "relevance": 1},
        ],
        "summaries/gemini-3-flash.json": summaries
        if summaries is not None
        else [
            {
                "original_fable_id": "aesop_1",
                "summaries": {"direct_moral": "Summary one."},
            },
            {
                "original_fable_id": "aesop_2",
                "summaries": {"direct_moral": "Summary two."},
            },
        ],
    }
    for relative_path, payload in payloads.items():
        (root / relative_path).write_text(json.dumps(payload), encoding="utf-8")


def test_load_benchmark_uses_released_ids_and_summary_aliases(tmp_path: Path):
    _write_release(tmp_path)

    benchmark = load_benchmark(tmp_path)

    assert [fable.doc_id for fable in benchmark.fables] == ["f1", "f2"]
    assert [query.doc_id for query in benchmark.queries] == ["q1", "q2"]
    assert benchmark.qrels == {"q1": frozenset({"f1"}), "q2": frozenset({"f2"})}
    assert benchmark.summaries["f1"]["direct_moral"] == "Summary one."


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "fables": [
                    {"doc_id": "f1", "alias": "a1", "text": "One"},
                    {"doc_id": "f1", "alias": "a2", "text": "Two"},
                ]
            },
            "Duplicate fable ID",
        ),
        (
            {
                "queries": [
                    {"doc_id": "q1", "text": "One"},
                    {"doc_id": "q1", "text": "Two"},
                ]
            },
            "Duplicate query ID",
        ),
        (
            {
                "qrels": [
                    {"query_id": "missing", "doc_id": "f1", "relevance": 1},
                    {"query_id": "q2", "doc_id": "f2", "relevance": 1},
                ]
            },
            "Unknown query ID",
        ),
        (
            {
                "qrels": [
                    {"query_id": "q1", "doc_id": "missing", "relevance": 1},
                    {"query_id": "q2", "doc_id": "f2", "relevance": 1},
                ]
            },
            "Unknown fable ID",
        ),
        (
            {
                "summaries": [
                    {
                        "original_fable_id": "aesop_1",
                        "summaries": {"direct_moral": "Summary one."},
                    }
                ]
            },
            "Missing summaries",
        ),
    ],
)
def test_load_benchmark_rejects_invalid_release(
    tmp_path: Path, overrides: dict, message: str
):
    _write_release(tmp_path, **overrides)

    with pytest.raises(ValueError, match=message):
        load_benchmark(tmp_path)


def test_load_benchmark_requires_relevance_for_every_query(tmp_path: Path):
    _write_release(
        tmp_path,
        qrels=[{"query_id": "q1", "doc_id": "f1", "relevance": 1}],
    )

    with pytest.raises(ValueError, match="Queries without relevant fables"):
        load_benchmark(tmp_path)
