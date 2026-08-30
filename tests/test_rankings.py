import json
from pathlib import Path

import pytest

from moralink.data import Benchmark, Fable, Query
from moralink.rankings import load_rankings

BENCHMARK = Benchmark(
    fables=(
        Fable(doc_id="f1", alias="a1", text="One"),
        Fable(doc_id="f2", alias="a2", text="Two"),
        Fable(doc_id="f3", alias="a3", text="Three"),
    ),
    queries=(Query(doc_id="q1", text="One"), Query(doc_id="q2", text="Two")),
    qrels={"q1": frozenset({"f1"}), "q2": frozenset({"f2"})},
    summaries={},
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_rankings_supports_ranking_v1(tmp_path: Path):
    path = tmp_path / "ranking.json"
    _write(
        path,
        {
            "schema_version": "ranking_v1",
            "n_queries": 2,
            "n_docs": 3,
            "ranking_depth": 2,
            "query_ids": ["q1", "q2"],
            "ranked_fable_ids": [["f1", "f2"], ["f2", "f3"]],
        },
    )

    assert load_rankings(path, BENCHMARK, expected_ranking_depth=2) == {
        "q1": ["f1", "f2"],
        "q2": ["f2", "f3"],
    }


def test_load_rankings_supports_legacy_queries_schema(tmp_path: Path):
    path = tmp_path / "ranking.json"
    _write(
        path,
        {
            "n_queries": 2,
            "n_docs": 3,
            "ranking_depth": 2,
            "queries": [
                {"query_id": "q1", "ranked_fable_ids": ["f1", "f2"]},
                {"query_id": "q2", "ranked_fable_ids": ["f2", "f3"]},
            ],
        },
    )

    assert load_rankings(path, BENCHMARK, expected_ranking_depth=2)["q2"] == [
        "f2",
        "f3",
    ]


@pytest.mark.parametrize(
    ("query_ids", "rankings", "message"),
    [
        (["q1", "q1"], [["f1", "f2"], ["f2", "f3"]], "duplicate query"),
        (["q1", "missing"], [["f1", "f2"], ["f2", "f3"]], "unknown query"),
        (["q1", "q2"], [["f1", "f1"], ["f2", "f3"]], "duplicate documents"),
        (["q1", "q2"], [["f1", "missing"], ["f2", "f3"]], "unknown fable"),
        (["q1", "q2"], [["f1"], ["f2", "f3"]], "expected 2"),
    ],
)
def test_load_rankings_rejects_invalid_payload(
    tmp_path: Path,
    query_ids: list[str],
    rankings: list[list[str]],
    message: str,
):
    path = tmp_path / "ranking.json"
    _write(
        path,
        {
            "schema_version": "ranking_v1",
            "n_queries": 2,
            "n_docs": 3,
            "ranking_depth": 2,
            "query_ids": query_ids,
            "ranked_fable_ids": rankings,
        },
    )

    with pytest.raises(ValueError, match=message):
        load_rankings(path, BENCHMARK, expected_ranking_depth=2)
