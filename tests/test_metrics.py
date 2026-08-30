import csv
import gzip
from pathlib import Path

import pytest

from moralink.metrics import (
    compute_paper_metrics,
    compute_standard_metrics,
    load_paper_contributions,
    paper_compatible_mrr,
    standard_mrr_at_k,
)


def test_paper_metric_and_standard_mrr_at_10_are_explicitly_different():
    ranking = [f"fable_{index:02d}" for index in range(1, 12)]
    rankings = {"query_1": ranking}
    qrels = {"query_1": {"fable_11"}}

    assert paper_compatible_mrr(rankings, qrels) == 1 / 11
    assert standard_mrr_at_k(rankings, qrels, k=10) == 0.0


def test_compact_ranking_requires_released_paper_contribution():
    rankings = {"query_1": [f"fable_{index:02d}" for index in range(1, 11)]}
    qrels = {"query_1": {"fable_11"}}

    with pytest.raises(ValueError, match="paper reciprocal-rank contributions"):
        compute_paper_metrics(rankings, qrels)


def test_paper_metrics_use_released_uncapped_contributions():
    rankings = {"q1": ["f1", "f2"], "q2": ["f1", "f2"]}
    qrels = {"q1": {"f1"}, "q2": {"f3"}}

    metrics = compute_paper_metrics(
        rankings,
        qrels,
        paper_reciprocal_ranks={"q1": 1.0, "q2": 0.25},
        k=2,
    )

    assert metrics == {"MRR@10": 0.625, "MAP@10": 0.5, "Recall@10": 0.5}


def test_standard_metrics_cap_all_metrics_at_k():
    rankings = {"q1": ["f1", "f2", "f3"], "q2": ["f2", "f1", "f3"]}
    qrels = {"q1": {"f1", "f3"}, "q2": {"f3"}}

    metrics = compute_standard_metrics(rankings, qrels, k=2)

    assert metrics["MRR@10"] == 0.5
    assert metrics["MAP@10"] == 0.25
    assert metrics["Recall@10"] == 0.25


def test_load_paper_contributions_selects_one_run(tmp_path: Path):
    path = tmp_path / "contributions.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "query_id", "MRR"])
        writer.writeheader()
        writer.writerows(
            [
                {"run_id": "run-a", "query_id": "q1", "MRR": "0.5"},
                {"run_id": "run-b", "query_id": "q1", "MRR": "1.0"},
                {"run_id": "run-a", "query_id": "q2", "MRR": "0.25"},
            ]
        )

    assert load_paper_contributions(path, "run-a") == {"q1": 0.5, "q2": 0.25}
