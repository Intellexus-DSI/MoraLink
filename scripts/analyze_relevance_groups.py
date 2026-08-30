#!/usr/bin/env python3
"""Reproduce the paper's one-versus-several relevance analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import yaml

from moralink.analysis.relevance_groups import (
    aggregate_relevance_groups,
    public_representation_fields,
)
from moralink.data import load_benchmark

OUTPUT_FIELDS = (
    "run_id",
    "model_alias",
    "model_id",
    "instruction_variant",
    "document_representation",
    "corpus_config",
    "subset",
    "n_queries",
    "MRR",
    "MAP@10",
    "Recall@10",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--per-query-metrics", type=Path, required=True)
    parser.add_argument("--official-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-source", default="gemini")
    args = parser.parse_args()

    benchmark = load_benchmark(args.data_dir, summary_source=args.summary_source)
    expected_queries = {query.doc_id for query in benchmark.queries}
    with gzip.open(
        args.per_query_metrics, "rt", encoding="utf-8", newline=""
    ) as handle:
        per_query_rows = list(csv.DictReader(handle))

    official_payload = yaml.safe_load(args.official_runs.read_text(encoding="utf-8"))
    official_runs = {row["run_id"]: row for row in official_payload["runs"]}
    if len(official_runs) != 144:
        raise ValueError(f"Expected 144 official runs, found {len(official_runs)}")

    by_run: dict[str, set[str]] = {}
    for row in per_query_rows:
        by_run.setdefault(row["run_id"], set()).add(row["query_id"])
    if set(by_run) != set(official_runs):
        raise ValueError("Per-query metrics do not cover the 144 official run IDs")
    incomplete = {
        run_id: sorted(expected_queries - query_ids)
        for run_id, query_ids in by_run.items()
        if query_ids != expected_queries
    }
    if incomplete:
        raise ValueError("Per-query metrics do not cover all 668 queries for every run")

    aggregates = aggregate_relevance_groups(per_query_rows)
    if len(aggregates) != 288:
        raise ValueError(f"Expected 288 aggregate rows, found {len(aggregates)}")
    output_rows = []
    for row in aggregates:
        run = official_runs[str(row["run_id"])]
        document_representation, corpus_config = public_representation_fields(
            run["document_representation"]
        )
        output_rows.append(
            {
                **run,
                **row,
                "document_representation": document_representation,
                "corpus_config": corpus_config,
            }
        )
    output_rows.sort(
        key=lambda row: (
            str(row["model_alias"]),
            str(row["instruction_variant"]),
            str(row["subset"]),
            {
                "raw": 0,
                "direct_moral_only": 1,
                "direct_moral_fable": 2,
            }[str(row["corpus_config"])],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({"runs": 144, "rows": 288, "output": str(args.output)}))


if __name__ == "__main__":
    main()
