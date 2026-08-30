#!/usr/bin/env python3
"""Reproduce the query-summary word-overlap analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import yaml

from moralink.analysis.word_overlap import analyze_word_overlap
from moralink.data import load_benchmark

SUMMARY_SOURCES = ("gemini", "gemma4-E2B", "gemma4-E4B", "gemma4-31B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rank-changes", type=Path, required=True)
    parser.add_argument("--official-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = load_benchmark(args.data_dir, summary_source="gemini")
    summary_collections = {
        source: load_benchmark(args.data_dir, summary_source=source).summaries
        for source in SUMMARY_SOURCES
    }
    with gzip.open(args.rank_changes, "rt", encoding="utf-8", newline="") as handle:
        rank_change_rows = list(csv.DictReader(handle))
    official_payload = yaml.safe_load(args.official_runs.read_text(encoding="utf-8"))
    official_runs = official_payload["runs"]

    system_rows, correlation_rows = analyze_word_overlap(
        benchmark,
        summary_collections,
        rank_change_rows,
        official_runs,
    )
    if len(rank_change_rows) != 106_212:
        raise ValueError(f"Expected 106,212 rank-change rows, found {len(rank_change_rows)}")
    if len(correlation_rows) != 159:
        raise ValueError(f"Expected 159 comparisons, found {len(correlation_rows)}")
    if any(row["n_queries"] != 668 for row in correlation_rows):
        raise ValueError("Every paired comparison must contain all 668 queries")
    correlations = [
        row["spearman_overlap_vs_reciprocal_rank_improvement"]
        for row in correlation_rows
    ]
    if any(value is None or float(value) <= 0 for value in correlations):
        raise ValueError("All 159 frozen Spearman correlations must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "overlap_by_system.csv", system_rows)
    _write_csv(args.output_dir / "rank_change_correlations.csv", correlation_rows)
    print(
        json.dumps(
            {
                "systems": len(system_rows),
                "comparisons": len(correlation_rows),
                "output_dir": str(args.output_dir),
            }
        )
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
