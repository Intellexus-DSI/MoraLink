#!/usr/bin/env python3
"""Evaluate a MoraLink ranking with paper-compatible or standard metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moralink.data import load_benchmark
from moralink.metrics import (
    compute_paper_metrics,
    compute_standard_metrics,
    load_paper_contributions,
)
from moralink.rankings import load_rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--summary-source", default="gemini")
    parser.add_argument(
        "--metric-mode",
        choices=("paper", "standard"),
        default="paper",
    )
    parser.add_argument("--paper-contributions", type=Path)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    benchmark = load_benchmark(args.data_dir, summary_source=args.summary_source)
    ranking_payload = json.loads(args.ranking.read_text(encoding="utf-8"))
    rankings = load_rankings(args.ranking, benchmark)

    if args.metric_mode == "paper":
        contributions = None
        if len(next(iter(rankings.values()))) < len(benchmark.fables):
            if args.paper_contributions is None:
                parser.error(
                    "--paper-contributions is required for compact paper-mode rankings"
                )
            run_id = ranking_payload.get("run_id")
            if not isinstance(run_id, str):
                parser.error("The ranking must include run_id in paper mode")
            contributions = load_paper_contributions(
                args.paper_contributions,
                run_id,
            )
        metrics = compute_paper_metrics(
            rankings,
            benchmark.qrels,
            paper_reciprocal_ranks=contributions,
            k=args.k,
        )
    else:
        metrics = compute_standard_metrics(rankings, benchmark.qrels, k=args.k)

    print(json.dumps({"mode": args.metric_mode, **metrics}, indent=2))


if __name__ == "__main__":
    main()
