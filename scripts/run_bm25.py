#!/usr/bin/env python3
"""Run the public MoraLink BM25 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moralink.bm25 import rank_bm25
from moralink.data import load_benchmark
from moralink.metrics import compute_standard_metrics, paper_compatible_mrr
from moralink.models import ModelSpec
from moralink.representations import build_document
from moralink.retrieval import save_ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=("fable_only", "summary_only", "summary_plus_fable"),
        required=True,
    )
    parser.add_argument("--summary-source", default="gemini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    benchmark = load_benchmark(args.data_dir, summary_source=args.summary_source)
    rankings = rank_bm25(
        [query.text for query in benchmark.queries],
        [
            build_document(
                fable,
                benchmark.summaries.get(fable.doc_id),
                args.representation,
            )
            for fable in benchmark.fables
        ],
        [query.doc_id for query in benchmark.queries],
        [fable.doc_id for fable in benchmark.fables],
        k1=args.k1,
        b=args.b,
    )
    metrics = compute_standard_metrics(rankings, benchmark.qrels)
    uncapped_mrr = paper_compatible_mrr(rankings, benchmark.qrels)
    spec = ModelSpec(
        alias="BM25",
        model_id="rank_bm25.BM25Okapi",
        batch_size=1,
        trust_remote_code=False,
    )
    run_id = f"BM25__{args.representation}__{args.summary_source}"
    save_ranking(
        args.output,
        rankings,
        run_id=run_id,
        model_spec=spec,
        instruction="no_instr",
        representation=args.representation,
        summary_source=args.summary_source,
    )
    print(
        json.dumps(
            {
                "model": "BM25",
                "queries": len(benchmark.queries),
                "fables": len(benchmark.fables),
                "output": str(args.output),
                "metric_mode": "standard cutoff at 10",
                "metrics": metrics,
                "uncapped_mrr": uncapped_mrr,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
