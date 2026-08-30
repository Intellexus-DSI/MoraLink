#!/usr/bin/env python3
"""Run the optional direct OpenAI or Google MoraLink ranking baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moralink.data import load_benchmark
from moralink.llm_ranking import (
    build_corpus_block,
    parse_ranked_ids,
    render_ranking_prompt,
    request_ranking,
)
from moralink.metrics import compute_standard_metrics
from moralink.models import ModelSpec
from moralink.representations import build_document
from moralink.retrieval import save_ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "google"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=("fable_only", "summary_only", "summary_plus_fable"),
        required=True,
    )
    parser.add_argument("--summary-source", default="gemini")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    benchmark = load_benchmark(args.data_dir, summary_source=args.summary_source)
    documents = [
        build_document(
            fable,
            benchmark.summaries.get(fable.doc_id),
            args.representation,
        )
        for fable in benchmark.fables
    ]
    corpus = build_corpus_block(list(benchmark.fables), documents)
    first_prompt = render_ranking_prompt(
        benchmark.queries[0].text,
        corpus,
        args.top_k,
    )
    if args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(first_prompt, encoding="utf-8")
        print(json.dumps({"dry_run": True, "output": str(args.output)}))
        return

    allowed_ids = {fable.doc_id for fable in benchmark.fables}
    rankings: dict[str, list[str]] = {}
    for query in benchmark.queries:
        prompt = render_ranking_prompt(query.text, corpus, args.top_k)
        response = request_ranking(args.provider, args.model, prompt)
        ranking = parse_ranked_ids(response, allowed_ids, top_k=args.top_k)
        if len(ranking) != args.top_k:
            raise ValueError(
                f"Query {query.doc_id} returned {len(ranking)} IDs; "
                f"expected {args.top_k}"
            )
        rankings[query.doc_id] = ranking

    metrics = compute_standard_metrics(rankings, benchmark.qrels, k=args.top_k)
    spec = ModelSpec(
        alias=args.model,
        model_id=args.model,
        batch_size=1,
        trust_remote_code=False,
    )
    run_id = (
        f"{args.model}__direct_llm__{args.representation}__{args.summary_source}"
    )
    save_ranking(
        args.output,
        rankings,
        run_id=run_id,
        model_spec=spec,
        instruction="no_instr",
        representation=args.representation,
        summary_source=args.summary_source,
        n_docs=len(benchmark.fables),
    )
    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "queries": len(benchmark.queries),
                "fables": len(benchmark.fables),
                "output": str(args.output),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
