#!/usr/bin/env python3
"""Run one public MoraLink zero-shot embedding experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moralink.data import load_benchmark
from moralink.models import ModelSpec, load_model_specs
from moralink.retrieval import run_zero_shot, save_ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Public alias or model ID")
    parser.add_argument(
        "--representation",
        choices=("fable_only", "summary_only", "summary_plus_fable"),
        required=True,
    )
    parser.add_argument(
        "--instruction",
        choices=("no_instr", "general"),
        required=True,
    )
    parser.add_argument("--summary-source", default="gemini")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    release_root = Path(__file__).resolve().parents[1]
    specs = load_model_specs(release_root / "configs" / "models.yaml")
    spec = _resolve_model(args.model, specs)
    benchmark = load_benchmark(args.data_dir, summary_source=args.summary_source)
    rankings, metrics = run_zero_shot(
        benchmark,
        spec,
        args.representation,
        args.instruction,
        batch_size=args.batch_size,
        device=args.device,
    )
    run_id = (
        f"{spec.alias}__{args.instruction}__{args.representation}__"
        f"{args.summary_source}"
    )
    save_ranking(
        args.output,
        rankings,
        run_id=run_id,
        model_spec=spec,
        instruction=args.instruction,
        representation=args.representation,
        summary_source=args.summary_source,
    )
    print(
        json.dumps(
            {
                "model": spec.alias,
                "device": args.device or "auto",
                "queries": len(benchmark.queries),
                "fables": len(benchmark.fables),
                "output": str(args.output),
                "metrics": metrics,
            },
            indent=2,
        )
    )


def _resolve_model(value: str, specs: dict[str, ModelSpec]) -> ModelSpec:
    if value in specs:
        return specs[value]
    matches = [spec for spec in specs.values() if spec.model_id == value]
    if len(matches) != 1:
        raise ValueError(f"Unknown model alias or ID: {value}")
    return matches[0]


if __name__ == "__main__":
    main()
