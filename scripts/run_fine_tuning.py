#!/usr/bin/env python3
"""Run or validate one reported MoraLink fine-tuning setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moralink.training import dry_run_summary, load_training_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--storal-path", type=Path)
    parser.add_argument("--tf1-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_training_config(args.config)
    shared = {
        "model_alias": args.model,
        "seed": args.seed,
        "storal_path": args.storal_path,
        "tf1_path": args.tf1_path,
    }
    if args.dry_run:
        print(json.dumps(dry_run_summary(config, **shared), indent=2))
        return
    train(config, output=args.output, **shared)


if __name__ == "__main__":
    main()
