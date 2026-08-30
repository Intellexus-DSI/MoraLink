#!/usr/bin/env python3
"""Validate a downloaded MoraLink data release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moralink.artifacts import validate_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory containing the downloaded artifacts",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/artifacts.yaml"),
        help="Artifact manifest to validate against",
    )
    args = parser.parse_args()

    summary = validate_artifacts(args.root, args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
