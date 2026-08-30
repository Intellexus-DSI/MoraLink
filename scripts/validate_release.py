#!/usr/bin/env python3
"""Validate a MoraLink code checkout and downloaded data release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moralink.release_validation import validate_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    print(
        json.dumps(
            validate_release(args.repository, args.data_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
