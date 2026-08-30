#!/usr/bin/env python3
"""Download the MoraLink public data release from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moralink.artifacts import download_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory in which to store the downloaded release",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision to download",
    )
    args = parser.parse_args()

    output_path = download_release(args.output_dir, revision=args.revision)
    print(json.dumps({"path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
