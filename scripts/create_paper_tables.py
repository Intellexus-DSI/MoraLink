#!/usr/bin/env python3
"""Create Markdown and LaTeX tables from the frozen MoraLink result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from moralink.tables import create_paper_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in create_paper_tables(args.results_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
