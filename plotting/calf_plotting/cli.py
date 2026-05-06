from __future__ import annotations

import argparse
from pathlib import Path

from .paper_figures import PLOTTING_ROOT, generate_all_figures, set_data_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all article figures from the local plotting snapshot."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PLOTTING_ROOT / "expdata" / "cleared-enriched",
        help="Directory with environment CSV folders. Defaults to expdata/cleared-enriched.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated PDFs. Defaults to the repository gfx/ directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_data_root(args.data_root)
    for path in generate_all_figures(args.output_dir):
        print(path)
