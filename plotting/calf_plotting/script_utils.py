from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from .paper_figures import PLOTTING_ROOT, set_data_root


def run_single_figure(
    figure_builder: Callable[[Path | None], Path], description: str
) -> None:
    parser = argparse.ArgumentParser(description=description)
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
        help="Directory for the generated PDF. Defaults to the repository gfx/ directory.",
    )
    args = parser.parse_args()
    set_data_root(args.data_root)
    print(figure_builder(args.output_dir))
