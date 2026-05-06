from __future__ import annotations

import argparse
from pathlib import Path

from calf_plotting.paper_figures import (
    PLOTTING_ROOT,
    generate_article_final_metrics_tables,
    set_data_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the final metrics tables used in main.tex."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PLOTTING_ROOT / "expdata" / "cleared-enriched",
        help="TD3-family data root. Defaults to expdata/cleared-enriched.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gfx"),
        help="Directory for final_metrics_tables.tex. Defaults to gfx/.",
    )
    args = parser.parse_args()

    set_data_root(args.data_root)
    print(generate_article_final_metrics_tables(args.output_dir))


if __name__ == "__main__":
    main()
