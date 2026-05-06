from calf_plotting.paper_figures import generate_backbone_comparison_drone
from calf_plotting.script_utils import run_single_figure


if __name__ == "__main__":
    run_single_figure(
        generate_backbone_comparison_drone,
        "Generate the backbone_comparison_drone article figure.",
    )
