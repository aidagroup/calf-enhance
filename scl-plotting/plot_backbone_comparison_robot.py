from calf_plotting.paper_figures import generate_backbone_comparison_robot
from calf_plotting.script_utils import run_single_figure


if __name__ == "__main__":
    run_single_figure(
        generate_backbone_comparison_robot,
        "Generate the backbone_comparison_robot article figure.",
    )
