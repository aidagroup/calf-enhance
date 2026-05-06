from calf_plotting.paper_figures import generate_goal_reaching_rate_comparison
from calf_plotting.script_utils import run_single_figure


if __name__ == "__main__":
    run_single_figure(
        generate_goal_reaching_rate_comparison,
        "Generate the goal_reaching_rate_comparison article figure.",
    )
