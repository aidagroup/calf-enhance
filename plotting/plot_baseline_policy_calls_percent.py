from calf_plotting.paper_figures import generate_baseline_policy_calls_percent
from calf_plotting.script_utils import run_single_figure


if __name__ == "__main__":
    run_single_figure(
        generate_baseline_policy_calls_percent,
        "Generate the baseline_policy_calls_percent article figure.",
    )
