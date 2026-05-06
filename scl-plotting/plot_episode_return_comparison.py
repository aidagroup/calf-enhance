from calf_plotting.paper_figures import generate_episode_return_comparison
from calf_plotting.script_utils import run_single_figure


if __name__ == "__main__":
    run_single_figure(
        generate_episode_return_comparison,
        "Generate the episode_return_comparison article figure.",
    )
