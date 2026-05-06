from calf_plotting.paper_figures import generate_ablation_episode_return_drone_robot
from calf_plotting.script_utils import run_single_figure


if __name__ == "__main__":
    run_single_figure(
        generate_ablation_episode_return_drone_robot,
        "Generate the ablation_episode_return_drone_robot article figure.",
    )
