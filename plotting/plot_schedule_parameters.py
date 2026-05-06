from calf_plotting.paper_figures import generate_schedule_parameters
from calf_plotting.script_utils import run_single_figure


if __name__ == "__main__":
    run_single_figure(
        generate_schedule_parameters,
        "Generate the schedule_parameters article figure.",
    )
