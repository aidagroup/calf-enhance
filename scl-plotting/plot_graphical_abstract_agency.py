from calf_plotting.paper_figures import generate_graphical_abstract_agency
from calf_plotting.script_utils import run_single_figure


if __name__ == "__main__":
    run_single_figure(
        generate_graphical_abstract_agency,
        "Generate the graphical abstract agency-transfer figure.",
    )
