import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_ellipse(a=3, b=2):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Add the ellipse patch
    ellipse = Ellipse(
        (0, 2), 2 * a, 2 * b, edgecolor="blue", facecolor="none", linewidth=2
    )
    ax.add_patch(ellipse)

    # Plot the center point
    ax.plot(0, 2, "ro", markersize=8)

    # Plot the semi-major and semi-minor axes
    ax.plot([-a, a], [2, 2], "r--", linewidth=1)
    ax.plot([0, 0], [2 - b, 2 + b], "r--", linewidth=1)

    # Show grid
    ax.grid(True)

    # Set limits with some padding
    padding = max(a, b) * 0.5
    ax.set_xlim(-a - padding, a + padding)
    ax.set_ylim(2 - b - padding, 2 + b + padding)

    # Add labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Ellipse: $\\frac{{x^2}}{{{a}^2}} + \\frac{{(y-2)^2}}{{{b}^2}} = 1$")

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Default values for a and b
    a, b = 3, 2
    plot_ellipse(a, b)
