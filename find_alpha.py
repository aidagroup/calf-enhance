import numpy as np
from scipy.optimize import root_scalar
from scipy.special import geometric_series


def find_alpha(b: float, T: int) -> float:
    """
    Find alpha such that sum_{t=0}^{T-1} alpha^t = b

    Args:
        b: The target sum
        T: The number of terms

    Returns:
        The value of alpha that satisfies the equation
    """

    def equation(alpha):
        if alpha >= 1.0 - 1e-6:
            return sum(alpha**t for t in range(T)) - b  # Special case when alpha = 1
        return geometric_series(alpha, T) - b

    # Try to find root in reasonable range
    result = root_scalar(equation, bracket=[0.00000, 1.0], method="brentq")
    return result.root


# Example usage
if __name__ == "__main__":
    b = 10.0  # Example target sum
    T = 1500  # Example number of terms

    alpha = find_alpha(b, T)
    print(f"For b={b} and T={T}, alpha={alpha:.6f}")

    # Verify the solution using geometric_series
    sum_terms = geometric_series(alpha, T)
    print(f"Verification: sum of terms = {sum_terms:.6f}")
