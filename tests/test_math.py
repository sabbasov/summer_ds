import os
import sys

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.gradient_descent import derivative
from scripts.la_foundations import calculate_determinant


def test_derivative_calculation() -> None:
    """Verify our calculus derivative logic works."""
    assert derivative(2.0) == 0.0
    assert derivative(5.0) == 6.0


def test_linear_algebra_determinant() -> None:
    """Verify that our system computes a matrix determinant accurately."""
    # Matrix: [[1, 3], [5, 7]] -> Determinant = (1*7) - (3*5) = 7 - 15 = -8
    matrix = np.array([[1, 3], [5, 7]])
    
    calculated_det = calculate_determinant(matrix)
    
    assert round(calculated_det, 1) == -8.0