from typing import Any

import numpy as np


def calculate_determinant(matrix: np.ndarray[Any, Any]) -> float:
    """Compute the matrix determinant and return it as a pure float."""
    return float(np.linalg.det(matrix))


def run_linear_algebra_pipeline() -> None:
    """Execute the core mathematical array operations workspace pipeline."""
    # create a 2x2 matrix
    matrix1: np.ndarray[Any, Any] = np.array([[1, 3], [5, 7]])
    print("2x2 Matrix:\n", matrix1)

    # create a 3x3 matrix
    matrix2: np.ndarray[Any, Any] = np.array([[2, 3, 5], [7, 14, 21], [1, 3, 5]])
    print("\n3x3 Matrix:\n", matrix2)

    matrix3: np.ndarray[Any, Any] = np.array([[1, 3], [5, 7]])
    matrix4: np.ndarray[Any, Any] = np.array([[2, 6], [4, 8]])

    result: np.ndarray[Any, Any] = np.dot(matrix3, matrix4)
    print("\nmatrix3 x matrix4: \n", result)

    transposed_result: np.ndarray[Any, Any] = np.transpose(result)
    print("\ntransposed result: \n", transposed_result)

    inversed_result: np.ndarray[Any, Any] = np.linalg.inv(result)
    print("\ninversed result: \n", inversed_result)

    dot_result_inverse: np.ndarray[Any, Any] = np.dot(result, inversed_result)
    print("\ndot product of result and its inverse: \n", dot_result_inverse)

    det_result: float = calculate_determinant(result)
    print("\ndeterminant of result: \n", det_result)

    result_flat: np.ndarray[Any, Any] = result.flatten()
    print("\nflattened result: \n", result_flat)

    eigenvalues, eigenvectors = np.linalg.eig(result)
    print(f"\neigenvalues: {eigenvalues}\n")
    print(f"eigenvectors:\n{eigenvectors}\n")


if __name__ == "__main__":
    run_linear_algebra_pipeline()