from __future__ import annotations

import numpy as np


def safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using a direct linear solver.
    """
    return np.linalg.solve(A, b)


def matrix_condition_number(A: np.ndarray) -> float:
    """
    Return the 2-norm condition number of a matrix.
    """
    return float(np.linalg.cond(A))