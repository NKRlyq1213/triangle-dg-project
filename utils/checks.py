from __future__ import annotations

import numpy as np

from geometry.barycentric import is_inside_triangle


def check_points_inside_triangle(
    xy: np.ndarray,
    vertices: np.ndarray,
    tol: float = 1e-12,
) -> bool:
    """
    Check whether all points lie inside or on the boundary of the triangle.
    """
    mask = is_inside_triangle(xy, vertices, tol=tol)
    return bool(np.all(mask))