from __future__ import annotations

import numpy as np


def reference_triangle_vertices() -> np.ndarray:
    """
    Return the reference triangle vertices in Cartesian coordinates.

    Vertices follow the convention:
        v1 = (0, 1)
        v2 = (0, 0)
        v3 = (1, 0)

    Returns
    -------
    np.ndarray
        Array of shape (3, 2).
    """
    return np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )


def reference_triangle_area() -> float:
    """
    Area of the reference triangle.
    """
    return 0.5


def reference_triangle_centroid() -> np.ndarray:
    """
    Centroid of the reference triangle.
    """
    verts = reference_triangle_vertices()
    return np.mean(verts, axis=0)