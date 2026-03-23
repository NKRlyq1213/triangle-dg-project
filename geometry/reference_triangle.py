from __future__ import annotations

import numpy as np


def reference_triangle_vertices() -> np.ndarray:
    """
    Return the reference triangle vertices in Hesthaven's (r, s) coordinates.

    Vertex convention:
        v1 = (-1, -1)
        v2 = ( 1, -1)
        v3 = (-1,  1)

    Returns
    -------
    np.ndarray
        Array of shape (3, 2).
    """
    return np.array(
        [
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [-1.0,  1.0],
        ],
        dtype=float,
    )


def reference_triangle_area() -> float:
    """
    Area of the reference triangle in (r, s).
    """
    return 2.0


def reference_triangle_centroid() -> np.ndarray:
    """
    Centroid of the reference triangle in (r, s).
    """
    verts = reference_triangle_vertices()
    return np.mean(verts, axis=0)