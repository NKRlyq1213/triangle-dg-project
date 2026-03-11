from __future__ import annotations

import numpy as np


def barycentric_to_cartesian(
    bary: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Convert barycentric coordinates to Cartesian coordinates.

    Parameters
    ----------
    bary : np.ndarray
        Shape (..., 3), barycentric coordinates.
    vertices : np.ndarray
        Shape (3, 2), triangle vertices.

    Returns
    -------
    np.ndarray
        Shape (..., 2), Cartesian coordinates.
    """
    bary = np.asarray(bary, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    return bary @ vertices


def cartesian_to_barycentric(
    xy: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Convert Cartesian coordinates to barycentric coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Shape (..., 2), Cartesian points.
    vertices : np.ndarray
        Shape (3, 2), triangle vertices.

    Returns
    -------
    np.ndarray
        Shape (..., 3), barycentric coordinates.
    """
    xy = np.asarray(xy, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    v1, v2, v3 = vertices
    A = np.array(
        [
            [v1[0], v2[0], v3[0]],
            [v1[1], v2[1], v3[1]],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    if xy.ndim == 1:
        rhs = np.array([xy[0], xy[1], 1.0], dtype=float)
        return np.linalg.solve(A, rhs)

    out = []
    for p in xy:
        rhs = np.array([p[0], p[1], 1.0], dtype=float)
        out.append(np.linalg.solve(A, rhs))
    return np.array(out, dtype=float)


def is_inside_triangle(
    xy: np.ndarray,
    vertices: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Check whether points are inside or on the boundary of a triangle.

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    bary = cartesian_to_barycentric(xy, vertices)
    return np.all(bary >= -tol, axis=-1)