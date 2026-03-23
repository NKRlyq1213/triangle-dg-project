from __future__ import annotations

import numpy as np

from .reference_triangle import reference_triangle_vertices


# Raw barycentric coordinates in the old note are associated with the old vertex order:
#   old v1 = (0, 1), old v2 = (0, 0), old v3 = (1, 0)
#
# We now want the new reference triangle order:
#   new v1 = (-1, -1), new v2 = ( 1, -1), new v3 = (-1,  1)
#
# The affine map from old standard triangle to the new (r, s) triangle is:
#   (x, y) -> (r, s) = (2x - 1, 2y - 1)
#
# Under this identification:
#   old v2 -> new v1
#   old v3 -> new v2
#   old v1 -> new v3
#
# Therefore, if bary_raw = [lambda_old_v1, lambda_old_v2, lambda_old_v3],
# then the corresponding barycentric coordinates with respect to the NEW
# reference triangle ordering are:
#   bary_new = [lambda_old_v2, lambda_old_v3, lambda_old_v1]
_RAW_TO_NEW_REF_PERM = np.array([1, 2, 0], dtype=int)


def barycentric_to_cartesian(
    bary: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Convert barycentric coordinates to Cartesian/reference coordinates.

    Parameters
    ----------
    bary : np.ndarray
        Shape (..., 3), barycentric coordinates.
    vertices : np.ndarray
        Shape (3, 2), triangle vertices.

    Returns
    -------
    np.ndarray
        Shape (..., 2), point coordinates.
    """
    bary = np.asarray(bary, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    if bary.shape[-1] != 3:
        raise ValueError("bary must have last dimension 3.")
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    return bary @ vertices


def cartesian_to_barycentric(
    xy: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Convert point coordinates to barycentric coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Shape (..., 2), point coordinates.
    vertices : np.ndarray
        Shape (3, 2), triangle vertices.

    Returns
    -------
    np.ndarray
        Shape (..., 3), barycentric coordinates.
    """
    xy = np.asarray(xy, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")
    if xy.shape[-1] != 2:
        raise ValueError("xy must have last dimension 2.")

    v1, v2, v3 = vertices
    A = np.array(
        [
            [v1[0], v2[0], v3[0]],
            [v1[1], v2[1], v3[1]],
            [1.0,   1.0,   1.0],
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
    """
    bary = cartesian_to_barycentric(xy, vertices)
    return np.all(bary >= -tol, axis=-1)


def raw_barycentric_to_reference_rs(bary_raw: np.ndarray) -> np.ndarray:
    """
    Convert raw barycentric coordinates from the old-note convention directly
    to points on the NEW reference triangle in (r, s).

    Parameters
    ----------
    bary_raw : np.ndarray
        Shape (..., 3), raw barycentric coordinates following the old-note
        vertex order.

    Returns
    -------
    np.ndarray
        Shape (..., 2), coordinates on the new reference triangle in (r, s).
    """
    bary_raw = np.asarray(bary_raw, dtype=float)

    if bary_raw.shape[-1] != 3:
        raise ValueError("bary_raw must have last dimension 3.")

    bary_new = bary_raw[..., _RAW_TO_NEW_REF_PERM]
    verts = reference_triangle_vertices()
    return barycentric_to_cartesian(bary_new, verts)