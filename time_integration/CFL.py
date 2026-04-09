from __future__ import annotations

import numpy as np


def triangle_edge_lengths(vertices: np.ndarray) -> np.ndarray:
    """
    Return the three edge lengths of one triangle.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (3, 2)

    Returns
    -------
    np.ndarray
        Shape (3,), edge lengths
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    p1, p2, p3 = vertices
    l12 = np.linalg.norm(p2 - p1)
    l23 = np.linalg.norm(p3 - p2)
    l31 = np.linalg.norm(p1 - p3)
    return np.array([l12, l23, l31], dtype=float)


def triangle_area(vertices: np.ndarray) -> float:
    """
    Geometric area of one triangle.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    p1, p2, p3 = vertices
    area = 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p2[1] - p1[1]) * (p3[0] - p1[0])
    )
    return float(area)


def triangle_min_altitude(vertices: np.ndarray) -> float:
    """
    Minimum altitude of one triangle.

    Using
        h_i = 2A / |edge_i|
    and taking the minimum over the three edges.
    """
    vertices = np.asarray(vertices, dtype=float)
    lengths = triangle_edge_lengths(vertices)
    area = triangle_area(vertices)

    if area <= 0.0:
        raise ValueError("Degenerate triangle detected.")

    altitudes = 2.0 * area / lengths
    return float(np.min(altitudes))


def mesh_min_altitude(VX: np.ndarray, VY: np.ndarray, EToV: np.ndarray) -> float:
    """
    Minimum triangle altitude over the whole mesh.
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)

    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    hmin = np.inf
    for vids in EToV:
        vertices = np.column_stack([VX[vids], VY[vids]])
        hmin = min(hmin, triangle_min_altitude(vertices))

    return float(hmin)


def vmax_from_uv(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute vmax = max sqrt(u^2 + v^2) from sampled velocity components.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")

    return float(np.max(np.sqrt(u**2 + v**2)))


def cfl_dt_from_h(
    cfl: float,
    h: float,
    N: int,
    vmax: float,
) -> float:
    r"""
    CFL step size based on the requested scaling:

        dt = CFL * h / (N^2 * vmax)

    Parameters
    ----------
    cfl : float
        CFL number
    h : float
        Element size scale
    N : int
        Polynomial order
    vmax : float
        Maximum advection speed

    Returns
    -------
    float
        Stable nominal time step
    """
    if cfl <= 0.0:
        raise ValueError("cfl must be positive.")
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if N <= 0:
        raise ValueError("N must be positive.")
    if vmax <= 0.0:
        raise ValueError("vmax must be positive.")

    return float(cfl * h / (N**2 * vmax))