from __future__ import annotations

import numpy as np


def _vertex_id(i: int, j: int, nx: int) -> int:
    """
    Map structured grid indices (i, j) to a global vertex id.

    We use row-major ordering:
        vid = j * (nx + 1) + i

    where
        i = 0, ..., nx
        j = 0, ..., ny
    """
    return j * (nx + 1) + i


def structured_square_tri_mesh(
    nx: int,
    ny: int,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    diagonal: str = "main",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a structured triangular mesh on a rectangular domain.

    The rectangle [x0, x1] x [y0, y1] is first divided into nx * ny
    rectangles, then each rectangle is split into 2 triangles.

    Parameters
    ----------
    nx, ny : int
        Number of rectangular cells in x- and y-direction.
        The final number of triangles is 2 * nx * ny.
    xlim, ylim : tuple[float, float]
        Domain bounds.
    diagonal : str
        How to split each rectangle:
        - "main": split along lower-left -> upper-right
        - "anti": split along upper-left -> lower-right

    Returns
    -------
    VX : np.ndarray
        x-coordinates of vertices, shape (Nv,)
    VY : np.ndarray
        y-coordinates of vertices, shape (Nv,)
    EToV : np.ndarray
        Element-to-vertex connectivity, shape (K, 3), dtype=int

    Notes
    -----
    All triangles are returned with counterclockwise vertex ordering.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")

    x0, x1 = xlim
    y0, y1 = ylim

    if not (x1 > x0 and y1 > y0):
        raise ValueError("Require x1 > x0 and y1 > y0.")

    if diagonal not in {"main", "anti"}:
        raise ValueError("diagonal must be either 'main' or 'anti'.")

    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    VX = X.reshape(-1)
    VY = Y.reshape(-1)

    elements: list[list[int]] = []

    for j in range(ny):
        for i in range(nx):
            v00 = _vertex_id(i,     j,     nx)  # lower-left
            v10 = _vertex_id(i + 1, j,     nx)  # lower-right
            v01 = _vertex_id(i,     j + 1, nx)  # upper-left
            v11 = _vertex_id(i + 1, j + 1, nx)  # upper-right

            if diagonal == "main":
                # diagonal: v00 -> v11
                # triangles:
                #   T1 = (v00, v10, v11)
                #   T2 = (v00, v11, v01)
                elements.append([v00, v10, v11])
                elements.append([v00, v11, v01])
            else:
                # diagonal: v01 -> v10
                # triangles:
                #   T1 = (v00, v10, v01)
                #   T2 = (v10, v11, v01)
                elements.append([v00, v10, v01])
                elements.append([v10, v11, v01])

    EToV = np.asarray(elements, dtype=int)
    return VX, VY, EToV


def triangle_signed_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Signed area of one triangle.

    Positive  -> counterclockwise
    Negative  -> clockwise
    Zero      -> degenerate
    """
    return 0.5 * (
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p2[1] - p1[1]) * (p3[0] - p1[0])
    )


def element_areas_and_orientations(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute element signed areas and a boolean orientation flag.

    Returns
    -------
    signed_areas : np.ndarray
        Shape (K,)
    is_ccw : np.ndarray
        Shape (K,), True if signed area > 0
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)

    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    signed_areas = np.zeros(EToV.shape[0], dtype=float)

    for k, (a, b, c) in enumerate(EToV):
        p1 = np.array([VX[a], VY[a]], dtype=float)
        p2 = np.array([VX[b], VY[b]], dtype=float)
        p3 = np.array([VX[c], VY[c]], dtype=float)
        signed_areas[k] = triangle_signed_area(p1, p2, p3)

    is_ccw = signed_areas > 0.0
    return signed_areas, is_ccw


def validate_mesh_orientation(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    tol: float = 1e-14,
) -> None:
    """
    Validate that all triangles are nondegenerate and counterclockwise.
    """
    signed_areas, is_ccw = element_areas_and_orientations(VX, VY, EToV)

    if np.any(np.abs(signed_areas) <= tol):
        bad = np.where(np.abs(signed_areas) <= tol)[0]
        raise ValueError(f"Degenerate triangles found at element indices: {bad.tolist()}")

    if np.any(~is_ccw):
        bad = np.where(~is_ccw)[0]
        raise ValueError(f"Clockwise triangles found at element indices: {bad.tolist()}")
    