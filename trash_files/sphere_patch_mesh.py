from __future__ import annotations

import numpy as np

from geometry.affine_map import map_ref_to_phys_points


# ---------------------------------------------------------------------
# Patch vertices on the 2D square [-R, R]^2
#
# Vertex order matters:
#   local (0,0), (1,0), (0,1)
#
# We choose the ordering to match the patch-local coordinates used in the
# sphere mapping formulas.
# ---------------------------------------------------------------------

_UNIT_PATCH_VERTICES: dict[int, np.ndarray] = {
    # T1: first quadrant, inner triangle
    1: np.array([
        [0.0, 0.0],   # local (0,0)
        [1.0, 0.0],   # local (1,0)
        [0.0, 1.0],   # local (0,1)
    ], dtype=float),

    # T2: first quadrant, outer triangle
    2: np.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=float),

    # T3: second quadrant, inner triangle
    3: np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=float),

    # T4: second quadrant, outer triangle
    4: np.array([
        [-1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=float),

    # T5: third quadrant, inner triangle
    5: np.array([
        [0.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=float),

    # T6: third quadrant, outer triangle
    6: np.array([
        [-1.0, -1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=float),

    # T7: fourth quadrant, inner triangle
    7: np.array([
        [0.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
    ], dtype=float),

    # T8: fourth quadrant, outer triangle
    8: np.array([
        [1.0, -1.0],
        [0.0, -1.0],
        [1.0, 0.0],
    ], dtype=float),
}


def patch_vertices_2d(patch_id: int, *, radius: float = 1.0) -> np.ndarray:
    """
    Return the three vertices of T_i in the physical square [-R, R]^2.

    Parameters
    ----------
    patch_id : int
        1,2,...,8
    radius : float
        Square half-width and sphere radius.

    Returns
    -------
    np.ndarray
        Shape (3, 2)
    """
    if patch_id not in _UNIT_PATCH_VERTICES:
        raise ValueError("patch_id must be one of 1,...,8.")
    if radius <= 0:
        raise ValueError("radius must be positive.")

    return radius * _UNIT_PATCH_VERTICES[patch_id]


def all_patch_vertices_2d(*, radius: float = 1.0) -> dict[int, np.ndarray]:
    """Return all patch triangles T1,...,T8."""
    return {pid: patch_vertices_2d(pid, radius=radius) for pid in range(1, 9)}


def local_xy_to_reference_rs(local_xy: np.ndarray) -> np.ndarray:
    """
    Convert local triangle coordinates (x, y) to repo reference triangle (r, s).

    local triangle:
        x >= 0, y >= 0, x + y <= 1

    repo reference triangle:
        v1 = (-1,-1), v2 = (1,-1), v3 = (-1,1)

    affine relation:
        r = 2x - 1
        s = 2y - 1
    """
    local_xy = np.asarray(local_xy, dtype=float)
    if local_xy.ndim != 2 or local_xy.shape[1] != 2:
        raise ValueError("local_xy must have shape (Np, 2).")

    x = local_xy[:, 0]
    y = local_xy[:, 1]

    if np.any(x < -1e-12) or np.any(y < -1e-12) or np.any(x + y > 1.0 + 1e-12):
        raise ValueError("local_xy must satisfy x>=0, y>=0, x+y<=1.")

    r = 2.0 * x - 1.0
    s = 2.0 * y - 1.0
    return np.column_stack([r, s])


def map_local_xy_to_patch_xy(
    local_xy: np.ndarray,
    patch_id: int,
    *,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Map local triangle points (x,y) to physical 2D points on T_i.

    We reuse repo's affine map:
        geometry.affine_map.map_ref_to_phys_points
    """
    rs = local_xy_to_reference_rs(local_xy)
    verts = patch_vertices_2d(patch_id, radius=radius)
    return map_ref_to_phys_points(rs, verts)


def uniform_local_submesh(nsub: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a uniform triangular submesh on the local triangle:
        x >= 0, y >= 0, x + y <= 1

    Parameters
    ----------
    nsub : int
        Number of subdivisions per edge.

    Returns
    -------
    points_local : np.ndarray
        Shape (Npts, 2)
    etoV : np.ndarray
        Shape (K, 3), connectivity of small triangles
    """
    if nsub < 1:
        raise ValueError("nsub must be >= 1.")

    points = []
    id_map = {}

    # grid points
    for i in range(nsub + 1):
        for j in range(nsub + 1 - i):
            idx = len(points)
            id_map[(i, j)] = idx
            points.append([i / nsub, j / nsub])

    points_local = np.array(points, dtype=float)

    # connectivity
    tris = []
    for i in range(nsub):
        for j in range(nsub - i):
            a = id_map[(i, j)]
            b = id_map[(i + 1, j)]
            c = id_map[(i, j + 1)]
            tris.append([a, b, c])

            # second triangle only if the upper-right point exists
            if i + j <= nsub - 2:
                d = id_map[(i + 1, j + 1)]
                tris.append([b, d, c])

    etoV = np.array(tris, dtype=int)
    return points_local, etoV


def patch_submesh_2d(
    patch_id: int,
    *,
    nsub: int,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the subdivided mesh on patch T_i in physical 2D coordinates.

    Returns
    -------
    points_phys : np.ndarray
        Shape (Npts, 2)
    etoV : np.ndarray
        Shape (K, 3)
    """
    points_local, etoV = uniform_local_submesh(nsub)
    points_phys = map_local_xy_to_patch_xy(points_local, patch_id, radius=radius)
    return points_phys, etoV