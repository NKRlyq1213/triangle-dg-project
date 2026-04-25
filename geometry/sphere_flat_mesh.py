from __future__ import annotations

import numpy as np


def macro_patch_triangles(R: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the 8 macro triangles on [-R,R]^2.

    Returns
    -------
    VX, VY : np.ndarray
        Unique macro vertices.
    EToV : np.ndarray
        Shape (8,3), CCW vertex order.
    patch_id : np.ndarray
        Shape (8,), values 1,...,8.

    Triangle convention
    -------------------
    T1: first quadrant,  near center, Z>=0
    T2: first quadrant,  near corner, Z<=0
    T3: second quadrant, near center, Z>=0
    T4: second quadrant, near corner, Z<=0
    T5: third quadrant,  near center, Z>=0
    T6: third quadrant,  near corner, Z<=0
    T7: fourth quadrant, near center, Z>=0
    T8: fourth quadrant, near corner, Z<=0
    """
    pts = [
        (0.0, 0.0),      # 0 center
        (R, 0.0),        # 1 east
        (R, R),          # 2 northeast
        (0.0, R),        # 3 north
        (-R, R),         # 4 northwest
        (-R, 0.0),       # 5 west
        (-R, -R),        # 6 southwest
        (0.0, -R),       # 7 south
        (R, -R),         # 8 southeast
    ]

    VX = np.array([p[0] for p in pts], dtype=float)
    VY = np.array([p[1] for p in pts], dtype=float)

    EToV = np.array(
        [
            [0, 1, 3],  # T1
            [1, 2, 3],  # T2
            [0, 3, 5],  # T3
            [3, 4, 5],  # T4
            [0, 5, 7],  # T5
            [5, 6, 7],  # T6
            [0, 7, 1],  # T7
            [7, 8, 1],  # T8
        ],
        dtype=int,
    )

    patch_id = np.arange(1, 9, dtype=int)
    return VX, VY, EToV, patch_id


def _signed_area(vertices: np.ndarray) -> float:
    v0, v1, v2 = vertices
    return 0.5 * (
        (v1[0] - v0[0]) * (v2[1] - v0[1])
        - (v1[1] - v0[1]) * (v2[0] - v0[0])
    )


def _add_vertex(
    xy: tuple[float, float],
    vertex_map: dict[tuple[float, float], int],
    vertices: list[tuple[float, float]],
    ndigits: int = 14,
) -> int:
    key = (round(float(xy[0]), ndigits), round(float(xy[1]), ndigits))
    if key in vertex_map:
        return vertex_map[key]
    idx = len(vertices)
    vertex_map[key] = idx
    vertices.append((float(xy[0]), float(xy[1])))
    return idx


def _subdivide_one_triangle(
    vertices: np.ndarray,
    n_sub: int,
    vertex_map: dict[tuple[float, float], int],
    global_vertices: list[tuple[float, float]],
    ndigits: int = 14,
) -> list[tuple[int, int, int]]:
    """
    Uniformly subdivide one CCW triangle into n_sub^2 small CCW triangles.
    """
    if n_sub < 1:
        raise ValueError("n_sub must be >= 1.")

    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3,2).")

    if _signed_area(vertices) <= 0.0:
        raise ValueError("Input triangle must be CCW.")

    local_to_global: dict[tuple[int, int], int] = {}

    for i in range(n_sub + 1):
        for j in range(n_sub + 1 - i):
            a = 1.0 - (i + j) / n_sub
            b = i / n_sub
            c = j / n_sub
            xy = a * vertices[0] + b * vertices[1] + c * vertices[2]
            local_to_global[(i, j)] = _add_vertex(
                (xy[0], xy[1]),
                vertex_map,
                global_vertices,
                ndigits=ndigits,
            )

    elems: list[tuple[int, int, int]] = []

    for i in range(n_sub):
        for j in range(n_sub - i):
            v00 = local_to_global[(i, j)]
            v10 = local_to_global[(i + 1, j)]
            v01 = local_to_global[(i, j + 1)]
            elems.append((v00, v10, v01))

            if i + j < n_sub - 1:
                v11 = local_to_global[(i + 1, j + 1)]
                elems.append((v10, v11, v01))

    return elems


def sphere_flat_square_mesh(
    n_sub: int = 1,
    R: float = 1.0,
    ndigits: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build refined 8-patch triangular mesh on [-R,R]^2.

    Parameters
    ----------
    n_sub : int
        Each macro triangle is subdivided into n_sub^2 small triangles.
        Total element count is 8*n_sub^2.
    R : float
        Sphere radius and square half-width.

    Returns
    -------
    VX, VY : np.ndarray
        Global vertex coordinates.
    EToV : np.ndarray
        Element-to-vertex connectivity, shape (K,3).
    elem_patch_id : np.ndarray
        Patch id per element, shape (K,).

    Notes
    -----
    This mesh is intentionally topologically flat. Sphere seam connectivity
    is a later layer.
    """
    if n_sub < 1:
        raise ValueError("n_sub must be >= 1.")

    VX0, VY0, EToV0, patch0 = macro_patch_triangles(R=R)

    vertex_map: dict[tuple[float, float], int] = {}
    vertices: list[tuple[float, float]] = []
    elems: list[tuple[int, int, int]] = []
    elem_patch: list[int] = []

    for k in range(EToV0.shape[0]):
        tri_vertices = np.column_stack([VX0[EToV0[k]], VY0[EToV0[k]]])
        sub_elems = _subdivide_one_triangle(
            tri_vertices,
            n_sub=n_sub,
            vertex_map=vertex_map,
            global_vertices=vertices,
            ndigits=ndigits,
        )
        elems.extend(sub_elems)
        elem_patch.extend([int(patch0[k])] * len(sub_elems))

    V = np.array(vertices, dtype=float)
    EToV = np.array(elems, dtype=int)
    elem_patch_id = np.array(elem_patch, dtype=int)

    return V[:, 0], V[:, 1], EToV, elem_patch_id


def element_centroids(VX: np.ndarray, VY: np.ndarray, EToV: np.ndarray) -> np.ndarray:
    VX = np.asarray(VX, dtype=float)
    VY = np.asarray(VY, dtype=float)
    EToV = np.asarray(EToV, dtype=int)
    return np.column_stack([
        np.mean(VX[EToV], axis=1),
        np.mean(VY[EToV], axis=1),
    ])


def mesh_summary(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_patch_id: np.ndarray,
) -> dict:
    """
    Basic mesh diagnostics.
    """
    VX = np.asarray(VX, dtype=float)
    VY = np.asarray(VY, dtype=float)
    EToV = np.asarray(EToV, dtype=int)
    elem_patch_id = np.asarray(elem_patch_id, dtype=int)

    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K,3).")
    if elem_patch_id.shape != (EToV.shape[0],):
        raise ValueError("elem_patch_id must have shape (K,).")

    areas = []
    for k in range(EToV.shape[0]):
        verts = np.column_stack([VX[EToV[k]], VY[EToV[k]]])
        areas.append(_signed_area(verts))

    areas = np.array(areas, dtype=float)

    return {
        "n_vertices": int(VX.size),
        "n_elements": int(EToV.shape[0]),
        "n_patches": int(np.unique(elem_patch_id).size),
        "min_area": float(np.min(areas)),
        "max_area": float(np.max(areas)),
        "total_area": float(np.sum(areas)),
        "elements_per_patch": {
            int(pid): int(np.sum(elem_patch_id == pid))
            for pid in np.unique(elem_patch_id)
        },
    }
