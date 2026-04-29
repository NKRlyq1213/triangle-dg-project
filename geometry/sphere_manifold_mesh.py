from __future__ import annotations

import numpy as np


def _normalize_to_sphere(point: np.ndarray, R: float) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    norm = float(np.linalg.norm(point))
    if norm <= 0.0:
        raise ValueError("Cannot project the zero vector to the sphere.")
    return R * point / norm


def _add_projected_vertex(
    point: np.ndarray,
    R: float,
    vertex_map: dict[tuple[float, float, float], int],
    vertices: list[np.ndarray],
    ndigits: int,
) -> int:
    xyz = _normalize_to_sphere(point, R=R)
    key = tuple(round(float(v), ndigits) for v in xyz)
    if key in vertex_map:
        return vertex_map[key]

    idx = len(vertices)
    vertex_map[key] = idx
    vertices.append(xyz)
    return idx


def _subdivide_octahedron_face(
    base_vertices: np.ndarray,
    n_div: int,
    R: float,
    vertex_map: dict[tuple[float, float, float], int],
    vertices: list[np.ndarray],
    ndigits: int,
) -> list[tuple[int, int, int]]:
    local_to_global: dict[tuple[int, int], int] = {}
    v0, v1, v2 = np.asarray(base_vertices, dtype=float)

    for i in range(n_div + 1):
        for j in range(n_div + 1 - i):
            a = 1.0 - (i + j) / n_div
            b = i / n_div
            c = j / n_div
            point = a * v0 + b * v1 + c * v2
            local_to_global[(i, j)] = _add_projected_vertex(
                point=point,
                R=R,
                vertex_map=vertex_map,
                vertices=vertices,
                ndigits=ndigits,
            )

    elems: list[tuple[int, int, int]] = []
    for i in range(n_div):
        for j in range(n_div - i):
            v00 = local_to_global[(i, j)]
            v10 = local_to_global[(i + 1, j)]
            v01 = local_to_global[(i, j + 1)]
            elems.append((v00, v10, v01))

            if i + j < n_div - 1:
                v11 = local_to_global[(i + 1, j + 1)]
                elems.append((v10, v11, v01))

    return elems


def generate_spherical_octahedron_mesh(
    n_div: int,
    R: float = 1.0,
    ndigits: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a closed triangular sphere mesh by subdividing an octahedron.

    Parameters
    ----------
    n_div : int
        Number of subdivisions per octahedron edge. The mesh has
        ``8 * n_div**2`` triangular elements.
    R : float
        Sphere radius.
    ndigits : int
        Rounding digits used to merge shared projected vertices.

    Returns
    -------
    nodes_xyz : np.ndarray
        Sphere vertices, shape (Nv, 3).
    EToV : np.ndarray
        Element-to-vertex connectivity, shape (K, 3), outward-oriented.
    """
    if n_div < 1:
        raise ValueError("n_div must be >= 1.")
    if R <= 0.0:
        raise ValueError("R must be positive.")

    base_nodes = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )

    # Outward orientation for each octahedron face.
    base_faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=int,
    )

    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[np.ndarray] = []
    elems: list[tuple[int, int, int]] = []

    for face in base_faces:
        elems.extend(
            _subdivide_octahedron_face(
                base_vertices=base_nodes[face],
                n_div=n_div,
                R=R,
                vertex_map=vertex_map,
                vertices=vertices,
                ndigits=ndigits,
            )
        )

    nodes_xyz = np.vstack(vertices).astype(float)
    EToV = np.asarray(elems, dtype=int)

    return nodes_xyz, EToV


def spherical_mesh_hmin(nodes_xyz: np.ndarray, EToV: np.ndarray) -> float:
    """
    Return the smallest straight-edge chord length in a triangular sphere mesh.
    """
    nodes_xyz = np.asarray(nodes_xyz, dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    if nodes_xyz.ndim != 2 or nodes_xyz.shape[1] != 3:
        raise ValueError("nodes_xyz must have shape (Nv, 3).")
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    tri = nodes_xyz[EToV]
    e01 = np.linalg.norm(tri[:, 1, :] - tri[:, 0, :], axis=1)
    e12 = np.linalg.norm(tri[:, 2, :] - tri[:, 1, :], axis=1)
    e20 = np.linalg.norm(tri[:, 0, :] - tri[:, 2, :], axis=1)
    return float(np.min(np.minimum(np.minimum(e01, e12), e20)))
