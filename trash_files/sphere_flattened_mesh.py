from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FlattenedSphereRawMesh:
    """
    Flattened 8-face sphere mesh on the global square [-1,1]^2.

    中文：
    - nodes: 全域 square 上的節點座標
    - EToV: 每個 triangle element 的 vertex connectivity
    - face_ids: 每個 element 屬於 T1,...,T8 哪一個基礎三角形
    - nsub: 每個 Ti 的邊被切成 nsub 段
    """

    nsub: int
    nodes: np.ndarray      # shape (Nv, 2)
    EToV: np.ndarray       # shape (K, 3)
    face_ids: np.ndarray   # shape (K,), values in 1,...,8


def base_triangles_flattened() -> dict[int, np.ndarray]:
    """
    Return the 8 base triangles on the flattened square [-1,1]^2.

    Vertex order is chosen to be CCW for every Ti.

    Points:
        C  = ( 0,  0)  north pole in flattened layout
        R  = ( 1,  0)  equator
        T  = ( 0,  1)  equator
        L  = (-1,  0)  equator
        B  = ( 0, -1)  equator

        TR = ( 1,  1)  south-pole copy
        TL = (-1,  1)  south-pole copy
        BL = (-1, -1)  south-pole copy
        BR = ( 1, -1)  south-pole copy
    """
    C = np.array([0.0, 0.0])
    R = np.array([1.0, 0.0])
    T = np.array([0.0, 1.0])
    L = np.array([-1.0, 0.0])
    B = np.array([0.0, -1.0])

    TR = np.array([1.0, 1.0])
    TL = np.array([-1.0, 1.0])
    BL = np.array([-1.0, -1.0])
    BR = np.array([1.0, -1.0])

    return {
        1: np.array([C, R, T], dtype=float),
        2: np.array([TR, T, R], dtype=float),
        3: np.array([C, T, L], dtype=float),
        4: np.array([TL, L, T], dtype=float),
        5: np.array([C, L, B], dtype=float),
        6: np.array([BL, B, L], dtype=float),
        7: np.array([C, B, R], dtype=float),
        8: np.array([BR, R, B], dtype=float),
    }


def triangle_signed_area(vertices: np.ndarray) -> float:
    """
    Signed area of one triangle.

    Positive means CCW orientation.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3,2).")

    p0, p1, p2 = vertices
    return 0.5 * float(
        (p1[0] - p0[0]) * (p2[1] - p0[1])
        - (p1[1] - p0[1]) * (p2[0] - p0[0])
    )


def assert_base_triangles_ccw() -> None:
    """
    Validate that all base triangles are CCW.
    """
    for face_id, vertices in base_triangles_flattened().items():
        area = triangle_signed_area(vertices)
        if area <= 0.0:
            raise ValueError(f"T{face_id} is not CCW. signed_area={area}")


def _node_key(x: float, y: float, *, ndigits: int = 14) -> tuple[float, float]:
    """
    Key for deduplicating global nodes.

    Because coordinates are rational multiples of 1/nsub, rounding is safe here.
    """
    return round(float(x), ndigits), round(float(y), ndigits)


def _get_or_create_node(
    nodes: list[list[float]],
    node_map: dict[tuple[float, float], int],
    point: np.ndarray,
) -> int:
    """
    Deduplicate flattened square nodes.
    """
    key = _node_key(point[0], point[1])
    if key not in node_map:
        node_map[key] = len(nodes)
        nodes.append([float(point[0]), float(point[1])])
    return node_map[key]


def _local_submesh_indices(nsub: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], np.ndarray]:
    """
    Build local integer grid on the reference local triangle.

    Local integer coordinates:
        i >= 0, j >= 0, i+j <= nsub

    Returns
    -------
    grid:
        List of integer pairs.
    id_map:
        Map (i,j) -> local node id.
    local_EToV:
        Sub-triangle connectivity on local nodes, shape (nsub^2, 3).
        All elements are CCW in local coordinates.
    """
    if nsub < 1:
        raise ValueError("nsub must be >= 1.")

    grid: list[tuple[int, int]] = []
    id_map: dict[tuple[int, int], int] = {}

    for i in range(nsub + 1):
        for j in range(nsub + 1 - i):
            id_map[(i, j)] = len(grid)
            grid.append((i, j))

    tris: list[list[int]] = []

    for i in range(nsub):
        for j in range(nsub - i):
            a = id_map[(i, j)]
            b = id_map[(i + 1, j)]
            c = id_map[(i, j + 1)]
            tris.append([a, b, c])

            if i + j <= nsub - 2:
                d = id_map[(i + 1, j + 1)]
                tris.append([b, d, c])

    return grid, id_map, np.asarray(tris, dtype=int)


def _local_integer_point_to_global(
    i: int,
    j: int,
    *,
    nsub: int,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Map local integer coordinate (i,j) to global flattened square coordinate.

    Local physical coordinates:
        xi  = i / nsub
        eta = j / nsub

    Global:
        P = v0 + xi*(v1-v0) + eta*(v2-v0)
    """
    xi = float(i) / float(nsub)
    eta = float(j) / float(nsub)

    v0, v1, v2 = vertices
    return v0 + xi * (v1 - v0) + eta * (v2 - v0)


def build_flattened_sphere_raw_mesh(nsub: int) -> FlattenedSphereRawMesh:
    """
    Build the flattened 8-face mesh on [-1,1]^2.

    Parameters
    ----------
    nsub:
        Number of subdivisions per edge of each Ti.

    Returns
    -------
    FlattenedSphereRawMesh

    Expected element count:
        K = 8 * nsub^2
    """
    if nsub < 1:
        raise ValueError("nsub must be >= 1.")

    assert_base_triangles_ccw()

    nodes: list[list[float]] = []
    node_map: dict[tuple[float, float], int] = {}

    EToV_list: list[list[int]] = []
    face_id_list: list[int] = []

    base_tris = base_triangles_flattened()
    local_grid, _, local_EToV = _local_submesh_indices(nsub)

    for face_id in range(1, 9):
        vertices = base_tris[face_id]

        # Build local node id -> global node id map for this base triangle.
        local_to_global: dict[int, int] = {}

        for local_id, (i, j) in enumerate(local_grid):
            point = _local_integer_point_to_global(
                i,
                j,
                nsub=nsub,
                vertices=vertices,
            )
            global_id = _get_or_create_node(nodes, node_map, point)
            local_to_global[local_id] = global_id

        for tri in local_EToV:
            global_tri = [
                local_to_global[int(tri[0])],
                local_to_global[int(tri[1])],
                local_to_global[int(tri[2])],
            ]

            # Safety check: enforce CCW after mapping.
            tri_vertices = np.asarray([nodes[idx] for idx in global_tri], dtype=float)
            area = triangle_signed_area(tri_vertices)
            if area <= 0.0:
                raise RuntimeError(
                    f"Generated non-CCW triangle on T{face_id}. area={area}"
                )

            EToV_list.append(global_tri)
            face_id_list.append(face_id)

    nodes_arr = np.asarray(nodes, dtype=float)
    EToV = np.asarray(EToV_list, dtype=int)
    face_ids = np.asarray(face_id_list, dtype=int)

    expected_K = 8 * nsub**2
    if EToV.shape[0] != expected_K:
        raise RuntimeError(
            f"Unexpected element count: got {EToV.shape[0]}, expected {expected_K}."
        )

    return FlattenedSphereRawMesh(
        nsub=int(nsub),
        nodes=nodes_arr,
        EToV=EToV,
        face_ids=face_ids,
    )


def mesh_element_areas(nodes: np.ndarray, EToV: np.ndarray) -> np.ndarray:
    """
    Return signed areas of all elements.
    """
    nodes = np.asarray(nodes, dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    areas = np.zeros(EToV.shape[0], dtype=float)
    for k in range(EToV.shape[0]):
        areas[k] = triangle_signed_area(nodes[EToV[k]])
    return areas


def mesh_size_summary(mesh: FlattenedSphereRawMesh) -> dict[str, int]:
    """
    Basic mesh size summary.
    """
    return {
        "nsub": int(mesh.nsub),
        "num_nodes": int(mesh.nodes.shape[0]),
        "num_elements": int(mesh.EToV.shape[0]),
        "num_elements_per_face": int(mesh.nsub**2),
        "expected_num_elements": int(8 * mesh.nsub**2),
    }