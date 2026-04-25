from __future__ import annotations

import numpy as np

from data.rule_registry import load_rule
from geometry.affine_map import map_ref_to_phys_points
from geometry.sphere_mapping import local_xy_to_sphere_xyz
from geometry.sphere_metrics import sqrtG, expected_sqrtG
from geometry.sphere_patches import all_patch_ids, patch_signs
from geometry.sphere_patch_mesh import uniform_local_submesh


def submesh_element_nodes_local(
    *,
    nsub: int,
    quad_table: str = "table2",
    quad_order: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build all quadrature / DG nodes inside every local sub-element.

    Returns
    -------
    points_local:
        Local submesh vertices, shape (Nv, 2).
    etoV:
        Sub-element connectivity, shape (K, 3).
    elem_nodes_local:
        Shape (K, Nq, 2).
    """
    points_local, etoV = uniform_local_submesh(nsub)

    rule = load_rule(quad_table, quad_order)
    rs_nodes = rule["rs"]

    K = etoV.shape[0]
    Nq = rs_nodes.shape[0]

    elem_nodes_local = np.zeros((K, Nq, 2), dtype=float)

    for k in range(K):
        local_vertices = points_local[etoV[k], :]
        elem_nodes_local[k, :, :] = map_ref_to_phys_points(
            rs_nodes,
            local_vertices,
        )

    return points_local, etoV, elem_nodes_local


def flatten_elem_nodes(elem_nodes_local: np.ndarray) -> np.ndarray:
    """
    Flatten shape (K, Nq, 2) to (K*Nq, 2).
    """
    K, Nq, _ = elem_nodes_local.shape
    return elem_nodes_local.reshape(K * Nq, 2)


def local_edge_points(edge: str, *, nsample: int = 33) -> np.ndarray:
    """
    Return sample points on one local triangle edge.

    Local triangle:
        x >= 0, y >= 0, x + y <= 1

    Edges:
        "y0"   : y = 0,       x in [0,1]
        "x0"   : x = 0,       y in [0,1]
        "sum1" : x + y = 1

    Returns
    -------
    local_xy:
        Shape (nsample, 2).
    """
    t = np.linspace(0.0, 1.0, nsample)

    if edge == "y0":
        return np.column_stack([t, np.zeros_like(t)])

    if edge == "x0":
        return np.column_stack([np.zeros_like(t), t])

    if edge == "sum1":
        return np.column_stack([t, 1.0 - t])

    raise ValueError("edge must be one of 'y0', 'x0', 'sum1'.")


def map_local_edge_to_sphere(
    patch_id: int,
    edge: str,
    *,
    radius: float = 1.0,
    nsample: int = 33,
) -> np.ndarray:
    """
    Map one local edge of patch T_i to 3D sphere coordinates.
    """
    local_xy = local_edge_points(edge, nsample=nsample)
    x = local_xy[:, 0]
    y = local_xy[:, 1]

    X, Y, Z = local_xy_to_sphere_xyz(
        x,
        y,
        patch_id,
        radius=radius,
    )

    return np.column_stack([X, Y, Z])


def max_pointwise_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Max Euclidean pointwise distance between two sampled curves.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")

    return float(np.max(np.linalg.norm(a - b, axis=1)))


def test_uniform_local_submesh_element_count():
    """
    A uniformly subdivided triangle should contain nsub^2 sub-triangles.
    """
    for nsub in [1, 2, 3, 5, 8]:
        points_local, etoV = uniform_local_submesh(nsub)
        assert etoV.shape[0] == nsub**2

        # All points should remain inside the local triangle.
        x = points_local[:, 0]
        y = points_local[:, 1]
        assert np.all(x >= -1e-12)
        assert np.all(y >= -1e-12)
        assert np.all(x + y <= 1.0 + 1e-12)


def test_submesh_nodes_are_on_sphere():
    """
    All quadrature / DG nodes inside every sub-element should lie on the sphere.
    """
    radius = 1.0
    _, _, elem_nodes_local = submesh_element_nodes_local(nsub=5)
    local_xy = flatten_elem_nodes(elem_nodes_local)

    x = local_xy[:, 0]
    y = local_xy[:, 1]

    for patch_id in all_patch_ids():
        X, Y, Z = local_xy_to_sphere_xyz(
            x,
            y,
            patch_id,
            radius=radius,
        )

        err = np.max(np.abs(X**2 + Y**2 + Z**2 - radius**2))
        assert err < 1e-12


def test_submesh_nodes_match_patch_signs():
    """
    Each patch should map to the correct sphere face sign region.
    """
    radius = 1.0
    _, _, elem_nodes_local = submesh_element_nodes_local(nsub=5)
    local_xy = flatten_elem_nodes(elem_nodes_local)

    x = local_xy[:, 0]
    y = local_xy[:, 1]

    for patch_id in all_patch_ids():
        sx, sy, sz = patch_signs(patch_id)

        X, Y, Z = local_xy_to_sphere_xyz(
            x,
            y,
            patch_id,
            radius=radius,
        )

        assert np.all(sx * X >= -1e-12)
        assert np.all(sy * Y >= -1e-12)
        assert np.all(sz * Z >= -1e-12)


def test_submesh_sqrtG_is_constant():
    """
    The equal-area mapping should have constant surface Jacobian sqrtG.

    For normalized local coordinates:
        sqrtG = pi * R^2
    """
    radius = 1.0
    _, _, elem_nodes_local = submesh_element_nodes_local(nsub=5)
    local_xy = flatten_elem_nodes(elem_nodes_local)

    x = local_xy[:, 0]
    y = local_xy[:, 1]

    for patch_id in all_patch_ids():
        J = sqrtG(x, y, patch_id, radius=radius)
        err = np.max(np.abs(J - expected_sqrtG(radius=radius)))
        assert err < 1e-11


def test_equator_edges_match_between_upper_and_lower_patch():
    """
    Upper/lower patches in the same quadrant should share the equator edge.

    Pairing:
        T1.sum1 == T2.sum1
        T3.sum1 == T4.sum1
        T5.sum1 == T6.sum1
        T7.sum1 == T8.sum1
    """
    radius = 1.0
    pairs = [
        (1, "sum1", 2, "sum1"),
        (3, "sum1", 4, "sum1"),
        (5, "sum1", 6, "sum1"),
        (7, "sum1", 8, "sum1"),
    ]

    for p_left, e_left, p_right, e_right in pairs:
        a = map_local_edge_to_sphere(
            p_left,
            e_left,
            radius=radius,
            nsample=51,
        )
        b = map_local_edge_to_sphere(
            p_right,
            e_right,
            radius=radius,
            nsample=51,
        )

        err = max_pointwise_distance(a, b)
        assert err < 1e-12


def test_upper_adjacent_sector_edges_match():
    """
    Adjacent upper hemisphere patches should share meridian edges.

    With current local convention:
        T1.x0 == T3.y0
        T3.x0 == T5.y0
        T5.x0 == T7.y0
        T7.x0 == T1.y0
    """
    radius = 1.0
    pairs = [
        (1, "x0", 3, "y0"),
        (3, "x0", 5, "y0"),
        (5, "x0", 7, "y0"),
        (7, "x0", 1, "y0"),
    ]

    for p_left, e_left, p_right, e_right in pairs:
        a = map_local_edge_to_sphere(
            p_left,
            e_left,
            radius=radius,
            nsample=51,
        )
        b = map_local_edge_to_sphere(
            p_right,
            e_right,
            radius=radius,
            nsample=51,
        )

        err = max_pointwise_distance(a, b)
        assert err < 1e-12


def test_lower_adjacent_sector_edges_match():
    """
    Adjacent lower hemisphere patches should share meridian edges.

    With current local convention:
        T2.x0 == T4.y0
        T4.x0 == T6.y0
        T6.x0 == T8.y0
        T8.x0 == T2.y0
    """
    radius = 1.0
    pairs = [
        (2, "x0", 4, "y0"),
        (4, "x0", 6, "y0"),
        (6, "x0", 8, "y0"),
        (8, "x0", 2, "y0"),
    ]

    for p_left, e_left, p_right, e_right in pairs:
        a = map_local_edge_to_sphere(
            p_left,
            e_left,
            radius=radius,
            nsample=51,
        )
        b = map_local_edge_to_sphere(
            p_right,
            e_right,
            radius=radius,
            nsample=51,
        )

        err = max_pointwise_distance(a, b)
        assert err < 1e-12