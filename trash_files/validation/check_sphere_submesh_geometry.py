from __future__ import annotations

from pathlib import Path

import numpy as np

from data.rule_registry import load_rule
from geometry.affine_map import map_ref_to_phys_points
from geometry.sphere_mapping import local_xy_to_sphere_xyz
from geometry.sphere_metrics_sdg_hardcoded import sqrtG_sdg_local as sqrtG
from geometry.sphere_metrics_sdg_hardcoded import expected_sqrtG_sdg as expected_sqrtG
from geometry.sphere_patches import all_patch_ids, patch_signs
from geometry.sphere_patch_mesh import uniform_local_submesh


def submesh_element_nodes_local(
    *,
    nsub: int,
    quad_table: str = "table1",
    quad_order: int = 4,
) -> np.ndarray:
    """
    Build all local nodes inside every sub-element.
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

    return elem_nodes_local.reshape(K * Nq, 2)


def local_edge_points(edge: str, *, nsample: int = 101) -> np.ndarray:
    t = np.linspace(0.0, 1.0, nsample)

    if edge == "y0":
        return np.column_stack([t, np.zeros_like(t)])

    if edge == "x0":
        return np.column_stack([np.zeros_like(t), t])

    if edge == "sum1":
        return np.column_stack([t, 1.0 - t])

    raise ValueError("edge must be one of 'y0', 'x0', 'sum1'.")


def map_edge(patch_id: int, edge: str, *, radius: float = 1.0) -> np.ndarray:
    local_xy = local_edge_points(edge)
    x = local_xy[:, 0]
    y = local_xy[:, 1]

    X, Y, Z = local_xy_to_sphere_xyz(
        x,
        y,
        patch_id,
        radius=radius,
    )

    return np.column_stack([X, Y, Z])


def max_curve_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def main() -> None:
    radius = 1.0
    nsub = 5
    quad_table = "table1"
    quad_order = 4

    local_xy = submesh_element_nodes_local(
        nsub=nsub,
        quad_table=quad_table,
        quad_order=quad_order,
    )

    x = local_xy[:, 0]
    y = local_xy[:, 1]

    print("=== Sphere submesh geometry diagnostics ===")
    print(f"radius     = {radius}")
    print(f"nsub       = {nsub}")
    print(f"quad rule  = {quad_table}, order={quad_order}")
    print(f"num nodes  = {local_xy.shape[0]} per patch")
    print()

    print("[1] radius identity")
    for patch_id in all_patch_ids():
        X, Y, Z = local_xy_to_sphere_xyz(
            x,
            y,
            patch_id,
            radius=radius,
        )
        err = np.max(np.abs(X**2 + Y**2 + Z**2 - radius**2))
        print(f"  patch {patch_id}: max |X^2+Y^2+Z^2-R^2| = {err:.3e}")
    print()

    print("[2] patch sign")
    for patch_id in all_patch_ids():
        sx, sy, sz = patch_signs(patch_id)
        X, Y, Z = local_xy_to_sphere_xyz(
            x,
            y,
            patch_id,
            radius=radius,
        )

        min_sx = np.min(sx * X)
        min_sy = np.min(sy * Y)
        min_sz = np.min(sz * Z)

        print(
            f"  patch {patch_id}: "
            f"min(sx*X)={min_sx:.3e}, "
            f"min(sy*Y)={min_sy:.3e}, "
            f"min(sz*Z)={min_sz:.3e}"
        )
    print()

    print("[3] sqrtG constant")
    for patch_id in all_patch_ids():
        J = sqrtG(x, y, patch_id, radius=radius)
        err = np.max(np.abs(J - expected_sqrtG(radius=radius)))
        print(f"  patch {patch_id}: max |sqrtG - pi R^2| = {err:.3e}")
    print()

    print("[4] equator shared edges")
    equator_pairs = [
        (1, "sum1", 2, "sum1"),
        (3, "sum1", 4, "sum1"),
        (5, "sum1", 6, "sum1"),
        (7, "sum1", 8, "sum1"),
    ]

    for p1, e1, p2, e2 in equator_pairs:
        err = max_curve_error(
            map_edge(p1, e1, radius=radius),
            map_edge(p2, e2, radius=radius),
        )
        print(f"  T{p1}.{e1} vs T{p2}.{e2}: max distance = {err:.3e}")
    print()

    print("[5] upper adjacent sector edges")
    upper_pairs = [
        (1, "x0", 3, "y0"),
        (3, "x0", 5, "y0"),
        (5, "x0", 7, "y0"),
        (7, "x0", 1, "y0"),
    ]

    for p1, e1, p2, e2 in upper_pairs:
        err = max_curve_error(
            map_edge(p1, e1, radius=radius),
            map_edge(p2, e2, radius=radius),
        )
        print(f"  T{p1}.{e1} vs T{p2}.{e2}: max distance = {err:.3e}")
    print()

    print("[6] lower adjacent sector edges")
    lower_pairs = [
        (2, "x0", 4, "y0"),
        (4, "x0", 6, "y0"),
        (6, "x0", 8, "y0"),
        (8, "x0", 2, "y0"),
    ]

    for p1, e1, p2, e2 in lower_pairs:
        err = max_curve_error(
            map_edge(p1, e1, radius=radius),
            map_edge(p2, e2, radius=radius),
        )
        print(f"  T{p1}.{e1} vs T{p2}.{e2}: max distance = {err:.3e}")


if __name__ == "__main__":
    main()