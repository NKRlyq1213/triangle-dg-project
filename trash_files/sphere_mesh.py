from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from data.rule_registry import load_rule
from geometry.affine_map import map_ref_to_phys_points
from geometry.sphere_mapping import (
    local_xy_to_patch_angles,
    local_xy_to_sphere_xyz,
)
from geometry.sphere_metrics_sdg_hardcoded import sqrtG_sdg_local as sqrtG
from geometry.sphere_patch_mesh import uniform_local_submesh
from geometry.sphere_patches import all_patch_ids
from geometry.sphere_trace_pairs import (
    SphereTracePair,
    expected_sphere_trace_pairs_bidirectional,
)


@dataclass(frozen=True)
class SpherePatchMesh:
    """
    Precomputed 8-patch sphere mesh.

    This object stores geometry data only.
    It does not assemble DG operators or RHS.

    中文：
    - 這是後續 RHS 會用的幾何資料層。
    - mapping / metric / patch_id 都在 build 階段算好。
    - time stepping 不應重新計算這些幾何量。
    """

    radius: float
    nsub: int
    quad_table: str
    quad_order: int

    # Local submesh template shared by all patches.
    points_local_template: np.ndarray      # shape (Nv_local, 2)
    etoV_local_template: np.ndarray        # shape (K_patch, 3)

    # Element metadata.
    elem_to_patch: np.ndarray              # shape (K_total,)
    elem_to_local_id: np.ndarray           # shape (K_total,)

    # Element vertices in patch-local coordinates.
    elem_vertices_local: np.ndarray        # shape (K_total, 3, 2)

    # Volume / quadrature nodes.
    volume_nodes_local: np.ndarray         # shape (K_total, Nq, 2)
    volume_lambda: np.ndarray              # shape (K_total, Nq)
    volume_theta: np.ndarray               # shape (K_total, Nq)
    volume_xyz: np.ndarray                 # shape (K_total, Nq, 3)
    volume_sqrtG: np.ndarray               # shape (K_total, Nq)

    # Directed patch-boundary trace pairs.
    patch_trace_pairs: tuple[SphereTracePair, ...]


def _build_local_element_nodes(
    points_local: np.ndarray,
    etoV_local: np.ndarray,
    rs_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build element vertices and volume nodes on the local triangle.

    Parameters
    ----------
    points_local:
        Local submesh vertices, shape (Nv, 2).
    etoV_local:
        Local element connectivity, shape (K_patch, 3).
    rs_nodes:
        Reference triangle nodes from repo quadrature rule, shape (Nq, 2).

    Returns
    -------
    elem_vertices_local:
        Shape (K_patch, 3, 2)
    elem_nodes_local:
        Shape (K_patch, Nq, 2)
    """
    K_patch = etoV_local.shape[0]
    Nq = rs_nodes.shape[0]

    elem_vertices_local = np.zeros((K_patch, 3, 2), dtype=float)
    elem_nodes_local = np.zeros((K_patch, Nq, 2), dtype=float)

    for k in range(K_patch):
        verts = points_local[etoV_local[k], :]
        elem_vertices_local[k, :, :] = verts
        elem_nodes_local[k, :, :] = map_ref_to_phys_points(rs_nodes, verts)

    return elem_vertices_local, elem_nodes_local


def build_sphere_patch_mesh(
    *,
    nsub: int,
    radius: float = 1.0,
    quad_table: str = "table2",
    quad_order: int = 4,
) -> SpherePatchMesh:
    """
    Build a precomputed 8-patch sphere mesh.

    Parameters
    ----------
    nsub:
        Number of subdivisions per patch edge.
        Each patch contains nsub^2 sub-elements.
    radius:
        Sphere radius.
    quad_table, quad_order:
        Existing repo quadrature rule.

    Returns
    -------
    SpherePatchMesh
        Geometry-only mesh object.
    """
    if nsub < 1:
        raise ValueError("nsub must be >= 1.")
    if radius <= 0:
        raise ValueError("radius must be positive.")

    # 1. Local submesh template.
    points_local, etoV_local = uniform_local_submesh(nsub)

    # 2. Existing repo quadrature rule.
    rule = load_rule(quad_table, quad_order)
    rs_nodes = rule["rs"]

    # 3. Local element nodes once.
    elem_vertices_one_patch, elem_nodes_one_patch = _build_local_element_nodes(
        points_local,
        etoV_local,
        rs_nodes,
    )

    K_patch = etoV_local.shape[0]
    Nq = rs_nodes.shape[0]
    patch_ids = list(all_patch_ids())
    K_total = len(patch_ids) * K_patch

    elem_to_patch = np.zeros(K_total, dtype=int)
    elem_to_local_id = np.zeros(K_total, dtype=int)

    elem_vertices_local = np.zeros((K_total, 3, 2), dtype=float)
    volume_nodes_local = np.zeros((K_total, Nq, 2), dtype=float)

    volume_lambda = np.zeros((K_total, Nq), dtype=float)
    volume_theta = np.zeros((K_total, Nq), dtype=float)
    volume_xyz = np.zeros((K_total, Nq, 3), dtype=float)
    volume_sqrtG = np.zeros((K_total, Nq), dtype=float)

    # 4. Fill patch blocks.
    row = 0
    for patch_id in patch_ids:
        sl = slice(row, row + K_patch)

        elem_to_patch[sl] = patch_id
        elem_to_local_id[sl] = np.arange(K_patch, dtype=int)

        elem_vertices_local[sl, :, :] = elem_vertices_one_patch
        volume_nodes_local[sl, :, :] = elem_nodes_one_patch

        x = elem_nodes_one_patch[:, :, 0]
        y = elem_nodes_one_patch[:, :, 1]

        lam, th = local_xy_to_patch_angles(
            x,
            y,
            patch_id,
        )

        X, Y, Z = local_xy_to_sphere_xyz(
            x,
            y,
            patch_id,
            radius=radius,
        )

        J = sqrtG(
            x,
            y,
            patch_id,
            radius=radius,
        )

        volume_lambda[sl, :] = lam
        volume_theta[sl, :] = th
        volume_xyz[sl, :, 0] = X
        volume_xyz[sl, :, 1] = Y
        volume_xyz[sl, :, 2] = Z
        volume_sqrtG[sl, :] = J

        row += K_patch

    return SpherePatchMesh(
        radius=radius,
        nsub=nsub,
        quad_table=quad_table,
        quad_order=quad_order,
        points_local_template=points_local,
        etoV_local_template=etoV_local,
        elem_to_patch=elem_to_patch,
        elem_to_local_id=elem_to_local_id,
        elem_vertices_local=elem_vertices_local,
        volume_nodes_local=volume_nodes_local,
        volume_lambda=volume_lambda,
        volume_theta=volume_theta,
        volume_xyz=volume_xyz,
        volume_sqrtG=volume_sqrtG,
        patch_trace_pairs=tuple(expected_sphere_trace_pairs_bidirectional()),
    )


def mesh_size_summary(mesh: SpherePatchMesh) -> dict[str, int]:
    """
    Return basic mesh size information.
    """
    K_total = mesh.elem_to_patch.shape[0]
    Nq = mesh.volume_nodes_local.shape[1]

    return {
        "num_patches": 8,
        "num_elements_total": K_total,
        "num_elements_per_patch": mesh.nsub**2,
        "num_volume_nodes_per_element": Nq,
        "num_volume_nodes_total": K_total * Nq,
        "num_directed_patch_trace_pairs": len(mesh.patch_trace_pairs),
    }


def flatten_volume_xyz(mesh: SpherePatchMesh) -> np.ndarray:
    """
    Flatten volume xyz nodes to shape (K_total*Nq, 3).
    """
    K, Nq, _ = mesh.volume_xyz.shape
    return mesh.volume_xyz.reshape(K * Nq, 3)


def flatten_volume_local_xy(mesh: SpherePatchMesh) -> np.ndarray:
    """
    Flatten volume local nodes to shape (K_total*Nq, 2).
    """
    K, Nq, _ = mesh.volume_nodes_local.shape
    return mesh.volume_nodes_local.reshape(K * Nq, 2)