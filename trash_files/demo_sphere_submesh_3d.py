from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.rule_registry import load_rule
from geometry.affine_map import map_ref_to_phys_points
from geometry.sphere_patches import all_patch_ids
from geometry.sphere_patch_mesh import uniform_local_submesh
from geometry.sphere_mapping import local_xy_to_sphere_xyz


def add_xyz_axes(ax, *, radius: float = 1.0, scale: float = 1.35) -> None:
    """
    Add visible X/Y/Z coordinate axes.

    中文：
    在 3D 球面圖中加入 +X, +Y, +Z 三個方向。
    """
    L = scale * radius

    ax.quiver(
        0.0, 0.0, 0.0,
        L, 0.0, 0.0,
        arrow_length_ratio=0.08,
        linewidth=1.8,
    )
    ax.text(L, 0.0, 0.0, "+X", fontsize=12)

    ax.quiver(
        0.0, 0.0, 0.0,
        0.0, L, 0.0,
        arrow_length_ratio=0.08,
        linewidth=1.8,
    )
    ax.text(0.0, L, 0.0, "+Y", fontsize=12)

    ax.quiver(
        0.0, 0.0, 0.0,
        0.0, 0.0, L,
        arrow_length_ratio=0.08,
        linewidth=1.8,
    )
    ax.text(0.0, 0.0, L, "+Z", fontsize=12)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)


def add_reference_sphere(ax, *, radius: float = 1.0) -> None:
    """
    Draw a light reference sphere wireframe.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, 72)
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 36)

    PHI, THETA = np.meshgrid(phi, theta)

    X = radius * np.cos(PHI) * np.cos(THETA)
    Y = radius * np.sin(PHI) * np.cos(THETA)
    Z = radius * np.sin(THETA)

    ax.plot_wireframe(X, Y, Z, linewidth=0.25, alpha=0.18)


def map_local_points_to_sphere(
    local_xy: np.ndarray,
    patch_id: int,
    *,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Map local triangle coordinates to 3D sphere coordinates.

    Parameters
    ----------
    local_xy:
        Shape (N, 2), local coordinates satisfying:
            x >= 0, y >= 0, x + y <= 1
    patch_id:
        1,...,8

    Returns
    -------
    xyz:
        Shape (N, 3)
    """
    local_xy = np.asarray(local_xy, dtype=float)

    x = local_xy[:, 0]
    y = local_xy[:, 1]

    X, Y, Z = local_xy_to_sphere_xyz(
        x,
        y,
        patch_id,
        radius=radius,
    )

    return np.column_stack([X, Y, Z])


def edge_samples_2d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    nsample: int = 12,
) -> np.ndarray:
    """
    Sample points on a local 2D edge.

    注意：
    sphere mapping 是 nonlinear。
    所以 3D element edge 不應只用兩端點連直線。
    這裡先在 local edge 上取樣，再映到 sphere。
    """
    t = np.linspace(0.0, 1.0, nsample)
    return (1.0 - t)[:, None] * a[None, :] + t[:, None] * b[None, :]


def draw_curved_triangle_edges(
    ax,
    local_vertices: np.ndarray,
    patch_id: int,
    *,
    radius: float = 1.0,
    nsample_per_edge: int = 12,
    linewidth: float = 0.55,
    alpha: float = 0.8,
) -> None:
    """
    Draw one sub-element boundary on the sphere.

    local_vertices:
        Shape (3, 2), vertices in local triangle coordinates.
    """
    edges = [
        (local_vertices[0], local_vertices[1]),
        (local_vertices[1], local_vertices[2]),
        (local_vertices[2], local_vertices[0]),
    ]

    for a, b in edges:
        local_edge = edge_samples_2d(
            a,
            b,
            nsample=nsample_per_edge,
        )
        xyz = map_local_points_to_sphere(
            local_edge,
            patch_id,
            radius=radius,
        )

        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            linewidth=linewidth,
            alpha=alpha,
        )


def element_nodes_on_local_submesh(
    points_local: np.ndarray,
    etoV: np.ndarray,
    *,
    quad_table: str = "table2",
    quad_order: int = 4,
) -> np.ndarray:
    """
    Put quadrature / DG nodes inside every local sub-element.

    Parameters
    ----------
    points_local:
        Submesh vertices on local triangle, shape (Nv, 2).
    etoV:
        Sub-element connectivity, shape (K, 3).
    quad_table, quad_order:
        Existing repo quadrature rule.

    Returns
    -------
    elem_nodes_local:
        Shape (K, Nq, 2)
    """
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

    return elem_nodes_local


def draw_patch_submesh_on_sphere(
    ax,
    patch_id: int,
    *,
    radius: float = 1.0,
    nsub: int = 5,
    quad_table: str = "table2",
    quad_order: int = 4,
    show_element_edges: bool = True,
    show_element_nodes: bool = True,
    show_patch_label: bool = True,
    node_size: float = 6.0,
) -> None:
    """
    Draw one patch T_i mapped to sphere with submesh and internal nodes.

    nsub:
        Number of subdivisions per edge.
        Each patch has nsub^2 small triangular elements.

    quad_order:
        Existing repo quadrature rule order.
        If quad_order=1, you may only see few nodes.
        For richer distribution, use quad_order=3 or 4.
    """
    points_local, etoV = uniform_local_submesh(nsub)

    # 1. Draw sub-element curved edges on sphere.
    if show_element_edges:
        for tri in etoV:
            local_vertices = points_local[tri, :]
            draw_curved_triangle_edges(
                ax,
                local_vertices,
                patch_id,
                radius=radius,
                nsample_per_edge=10,
                linewidth=0.45,
                alpha=0.75,
            )

    # 2. Draw quadrature / DG nodes inside every sub-element.
    if show_element_nodes:
        elem_nodes_local = element_nodes_on_local_submesh(
            points_local,
            etoV,
            quad_table=quad_table,
            quad_order=quad_order,
        )

        K, Nq, _ = elem_nodes_local.shape
        flat_local = elem_nodes_local.reshape(K * Nq, 2)

        xyz = map_local_points_to_sphere(
            flat_local,
            patch_id,
            radius=radius,
        )

        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=node_size,
            alpha=0.85,
            marker=".",
        )

    # 3. Patch label near local centroid.
    if show_patch_label:
        center_local = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        center_xyz = map_local_points_to_sphere(
            center_local,
            patch_id,
            radius=radius,
        )[0]

        ax.text(
            center_xyz[0],
            center_xyz[1],
            center_xyz[2],
            f"T{patch_id}",
            fontsize=11,
            ha="center",
            va="center",
        )


def plot_sphere_submesh_3d(
    outdir: Path,
    *,
    radius: float = 1.0,
    nsub: int = 5,
    quad_table: str = "table2",
    quad_order: int = 4,
    show_reference_sphere: bool = True,
    show_element_edges: bool = True,
    show_element_nodes: bool = True,
    show_patch_labels: bool = True,
) -> None:
    """
    Plot all 8 sphere patches with submesh and element nodes.
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    if show_reference_sphere:
        add_reference_sphere(ax, radius=radius)

    for patch_id in [1]:
        draw_patch_submesh_on_sphere(
            ax,
            patch_id,
            radius=radius,
            nsub=nsub,
            quad_table=quad_table,
            quad_order=quad_order,
            show_element_edges=show_element_edges,
            show_element_nodes=show_element_nodes,
            show_patch_label=show_patch_labels,
            node_size=6.0,
        )

    add_xyz_axes(ax, radius=radius)
    ax.view_init(elev=25, azim=35)

    ax.set_title(
        f"3D sphere submesh with element nodes "
        f"(nsub={nsub}, {quad_table}, order={quad_order})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "sphere_submesh_3d_nodes.png", dpi=240)
    plt.close(fig)


def main() -> None:
    outdir = Path("experiments_outputs/sphere_mapping_3d")

    plot_sphere_submesh_3d(
        outdir,
        radius=1.0,
        nsub=2,
        quad_table="table1",
        quad_order=4,
        show_reference_sphere=True,
        show_element_edges=True,
        show_element_nodes=True,
        show_patch_labels=True,
    )

    print(f"[done] figures written to {outdir}")


if __name__ == "__main__":
    main()