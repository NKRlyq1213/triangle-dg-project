from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.rule_registry import load_rule
from geometry.affine_map import map_ref_to_phys_points
from geometry.sphere_patch_mesh import (
    all_patch_vertices_2d,
    map_local_xy_to_patch_xy,
    patch_submesh_2d,
    uniform_local_submesh,
)


def add_square_guides(ax, *, radius: float = 1.0) -> None:
    """
    Draw square boundary and coordinate axes.
    """
    R = radius
    boundary = np.array([
        [-R, -R],
        [ R, -R],
        [ R,  R],
        [-R,  R],
        [-R, -R],
    ], dtype=float)

    ax.plot(boundary[:, 0], boundary[:, 1], linewidth=1.5)
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, linewidth=1.0, alpha=0.5)

    ax.set_xlim(-1.1 * R, 1.1 * R)
    ax.set_ylim(-1.1 * R, 1.1 * R)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def draw_triangle(ax, verts: np.ndarray, *, close: bool = True, label: str | None = None) -> None:
    """
    Draw one triangle boundary and optionally place a label at centroid.
    """
    poly = verts
    if close:
        poly = np.vstack([verts, verts[0]])

    ax.plot(poly[:, 0], poly[:, 1], linewidth=1.5)

    if label is not None:
        c = np.mean(verts, axis=0)
        ax.text(c[0], c[1], label, fontsize=12, ha="center", va="center")


def plot_patch_partition_2d(
    outdir: Path,
    *,
    radius: float = 1.0,
    show_quadrature_nodes: bool = True,
    quad_table: str = "table2",
    quad_order: int = 4,
) -> None:
    """
    Plot the eight patch triangles T1,...,T8 on the 2D square.
    Optionally overlay existing repo quadrature nodes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    add_square_guides(ax, radius=radius)

    patch_dict = all_patch_vertices_2d(radius=radius)

    for patch_id, verts in patch_dict.items():
        draw_triangle(ax, verts, label=f"T{patch_id}")

        if show_quadrature_nodes:
            # Reuse existing rule from repo
            rule = load_rule(quad_table, quad_order)
            rs = rule["rs"]

            # convert rs -> local xy
            x = 0.5 * (rs[:, 0] + 1.0)
            y = 0.5 * (rs[:, 1] + 1.0)
            local_xy = np.column_stack([x, y])

            xy_phys = map_local_xy_to_patch_xy(local_xy, patch_id, radius=radius)
            ax.scatter(xy_phys[:, 0], xy_phys[:, 1], s=20, alpha=0.8)

    ax.set_title("2D patch partition on the square: T1,...,T8")

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "patch_partition_2d.png", dpi=200)
    plt.close(fig)


def draw_submesh(ax, points: np.ndarray, etoV: np.ndarray, *, linewidth: float = 0.8) -> None:
    """
    Draw triangular submesh edges.
    """
    for tri in etoV:
        poly = np.vstack([points[tri], points[tri[0]]])
        ax.plot(poly[:, 0], poly[:, 1], linewidth=linewidth)

def element_nodes_on_submesh(
    points: np.ndarray,
    etoV: np.ndarray,
    *,
    quad_table: str = "table2",
    quad_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map reference triangle quadrature nodes to every small sub-element.

    Parameters
    ----------
    points:
        Physical 2D mesh vertices, shape (Nv, 2).
    etoV:
        Element-to-vertex connectivity, shape (K, 3).
    quad_table, quad_order:
        Existing repo quadrature rule.

    Returns
    -------
    elem_nodes:
        Shape (K, Nq, 2).
        elem_nodes[k, q, :] is the q-th node inside element k.
    elem_centers:
        Shape (K, 2).
        Element centroids, useful for labeling.
    """
    points = np.asarray(points, dtype=float)
    etoV = np.asarray(etoV, dtype=int)

    rule = load_rule(quad_table, quad_order)
    rs_nodes = rule["rs"]  # repo reference triangle nodes, shape (Nq, 2)

    K = etoV.shape[0]
    Nq = rs_nodes.shape[0]

    elem_nodes = np.zeros((K, Nq, 2), dtype=float)
    elem_centers = np.zeros((K, 2), dtype=float)

    for k in range(K):
        verts = points[etoV[k], :]  # shape (3, 2)
        elem_nodes[k, :, :] = map_ref_to_phys_points(rs_nodes, verts)
        elem_centers[k, :] = np.mean(verts, axis=0)

    return elem_nodes, elem_centers


def draw_element_nodes(
    ax,
    points: np.ndarray,
    etoV: np.ndarray,
    *,
    quad_table: str = "table2",
    quad_order: int = 4,
    node_size: float = 8.0,
    show_element_centers: bool = False,
    show_element_ids: bool = False,
) -> None:
    """
    Draw DG/quadrature nodes inside every sub-element.

    中文：
    - 每個小三角形都會放一組 reference triangle quadrature nodes。
    - 這不是只畫一個點，而是每個 element 都有 Nq 個節點。
    """
    elem_nodes, elem_centers = element_nodes_on_submesh(
        points,
        etoV,
        quad_table=quad_table,
        quad_order=quad_order,
    )

    K, Nq, _ = elem_nodes.shape
    flat_nodes = elem_nodes.reshape(K * Nq, 2)

    ax.scatter(
        flat_nodes[:, 0],
        flat_nodes[:, 1],
        s=node_size,
        alpha=0.85,
        marker=".",
        label=f"{quad_table}, order={quad_order} nodes per sub-element",
    )

    if show_element_centers:
        ax.scatter(
            elem_centers[:, 0],
            elem_centers[:, 1],
            s=10,
            alpha=0.7,
            marker="x",
            label="element centers",
        )

    if show_element_ids:
        for k, c in enumerate(elem_centers):
            ax.text(c[0], c[1], str(k), fontsize=6, ha="center", va="center")

def plot_patch_partition_2d_submesh(
    outdir: Path,
    *,
    radius: float = 1.0,
    nsub: int = 4,
    show_patch_labels: bool = True,
    show_submesh_vertices: bool = False,
    show_element_nodes: bool = True,
    show_element_centers: bool = False,
    show_element_ids: bool = False,
    quad_table: str = "table2",
    quad_order: int = 4,
) -> None:
    """
    Plot the 2D square partition with uniform triangular submesh on each T_i.

    New feature:
        show_element_nodes=True
    will draw quadrature / DG nodes inside every sub-element.

    Parameters
    ----------
    nsub:
        Number of edge subdivisions per patch.
        Each patch has nsub^2 small triangles.
    quad_table, quad_order:
        Existing repo quadrature rule used as local element nodes.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    add_square_guides(ax, radius=radius)

    patch_dict = all_patch_vertices_2d(radius=radius)

    for patch_id, verts in patch_dict.items():
        points_phys, etoV = patch_submesh_2d(
            patch_id,
            nsub=nsub,
            radius=radius,
        )

        # 1. Draw small triangle element edges.
        draw_submesh(ax, points_phys, etoV, linewidth=0.7)

        # 2. Draw patch boundary and label.
        draw_triangle(
            ax,
            verts,
            label=(f"T{patch_id}" if show_patch_labels else None),
        )

        # 3. Draw submesh vertices if needed.
        if show_submesh_vertices:
            ax.scatter(
                points_phys[:, 0],
                points_phys[:, 1],
                s=12,
                alpha=0.7,
                marker="o",
                label="submesh vertices" if patch_id == 1 else None,
            )

        # 4. Draw quadrature / DG nodes inside every sub-element.
        if show_element_nodes:
            draw_element_nodes(
                ax,
                points_phys,
                etoV,
                quad_table=quad_table,
                quad_order=quad_order,
                node_size=7.0,
                show_element_centers=show_element_centers,
                show_element_ids=show_element_ids,
            )

    ax.set_title(
        f"2D patch partition with submesh nodes "
        f"(nsub={nsub}, {quad_table}, order={quad_order})"
    )

    # Avoid repeated legend entries.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="upper right")

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "patch_partition_2d_submesh_nodes.png", dpi=220)
    plt.close(fig)


def plot_local_reference_submesh(
    outdir: Path,
    *,
    nsub: int = 4,
) -> None:
    """
    Optional figure:
    show the subdivision on the local reference triangle itself.
    This is useful to verify the element pattern before mapping to T_i.
    """
    points_local, etoV = uniform_local_submesh(nsub)

    fig, ax = plt.subplots(figsize=(6, 6))
    for tri in etoV:
        poly = np.vstack([points_local[tri], points_local[tri[0]]])
        ax.plot(poly[:, 0], poly[:, 1], linewidth=0.8)

    ax.scatter(points_local[:, 0], points_local[:, 1], s=15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    ax.set_title(f"Local triangle uniform submesh (nsub={nsub})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "local_triangle_submesh.png", dpi=200)
    plt.close(fig)


def main() -> None:
    outdir = Path("experiments_outputs/sphere_mapping_2d")

    plot_patch_partition_2d(
        outdir,
        radius=1.0,
        show_quadrature_nodes=True,
        quad_table="table2",
        quad_order=4,
    )

    plot_patch_partition_2d_submesh(
        outdir,
        radius=1.0,
        nsub=5,
        show_patch_labels=True,
        show_submesh_vertices=False,
        show_element_nodes=True,
        show_element_centers=False,
        show_element_ids=False,
        quad_table="table2",
        quad_order=4,
    )

    plot_local_reference_submesh(
        outdir,
        nsub=5,
    )

    print(f"[done] figures written to {outdir}")


if __name__ == "__main__":
    main()