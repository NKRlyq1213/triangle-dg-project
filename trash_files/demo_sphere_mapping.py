from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geometry.sphere_patches import all_patch_ids
from geometry.sphere_mapping import (
    local_xy_to_square_xy,
    local_xy_to_sphere_xyz,
)


def add_xyz_axes(ax, *, radius: float = 1.0, scale: float = 1.25) -> None:
    """
    Add visible X/Y/Z coordinate axes to a 3D matplotlib plot.
    """
    L = scale * radius

    ax.quiver(0.0, 0.0, 0.0, L, 0.0, 0.0, arrow_length_ratio=0.08, linewidth=1.8)
    ax.text(L, 0.0, 0.0, "+X", fontsize=12)

    ax.quiver(0.0, 0.0, 0.0, 0.0, L, 0.0, arrow_length_ratio=0.08, linewidth=1.8)
    ax.text(0.0, L, 0.0, "+Y", fontsize=12)

    ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, L, arrow_length_ratio=0.08, linewidth=1.8)
    ax.text(0.0, 0.0, L, "+Z", fontsize=12)

    ax.text(-L, 0.0, 0.0, "-X", fontsize=10)
    ax.text(0.0, -L, 0.0, "-Y", fontsize=10)
    ax.text(0.0, 0.0, -L, "-Z", fontsize=10)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)


def build_local_subtriangulation(nsub: int):
    """
    Build a uniform subdivision of the normalized local triangle:
        x >= 0, y >= 0, x + y <= 1

    Returns
    -------
    x, y : ndarray, shape (Np,)
        Local node coordinates.
    tris : ndarray, shape (Nt, 3)
        Connectivity of subdivided triangles.
    """
    if nsub < 1:
        raise ValueError("nsub must be >= 1.")

    points = []
    index = {}

    # lattice points
    for i in range(nsub + 1):
        for j in range(nsub + 1 - i):
            idx = len(points)
            index[(i, j)] = idx
            points.append((i / nsub, j / nsub))

    points = np.array(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]

    tris = []

    # subdivided small triangles
    for i in range(nsub):
        for j in range(nsub - i):
            a = index[(i, j)]
            b = index[(i + 1, j)]
            c = index[(i, j + 1)]
            tris.append((a, b, c))

            if i + j <= nsub - 2:
                d = index[(i + 1, j + 1)]
                tris.append((b, d, c))

    tris = np.array(tris, dtype=int)
    return x, y, tris


def triangle_edges(tris: np.ndarray):
    """
    Convert triangle connectivity to unique edges for wireframe plotting.
    """
    edges = set()
    for tri in tris:
        i, j, k = tri
        for a, b in [(i, j), (j, k), (k, i)]:
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return sorted(edges)


def patch_polygon_vertices(patch_id: int, radius: float = 1.0) -> np.ndarray:
    """
    Return the 2D polygon vertices of Ti in the square.
    """
    R = radius

    if patch_id == 1:
        return np.array([[0, 0], [R, 0], [0, R]], dtype=float)
    if patch_id == 2:
        return np.array([[R, R], [R, 0], [0, R]], dtype=float)
    if patch_id == 3:
        return np.array([[0, 0], [0, R], [-R, 0]], dtype=float)
    if patch_id == 4:
        return np.array([[-R, R], [0, R], [-R, 0]], dtype=float)
    if patch_id == 5:
        return np.array([[0, 0], [-R, 0], [0, -R]], dtype=float)
    if patch_id == 6:
        return np.array([[-R, -R], [-R, 0], [0, -R]], dtype=float)
    if patch_id == 7:
        return np.array([[0, 0], [0, -R], [R, 0]], dtype=float)
    if patch_id == 8:
        return np.array([[R, -R], [R, 0], [0, -R]], dtype=float)

    raise ValueError("patch_id must be one of 1,...,8.")


def plot_square_patch_layout(outdir: Path, *, radius: float = 1.0) -> None:
    """
    Plot only the 2D Ti layout on the square.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # square boundary
    sq = np.array(
        [
            [-radius, -radius],
            [ radius, -radius],
            [ radius,  radius],
            [-radius,  radius],
            [-radius, -radius],
        ],
        dtype=float,
    )
    ax.plot(sq[:, 0], sq[:, 1], linewidth=2.0)

    # draw each Ti
    for patch_id in all_patch_ids():
        poly = patch_polygon_vertices(patch_id, radius=radius)
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=1.5)

        cx = np.mean(poly[:, 0])
        cy = np.mean(poly[:, 1])
        ax.text(cx, cy, f"T{patch_id}", fontsize=12, ha="center", va="center")

    # center cross
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)

    ax.set_title("2D square partition into T1,...,T8")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1 * radius, 1.1 * radius)
    ax.set_ylim(-1.1 * radius, 1.1 * radius)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "square_patch_layout.png", dpi=200)
    plt.close(fig)


def plot_square_patch_mesh(
    outdir: Path,
    *,
    radius: float = 1.0,
    nsub: int = 4,
) -> None:
    """
    Plot 2D square Ti layout together with uniform triangular subdivision.
    """
    x, y, tris = build_local_subtriangulation(nsub=nsub)

    fig, ax = plt.subplots(figsize=(8, 8))

    # square boundary
    sq = np.array(
        [
            [-radius, -radius],
            [ radius, -radius],
            [ radius,  radius],
            [-radius,  radius],
            [-radius, -radius],
        ],
        dtype=float,
    )
    ax.plot(sq[:, 0], sq[:, 1], linewidth=2.0)

    for patch_id in all_patch_ids():
        X2, Y2 = local_xy_to_square_xy(x, y, patch_id, radius=radius)

        # draw subdivided mesh
        for tri in tris:
            loop = np.r_[tri, tri[0]]
            ax.plot(X2[loop], Y2[loop], linewidth=0.8)

        # label patch by centroid of local triangle
        cx, cy = local_xy_to_square_xy(
            np.array([1.0 / 3.0]),
            np.array([1.0 / 3.0]),
            patch_id,
            radius=radius,
        )
        ax.text(cx[0], cy[0], f"T{patch_id}", fontsize=12, ha="center", va="center")

    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)

    ax.set_title(f"2D Ti layout with uniform subdivision (nsub={nsub})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1 * radius, 1.1 * radius)
    ax.set_ylim(-1.1 * radius, 1.1 * radius)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "square_patch_mesh.png", dpi=200)
    plt.close(fig)


def plot_sphere_patch_mesh_3d(
    outdir: Path,
    *,
    radius: float = 1.0,
    nsub: int = 4,
) -> None:
    """
    Plot the same subdivided mesh after mapping to the sphere.
    """
    x, y, tris = build_local_subtriangulation(nsub=nsub)
    edges = triangle_edges(tris)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for patch_id in [1]:  # only plot upper hemisphere patches for better visibility
        X, Y, Z = local_xy_to_sphere_xyz(x, y, patch_id, radius=radius)

        # draw wireframe edges
        for i, j in edges:
            ax.plot(
                [X[i], X[j]],
                [Y[i], Y[j]],
                [Z[i], Z[j]],
                linewidth=0.8,
            )

        # label patch
        cx, cy, cz = local_xy_to_sphere_xyz(
            np.array([1.0 / 3.0]),
            np.array([1.0 / 3.0]),
            patch_id,
            radius=radius,
        )
        ax.text(cx[0], cy[0], cz[0], f"T{patch_id}", fontsize=11)

    add_xyz_axes(ax, radius=radius)
    ax.view_init(elev=25, azim=35)

    ax.set_title(f"Sphere patch mesh after subdivision (nsub={nsub})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "sphere_patch_mesh_3d.png", dpi=200)
    plt.close(fig)


def main() -> None:
    outdir = Path("experiments_outputs/sphere_mapping")

    # 1. just the 2D patch distribution Ti
    plot_square_patch_layout(outdir, radius=1.0)

    # 2. 2D subdivision inside each Ti
    plot_square_patch_mesh(outdir, radius=1.0, nsub=5)

    # 3. mapped 3D mesh on the sphere
    plot_sphere_patch_mesh_3d(outdir, radius=1.0, nsub=5)

    print(f"[done] figures written to {outdir}")


if __name__ == "__main__":
    main()