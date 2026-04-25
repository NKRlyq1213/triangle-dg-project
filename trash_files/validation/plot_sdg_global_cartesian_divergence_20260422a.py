from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from operators.sdg_cartesian_divergence_20260422a import (
    build_sdg_cartesian_divergence_setup_20260422a,
    sdg_cartesian_divergence_of_constant_state_20260422a,
)


def build_plot_triangulation(X: np.ndarray, Y: np.ndarray, rule: dict):
    """
    把每個 element 的 nodal points 組成可繪圖的全域 triangulation。

    Parameters
    ----------
    X, Y : ndarray, shape (K, Np)
        每個 element 上的 global flattened coordinates.
    rule : dict
        table1 rule，需包含 "rs".

    Returns
    -------
    x_all, y_all : ndarray, shape (K*Np,)
    triangles_all : ndarray, shape (Nt, 3)
    """
    rs = np.asarray(rule["rs"], dtype=float)
    K, Np = X.shape

    # 在 reference triangle 上先做一次三角剖分
    tri_ref = mtri.Triangulation(rs[:, 0], rs[:, 1])
    triangles_ref = tri_ref.triangles  # shape (Nt_ref, 3)

    x_all = X.reshape(-1)
    y_all = Y.reshape(-1)

    triangles_list = []
    for k in range(K):
        offset = k * Np
        triangles_list.append(triangles_ref + offset)

    triangles_all = np.vstack(triangles_list)
    return x_all, y_all, triangles_all


def plot_divergence_on_square(
    *,
    nsub: int = 8,
    order: int = 4,
    N: int = 4,
    radius: float = 1.0,
    alpha0: float = np.pi / 4.0,
    cmap_main: str = "coolwarm",
    cmap_log: str = "viridis",
    log_eps: float = 1e-14,
    show_mesh_lines: bool = True,
):
    """
    在 flattened square [-1,1]^2 上畫 divergence 圖。
    """
    setup = build_sdg_cartesian_divergence_setup_20260422a(
        nsub=nsub,
        order=order,
        N=N,
        radius=radius,
    )

    div, diag = sdg_cartesian_divergence_of_constant_state_20260422a(
        setup,
        u0=1.0,
        alpha0=alpha0,
    )

    X = setup.X
    Y = setup.Y

    x_all, y_all, triangles_all = build_plot_triangulation(X, Y, setup.rule)
    div_all = div.reshape(-1)
    log_div_all = np.log10(np.abs(div_all) + log_eps)

    triang = mtri.Triangulation(x_all, y_all, triangles_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -------------------------------------------------
    # Plot 1: div
    # -------------------------------------------------
    tpc1 = axes[0].tripcolor(
        triang,
        div_all,
        shading="gouraud",
        cmap=cmap_main,
    )
    cbar1 = fig.colorbar(tpc1, ax=axes[0])
    cbar1.set_label(r"$\partial_x(u^1 q)+\partial_y(u^2 q)$")

    if show_mesh_lines:
        axes[0].triplot(triang, linewidth=0.2)

    axes[0].set_title(
        rf"Divergence on $[-1,1]^2$ (q=1, $\alpha_0={alpha0:.4f}$, nsub={nsub})"
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_aspect("equal")

    # 標出可能的 pole copies
    pole_points = np.array([
        [0.0, 0.0],   # north pole
        [1.0, 1.0],   # south pole copies
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
    ])
    axes[0].scatter(pole_points[:, 0], pole_points[:, 1], marker="x", s=50)
    for px, py in pole_points:
        axes[0].text(px, py, f"({px:.0f},{py:.0f})", fontsize=8, ha="left", va="bottom")

    # -------------------------------------------------
    # Plot 2: log10(|div| + eps)
    # -------------------------------------------------
    tpc2 = axes[1].tripcolor(
        triang,
        log_div_all,
        shading="gouraud",
        cmap=cmap_log,
    )
    cbar2 = fig.colorbar(tpc2, ax=axes[1])
    cbar2.set_label(r"$\log_{10}(|\mathrm{div}|+\varepsilon)$")

    if show_mesh_lines:
        axes[1].triplot(triang, linewidth=0.2)

    axes[1].set_title(
        rf"$\log_{{10}}(|\mathrm{{div}}|+\varepsilon)$ on $[-1,1]^2$"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[1].set_aspect("equal")

    axes[1].scatter(pole_points[:, 0], pole_points[:, 1], marker="x", s=50)
    for px, py in pole_points:
        axes[1].text(px, py, f"({px:.0f},{py:.0f})", fontsize=8, ha="left", va="bottom")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsub", type=int, default=4)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=np.pi / 4.0)
    parser.add_argument("--no-mesh-lines", action="store_true")

    args = parser.parse_args()

    plot_divergence_on_square(
        nsub=args.nsub,
        order=args.order,
        N=args.N,
        radius=args.radius,
        alpha0=args.alpha0,
        show_mesh_lines=not args.no_mesh_lines,
    )


if __name__ == "__main__":
    main()