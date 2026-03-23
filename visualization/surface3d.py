from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _axis_limits_from_vertices(vertices: np.ndarray, pad_ratio: float = 0.06):
    vertices = np.asarray(vertices, dtype=float)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = np.maximum(maxs - mins, 1e-14)
    pad = pad_ratio * span
    return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])


def plot_triangle_surface3d(
    rs_eval: np.ndarray,
    z_eval: np.ndarray,
    vertices: np.ndarray,
    nodes: np.ndarray | None = None,
    title: str | None = None,
    zlabel: str = "value",
    elev: float = 28.0,
    azim: float = -58.0,
    ax=None,
):
    """
    Plot a 3D surface over a triangular domain in (r, s).
    """
    rs_eval = np.asarray(rs_eval, dtype=float)
    z_eval = np.asarray(z_eval, dtype=float).reshape(-1)

    if rs_eval.ndim != 2 or rs_eval.shape[1] != 2:
        raise ValueError("rs_eval must have shape (n_points, 2).")
    if rs_eval.shape[0] != z_eval.size:
        raise ValueError("rs_eval and z_eval size mismatch.")

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    triang = mtri.Triangulation(rs_eval[:, 0], rs_eval[:, 1])

    surf = ax.plot_trisurf(
        triang,
        z_eval,
        linewidth=0.2,
        antialiased=True,
        alpha=0.95,
    )
    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08)

    z0 = float(np.min(z_eval))
    tri = np.vstack([vertices, vertices[0]])
    ax.plot(tri[:, 0], tri[:, 1], z0, linewidth=1.5)

    if nodes is not None:
        nodes = np.asarray(nodes, dtype=float)
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            np.full(nodes.shape[0], z0),
            s=16,
            marker="o",
            alpha=0.9,
        )

    xlim, ylim = _axis_limits_from_vertices(vertices)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel("r")
    ax.set_ylabel("s")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)

    if title is not None:
        ax.set_title(title)

    return fig, ax