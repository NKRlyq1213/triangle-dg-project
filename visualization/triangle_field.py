from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _triangle_closed(vertices: np.ndarray) -> np.ndarray:
    return np.vstack([vertices, vertices[0]])


def _axis_limits_from_vertices(vertices: np.ndarray, pad_ratio: float = 0.06) -> tuple[tuple[float, float], tuple[float, float]]:
    vertices = np.asarray(vertices, dtype=float)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = np.maximum(maxs - mins, 1e-14)
    pad = pad_ratio * span
    return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])


def plot_triangle_field(
    rs_eval: np.ndarray,
    u_eval: np.ndarray,
    vertices: np.ndarray,
    nodes: np.ndarray | None = None,
    title: str | None = None,
    levels: int = 20,
    show_nodes: bool = True,
    ax=None,
):
    """
    Plot a scalar field inside a triangle using tricontourf in (r, s).
    """
    rs_eval = np.asarray(rs_eval, dtype=float)
    u_eval = np.asarray(u_eval, dtype=float).reshape(-1)

    if rs_eval.ndim != 2 or rs_eval.shape[1] != 2:
        raise ValueError("rs_eval must have shape (n_points, 2).")
    if rs_eval.shape[0] != u_eval.size:
        raise ValueError("rs_eval and u_eval size mismatch.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig = ax.figure

    triang = mtri.Triangulation(rs_eval[:, 0], rs_eval[:, 1])
    contour = ax.tricontourf(triang, u_eval, levels=levels)
    fig.colorbar(contour, ax=ax)

    tri = _triangle_closed(vertices)
    ax.plot(tri[:, 0], tri[:, 1], linewidth=1.5)

    if show_nodes and nodes is not None:
        nodes = np.asarray(nodes, dtype=float)
        ax.scatter(nodes[:, 0], nodes[:, 1], s=15, marker="o", alpha=0.8)

    xlim, ylim = _axis_limits_from_vertices(vertices)

    ax.set_aspect("equal")
    ax.set_xlabel("r")
    ax.set_ylabel("s")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2)

    if title is not None:
        ax.set_title(title)

    return fig, ax