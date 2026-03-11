from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _triangle_closed(vertices: np.ndarray) -> np.ndarray:
    return np.vstack([vertices, vertices[0]])


def plot_triangle_field(
    xy_eval: np.ndarray,
    u_eval: np.ndarray,
    vertices: np.ndarray,
    nodes: np.ndarray | None = None,
    title: str | None = None,
    levels: int = 20,
    show_nodes: bool = True,
    ax=None,
):
    """
    Plot a scalar field inside a triangle using tricontourf.

    Parameters
    ----------
    xy_eval : np.ndarray
        Evaluation points of shape (n_points, 2).
    u_eval : np.ndarray
        Scalar values at evaluation points of shape (n_points,).
    vertices : np.ndarray
        Triangle vertices of shape (3, 2).
    nodes : np.ndarray | None
        Optional nodal points to overlay, shape (n_nodes, 2).
    title : str | None
        Plot title.
    levels : int
        Number of contour levels.
    show_nodes : bool
        Whether to overlay nodal points.
    ax : matplotlib axis | None
        Existing axis if provided.

    Returns
    -------
    tuple[fig, ax]
    """
    xy_eval = np.asarray(xy_eval, dtype=float)
    u_eval = np.asarray(u_eval, dtype=float).reshape(-1)

    if xy_eval.ndim != 2 or xy_eval.shape[1] != 2:
        raise ValueError("xy_eval must have shape (n_points, 2).")
    if xy_eval.shape[0] != u_eval.size:
        raise ValueError("xy_eval and u_eval size mismatch.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig = ax.figure

    triang = mtri.Triangulation(xy_eval[:, 0], xy_eval[:, 1])
    contour = ax.tricontourf(triang, u_eval, levels=levels)
    fig.colorbar(contour, ax=ax)

    tri = _triangle_closed(vertices)
    ax.plot(tri[:, 0], tri[:, 1], linewidth=1.5)

    if show_nodes and nodes is not None:
        nodes = np.asarray(nodes, dtype=float)
        ax.scatter(nodes[:, 0], nodes[:, 1], s=15, marker="o", alpha=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel("xi")
    ax.set_ylabel("eta")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    if title is not None:
        ax.set_title(title)

    return fig, ax
