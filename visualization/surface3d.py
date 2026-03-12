from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_triangle_surface3d(
    xy_eval: np.ndarray,
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
    Plot a 3D surface over a triangular domain.

    Parameters
    ----------
    xy_eval : np.ndarray
        Evaluation points of shape (n_points, 2).
    z_eval : np.ndarray
        Scalar values at evaluation points of shape (n_points,).
    vertices : np.ndarray
        Triangle vertices of shape (3, 2).
    nodes : np.ndarray | None
        Optional nodal points to overlay on the xy-plane.
    title : str | None
        Plot title.
    zlabel : str
        Z-axis label.
    elev, azim : float
        View angles for 3D plot.
    ax : matplotlib axis | None
        Existing 3D axis if provided.

    Returns
    -------
    tuple[fig, ax]
    """
    xy_eval = np.asarray(xy_eval, dtype=float)
    z_eval = np.asarray(z_eval, dtype=float).reshape(-1)

    if xy_eval.ndim != 2 or xy_eval.shape[1] != 2:
        raise ValueError("xy_eval must have shape (n_points, 2).")
    if xy_eval.shape[0] != z_eval.size:
        raise ValueError("xy_eval and z_eval size mismatch.")

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    triang = mtri.Triangulation(xy_eval[:, 0], xy_eval[:, 1])

    surf = ax.plot_trisurf(
        triang,
        z_eval,
        linewidth=0.2,
        antialiased=True,
        alpha=0.95,
    )
    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08)

    # triangle boundary on z = min(z)
    z0 = float(np.min(z_eval))
    tri = np.vstack([vertices, vertices[0]])
    ax.plot(tri[:, 0], tri[:, 1], z0, linewidth=1.5)

    # optional nodal points projected onto z = min(z)
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

    ax.set_xlabel("xi")
    ax.set_ylabel("eta")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)

    if title is not None:
        ax.set_title(title)

    return fig, ax