from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _triangle_closed(vertices: np.ndarray) -> np.ndarray:
    return np.vstack([vertices, vertices[0]])


def plot_nodes(
    rule: dict,
    vertices: np.ndarray,
    annotate: bool = False,
    show_boundary_only: bool = False,
    title: str | None = None,
    ax=None,
):
    """
    Plot quadrature/nodal points in the reference triangle.

    Parameters
    ----------
    rule : dict
        Output of load_table1_rule / load_table2_rule.
    vertices : np.ndarray
        Triangle vertices, shape (3,2).
    annotate : bool
        Whether to annotate point indices.
    show_boundary_only : bool
        If True and edge_mask exists, only show boundary-marked nodes.
    title : str | None
        Plot title.
    ax : matplotlib axis | None
        Existing axis if provided.

    Returns
    -------
    tuple[fig, ax]
    """
    xy = np.asarray(rule["xy"], dtype=float)

    mask = np.ones(len(xy), dtype=bool)
    if show_boundary_only and "edge_mask" in rule:
        mask = np.asarray(rule["edge_mask"], dtype=bool)

    xy_plot = xy[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    tri = _triangle_closed(vertices)
    ax.plot(tri[:, 0], tri[:, 1], linewidth=1.5)

    if "edge_mask" in rule and not show_boundary_only:
        edge_mask = np.asarray(rule["edge_mask"], dtype=bool)
        int_mask = ~edge_mask

        if np.any(int_mask):
            ax.scatter(
                xy[int_mask, 0],
                xy[int_mask, 1],
                s=50,
                marker="o",
                label="interior",
            )
        if np.any(edge_mask):
            ax.scatter(
                xy[edge_mask, 0],
                xy[edge_mask, 1],
                s=55,
                marker="s",
                label="edge-marked",
            )
        ax.legend()
    else:
        ax.scatter(xy_plot[:, 0], xy_plot[:, 1], s=50, marker="o")

    if annotate:
        idxs = np.where(mask)[0]
        for local_k, global_k in enumerate(idxs):
            xk, yk = xy[global_k]
            ax.text(xk, yk, str(global_k), fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("xi")
    ax.set_ylabel("eta")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)

    if title is None:
        title = f"{rule['table']} order {rule['order']}"
    ax.set_title(title)

    return fig, ax
