from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _axis_limits_from_vertices(vertices: np.ndarray, pad_ratio: float = 0.06) -> tuple[tuple[float, float], tuple[float, float]]:
    vertices = np.asarray(vertices, dtype=float)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = np.maximum(maxs - mins, 1e-14)
    pad = pad_ratio * span
    return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])


def plot_radial_field(
    rs_eval: np.ndarray,
    u_eval: np.ndarray,
    vertices: np.ndarray,
    nodes: np.ndarray | None = None,
    title: str | None = None,
    ax=None,
):
    """
    Scatter-style visualization for centroid-based star sampling in (r, s).
    """
    rs_eval = np.asarray(rs_eval, dtype=float)
    u_eval = np.asarray(u_eval, dtype=float).reshape(-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig = ax.figure

    sc = ax.scatter(rs_eval[:, 0], rs_eval[:, 1], c=u_eval, s=12)
    fig.colorbar(sc, ax=ax)

    tri = np.vstack([vertices, vertices[0]])
    ax.plot(tri[:, 0], tri[:, 1], linewidth=1.5)

    if nodes is not None:
        nodes = np.asarray(nodes, dtype=float)
        ax.scatter(nodes[:, 0], nodes[:, 1], s=14, marker="o", alpha=0.8)

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


def plot_radial_profile(
    rho: np.ndarray,
    u: np.ndarray,
    theta_ids: np.ndarray,
    n_curves_to_show: int = 8,
    title: str | None = None,
    ax=None,
):
    """
    Plot several radial profiles u(rho) for selected angle indices.
    """
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)
    theta_ids = np.asarray(theta_ids, dtype=int).reshape(-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    unique_ids = np.unique(theta_ids)
    if unique_ids.size == 0:
        raise ValueError("theta_ids is empty.")

    step = max(1, unique_ids.size // n_curves_to_show)
    chosen = unique_ids[::step][:n_curves_to_show]

    for tid in chosen:
        mask = theta_ids == tid
        order = np.argsort(rho[mask])
        ax.plot(rho[mask][order], u[mask][order], label=f"theta_id={tid}")

    ax.set_xlabel("normalized radius rho")
    ax.set_ylabel("field value")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    if title is not None:
        ax.set_title(title)

    return fig, ax
