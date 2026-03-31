from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _triangle_closed(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    return np.vstack([vertices, vertices[0]])


def _axis_limits_from_xy(x: np.ndarray, y: np.ndarray, pad_ratio: float = 0.06):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    dx = max(xmax - xmin, 1e-14)
    dy = max(ymax - ymin, 1e-14)

    return (
        (xmin - pad_ratio * dx, xmax + pad_ratio * dx),
        (ymin - pad_ratio * dy, ymax + pad_ratio * dy),
    )


def build_reference_triangulation(rs: np.ndarray) -> np.ndarray:
    """
    Build a local triangulation on reference points rs.

    Parameters
    ----------
    rs : np.ndarray
        Reference points of shape (Np, 2)

    Returns
    -------
    np.ndarray
        Local triangle connectivity of shape (Nt, 3)
    """
    rs = np.asarray(rs, dtype=float)
    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs must have shape (Np, 2).")

    triang = mtri.Triangulation(rs[:, 0], rs[:, 1])
    return triang.triangles.copy()


def lift_local_triangulation_to_global(
    local_triangles: np.ndarray,
    K: int,
    nloc: int,
) -> np.ndarray:
    """
    Repeat one local triangulation for K elements.

    Parameters
    ----------
    local_triangles : np.ndarray
        Shape (Nt, 3), local connectivity
    K : int
        Number of elements
    nloc : int
        Number of local points per element

    Returns
    -------
    np.ndarray
        Global connectivity of shape (K*Nt, 3)
    """
    local_triangles = np.asarray(local_triangles, dtype=int)
    if local_triangles.ndim != 2 or local_triangles.shape[1] != 3:
        raise ValueError("local_triangles must have shape (Nt, 3).")
    if K < 1 or nloc < 1:
        raise ValueError("K and nloc must be >= 1.")

    out = []
    for k in range(K):
        out.append(local_triangles + k * nloc)
    return np.vstack(out)


def plot_reference_rule_nodes(
    rs_nodes: np.ndarray,
    vertices: np.ndarray,
    title: str | None = None,
    annotate_indices: bool = False,
    ax=None,
):
    """
    Plot table nodes on the reference triangle.
    """
    rs_nodes = np.asarray(rs_nodes, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    if rs_nodes.ndim != 2 or rs_nodes.shape[1] != 2:
        raise ValueError("rs_nodes must have shape (Np, 2).")
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 6.0))
    else:
        fig = ax.figure

    tri = _triangle_closed(vertices)
    ax.plot(tri[:, 0], tri[:, 1], linewidth=1.6)
    ax.scatter(rs_nodes[:, 0], rs_nodes[:, 1], s=28, zorder=3)

    if annotate_indices:
        for i, (r, s) in enumerate(rs_nodes):
            ax.text(r, s, f"{i}", fontsize=8, ha="left", va="bottom")

    xlim, ylim = _axis_limits_from_xy(vertices[:, 0], vertices[:, 1])

    ax.set_aspect("equal")
    ax.set_xlabel("r")
    ax.set_ylabel("s")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)

    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_physical_mesh_nodes(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    X_nodes: np.ndarray,
    Y_nodes: np.ndarray,
    title: str | None = None,
    show_element_ids: bool = True,
    show_vertex_ids: bool = False,
    node_size: float = 14.0,
    ax=None,
):
    """
    Plot the physical triangular mesh and mapped table nodes.
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)
    X_nodes = np.asarray(X_nodes, dtype=float)
    Y_nodes = np.asarray(Y_nodes, dtype=float)

    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")
    if X_nodes.shape != Y_nodes.shape:
        raise ValueError("X_nodes and Y_nodes must have the same shape.")
    if X_nodes.ndim != 2 or X_nodes.shape[0] != EToV.shape[0]:
        raise ValueError("X_nodes and Y_nodes must have shape (K, Np).")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.2))
    else:
        fig = ax.figure

    # mesh edges
    for k, tri_vids in enumerate(EToV):
        verts = np.column_stack([VX[tri_vids], VY[tri_vids]])
        closed = _triangle_closed(verts)
        ax.plot(closed[:, 0], closed[:, 1], linewidth=1.2)

        if show_element_ids:
            cx = np.mean(verts[:, 0])
            cy = np.mean(verts[:, 1])
            ax.text(cx, cy, f"E{k}", fontsize=8, ha="center", va="center")

    # nodes
    ax.scatter(X_nodes.ravel(), Y_nodes.ravel(), s=node_size, alpha=0.85, zorder=3)

    if show_vertex_ids:
        for vid, (xv, yv) in enumerate(zip(VX, VY)):
            ax.text(xv, yv, f"V{vid}", fontsize=8, ha="left", va="bottom")

    xlim, ylim = _axis_limits_from_xy(VX, VY)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)

    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_physical_field_and_nodes(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    U_eval: np.ndarray,
    local_triangles: np.ndarray,
    X_nodes: np.ndarray | None = None,
    Y_nodes: np.ndarray | None = None,
    title: str | None = None,
    levels: int = 20,
    show_nodes: bool = True,
    show_mesh_edges: bool = True,
    node_size: float = 10.0,
    ax=None,
):
    """
    Plot a scalar field on the physical mesh using per-element lifted triangulation,
    with optional overlay of mapped table nodes.
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)
    X_eval = np.asarray(X_eval, dtype=float)
    Y_eval = np.asarray(Y_eval, dtype=float)
    U_eval = np.asarray(U_eval, dtype=float)
    local_triangles = np.asarray(local_triangles, dtype=int)

    if X_eval.shape != Y_eval.shape or X_eval.shape != U_eval.shape:
        raise ValueError("X_eval, Y_eval, U_eval must have the same shape.")
    if X_eval.ndim != 2 or X_eval.shape[0] != EToV.shape[0]:
        raise ValueError("X_eval, Y_eval, U_eval must have shape (K, Nq).")
    if local_triangles.ndim != 2 or local_triangles.shape[1] != 3:
        raise ValueError("local_triangles must have shape (Nt, 3).")

    K, Nq = X_eval.shape
    global_triangles = lift_local_triangulation_to_global(local_triangles, K, Nq)

    xg = X_eval.ravel()
    yg = Y_eval.ravel()
    ug = U_eval.ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.2))
    else:
        fig = ax.figure

    triang = mtri.Triangulation(xg, yg, triangles=global_triangles)
    contour = ax.tricontourf(triang, ug, levels=levels)
    fig.colorbar(contour, ax=ax)

    if show_mesh_edges:
        for tri_vids in EToV:
            verts = np.column_stack([VX[tri_vids], VY[tri_vids]])
            closed = _triangle_closed(verts)
            ax.plot(closed[:, 0], closed[:, 1], linewidth=1.0, color="k", alpha=0.8)

    if show_nodes and X_nodes is not None and Y_nodes is not None:
        X_nodes = np.asarray(X_nodes, dtype=float)
        Y_nodes = np.asarray(Y_nodes, dtype=float)
        if X_nodes.shape != Y_nodes.shape:
            raise ValueError("X_nodes and Y_nodes must have the same shape.")
        ax.scatter(X_nodes.ravel(), Y_nodes.ravel(), s=node_size, alpha=0.85, zorder=3)

    xlim, ylim = _axis_limits_from_xy(VX, VY)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.20)

    if title is not None:
        ax.set_title(title)

    return fig, ax