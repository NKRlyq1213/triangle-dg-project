from __future__ import annotations

import numpy as np


def reference_shape_functions(r, s) -> np.ndarray:
    """
    Affine shape functions on Hesthaven's reference triangle:
        v1 = (-1, -1)
        v2 = ( 1, -1)
        v3 = (-1,  1)

    For each point (r, s), return:
        [phi1, phi2, phi3]
    such that
        x(r,s) = phi1*x1 + phi2*x2 + phi3*x3
        y(r,s) = phi1*y1 + phi2*y2 + phi3*y3

    Parameters
    ----------
    r, s : array_like
        Reference coordinates with the same shape.

    Returns
    -------
    np.ndarray
        Shape (..., 3)
    """
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    if r.shape != s.shape:
        raise ValueError("r and s must have the same shape.")

    phi1 = -(r + s) / 2.0
    phi2 = (1.0 + r) / 2.0
    phi3 = (1.0 + s) / 2.0

    return np.stack([phi1, phi2, phi3], axis=-1)


def map_ref_to_phys(
    r,
    s,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map points (r, s) from the reference triangle to a physical triangle.

    Parameters
    ----------
    r, s : array_like
        Reference coordinates with the same shape.
    vertices : np.ndarray
        Physical triangle vertices of shape (3, 2), ordered as:
            v1, v2, v3

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x, y), each with the same shape as r and s
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    phi = reference_shape_functions(r, s)   # (..., 3)

    x = phi[..., 0] * vertices[0, 0] + phi[..., 1] * vertices[1, 0] + phi[..., 2] * vertices[2, 0]
    y = phi[..., 0] * vertices[0, 1] + phi[..., 1] * vertices[1, 1] + phi[..., 2] * vertices[2, 1]
    return x, y


def map_ref_to_phys_points(
    rs: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Vectorized wrapper:
        rs.shape = (Np, 2)
        vertices.shape = (3, 2)

    Returns
    -------
    np.ndarray
        Physical points of shape (Np, 2)
    """
    rs = np.asarray(rs, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs must have shape (Np, 2).")
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    x, y = map_ref_to_phys(rs[:, 0], rs[:, 1], vertices)
    return np.column_stack([x, y])


def element_vertices(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_id: int,
) -> np.ndarray:
    """
    Extract the 3 physical vertices of one element.

    Parameters
    ----------
    VX, VY : np.ndarray
        Global vertex coordinates, shape (Nv,)
    EToV : np.ndarray
        Element-to-vertex connectivity, shape (K, 3)
    elem_id : int
        Element index

    Returns
    -------
    np.ndarray
        Vertices of the selected element, shape (3, 2)
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)

    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")
    if not (0 <= elem_id < EToV.shape[0]):
        raise IndexError("elem_id out of range.")

    vids = EToV[elem_id]
    return np.column_stack([VX[vids], VY[vids]])


def map_reference_nodes_to_element(
    rs_nodes: np.ndarray,
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_id: int,
) -> np.ndarray:
    """
    Map reference nodes of shape (Np, 2) to one physical element.

    Returns
    -------
    np.ndarray
        Physical coordinates of shape (Np, 2)
    """
    verts = element_vertices(VX, VY, EToV, elem_id)
    return map_ref_to_phys_points(rs_nodes, verts)


def map_reference_nodes_to_all_elements(
    rs_nodes: np.ndarray,
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map the same reference nodes to all elements.

    Parameters
    ----------
    rs_nodes : np.ndarray
        Reference nodes, shape (Np, 2)
    VX, VY : np.ndarray
        Global vertex coordinates
    EToV : np.ndarray
        Element-to-vertex connectivity, shape (K, 3)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X, Y with shape (K, Np)
        where
            X[k, i], Y[k, i]
        are the physical coordinates of node i in element k.
    """
    rs_nodes = np.asarray(rs_nodes, dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    if rs_nodes.ndim != 2 or rs_nodes.shape[1] != 2:
        raise ValueError("rs_nodes must have shape (Np, 2).")
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    K = EToV.shape[0]
    Np = rs_nodes.shape[0]

    X = np.zeros((K, Np), dtype=float)
    Y = np.zeros((K, Np), dtype=float)

    for k in range(K):
        verts = element_vertices(VX, VY, EToV, k)
        xk, yk = map_ref_to_phys(rs_nodes[:, 0], rs_nodes[:, 1], verts)
        X[k, :] = xk
        Y[k, :] = yk

    return X, Y