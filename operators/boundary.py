from __future__ import annotations

import numpy as np

from data.edge_rules import edge_gl1d_rule
from operators.mass import mass_matrix_from_quadrature
from operators.vandermonde2d import vandermonde2d


def xy_to_rs(xy: np.ndarray) -> np.ndarray:
    """
    Convert reference-triangle Cartesian coordinates (xi, eta) to (r, s).

    Reference triangle:
        (xi, eta) in {(0,1), (0,0), (1,0)}

    Mapping:
        r = 2*xi - 1
        s = 2*eta - 1
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (n_points, 2).")

    rs = np.empty_like(xy)
    rs[:, 0] = 2.0 * xy[:, 0] - 1.0
    rs[:, 1] = 2.0 * xy[:, 1] - 1.0
    return rs


def edge_nodes_rs(edge_id: int, n_edge: int) -> np.ndarray:
    """
    Return GL1D edge nodes mapped onto the reference triangle, in (r,s).
    """
    rule = edge_gl1d_rule(edge_id=edge_id, n=n_edge)
    return xy_to_rs(rule.xy)


def edge_vandermonde2d(
    N: int,
    edge_id: int,
    n_edge: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the edge evaluation Vandermonde matrix on GL1D nodes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (V_edge, rs_edge)
    """
    rs_edge = edge_nodes_rs(edge_id=edge_id, n_edge=n_edge)
    r_e = rs_edge[:, 0]
    s_e = rs_edge[:, 1]
    V_edge = vandermonde2d(N, r_e, s_e)
    return V_edge, rs_edge


def volume_to_edge_operator(
    V_vol: np.ndarray,
    weights: np.ndarray,
    V_edge: np.ndarray,
    area: float = 1.0,
) -> np.ndarray:
    """
    Build the volume-to-edge trace/evaluation operator:

        E = V_edge (A V^T W V)^(-1) (A V^T W)

    Parameters
    ----------
    V_vol : np.ndarray
        Volume Vandermonde matrix of shape (n_vol, n_modes).
    weights : np.ndarray
        Volume quadrature weights of shape (n_vol,).
    V_edge : np.ndarray
        Edge evaluation Vandermonde matrix of shape (n_edge, n_modes).
    area : float
        Geometric area factor.

    Returns
    -------
    np.ndarray
        Trace operator E of shape (n_edge, n_vol).
    """
    V_vol = np.asarray(V_vol, dtype=float)
    V_edge = np.asarray(V_edge, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if V_vol.ndim != 2 or V_edge.ndim != 2:
        raise ValueError("V_vol and V_edge must be 2D.")
    if V_vol.shape[0] != weights.size:
        raise ValueError("weights size must match V_vol.shape[0].")
    if V_vol.shape[1] != V_edge.shape[1]:
        raise ValueError("V_vol and V_edge must have the same number of modes.")

    M = mass_matrix_from_quadrature(V_vol, weights, area=area)
    rhs = area * (V_vol.T * weights[None, :])
    proj = np.linalg.solve(M, rhs)

    E = V_edge @ proj
    return E


def evaluate_on_edge(
    u_vol: np.ndarray,
    V_vol: np.ndarray,
    weights: np.ndarray,
    N: int,
    edge_id: int,
    n_edge: int,
    area: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate volume nodal/sample values on one triangle edge via weighted projection.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (u_edge, rs_edge, E_edge)
    """
    u_vol = np.asarray(u_vol, dtype=float).reshape(-1)
    V_vol = np.asarray(V_vol, dtype=float)

    if V_vol.shape[0] != u_vol.size:
        raise ValueError("u_vol size must match V_vol.shape[0].")

    V_edge, rs_edge = edge_vandermonde2d(N=N, edge_id=edge_id, n_edge=n_edge)
    E_edge = volume_to_edge_operator(
        V_vol=V_vol,
        weights=weights,
        V_edge=V_edge,
        area=area,
    )
    u_edge = E_edge @ u_vol
    return u_edge, rs_edge, E_edge


def evaluate_on_all_edges(
    u_vol: np.ndarray,
    V_vol: np.ndarray,
    weights: np.ndarray,
    N: int,
    n_edge: int,
    area: float = 1.0,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Evaluate volume nodal/sample values on all three triangle edges.
    """
    out: dict[int, dict[str, np.ndarray]] = {}
    for edge_id in [1, 2, 3]:
        u_edge, rs_edge, E_edge = evaluate_on_edge(
            u_vol=u_vol,
            V_vol=V_vol,
            weights=weights,
            N=N,
            edge_id=edge_id,
            n_edge=n_edge,
            area=area,
        )
        out[edge_id] = {
            "u_edge": u_edge,
            "rs_edge": rs_edge,
            "E_edge": E_edge,
        }
    return out
