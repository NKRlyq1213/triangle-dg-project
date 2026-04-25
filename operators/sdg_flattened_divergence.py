from __future__ import annotations

import numpy as np

from geometry.metrics import affine_geometric_factors_from_mesh, divergence_2d
from geometry.reference_triangle import reference_triangle_area
from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)


def build_table1_reference_diff_operators(rule: dict, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build reference differentiation matrices Dr, Ds on Table1 nodes.

    Parameters
    ----------
    rule : dict
        Output from data.table1_rules.load_table1_rule(order).
    N : int
        Polynomial degree used for the modal basis.

    Returns
    -------
    Dr, Ds : np.ndarray
        Differentiation matrices acting on nodal values on Table1 nodes.

    Notes
    -----
    If V is square, use exact nodal differentiation.
    If V is tall, use weighted projection differentiation.
    """
    rs = np.asarray(rule["rs"], dtype=float)
    w = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])

    if V.shape[0] == V.shape[1]:
        return differentiation_matrices_square(V, Vr, Vs)

    return differentiation_matrices_weighted(
        V=V,
        Vr=Vr,
        Vs=Vs,
        weights=w,
        area=reference_triangle_area(),
    )


def sdg_flattened_cartesian_divergence(
    q: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    rs_nodes: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Compute volume-only SDG flattened Cartesian divergence:

        div = ∂x(u1*q) + ∂y(u2*q)

    on the flattened square mesh.

    Parameters
    ----------
    q, u1, u2 : np.ndarray
        Shape (K,Np). Nodal scalar field and flattened velocity components.
    Dr, Ds : np.ndarray
        Reference differentiation matrices.
    VX, VY, EToV : np.ndarray
        Flattened-square triangular mesh.
    rs_nodes : np.ndarray
        Table1 reference nodes, shape (Np,2).
    mask : np.ndarray | None
        Optional bad-node mask. If provided, masked output is set to NaN.

    Returns
    -------
    div : np.ndarray
        Shape (K,Np).
    """
    q = np.asarray(q, dtype=float)
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)

    if not (q.shape == u1.shape == u2.shape):
        raise ValueError("q, u1, u2 must have the same shape.")
    if q.ndim != 2:
        raise ValueError("q, u1, u2 must have shape (K,Np).")

    Fx = u1 * q
    Fy = u2 * q

    geom = affine_geometric_factors_from_mesh(
        VX=VX,
        VY=VY,
        EToV=EToV,
        rs_nodes=rs_nodes,
    )

    div = divergence_2d(
        Fx=Fx,
        Fy=Fy,
        Dr=Dr,
        Ds=Ds,
        rx=geom["rx"],
        sx=geom["sx"],
        ry=geom["ry"],
        sy=geom["sy"],
    )

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != div.shape:
            raise ValueError("mask must have the same shape as div.")
        div = np.where(mask, np.nan, div)

    return div


def divergence_stats_by_patch(
    values: np.ndarray,
    patch_id: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[int, dict[str, float]]:
    """
    Per-patch min/max/mean statistics for diagnostics.
    """
    values = np.asarray(values, dtype=float)
    patch_id = np.asarray(patch_id, dtype=int)

    if values.shape != patch_id.shape:
        raise ValueError("values and patch_id must have the same shape.")

    if mask is None:
        good = np.isfinite(values)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("mask must have the same shape as values.")
        good = (~mask) & np.isfinite(values)

    out: dict[int, dict[str, float]] = {}

    for pid in range(1, 9):
        sel = (patch_id == pid) & good
        vals = values[sel]
        if vals.size == 0:
            out[pid] = {
                "n": 0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "linf": float("nan"),
            }
            continue

        out[pid] = {
            "n": int(vals.size),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "linf": float(np.max(np.abs(vals))),
        }

    return out
