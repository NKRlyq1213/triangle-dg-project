from __future__ import annotations

import numpy as np

from operators.divergence_split import mapped_divergence_split_2d
from operators.exchange import (
    pair_face_traces,
    interior_face_pair_mismatches,
)


def volume_term_split_conservative(
    q_elem: np.ndarray,
    u_elem: np.ndarray,
    v_elem: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict,
) -> np.ndarray:
    """
    Volume term for the conservative split form:

        q_t + div(V q) = 0,   V = (u, v)

    implemented as

        RHS_vol = - div_h^split(V q)
    """
    q_elem = np.asarray(q_elem, dtype=float)
    u_elem = np.asarray(u_elem, dtype=float)
    v_elem = np.asarray(v_elem, dtype=float)

    return -mapped_divergence_split_2d(
        v=q_elem,
        a=u_elem,
        b=v_elem,
        Dr=Dr,
        Ds=Ds,
        xr=geom["xr"],
        xs=geom["xs"],
        yr=geom["yr"],
        ys=geom["ys"],
        J=geom["J"],
    )


def upwind_flux_and_penalty(
    ndotV: np.ndarray,
    qM: np.ndarray,
    qP: np.ndarray,
    tau: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Numerical flux from the document:

        f   = (n · V) qM
        f*  = 1/2[(n·V)qM + (n·V)qP] + (1-tau)/2 |n·V| (qM - qP)
        p   = f - f*

    For pure upwind, use tau = 0.
    """
    ndotV = np.asarray(ndotV, dtype=float)
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float)

    f = ndotV * qM
    fstar = 0.5 * (ndotV * qM + ndotV * qP) + 0.5 * (1.0 - tau) * np.abs(ndotV) * (qM - qP)
    p = f - fstar
    return f, fstar, p

def upwind_penalty_simplified(
    ndotV: np.ndarray,
    qM: np.ndarray,
    qP: np.ndarray,
) -> np.ndarray:
    """
    Simplified penalty term for pure upwind flux (tau = 0):

        p = f - f*
          = 0                      if n·V >= 0
          = (n·V) (qM - qP)       if n·V < 0

    Equivalently:
        p = min(n·V, 0) * (qM - qP)
    """
    ndotV = np.asarray(ndotV, dtype=float)
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float)
    if ndotV.shape != qM.shape or qM.shape != qP.shape:
        raise ValueError("ndotV, qM, qP must have the same shape.")

    return np.minimum(ndotV, 0.0) * (qM - qP)

def fill_boundary_exterior_state_upwind(
    qM: np.ndarray,
    qP: np.ndarray,
    ndotV: np.ndarray,
    is_boundary: np.ndarray,
    q_boundary_exact: np.ndarray,
) -> np.ndarray:
    """
    Fill exterior trace qP on boundary faces.

    Boundary policy
    ---------------
    - inflow  (n·V < 0): use exact boundary data
    - outflow (n·V >= 0): use qM itself

    Parameters
    ----------
    qM, qP, ndotV : np.ndarray
        Shape (K, 3, Nfp)
    is_boundary : np.ndarray
        Shape (K, 3)
    q_boundary_exact : np.ndarray
        Shape (K, 3, Nfp)

    Returns
    -------
    np.ndarray
        Boundary-filled qP, same shape as qM
    """
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float).copy()
    ndotV = np.asarray(ndotV, dtype=float)
    is_boundary = np.asarray(is_boundary, dtype=bool)
    q_boundary_exact = np.asarray(q_boundary_exact, dtype=float)

    if qM.shape != qP.shape or qM.shape != ndotV.shape or qM.shape != q_boundary_exact.shape:
        raise ValueError("qM, qP, ndotV, q_boundary_exact must all have the same shape.")
    if is_boundary.shape != qM.shape[:2]:
        raise ValueError("is_boundary must have shape (K, 3).")

    bd = is_boundary[:, :, None]
    inflow = bd & (ndotV < 0.0)
    outflow = bd & (ndotV >= 0.0)

    qP[inflow] = q_boundary_exact[inflow]
    qP[outflow] = qM[outflow]

    return qP

def fill_exterior_state(
    qP_exchange: np.ndarray,
    is_boundary: np.ndarray,
    q_boundary_exact: np.ndarray,
) -> np.ndarray:
    """
    Exterior state policy:
    - interior faces: use exchanged neighbor trace
    - physical boundary faces: use exact boundary data
    """
    qP_exchange = np.asarray(qP_exchange, dtype=float)
    q_boundary_exact = np.asarray(q_boundary_exact, dtype=float)
    is_boundary = np.asarray(is_boundary, dtype=bool)

    if qP_exchange.shape != q_boundary_exact.shape:
        raise ValueError("qP_exchange and q_boundary_exact must have the same shape.")
    if is_boundary.shape != qP_exchange.shape[:2]:
        raise ValueError("is_boundary must have shape (K, 3).")

    bd = is_boundary[:, :, None]   # shape (K, 3, 1)
    qP = np.where(bd, q_boundary_exact, qP_exchange)
    return qP

def surface_term_from_exchange(
    q_elem: np.ndarray,
    rule: dict,
    trace: dict,
    conn: dict,
    face_geom: dict,
    q_boundary,
    velocity,
    t: float = 0.0,
    tau: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Surface term using actual interior face exchange plus exact inflow boundary data.

    Current scope
    -------------
    - Table 1 only (embedded face nodes)
    - affine triangles only
    """
    q_elem = np.asarray(q_elem, dtype=float)
    if q_elem.ndim != 2:
        raise ValueError("q_elem must have shape (K, Np).")

    if trace.get("trace_mode", None) != "embedded":
        raise ValueError("This phase currently supports Table 1 embedded trace only.")

    paired = pair_face_traces(
        u_elem=q_elem,
        conn=conn,
        trace=trace,
        boundary_fill_value=np.nan,
    )

    qM = np.asarray(paired["uM"], dtype=float)
    qP = np.asarray(paired["uP"], dtype=float)

    x_face = np.asarray(face_geom["x_face"], dtype=float)
    y_face = np.asarray(face_geom["y_face"], dtype=float)
    nx = np.asarray(face_geom["nx"], dtype=float)
    ny = np.asarray(face_geom["ny"], dtype=float)

    u_face, v_face = velocity(x_face, y_face, t)
    u_face = np.asarray(u_face, dtype=float)
    v_face = np.asarray(v_face, dtype=float)

    ndotV = nx * u_face + ny * v_face

    qB = q_boundary(x_face, y_face, t)
    qB = np.asarray(qB, dtype=float)

    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)
    qP_filled = fill_exterior_state(
        qP_exchange=qP,
        is_boundary=is_boundary,
        q_boundary_exact=qB,
    )

    # pure upwind simplification from the document
    p = upwind_penalty_simplified(
        ndotV=ndotV,
        qM=qM,
        qP=qP_filled,
    )
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    K, Np = q_elem.shape
    surface_rhs = np.zeros((K, Np), dtype=float)

    for k in range(K):
        Ak = float(face_geom["area"][k])

        for face_id in (1, 2, 3):
            ids = np.asarray(trace["face_node_ids"][face_id], dtype=int)
            we = np.asarray(trace["face_weights"][face_id], dtype=float).reshape(-1)
            Lf = float(face_geom["length"][k, face_id - 1])

            # |T|^{-1} W^{-1} E^T W^e p
            surface_rhs[k, ids] += (Lf / Ak) * (we / ws[ids]) * p[k, face_id - 1, :]

    mismatches = interior_face_pair_mismatches(paired)

    diagnostics = {
        "qM": qM,
        "qP_before_boundary_fill": np.asarray(paired["uP"], dtype=float),
        "qP": qP_filled,
        "qB": qB,
        "u_face": u_face,
        "v_face": v_face,
        "ndotV": ndotV,
        "p": p,
        "x_face": x_face,
        "y_face": y_face,
        "nx": nx,
        "ny": ny,
        "interior_mismatches": mismatches,
    }
    return surface_rhs, diagnostics


def rhs_split_conservative_exchange(
    q_elem: np.ndarray,
    u_elem: np.ndarray,
    v_elem: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict,
    rule: dict,
    trace: dict,
    conn: dict,
    face_geom: dict,
    q_boundary,
    velocity,
    t: float = 0.0,
    tau: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Full semi-discrete RHS using actual interior face exchange.

        RHS = volume_rhs + surface_rhs
    """
    volume_rhs = volume_term_split_conservative(
        q_elem=q_elem,
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
    )

    surface_rhs, surface_diag = surface_term_from_exchange(
        q_elem=q_elem,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_boundary,
        velocity=velocity,
        t=t,
        tau=tau,
    )

    total_rhs = volume_rhs + surface_rhs

    diagnostics = dict(surface_diag)
    diagnostics["volume_rhs"] = volume_rhs
    diagnostics["surface_rhs"] = surface_rhs
    diagnostics["total_rhs"] = total_rhs

    return total_rhs, diagnostics