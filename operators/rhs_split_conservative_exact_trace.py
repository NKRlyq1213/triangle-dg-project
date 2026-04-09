from __future__ import annotations

import numpy as np

from operators.divergence_split import mapped_divergence_split_2d
from operators.exchange import evaluate_all_face_values
from geometry.face_metrics import affine_face_geometry_from_mesh


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
    Upwind-family numerical flux from the document:

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


def surface_term_from_exact_trace(
    q_elem: np.ndarray,
    rule: dict,
    trace: dict,
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    q_exact,
    velocity,
    t: float = 0.0,
    tau: float = 0.0,
    face_geom: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Surface term using exact trace values on all faces.

    Current scope
    -------------
    - Table 1 only (embedded face nodes)
    - affine triangles only

    Returns
    -------
    surface_rhs : np.ndarray
        Shape (K, Np)
    diagnostics : dict
        Contains qM, qP, ndotV, f, fstar, p, x_face, y_face
    """
    q_elem = np.asarray(q_elem, dtype=float)
    if q_elem.ndim != 2:
        raise ValueError("q_elem must have shape (K, Np).")

    if trace.get("trace_mode", None) != "embedded":
        raise ValueError("This phase-1 implementation only supports embedded Table 1 trace.")

    if face_geom is None:
        face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    qM = evaluate_all_face_values(q_elem, trace)

    x_face = face_geom["x_face"]
    y_face = face_geom["y_face"]
    nx = face_geom["nx"]
    ny = face_geom["ny"]

    qP = q_exact(x_face, y_face, t)
    u_face, v_face = velocity(x_face, y_face, t)

    qP = np.asarray(qP, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    v_face = np.asarray(v_face, dtype=float)

    ndotV = nx * u_face + ny * v_face
    f, fstar, p = upwind_flux_and_penalty(ndotV, qM, qP, tau=tau)

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
            # with W^e = |edge| * diag(we)
            surface_rhs[k, ids] += (Lf / Ak) * (we / ws[ids]) * p[k, face_id - 1, :]

    diagnostics = {
        "qM": qM,
        "qP": qP,
        "u_face": u_face,
        "v_face": v_face,
        "ndotV": ndotV,
        "f": f,
        "fstar": fstar,
        "p": p,
        "x_face": x_face,
        "y_face": y_face,
        "nx": nx,
        "ny": ny,
    }
    return surface_rhs, diagnostics


def rhs_split_conservative_exact_trace(
    q_elem: np.ndarray,
    u_elem: np.ndarray,
    v_elem: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict,
    rule: dict,
    trace: dict,
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    q_exact,
    velocity,
    t: float = 0.0,
    tau: float = 0.0,
    face_geom: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Full semi-discrete RHS:

        RHS = volume_rhs + surface_rhs

    under exact-trace substitution on all faces.
    """
    volume_rhs = volume_term_split_conservative(
        q_elem=q_elem,
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
    )

    surface_rhs, surface_diag = surface_term_from_exact_trace(
        q_elem=q_elem,
        rule=rule,
        trace=trace,
        VX=VX,
        VY=VY,
        EToV=EToV,
        q_exact=q_exact,
        velocity=velocity,
        t=t,
        tau=tau,
        face_geom=face_geom,
    )

    total_rhs = volume_rhs + surface_rhs

    diagnostics = dict(surface_diag)
    diagnostics["volume_rhs"] = volume_rhs
    diagnostics["surface_rhs"] = surface_rhs
    diagnostics["total_rhs"] = total_rhs

    return total_rhs, diagnostics