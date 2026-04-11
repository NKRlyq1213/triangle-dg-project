from __future__ import annotations

import importlib
import numpy as np

try:
    njit = importlib.import_module("numba").njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    _NUMBA_AVAILABLE = False

from operators.divergence_split import mapped_divergence_split_2d
from operators.exchange import (
    pair_face_traces,
    interior_face_pair_mismatches,
)


def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _surface_rhs_kernel_inplace(
        p: np.ndarray,
        area: np.ndarray,
        length: np.ndarray,
        ids_f1: np.ndarray,
        ids_f2: np.ndarray,
        ids_f3: np.ndarray,
        wr_f1: np.ndarray,
        wr_f2: np.ndarray,
        wr_f3: np.ndarray,
        surface_rhs: np.ndarray,
    ) -> None:
        K = p.shape[0]
        nfp = p.shape[2]

        for k in range(K):
            s1 = length[k, 0] / area[k]
            for i in range(nfp):
                surface_rhs[k, ids_f1[i]] += s1 * wr_f1[i] * p[k, 0, i]

            s2 = length[k, 1] / area[k]
            for i in range(nfp):
                surface_rhs[k, ids_f2[i]] += s2 * wr_f2[i] * p[k, 1, i]

            s3 = length[k, 2] / area[k]
            for i in range(nfp):
                surface_rhs[k, ids_f3[i]] += s3 * wr_f3[i] * p[k, 2, i]

else:
    _surface_rhs_kernel_inplace = None


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
    compute_mismatches: bool = True,
    return_diagnostics: bool = True,
    use_numba: bool | None = None,
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
        use_numba=use_numba,
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
    inv_ws = 1.0 / ws

    K, Np = q_elem.shape
    surface_rhs = np.zeros((K, Np), dtype=float)

    area = np.asarray(face_geom["area"], dtype=float).reshape(-1)
    length = np.asarray(face_geom["length"], dtype=float)

    face_node_ids = {
        face_id: np.asarray(trace["face_node_ids"][face_id], dtype=int).reshape(-1)
        for face_id in (1, 2, 3)
    }
    face_weight_over_ws = {
        face_id: np.asarray(trace["face_weights"][face_id], dtype=float).reshape(-1)
        * inv_ws[face_node_ids[face_id]]
        for face_id in (1, 2, 3)
    }

    should_use_numba = _should_use_numba(use_numba)
    nfp = p.shape[2]

    if should_use_numba:
        ids_f1 = np.ascontiguousarray(face_node_ids[1], dtype=np.int64)
        ids_f2 = np.ascontiguousarray(face_node_ids[2], dtype=np.int64)
        ids_f3 = np.ascontiguousarray(face_node_ids[3], dtype=np.int64)

        wr_f1 = np.ascontiguousarray(face_weight_over_ws[1], dtype=np.float64)
        wr_f2 = np.ascontiguousarray(face_weight_over_ws[2], dtype=np.float64)
        wr_f3 = np.ascontiguousarray(face_weight_over_ws[3], dtype=np.float64)

        if (
            ids_f1.size != nfp
            or ids_f2.size != nfp
            or ids_f3.size != nfp
            or wr_f1.size != nfp
            or wr_f2.size != nfp
            or wr_f3.size != nfp
        ):
            raise ValueError("Trace face-node count must match p.shape[2].")

        _surface_rhs_kernel_inplace(
            p=np.ascontiguousarray(p, dtype=np.float64),
            area=np.ascontiguousarray(area, dtype=np.float64),
            length=np.ascontiguousarray(length, dtype=np.float64),
            ids_f1=ids_f1,
            ids_f2=ids_f2,
            ids_f3=ids_f3,
            wr_f1=wr_f1,
            wr_f2=wr_f2,
            wr_f3=wr_f3,
            surface_rhs=surface_rhs,
        )
    else:
        for face_id in (1, 2, 3):
            jf = face_id - 1
            ids = face_node_ids[face_id]
            w_ratio = face_weight_over_ws[face_id]

            if ids.size != p.shape[2] or w_ratio.size != p.shape[2]:
                raise ValueError("Trace face-node count must match p.shape[2].")

            # Batch form of |T|^{-1} W^{-1} E^T W^e p on one face for all elements.
            scale = (length[:, jf] / area)[:, None]
            face_contrib = scale * w_ratio[None, :] * p[:, jf, :]

            if np.unique(ids).size == ids.size:
                surface_rhs[:, ids] += face_contrib
            else:
                row_idx = np.broadcast_to(np.arange(K)[:, None], face_contrib.shape)
                col_idx = np.broadcast_to(ids[None, :], face_contrib.shape)
                np.add.at(surface_rhs, (row_idx, col_idx), face_contrib)

    if not return_diagnostics:
        return surface_rhs, {}

    mismatches = interior_face_pair_mismatches(paired) if compute_mismatches else []

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
    compute_mismatches: bool = True,
    return_diagnostics: bool = True,
    use_numba: bool | None = None,
    state_projector: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Full semi-discrete RHS using actual interior face exchange.

        RHS = volume_rhs + surface_rhs
    """
    q_work = np.asarray(q_elem, dtype=float)
    projector = None
    if state_projector is not None:
        projector = np.asarray(state_projector, dtype=float)
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise ValueError("state_projector must be a square 2D array.")
        if projector.shape[0] != q_work.shape[1]:
            raise ValueError("state_projector size must match q_elem.shape[1].")
        q_work = q_work @ projector.T

    volume_rhs = volume_term_split_conservative(
        q_elem=q_work,
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
    )

    surface_rhs, surface_diag = surface_term_from_exchange(
        q_elem=q_work,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_boundary,
        velocity=velocity,
        t=t,
        tau=tau,
        compute_mismatches=compute_mismatches,
        return_diagnostics=return_diagnostics,
        use_numba=use_numba,
    )

    total_rhs = volume_rhs + surface_rhs
    if projector is not None:
        total_rhs = total_rhs @ projector.T

    if not return_diagnostics:
        return total_rhs, {}

    diagnostics = dict(surface_diag)
    diagnostics["volume_rhs"] = volume_rhs
    diagnostics["surface_rhs"] = surface_rhs
    diagnostics["total_rhs"] = total_rhs

    return total_rhs, diagnostics