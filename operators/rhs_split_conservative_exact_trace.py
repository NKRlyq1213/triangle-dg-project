from __future__ import annotations

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _wrap(func):
            return func

        return _wrap

from operators.divergence_split import mapped_divergence_split_2d
from operators.exchange import evaluate_all_face_values
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.connectivity import build_face_connectivity
from operators.rhs_split_conservative_exchange import (
    _apply_exact_source_q_correction,
    _build_boundary_state_from_opposite_boundary,
    build_face_tau_array,
    build_surface_exchange_cache,
)


def _get_trace_face_arrays(trace: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Return face-node ids and face quadrature weights as contiguous arrays.

    Cached on `trace` to avoid repeated Python/dict -> ndarray conversions
    inside the RHS loop.
    """
    ids_key = "_face_node_ids_array"
    w_key = "_face_weights_array"

    face_node_ids = trace.get(ids_key)
    face_weights = trace.get(w_key)
    if face_node_ids is not None and face_weights is not None:
        return face_node_ids, face_weights

    face_node_ids = np.ascontiguousarray(
        np.stack(
            [np.asarray(trace["face_node_ids"][face_id], dtype=np.int64) for face_id in (1, 2, 3)],
            axis=0,
        ),
        dtype=np.int64,
    )
    face_weights = np.ascontiguousarray(
        np.stack(
            [np.asarray(trace["face_weights"][face_id], dtype=float).reshape(-1) for face_id in (1, 2, 3)],
            axis=0,
        ),
        dtype=float,
    )

    trace[ids_key] = face_node_ids
    trace[w_key] = face_weights
    return face_node_ids, face_weights


@njit(cache=True)
def _surface_lift_numba_kernel(
    p: np.ndarray,
    face_node_ids: np.ndarray,
    face_weight_scale: np.ndarray,
    length_over_area: np.ndarray,
    Np: int,
) -> np.ndarray:
    K = p.shape[0]
    Nfp = p.shape[2]
    out = np.zeros((K, Np), dtype=np.float64)

    for k in range(K):
        for f in range(3):
            s = length_over_area[k, f]
            for j in range(Nfp):
                nid = face_node_ids[f, j]
                out[k, nid] += s * face_weight_scale[f, j] * p[k, f, j]

    return out


def _surface_lift_vectorized(
    p: np.ndarray,
    face_node_ids: np.ndarray,
    face_weight_scale: np.ndarray,
    length_over_area: np.ndarray,
    Np: int,
) -> np.ndarray:
    K = p.shape[0]
    out = np.zeros((K, Np), dtype=float)

    for f in range(3):
        ids = face_node_ids[f]
        out[:, ids] += (
            length_over_area[:, f][:, None]
            * face_weight_scale[f, :][None, :]
            * p[:, f, :]
        )

    return out


def _surface_lift_exact_trace(
    p: np.ndarray,
    face_node_ids: np.ndarray,
    face_weights: np.ndarray,
    length: np.ndarray,
    area: np.ndarray,
    ws: np.ndarray,
    *,
    use_numba: bool,
) -> np.ndarray:
    ws = np.asarray(ws, dtype=float).reshape(-1)
    inv_ws = 1.0 / ws

    length_over_area = np.ascontiguousarray(
        np.asarray(length, dtype=float) / np.asarray(area, dtype=float)[:, None],
        dtype=float,
    )
    face_weight_scale = np.ascontiguousarray(
        np.asarray(face_weights, dtype=float) * inv_ws[np.asarray(face_node_ids, dtype=np.int64)],
        dtype=float,
    )

    p_arr = np.ascontiguousarray(np.asarray(p, dtype=float), dtype=float)
    ids_arr = np.ascontiguousarray(np.asarray(face_node_ids, dtype=np.int64), dtype=np.int64)

    if use_numba and _NUMBA_AVAILABLE:
        return _surface_lift_numba_kernel(
            p_arr,
            ids_arr,
            face_weight_scale,
            length_over_area,
            int(ws.size),
        )

    return _surface_lift_vectorized(
        p_arr,
        ids_arr,
        face_weight_scale,
        length_over_area,
        int(ws.size),
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
    tau: float | np.ndarray = 0.0,
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
    tau = np.asarray(tau, dtype=float)
    if tau.ndim == 0:
        tau = float(tau)
    elif tau.shape != ndotV.shape:
        raise ValueError("tau array must have the same shape as ndotV, qM, qP.")

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
    tau_interior: float | None = None,
    tau_qb: float | None = None,
    face_geom: dict | None = None,
    q_boundary=None,
    physical_boundary_mode: str = "exact_qb",
    q_boundary_correction=None,
    q_boundary_correction_mode: str = "all",
    use_numba: bool = False,
    conn: dict | None = None,
    surface_cache: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Surface term using exact trace values on interior faces.

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
    if conn is None:
        conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    cache = (
        build_surface_exchange_cache(rule, trace, conn, face_geom)
        if surface_cache is None
        else surface_cache
    )

    qM = evaluate_all_face_values(q_elem, trace)

    x_face = np.asarray(cache["x_face"], dtype=float)
    y_face = np.asarray(cache["y_face"], dtype=float)
    nx = np.asarray(cache["nx"], dtype=float)
    ny = np.asarray(cache["ny"], dtype=float)
    is_boundary = np.asarray(cache["is_boundary"], dtype=bool)
    interior_faces = ~is_boundary

    qB_interior_exact = q_exact(x_face, y_face, t)
    u_face, v_face = velocity(x_face, y_face, t)

    qB_interior_exact = np.asarray(qB_interior_exact, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    v_face = np.asarray(v_face, dtype=float)
    if qB_interior_exact.shape != qM.shape:
        raise ValueError("q_exact must return arrays with shape (K, 3, Nfp).")

    ndotV = nx * u_face + ny * v_face
    boundary_mode = str(physical_boundary_mode).strip().lower()
    if boundary_mode not in ("exact_qb", "opposite_boundary"):
        raise ValueError("physical_boundary_mode must be one of: 'exact_qb', 'opposite_boundary'.")

    qP = np.empty_like(qM)
    qP[interior_faces] = qB_interior_exact[interior_faces]

    qB_boundary_exact = None
    if boundary_mode == "exact_qb":
        q_boundary_fn = q_exact if q_boundary is None else q_boundary
        qB_boundary_exact = np.asarray(q_boundary_fn(x_face, y_face, t), dtype=float)
        if qB_boundary_exact.shape != qM.shape:
            raise ValueError("q_boundary must return arrays with shape (K, 3, Nfp).")
        qP[is_boundary] = qB_boundary_exact[is_boundary]
    else:
        qP_boundary = _build_boundary_state_from_opposite_boundary(q_elem=q_elem, cache=cache)
        qP[is_boundary] = qP_boundary[is_boundary]

    qB_exact = np.full_like(qM, np.nan)
    qB_exact[interior_faces] = qB_interior_exact[interior_faces]
    exact_source_faces = np.array(interior_faces, copy=True)
    if qB_boundary_exact is not None:
        qB_exact[is_boundary] = qB_boundary_exact[is_boundary]
        exact_source_faces |= is_boundary

    if q_boundary_correction is not None:
        qP_exact_corrected = _apply_exact_source_q_correction(
            q_exact_source=qB_exact,
            x_face=x_face,
            y_face=y_face,
            t=t,
            qM=qM,
            ndotV=ndotV,
            active_faces=exact_source_faces,
            q_boundary_correction=q_boundary_correction,
            q_boundary_correction_mode=q_boundary_correction_mode,
        )
        qP[exact_source_faces] = qP_exact_corrected[exact_source_faces]

    tau_interior_eff, tau_qb_eff, tau_face = build_face_tau_array(
        is_boundary=is_boundary,
        face_shape=qM.shape,
        physical_boundary_mode=boundary_mode,
        tau=tau,
        tau_interior=tau_interior,
        tau_qb=tau_qb,
    )

    f, fstar, p = upwind_flux_and_penalty(ndotV, qM, qP, tau=tau_face)

    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    face_node_ids, face_weights = _get_trace_face_arrays(trace)

    surface_rhs = _surface_lift_exact_trace(
        p=p,
        face_node_ids=face_node_ids,
        face_weights=face_weights,
        length=np.asarray(face_geom["length"], dtype=float),
        area=np.asarray(face_geom["area"], dtype=float),
        ws=ws,
        use_numba=use_numba,
    )

    qP_interior = np.full_like(qM, np.nan)
    qP_interior[interior_faces] = qP[interior_faces]
    qP_boundary = np.full_like(qM, np.nan)
    qP_boundary[is_boundary] = qP[is_boundary]

    diagnostics = {
        "qM": qM,
        "qP_interior": qP_interior,
        "qP_boundary": qP_boundary,
        "qB_exact": qB_exact,
        "qP_exact": qB_exact,
        "qP": qP,
        "tau_interior": float(tau_interior_eff),
        "tau_qb": float(tau_qb_eff),
        "tau_face": tau_face,
        "physical_boundary_mode": boundary_mode,
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
    tau_interior: float | None = None,
    tau_qb: float | None = None,
    face_geom: dict | None = None,
    q_boundary=None,
    physical_boundary_mode: str = "exact_qb",
    q_boundary_correction=None,
    q_boundary_correction_mode: str = "all",
    use_numba: bool = False,
    conn: dict | None = None,
    surface_cache: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Full semi-discrete RHS:

        RHS = volume_rhs + surface_rhs

    with exact-trace substitution on interior faces and physical-boundary
    exterior data selected independently.
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
        tau_interior=tau_interior,
        tau_qb=tau_qb,
        face_geom=face_geom,
        q_boundary=q_boundary,
        physical_boundary_mode=physical_boundary_mode,
        q_boundary_correction=q_boundary_correction,
        q_boundary_correction_mode=q_boundary_correction_mode,
        use_numba=use_numba,
        conn=conn,
        surface_cache=surface_cache,
    )

    total_rhs = volume_rhs + surface_rhs

    diagnostics = dict(surface_diag)
    diagnostics["volume_rhs"] = volume_rhs
    diagnostics["surface_rhs"] = surface_rhs
    diagnostics["total_rhs"] = total_rhs

    return total_rhs, diagnostics
