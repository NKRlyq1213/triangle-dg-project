from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np

from data.table1_rules import load_table1_rule
from geometry.sphere_manifold_metrics import ManifoldGeometryCache
from operators.exchange import evaluate_all_face_values, pair_face_traces
from operators.manifold_rhs import build_manifold_face_connectivity
from operators.trace_policy import build_trace_policy
from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d

try:
    _numba = importlib.import_module("numba")
    njit = _numba.njit
    prange = getattr(_numba, "prange", range)
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
    prange = range
    _NUMBA_AVAILABLE = False


_VALID_FLUX_TYPES = {"upwind", "central", "lax_friedrichs", "lf", "LF"}


def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


def _normalize_flux_type(flux_type: str) -> str:
    flux = str(flux_type).strip()
    if flux == "LF":
        return "lax_friedrichs"
    flux = flux.lower()
    if flux == "lf":
        return "lax_friedrichs"
    if flux not in {"upwind", "central", "lax_friedrichs"}:
        raise ValueError("flux_type must be one of: upwind, central, lax_friedrichs.")
    return flux


@dataclass(frozen=True)
class ManifoldSBPChanReferenceOperators:
    rule: dict
    trace: dict
    rs_nodes: np.ndarray
    weights_2d: np.ndarray
    weights_1d: np.ndarray
    V: np.ndarray
    Vr: np.ndarray
    Vs: np.ndarray
    P: np.ndarray
    Dr_sbp: np.ndarray
    Ds_sbp: np.ndarray
    E: np.ndarray
    Q_tilde_r: np.ndarray
    Q_tilde_s: np.ndarray
    face_weights_flat: np.ndarray
    nr_flat: np.ndarray
    ns_flat: np.ndarray
    nfp: int


def _face_extraction_matrix(trace: dict, Np: int) -> np.ndarray:
    nfp = int(trace["nfp"])
    E = np.zeros((3 * nfp, Np), dtype=float)
    for face_id in (1, 2, 3):
        ids = np.asarray(trace["face_node_ids"][face_id], dtype=int)
        rows = slice((face_id - 1) * nfp, face_id * nfp)
        E[rows, ids] = 1.0
    return E


def _flat_face_weights(trace: dict) -> np.ndarray:
    return np.concatenate(
        [np.asarray(trace["face_weights"][face_id], dtype=float) for face_id in (1, 2, 3)]
    )


def _build_sbp_differentiation_matrices(
    V: np.ndarray,
    Vr: np.ndarray,
    Vs: np.ndarray,
    weights_2d: np.ndarray,
    weights_1d: np.ndarray,
    trace: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chan-style corrected SBP derivative operators on embedded Table-1 nodes.

    The construction follows the notebook logic but replaces fmask with the
    repository trace policy, so the point and face ordering stay repository-native.
    """
    weights_2d = np.asarray(weights_2d, dtype=float).reshape(-1)
    weights_1d = np.asarray(weights_1d, dtype=float).reshape(-1)
    Np = weights_2d.size

    W = np.diag(weights_2d)
    M_modal = V.T @ W @ V
    P = np.linalg.solve(M_modal, V.T @ W)

    Dr_sbp = Vr @ P
    Ds_sbp = Vs @ P

    left_term = (np.eye(Np) + V @ P).T
    right_term = np.eye(Np) - V @ P

    sum_r = np.zeros((Np, Np), dtype=float)
    sum_s = np.zeros((Np, Np), dtype=float)

    # Repository local face convention:
    # face 1: r + s = 0  -> outward reference normal ( 1,  1)
    # face 2: r     = -1 -> outward reference normal (-1,  0)
    # face 3: s     = -1 -> outward reference normal ( 0, -1)
    face_normals = {
        1: (1.0, 1.0),
        2: (-1.0, 0.0),
        3: (0.0, -1.0),
    }

    for face_id in (1, 2, 3):
        ids = np.asarray(trace["face_node_ids"][face_id], dtype=int)
        wf = np.asarray(trace["face_weights"][face_id], dtype=float)
        E_face = np.zeros((Np, Np), dtype=float)
        E_face[ids, ids] = wf

        nr, ns = face_normals[face_id]
        correction = left_term @ E_face @ right_term
        sum_r += nr * correction
        sum_s += ns * correction

    invW = np.diag(1.0 / weights_2d)
    Dr_sbp = Dr_sbp + 0.5 * invW @ sum_r
    Ds_sbp = Ds_sbp + 0.5 * invW @ sum_s

    return Dr_sbp, Ds_sbp, P


def _build_hybrid_q_operators(
    Dr_sbp: np.ndarray,
    Ds_sbp: np.ndarray,
    weights_2d: np.ndarray,
    face_weights_flat: np.ndarray,
    E: np.ndarray,
    nr_flat: np.ndarray,
    ns_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    W = np.diag(np.asarray(weights_2d, dtype=float))
    Q_r = W @ Dr_sbp
    Q_s = W @ Ds_sbp

    S_r = 0.5 * (Q_r - Q_r.T)
    S_s = 0.5 * (Q_s - Q_s.T)

    B_r = np.diag(face_weights_flat * nr_flat)
    B_s = np.diag(face_weights_flat * ns_flat)

    E_T_Br = E.T @ B_r
    E_T_Bs = E.T @ B_s

    Q_tilde_r = np.block(
        [[S_r, 0.5 * E_T_Br], [-0.5 * E_T_Br.T, 0.5 * B_r]]
    )
    Q_tilde_s = np.block(
        [[S_s, 0.5 * E_T_Bs], [-0.5 * E_T_Bs.T, 0.5 * B_s]]
    )
    return Q_tilde_r, Q_tilde_s


def build_manifold_sbp_chan_table1_k4_reference_operators() -> ManifoldSBPChanReferenceOperators:
    rule = load_table1_rule(4)

    # Use repository canonical Table-1 node order.
    rs_nodes = np.asarray(rule["rs"], dtype=float)

    weights_2d = np.asarray(rule["ws"], dtype=float).reshape(-1)
    weights_1d = np.asarray(rule["we"], dtype=float).reshape(-1)

    trace = build_trace_policy(rule, N=4)
    Np = rs_nodes.shape[0]
    nfp = int(trace["nfp"])

    V = vandermonde2d(4, rs_nodes[:, 0], rs_nodes[:, 1])
    Vr, Vs = grad_vandermonde2d(4, rs_nodes[:, 0], rs_nodes[:, 1])

    Dr_sbp, Ds_sbp, P = _build_sbp_differentiation_matrices(
        V=V,
        Vr=Vr,
        Vs=Vs,
        weights_2d=weights_2d,
        weights_1d=weights_1d,
        trace=trace,
    )

    E = _face_extraction_matrix(trace, Np=Np)
    face_weights_flat = _flat_face_weights(trace)

    nr_by_face = np.array([1.0, -1.0, 0.0], dtype=float)
    ns_by_face = np.array([1.0, 0.0, -1.0], dtype=float)
    nr_flat = np.repeat(nr_by_face, nfp)
    ns_flat = np.repeat(ns_by_face, nfp)

    Q_tilde_r, Q_tilde_s = _build_hybrid_q_operators(
        Dr_sbp=Dr_sbp,
        Ds_sbp=Ds_sbp,
        weights_2d=weights_2d,
        face_weights_flat=face_weights_flat,
        E=E,
        nr_flat=nr_flat,
        ns_flat=ns_flat,
    )

    return ManifoldSBPChanReferenceOperators(
        rule=rule,
        trace=trace,
        rs_nodes=rs_nodes,
        weights_2d=weights_2d,
        weights_1d=weights_1d,
        V=V,
        Vr=Vr,
        Vs=Vs,
        P=P,
        Dr_sbp=Dr_sbp,
        Ds_sbp=Ds_sbp,
        E=E,
        Q_tilde_r=Q_tilde_r,
        Q_tilde_s=Q_tilde_s,
        face_weights_flat=face_weights_flat,
        nr_flat=nr_flat,
        ns_flat=ns_flat,
        nfp=nfp,
    )


def manifold_sbp_contravariant_flux(
    geom: ManifoldGeometryCache,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if not (U.shape == V.shape == W.shape == geom.X.shape):
        raise ValueError("U, V, W must match geometry nodal shape.")

    u_tilde = geom.a1x * U + geom.a1y * V + geom.a1z * W
    v_tilde = geom.a2x * U + geom.a2y * V + geom.a2z * W
    return geom.J * u_tilde, geom.J * v_tilde


def _two_point_flux_linear_numpy(Q_tilde: np.ndarray, metric_flux: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Algebraically optimized version of

        sum_j 0.5 * Q_ij * (a_i + a_j) * (q_i + q_j).

    All arrays use repository shape (K, N).
    """
    Q_tilde = np.asarray(Q_tilde, dtype=float)
    metric_flux = np.asarray(metric_flux, dtype=float)
    q = np.asarray(q, dtype=float)

    row_sum = np.sum(Q_tilde, axis=1)
    aq = metric_flux * q
    return 0.5 * (
        aq * row_sum[None, :]
        + metric_flux * (q @ Q_tilde.T)
        + q * (metric_flux @ Q_tilde.T)
        + aq @ Q_tilde.T
    )


@njit(cache=True, parallel=True)
def _two_point_flux_linear_numba(
    Q_tilde: np.ndarray,
    metric_flux: np.ndarray,
    q: np.ndarray,
    out: np.ndarray,
) -> None:
    K, N = q.shape
    for k in prange(K):
        for i in range(N):
            val = 0.0
            ai = metric_flux[k, i]
            qi = q[k, i]
            for j in range(N):
                val += 0.5 * Q_tilde[i, j] * (ai + metric_flux[k, j]) * (qi + q[k, j])
            out[k, i] = val


def _two_point_flux_linear(
    Q_tilde: np.ndarray,
    metric_flux: np.ndarray,
    q: np.ndarray,
    use_numba: bool | None,
) -> np.ndarray:
    if _should_use_numba(use_numba):
        out = np.zeros_like(q)
        _two_point_flux_linear_numba(
            np.ascontiguousarray(Q_tilde),
            np.ascontiguousarray(metric_flux),
            np.ascontiguousarray(q),
            out,
        )
        return out
    return _two_point_flux_linear_numpy(Q_tilde, metric_flux, q)


def _as_face_flat(face_values: np.ndarray) -> np.ndarray:
    face_values = np.asarray(face_values, dtype=float)
    if face_values.ndim != 3:
        raise ValueError("face_values must have shape (K, 3, nfp).")
    return face_values.reshape(face_values.shape[0], face_values.shape[1] * face_values.shape[2])


def manifold_sbp_chan2019_rhs_exchange(
    q: np.ndarray,
    geom: ManifoldGeometryCache,
    velocity_xyz,
    ref_ops: ManifoldSBPChanReferenceOperators | None = None,
    conn: dict | None = None,
    flux_type: str = "upwind",
    alpha_lf: float = 1.0,
    global_vmax: float | None = None,
    t: float = 0.0,
    use_numba: bool | None = True,
) -> np.ndarray:
    """
    Chan-style hybridized SBP manifold DG RHS.

    The semi-discrete form is

        q_t = - M_J^{-1} [ D_SBP(J a^1 繚 u, q) + D_SBP(J a^2 繚 u, q)
                          + SAT(f* - f_M) ]

    with Table-1 embedded surface nodes and repository-native face pairing.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != geom.X.shape:
        raise ValueError("q must match geometry nodal shape (K, Np).")
    if alpha_lf <= 0.0:
        raise ValueError("alpha_lf must be positive.")

    if ref_ops is None:
        ref_ops = build_manifold_sbp_chan_table1_k4_reference_operators()
    if conn is None:
        conn = build_manifold_face_connectivity(geom.EToV)

    flux_type = _normalize_flux_type(flux_type)

    if callable(velocity_xyz):
        try:
            U, V, W = velocity_xyz(geom.X, geom.Y, geom.Z, t=t)
        except TypeError:
            U, V, W = velocity_xyz(geom.X, geom.Y, geom.Z)
    else:
        U, V, W = velocity_xyz
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    Ju, Jv = manifold_sbp_contravariant_flux(geom, U, V, W)

    paired = pair_face_traces(q, conn=conn, trace=ref_ops.trace, use_numba=use_numba)
    q_M_face = np.asarray(paired["uM"], dtype=float)
    q_P_face = np.asarray(paired["uP"], dtype=float)

    q_face = _as_face_flat(q_M_face)
    qP_face = _as_face_flat(q_P_face)

    Ju_face = _as_face_flat(evaluate_all_face_values(Ju, ref_ops.trace, use_numba=use_numba))
    Jv_face = _as_face_flat(evaluate_all_face_values(Jv, ref_ops.trace, use_numba=use_numba))
    J_face = _as_face_flat(evaluate_all_face_values(geom.J, ref_ops.trace, use_numba=use_numba))

    q_ext = np.concatenate([q, q_face], axis=1)
    Ju_ext = np.concatenate([Ju, Ju_face], axis=1)
    Jv_ext = np.concatenate([Jv, Jv_face], axis=1)

    Np = q.shape[1]

    vol_r_ext = _two_point_flux_linear(ref_ops.Q_tilde_r, Ju_ext, q_ext, use_numba=use_numba)
    vol_s_ext = _two_point_flux_linear(ref_ops.Q_tilde_s, Jv_ext, q_ext, use_numba=use_numba)

    vol_term = (
        vol_r_ext[:, :Np]
        + vol_r_ext[:, Np:] @ ref_ops.E
        + vol_s_ext[:, :Np]
        + vol_s_ext[:, Np:] @ ref_ops.E
    )

    vn_M = ref_ops.nr_flat[None, :] * Ju_face + ref_ops.ns_flat[None, :] * Jv_face
    if flux_type == "upwind":
        c_val = alpha_lf * np.abs(vn_M)
    elif flux_type == "lax_friedrichs":
        if global_vmax is None:
            global_vmax = float(np.max(np.sqrt(U * U + V * V + W * W)))
        c_val = alpha_lf * float(global_vmax) * J_face
    elif flux_type == "central":
        c_val = 0.0
    else:  # pragma: no cover
        raise ValueError("Unsupported flux_type.")

    f_star = vn_M * 0.5 * (q_face + qP_face) + 0.5 * c_val * (q_face - qP_face)
    penalty = (f_star - vn_M * q_face) * ref_ops.face_weights_flat[None, :]
    surface_integral = penalty @ ref_ops.E

    return -(vol_term + surface_integral) / (ref_ops.weights_2d[None, :] * geom.J)


def constant_state_rhs_diagnostic(
    geom: ManifoldGeometryCache,
    velocity_xyz,
    ref_ops: ManifoldSBPChanReferenceOperators | None = None,
    conn: dict | None = None,
    flux_type: str = "upwind",
    use_numba: bool | None = True,
) -> dict[str, float | np.ndarray]:
    if ref_ops is None:
        ref_ops = build_manifold_sbp_chan_table1_k4_reference_operators()
    q = np.ones_like(geom.X)
    rhs = manifold_sbp_chan2019_rhs_exchange(
        q=q,
        geom=geom,
        velocity_xyz=velocity_xyz,
        ref_ops=ref_ops,
        conn=conn,
        flux_type=flux_type,
        use_numba=use_numba,
    )
    return {
        "rhs": rhs,
        "max_rhs_abs": float(np.max(np.abs(rhs))),
        "weighted_rhs_mass": float(np.sum(rhs * geom.J * ref_ops.weights_2d[None, :])),
    }

