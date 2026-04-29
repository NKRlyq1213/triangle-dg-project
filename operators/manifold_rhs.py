from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.table1_rules import load_table1_rule
from geometry.connectivity import build_face_connectivity
from geometry.sphere_manifold_metrics import ManifoldGeometryCache
from operators.exchange import evaluate_all_face_values, pair_face_traces
from operators.sdg_flattened_divergence import build_table1_reference_diff_operators
from operators.trace_policy import build_trace_policy
from operators.vandermonde2d import vandermonde2d

try:
    import importlib
    _numba = importlib.import_module("numba")
    njit = _numba.njit
    prange = getattr(_numba, 'prange', range)
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = lambda cache: lambda f: f
    prange = range
    _NUMBA_AVAILABLE = False

def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


_VALID_FLUX_TYPES = {"upwind", "central", "lax_friedrichs"}


def _normalize_flux_type(flux_type: str) -> str:
    flux = str(flux_type).strip().lower()
    if flux not in _VALID_FLUX_TYPES:
        raise ValueError("flux_type must be 'upwind', 'central', or 'lax_friedrichs'.")
    return flux


def _apply_reference_operator(D: np.ndarray, u: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    u = np.asarray(u, dtype=float)

    if u.ndim != 2 or u.shape[1] != D.shape[0]:
        raise ValueError("u must have shape (K, Np), with Np matching D.")
    return u @ D.T


@dataclass(frozen=True)
class ManifoldReferenceOperators:
    rule: dict
    trace: dict
    rs_nodes: np.ndarray
    weights_2d: np.ndarray
    Dr: np.ndarray
    Ds: np.ndarray
    lift: np.ndarray
    face_extraction: np.ndarray
    face_node_ids: dict[int, np.ndarray]
    face_weights: np.ndarray
    nr: np.ndarray
    ns: np.ndarray


def build_manifold_table1_k4_reference_operators() -> ManifoldReferenceOperators:
    """
    Build the fixed Table1 k=4 reference operators used by the manifold study.
    """
    rule = load_table1_rule(4)
    bary_raw = np.asarray(rule["bary"], dtype=float)
    rs_nodes = np.column_stack(
        [
            2.0 * bary_raw[:, 2] - 1.0,
            2.0 * bary_raw[:, 1] - 1.0,
        ]
    )
    rule = dict(rule)
    rule["rs"] = rs_nodes
    weights = np.asarray(rule["ws"], dtype=float).reshape(-1)

    Dr, Ds = build_table1_reference_diff_operators(rule, N=4)
    trace = build_trace_policy(rule, N=4)

    V = vandermonde2d(4, rs_nodes[:, 0], rs_nodes[:, 1])
    # Table1 weights in this project sum to one, matching the notebook's
    # M_modal = V.T @ diag(weights) @ V convention for the projected lift.
    M_modal = V.T @ (weights[:, None] * V)
    lift = V @ np.linalg.inv(M_modal) @ V.T

    nfp = int(trace["nfp"])
    Np = rs_nodes.shape[0]
    face_extraction = np.zeros((3 * nfp, Np), dtype=float)
    face_weights = np.zeros((3 * nfp,), dtype=float)

    for face_id in (1, 2, 3):
        rows = slice((face_id - 1) * nfp, face_id * nfp)
        row_ids = np.arange((face_id - 1) * nfp, face_id * nfp, dtype=int)
        ids = np.asarray(trace["face_node_ids"][face_id], dtype=int)
        face_extraction[row_ids, ids] = 1.0
        face_weights[rows] = np.asarray(trace["face_weights"][face_id], dtype=float)

    # Local face convention from geometry.connectivity / trace_policy:
    # face 1: r + s = 0  -> outward reference normal ( 1,  1)
    # face 2: r     = -1 -> outward reference normal (-1,  0)
    # face 3: s     = -1 -> outward reference normal ( 0, -1)
    nr_face = np.array([1.0, -1.0, 0.0], dtype=float)
    ns_face = np.array([1.0, 0.0, -1.0], dtype=float)

    return ManifoldReferenceOperators(
        rule=rule,
        trace=trace,
        rs_nodes=rs_nodes,
        weights_2d=weights,
        Dr=Dr,
        Ds=Ds,
        lift=lift,
        face_extraction=face_extraction,
        face_node_ids={int(k): np.asarray(v, dtype=int) for k, v in trace["face_node_ids"].items()},
        face_weights=face_weights,
        nr=np.repeat(nr_face, nfp),
        ns=np.repeat(ns_face, nfp),
    )


@dataclass(frozen=True)
class ManifoldExchangeCache:
    conn: dict
    trace: dict
    face_weights: np.ndarray
    nr: np.ndarray
    ns: np.ndarray
    J_face: np.ndarray | None = None
    u_tilde_face: np.ndarray | None = None
    v_tilde_face: np.ndarray | None = None


def build_manifold_face_connectivity(EToV: np.ndarray) -> dict:
    """
    Build closed-surface face connectivity using existing topological exchange.
    """
    EToV = np.asarray(EToV, dtype=int)
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    vertex_ids = np.arange(int(np.max(EToV)) + 1, dtype=float)
    conn = build_face_connectivity(
        VX=vertex_ids,
        VY=np.zeros_like(vertex_ids),
        EToV=EToV,
        classify_boundary=None,
    )
    if np.any(np.asarray(conn["is_boundary"], dtype=bool)):
        raise ValueError("Manifold sphere mesh must be closed; boundary faces were found.")
    return conn


def _face_matrix_from_trace_values(trace_values: dict[int, np.ndarray], K: int) -> np.ndarray:
    nfp = len(np.asarray(trace_values[1]).reshape(-1))
    out = np.zeros((K, 3, nfp), dtype=float)
    for face_id in (1, 2, 3):
        out[:, face_id - 1, :] = np.asarray(trace_values[face_id], dtype=float).reshape(1, nfp)
    return out


def _evaluate_face_values(u: np.ndarray, trace: dict, use_numba: bool | None = None) -> np.ndarray:
    return evaluate_all_face_values(np.asarray(u, dtype=float), trace, use_numba=use_numba)


def build_manifold_exchange_cache(
    EToV: np.ndarray,
    ref_ops: ManifoldReferenceOperators | None = None,
    geom: ManifoldGeometryCache | None = None,
    U: np.ndarray | None = None,
    V: np.ndarray | None = None,
    W: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> ManifoldExchangeCache:
    """
    Cache the closed-sphere connectivity and embedded Table1 face data.

    If geometry and velocity are supplied, face metric/contravariant velocity
    values are precomputed for repeated time-stepping RHS calls.
    """
    if ref_ops is None:
        ref_ops = build_manifold_table1_k4_reference_operators()

    conn = build_manifold_face_connectivity(EToV)
    trace = ref_ops.trace
    K = np.asarray(EToV, dtype=int).shape[0]

    face_weights = _face_matrix_from_trace_values(trace["face_weights"], K)
    nr_face = np.array([1.0, -1.0, 0.0], dtype=float)
    ns_face = np.array([1.0, 0.0, -1.0], dtype=float)
    nfp = int(trace["nfp"])
    nr = np.repeat(nr_face.reshape(1, 3, 1), K, axis=0)
    nr = np.repeat(nr, nfp, axis=2)
    ns = np.repeat(ns_face.reshape(1, 3, 1), K, axis=0)
    ns = np.repeat(ns, nfp, axis=2)

    J_face = None
    u_tilde_face = None
    v_tilde_face = None
    if geom is not None and U is not None and V is not None and W is not None:
        u_tilde, v_tilde = manifold_contravariant_velocity(geom, U, V, W)
        J_face = _evaluate_face_values(geom.J, trace, use_numba=use_numba)
        u_tilde_face = _evaluate_face_values(u_tilde, trace, use_numba=use_numba)
        v_tilde_face = _evaluate_face_values(v_tilde, trace, use_numba=use_numba)

    return ManifoldExchangeCache(
        conn=conn,
        trace=trace,
        face_weights=face_weights,
        nr=nr,
        ns=ns,
        J_face=J_face,
        u_tilde_face=u_tilde_face,
        v_tilde_face=v_tilde_face,
    )


def build_manifold_vmaps(
    EToV: np.ndarray,
    face_node_ids: dict[int, np.ndarray],
    Np: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build interior trace maps for a closed conforming triangular surface mesh.
    """
    EToV = np.asarray(EToV, dtype=int)
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    conn = build_manifold_face_connectivity(EToV)
    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)

    K = EToV.shape[0]
    if Np is None:
        Np = max(int(np.max(ids)) for ids in face_node_ids.values()) + 1
    Np = int(Np)
    if Np <= max(int(np.max(ids)) for ids in face_node_ids.values()):
        raise ValueError("Np must be larger than all face node indices.")
    nfp = len(face_node_ids[1])
    vmapM = np.zeros((3 * nfp, K), dtype=int)
    vmapP = np.zeros((3 * nfp, K), dtype=int)

    EToE = np.asarray(conn["EToE"], dtype=int)
    EToF = np.asarray(conn["EToF"], dtype=int)
    face_flip = np.asarray(conn["face_flip"], dtype=bool)

    for k in range(K):
        for face_id in (1, 2, 3):
            rows = slice((face_id - 1) * nfp, face_id * nfp)
            local_ids = np.asarray(face_node_ids[face_id], dtype=int)
            vmapM[rows, k] = k * Np + local_ids

            nbr = int(EToE[k, face_id - 1])
            nbr_face = int(EToF[k, face_id - 1])
            nbr_ids = np.asarray(face_node_ids[nbr_face], dtype=int)
            if bool(face_flip[k, face_id - 1]):
                nbr_ids = nbr_ids[::-1]
            vmapP[rows, k] = nbr * Np + nbr_ids

    return vmapM, vmapP, is_boundary


def pair_manifold_face_traces(
    q: np.ndarray,
    conn: dict,
    trace: dict,
    boundary_fill_value: float = np.nan,
    use_numba: bool | None = None,
) -> dict:
    """
    Pair Table1 embedded face traces on a closed manifold mesh.
    """
    return pair_face_traces(
        q,
        conn=conn,
        trace=trace,
        boundary_fill_value=boundary_fill_value,
        use_numba=use_numba,
    )


def manifold_contravariant_velocity(
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
    return u_tilde, v_tilde


@njit(cache=True, parallel=True)
def _manifold_volume_divergence_kernel(
    q: np.ndarray,
    u_tilde: np.ndarray,
    v_tilde: np.ndarray,
    J: np.ndarray,
    Dr_T: np.ndarray,
    Ds_T: np.ndarray,
    out: np.ndarray,
) -> None:
    K, Np = q.shape
    for k in prange(K):
        for i in range(Np):
            d1 = 0.0
            d2 = 0.0
            d3 = 0.0

            # evaluate dot products
            # d1: D @ (J u q) + D @ (J v q)
            # d2: u D @ q + v D @ q
            # d3: (q/J) * (D @ (J u) + D @ (J v))
            
            for j in range(Np):
                _Dr = Dr_T[j, i]
                _Ds = Ds_T[j, i]
                
                qu_j = q[k, j] * u_tilde[k, j]
                qv_j = q[k, j] * v_tilde[k, j]
                J_j = J[k, j]
                
                J_u_q = J_j * qu_j
                J_v_q = J_j * qv_j
                
                d1 += _Dr * J_u_q + _Ds * J_v_q
                d2 += _Dr * q[k, j] * u_tilde[k, i] + _Ds * q[k, j] * v_tilde[k, i]
                d3 += _Dr * J_j * u_tilde[k, j] + _Ds * J_j * v_tilde[k, j]

            out[k, i] = 0.5 * (d1 / J[k, i] + d2 + (q[k, i] / J[k, i]) * d3)


def manifold_volume_divergence(
    q: np.ndarray,
    geom: ManifoldGeometryCache,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    use_numba: bool | None = None,
) -> np.ndarray:
    """
    Split-form manifold divergence contribution without the leading RHS sign.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != geom.X.shape:
        raise ValueError("q must match geometry nodal shape.")

    u_tilde, v_tilde = manifold_contravariant_velocity(geom, U, V, W)
    J = geom.J

    if _should_use_numba(use_numba):
        out = np.zeros_like(q)
        # Numba fails if passing Fortran or non-contiguous arrays, occasionally 
        Dr_T = np.ascontiguousarray(Dr.T)
        Ds_T = np.ascontiguousarray(Ds.T)
        _manifold_volume_divergence_kernel(
            q, u_tilde, v_tilde, J, Dr_T, Ds_T, out
        )
        return out

    J_u_q = J * u_tilde * q
    J_v_q = J * v_tilde * q
    term1 = (_apply_reference_operator(Dr, J_u_q) + _apply_reference_operator(Ds, J_v_q)) / J

    term2 = (
        u_tilde * _apply_reference_operator(Dr, q)
        + v_tilde * _apply_reference_operator(Ds, q)
    )

    J_u = J * u_tilde
    J_v = J * v_tilde
    term3 = (q / J) * (
        _apply_reference_operator(Dr, J_u)
        + _apply_reference_operator(Ds, J_v)
    )

    return 0.5 * (term1 + term2 + term3)


def manifold_surface_term(
    q: np.ndarray,
    geom: ManifoldGeometryCache,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    ref_ops: ManifoldReferenceOperators,
    vmapM: np.ndarray,
    vmapP: np.ndarray,
    flux_type: str = "upwind",
    alpha_lf: float = 1.0,
) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != geom.X.shape:
        raise ValueError("q must match geometry nodal shape.")

    flux_type = _normalize_flux_type(flux_type)
    u_tilde, v_tilde = manifold_contravariant_velocity(geom, U, V, W)

    q_flat = q.reshape(-1)
    q_M = q_flat[vmapM]
    q_P = q_flat[vmapP]

    J_M = geom.J.reshape(-1)[vmapM]
    u_M = u_tilde.reshape(-1)[vmapM]
    v_M = v_tilde.reshape(-1)[vmapM]

    vn_sJ = J_M * (ref_ops.nr[:, None] * u_M + ref_ops.ns[:, None] * v_M)
    if flux_type == "upwind":
        c_val = alpha_lf * np.abs(vn_sJ)
    elif flux_type == "lax_friedrichs":
        c_val = alpha_lf * float(np.max(np.abs(vn_sJ)))
    elif flux_type == "central":
        c_val = 0.0
    else:
        raise ValueError("flux_type must be 'upwind', 'central', or 'lax_friedrichs'.")

    penalty = 0.5 * (vn_sJ - c_val) * (q_M - q_P)
    scaled_penalty = penalty * ref_ops.face_weights[:, None]
    surface_integral = ref_ops.face_extraction.T @ scaled_penalty
    return (ref_ops.lift @ surface_integral).T / geom.J


def _resolve_velocity_xyz(
    velocity_xyz,
    geom: ManifoldGeometryCache,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if callable(velocity_xyz):
        try:
            out = velocity_xyz(geom.X, geom.Y, geom.Z, t=t)
        except TypeError:
            out = velocity_xyz(geom.X, geom.Y, geom.Z)
    else:
        out = velocity_xyz

    if not isinstance(out, tuple) or len(out) != 3:
        raise ValueError("velocity_xyz must be a callable or a tuple (U, V, W).")
    U, V, W = (np.asarray(part, dtype=float) for part in out)
    if not (U.shape == V.shape == W.shape == geom.X.shape):
        raise ValueError("velocity components must match geometry nodal shape.")
    return U, V, W


@njit(cache=True, parallel=True)
def _manifold_surface_term_kernel(
    q_M: np.ndarray,
    q_P: np.ndarray,
    J_face: np.ndarray,
    u_face: np.ndarray,
    v_face: np.ndarray,
    nr: np.ndarray,
    ns: np.ndarray,
    face_weights: np.ndarray,
    alpha_lf: float,
    flux_type_upwind: bool,
    flux_type_lf: bool,
    lf_speed: float,
    lift_edge: np.ndarray,
    J: np.ndarray,
    out: np.ndarray
) -> None:
    K, _3, nfp = q_M.shape
    Np = lift_edge.shape[0]

    for k in prange(K):
        for i in range(Np):
            val = 0.0
            
            for jf in range(3):
                for node in range(nfp):
                    _J_face = J_face[k, jf, node]
                    _nr = nr[jf, node]
                    _ns = ns[jf, node]
                    _u = u_face[k, jf, node]
                    _v = v_face[k, jf, node]
                    
                    vn_sJ = _J_face * (_nr * _u + _ns * _v)
                    c_val = 0.0
                    if flux_type_upwind:
                        c_val = alpha_lf * abs(vn_sJ)
                    elif flux_type_lf:
                        c_val = alpha_lf * lf_speed
                        
                    penalty = 0.5 * (vn_sJ - c_val) * (q_M[k, jf, node] - q_P[k, jf, node])
                    scaled = penalty * face_weights[jf, node]
                    
                    # scaled_penalty index maps to jf * nfp + node
                    face_idx = jf * nfp + node
                    val += lift_edge[i, face_idx] * scaled
                    
            out[k, i] = val / J[k, i]


def manifold_surface_term_from_exchange(
    q: np.ndarray,
    geom: ManifoldGeometryCache,
    velocity_xyz,
    ref_ops: ManifoldReferenceOperators,
    exchange_cache: ManifoldExchangeCache,
    flux_type: str = "upwind",
    alpha_lf: float = 1.0,
    t: float = 0.0,
    use_numba: bool | None = None,
) -> np.ndarray:
    """
    Surface penalty using operators.exchange.pair_face_traces for face exchange.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != geom.X.shape:
        raise ValueError("q must match geometry nodal shape.")

    flux_type = _normalize_flux_type(flux_type)
    paired = pair_manifold_face_traces(
        q,
        conn=exchange_cache.conn,
        trace=exchange_cache.trace,
        use_numba=use_numba,
    )
    q_M = np.asarray(paired["uM"], dtype=float)
    q_P = np.asarray(paired["uP"], dtype=float)

    if (
        exchange_cache.J_face is not None
        and exchange_cache.u_tilde_face is not None
        and exchange_cache.v_tilde_face is not None
    ):
        J_face = exchange_cache.J_face
        u_face = exchange_cache.u_tilde_face
        v_face = exchange_cache.v_tilde_face
    else:
        U, V, W = _resolve_velocity_xyz(velocity_xyz, geom, t=t)
        u_tilde, v_tilde = manifold_contravariant_velocity(geom, U, V, W)
        J_face = _evaluate_face_values(geom.J, exchange_cache.trace, use_numba=use_numba)
        u_face = _evaluate_face_values(u_tilde, exchange_cache.trace, use_numba=use_numba)
        v_face = _evaluate_face_values(v_tilde, exchange_cache.trace, use_numba=use_numba)

    vn_sJ = J_face * (exchange_cache.nr * u_face + exchange_cache.ns * v_face)
    lf_speed = float(np.max(np.abs(vn_sJ)))

    if _should_use_numba(use_numba):
        out = np.zeros_like(q)
        lift_edge = np.ascontiguousarray(ref_ops.lift @ ref_ops.face_extraction.T)

        nr = exchange_cache.nr
        ns = exchange_cache.ns
        fw = exchange_cache.face_weights

        if nr.ndim == 3:
            nr = nr[0]
            ns = ns[0]
        if fw.ndim == 3:
            fw = fw[0]

        flux_is_upwind = flux_type == "upwind"
        flux_is_lf = flux_type == "lax_friedrichs"

        _manifold_surface_term_kernel(
            q_M, q_P, J_face, u_face, v_face,
            nr, ns, fw,
            alpha_lf,
            flux_is_upwind,
            flux_is_lf,
            lf_speed,
            lift_edge,
            geom.J,
            out
        )
        return out

    if flux_type == "upwind":
        c_val = alpha_lf * np.abs(vn_sJ)
    elif flux_type == "lax_friedrichs":
        c_val = alpha_lf * lf_speed
    elif flux_type == "central":
        c_val = 0.0
    else:
        raise ValueError("flux_type must be 'upwind', 'central', or 'lax_friedrichs'.")

    penalty = 0.5 * (vn_sJ - c_val) * (q_M - q_P)
    scaled = penalty * exchange_cache.face_weights
    scaled_penalty = scaled.transpose(1, 2, 0).reshape(3 * int(exchange_cache.trace["nfp"]), q.shape[0])
    surface_integral = ref_ops.face_extraction.T @ scaled_penalty
    return (ref_ops.lift @ surface_integral).T / geom.J


def manifold_rhs(
    q: np.ndarray,
    geom: ManifoldGeometryCache,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    ref_ops: ManifoldReferenceOperators,
    vmapM: np.ndarray,
    vmapP: np.ndarray,
    flux_type: str = "upwind",
    alpha_lf: float = 1.0,
) -> np.ndarray:
    div = manifold_volume_divergence(
        q=q,
        geom=geom,
        U=U,
        V=V,
        W=W,
        Dr=ref_ops.Dr,
        Ds=ref_ops.Ds,
    )
    surface = manifold_surface_term(
        q=q,
        geom=geom,
        U=U,
        V=V,
        W=W,
        ref_ops=ref_ops,
        vmapM=vmapM,
        vmapP=vmapP,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
    )
    return -div + surface


def manifold_rhs_exchange(
    q: np.ndarray,
    geom: ManifoldGeometryCache,
    velocity_xyz,
    exchange_cache: ManifoldExchangeCache,
    ref_ops: ManifoldReferenceOperators | None = None,
    flux_type: str = "upwind",
    alpha_lf: float = 1.0,
    t: float = 0.0,
    use_numba: bool | None = None,
) -> np.ndarray:
    """
    Main manifold RHS path for time stepping, using pair_face_traces exchange.
    """
    if ref_ops is None:
        ref_ops = build_manifold_table1_k4_reference_operators()

    U, V, W = _resolve_velocity_xyz(velocity_xyz, geom, t=t)
    div = manifold_volume_divergence(
        q=q,
        geom=geom,
        U=U,
        V=V,
        W=W,
        Dr=ref_ops.Dr,
        Ds=ref_ops.Ds,
    )
    surface = manifold_surface_term_from_exchange(
        q=q,
        geom=geom,
        velocity_xyz=(U, V, W),
        ref_ops=ref_ops,
        exchange_cache=exchange_cache,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        t=t,
        use_numba=use_numba,
    )
    return -div + surface


def manifold_rhs_constant_field(
    geom: ManifoldGeometryCache,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    ref_ops: ManifoldReferenceOperators | None = None,
) -> dict[str, np.ndarray | float]:
    """
    Evaluate the constant-state free-stream divergence diagnostic.
    """
    if ref_ops is None:
        ref_ops = build_manifold_table1_k4_reference_operators()

    q = np.ones_like(geom.X)
    vmapM, vmapP, _ = build_manifold_vmaps(
        geom.EToV,
        ref_ops.face_node_ids,
        Np=geom.X.shape[1],
    )

    volume_div = manifold_volume_divergence(q, geom, U, V, W, ref_ops.Dr, ref_ops.Ds)
    surface = manifold_surface_term(q, geom, U, V, W, ref_ops, vmapM, vmapP)
    rhs = -volume_div + surface

    return {
        "q": q,
        "divergence": volume_div,
        "surface_term": surface,
        "rhs": rhs,
        "max_surface_abs": float(np.max(np.abs(surface))),
        "max_rhs_abs": float(np.max(np.abs(rhs))),
    }


def constant_field_divergence_error(*args, **kwargs) -> dict[str, np.ndarray | float]:
    """
    Backward-readable alias for the public constant-field diagnostic.
    """
    return manifold_rhs_constant_field(*args, **kwargs)
