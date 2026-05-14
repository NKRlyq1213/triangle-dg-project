# Generate NumPy/Numba SBP Manifold DG files from the two notebooks.
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File .\tools\generate_sbp_chan2019_files.ps1
#
# This script only creates new files. It does not edit existing repository files.

$ErrorActionPreference = "Stop"

function Write-RepoFile {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Content
    )
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    Set-Content -Path $Path -Value $Content -Encoding UTF8
    Write-Host "[OK] wrote $Path"
}

Write-RepoFile -Path "geometry/sphere_projected_exact_metrics.py" -Content @'
from __future__ import annotations

import numpy as np

from geometry.sphere_manifold_metrics import (
    ManifoldGeometryCache,
    map_reference_nodes_to_sphere,
)


def build_exact_projected_sphere_geometry_cache(
    nodes_xyz: np.ndarray,
    EToV: np.ndarray,
    rs_nodes: np.ndarray,
    R: float = 1.0,
    tol: float = 1.0e-14,
) -> ManifoldGeometryCache:
    """
    Build analytical metrics for the radial projection of planar triangles to a sphere.

    This is the NumPy version of the exact metric construction used in the SBP
    manifold DG notebooks.  The returned arrays follow the repository convention:

        shape = (K, Np)

    where K is the number of elements and Np is the number of Table-1 volume nodes.
    """
    nodes_xyz = np.asarray(nodes_xyz, dtype=float)
    EToV = np.asarray(EToV, dtype=int)
    rs_nodes = np.asarray(rs_nodes, dtype=float)

    if nodes_xyz.ndim != 2 or nodes_xyz.shape[1] != 3:
        raise ValueError("nodes_xyz must have shape (Nv, 3).")
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")
    if rs_nodes.ndim != 2 or rs_nodes.shape[1] != 2:
        raise ValueError("rs_nodes must have shape (Np, 2).")
    if R <= 0.0:
        raise ValueError("R must be positive.")

    X, Y, Z = map_reference_nodes_to_sphere(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=rs_nodes,
        R=R,
    )

    r = rs_nodes[:, 0]
    s = rs_nodes[:, 1]
    L1 = -(r + s) / 2.0
    L2 = (1.0 + r) / 2.0
    L3 = (1.0 + s) / 2.0

    K = EToV.shape[0]
    Np = rs_nodes.shape[0]

    J = np.zeros((K, Np), dtype=float)
    a1x = np.zeros((K, Np), dtype=float)
    a1y = np.zeros((K, Np), dtype=float)
    a1z = np.zeros((K, Np), dtype=float)
    a2x = np.zeros((K, Np), dtype=float)
    a2y = np.zeros((K, Np), dtype=float)
    a2z = np.zeros((K, Np), dtype=float)

    for k in range(K):
        v1, v2, v3 = nodes_xyz[EToV[k]]

        x_flat = L1 * v1[0] + L2 * v2[0] + L3 * v3[0]
        y_flat = L1 * v1[1] + L2 * v2[1] + L3 * v3[1]
        z_flat = L1 * v1[2] + L2 * v2[2] + L3 * v3[2]

        norm_x = np.sqrt(x_flat * x_flat + y_flat * y_flat + z_flat * z_flat)
        if np.any(norm_x <= tol):
            raise ValueError("Reference node interpolation hit the zero vector.")

        dr_x = -0.5 * v1[0] + 0.5 * v2[0]
        dr_y = -0.5 * v1[1] + 0.5 * v2[1]
        dr_z = -0.5 * v1[2] + 0.5 * v2[2]

        ds_x = -0.5 * v1[0] + 0.5 * v3[0]
        ds_y = -0.5 * v1[1] + 0.5 * v3[1]
        ds_z = -0.5 * v1[2] + 0.5 * v3[2]

        cross_x = dr_y * ds_z - dr_z * ds_y
        cross_y = dr_z * ds_x - dr_x * ds_z
        cross_z = dr_x * ds_y - dr_y * ds_x

        h_signed = v1[0] * cross_x + v1[1] * cross_y + v1[2] * cross_z
        h_abs = abs(float(h_signed))
        if h_abs <= tol:
            raise ValueError("Degenerate projected triangle detected: |h| too small.")

        J[k, :] = (R * R * h_abs) / (norm_x ** 3)

        factor = norm_x / (R * h_abs)

        # a^1 = (|x| / (R |h|)) * (d_s x x_flat)
        a1x[k, :] = factor * (ds_y * z_flat - ds_z * y_flat)
        a1y[k, :] = factor * (ds_z * x_flat - ds_x * z_flat)
        a1z[k, :] = factor * (ds_x * y_flat - ds_y * x_flat)

        # a^2 = (|x| / (R |h|)) * (x_flat x d_r)
        a2x[k, :] = factor * (y_flat * dr_z - z_flat * dr_y)
        a2y[k, :] = factor * (z_flat * dr_x - x_flat * dr_z)
        a2z[k, :] = factor * (x_flat * dr_y - y_flat * dr_x)

    nx = X / R
    ny = Y / R
    nz = Z / R

    return ManifoldGeometryCache(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=rs_nodes,
        X=X,
        Y=Y,
        Z=Z,
        J=J,
        nx=nx,
        ny=ny,
        nz=nz,
        a1x=a1x,
        a1y=a1y,
        a1z=a1z,
        a2x=a2x,
        a2y=a2y,
        a2z=a2z,
    )

'@

Write-RepoFile -Path "operators/manifold_sbp_chan2019.py" -Content @'
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
    bary = np.asarray(rule["bary"], dtype=float)

    # Same Table-1 point order as the repository.
    rs_nodes = np.column_stack([2.0 * bary[:, 2] - 1.0, 2.0 * bary[:, 1] - 1.0])
    rule = dict(rule)
    rule["rs"] = rs_nodes

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

        q_t = - M_J^{-1} [ D_SBP(J a^1 · u, q) + D_SBP(J a^2 · u, q)
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

'@

Write-RepoFile -Path "experiments/manifold_sbp_chan2019_convergence.py" -Content @'
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from geometry.sphere_manifold_mesh import (
    generate_spherical_octahedron_mesh,
    spherical_mesh_hmin,
)
from geometry.sphere_projected_exact_metrics import build_exact_projected_sphere_geometry_cache
from operators.manifold_rhs import build_manifold_face_connectivity
from operators.manifold_sbp_chan2019 import (
    build_manifold_sbp_chan_table1_k4_reference_operators,
    constant_state_rhs_diagnostic,
    manifold_sbp_chan2019_rhs_exchange,
)
from problems.sphere_advection import (
    constant_field_xyz,
    exact_gaussian_bell_xyz,
    solid_body_velocity_xyz,
)
from time_integration.CFL import cfl_dt_from_h
from time_integration.lsrk54 import integrate_lsrk54, is_tf_reached

from experiments.manifold_div_h_convergence import (
    compute_convergence_rate,
    manifold_weighted_mass,
    manifold_weighted_norms,
)


_VALID_FIELD_CASES = {"gaussian", "constant"}
_VALID_FLUX_TYPES = {"upwind", "central", "lax_friedrichs"}


@dataclass(frozen=True)
class ManifoldSBPChan2019Config:
    mesh_levels: tuple[int, ...] = (2, 4, 8)
    R: float = 1.0
    u0: float = 1.0
    alpha0: float = math.pi / 4.0
    cfl: float = 0.5
    tf: float = 1.0
    N: int = 4
    gaussian_width: float = 1.0 / math.sqrt(10.0)
    field_case: str = "gaussian"
    constant_value: float = 1.0
    flux_type: str = "upwind"
    alpha_lf: float = 1.0
    use_numba: bool | None = True
    record_history: bool = False
    verbose: bool = True


def _validate_config(config: ManifoldSBPChan2019Config) -> None:
    if not config.mesh_levels:
        raise ValueError("mesh_levels must not be empty.")
    if any(int(level) < 1 for level in config.mesh_levels):
        raise ValueError("mesh_levels must contain positive integers.")
    if config.R <= 0.0:
        raise ValueError("R must be positive.")
    if config.cfl <= 0.0:
        raise ValueError("cfl must be positive.")
    if config.tf < 0.0:
        raise ValueError("tf must be non-negative.")
    if config.N != 4:
        raise ValueError("This implementation is fixed to Table1 k=4 / N=4.")
    if config.gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be positive.")
    if str(config.field_case).lower() not in _VALID_FIELD_CASES:
        raise ValueError("field_case must be one of: gaussian, constant.")
    if str(config.flux_type).lower() not in _VALID_FLUX_TYPES:
        raise ValueError("flux_type must be one of: upwind, central, lax_friedrichs.")
    if config.alpha_lf <= 0.0:
        raise ValueError("alpha_lf must be positive.")


def _field_case(config: ManifoldSBPChan2019Config) -> str:
    return str(config.field_case).strip().lower()


def _flux_type(config: ManifoldSBPChan2019Config) -> str:
    return str(config.flux_type).strip().lower()


def _max_speed(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> float:
    return float(np.max(np.sqrt(U * U + V * V + W * W)))


def _make_reference_field_getter(geom, config: ManifoldSBPChan2019Config):
    field_case = _field_case(config)
    if field_case == "gaussian":
        center_xyz = (0.0, config.R, 0.0)

        def q_ref(t: float) -> np.ndarray:
            return exact_gaussian_bell_xyz(
                geom.X,
                geom.Y,
                geom.Z,
                t=float(t),
                u0=config.u0,
                R=config.R,
                alpha0=config.alpha0,
                width=config.gaussian_width,
                center_xyz=center_xyz,
            )

        return q_ref(0.0), q_ref

    def q_ref(t: float) -> np.ndarray:
        del t
        return constant_field_xyz(geom.X, geom.Y, geom.Z, value=config.constant_value)

    return q_ref(0.0), q_ref


def _mass_relative_error(mass: float, mass0: float) -> float:
    if abs(mass0) <= np.finfo(float).tiny:
        return math.nan
    return float(mass - mass0) / float(mass0)


def _record_error(q, t, q_ref_getter, geom, weights_2d) -> tuple[np.ndarray, dict[str, float]]:
    q_ref = np.asarray(q_ref_getter(float(t)), dtype=float)
    err = np.asarray(q, dtype=float) - q_ref
    norms = manifold_weighted_norms(err, geom.J, weights_2d)
    norms["max_abs_error"] = float(np.max(np.abs(err)))
    return q_ref, norms


def _run_one_level(
    config: ManifoldSBPChan2019Config,
    n_div: int,
    ref_ops,
) -> dict:
    t_start = perf_counter()

    nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=n_div, R=config.R)
    geom = build_exact_projected_sphere_geometry_cache(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=ref_ops.rs_nodes,
        R=config.R,
    )
    conn = build_manifold_face_connectivity(EToV)

    U, V, W = solid_body_velocity_xyz(
        geom.X,
        geom.Y,
        geom.Z,
        u0=config.u0,
        alpha0=config.alpha0,
    )
    velocity_xyz = (U, V, W)
    vmax = _max_speed(U, V, W)
    h = spherical_mesh_hmin(nodes_xyz, EToV)
    dt = cfl_dt_from_h(cfl=config.cfl, h=h, N=config.N + 1, vmax=vmax)

    q0, q_ref_getter = _make_reference_field_getter(geom, config)
    mass0 = manifold_weighted_mass(q0, geom.J, ref_ops.weights_2d)

    if config.verbose:
        diag = constant_state_rhs_diagnostic(
            geom=geom,
            velocity_xyz=velocity_xyz,
            ref_ops=ref_ops,
            conn=conn,
            flux_type=config.flux_type,
            use_numba=config.use_numba,
        )
        print(
            f"[SBP Chan2019] n_div={n_div} constant RHS Linf="
            f"{float(diag['max_rhs_abs']):.6e}, massRHS={float(diag['weighted_rhs_mass']):.6e}"
        )

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        return manifold_sbp_chan2019_rhs_exchange(
            q=q,
            geom=geom,
            velocity_xyz=velocity_xyz,
            ref_ops=ref_ops,
            conn=conn,
            flux_type=config.flux_type,
            alpha_lf=config.alpha_lf,
            global_vmax=vmax,
            t=t,
            use_numba=config.use_numba,
        )

    step_ids: list[int] = []
    times: list[float] = []
    l2_errors: list[float] = []
    linf_errors: list[float] = []
    masses: list[float] = []
    mass_errors: list[float] = []
    mass_rel_errors: list[float] = []

    if config.record_history:
        _, norms0 = _record_error(q0, 0.0, q_ref_getter, geom, ref_ops.weights_2d)
        step_ids.append(0)
        times.append(0.0)
        l2_errors.append(float(norms0["L2"]))
        linf_errors.append(float(norms0["Linf"]))
        masses.append(float(mass0))
        mass_errors.append(0.0)
        mass_rel_errors.append(0.0)

    def _post_step_transform(t_step: float, q_step: np.ndarray) -> np.ndarray:
        _, norms_step = _record_error(q_step, t_step, q_ref_getter, geom, ref_ops.weights_2d)
        mass_step = float(manifold_weighted_mass(q_step, geom.J, ref_ops.weights_2d))
        step_ids.append(len(step_ids))
        times.append(float(t_step))
        l2_errors.append(float(norms_step["L2"]))
        linf_errors.append(float(norms_step["Linf"]))
        masses.append(mass_step)
        mass_errors.append(float(mass_step - mass0))
        mass_rel_errors.append(float(_mass_relative_error(mass_step, mass0)))
        return q_step

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=0.0,
        tf=config.tf,
        dt=dt,
        post_step_transform=_post_step_transform if config.record_history else None,
    )

    _, norms_final = _record_error(qf, tf_used, q_ref_getter, geom, ref_ops.weights_2d)
    mass_final = float(manifold_weighted_mass(qf, geom.J, ref_ops.weights_2d))

    row = {
        "n_div": int(n_div),
        "K": int(EToV.shape[0]),
        "Nv": int(nodes_xyz.shape[0]),
        "Np": int(ref_ops.rs_nodes.shape[0]),
        "total_dof": int(EToV.shape[0] * ref_ops.rs_nodes.shape[0]),
        "h": float(h),
        "dt": float(dt),
        "tf_target": float(config.tf),
        "tf": float(tf_used),
        "reached_tf": bool(is_tf_reached(tf_used, config.tf)),
        "nsteps": int(nsteps),
        "field_case": _field_case(config),
        "flux_type": _flux_type(config),
        "alpha_lf": float(config.alpha_lf),
        "mass0": float(mass0),
        "mass": float(mass_final),
        "mass_error": float(mass_final - mass0),
        "mass_rel_error": float(_mass_relative_error(mass_final, mass0)),
        "L2_error": float(norms_final["L2"]),
        "Linf_error": float(norms_final["Linf"]),
        "max_abs_error": float(norms_final["max_abs_error"]),
        "elapsed_sec": float(perf_counter() - t_start),
    }

    if config.record_history:
        row["history"] = {
            "mesh_level": int(n_div),
            "h": float(h),
            "field_case": _field_case(config),
            "flux_type": _flux_type(config),
            "alpha_lf": float(config.alpha_lf),
            "mass0": float(mass0),
            "step_ids": np.asarray(step_ids, dtype=int),
            "times": np.asarray(times, dtype=float),
            "l2": np.asarray(l2_errors, dtype=float),
            "linf": np.asarray(linf_errors, dtype=float),
            "mass": np.asarray(masses, dtype=float),
            "mass_error": np.asarray(mass_errors, dtype=float),
            "mass_rel_error": np.asarray(mass_rel_errors, dtype=float),
            "reached_tf": bool(row["reached_tf"]),
            "tf_used": float(tf_used),
            "nsteps": int(nsteps),
        }

    if config.verbose:
        print(
            f"[SBP Chan2019] field={row['field_case']:>8s} flux={row['flux_type']:>14s} "
            f"n_div={row['n_div']:3d} K={row['K']:6d} h={row['h']:.6e} "
            f"dt={row['dt']:.3e} steps={row['nsteps']:6d} "
            f"L2={row['L2_error']:.6e} Linf={row['Linf_error']:.6e} "
            f"mass_err={row['mass_error']:.3e} time={row['elapsed_sec']:.2f}s"
        )

    return row


def run_manifold_sbp_chan2019_convergence(
    config: ManifoldSBPChan2019Config,
) -> list[dict]:
    _validate_config(config)
    ref_ops = build_manifold_sbp_chan_table1_k4_reference_operators()
    results = [_run_one_level(config, int(level), ref_ops) for level in config.mesh_levels]

    hs = [float(row["h"]) for row in results]
    l2_rates = compute_convergence_rate([float(row["L2_error"]) for row in results], hs)
    linf_rates = compute_convergence_rate([float(row["Linf_error"]) for row in results], hs)

    for row, rate_l2, rate_linf in zip(results, l2_rates, linf_rates):
        row["rate_L2"] = float(rate_l2)
        row["rate_Linf"] = float(rate_linf)

    return results


def extract_time_histories(results: list[dict]) -> list[dict]:
    return [row["history"] for row in results if isinstance(row.get("history"), dict)]


def print_results_table(results: list[dict]) -> None:
    header = (
        f"{'field':>10s} {'flux':>14s} {'n_div':>6s} {'K':>9s} "
        f"{'h':>12s} {'dt':>12s} {'steps':>8s} "
        f"{'L2':>14s} {'rate':>8s} {'Linf':>14s} {'rate':>8s} {'mass_err':>12s}"
    )
    print(header)
    print("-" * len(header))

    def fmt_rate(value: float) -> str:
        return "   -   " if not np.isfinite(value) else f"{value:8.3f}"

    for row in results:
        print(
            f"{str(row['field_case']):>10s} {str(row['flux_type']):>14s} "
            f"{row['n_div']:6d} {row['K']:9d} {row['h']:12.4e} "
            f"{row['dt']:12.4e} {row['nsteps']:8d} "
            f"{row['L2_error']:14.6e} {fmt_rate(float(row['rate_L2']))} "
            f"{row['Linf_error']:14.6e} {fmt_rate(float(row['rate_Linf']))} "
            f"{row['mass_error']:12.4e}"
        )


def _summary_fieldnames(results: list[dict]) -> list[str]:
    return [
        key for key, value in results[0].items()
        if not isinstance(value, (dict, list, tuple, np.ndarray))
    ]


def save_results_csv(results: list[dict], filepath: str | Path) -> None:
    if not results:
        raise ValueError("results is empty.")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _summary_fieldnames(results)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row[name] for name in fieldnames})


def save_time_history_csv(results: list[dict], filepath: str | Path) -> None:
    histories = extract_time_histories(results)
    if not histories:
        raise ValueError("results does not contain recorded histories.")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mesh_level",
        "h",
        "field_case",
        "flux_type",
        "alpha_lf",
        "mass0",
        "step_index",
        "time",
        "L2_error",
        "Linf_error",
        "mass",
        "mass_error",
        "mass_rel_error",
        "reached_tf",
        "tf_used",
        "nsteps",
    ]
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for history in histories:
            times = np.asarray(history["times"], dtype=float)
            for i in range(times.size):
                writer.writerow(
                    {
                        "mesh_level": int(history["mesh_level"]),
                        "h": float(history["h"]),
                        "field_case": str(history["field_case"]),
                        "flux_type": str(history["flux_type"]),
                        "alpha_lf": float(history["alpha_lf"]),
                        "mass0": float(history["mass0"]),
                        "step_index": int(history["step_ids"][i]),
                        "time": float(times[i]),
                        "L2_error": float(history["l2"][i]),
                        "Linf_error": float(history["linf"][i]),
                        "mass": float(history["mass"][i]),
                        "mass_error": float(history["mass_error"][i]),
                        "mass_rel_error": float(history["mass_rel_error"][i]),
                        "reached_tf": bool(history["reached_tf"]),
                        "tf_used": float(history["tf_used"]),
                        "nsteps": int(history["nsteps"]),
                    }
                )

'@

Write-RepoFile -Path "cli/run_manifold_sbp_chan2019_convergence.py" -Content @'
from __future__ import annotations

import argparse
import math

from experiments.manifold_sbp_chan2019_convergence import (
    ManifoldSBPChan2019Config,
    extract_time_histories,
    print_results_table,
    run_manifold_sbp_chan2019_convergence,
    save_results_csv,
    save_time_history_csv,
)
from experiments.output_paths import experiments_output_dir
from visualization.manifold_diagnostics import (
    plot_manifold_l2_error_vs_time,
    plot_manifold_lsrk_convergence,
    plot_manifold_mass_error_vs_time,
)


def _parse_levels(text: str) -> tuple[int, ...]:
    levels = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not levels:
        raise argparse.ArgumentTypeError("mesh levels must not be empty.")
    return levels


def _float_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(float(value)).replace(".", "p").replace("-", "m")


def _slug(args: argparse.Namespace) -> str:
    base = f"{args.field_case}_{args.flux_type}_tf{_float_slug(args.tf)}_cfl{_float_slug(args.cfl)}"
    if args.flux_type == "lax_friedrichs":
        base += f"_a{_float_slug(args.alpha_lf)}"
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Table1 k=4 Chan-style SBP manifold DG convergence study.",
    )
    parser.add_argument("--mesh-levels", type=_parse_levels, default=(2, 4, 8))
    parser.add_argument("--tf", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=math.pi / 4.0)
    parser.add_argument("--gaussian-width", type=float, default=1.0 / math.sqrt(10.0))
    parser.add_argument("--field-case", choices=("gaussian", "constant"), default="gaussian")
    parser.add_argument(
        "--flux-type",
        choices=("upwind", "central", "lax_friedrichs"),
        default="upwind",
    )
    parser.add_argument("--alpha-lf", type=float, default=1.0)
    parser.add_argument("--constant-value", type=float, default=1.0)
    parser.add_argument("--use-numba", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action="store_true", help="Save optional diagnostic figures.")
    args = parser.parse_args()

    output_dir = experiments_output_dir(__file__, "manifold_sbp_chan2019_convergence")

    config = ManifoldSBPChan2019Config(
        mesh_levels=args.mesh_levels,
        R=args.R,
        u0=args.u0,
        alpha0=args.alpha0,
        cfl=args.cfl,
        tf=args.tf,
        gaussian_width=args.gaussian_width,
        field_case=args.field_case,
        flux_type=args.flux_type,
        alpha_lf=args.alpha_lf,
        constant_value=args.constant_value,
        use_numba=args.use_numba,
        record_history=args.record_history,
        verbose=True,
    )

    results = run_manifold_sbp_chan2019_convergence(config)
    histories = extract_time_histories(results)
    slug = _slug(args)

    csv_path = output_dir / f"manifold_sbp_chan2019_table1_k4_{slug}.csv"
    history_csv_path = output_dir / f"manifold_sbp_chan2019_time_history_table1_k4_{slug}.csv"

    print()
    print_results_table(results)
    save_results_csv(results, csv_path)
    print()
    print("[OK] wrote " + str(csv_path))

    if histories:
        save_time_history_csv(results, history_csv_path)
        print("[OK] wrote " + str(history_csv_path))

    if args.plot:
        fig_path = output_dir / f"manifold_sbp_chan2019_convergence_table1_k4_{slug}.png"
        error_fig_path = output_dir / f"manifold_sbp_chan2019_error_vs_time_table1_k4_{slug}.png"
        mass_fig_path = output_dir / f"manifold_sbp_chan2019_mass_vs_time_table1_k4_{slug}.png"
        plot_manifold_lsrk_convergence(results, fig_path)
        plot_manifold_l2_error_vs_time(histories, error_fig_path)
        plot_manifold_mass_error_vs_time(histories, mass_fig_path)
        print("[OK] wrote " + str(fig_path))
        print("[OK] wrote " + str(error_fig_path))
        print("[OK] wrote " + str(mass_fig_path))


if __name__ == "__main__":
    main()

'@

Write-Host ""
Write-Host "Generated SBP Chan2019 manifold DG files."
Write-Host "Example:"
Write-Host "  python -m cli.run_manifold_sbp_chan2019_convergence --mesh-levels 2,4,8 --tf 1.0 --cfl 0.5 --use-numba --plot"
