from __future__ import annotations

import numpy as np

from data.table1_rules import load_table1_rule
from experiments.lsrk_h_convergence import build_projected_inverse_mass_from_rule
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from operators.mass import mass_matrix_from_quadrature
from operators.trace_policy import build_trace_policy
from operators.rhs_split_conservative_exchange import (
    _lift_surface_penalty_to_volume,
    build_surface_exchange_cache,
    surface_term_from_exchange,
)
from operators.vandermonde2d import vandermonde2d


def q_boundary_zero(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return np.zeros_like(x)


def velocity_one_one(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(x), np.ones_like(y)


def _reference_surface_lift_from_penalty(
    p: np.ndarray,
    rule: dict,
    trace: dict,
    face_geom: dict,
    Np: int,
) -> np.ndarray:
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    inv_ws = 1.0 / ws
    area = np.asarray(face_geom["area"], dtype=float).reshape(-1)
    length = np.asarray(face_geom["length"], dtype=float)

    K = p.shape[0]
    out = np.zeros((K, Np), dtype=float)

    for face_id in (1, 2, 3):
        jf = face_id - 1
        ids = np.asarray(trace["face_node_ids"][face_id], dtype=int).reshape(-1)
        we = np.asarray(trace["face_weights"][face_id], dtype=float).reshape(-1)
        w_ratio = we * inv_ws[ids]

        scale = (length[:, jf] / area)[:, None]
        face_contrib = scale * w_ratio[None, :] * p[:, jf, :]

        if np.unique(ids).size == ids.size:
            out[:, ids] += face_contrib
        else:
            row_idx = np.broadcast_to(np.arange(K)[:, None], face_contrib.shape)
            col_idx = np.broadcast_to(ids[None, :], face_contrib.shape)
            np.add.at(out, (row_idx, col_idx), face_contrib)

    return out


def test_exchange_surface_lift_scaling_matches_reference() -> None:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    K = int(EToV.shape[0])
    Np = int(len(rule["ws"]))

    rng = np.random.default_rng(123)
    q_elem = rng.normal(size=(K, Np))

    surface_rhs, diag = surface_term_from_exchange(
        q_elem=q_elem,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_boundary_zero,
        velocity=velocity_one_one,
        t=0.0,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=True,
        use_numba=False,
    )

    p = np.asarray(diag["p"], dtype=float)
    assert np.max(np.abs(p)) > 1e-10, "penalty is unexpectedly zero; test setup is degenerate"

    surface_ref = _reference_surface_lift_from_penalty(
        p=p,
        rule=rule,
        trace=trace,
        face_geom=face_geom,
        Np=Np,
    )

    assert np.allclose(surface_rhs, surface_ref, atol=1e-12, rtol=1e-12)


def test_surface_inverse_mass_matrix_path_does_not_double_apply_inv_ws() -> None:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)
    cache = build_surface_exchange_cache(
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
    )

    K = int(EToV.shape[0])
    Np = int(len(rule["ws"]))
    nfp = int(trace["nfp"])

    rng = np.random.default_rng(456)
    p = rng.normal(size=(K, 3, nfp))

    inv_ws = 1.0 / np.asarray(rule["ws"], dtype=float).reshape(-1)
    surface_inverse_mass_t = np.ascontiguousarray(np.diag(inv_ws), dtype=float)

    lifted = _lift_surface_penalty_to_volume(
        p=p,
        cache=cache,
        use_numba=False,
        surface_inverse_mass_T=surface_inverse_mass_t,
    )
    lifted_ref = _reference_surface_lift_from_penalty(
        p=p,
        rule=rule,
        trace=trace,
        face_geom=face_geom,
        Np=Np,
    )

    assert np.allclose(lifted, lifted_ref, atol=1e-12, rtol=1e-12)


def test_projected_inverse_mass_is_direct_v_minv_vt_operator() -> None:
    rule = load_table1_rule(4)
    N = 4

    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    M = mass_matrix_from_quadrature(V, ws, area=0.5)

    projected_inverse_mass = build_projected_inverse_mass_from_rule(rule, N)
    direct = V @ np.linalg.solve(M, 0.5 * V.T)
    projector = V @ np.linalg.solve(M, 0.5 * (V.T * ws[None, :]))
    legacy = projector * (1.0 / ws)[None, :]

    assert np.allclose(projected_inverse_mass, direct, atol=1e-12, rtol=1e-12)
    assert np.allclose(projected_inverse_mass, legacy, atol=1e-12, rtol=1e-12)
