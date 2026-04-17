from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from data.table1_rules import load_table1_rule
from experiments.lsrk_h_convergence import (
    build_projected_inverse_mass_from_rule,
    build_reference_diff_operators_from_rule,
)
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.metrics import affine_geometric_factors_from_mesh
from operators.rhs_split_conservative_exchange import (
    build_surface_exchange_cache,
    build_volume_split_cache,
    rhs_split_conservative_exchange,
)
from operators.trace_policy import build_trace_policy
from time_integration.lsrk54 import lsrk54_step


_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_SIMPLEX_ROOT = _WORKSPACE_ROOT / "Simplex-DG-solver"
if not _SIMPLEX_ROOT.exists():
    pytest.skip(
        "Simplex-DG-solver repo not found; skipping simplex_strict parity tests.",
        allow_module_level=True,
    )

if str(_SIMPLEX_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIMPLEX_ROOT))

from src.bases.vandermonde import grad_vandermonde_2d_dubiner, vandermonde_2d_dubiner
from src.core.connectivity import build_connectivity
from src.core.generators import get_reference_data
from src.geometry.mappings import rs_to_xy
from src.geometry.metrics import compute_geometric_factors_batch
from src.reconstruction import build_differentiation_matrices, build_fmask_table1


def _velocity_one_one(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    del t
    return np.ones_like(x), np.ones_like(y)


def _q_boundary_nonperiodic(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return (
        np.sin(5.0 * x + 0.3 - 0.9 * t)
        + 0.37 * np.cos(7.0 * y + 0.1 + 0.6 * t)
        + 0.11 * np.sin(3.0 * x * y + 0.2 * t)
    )


def _mesh_min_edge_length(VX: np.ndarray, VY: np.ndarray, EToV: np.ndarray) -> float:
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)

    hmin = np.inf
    for vids in EToV:
        vertices = np.column_stack([VX[vids], VY[vids]])
        e01 = float(np.linalg.norm(vertices[1] - vertices[0]))
        e12 = float(np.linalg.norm(vertices[2] - vertices[1]))
        e20 = float(np.linalg.norm(vertices[0] - vertices[2]))
        hmin = min(hmin, e01, e12, e20)
    return float(hmin)


def _generate_square_mesh_main_diagonal(n_div: int) -> tuple[np.ndarray, np.ndarray, int]:
    grid = np.linspace(0.0, 1.0, int(n_div) + 1)
    xg, yg = np.meshgrid(grid, grid, indexing="xy")
    nodes = np.column_stack((xg.ravel(), yg.ravel()))

    def node_id(i: int, j: int) -> int:
        return j * (int(n_div) + 1) + i

    triangles: list[list[int]] = []
    for j in range(int(n_div)):
        for i in range(int(n_div)):
            v00 = node_id(i, j)
            v10 = node_id(i + 1, j)
            v01 = node_id(i, j + 1)
            v11 = node_id(i + 1, j + 1)
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])

    etov = np.asarray(triangles, dtype=int)
    return nodes, etov, int(etov.shape[0])


def _build_global_index_maps(
    *,
    etov: np.ndarray,
    etoe: np.ndarray,
    etof: np.ndarray,
    xi_ref: np.ndarray,
    eta_ref: np.ndarray,
    np_nodes: int,
    weights_1d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_elem = int(etov.shape[0])
    nfp = int(len(weights_1d))

    bary_coords = np.column_stack(
        [
            (-xi_ref - eta_ref) / 2.0,
            (xi_ref + 1.0) / 2.0,
            (eta_ref + 1.0) / 2.0,
        ]
    )
    fmask = build_fmask_table1(bary_coords)

    vmap_m = np.zeros((3 * nfp, k_elem), dtype=int)
    vmap_p = np.zeros((3 * nfp, k_elem), dtype=int)

    for k in range(k_elem):
        for face in range(3):
            local_nodes = fmask[:, face]
            interior_indices = k * np_nodes + local_nodes
            vmap_m[face * nfp : (face + 1) * nfp, k] = interior_indices

            k_neighbor = int(etoe[k, face])
            f_neighbor = int(etof[k, face])
            if k_neighbor == k:
                vmap_p[face * nfp : (face + 1) * nfp, k] = interior_indices
            else:
                neighbor_local_nodes = fmask[:, f_neighbor]
                neighbor_indices = k_neighbor * np_nodes + neighbor_local_nodes
                vmap_p[face * nfp : (face + 1) * nfp, k] = neighbor_indices[::-1]

    boundary_mask = vmap_m == vmap_p
    return vmap_m, vmap_p, boundary_mask, fmask


def _coord_key(x_val: float, y_val: float, tol: float) -> tuple[int, int]:
    return int(np.round(x_val / tol)), int(np.round(y_val / tol))


def _apply_periodic_vmapP(
    *,
    vmap_m: np.ndarray,
    vmap_p: np.ndarray,
    boundary_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    vmap_p_periodic = np.array(vmap_p, copy=True)

    x_flat = x.flatten(order="F")
    y_flat = y.flatten(order="F")

    boundary_gids = np.unique(vmap_m[boundary_mask])
    coord_lookup: dict[tuple[int, int], list[int]] = {}
    for gid in boundary_gids:
        key = _coord_key(float(x_flat[gid]), float(y_flat[gid]), tol)
        coord_lookup.setdefault(key, []).append(int(gid))

    boundary_slots = np.argwhere(boundary_mask)
    for slot_row, slot_col in boundary_slots:
        gid_m = int(vmap_m[slot_row, slot_col])
        x_m = float(x_flat[gid_m])
        y_m = float(y_flat[gid_m])

        x_target = 1.0 if np.isclose(x_m, 0.0, atol=tol) else (0.0 if np.isclose(x_m, 1.0, atol=tol) else x_m)
        y_target = 1.0 if np.isclose(y_m, 0.0, atol=tol) else (0.0 if np.isclose(y_m, 1.0, atol=tol) else y_m)

        key_target = _coord_key(x_target, y_target, tol)
        candidates = coord_lookup.get(key_target, [])
        if not candidates:
            raise ValueError("No periodic counterpart found for boundary node.")

        gid_p = next((cand for cand in candidates if cand != gid_m), candidates[0])
        vmap_p_periodic[slot_row, slot_col] = gid_p

    return vmap_p_periodic


def _compute_face_geometry(
    *,
    nodes: np.ndarray,
    etov: np.ndarray,
    weights_1d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_elem = int(etov.shape[0])
    nfp = int(len(weights_1d))

    j_face = np.zeros((3, k_elem))
    nx_array = np.zeros((3, k_elem))
    ny_array = np.zeros((3, k_elem))

    for k_id in range(k_elem):
        v1, v2, v3 = nodes[etov[k_id]]
        edges = [v2 - v1, v3 - v2, v1 - v3]
        for face in range(3):
            dx, dy = edges[face]
            length = np.hypot(dx, dy)
            j_face[face, k_id] = length / 2.0
            nx_array[face, k_id] = dy / length
            ny_array[face, k_id] = -dx / length

    nx_expanded = np.repeat(nx_array, nfp, axis=0)
    ny_expanded = np.repeat(ny_array, nfp, axis=0)
    j_face_expanded = np.repeat(j_face, nfp, axis=0)
    return nx_expanded, ny_expanded, j_face_expanded


def _build_triangle_context(*, n: int, k: int) -> dict:
    rule = load_table1_rule(k)
    trace = build_trace_policy(rule)
    Dr, Ds = build_reference_diff_operators_from_rule(rule, k)

    VX, VY, EToV = structured_square_tri_mesh(nx=n, ny=n, diagonal="main")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    u_elem, v_elem = _velocity_one_one(X, Y, t=0.0)
    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    surface_cache = build_surface_exchange_cache(
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        face_order_mode="simplex_strict",
    )
    volume_cache = build_volume_split_cache(
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
    )

    surface_inverse_mass = build_projected_inverse_mass_from_rule(rule, k)
    surface_inverse_mass_t = np.ascontiguousarray(surface_inverse_mass.T, dtype=float)

    ndotV_precomputed = np.ascontiguousarray(
        np.asarray(face_geom["nx"], dtype=float) + np.asarray(face_geom["ny"], dtype=float),
        dtype=float,
    )
    ndotV_flat_precomputed = np.ascontiguousarray(
        ndotV_precomputed.reshape(-1, int(trace["nfp"])),
        dtype=float,
    )

    hmin = _mesh_min_edge_length(VX, VY, EToV)
    vmax = float(np.sqrt(2.0))
    dt_nominal = float(hmin / (((k + 1) ** 2) * vmax))

    return {
        "rule": rule,
        "trace": trace,
        "Dr": Dr,
        "Ds": Ds,
        "VX": VX,
        "VY": VY,
        "EToV": EToV,
        "conn": conn,
        "X": X,
        "Y": Y,
        "geom": geom,
        "face_geom": face_geom,
        "u_elem": u_elem,
        "v_elem": v_elem,
        "surface_cache": surface_cache,
        "volume_cache": volume_cache,
        "surface_inverse_mass_t": surface_inverse_mass_t,
        "ndotV_precomputed": ndotV_precomputed,
        "ndotV_flat_precomputed": ndotV_flat_precomputed,
        "K": int(EToV.shape[0]),
        "Np": int(len(rule["ws"])),
        "dt": dt_nominal,
    }


def _build_simplex_context(*, n: int, k: int) -> dict:
    ref_data = get_reference_data("table1", k)
    xi_ref = np.asarray(ref_data["xi"], dtype=float)
    eta_ref = np.asarray(ref_data["eta"], dtype=float)
    weights_ref = np.asarray(ref_data["weights"], dtype=float)
    weights_1d = np.asarray(ref_data["weights_1d"], dtype=float)
    np_nodes = int(len(xi_ref))
    nfp = int(len(weights_1d))

    nodes, etov, k_elem = _generate_square_mesh_main_diagonal(n)
    etoe, etof = build_connectivity(etov)

    vmap_m, vmap_p, boundary_mask, fmask = _build_global_index_maps(
        etov=etov,
        etoe=etoe,
        etof=etof,
        xi_ref=xi_ref,
        eta_ref=eta_ref,
        np_nodes=np_nodes,
        weights_1d=weights_1d,
    )

    v_nodal = vandermonde_2d_dubiner(xi_ref, eta_ref, k)
    vr, vs = grad_vandermonde_2d_dubiner(xi_ref, eta_ref, k)
    d_r_ref, d_s_ref = build_differentiation_matrices(v_nodal, vr, vs, w=weights_ref)

    w_diag = np.diag(weights_ref)
    m_modal = v_nodal.T @ w_diag @ v_nodal
    m_inv_projected = v_nodal @ np.linalg.inv(m_modal) @ v_nodal.T

    E = np.zeros((3 * nfp, np_nodes), dtype=float)
    for face in range(3):
        for local_i, node_idx in enumerate(fmask[:, face]):
            E[face * nfp + local_i, node_idx] = 1.0

    vertices = nodes[etov]
    metrics = compute_geometric_factors_batch(vertices)
    J = np.asarray(metrics["J"], dtype=float)
    rx = np.asarray(metrics["rx"], dtype=float)
    ry = np.asarray(metrics["ry"], dtype=float)
    sx = np.asarray(metrics["sx"], dtype=float)
    sy = np.asarray(metrics["sy"], dtype=float)

    X = np.zeros((np_nodes, k_elem), dtype=float)
    Y = np.zeros((np_nodes, k_elem), dtype=float)
    for k_id in range(k_elem):
        v1, v2, v3 = nodes[etov[k_id]]
        x_phys, y_phys = rs_to_xy(xi_ref, eta_ref, v1, v2, v3)
        X[:, k_id] = x_phys
        Y[:, k_id] = y_phys

    vmap_p_periodic = _apply_periodic_vmapP(
        vmap_m=vmap_m,
        vmap_p=vmap_p,
        boundary_mask=boundary_mask,
        x=X,
        y=Y,
        tol=1e-12,
    )

    nx_expanded, ny_expanded, j_face_expanded = _compute_face_geometry(
        nodes=nodes,
        etov=etov,
        weights_1d=weights_1d,
    )

    return {
        "xi_ref": xi_ref,
        "eta_ref": eta_ref,
        "weights_1d": weights_1d,
        "np_nodes": np_nodes,
        "nfp": nfp,
        "nodes": nodes,
        "etov": etov,
        "k_elem": k_elem,
        "vmap_m": vmap_m,
        "vmap_p": vmap_p,
        "vmap_p_periodic": vmap_p_periodic,
        "boundary_mask": boundary_mask,
        "D_r": np.asarray(d_r_ref, dtype=float),
        "D_s": np.asarray(d_s_ref, dtype=float),
        "E": np.asarray(E, dtype=float),
        "J": J,
        "rx": rx,
        "ry": ry,
        "sx": sx,
        "sy": sy,
        "X": X,
        "Y": Y,
        "M_inv_projected": np.asarray(m_inv_projected, dtype=float),
        "nx_expanded": np.asarray(nx_expanded, dtype=float),
        "ny_expanded": np.asarray(ny_expanded, dtype=float),
        "j_face_expanded": np.asarray(j_face_expanded, dtype=float),
    }


def _build_local_permutation(triangle: dict, simplex: dict, tol: float = 1e-12) -> np.ndarray:
    X_tri = np.asarray(triangle["X"], dtype=float)
    Y_tri = np.asarray(triangle["Y"], dtype=float)
    X_sx = np.asarray(simplex["X"], dtype=float)
    Y_sx = np.asarray(simplex["Y"], dtype=float)

    K = int(triangle["K"])
    Np = int(triangle["Np"])

    tri_to_sx = np.empty((K, Np), dtype=int)
    for k in range(K):
        lookup: dict[tuple[int, int], int] = {}
        for i_sx in range(Np):
            key = _coord_key(float(X_sx[i_sx, k]), float(Y_sx[i_sx, k]), tol)
            lookup[key] = i_sx

        for i_tri in range(Np):
            key = _coord_key(float(X_tri[k, i_tri]), float(Y_tri[k, i_tri]), tol)
            if key not in lookup:
                raise ValueError("Failed to map triangle node to simplex node.")
            tri_to_sx[k, i_tri] = int(lookup[key])

    return tri_to_sx


def _tri_to_simplex(q_tri: np.ndarray, tri_to_sx: np.ndarray) -> np.ndarray:
    q_tri = np.asarray(q_tri, dtype=float)
    K, Np = q_tri.shape
    out = np.empty((Np, K), dtype=float)
    for k in range(K):
        out[tri_to_sx[k], k] = q_tri[k]
    return out


def _simplex_to_tri(q_sx: np.ndarray, tri_to_sx: np.ndarray) -> np.ndarray:
    q_sx = np.asarray(q_sx, dtype=float)
    K, Np = tri_to_sx.shape
    out = np.empty((K, Np), dtype=float)
    for k in range(K):
        out[k] = q_sx[tri_to_sx[k], k]
    return out


def _eval_triangle_rhs(
    triangle: dict,
    q_tri: np.ndarray,
    *,
    t: float,
    boundary_mode: str,
    q_boundary,
    return_diagnostics: bool,
) -> tuple[np.ndarray, dict]:
    return rhs_split_conservative_exchange(
        q_elem=np.asarray(q_tri, dtype=float),
        u_elem=triangle["u_elem"],
        v_elem=triangle["v_elem"],
        Dr=triangle["Dr"],
        Ds=triangle["Ds"],
        geom=triangle["geom"],
        rule=triangle["rule"],
        trace=triangle["trace"],
        conn=triangle["conn"],
        face_geom=triangle["face_geom"],
        q_boundary=q_boundary,
        velocity=_velocity_one_one,
        t=float(t),
        tau=0.0,
        tau_interior=0.0,
        tau_qb=0.0,
        compute_mismatches=False,
        return_diagnostics=bool(return_diagnostics),
        use_numba=False,
        surface_backend="face-major",
        surface_cache=triangle["surface_cache"],
        ndotV_precomputed=triangle["ndotV_precomputed"],
        ndotV_flat_precomputed=triangle["ndotV_flat_precomputed"],
        surface_inverse_mass_T=triangle["surface_inverse_mass_t"],
        volume_split_cache=triangle["volume_cache"],
        physical_boundary_mode=boundary_mode,
        face_order_mode="simplex_strict",
    )


def _eval_simplex_rhs(
    simplex: dict,
    q_sx: np.ndarray,
    *,
    t: float,
    boundary_mode: str,
    q_boundary,
) -> tuple[np.ndarray, dict]:
    q = np.asarray(q_sx, dtype=float)

    u_arr, v_arr = _velocity_one_one(simplex["X"], simplex["Y"], t=t)

    alpha = simplex["J"][None, :] * (simplex["rx"][None, :] * u_arr + simplex["ry"][None, :] * v_arr)
    beta = simplex["J"][None, :] * (simplex["sx"][None, :] * u_arr + simplex["sy"][None, :] * v_arr)

    split_r = 0.5 * (
        simplex["D_r"] @ (alpha * q)
        + alpha * (simplex["D_r"] @ q)
        + q * (simplex["D_r"] @ alpha)
    )
    split_s = 0.5 * (
        simplex["D_s"] @ (beta * q)
        + beta * (simplex["D_s"] @ q)
        + q * (simplex["D_s"] @ beta)
    )
    volume_rhs = -(split_r + split_s) / simplex["J"][None, :]

    q_flat = q.flatten(order="F")
    qM = q_flat[simplex["vmap_m"]]

    if boundary_mode == "opposite_boundary":
        qP = q_flat[simplex["vmap_p_periodic"]]
    elif boundary_mode == "exact_qb":
        if q_boundary is None:
            raise ValueError("q_boundary callback is required for exact_qb mode.")
        qP = q_flat[simplex["vmap_p"]].copy()
        x_m = simplex["X"].flatten(order="F")[simplex["vmap_m"]]
        y_m = simplex["Y"].flatten(order="F")[simplex["vmap_m"]]
        qB = np.asarray(q_boundary(x_m, y_m, float(t)), dtype=float)
        if qB.shape != qP.shape:
            qB = np.broadcast_to(qB, qP.shape)
        qP[simplex["boundary_mask"]] = qB[simplex["boundary_mask"]]
    else:
        raise ValueError("boundary_mode must be 'opposite_boundary' or 'exact_qb'.")

    u_flat = u_arr.flatten(order="F")
    v_flat = v_arr.flatten(order="F")
    ndotV = (
        simplex["nx_expanded"] * u_flat[simplex["vmap_m"]]
        + simplex["ny_expanded"] * v_flat[simplex["vmap_m"]]
    )

    p = np.minimum(ndotV, 0.0) * (qM - qP)
    face_w = np.tile(simplex["weights_1d"], 3)
    scaled_penalty = p * face_w[:, None] * simplex["j_face_expanded"]

    surface_integral = simplex["E"].T @ scaled_penalty
    surface_rhs = (simplex["M_inv_projected"] @ surface_integral) / simplex["J"][None, :]

    total_rhs = volume_rhs + surface_rhs
    return total_rhs, {"volume_rhs": volume_rhs, "surface_rhs": surface_rhs}


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


@pytest.fixture(scope="module")
def parity_context() -> dict:
    n = 8
    k = 4
    triangle = _build_triangle_context(n=n, k=k)
    simplex = _build_simplex_context(n=n, k=k)
    tri_to_sx = _build_local_permutation(triangle, simplex)
    return {
        "triangle": triangle,
        "simplex": simplex,
        "tri_to_sx": tri_to_sx,
    }


def _random_state(triangle: dict, *, seed: int, scale: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return float(scale) * rng.normal(size=(triangle["K"], triangle["Np"]))


def test_simplex_strict_stage0_rhs_parity_random_state(
    parity_context: dict,
) -> None:
    triangle = parity_context["triangle"]
    simplex = parity_context["simplex"]
    tri_to_sx = parity_context["tri_to_sx"]
    boundary_mode = "opposite_boundary"

    q0_tri = _random_state(triangle, seed=20260417, scale=0.1)
    q0_sx = _tri_to_simplex(q0_tri, tri_to_sx)

    q_boundary = None

    rhs_tri, diag_tri = _eval_triangle_rhs(
        triangle,
        q0_tri,
        t=0.0,
        boundary_mode=boundary_mode,
        q_boundary=q_boundary,
        return_diagnostics=True,
    )
    rhs_sx, diag_sx = _eval_simplex_rhs(
        simplex,
        q0_sx,
        t=0.0,
        boundary_mode=boundary_mode,
        q_boundary=q_boundary,
    )

    rhs_sx_tri = _simplex_to_tri(rhs_sx, tri_to_sx)
    vol_sx_tri = _simplex_to_tri(diag_sx["volume_rhs"], tri_to_sx)
    surf_sx_tri = _simplex_to_tri(diag_sx["surface_rhs"], tri_to_sx)

    vol_diff = _max_abs_diff(diag_tri["volume_rhs"], vol_sx_tri)
    surf_diff = _max_abs_diff(diag_tri["surface_rhs"], surf_sx_tri)
    rhs_diff = _max_abs_diff(rhs_tri, rhs_sx_tri)

    assert vol_diff <= 1e-12, f"Stage0 volume RHS parity failed: {vol_diff:.3e}"
    assert surf_diff <= 1e-12, f"Stage0 surface RHS parity failed: {surf_diff:.3e}"
    assert rhs_diff <= 1e-12, f"Stage0 total RHS parity failed: {rhs_diff:.3e}"


@pytest.mark.parametrize("boundary_mode", ("opposite_boundary", "exact_qb"))
def test_simplex_strict_one_full_dt_parity(
    parity_context: dict,
    boundary_mode: str,
) -> None:
    triangle = parity_context["triangle"]
    simplex = parity_context["simplex"]
    tri_to_sx = parity_context["tri_to_sx"]

    q0_tri = _random_state(triangle, seed=20260418, scale=0.1)
    q0_sx = _tri_to_simplex(q0_tri, tri_to_sx)
    q_boundary = _q_boundary_nonperiodic if boundary_mode == "exact_qb" else None

    dt = float(triangle["dt"])

    def rhs_triangle(t: float, q: np.ndarray) -> np.ndarray:
        rhs, _ = _eval_triangle_rhs(
            triangle,
            q,
            t=t,
            boundary_mode=boundary_mode,
            q_boundary=q_boundary,
            return_diagnostics=False,
        )
        return rhs

    def rhs_simplex(t: float, q: np.ndarray) -> np.ndarray:
        rhs, _ = _eval_simplex_rhs(
            simplex,
            q,
            t=t,
            boundary_mode=boundary_mode,
            q_boundary=q_boundary,
        )
        return rhs

    q1_tri = lsrk54_step(rhs=rhs_triangle, t=0.0, q=q0_tri, dt=dt)
    q1_sx = lsrk54_step(rhs=rhs_simplex, t=0.0, q=q0_sx, dt=dt)

    q1_sx_tri = _simplex_to_tri(q1_sx, tri_to_sx)
    q1_diff = _max_abs_diff(q1_tri, q1_sx_tri)

    assert q1_diff <= 1e-12, f"One-full-dt parity failed: {q1_diff:.3e}"
