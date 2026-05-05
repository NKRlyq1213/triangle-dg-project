from __future__ import annotations

import numpy as np

from operators.vandermonde2d import vandermonde2d
from geometry.reference_triangle import reference_triangle_area
from operators.mass import mass_matrix_from_quadrature
from operators.rhs_split_conservative_exchange import numerical_flux_penalty
from operators.rhs_split_conservative_exact_trace import _surface_lift_exact_trace


def build_projected_surface_inverse_mass_T(
    *,
    N: int,
    rule: dict | None = None,
    rs_nodes: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    area: float | None = None,
) -> np.ndarray:
    r"""
    Build projected inverse-mass transpose compatible with repo lifting.

    This supports the SDG/Table1 overdetermined case:

        V.shape = (Nq, Nm), with Nq > Nm.

    For Table1 order=4, N=4:

        V.shape = (22, 15)

    The projected inverse mass used by the repo is:

        M = A V^T W V
        P = V M^{-1} (A V^T)

    where:
        A = reference triangle area
        W = diag(volume quadrature weights)

    The repo surface lifting applies right multiplication:

        surface_rhs = surface_integral @ surface_inverse_mass_t
        surface_rhs /= area_flat

    Therefore this function returns:

        P.T

    so that right multiplication matches the existing repo convention.
    """
    if rule is not None:
        rs = np.asarray(rule["rs"], dtype=float)
        w = np.asarray(rule["ws"], dtype=float).reshape(-1)
        A = reference_triangle_area() if area is None else float(area)
    else:
        if rs_nodes is None or weights is None:
            raise ValueError(
                "Either provide rule, or provide both rs_nodes and weights."
            )
        rs = np.asarray(rs_nodes, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)
        A = reference_triangle_area() if area is None else float(area)

    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs nodes must have shape (Nq, 2).")
    if w.shape != (rs.shape[0],):
        raise ValueError("weights must have shape (Nq,).")
    if np.any(w <= 0.0):
        raise ValueError("volume quadrature weights must be strictly positive.")

    V = vandermonde2d(
        N,
        rs[:, 0],
        rs[:, 1],
    )

    if V.shape[0] != rs.shape[0]:
        raise ValueError("Vandermonde row count must match number of nodes.")

    M = mass_matrix_from_quadrature(
        V,
        w,
        area=A,
    )

    rhs = A * V.T

    # Avoid forming inv(M) explicitly:
    # solve M X = rhs, then P = V X.
    X = np.linalg.solve(M, rhs)
    P = V @ X

    if P.shape != (rs.shape[0], rs.shape[0]):
        raise ValueError(
            "Projected inverse mass must have shape (Nq, Nq). "
            f"Got {P.shape}."
        )

    return np.ascontiguousarray(P.T, dtype=float)

def build_face_weight_array_from_rule(
    rule: dict,
    ref_face,
) -> np.ndarray:
    """
    Build face_weights array with shape (3, Nfp) from Table1 edge weights.

    Important:
    - ref_face.face_wratio stores we / ws.
    - repo projected lifting expects raw face weights we.
    """
    we = np.asarray(rule["we"], dtype=float).reshape(-1)

    face_weights = []
    for f in range(3):
        ids = np.asarray(ref_face.face_node_ids[f], dtype=int)
        wf = we[ids]
        if np.any(~np.isfinite(wf)):
            raise ValueError(f"Face {f} contains invalid edge weights.")
        face_weights.append(wf)

    return np.asarray(face_weights, dtype=float)


def collect_surface_penalty_array(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    ref_face,
    conn,
    *,
    tau: float = 0.0,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
) -> np.ndarray:
    r"""
    Collect face penalty p with shape (K, 3, Nfp).

    Penalty convention is exactly the repo convention:

        p = numerical_flux_penalty(ndotF, qM, qP, tau)

    where:

        ndotF = n_x Fx_area + n_y Fy_area.

    Boundary modes
    --------------
    same_state:
        qP = qM on boundary. Useful for constant-state preservation tests.
    constant:
        qP = q_boundary_value on boundary.
    """
    q = np.asarray(q, dtype=float)
    Fx_area = np.asarray(Fx_area, dtype=float)
    Fy_area = np.asarray(Fy_area, dtype=float)

    if not (q.shape == Fx_area.shape == Fy_area.shape):
        raise ValueError("q, Fx_area, Fy_area must have the same shape.")
    if q.ndim != 2:
        raise ValueError("q must have shape (K, Np).")

    mode = str(boundary_mode).strip().lower()
    if mode not in ("same_state", "constant"):
        raise ValueError("boundary_mode must be 'same_state' or 'constant'.")

    K, _Np = q.shape
    nfp = ref_face.nfp

    p = np.zeros((K, 3, nfp), dtype=float)

    for k in range(K):
        for f in range(3):
            ids = np.asarray(ref_face.face_node_ids[f], dtype=int)

            qM = q[k, ids]
            FxM = Fx_area[k, ids]
            FyM = Fy_area[k, ids]

            ndotF = (
                conn.face_normal_x[k, f] * FxM
                + conn.face_normal_y[k, f] * FyM
            )

            if conn.is_boundary[k, f]:
                if mode == "same_state":
                    qP = qM
                else:
                    qP = np.full_like(qM, float(q_boundary_value))
            else:
                kp = conn.neighbor_elem[k, f]
                idsP = conn.neighbor_node_ids[k, f, :]
                qP = q[kp, idsP]

            p[k, f, :] = numerical_flux_penalty(
                ndotF,
                qM,
                qP,
                tau=tau,
            )

    return p


def sdg_surface_penalty_rhs_repo_lift(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    rule: dict,
    ref_face,
    conn,
    *,
    J_area: float,
    tau: float = 0.0,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
    surface_inverse_mass_T: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Surface penalty RHS using the repo's lifting convention.

    If surface_inverse_mass_T is None:
        uses repo diagonal lifting.

    If surface_inverse_mass_T is provided:
        uses repo projected lifting:

            surface_integral @ surface_inverse_mass_T / area_flat.

    Since our equation is the equal-area sphere form:

        d_t(J_area q) + div(F_area q) = 0,

    we additionally divide the flat lifted surface term by J_area:

        q_t_surface = flat_surface_rhs / J_area.
    """
    if J_area == 0.0:
        raise ValueError("J_area must be nonzero.")

    p = collect_surface_penalty_array(
        q,
        Fx_area,
        Fy_area,
        ref_face,
        conn,
        tau=tau,
        boundary_mode=boundary_mode,
        q_boundary_value=q_boundary_value,
    )

    face_node_ids = np.asarray(
        [np.asarray(ref_face.face_node_ids[f], dtype=np.int64) for f in range(3)],
        dtype=np.int64,
    )

    face_weights = build_face_weight_array_from_rule(
        rule,
        ref_face,
    )

    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    flat_surface_rhs = _surface_lift_exact_trace(
        p=p,
        face_node_ids=face_node_ids,
        face_weights=face_weights,
        length=np.asarray(conn.face_length, dtype=float),
        area=np.asarray(conn.area_flat, dtype=float),
        ws=ws,
        surface_inverse_mass_t=surface_inverse_mass_T,
        use_numba=False,
    )

    return flat_surface_rhs / float(J_area)


def sdg_surface_penalty_rhs_diagonal_repo_lift(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    rule: dict,
    ref_face,
    conn,
    *,
    J_area: float,
    tau: float = 0.0,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
) -> np.ndarray:
    """
    Explicit wrapper for repo diagonal lifting.
    """
    return sdg_surface_penalty_rhs_repo_lift(
        q,
        Fx_area,
        Fy_area,
        rule,
        ref_face,
        conn,
        J_area=J_area,
        tau=tau,
        boundary_mode=boundary_mode,
        q_boundary_value=q_boundary_value,
        surface_inverse_mass_T=None,
    )


def sdg_surface_penalty_rhs_projected_repo_lift(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    rule: dict,
    ref_face,
    conn,
    *,
    N: int,
    rs_nodes: np.ndarray,
    J_area: float,
    tau: float = 0.0,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
    surface_inverse_mass_T: np.ndarray | None = None,
) -> np.ndarray:
    """
    Explicit wrapper for repo projected lifting.

    If surface_inverse_mass_T is not supplied, it is constructed from
    Vandermonde data.
    """
    if surface_inverse_mass_T is None:
        surface_inverse_mass_T = build_projected_surface_inverse_mass_T(N=N, rule=rule)

    return sdg_surface_penalty_rhs_repo_lift(
        q,
        Fx_area,
        Fy_area,
        rule,
        ref_face,
        conn,
        J_area=J_area,
        tau=tau,
        boundary_mode=boundary_mode,
        q_boundary_value=q_boundary_value,
        surface_inverse_mass_T=surface_inverse_mass_T,
    )

