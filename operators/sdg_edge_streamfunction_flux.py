from __future__ import annotations

import numpy as np

from operators.rhs_split_conservative_exact_trace import _surface_lift_exact_trace
from operators.sdg_projected_surface_flux import (
    build_face_weight_array_from_rule,
    build_projected_surface_inverse_mass_T,
)


def lagrange_derivative_matrix_1d(x: np.ndarray) -> np.ndarray:
    """
    Differentiation matrix for Lagrange interpolation nodes x.

    D[i,j] = l_j'(x_i)

    This is used only on the small number of face nodes.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size

    if n < 2:
        raise ValueError("Need at least two 1D nodes.")

    w = np.ones(n, dtype=float)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (x[j] - x[k])

    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / w[i] / (x[i] - x[j])

    D[np.diag_indices(n)] = -np.sum(D, axis=1)

    return D


def reference_face_parameters(
    rs_nodes: np.ndarray,
    ref_face,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return local edge parameter t in [0,1] for each reference face.

    Must match face ordering used in build_reference_face_data:

        face 0: v2 -> v3, t = (s + 1)/2
        face 1: v3 -> v1, t = (1 - s)/2
        face 2: v1 -> v2, t = (r + 1)/2
    """
    rs_nodes = np.asarray(rs_nodes, dtype=float)

    r = rs_nodes[:, 0]
    s = rs_nodes[:, 1]

    ts = []

    for f in range(3):
        ids = np.asarray(ref_face.face_node_ids[f], dtype=int)

        if f == 0:
            t = 0.5 * (s[ids] + 1.0)
        elif f == 1:
            t = 0.5 * (1.0 - s[ids])
        elif f == 2:
            t = 0.5 * (r[ids] + 1.0)
        else:
            raise ValueError("Invalid face id.")

        ts.append(np.asarray(t, dtype=float))

    return tuple(ts)


def build_reference_face_derivative_matrices(
    rs_nodes: np.ndarray,
    ref_face,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build D_t matrices along each oriented reference face.
    """
    ts = reference_face_parameters(rs_nodes, ref_face)
    return tuple(lagrange_derivative_matrix_1d(t) for t in ts)


def edge_streamfunction_normal_flux(
    psi: np.ndarray,
    ref_face,
    conn,
    face_Dt: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    r"""
    Compute face normal area flux from streamfunction trace.

    For the flattened coordinates:

        F = (-psi_y, psi_x)

    Along an oriented boundary edge with unit tangent t and outward normal n,
    using the same CCW local face orientation as the mesh,

        F dot n = - d_s psi

    where s is physical arclength along the oriented local face.

    If t in [0,1] parametrizes the edge, ds = length * dt, hence:

        a = F dot n = -(1/length) D_t psi.

    Returns
    -------
    a:
        shape (K, 3, Nfp), local outward normal flux on each element face.
    """
    psi = np.asarray(psi, dtype=float)

    if psi.ndim != 2:
        raise ValueError("psi must have shape (K, Np).")

    K, _Np = psi.shape
    nfp = ref_face.nfp

    a = np.zeros((K, 3, nfp), dtype=float)

    for k in range(K):
        for f in range(3):
            ids = np.asarray(ref_face.face_node_ids[f], dtype=int)
            psi_f = psi[k, ids]
            Dt = face_Dt[f]

            if Dt.shape != (nfp, nfp):
                raise ValueError("Face derivative matrix shape mismatch.")

            a[k, f, :] = -(Dt @ psi_f) / conn.face_length[k, f]

    return a


def collect_pairwise_strong_penalty_from_edge_flux(
    q: np.ndarray,
    a_face: np.ndarray,
    ref_face,
    conn,
    *,
    tau: float = 0.0,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
) -> np.ndarray:
    r"""
    Assemble p = f - fhat pairwise from edge-based normal flux.

    Interior face:
        owner normal orientation:
            aM = a_face[k,f]
            aP_owner = -a_face[kp,fp] reordered to owner nodes

        fhat = 0.5(aM qM + aP qP)
               -0.5(1-tau) max(|aM|, |aP|)(qP - qM)

        pM = aM qM - fhat
        pP = aP_local qP - fhat_local

    The neighbor receives the corresponding local p values.

    This preserves constant state if the edge flux is single-valued.
    """
    q = np.asarray(q, dtype=float)
    a_face = np.asarray(a_face, dtype=float)

    if q.ndim != 2:
        raise ValueError("q must have shape (K, Np).")
    if a_face.ndim != 3 or a_face.shape[0] != q.shape[0] or a_face.shape[1] != 3:
        raise ValueError("a_face must have shape (K, 3, Nfp).")

    mode = str(boundary_mode).strip().lower()
    if mode not in ("same_state", "constant"):
        raise ValueError("boundary_mode must be 'same_state' or 'constant'.")

    K, _Np = q.shape
    nfp = ref_face.nfp
    tau = float(tau)

    p = np.zeros((K, 3, nfp), dtype=float)

    for k in range(K):
        for f in range(3):
            kp = conn.neighbor_elem[k, f]
            fp = conn.neighbor_face[k, f]

            idsM = np.asarray(ref_face.face_node_ids[f], dtype=int)

            qM = q[k, idsM]
            aM = a_face[k, f, :]

            if kp < 0:
                if mode == "same_state":
                    qP = qM
                    aP_owner = aM
                else:
                    qP = np.full_like(qM, float(q_boundary_value))
                    aP_owner = aM

                alpha = np.maximum(np.abs(aM), np.abs(aP_owner))
                fhat = (
                    0.5 * (aM * qM + aP_owner * qP)
                    - 0.5 * (1.0 - tau) * alpha * (qP - qM)
                )

                p[k, f, :] = aM * qM - fhat
                continue

            # Count each interior/seam face only once.
            if (kp < k) or (kp == k and fp < f):
                continue

            idsP = np.asarray(conn.neighbor_node_ids[k, f, :], dtype=int)
            neighbor_face_ids = np.asarray(ref_face.face_node_ids[fp], dtype=int)

            reorder = []
            for nid in idsP:
                loc = np.nonzero(neighbor_face_ids == nid)[0]
                if loc.size != 1:
                    raise ValueError("Failed to locate matched neighbor face node.")
                reorder.append(int(loc[0]))
            reorder = np.asarray(reorder, dtype=int)

            qP = q[kp, idsP]

            # Neighbor local outward flux, reordered to owner-node order.
            aP_local_reordered = a_face[kp, fp, reorder]

            # Express plus-side flux using owner normal orientation.
            aP_owner = -aP_local_reordered

            alpha = np.maximum(np.abs(aM), np.abs(aP_owner))

            fhat_owner = (
                0.5 * (aM * qM + aP_owner * qP)
                - 0.5 * (1.0 - tau) * alpha * (qP - qM)
            )

            # Owner penalty.
            p[k, f, :] = aM * qM - fhat_owner

            # Neighbor local fhat is negative of owner fhat.
            p_neighbor_reordered = (
                aP_local_reordered * qP
                - (-fhat_owner)
            )

            # Scatter back to neighbor face-node order.
            for i, loc in enumerate(reorder):
                p[kp, fp, int(loc)] = p_neighbor_reordered[i]

    return p


def sdg_surface_edge_streamfunction_strong_rhs_projected(
    q: np.ndarray,
    psi: np.ndarray,
    rule: dict,
    rs_nodes: np.ndarray,
    ref_face,
    conn,
    *,
    N: int,
    J_area: float,
    tau: float = 0.0,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
    surface_inverse_mass_T: np.ndarray | None = None,
    face_Dt: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Projected surface RHS using edge streamfunction flux.

    Returns:
        rhs_surface, p, a_face
    """
    if J_area == 0.0:
        raise ValueError("J_area must be nonzero.")

    if face_Dt is None:
        face_Dt = build_reference_face_derivative_matrices(rs_nodes, ref_face)

    if surface_inverse_mass_T is None:
        surface_inverse_mass_T = build_projected_surface_inverse_mass_T(
            N=N,
            rule=rule,
        )

    a_face = edge_streamfunction_normal_flux(
        psi,
        ref_face,
        conn,
        face_Dt,
    )

    p = collect_pairwise_strong_penalty_from_edge_flux(
        q,
        a_face,
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

    from operators.sdg_projected_surface_flux import build_face_weight_array_from_rule

    face_weights = build_face_weight_array_from_rule(rule, ref_face)
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

    return flat_surface_rhs / float(J_area), p, a_face


def edge_flux_pair_error(
    a_face: np.ndarray,
    ref_face,
    conn,
) -> dict[str, float | int]:
    """
    Check whether local edge streamfunction flux is pairwise opposite.
    """
    max_err = 0.0
    sum_l2 = 0.0
    n_pairs = 0

    K = a_face.shape[0]

    for k in range(K):
        for f in range(3):
            kp = conn.neighbor_elem[k, f]
            fp = conn.neighbor_face[k, f]

            if kp < 0:
                continue
            if (kp < k) or (kp == k and fp < f):
                continue

            idsP = np.asarray(conn.neighbor_node_ids[k, f, :], dtype=int)
            neighbor_face_ids = np.asarray(ref_face.face_node_ids[fp], dtype=int)

            reorder = []
            for nid in idsP:
                loc = np.nonzero(neighbor_face_ids == nid)[0]
                if loc.size != 1:
                    raise ValueError("Failed to locate matched neighbor face node.")
                reorder.append(int(loc[0]))
            reorder = np.asarray(reorder, dtype=int)

            err = a_face[k, f, :] + a_face[kp, fp, reorder]

            max_err = max(max_err, float(np.max(np.abs(err))))
            sum_l2 += float(np.sum(err * err))
            n_pairs += 1

    return {
        "edge_flux_pair_max_error": float(max_err),
        "edge_flux_pair_l2_unweighted": float(np.sqrt(max(sum_l2, 0.0))),
        "n_pairs": int(n_pairs),
    }
