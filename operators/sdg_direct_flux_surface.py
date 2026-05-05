from __future__ import annotations

import numpy as np

from operators.rhs_split_conservative_exchange import numerical_flux_penalty
from operators.rhs_split_conservative_exact_trace import _surface_lift_exact_trace
from operators.sdg_projected_surface_flux import (
    build_face_weight_array_from_rule,
    build_projected_surface_inverse_mass_T,
)


def collect_direct_numerical_flux_array(
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
    Collect oriented numerical flux fhat on every owner face.

    Shape:
        fhat.shape = (K, 3, Nfp)

    Let:

        fM = ndotF * qM

    and repo penalty convention:

        p = fM - fhat

    Therefore:

        fhat = fM - p

    where p is computed by the repo's numerical_flux_penalty.

    For an interior face, the neighbor element sees the opposite normal, so
    the corresponding fhat should be equal and opposite up to roundoff.
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

    fhat = np.zeros((K, 3, nfp), dtype=float)

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

            penalty = numerical_flux_penalty(
                ndotF,
                qM,
                qP,
                tau=tau,
            )

            fM = ndotF * qM
            fhat[k, f, :] = fM - penalty

    return fhat



def collect_pairwise_direct_numerical_flux_array(
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
    Collect oriented numerical flux fhat using single-edge pairwise assembly.

    For an interior face, compute fhat only once with the owner-side outward
    normal, then assign opposite signs to the two neighboring elements:

        fhat^- =  fhat
        fhat^+ = -fhat

    This enforces pairwise conservation algebraically:

        fhat^- + fhat^+ = 0

    For variable or discontinuous one-sided flux traces, use:

        aM = nM dot F_M
        aP = nM dot F_P = -(nP dot F_P)

        fhat = 0.5 * (aM qM + aP qP)
               -0.5 * (1 - tau) * max(|aM|, |aP|) * (qP - qM)

    Boundary mode:
        same_state:
            qP = qM and aP = aM, hence fhat = aM qM.
        constant:
            qP = q_boundary_value and aP = aM.
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

    fhat = np.zeros((K, 3, nfp), dtype=float)

    tau = float(tau)

    for k in range(K):
        for f in range(3):
            kp = conn.neighbor_elem[k, f]
            fp = conn.neighbor_face[k, f]

            # Boundary face: no paired neighbor.
            if kp < 0:
                idsM = np.asarray(ref_face.face_node_ids[f], dtype=int)

                qM = q[k, idsM]
                FxM = Fx_area[k, idsM]
                FyM = Fy_area[k, idsM]

                aM = (
                    conn.face_normal_x[k, f] * FxM
                    + conn.face_normal_y[k, f] * FyM
                )

                if mode == "same_state":
                    qP = qM
                else:
                    qP = np.full_like(qM, float(q_boundary_value))

                aP = aM
                alpha = np.maximum(np.abs(aM), np.abs(aP))

                fhat[k, f, :] = (
                    0.5 * (aM * qM + aP * qP)
                    - 0.5 * (1.0 - tau) * alpha * (qP - qM)
                )
                continue

            # Count each interior face only once.
            if (kp < k) or (kp == k and fp < f):
                continue

            idsM = np.asarray(ref_face.face_node_ids[f], dtype=int)
            idsP = np.asarray(conn.neighbor_node_ids[k, f, :], dtype=int)

            qM = q[k, idsM]
            qP = q[kp, idsP]

            FxM = Fx_area[k, idsM]
            FyM = Fy_area[k, idsM]

            FxP = Fx_area[kp, idsP]
            FyP = Fy_area[kp, idsP]

            aM = (
                conn.face_normal_x[k, f] * FxM
                + conn.face_normal_y[k, f] * FyM
            )

            # Neighbor outward normal is opposite to owner normal geometrically.
            # Use owner-normal orientation for the plus-side flux.
            aP_owner = -(
                conn.face_normal_x[kp, fp] * FxP
                + conn.face_normal_y[kp, fp] * FyP
            )

            alpha = np.maximum(np.abs(aM), np.abs(aP_owner))

            fhat_owner = (
                0.5 * (aM * qM + aP_owner * qP)
                - 0.5 * (1.0 - tau) * alpha * (qP - qM)
            )

            # Assign to owner side.
            fhat[k, f, :] = fhat_owner

            # Assign opposite flux to neighbor side in neighbor face-node order.
            neighbor_face_ids = np.asarray(ref_face.face_node_ids[fp], dtype=int)

            for i, nid in enumerate(idsP):
                loc = np.nonzero(neighbor_face_ids == nid)[0]
                if loc.size != 1:
                    raise ValueError(
                        "Failed to locate matched neighbor face node during pairwise flux assembly."
                    )
                fhat[kp, fp, int(loc[0])] = -fhat_owner[i]

    return fhat


def sdg_surface_direct_flux_rhs_repo_lift(
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
    Direct numerical flux surface RHS.

    PDE form:

        d_t(J q) + div(F_area q) = 0

    Direct flux surface contribution:

        R_surf = - 1/J * M^{-1} int_face fhat phi ds

    The repo lifting routine computes:

        lift(p) = M^{-1} int_face p phi ds

    so we pass:

        p = -fhat

    and then divide by J_area.
    """
    if J_area == 0.0:
        raise ValueError("J_area must be nonzero.")

    fhat = collect_pairwise_direct_numerical_flux_array(
        q,
        Fx_area,
        Fy_area,
        ref_face,
        conn,
        tau=tau,
        boundary_mode=boundary_mode,
        q_boundary_value=q_boundary_value,
    )

    p_direct = -fhat

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
        p=p_direct,
        face_node_ids=face_node_ids,
        face_weights=face_weights,
        length=np.asarray(conn.face_length, dtype=float),
        area=np.asarray(conn.area_flat, dtype=float),
        ws=ws,
        surface_inverse_mass_t=surface_inverse_mass_T,
        use_numba=False,
    )

    return flat_surface_rhs / float(J_area)


def sdg_surface_direct_flux_rhs_projected(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    rule: dict,
    ref_face,
    conn,
    *,
    N: int,
    J_area: float,
    tau: float = 0.0,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
    surface_inverse_mass_T: np.ndarray | None = None,
) -> np.ndarray:
    """
    Projected inverse-mass version of direct numerical flux surface RHS.
    """
    if surface_inverse_mass_T is None:
        surface_inverse_mass_T = build_projected_surface_inverse_mass_T(
            N=N,
            rule=rule,
        )

    return sdg_surface_direct_flux_rhs_repo_lift(
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


def sdg_surface_direct_flux_rhs_diagonal(
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
    Diagonal inverse-mass version of direct numerical flux surface RHS.
    """
    return sdg_surface_direct_flux_rhs_repo_lift(
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


def internal_direct_flux_pair_balance(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    ref_face,
    conn,
    *,
    tau: float = 0.0,
) -> dict[str, float]:
    r"""
    Diagnostic for pairwise conservation of oriented fhat.

    For every interior face pair, owner-side fhat plus neighbor-side fhat
    should be zero at matched face nodes.

    This does not include lifting, only the raw oriented numerical flux.
    """
    fhat = collect_pairwise_direct_numerical_flux_array(
        q,
        Fx_area,
        Fy_area,
        ref_face,
        conn,
        tau=tau,
        boundary_mode="same_state",
    )

    max_pair_error = 0.0
    sum_pair_l2 = 0.0
    n_pairs = 0

    K = q.shape[0]

    for k in range(K):
        for f in range(3):
            kp = conn.neighbor_elem[k, f]
            fp = conn.neighbor_face[k, f]

            if kp < 0:
                continue

            # Count each interior face once.
            if (kp < k) or (kp == k and fp < f):
                continue

            idsP_on_neighbor = conn.neighbor_node_ids[k, f, :]

            # Need to reorder neighbor fhat values by matched node ids.
            # fhat[kp, fp, :] is stored in neighbor's own face-node order.
            neighbor_face_ids = np.asarray(ref_face.face_node_ids[fp], dtype=int)

            reorder = []
            for nid in idsP_on_neighbor:
                loc = np.nonzero(neighbor_face_ids == nid)[0]
                if loc.size != 1:
                    raise ValueError("Failed to locate matched neighbor face node.")
                reorder.append(int(loc[0]))

            reorder = np.asarray(reorder, dtype=int)

            err = fhat[k, f, :] + fhat[kp, fp, reorder]

            max_pair_error = max(max_pair_error, float(np.max(np.abs(err))))
            sum_pair_l2 += float(np.sum(err * err))
            n_pairs += 1

    return {
        "max_pair_error": float(max_pair_error),
        "pair_l2_unweighted": float(np.sqrt(max(sum_pair_l2, 0.0))),
        "n_pairs": int(n_pairs),
    }

