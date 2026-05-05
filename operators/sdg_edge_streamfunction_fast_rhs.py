from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap


from operators.sdg_edge_streamfunction_closed_sphere_rhs import (
    EdgeStreamfunctionClosedSphereSDGOperator,
    build_edge_streamfunction_closed_sphere_sdg_operator,
)
from operators.sdg_conservative_rhs import sdg_conservative_volume_rhs
from operators.sdg_projected_surface_flux import build_face_weight_array_from_rule


@njit(cache=True)
def _edge_streamfunction_normal_flux_kernel(
    psi: np.ndarray,
    face_node_ids: np.ndarray,
    face_Dt: np.ndarray,
    face_length: np.ndarray,
) -> np.ndarray:
    """
    Compute a = F dot n = - d_s psi on all faces.

    Shapes
    ------
    psi           : (K, Np)
    face_node_ids : (3, Nfp)
    face_Dt       : (3, Nfp, Nfp)
    face_length   : (K, 3)
    """
    K = psi.shape[0]
    Nfp = face_node_ids.shape[1]

    a_face = np.zeros((K, 3, Nfp), dtype=np.float64)

    for k in range(K):
        for f in range(3):
            length = face_length[k, f]
            for i in range(Nfp):
                val = 0.0
                for j in range(Nfp):
                    nid = face_node_ids[f, j]
                    val += face_Dt[f, i, j] * psi[k, nid]
                a_face[k, f, i] = -val / length

    return a_face


@njit(cache=True)
def _pairwise_strong_penalty_kernel(
    q: np.ndarray,
    a_face: np.ndarray,
    face_node_ids: np.ndarray,
    neighbor_elem: np.ndarray,
    neighbor_face: np.ndarray,
    neighbor_node_ids: np.ndarray,
    neighbor_face_locs: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Assemble p = f - fhat using pairwise common flux.

    This kernel assumes boundary faces, if any, use same_state boundary mode.
    For closed sphere seam connectivity, no unmatched boundary faces should remain.
    """
    K = q.shape[0]
    Nfp = face_node_ids.shape[1]

    p = np.zeros((K, 3, Nfp), dtype=np.float64)

    for k in range(K):
        for f in range(3):
            kp = neighbor_elem[k, f]
            fp = neighbor_face[k, f]

            # Boundary fallback: same_state => qP=qM, aP=aM.
            if kp < 0:
                for i in range(Nfp):
                    nidM = face_node_ids[f, i]
                    qM = q[k, nidM]
                    aM = a_face[k, f, i]
                    qP = qM
                    aP = aM
                    alpha = abs(aM)
                    if abs(aP) > alpha:
                        alpha = abs(aP)

                    fhat = 0.5 * (aM * qM + aP * qP) - 0.5 * (1.0 - tau) * alpha * (qP - qM)
                    p[k, f, i] = aM * qM - fhat
                continue

            # Count each interior/seam face only once.
            if (kp < k) or ((kp == k) and (fp < f)):
                continue

            for i in range(Nfp):
                nidM = face_node_ids[f, i]
                nidP = neighbor_node_ids[k, f, i]
                locP = neighbor_face_locs[k, f, i]

                qM = q[k, nidM]
                qP = q[kp, nidP]

                aM = a_face[k, f, i]
                aP_local = a_face[kp, fp, locP]

                # plus-side normal flux using owner normal orientation
                aP_owner = -aP_local

                alpha = abs(aM)
                if abs(aP_owner) > alpha:
                    alpha = abs(aP_owner)

                fhat_owner = (
                    0.5 * (aM * qM + aP_owner * qP)
                    - 0.5 * (1.0 - tau) * alpha * (qP - qM)
                )

                # Owner-side penalty.
                p[k, f, i] = aM * qM - fhat_owner

                # Neighbor-side local fhat is -fhat_owner.
                p_neighbor = aP_local * qP - (-fhat_owner)
                p[kp, fp, locP] = p_neighbor

    return p


@njit(cache=True)
def _surface_integral_accumulate_kernel(
    p: np.ndarray,
    face_node_ids: np.ndarray,
    face_weights: np.ndarray,
    face_length: np.ndarray,
    Np: int,
) -> np.ndarray:
    """
    Accumulate:

        b_j = sum_faces int_face p phi_j ds

    Since Table1 face nodes are embedded volume nodes, this is sparse.
    """
    K = p.shape[0]
    Nfp = p.shape[2]

    out = np.zeros((K, Np), dtype=np.float64)

    for k in range(K):
        for f in range(3):
            length = face_length[k, f]
            for j in range(Nfp):
                nid = face_node_ids[f, j]
                out[k, nid] += length * face_weights[f, j] * p[k, f, j]

    return out


@njit(cache=True)
def _global_mass_residual_kernel(
    rhs: np.ndarray,
    weights_surface: np.ndarray,
    bad_mask: np.ndarray,
) -> float:
    K = rhs.shape[0]
    Np = rhs.shape[1]

    total = 0.0
    for k in range(K):
        for i in range(Np):
            if not bad_mask[k, i]:
                total += weights_surface[k, i] * rhs[k, i]

    return total


@njit(cache=True)
def _area_from_weights_kernel(
    weights_surface: np.ndarray,
    bad_mask: np.ndarray,
) -> float:
    K = weights_surface.shape[0]
    Np = weights_surface.shape[1]

    total = 0.0
    for k in range(K):
        for i in range(Np):
            if not bad_mask[k, i]:
                total += weights_surface[k, i]

    return total


@njit(cache=True)
def _apply_global_correction_kernel(
    rhs: np.ndarray,
    bad_mask: np.ndarray,
    c: float,
) -> np.ndarray:
    K = rhs.shape[0]
    Np = rhs.shape[1]

    out = np.empty_like(rhs)

    for k in range(K):
        for i in range(Np):
            if bad_mask[k, i]:
                out[k, i] = np.nan
            else:
                out[k, i] = rhs[k, i] + c

    return out


def _as_face_node_ids_array(ref_face) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(
            [np.asarray(ref_face.face_node_ids[f], dtype=np.int64) for f in range(3)],
            dtype=np.int64,
        )
    )


def _build_neighbor_face_locs(ref_face, conn) -> np.ndarray:
    """
    For each owner face node i, store the local index on the neighbor face.

        neighbor_face_locs[k,f,i] = locP

    where:

        ref_face.face_node_ids[fp][locP] == conn.neighbor_node_ids[k,f,i]
    """
    neighbor_elem = np.asarray(conn.neighbor_elem, dtype=np.int64)
    neighbor_face = np.asarray(conn.neighbor_face, dtype=np.int64)
    neighbor_node_ids = np.asarray(conn.neighbor_node_ids, dtype=np.int64)

    K = neighbor_elem.shape[0]
    Nfp = neighbor_node_ids.shape[2]

    locs = -np.ones((K, 3, Nfp), dtype=np.int64)

    for k in range(K):
        for f in range(3):
            fp = neighbor_face[k, f]
            if fp < 0:
                continue

            neighbor_face_ids = np.asarray(ref_face.face_node_ids[fp], dtype=np.int64)

            for i in range(Nfp):
                nid = neighbor_node_ids[k, f, i]
                found = np.nonzero(neighbor_face_ids == nid)[0]
                if found.size != 1:
                    raise ValueError("Failed to locate matched neighbor face node.")
                locs[k, f, i] = int(found[0])

    return np.ascontiguousarray(locs, dtype=np.int64)


def build_fast_edge_rhs_cache(op: EdgeStreamfunctionClosedSphereSDGOperator) -> dict:
    """
    Build static arrays for the accelerated edge-streamfunction RHS.

    This should be called once after building the operator.
    """
    base = op.base

    face_node_ids = _as_face_node_ids_array(base.ref_face)

    face_Dt = np.ascontiguousarray(
        np.asarray(op.face_Dt, dtype=np.float64),
        dtype=np.float64,
    )

    face_weights = np.ascontiguousarray(
        build_face_weight_array_from_rule(base.rule, base.ref_face),
        dtype=np.float64,
    )

    neighbor_elem = np.ascontiguousarray(
        np.asarray(base.conn_seam.neighbor_elem, dtype=np.int64),
        dtype=np.int64,
    )
    neighbor_face = np.ascontiguousarray(
        np.asarray(base.conn_seam.neighbor_face, dtype=np.int64),
        dtype=np.int64,
    )
    neighbor_node_ids = np.ascontiguousarray(
        np.asarray(base.conn_seam.neighbor_node_ids, dtype=np.int64),
        dtype=np.int64,
    )
    neighbor_face_locs = _build_neighbor_face_locs(base.ref_face, base.conn_seam)

    face_length = np.ascontiguousarray(
        np.asarray(base.conn_seam.face_length, dtype=np.float64),
        dtype=np.float64,
    )

    area_flat = np.ascontiguousarray(
        np.asarray(base.conn_seam.area_flat, dtype=np.float64),
        dtype=np.float64,
    )

    weights_surface = np.ascontiguousarray(
        np.asarray(base.weights_surface, dtype=np.float64),
        dtype=np.float64,
    )

    bad_mask = np.ascontiguousarray(
        np.asarray(base.cache.bad_mask, dtype=np.bool_),
        dtype=np.bool_,
    )

    psi = np.ascontiguousarray(
        np.asarray(base.psi, dtype=np.float64),
        dtype=np.float64,
    )

    a_face = _edge_streamfunction_normal_flux_kernel(
        psi,
        face_node_ids,
        face_Dt,
        face_length,
    )

    total_area = _area_from_weights_kernel(weights_surface, bad_mask)

    return {
        "face_node_ids": face_node_ids,
        "face_Dt": face_Dt,
        "face_weights": face_weights,
        "neighbor_elem": neighbor_elem,
        "neighbor_face": neighbor_face,
        "neighbor_node_ids": neighbor_node_ids,
        "neighbor_face_locs": neighbor_face_locs,
        "face_length": face_length,
        "area_flat": area_flat,
        "weights_surface": weights_surface,
        "bad_mask": bad_mask,
        "psi": psi,
        "a_face": a_face,
        "total_area": float(total_area),
        "Np": int(base.Fx_area.shape[1]),
    }


def edge_streamfunction_closed_sphere_rhs_global_corrected_fast(
    q: np.ndarray,
    op: EdgeStreamfunctionClosedSphereSDGOperator,
    *,
    fast_cache: dict | None = None,
    return_info: bool = False,
):
    """
    Fast RHS for the verified formulation:

        edge-streamfunction strong correction + global mass correction.

    Surface part is accelerated with Numba kernels.
    Volume part still uses the existing vectorized conservative divergence.
    """
    base = op.base

    q = np.ascontiguousarray(np.asarray(q, dtype=np.float64), dtype=np.float64)

    if q.shape != base.Fx_area.shape:
        raise ValueError(
            f"q shape {q.shape} does not match operator shape {base.Fx_area.shape}."
        )

    if fast_cache is None:
        fast_cache = build_fast_edge_rhs_cache(op)

    rhs_volume = sdg_conservative_volume_rhs(
        q,
        base.Fx_area,
        base.Fy_area,
        base.Dr,
        base.Ds,
        base.geom["rx"],
        base.geom["sx"],
        base.geom["ry"],
        base.geom["sy"],
        J_area=base.J_area,
        mask=base.cache.bad_mask,
    )

    p = _pairwise_strong_penalty_kernel(
        q,
        fast_cache["a_face"],
        fast_cache["face_node_ids"],
        fast_cache["neighbor_elem"],
        fast_cache["neighbor_face"],
        fast_cache["neighbor_node_ids"],
        fast_cache["neighbor_face_locs"],
        float(base.tau),
    )

    surface_integral = _surface_integral_accumulate_kernel(
        p,
        fast_cache["face_node_ids"],
        fast_cache["face_weights"],
        fast_cache["face_length"],
        int(fast_cache["Np"]),
    )

    rhs_surface = surface_integral @ base.MinvT
    rhs_surface /= fast_cache["area_flat"][:, None]
    rhs_surface /= float(base.J_area)

    rhs_raw = rhs_volume + rhs_surface

    I_raw = _global_mass_residual_kernel(
        np.ascontiguousarray(rhs_raw, dtype=np.float64),
        fast_cache["weights_surface"],
        fast_cache["bad_mask"],
    )

    c = -I_raw / float(fast_cache["total_area"])

    rhs_corr = _apply_global_correction_kernel(
        np.ascontiguousarray(rhs_raw, dtype=np.float64),
        fast_cache["bad_mask"],
        c,
    )

    if not return_info:
        return rhs_corr

    I_corr = _global_mass_residual_kernel(
        np.ascontiguousarray(rhs_corr, dtype=np.float64),
        fast_cache["weights_surface"],
        fast_cache["bad_mask"],
    )

    info = {
        "global_mass_residual_before": float(I_raw),
        "global_mass_residual_after": float(I_corr),
        "global_correction_constant": float(c),
        "total_area": float(fast_cache["total_area"]),
        "numba_available": bool(_NUMBA_AVAILABLE),
    }

    return rhs_corr, info


def build_fast_operator(
    *,
    nsub: int,
    order: int = 4,
    N: int | None = None,
    R: float = 1.0,
    u0: float = 1.0,
    alpha0: float = math.pi / 4.0,
    tau: float = 0.0,
    seam_tol: float = 1.0e-10,
):
    return build_edge_streamfunction_closed_sphere_sdg_operator(
        nsub=nsub,
        order=order,
        N=N,
        R=R,
        u0=u0,
        alpha0=alpha0,
        tau=tau,
        seam_tol=seam_tol,
    )
