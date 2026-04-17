from __future__ import annotations

import importlib
import numpy as np

try:
    njit = importlib.import_module("numba").njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    _NUMBA_AVAILABLE = False

from operators.divergence_split import mapped_divergence_split_2d
from operators.divergence_split import build_mapped_divergence_split_cache_2d
from operators.exchange import (
    pair_face_traces,
    interior_face_pair_mismatches,
)
from geometry.reference_triangle import reference_triangle_area


def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


def _as_c_f64(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.float64 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _resolve_surface_backend(
    surface_backend: str | None,
    *,
    use_numba: bool | None,
    compute_mismatches: bool,
    return_diagnostics: bool,
) -> str:
    name = "auto" if surface_backend is None else str(surface_backend).lower().strip()
    perf_mode = (not compute_mismatches) and (not return_diagnostics)

    if name in ("legacy", "face-major"):
        return name

    if name == "auto":
        # In pure performance mode, face-major has the fastest numba path
        # (fused qP + penalty + lift kernel). Keep legacy for diagnostic mode.
        if _should_use_numba(use_numba):
            if perf_mode:
                return "face-major"
            return "legacy"

        # Without numba, prefer face-major only in pure performance mode.
        if perf_mode:
            return "face-major"
        return "legacy"

    raise ValueError("surface_backend must be 'auto', 'legacy', or 'face-major'.")


def _resolve_face_order_mode(face_order_mode: str | None) -> str:
    mode = "triangle" if face_order_mode is None else str(face_order_mode).strip().lower()
    if mode not in ("triangle", "simplex", "simplex_strict"):
        raise ValueError("face_order_mode must be one of: 'triangle', 'simplex', 'simplex_strict'.")
    return mode


def _is_simplex_like_face_mode(face_order_mode: str) -> bool:
    mode = _resolve_face_order_mode(face_order_mode)
    return mode in ("simplex", "simplex_strict")


def _canonical_face_axis_mode(face_order_mode: str) -> str:
    return "simplex" if _is_simplex_like_face_mode(face_order_mode) else "triangle"


def _resolve_surface_lift_scale(
    *,
    face_order_mode: str,
    surface_inverse_mass_T: np.ndarray | None,
) -> float:
    mode = _resolve_face_order_mode(face_order_mode)
    if mode != "simplex_strict":
        return 1.0
    if surface_inverse_mass_T is None:
        raise ValueError(
            "face_order_mode='simplex_strict' requires surface_inverse_mass_T (projected inverse-mass lifting)."
        )
    return float(reference_triangle_area())


def _face_axis_permutations(face_order_mode: str) -> tuple[np.ndarray, np.ndarray]:
    mode = _resolve_face_order_mode(face_order_mode)
    if _is_simplex_like_face_mode(mode):
        # Triangle trace order B=[v2v3,v3v1,v1v2] -> Simplex order A=[v1v2,v2v3,v3v1].
        face_perm_new_to_old = np.asarray([2, 0, 1], dtype=np.int64)
    else:
        face_perm_new_to_old = np.asarray([0, 1, 2], dtype=np.int64)

    face_perm_old_to_new = np.empty(3, dtype=np.int64)
    face_perm_old_to_new[face_perm_new_to_old] = np.arange(3, dtype=np.int64)
    return face_perm_new_to_old, face_perm_old_to_new


def _build_flat_face_index_maps(
    K: int,
    face_perm_new_to_old: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nf = 3 * int(K)
    flat_new_to_old = np.empty(nf, dtype=np.int64)
    flat_old_to_new = np.empty(nf, dtype=np.int64)

    for k in range(int(K)):
        base = 3 * k
        for jf_new in range(3):
            idx_new = base + jf_new
            idx_old = base + int(face_perm_new_to_old[jf_new])
            flat_new_to_old[idx_new] = idx_old
            flat_old_to_new[idx_old] = idx_new

    return flat_new_to_old, flat_old_to_new


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _surface_rhs_kernel_inplace(
        p: np.ndarray,
        area: np.ndarray,
        length: np.ndarray,
        ids_f1: np.ndarray,
        ids_f2: np.ndarray,
        ids_f3: np.ndarray,
        wr_f1: np.ndarray,
        wr_f2: np.ndarray,
        wr_f3: np.ndarray,
        surface_rhs: np.ndarray,
    ) -> None:
        K = p.shape[0]
        nfp = p.shape[2]

        for k in range(K):
            s1 = length[k, 0] / area[k]
            for i in range(nfp):
                surface_rhs[k, ids_f1[i]] += s1 * wr_f1[i] * p[k, 0, i]

            s2 = length[k, 1] / area[k]
            for i in range(nfp):
                surface_rhs[k, ids_f2[i]] += s2 * wr_f2[i] * p[k, 1, i]

            s3 = length[k, 2] / area[k]
            for i in range(nfp):
                surface_rhs[k, ids_f3[i]] += s3 * wr_f3[i] * p[k, 2, i]


    @njit(cache=True)
    def _face_major_surface_rhs_flat_kernel_inplace(
        q_elem: np.ndarray,
        ndotV_flat: np.ndarray,
        qB_flat: np.ndarray,
        tau_interior: float,
        tau_boundary: float,
        owner_elem_flat: np.ndarray,
        owner_node_ids_flat: np.ndarray,
        owner_wratio_flat: np.ndarray,
        length_over_area_flat: np.ndarray,
        boundary_flat: np.ndarray,
        nbr_elem_flat: np.ndarray,
        nbr_node_ids_flat: np.ndarray,
        surface_rhs: np.ndarray,
    ) -> None:
        nf = ndotV_flat.shape[0]
        nfp = ndotV_flat.shape[1]

        for face_idx in range(nf):
            k = owner_elem_flat[face_idx]
            scale = length_over_area_flat[face_idx]
            is_boundary = boundary_flat[face_idx]
            nbr_k = nbr_elem_flat[face_idx]

            for i in range(nfp):
                m_id = owner_node_ids_flat[face_idx, i]
                qM = q_elem[k, m_id]

                if is_boundary:
                    qP = qB_flat[face_idx, i]
                    tau_face = tau_boundary
                else:
                    qP = q_elem[nbr_k, nbr_node_ids_flat[face_idx, i]]
                    tau_face = tau_interior

                ndv = ndotV_flat[face_idx, i]
                coeff = 0.5 * (ndv - (1.0 - tau_face) * abs(ndv))
                if coeff != 0.0:
                    surface_rhs[k, m_id] += (
                        scale
                        * owner_wratio_flat[face_idx, i]
                        * coeff
                        * (qM - qP)
                    )


    @njit(cache=True)
    def _face_major_surface_rhs_pair_kernel_inplace(
        q_elem: np.ndarray,
        ndotV_flat: np.ndarray,
        qB_flat: np.ndarray,
        tau_interior: float,
        tau_boundary: float,
        owner_elem_flat: np.ndarray,
        owner_node_ids_flat: np.ndarray,
        owner_wratio_flat: np.ndarray,
        length_over_area_flat: np.ndarray,
        boundary_face_idx: np.ndarray,
        pair_face_a: np.ndarray,
        pair_face_b: np.ndarray,
        nbr_node_ids_flat: np.ndarray,
        surface_rhs: np.ndarray,
    ) -> None:
        nfp = ndotV_flat.shape[1]

        for b in range(boundary_face_idx.size):
            face_idx = boundary_face_idx[b]
            k = owner_elem_flat[face_idx]
            scale = length_over_area_flat[face_idx]

            for i in range(nfp):
                m_id = owner_node_ids_flat[face_idx, i]
                qM = q_elem[k, m_id]
                qP = qB_flat[face_idx, i]
                ndv = ndotV_flat[face_idx, i]

                coeff = 0.5 * (ndv - (1.0 - tau_boundary) * abs(ndv))
                if coeff != 0.0:
                    surface_rhs[k, m_id] += (
                        scale
                        * owner_wratio_flat[face_idx, i]
                        * coeff
                        * (qM - qP)
                    )

        for p in range(pair_face_a.size):
            fa = pair_face_a[p]
            fb = pair_face_b[p]

            ka = owner_elem_flat[fa]
            kb = owner_elem_flat[fb]

            scale_a = length_over_area_flat[fa]
            scale_b = length_over_area_flat[fb]

            for i in range(nfp):
                a_mid = owner_node_ids_flat[fa, i]
                b_mid = owner_node_ids_flat[fb, i]

                a_nid = nbr_node_ids_flat[fa, i]
                b_nid = nbr_node_ids_flat[fb, i]

                qMa = q_elem[ka, a_mid]
                qPa = q_elem[kb, a_nid]
                ndv_a = ndotV_flat[fa, i]
                coeff_a = 0.5 * (ndv_a - (1.0 - tau_interior) * abs(ndv_a))
                if coeff_a != 0.0:
                    surface_rhs[ka, a_mid] += (
                        scale_a
                        * owner_wratio_flat[fa, i]
                        * coeff_a
                        * (qMa - qPa)
                    )

                qMb = q_elem[kb, b_mid]
                qPb = q_elem[ka, b_nid]
                ndv_b = ndotV_flat[fb, i]
                coeff_b = 0.5 * (ndv_b - (1.0 - tau_interior) * abs(ndv_b))
                if coeff_b != 0.0:
                    surface_rhs[kb, b_mid] += (
                        scale_b
                        * owner_wratio_flat[fb, i]
                        * coeff_b
                        * (qMb - qPb)
                    )


    @njit(cache=True)
    def _boundary_state_from_opposite_kernel_inplace(
        q_elem: np.ndarray,
        boundary_face_idx: np.ndarray,
        opposite_face_flat: np.ndarray,
        opposite_flip_flat: np.ndarray,
        owner_elem_flat: np.ndarray,
        owner_node_ids_flat: np.ndarray,
        qB_flat: np.ndarray,
    ) -> None:
        nfp = owner_node_ids_flat.shape[1]
        for b in range(boundary_face_idx.size):
            idx = boundary_face_idx[b]
            opp_idx = opposite_face_flat[idx]
            opp_elem = owner_elem_flat[opp_idx]
            flip = opposite_flip_flat[idx]

            if flip:
                for i in range(nfp):
                    src_i = nfp - 1 - i
                    qB_flat[idx, i] = q_elem[opp_elem, owner_node_ids_flat[opp_idx, src_i]]
            else:
                for i in range(nfp):
                    qB_flat[idx, i] = q_elem[opp_elem, owner_node_ids_flat[opp_idx, i]]


    @njit(cache=True)
    def _boundary_state_from_periodic_vmap_kernel_inplace(
        q_elem: np.ndarray,
        boundary_face_idx: np.ndarray,
        partner_elem_flat: np.ndarray,
        partner_node_ids_flat: np.ndarray,
        qB_flat: np.ndarray,
    ) -> None:
        nfp = partner_node_ids_flat.shape[1]
        for b in range(boundary_face_idx.size):
            idx = boundary_face_idx[b]
            for i in range(nfp):
                qB_flat[idx, i] = q_elem[
                    partner_elem_flat[idx, i],
                    partner_node_ids_flat[idx, i],
                ]


    @njit(cache=True)
    def _accumulate_projected_surface_integral_kernel_inplace(
        p: np.ndarray,
        owner_elem_flat: np.ndarray,
        owner_node_ids_flat: np.ndarray,
        face_weight_flat: np.ndarray,
        length_flat: np.ndarray,
        surface_integral: np.ndarray,
    ) -> None:
        nf = owner_elem_flat.size
        nfp = owner_node_ids_flat.shape[1]

        for face_idx in range(nf):
            k = owner_elem_flat[face_idx]
            jf = face_idx - 3 * k
            face_length = length_flat[face_idx]
            for i in range(nfp):
                node_id = owner_node_ids_flat[face_idx, i]
                surface_integral[k, node_id] += (
                    face_length * face_weight_flat[face_idx, i] * p[k, jf, i]
                )

else:
    _surface_rhs_kernel_inplace = None
    _face_major_surface_rhs_flat_kernel_inplace = None
    _face_major_surface_rhs_pair_kernel_inplace = None
    _boundary_state_from_opposite_kernel_inplace = None
    _boundary_state_from_periodic_vmap_kernel_inplace = None
    _accumulate_projected_surface_integral_kernel_inplace = None


def volume_term_split_conservative(
    q_elem: np.ndarray,
    u_elem: np.ndarray,
    v_elem: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict,
    use_numba: bool | None = None,
    split_cache: dict | None = None,
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
        use_numba=use_numba,
        split_cache=split_cache,
    )


def build_volume_split_cache(
    u_elem: np.ndarray,
    v_elem: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict,
) -> dict:
    """
    Build reusable split-divergence coefficients for fixed velocity/geometry.
    """
    return build_mapped_divergence_split_cache_2d(
        a=np.asarray(u_elem, dtype=float),
        b=np.asarray(v_elem, dtype=float),
        Dr=np.asarray(Dr, dtype=float),
        Ds=np.asarray(Ds, dtype=float),
        xr=np.asarray(geom["xr"], dtype=float),
        xs=np.asarray(geom["xs"], dtype=float),
        yr=np.asarray(geom["yr"], dtype=float),
        ys=np.asarray(geom["ys"], dtype=float),
        J=np.asarray(geom["J"], dtype=float),
    )


def resolve_effective_taus(
    *,
    tau: float = 0.0,
    tau_interior: float | None = None,
    tau_qb: float | None = None,
) -> tuple[float, float]:
    tau_shared = float(tau)
    tau_interior_eff = tau_shared if tau_interior is None else float(tau_interior)
    tau_qb_eff = tau_shared if tau_qb is None else float(tau_qb)

    if not np.isfinite(tau_shared):
        raise ValueError("tau must be finite.")
    if not np.isfinite(tau_interior_eff):
        raise ValueError("tau_interior must be finite when provided.")
    if not np.isfinite(tau_qb_eff):
        raise ValueError("tau_qb must be finite when provided.")

    return tau_interior_eff, tau_qb_eff


def build_face_tau_array(
    *,
    is_boundary: np.ndarray,
    face_shape: tuple[int, int, int],
    physical_boundary_mode: str,
    tau: float = 0.0,
    tau_interior: float | None = None,
    tau_qb: float | None = None,
) -> tuple[float, float, np.ndarray]:
    is_boundary = np.asarray(is_boundary, dtype=bool)
    if is_boundary.shape != face_shape[:2]:
        raise ValueError("is_boundary must have shape (K, 3) matching face_shape.")

    tau_interior_eff, tau_qb_eff = resolve_effective_taus(
        tau=tau,
        tau_interior=tau_interior,
        tau_qb=tau_qb,
    )
    tau_face = np.full(face_shape, tau_interior_eff, dtype=float)

    if str(physical_boundary_mode).strip().lower() == "exact_qb":
        tau_face[np.broadcast_to(is_boundary[:, :, None], face_shape)] = tau_qb_eff

    return tau_interior_eff, tau_qb_eff, tau_face


def _coerce_tau_argument(
    tau: float | np.ndarray,
    *,
    shape: tuple[int, ...],
) -> float | np.ndarray:
    tau_arr = np.asarray(tau, dtype=float)
    if tau_arr.ndim == 0:
        tau_scalar = float(tau_arr)
        if not np.isfinite(tau_scalar):
            raise ValueError("tau must be finite.")
        return tau_scalar
    if tau_arr.shape != shape:
        raise ValueError("tau array must have the same shape as ndotV, qM, qP.")
    if not np.all(np.isfinite(tau_arr)):
        raise ValueError("tau array must be finite.")
    return tau_arr


def upwind_flux_and_penalty(
    ndotV: np.ndarray,
    qM: np.ndarray,
    qP: np.ndarray,
    tau: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Numerical flux from the document:

        f   = (n · V) qM
        f*  = 1/2[(n·V)qM + (n·V)qP] + (1-tau)/2 |n·V| (qM - qP)
        p   = f - f*

    For pure upwind, use tau = 0.
    """
    ndotV = np.asarray(ndotV, dtype=float)
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float)
    tau_value = _coerce_tau_argument(tau, shape=ndotV.shape)

    f = ndotV * qM
    fstar = (
        0.5 * (ndotV * qM + ndotV * qP)
        + 0.5 * (1.0 - tau_value) * np.abs(ndotV) * (qM - qP)
    )
    p = f - fstar
    return f, fstar, p

def upwind_penalty_simplified(
    ndotV: np.ndarray,
    qM: np.ndarray,
    qP: np.ndarray,
) -> np.ndarray:
    """
    Simplified penalty term for pure upwind flux (tau = 0):

        p = f - f*
          = 0                      if n·V >= 0
          = (n·V) (qM - qP)       if n·V < 0

    Equivalently:
        p = min(n·V, 0) * (qM - qP)
    """
    ndotV = np.asarray(ndotV, dtype=float)
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float)
    if ndotV.shape != qM.shape or qM.shape != qP.shape:
        raise ValueError("ndotV, qM, qP must have the same shape.")

    return np.minimum(ndotV, 0.0) * (qM - qP)


def numerical_flux_penalty(
    ndotV: np.ndarray,
    qM: np.ndarray,
    qP: np.ndarray,
    tau: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Penalty term associated with the upwind-family numerical flux.

    For `tau=0`, this reduces to the pure-upwind shortcut

        p = min(n·V, 0) * (qM - qP).
    """
    ndotV = np.asarray(ndotV, dtype=float)
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float)

    if ndotV.shape != qM.shape or qM.shape != qP.shape:
        raise ValueError("ndotV, qM, qP must have the same shape.")

    tau_value = _coerce_tau_argument(tau, shape=ndotV.shape)
    if isinstance(tau_value, float) and tau_value == 0.0:
        return upwind_penalty_simplified(ndotV=ndotV, qM=qM, qP=qP)

    coeff = 0.5 * (ndotV - (1.0 - tau_value) * np.abs(ndotV))
    return coeff * (qM - qP)

def fill_boundary_exterior_state_upwind(
    qM: np.ndarray,
    qP: np.ndarray,
    ndotV: np.ndarray,
    is_boundary: np.ndarray,
    q_boundary_exact: np.ndarray,
) -> np.ndarray:
    """
    Fill exterior trace qP on boundary faces.

    Boundary policy
    ---------------
    - inflow  (n·V < 0): use exact boundary data
    - outflow (n·V >= 0): use qM itself

    Parameters
    ----------
    qM, qP, ndotV : np.ndarray
        Shape (K, 3, Nfp)
    is_boundary : np.ndarray
        Shape (K, 3)
    q_boundary_exact : np.ndarray
        Shape (K, 3, Nfp)

    Returns
    -------
    np.ndarray
        Boundary-filled qP, same shape as qM
    """
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float).copy()
    ndotV = np.asarray(ndotV, dtype=float)
    is_boundary = np.asarray(is_boundary, dtype=bool)
    q_boundary_exact = np.asarray(q_boundary_exact, dtype=float)

    if qM.shape != qP.shape or qM.shape != ndotV.shape or qM.shape != q_boundary_exact.shape:
        raise ValueError("qM, qP, ndotV, q_boundary_exact must all have the same shape.")
    if is_boundary.shape != qM.shape[:2]:
        raise ValueError("is_boundary must have shape (K, 3).")

    bd = is_boundary[:, :, None]
    inflow = bd & (ndotV < 0.0)
    outflow = bd & (ndotV >= 0.0)

    qP[inflow] = q_boundary_exact[inflow]
    qP[outflow] = qM[outflow]

    return qP

def fill_exterior_state(
    qP_exchange: np.ndarray,
    is_boundary: np.ndarray,
    q_boundary_exact: np.ndarray,
) -> np.ndarray:
    """
    Exterior state policy:
    - interior faces: use exchanged neighbor trace
    - physical boundary faces: use exact boundary data
    """
    qP_exchange = np.asarray(qP_exchange, dtype=float)
    q_boundary_exact = np.asarray(q_boundary_exact, dtype=float)
    is_boundary = np.asarray(is_boundary, dtype=bool)

    if qP_exchange.shape != q_boundary_exact.shape:
        raise ValueError("qP_exchange and q_boundary_exact must have the same shape.")
    if is_boundary.shape != qP_exchange.shape[:2]:
        raise ValueError("is_boundary must have shape (K, 3).")

    bd = is_boundary[:, :, None]   # shape (K, 3, 1)
    qP = np.where(bd, q_boundary_exact, qP_exchange)
    return qP


def _apply_exact_source_q_correction(
    q_exact_source: np.ndarray,
    *,
    x_face: np.ndarray,
    y_face: np.ndarray,
    t: float,
    qM: np.ndarray,
    ndotV: np.ndarray,
    active_faces: np.ndarray,
    q_boundary_correction,
    q_boundary_correction_mode: str,
) -> np.ndarray:
    """
    Apply user-supplied correction to exact-source traces.

    Expected callback signature:
        corr = q_boundary_correction(x_face, y_face, t, qM, ndotV, active_faces, q_exact_source)

    where corr can be broadcast to shape (K, 3, Nfp). `active_faces` marks
    which faces currently use exact data as the exterior source.
    """
    if q_boundary_correction is None:
        return q_exact_source

    mode = str(q_boundary_correction_mode).strip().lower()
    if mode not in ("inflow", "boundary", "all"):
        raise ValueError("q_boundary_correction_mode must be one of: 'inflow', 'boundary', 'all'.")

    corr = q_boundary_correction(
        x_face,
        y_face,
        t,
        qM,
        ndotV,
        active_faces,
        q_exact_source,
    )
    corr = np.asarray(corr, dtype=float)
    if corr.shape != q_exact_source.shape:
        try:
            corr = np.broadcast_to(corr, q_exact_source.shape)
        except ValueError as exc:
            raise ValueError(
                "q_boundary_correction must return an array broadcastable to shape (K, 3, Nfp)."
            ) from exc

    qP = np.array(q_exact_source, dtype=float, copy=True)
    active = np.asarray(active_faces, dtype=bool)[:, :, None]
    if mode == "inflow":
        mask = active & (ndotV < 0.0)
    else:
        mask = np.broadcast_to(active, q_exact_source.shape)
    qP[mask] += corr[mask]
    return qP


def _pair_boundary_faces_by_axis(
    faces_a: list[tuple[int, int]],
    faces_b: list[tuple[int, int]],
    face_midpoints: np.ndarray,
    *,
    axis: int,
    label_a: str,
    label_b: str,
    tol: float = 1e-12,
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    if len(faces_a) != len(faces_b):
        raise ValueError(
            f"Boundary group size mismatch for opposite pairing: {label_a}={len(faces_a)}, {label_b}={len(faces_b)}."
        )

    if len(faces_a) == 0:
        return []

    def _coord(face: tuple[int, int]) -> float:
        k, f = int(face[0]), int(face[1])
        return float(face_midpoints[k, f - 1, axis])

    left_sorted = sorted(((int(k), int(f)) for k, f in faces_a), key=_coord)
    right_sorted = sorted(((int(k), int(f)) for k, f in faces_b), key=_coord)

    pairs: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for fa, fb in zip(left_sorted, right_sorted):
        ca = _coord(fa)
        cb = _coord(fb)
        if not np.isclose(ca, cb, atol=tol, rtol=tol):
            raise ValueError(
                f"Opposite-boundary pairing mismatch between {label_a} and {label_b}: {ca} vs {cb}."
            )
        pairs.append((fa, fb, axis))
    return pairs


def _opposite_pair_needs_flip(
    x_face: np.ndarray,
    y_face: np.ndarray,
    *,
    ka: int,
    fa: int,
    kb: int,
    fb: int,
    axis: int,
) -> bool:
    if axis == 1:
        a = np.asarray(y_face[ka, fa - 1, :], dtype=float)
        b = np.asarray(y_face[kb, fb - 1, :], dtype=float)
    else:
        a = np.asarray(x_face[ka, fa - 1, :], dtype=float)
        b = np.asarray(x_face[kb, fb - 1, :], dtype=float)

    if a.size <= 1:
        return False

    d_same = float(np.max(np.abs(a - b)))
    d_flip = float(np.max(np.abs(a - b[::-1])))
    if np.isclose(d_same, d_flip, atol=1e-14, rtol=1e-12):
        ax = float(x_face[ka, fa - 1, -1] - x_face[ka, fa - 1, 0])
        ay = float(y_face[ka, fa - 1, -1] - y_face[ka, fa - 1, 0])
        bx = float(x_face[kb, fb - 1, -1] - x_face[kb, fb - 1, 0])
        by = float(y_face[kb, fb - 1, -1] - y_face[kb, fb - 1, 0])
        return (ax * bx + ay * by) < 0.0
    return d_flip < d_same


def _build_opposite_boundary_face_map(
    conn: dict,
    face_geom: dict,
) -> tuple[np.ndarray, np.ndarray]:
    EToE = np.asarray(conn["EToE"], dtype=int)
    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)
    face_midpoints_raw = conn.get("face_midpoints", None)
    boundary_groups = conn.get("boundary_groups", {})

    K = int(EToE.shape[0])
    nf = 3 * K
    opposite_face_flat = -np.ones(nf, dtype=np.int64)
    opposite_flip_flat = np.zeros(nf, dtype=np.bool_)

    boundary_flat = is_boundary.reshape(-1)
    if not np.any(boundary_flat):
        return opposite_face_flat, opposite_flip_flat

    if face_midpoints_raw is None:
        return opposite_face_flat, opposite_flip_flat
    face_midpoints = np.asarray(face_midpoints_raw, dtype=float)

    if face_midpoints.shape != (K, 3, 2):
        return opposite_face_flat, opposite_flip_flat

    required_groups = ("left", "right", "bottom", "top")
    missing = [g for g in required_groups if g not in boundary_groups]
    if missing:
        return opposite_face_flat, opposite_flip_flat

    x_face = np.asarray(face_geom["x_face"], dtype=float)
    y_face = np.asarray(face_geom["y_face"], dtype=float)

    pairs = []
    pairs.extend(
        _pair_boundary_faces_by_axis(
            boundary_groups["left"],
            boundary_groups["right"],
            face_midpoints,
            axis=1,
            label_a="left",
            label_b="right",
        )
    )
    pairs.extend(
        _pair_boundary_faces_by_axis(
            boundary_groups["bottom"],
            boundary_groups["top"],
            face_midpoints,
            axis=0,
            label_a="bottom",
            label_b="top",
        )
    )

    paired_idx: set[int] = set()
    for (ka, fa), (kb, fb), axis in pairs:
        ia = int(ka) * 3 + (int(fa) - 1)
        ib = int(kb) * 3 + (int(fb) - 1)
        if ia in paired_idx or ib in paired_idx:
            raise ValueError("Boundary face paired more than once while building opposite-boundary map.")

        flip = _opposite_pair_needs_flip(
            x_face,
            y_face,
            ka=int(ka),
            fa=int(fa),
            kb=int(kb),
            fb=int(fb),
            axis=axis,
        )

        opposite_face_flat[ia] = ib
        opposite_face_flat[ib] = ia
        opposite_flip_flat[ia] = flip
        opposite_flip_flat[ib] = flip
        paired_idx.add(ia)
        paired_idx.add(ib)

    unpaired_boundary = np.nonzero(boundary_flat & (opposite_face_flat < 0))[0]
    if unpaired_boundary.size > 0:
        raise ValueError(
            "Opposite-boundary map is incomplete for current mesh; "
            f"unpaired boundary faces: {int(unpaired_boundary.size)}."
        )

    return opposite_face_flat, opposite_flip_flat


def _build_boundary_state_from_opposite_boundary(
    q_elem: np.ndarray,
    cache: dict,
    use_numba: bool | None = None,
) -> np.ndarray:
    q_elem = np.asarray(q_elem, dtype=float)
    K = int(cache["K"])
    nfp = int(cache["nfp"])
    nf = 3 * K

    opposite_face_flat = cache.get("opposite_boundary_face_flat_numba", None)
    opposite_flip_flat = cache.get("opposite_boundary_flip_flat_numba", None)
    if opposite_face_flat is None or opposite_flip_flat is None:
        raise ValueError(
            "surface_cache missing opposite-boundary map; rebuild cache with compatible connectivity for opposite_boundary mode."
        )

    owner_elem = np.asarray(cache["owner_elem_flat_numba"], dtype=np.int64)
    owner_node_ids = np.asarray(cache["owner_node_ids_flat_numba"], dtype=np.int64)
    boundary_flat = np.asarray(cache["boundary_flat"], dtype=bool)
    opposite_face_flat = np.asarray(opposite_face_flat, dtype=np.int64)
    opposite_flip_flat = np.asarray(opposite_flip_flat, dtype=np.bool_)

    boundary_idx = np.nonzero(boundary_flat)[0]
    if np.any(opposite_face_flat[boundary_idx] < 0):
        raise ValueError("Encountered unpaired boundary face in opposite_boundary mode.")

    qB_flat = np.zeros((nf, nfp), dtype=float)
    if _should_use_numba(use_numba):
        q_elem_numba = q_elem
        if q_elem_numba.dtype != np.float64 or (not q_elem_numba.flags.c_contiguous):
            q_elem_numba = np.ascontiguousarray(q_elem_numba, dtype=np.float64)

        _boundary_state_from_opposite_kernel_inplace(
            q_elem=q_elem_numba,
            boundary_face_idx=np.ascontiguousarray(boundary_idx, dtype=np.int64),
            opposite_face_flat=opposite_face_flat,
            opposite_flip_flat=opposite_flip_flat,
            owner_elem_flat=owner_elem,
            owner_node_ids_flat=owner_node_ids,
            qB_flat=qB_flat,
        )
    else:
        for idx in boundary_idx:
            opp_idx = int(opposite_face_flat[idx])
            vals = q_elem[owner_elem[opp_idx], owner_node_ids[opp_idx, :]]
            if bool(opposite_flip_flat[idx]):
                vals = vals[::-1]
            qB_flat[idx, :] = vals

    return qB_flat.reshape(K, 3, nfp)


def _coord_key(x_val: float, y_val: float, tol: float) -> tuple[int, int]:
    return (int(np.round(float(x_val) / tol)), int(np.round(float(y_val) / tol)))


def _build_periodic_boundary_node_map(
    X_nodes: np.ndarray,
    Y_nodes: np.ndarray,
    *,
    owner_elem_flat: np.ndarray,
    owner_node_ids_flat: np.ndarray,
    boundary_flat: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    X_nodes = np.asarray(X_nodes, dtype=float)
    Y_nodes = np.asarray(Y_nodes, dtype=float)
    owner_elem_flat = np.asarray(owner_elem_flat, dtype=np.int64)
    owner_node_ids_flat = np.asarray(owner_node_ids_flat, dtype=np.int64)
    boundary_flat = np.asarray(boundary_flat, dtype=bool).reshape(-1)

    if X_nodes.shape != Y_nodes.shape or X_nodes.ndim != 2:
        raise ValueError("X_nodes and Y_nodes must both have shape (K, Np).")
    if owner_node_ids_flat.ndim != 2:
        raise ValueError("owner_node_ids_flat must have shape (3*K, Nfp).")
    if owner_elem_flat.ndim != 1 or owner_elem_flat.size != owner_node_ids_flat.shape[0]:
        raise ValueError("owner_elem_flat must have shape (3*K,) matching owner_node_ids_flat.")
    if boundary_flat.size != owner_node_ids_flat.shape[0]:
        raise ValueError("boundary_flat must have shape (3*K,) matching owner_node_ids_flat.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    K, Np = X_nodes.shape
    nf, nfp = owner_node_ids_flat.shape
    if nf != 3 * K:
        raise ValueError("owner_node_ids_flat must have shape (3*K, Nfp) matching X_nodes.")

    partner_elem_flat = -np.ones((nf, nfp), dtype=np.int64)
    partner_node_ids_flat = -np.ones((nf, nfp), dtype=np.int64)

    x_flat = X_nodes.reshape(-1)
    y_flat = Y_nodes.reshape(-1)
    xmin = float(np.min(x_flat))
    xmax = float(np.max(x_flat))
    ymin = float(np.min(y_flat))
    ymax = float(np.max(y_flat))

    coord_lookup: dict[tuple[int, int], list[int]] = {}
    for gid in range(x_flat.size):
        key = _coord_key(x_flat[gid], y_flat[gid], tol)
        coord_lookup.setdefault(key, []).append(int(gid))

    boundary_idx = np.nonzero(boundary_flat)[0]
    for face_idx in boundary_idx:
        elem_idx = int(owner_elem_flat[face_idx])
        for j in range(nfp):
            node_id = int(owner_node_ids_flat[face_idx, j])
            gid_m = elem_idx * Np + node_id

            x_m = float(X_nodes[elem_idx, node_id])
            y_m = float(Y_nodes[elem_idx, node_id])
            x_target = x_m
            y_target = y_m

            if np.isclose(x_m, xmin, atol=tol, rtol=tol):
                x_target = xmax
            elif np.isclose(x_m, xmax, atol=tol, rtol=tol):
                x_target = xmin

            if np.isclose(y_m, ymin, atol=tol, rtol=tol):
                y_target = ymax
            elif np.isclose(y_m, ymax, atol=tol, rtol=tol):
                y_target = ymin

            key_target = _coord_key(x_target, y_target, tol)
            candidates = coord_lookup.get(key_target, [])
            if not candidates:
                raise ValueError(
                    "Unable to build periodic_vmap boundary map: no nodal counterpart "
                    f"for boundary node ({x_m:.16e}, {y_m:.16e}) targeting "
                    f"({x_target:.16e}, {y_target:.16e})."
                )

            gid_p = next((cand for cand in candidates if cand != gid_m), candidates[0])
            partner_elem_flat[face_idx, j] = gid_p // Np
            partner_node_ids_flat[face_idx, j] = gid_p % Np

    missing = np.nonzero(boundary_flat[:, None] & (partner_elem_flat < 0))
    if missing[0].size > 0:
        raise ValueError("Periodic_vmap boundary map is incomplete for the current mesh.")

    return partner_elem_flat, partner_node_ids_flat


def _build_boundary_state_from_periodic_vmap(
    q_elem: np.ndarray,
    cache: dict,
    use_numba: bool | None = None,
) -> np.ndarray:
    q_elem = np.asarray(q_elem, dtype=float)
    K = int(cache["K"])
    nfp = int(cache["nfp"])
    nf = 3 * K

    partner_elem_flat = cache.get("periodic_vmap_elem_flat_numba", None)
    partner_node_ids_flat = cache.get("periodic_vmap_node_ids_flat_numba", None)
    if partner_elem_flat is None or partner_node_ids_flat is None:
        raise ValueError(
            "surface_cache missing periodic_vmap node map; rebuild cache with nodal coordinates "
            "for physical_boundary_mode='periodic_vmap'."
        )

    partner_elem_flat = np.asarray(partner_elem_flat, dtype=np.int64)
    partner_node_ids_flat = np.asarray(partner_node_ids_flat, dtype=np.int64)
    boundary_flat = np.asarray(cache["boundary_flat"], dtype=bool)
    if partner_elem_flat.shape != (nf, nfp) or partner_node_ids_flat.shape != (nf, nfp):
        raise ValueError("periodic_vmap node map must have shape (3*K, Nfp).")

    qB_flat = np.zeros((nf, nfp), dtype=float)
    boundary_idx = np.asarray(
        cache.get("boundary_face_idx_numba", np.nonzero(boundary_flat)[0].astype(np.int64)),
        dtype=np.int64,
    )
    if _should_use_numba(use_numba):
        q_elem_numba = q_elem
        if q_elem_numba.dtype != np.float64 or (not q_elem_numba.flags.c_contiguous):
            q_elem_numba = np.ascontiguousarray(q_elem_numba, dtype=np.float64)

        _boundary_state_from_periodic_vmap_kernel_inplace(
            q_elem=q_elem_numba,
            boundary_face_idx=np.ascontiguousarray(boundary_idx, dtype=np.int64),
            partner_elem_flat=partner_elem_flat,
            partner_node_ids_flat=partner_node_ids_flat,
            qB_flat=qB_flat,
        )
    else:
        for idx in boundary_idx:
            qB_flat[idx, :] = q_elem[partner_elem_flat[idx, :], partner_node_ids_flat[idx, :]]

    return qB_flat.reshape(K, 3, nfp)


def build_surface_exchange_cache(
    rule: dict,
    trace: dict,
    conn: dict,
    face_geom: dict,
    face_order_mode: str = "triangle",
    X_nodes: np.ndarray | None = None,
    Y_nodes: np.ndarray | None = None,
) -> dict:
    """
    Precompute static data used by the surface exchange term.

    The cache is valid as long as rule/trace/connectivity/face geometry do not
    change (typically fixed for one mesh level in a time loop).
    """
    if trace.get("trace_mode", None) != "embedded":
        raise ValueError("Surface cache currently supports embedded traces only.")

    face_mode = _resolve_face_order_mode(face_order_mode)
    face_perm_new_to_old, face_perm_old_to_new = _face_axis_permutations(face_mode)

    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    inv_ws = 1.0 / ws

    ids_faces_raw = []
    we_faces_raw = []
    wr_faces_raw = []
    ids_unique_raw = []
    nfp = int(trace["nfp"])

    for face_id in (1, 2, 3):
        ids = np.asarray(trace["face_node_ids"][face_id], dtype=int).reshape(-1)
        if ids.size != nfp:
            raise ValueError("Trace face-node count must match trace['nfp'].")

        we = np.asarray(trace["face_weights"][face_id], dtype=float).reshape(-1)
        if we.size != nfp:
            raise ValueError("Trace face-weight count must match trace['nfp'].")

        ids_faces_raw.append(ids)
        we_faces_raw.append(we)
        wr_faces_raw.append(we * inv_ws[ids])
        ids_unique_raw.append(np.unique(ids).size == ids.size)

    ids_faces = [ids_faces_raw[int(jf_old)] for jf_old in face_perm_new_to_old]
    we_faces = [we_faces_raw[int(jf_old)] for jf_old in face_perm_new_to_old]
    wr_faces = [wr_faces_raw[int(jf_old)] for jf_old in face_perm_new_to_old]
    ids_unique = [ids_unique_raw[int(jf_old)] for jf_old in face_perm_new_to_old]

    area = np.asarray(face_geom["area"], dtype=float).reshape(-1)
    length_raw = np.asarray(face_geom["length"], dtype=float)
    length = np.ascontiguousarray(length_raw[:, face_perm_new_to_old], dtype=float)
    if length.shape != (area.size, 3):
        raise ValueError("face_geom['length'] must have shape (K, 3).")

    EToE_raw = np.asarray(conn["EToE"], dtype=int)
    EToF_raw = np.asarray(conn["EToF"], dtype=int)
    is_boundary_raw = np.asarray(conn["is_boundary"], dtype=bool)
    face_flip_raw = np.asarray(conn["face_flip"], dtype=bool)

    K = int(EToE_raw.shape[0])
    if EToE_raw.shape != (K, 3):
        raise ValueError("conn['EToE'] must have shape (K, 3).")
    if (
        EToF_raw.shape != (K, 3)
        or is_boundary_raw.shape != (K, 3)
        or face_flip_raw.shape != (K, 3)
    ):
        raise ValueError("conn face arrays must have shape (K, 3).")

    EToE = np.ascontiguousarray(EToE_raw[:, face_perm_new_to_old], dtype=int)
    EToF_old = np.asarray(EToF_raw[:, face_perm_new_to_old], dtype=int)
    EToF = np.ascontiguousarray(face_perm_old_to_new[EToF_old - 1] + 1, dtype=int)
    is_boundary = np.ascontiguousarray(is_boundary_raw[:, face_perm_new_to_old], dtype=bool)
    face_flip = np.ascontiguousarray(face_flip_raw[:, face_perm_new_to_old], dtype=bool)

    x_face_raw = np.asarray(face_geom["x_face"], dtype=float)
    y_face_raw = np.asarray(face_geom["y_face"], dtype=float)
    nx_raw = np.asarray(face_geom["nx"], dtype=float)
    ny_raw = np.asarray(face_geom["ny"], dtype=float)
    x_face = np.ascontiguousarray(x_face_raw[:, face_perm_new_to_old, :], dtype=float)
    y_face = np.ascontiguousarray(y_face_raw[:, face_perm_new_to_old, :], dtype=float)
    nx = np.ascontiguousarray(nx_raw[:, face_perm_new_to_old, :], dtype=float)
    ny = np.ascontiguousarray(ny_raw[:, face_perm_new_to_old, :], dtype=float)
    expected_face_shape = (K, 3, nfp)
    if (
        x_face.shape != expected_face_shape
        or y_face.shape != expected_face_shape
        or nx.shape != expected_face_shape
        or ny.shape != expected_face_shape
    ):
        raise ValueError("face_geom face arrays must have shape (K, 3, Nfp).")

    nbr_face_linear = EToE * 3 + (EToF - 1)
    boundary_flat = is_boundary.reshape(-1)
    flip_flat = face_flip.reshape(-1)

    area_numba = np.ascontiguousarray(area, dtype=np.float64)
    length_numba = np.ascontiguousarray(length, dtype=np.float64)
    neighbor_face_flat_numba = np.ascontiguousarray(nbr_face_linear.reshape(-1), dtype=np.int64)
    boundary_flat_numba = np.ascontiguousarray(boundary_flat, dtype=np.bool_)
    flip_flat_numba = np.ascontiguousarray(flip_flat, dtype=np.bool_)
    length_over_area_flat_numba = np.ascontiguousarray((length / area[:, None]).reshape(-1), dtype=np.float64)

    nf = 3 * K
    owner_elem_flat_numba = np.repeat(np.arange(K, dtype=np.int64), 3)
    owner_node_ids_flat_numba = np.empty((nf, nfp), dtype=np.int64)
    owner_wratio_flat_numba = np.empty((nf, nfp), dtype=np.float64)
    face_weight_flat_numba = np.empty((nf, nfp), dtype=np.float64)
    length_flat_numba = np.ascontiguousarray(length.reshape(-1), dtype=np.float64)

    ids_faces_numba = tuple(np.ascontiguousarray(ids, dtype=np.int64) for ids in ids_faces)
    wr_faces_numba = tuple(np.ascontiguousarray(wr, dtype=np.float64) for wr in wr_faces)
    we_faces_numba = tuple(np.ascontiguousarray(we, dtype=np.float64) for we in we_faces)
    we_faces_matrix = np.ascontiguousarray(np.stack(we_faces, axis=0), dtype=float)

    E_face_matrix = np.zeros((3 * nfp, int(ws.size)), dtype=float)
    for jf in range(3):
        row0 = jf * nfp
        for i_local, node_id in enumerate(ids_faces[jf]):
            E_face_matrix[row0 + i_local, int(node_id)] = 1.0
    E_face_matrix = np.ascontiguousarray(E_face_matrix, dtype=float)

    for jf in range(3):
        rows = np.arange(jf, nf, 3, dtype=np.int64)
        owner_node_ids_flat_numba[rows, :] = ids_faces_numba[jf][None, :]
        owner_wratio_flat_numba[rows, :] = wr_faces_numba[jf][None, :]
        face_weight_flat_numba[rows, :] = we_faces_numba[jf][None, :]

    nbr_elem_flat_numba = np.empty(nf, dtype=np.int64)
    nbr_node_ids_flat_numba = np.empty((nf, nfp), dtype=np.int64)

    nbr_face_flat = nbr_face_linear.reshape(-1)
    for face_idx in range(nf):
        if boundary_flat[face_idx]:
            nbr_elem_flat_numba[face_idx] = -1
            nbr_node_ids_flat_numba[face_idx, :] = owner_node_ids_flat_numba[face_idx, :]
            continue

        nbr_face = int(nbr_face_flat[face_idx])
        nbr_k = nbr_face // 3
        nbr_jf = nbr_face - 3 * nbr_k

        nbr_elem_flat_numba[face_idx] = nbr_k
        nbr_ids = ids_faces_numba[nbr_jf]
        if flip_flat[face_idx]:
            nbr_node_ids_flat_numba[face_idx, :] = nbr_ids[::-1]
        else:
            nbr_node_ids_flat_numba[face_idx, :] = nbr_ids

    boundary_face_idx_numba = np.ascontiguousarray(
        np.nonzero(boundary_flat)[0].astype(np.int64),
        dtype=np.int64,
    )

    pair_owner_mask = (~boundary_flat) & (
        np.arange(nf, dtype=np.int64) < nbr_face_flat.astype(np.int64)
    )
    pair_face_a_numba = np.ascontiguousarray(
        np.nonzero(pair_owner_mask)[0].astype(np.int64),
        dtype=np.int64,
    )
    pair_face_b_numba = np.ascontiguousarray(
        nbr_face_flat[pair_face_a_numba].astype(np.int64),
        dtype=np.int64,
    )

    opposite_face_flat_old, opposite_flip_flat_old = _build_opposite_boundary_face_map(
        conn=conn,
        face_geom=face_geom,
    )

    flat_new_to_old, flat_old_to_new = _build_flat_face_index_maps(
        K=K,
        face_perm_new_to_old=face_perm_new_to_old,
    )
    opposite_face_flat = -np.ones(nf, dtype=np.int64)
    opposite_flip_flat = np.zeros(nf, dtype=np.bool_)
    for idx_new in range(nf):
        idx_old = int(flat_new_to_old[idx_new])
        opp_old = int(opposite_face_flat_old[idx_old])
        if opp_old >= 0:
            opposite_face_flat[idx_new] = int(flat_old_to_new[opp_old])
        opposite_flip_flat[idx_new] = bool(opposite_flip_flat_old[idx_old])

    opposite_face_flat_numba = np.ascontiguousarray(opposite_face_flat, dtype=np.int64)
    opposite_flip_flat_numba = np.ascontiguousarray(opposite_flip_flat, dtype=np.bool_)

    periodic_vmap_elem_flat = None
    periodic_vmap_node_ids_flat = None
    periodic_vmap_elem_flat_numba = None
    periodic_vmap_node_ids_flat_numba = None
    if X_nodes is not None or Y_nodes is not None:
        if X_nodes is None or Y_nodes is None:
            raise ValueError("X_nodes and Y_nodes must either both be provided or both be None.")
        periodic_vmap_elem_flat, periodic_vmap_node_ids_flat = _build_periodic_boundary_node_map(
            X_nodes,
            Y_nodes,
            owner_elem_flat=owner_elem_flat_numba,
            owner_node_ids_flat=owner_node_ids_flat_numba,
            boundary_flat=boundary_flat,
        )
        periodic_vmap_elem_flat_numba = np.ascontiguousarray(periodic_vmap_elem_flat, dtype=np.int64)
        periodic_vmap_node_ids_flat_numba = np.ascontiguousarray(periodic_vmap_node_ids_flat, dtype=np.int64)

    return {
        "face_order_mode": face_mode,
        "face_perm_new_to_old": np.ascontiguousarray(face_perm_new_to_old, dtype=np.int64),
        "face_perm_old_to_new": np.ascontiguousarray(face_perm_old_to_new, dtype=np.int64),
        "flat_face_new_to_old": np.ascontiguousarray(flat_new_to_old, dtype=np.int64),
        "flat_face_old_to_new": np.ascontiguousarray(flat_old_to_new, dtype=np.int64),
        "K": K,
        "Np": int(ws.size),
        "nfp": nfp,
        "area": area,
        "length": length,
        "length_over_area": length / area[:, None],
        "x_face": x_face,
        "y_face": y_face,
        "nx": nx,
        "ny": ny,
        "EToE": EToE,
        "EToF": EToF,
        "is_boundary": is_boundary,
        "face_flip": face_flip,
        "ids_faces": tuple(ids_faces),
        "we_faces": tuple(we_faces),
        "wr_faces": tuple(wr_faces),
        "ids_unique": tuple(ids_unique),
        "we_faces_matrix": we_faces_matrix,
        "E_face_matrix": E_face_matrix,
        "ids_faces_numba": ids_faces_numba,
        "wr_faces_numba": wr_faces_numba,
        "we_faces_numba": we_faces_numba,
        "area_numba": area_numba,
        "length_numba": length_numba,
        "length_flat_numba": length_flat_numba,
        "length_over_area_flat_numba": length_over_area_flat_numba,
        "neighbor_face_flat": nbr_face_linear.reshape(-1),
        "neighbor_face_flat_numba": neighbor_face_flat_numba,
        "boundary_flat": boundary_flat,
        "boundary_flat_numba": boundary_flat_numba,
        "interior_flat": ~boundary_flat,
        "flip_flat": flip_flat,
        "flip_flat_numba": flip_flat_numba,
        "owner_elem_flat_numba": np.ascontiguousarray(owner_elem_flat_numba, dtype=np.int64),
        "owner_node_ids_flat_numba": np.ascontiguousarray(owner_node_ids_flat_numba, dtype=np.int64),
        "owner_wratio_flat_numba": np.ascontiguousarray(owner_wratio_flat_numba, dtype=np.float64),
        "face_weight_flat_numba": np.ascontiguousarray(face_weight_flat_numba, dtype=np.float64),
        "nbr_elem_flat_numba": np.ascontiguousarray(nbr_elem_flat_numba, dtype=np.int64),
        "nbr_node_ids_flat_numba": np.ascontiguousarray(nbr_node_ids_flat_numba, dtype=np.int64),
        "boundary_face_idx_numba": boundary_face_idx_numba,
        "pair_face_a_numba": pair_face_a_numba,
        "pair_face_b_numba": pair_face_b_numba,
        "opposite_boundary_face_flat": opposite_face_flat,
        "opposite_boundary_flip_flat": opposite_flip_flat,
        "opposite_boundary_face_flat_numba": opposite_face_flat_numba,
        "opposite_boundary_flip_flat_numba": opposite_flip_flat_numba,
        "periodic_vmap_elem_flat": periodic_vmap_elem_flat,
        "periodic_vmap_node_ids_flat": periodic_vmap_node_ids_flat,
        "periodic_vmap_elem_flat_numba": periodic_vmap_elem_flat_numba,
        "periodic_vmap_node_ids_flat_numba": periodic_vmap_node_ids_flat_numba,
    }


def _lift_surface_penalty_to_volume(
    p: np.ndarray,
    cache: dict,
    use_numba: bool | None,
    surface_inverse_mass_T: np.ndarray | None = None,
    lift_scale: float = 1.0,
) -> np.ndarray:
    K = int(cache["K"])
    Np = int(cache["Np"])
    lift_scale = float(lift_scale)
    if (not np.isfinite(lift_scale)) or lift_scale <= 0.0:
        raise ValueError("lift_scale must be a finite positive scalar.")

    if surface_inverse_mass_T is not None:
        surface_inverse_mass_t = np.asarray(surface_inverse_mass_T, dtype=float)
        if (
            surface_inverse_mass_t.ndim != 2
            or surface_inverse_mass_t.shape[0] != Np
            or surface_inverse_mass_t.shape[1] != Np
        ):
            raise ValueError("surface_inverse_mass_T must be a square (Np, Np) array.")

        if _should_use_numba(use_numba):
            owner_elem_flat = np.asarray(cache["owner_elem_flat_numba"], dtype=np.int64)
            owner_node_ids_flat = np.asarray(cache["owner_node_ids_flat_numba"], dtype=np.int64)
            face_weight_flat = np.asarray(cache["face_weight_flat_numba"], dtype=np.float64)
            length_flat = np.asarray(cache["length_flat_numba"], dtype=np.float64)
            area_numba = np.asarray(cache.get("area_numba", cache["area"]), dtype=np.float64)

            nf = int(owner_elem_flat.size)
            if owner_node_ids_flat.shape != (nf, p.shape[2]):
                raise ValueError("owner_node_ids_flat_numba shape mismatch in surface cache.")
            if face_weight_flat.shape != (nf, p.shape[2]):
                raise ValueError("face_weight_flat_numba shape mismatch in surface cache.")
            if length_flat.shape != (nf,):
                raise ValueError("length_flat_numba shape mismatch in surface cache.")
            if area_numba.shape != (K,):
                raise ValueError("area_numba shape mismatch in surface cache.")

            p_numba = p
            if p_numba.dtype != np.float64 or (not p_numba.flags.c_contiguous):
                p_numba = np.ascontiguousarray(p_numba, dtype=np.float64)

            surface_integral = np.zeros((K, Np), dtype=float)
            _accumulate_projected_surface_integral_kernel_inplace(
                p=p_numba,
                owner_elem_flat=owner_elem_flat,
                owner_node_ids_flat=owner_node_ids_flat,
                face_weight_flat=face_weight_flat,
                length_flat=length_flat,
                surface_integral=surface_integral,
            )

            surface_rhs = surface_integral @ surface_inverse_mass_t
            surface_rhs /= area_numba[:, None]
            if lift_scale != 1.0:
                surface_rhs *= lift_scale
            return surface_rhs

        if lift_scale != 1.0:
            length = np.asarray(cache["length"], dtype=float)
            area = np.asarray(cache["area"], dtype=float)
            we_faces_matrix = np.asarray(cache["we_faces_matrix"], dtype=float)
            E_face_matrix = np.asarray(cache["E_face_matrix"], dtype=float)
            if we_faces_matrix.shape != (3, p.shape[2]):
                raise ValueError("we_faces_matrix shape mismatch in surface cache.")
            if E_face_matrix.shape != (3 * p.shape[2], Np):
                raise ValueError("E_face_matrix shape mismatch in surface cache.")

            scaled_penalty = (
                p * we_faces_matrix[None, :, :] * length[:, :, None]
            ).transpose(1, 2, 0).reshape(3 * p.shape[2], K)
            surface_integral = E_face_matrix.T @ scaled_penalty
            surface_rhs = surface_integral.T @ surface_inverse_mass_t
            surface_rhs /= area[:, None]
            surface_rhs *= lift_scale
            return surface_rhs

        ids_faces = cache["ids_faces"]
        we_faces = cache["we_faces"]
        ids_unique = cache["ids_unique"]
        length = np.asarray(cache["length"], dtype=float)
        area = np.asarray(cache["area"], dtype=float)

        surface_integral = np.zeros((K, Np), dtype=float)
        for jf in range(3):
            ids = ids_faces[jf]
            we = np.asarray(we_faces[jf], dtype=float)
            face_contrib = length[:, jf][:, None] * we[None, :] * p[:, jf, :]

            if ids_unique[jf]:
                surface_integral[:, ids] += face_contrib
            else:
                row_idx = np.broadcast_to(np.arange(K)[:, None], face_contrib.shape)
                col_idx = np.broadcast_to(ids[None, :], face_contrib.shape)
                np.add.at(surface_integral, (row_idx, col_idx), face_contrib)

        surface_rhs = surface_integral @ surface_inverse_mass_t
        surface_rhs /= area[:, None]
        if lift_scale != 1.0:
            surface_rhs *= lift_scale
        return surface_rhs

    surface_rhs = np.zeros((K, Np), dtype=float)

    if _should_use_numba(use_numba):
        ids_f1, ids_f2, ids_f3 = cache["ids_faces_numba"]
        wr_f1, wr_f2, wr_f3 = cache["wr_faces_numba"]
        area_numba = np.asarray(cache.get("area_numba", cache["area"]), dtype=np.float64)
        length_numba = np.asarray(cache.get("length_numba", cache["length"]), dtype=np.float64)

        nfp = p.shape[2]
        if (
            ids_f1.size != nfp
            or ids_f2.size != nfp
            or ids_f3.size != nfp
            or wr_f1.size != nfp
            or wr_f2.size != nfp
            or wr_f3.size != nfp
        ):
            raise ValueError("Trace face-node count must match p.shape[2].")

        _surface_rhs_kernel_inplace(
            p=np.ascontiguousarray(p, dtype=np.float64),
            area=area_numba,
            length=length_numba,
            ids_f1=ids_f1,
            ids_f2=ids_f2,
            ids_f3=ids_f3,
            wr_f1=wr_f1,
            wr_f2=wr_f2,
            wr_f3=wr_f3,
            surface_rhs=surface_rhs,
        )
        return surface_rhs

    ids_faces = cache["ids_faces"]
    wr_faces = cache["wr_faces"]
    ids_unique = cache["ids_unique"]
    length_over_area = np.asarray(cache["length_over_area"], dtype=float)

    for jf in range(3):
        ids = ids_faces[jf]
        w_ratio = wr_faces[jf]
        face_contrib = length_over_area[:, jf][:, None] * w_ratio[None, :] * p[:, jf, :]

        if ids_unique[jf]:
            surface_rhs[:, ids] += face_contrib
        else:
            row_idx = np.broadcast_to(np.arange(K)[:, None], face_contrib.shape)
            col_idx = np.broadcast_to(ids[None, :], face_contrib.shape)
            np.add.at(surface_rhs, (row_idx, col_idx), face_contrib)

    if lift_scale != 1.0:
        surface_rhs *= lift_scale

    return surface_rhs

def surface_term_from_exchange(
    q_elem: np.ndarray,
    rule: dict,
    trace: dict,
    conn: dict,
    face_geom: dict,
    q_boundary,
    velocity,
    t: float = 0.0,
    tau: float = 0.0,
    tau_interior: float | None = None,
    tau_qb: float | None = None,
    compute_mismatches: bool = True,
    return_diagnostics: bool = True,
    use_numba: bool | None = None,
    surface_backend: str | None = None,
    surface_cache: dict | None = None,
    ndotV_precomputed: np.ndarray | None = None,
    ndotV_flat_precomputed: np.ndarray | None = None,
    surface_inverse_mass_T: np.ndarray | None = None,
    physical_boundary_mode: str = "exact_qb",
    face_order_mode: str = "triangle",
    q_boundary_correction=None,
    q_boundary_correction_mode: str = "inflow",
    volume_split_cache: dict | None = None,
    X_nodes: np.ndarray | None = None,
    Y_nodes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Surface term using actual interior face exchange plus prescribed boundary traces.

    Current scope
    -------------
    - Table 1 only (embedded face nodes)
    - affine triangles only
    """
    q_elem = np.asarray(q_elem, dtype=float)
    if q_elem.ndim != 2:
        raise ValueError("q_elem must have shape (K, Np).")

    if trace.get("trace_mode", None) != "embedded":
        raise ValueError("This phase currently supports Table 1 embedded trace only.")

    requested_face_mode = _resolve_face_order_mode(face_order_mode)
    requested_axis_mode = _canonical_face_axis_mode(requested_face_mode)
    boundary_mode = str(physical_boundary_mode).strip().lower()
    if boundary_mode not in ("exact_qb", "opposite_boundary", "periodic_vmap"):
        raise ValueError(
            "physical_boundary_mode must be one of: 'exact_qb', 'opposite_boundary', 'periodic_vmap'."
        )

    cache = (
        build_surface_exchange_cache(
            rule,
            trace,
            conn,
            face_geom,
            face_order_mode=requested_face_mode,
            X_nodes=X_nodes if boundary_mode == "periodic_vmap" else None,
            Y_nodes=Y_nodes if boundary_mode == "periodic_vmap" else None,
        )
        if surface_cache is None
        else surface_cache
    )
    cache_face_mode = _resolve_face_order_mode(cache.get("face_order_mode", "triangle"))
    cache_axis_mode = _canonical_face_axis_mode(cache_face_mode)
    if cache_axis_mode != requested_axis_mode:
        raise ValueError(
            "surface_cache face-order mode does not match requested face_order_mode."
        )
    surface_lift_scale = _resolve_surface_lift_scale(
        face_order_mode=requested_face_mode,
        surface_inverse_mass_T=surface_inverse_mass_T,
    )
    face_perm_new_to_old = np.asarray(
        cache.get("face_perm_new_to_old", np.asarray([0, 1, 2], dtype=np.int64)),
        dtype=np.int64,
    )

    K, Np = q_elem.shape
    if int(cache["K"]) != K or int(cache["Np"]) != Np:
        raise ValueError("surface_cache does not match q_elem shape.")

    nfp = int(cache["nfp"])
    backend = _resolve_surface_backend(
        surface_backend,
        use_numba=use_numba,
        compute_mismatches=compute_mismatches,
        return_diagnostics=return_diagnostics,
    )

    x_face = np.asarray(cache["x_face"], dtype=float)
    y_face = np.asarray(cache["y_face"], dtype=float)
    nx = np.asarray(cache["nx"], dtype=float)
    ny = np.asarray(cache["ny"], dtype=float)

    u_face = None
    v_face = None
    if ndotV_precomputed is None:
        u_face, v_face = velocity(x_face, y_face, t)
        u_face = np.asarray(u_face, dtype=float)
        v_face = np.asarray(v_face, dtype=float)
        ndotV = nx * u_face + ny * v_face
    else:
        ndotV = np.asarray(ndotV_precomputed, dtype=float)
        if ndotV.shape != (K, 3, nfp):
            raise ValueError("ndotV_precomputed must have shape (K, 3, Nfp).")
        if _is_simplex_like_face_mode(cache_face_mode):
            ndotV = np.ascontiguousarray(ndotV[:, face_perm_new_to_old, :], dtype=float)
        if return_diagnostics:
            u_face, v_face = velocity(x_face, y_face, t)
            u_face = np.asarray(u_face, dtype=float)
            v_face = np.asarray(v_face, dtype=float)

    if boundary_mode == "periodic_vmap" and cache.get("periodic_vmap_elem_flat_numba", None) is None:
        raise ValueError(
            "physical_boundary_mode='periodic_vmap' requires a periodic node map in surface_cache; "
            "provide X_nodes and Y_nodes when building the cache."
        )
    if q_boundary_correction is not None and boundary_mode != "exact_qb":
        raise ValueError(
            "q_boundary_correction requires an exact boundary source; "
            "only physical_boundary_mode='exact_qb' provides one."
        )

    qB_exact = None
    if boundary_mode == "exact_qb":
        if q_boundary is None:
            raise ValueError("q_boundary callback is required when physical_boundary_mode='exact_qb'.")
        qB_exact = np.asarray(q_boundary(x_face, y_face, t), dtype=float)
        qP_boundary = np.array(qB_exact, dtype=float, copy=True)
    elif boundary_mode == "periodic_vmap":
        qP_boundary = _build_boundary_state_from_periodic_vmap(
            q_elem=q_elem,
            cache=cache,
            use_numba=use_numba,
        )
    else:
        qP_boundary = _build_boundary_state_from_opposite_boundary(
            q_elem=q_elem,
            cache=cache,
            use_numba=use_numba,
        )

    if ndotV.shape != (K, 3, nfp) or qP_boundary.shape != (K, 3, nfp):
        raise ValueError("velocity/boundary callbacks must return arrays with shape (K, 3, Nfp).")
    if qB_exact is not None and qB_exact.shape != (K, 3, nfp):
        raise ValueError("q_boundary callback must return arrays with shape (K, 3, Nfp).")

    tau_interior_eff, tau_qb_eff, tau_face = build_face_tau_array(
        is_boundary=cache["is_boundary"],
        face_shape=(K, 3, nfp),
        physical_boundary_mode=boundary_mode,
        tau=tau,
        tau_interior=tau_interior,
        tau_qb=tau_qb,
    )
    tau_boundary_eff = tau_qb_eff if boundary_mode == "exact_qb" else tau_interior_eff

    qM_face_prefetched = None
    if q_boundary_correction is not None and boundary_mode == "exact_qb":
        ids_faces = cache["ids_faces"]
        qM_face_prefetched = np.empty((K, 3, nfp), dtype=float)
        for jf in range(3):
            qM_face_prefetched[:, jf, :] = q_elem[:, ids_faces[jf]]

        qP_boundary = _apply_exact_source_q_correction(
            q_exact_source=qB_exact,
            x_face=x_face,
            y_face=y_face,
            t=t,
            qM=qM_face_prefetched,
            ndotV=ndotV,
            active_faces=cache["is_boundary"],
            q_boundary_correction=q_boundary_correction,
            q_boundary_correction_mode=q_boundary_correction_mode,
        )

    if backend == "face-major":
        if _should_use_numba(use_numba) and (not return_diagnostics) and (surface_inverse_mass_T is None):
            surface_rhs = np.zeros((K, Np), dtype=float)

            if ndotV_flat_precomputed is None or _is_simplex_like_face_mode(cache_face_mode):
                ndotV_flat = ndotV.reshape(K * 3, nfp)
                if ndotV_flat.dtype != np.float64 or (not ndotV_flat.flags.c_contiguous):
                    ndotV_flat = np.ascontiguousarray(ndotV_flat, dtype=np.float64)
            else:
                ndotV_flat = _as_c_f64(ndotV_flat_precomputed)
                if ndotV_flat.shape != (K * 3, nfp):
                    raise ValueError("ndotV_flat_precomputed must have shape (K*3, Nfp).")

            qP_boundary_flat = qP_boundary.reshape(K * 3, nfp)
            if qP_boundary_flat.dtype != np.float64 or (not qP_boundary_flat.flags.c_contiguous):
                qP_boundary_flat = np.ascontiguousarray(qP_boundary_flat, dtype=np.float64)

            q_elem_numba = q_elem
            if q_elem_numba.dtype != np.float64 or (not q_elem_numba.flags.c_contiguous):
                q_elem_numba = np.ascontiguousarray(q_elem_numba, dtype=np.float64)

            if (
                "pair_face_a_numba" in cache
                and "pair_face_b_numba" in cache
                and "boundary_face_idx_numba" in cache
            ):
                _face_major_surface_rhs_pair_kernel_inplace(
                    q_elem=q_elem_numba,
                    ndotV_flat=ndotV_flat,
                    qB_flat=qP_boundary_flat,
                    tau_interior=float(tau_interior_eff),
                    tau_boundary=float(tau_boundary_eff),
                    owner_elem_flat=cache["owner_elem_flat_numba"],
                    owner_node_ids_flat=cache["owner_node_ids_flat_numba"],
                    owner_wratio_flat=cache["owner_wratio_flat_numba"],
                    length_over_area_flat=cache["length_over_area_flat_numba"],
                    boundary_face_idx=cache["boundary_face_idx_numba"],
                    pair_face_a=cache["pair_face_a_numba"],
                    pair_face_b=cache["pair_face_b_numba"],
                    nbr_node_ids_flat=cache["nbr_node_ids_flat_numba"],
                    surface_rhs=surface_rhs,
                )
            else:
                _face_major_surface_rhs_flat_kernel_inplace(
                    q_elem=q_elem_numba,
                    ndotV_flat=ndotV_flat,
                    qB_flat=qP_boundary_flat,
                    tau_interior=float(tau_interior_eff),
                    tau_boundary=float(tau_boundary_eff),
                    owner_elem_flat=cache["owner_elem_flat_numba"],
                    owner_node_ids_flat=cache["owner_node_ids_flat_numba"],
                    owner_wratio_flat=cache["owner_wratio_flat_numba"],
                    length_over_area_flat=cache["length_over_area_flat_numba"],
                    boundary_flat=cache["boundary_flat_numba"],
                    nbr_elem_flat=cache["nbr_elem_flat_numba"],
                    nbr_node_ids_flat=cache["nbr_node_ids_flat_numba"],
                    surface_rhs=surface_rhs,
                )
            return surface_rhs, {}

        ids_faces = cache["ids_faces"]
        if qM_face_prefetched is None:
            qM = np.empty((K, 3, nfp), dtype=float)
            for jf in range(3):
                qM[:, jf, :] = q_elem[:, ids_faces[jf]]
        else:
            qM = qM_face_prefetched

        qM_flat = qM.reshape(K * 3, nfp)
        ndotV_flat = ndotV.reshape(K * 3, nfp)
        qP_boundary_flat = qP_boundary.reshape(K * 3, nfp)

        interior_flat = cache["interior_flat"]
        boundary_flat = cache["boundary_flat"]
        flip_flat = cache["flip_flat"]
        nbr_flat = cache["neighbor_face_flat"]

        qP_before_flat = None
        if return_diagnostics:
            qP_before_flat = np.empty_like(qM_flat)
            qP_before_flat[boundary_flat] = np.nan
            qP_before_flat[interior_flat] = qM_flat[nbr_flat[interior_flat]]

            flip_interior = flip_flat & interior_flat
            if np.any(flip_interior):
                qP_before_flat[flip_interior] = qP_before_flat[flip_interior, ::-1]

            qP_flat = qP_before_flat.copy()
            qP_flat[boundary_flat] = qP_boundary_flat[boundary_flat]
        else:
            qP_flat = np.empty_like(qM_flat)
            qP_flat[interior_flat] = qM_flat[nbr_flat[interior_flat]]

            flip_interior = flip_flat & interior_flat
            if np.any(flip_interior):
                qP_flat[flip_interior] = qP_flat[flip_interior, ::-1]

            qP_flat[boundary_flat] = qP_boundary_flat[boundary_flat]

        p = numerical_flux_penalty(
            ndotV=ndotV_flat,
            qM=qM_flat,
            qP=qP_flat,
            tau=tau_face.reshape(K * 3, nfp),
        )
        p = p.reshape(K, 3, nfp)
        surface_rhs = _lift_surface_penalty_to_volume(
            p,
            cache,
            use_numba=use_numba,
            surface_inverse_mass_T=surface_inverse_mass_T,
            lift_scale=surface_lift_scale,
        )

        if not return_diagnostics:
            return surface_rhs, {}

        qP_filled = qP_flat.reshape(K, 3, nfp)
        qP_before = qP_before_flat.reshape(K, 3, nfp)

        if compute_mismatches:
            mismatches = interior_face_pair_mismatches(
                {
                    "uM": qM,
                    "uP": qP_before,
                    "EToE": cache["EToE"],
                    "EToF": cache["EToF"],
                    "is_boundary": cache["is_boundary"],
                }
            )
        else:
            mismatches = []

        diagnostics = {
            "qM": qM,
            "qP_interior": qP_before,
            "qP_before_boundary_fill": qP_before,
            "qP_boundary": qP_boundary,
            "qP": qP_filled,
            "qB_exact": qB_exact,
            "qB": qB_exact,
            "tau_interior": float(tau_interior_eff),
            "tau_qb": float(tau_qb_eff),
            "tau_face": tau_face,
            "physical_boundary_mode": boundary_mode,
            "u_face": u_face,
            "v_face": v_face,
            "ndotV": ndotV,
            "p": p,
            "x_face": x_face,
            "y_face": y_face,
            "nx": nx,
            "ny": ny,
            "interior_mismatches": mismatches,
            "surface_backend": "face-major",
            "face_order_mode": requested_face_mode,
            "surface_lift_scale": float(surface_lift_scale),
        }
        return surface_rhs, diagnostics

    paired = pair_face_traces(
        u_elem=q_elem,
        conn=conn,
        trace=trace,
        boundary_fill_value=np.nan,
        use_numba=use_numba,
    )

    qM = np.asarray(paired["uM"], dtype=float)
    qP = np.asarray(paired["uP"], dtype=float)

    if _is_simplex_like_face_mode(cache_face_mode):
        qM = np.ascontiguousarray(qM[:, face_perm_new_to_old, :], dtype=float)
        qP = np.ascontiguousarray(qP[:, face_perm_new_to_old, :], dtype=float)

    is_boundary = np.asarray(cache["is_boundary"], dtype=bool)
    qP_filled = fill_exterior_state(
        qP_exchange=qP,
        is_boundary=is_boundary,
        q_boundary_exact=qP_boundary,
    )

    p = numerical_flux_penalty(
        ndotV=ndotV,
        qM=qM,
        qP=qP_filled,
        tau=tau_face,
    )
    surface_rhs = _lift_surface_penalty_to_volume(
        p,
        cache,
        use_numba=use_numba,
        surface_inverse_mass_T=surface_inverse_mass_T,
        lift_scale=surface_lift_scale,
    )

    if not return_diagnostics:
        return surface_rhs, {}

    if compute_mismatches:
        mismatches = interior_face_pair_mismatches(
            {
                "uM": qM,
                "uP": qP,
                "EToE": cache["EToE"],
                "EToF": cache["EToF"],
                "is_boundary": cache["is_boundary"],
            }
        )
    else:
        mismatches = []

    diagnostics = {
        "qM": qM,
        "qP_interior": qP,
        "qP_before_boundary_fill": qP,
        "qP_boundary": qP_boundary,
        "qP": qP_filled,
        "qB_exact": qB_exact,
        "qB": qB_exact,
        "tau_interior": float(tau_interior_eff),
        "tau_qb": float(tau_qb_eff),
        "tau_face": tau_face,
        "physical_boundary_mode": boundary_mode,
        "u_face": u_face,
        "v_face": v_face,
        "ndotV": ndotV,
        "p": p,
        "x_face": x_face,
        "y_face": y_face,
        "nx": nx,
        "ny": ny,
        "interior_mismatches": mismatches,
        "surface_backend": "legacy",
        "face_order_mode": requested_face_mode,
        "surface_lift_scale": float(surface_lift_scale),
    }
    return surface_rhs, diagnostics


def rhs_split_conservative_exchange(
    q_elem: np.ndarray,
    u_elem: np.ndarray,
    v_elem: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict,
    rule: dict,
    trace: dict,
    conn: dict,
    face_geom: dict,
    q_boundary,
    velocity,
    t: float = 0.0,
    tau: float = 0.0,
    tau_interior: float | None = None,
    tau_qb: float | None = None,
    compute_mismatches: bool = True,
    return_diagnostics: bool = True,
    use_numba: bool | None = None,
    surface_backend: str | None = None,
    surface_cache: dict | None = None,
    ndotV_precomputed: np.ndarray | None = None,
    ndotV_flat_precomputed: np.ndarray | None = None,
    surface_inverse_mass_T: np.ndarray | None = None,
    volume_split_cache: dict | None = None,
    physical_boundary_mode: str = "exact_qb",
    face_order_mode: str = "triangle",
    q_boundary_correction=None,
    q_boundary_correction_mode: str = "inflow",
    X_nodes: np.ndarray | None = None,
    Y_nodes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Full semi-discrete RHS using actual interior face exchange.

        RHS = volume_rhs + surface_rhs
    """
    q_work = np.asarray(q_elem, dtype=float)

    volume_rhs = volume_term_split_conservative(
        q_elem=q_work,
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
        use_numba=use_numba,
        split_cache=volume_split_cache,
    )

    surface_rhs, surface_diag = surface_term_from_exchange(
        q_elem=q_work,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_boundary,
        velocity=velocity,
        t=t,
        tau=tau,
        tau_interior=tau_interior,
        tau_qb=tau_qb,
        compute_mismatches=compute_mismatches,
        return_diagnostics=return_diagnostics,
        use_numba=use_numba,
        surface_backend=surface_backend,
        surface_cache=surface_cache,
        ndotV_precomputed=ndotV_precomputed,
        ndotV_flat_precomputed=ndotV_flat_precomputed,
        surface_inverse_mass_T=surface_inverse_mass_T,
        physical_boundary_mode=physical_boundary_mode,
        face_order_mode=face_order_mode,
        q_boundary_correction=q_boundary_correction,
        q_boundary_correction_mode=q_boundary_correction_mode,
        X_nodes=X_nodes,
        Y_nodes=Y_nodes,
    )

    if not return_diagnostics:
        total_rhs = volume_rhs
        total_rhs += surface_rhs
        return total_rhs, {}

    total_rhs = volume_rhs + surface_rhs

    diagnostics = dict(surface_diag)
    diagnostics["volume_rhs"] = volume_rhs
    diagnostics["surface_rhs"] = surface_rhs
    diagnostics["total_rhs"] = total_rhs

    return total_rhs, diagnostics
