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
                else:
                    qP = q_elem[nbr_k, nbr_node_ids_flat[face_idx, i]]

                ndv = ndotV_flat[face_idx, i]
                if ndv < 0.0:
                    surface_rhs[k, m_id] += (
                        scale
                        * owner_wratio_flat[face_idx, i]
                        * ndv
                        * (qM - qP)
                    )


    @njit(cache=True)
    def _face_major_surface_rhs_pair_kernel_inplace(
        q_elem: np.ndarray,
        ndotV_flat: np.ndarray,
        qB_flat: np.ndarray,
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

                if ndv < 0.0:
                    surface_rhs[k, m_id] += (
                        scale
                        * owner_wratio_flat[face_idx, i]
                        * ndv
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
                if ndv_a < 0.0:
                    surface_rhs[ka, a_mid] += (
                        scale_a
                        * owner_wratio_flat[fa, i]
                        * ndv_a
                        * (qMa - qPa)
                    )

                qMb = q_elem[kb, b_mid]
                qPb = q_elem[ka, b_nid]
                ndv_b = ndotV_flat[fb, i]
                if ndv_b < 0.0:
                    surface_rhs[kb, b_mid] += (
                        scale_b
                        * owner_wratio_flat[fb, i]
                        * ndv_b
                        * (qMb - qPb)
                    )

else:
    _surface_rhs_kernel_inplace = None
    _face_major_surface_rhs_flat_kernel_inplace = None
    _face_major_surface_rhs_pair_kernel_inplace = None


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


def upwind_flux_and_penalty(
    ndotV: np.ndarray,
    qM: np.ndarray,
    qP: np.ndarray,
    tau: float = 0.0,
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

    f = ndotV * qM
    fstar = 0.5 * (ndotV * qM + ndotV * qP) + 0.5 * (1.0 - tau) * np.abs(ndotV) * (qM - qP)
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


def _apply_boundary_q_correction(
    q_boundary_exact: np.ndarray,
    *,
    x_face: np.ndarray,
    y_face: np.ndarray,
    t: float,
    qM: np.ndarray,
    ndotV: np.ndarray,
    is_boundary: np.ndarray,
    q_boundary_correction,
    q_boundary_correction_mode: str,
) -> np.ndarray:
    """
    Apply user-supplied boundary correction to exact boundary traces.

    Expected callback signature:
        corr = q_boundary_correction(x_face, y_face, t, qM, ndotV, is_boundary, q_boundary_exact)

    where corr can be broadcast to shape (K, 3, Nfp).
    """
    if q_boundary_correction is None:
        return q_boundary_exact

    mode = str(q_boundary_correction_mode).strip().lower()
    if mode not in ("inflow", "boundary", "all"):
        raise ValueError("q_boundary_correction_mode must be one of: 'inflow', 'boundary', 'all'.")

    corr = q_boundary_correction(
        x_face,
        y_face,
        t,
        qM,
        ndotV,
        is_boundary,
        q_boundary_exact,
    )
    corr = np.asarray(corr, dtype=float)
    if corr.shape != q_boundary_exact.shape:
        try:
            corr = np.broadcast_to(corr, q_boundary_exact.shape)
        except ValueError as exc:
            raise ValueError(
                "q_boundary_correction must return an array broadcastable to shape (K, 3, Nfp)."
            ) from exc

    qB = np.array(q_boundary_exact, dtype=float, copy=True)
    bd = np.asarray(is_boundary, dtype=bool)[:, :, None]
    if mode == "inflow":
        mask = bd & (ndotV < 0.0)
    else:
        mask = np.broadcast_to(bd, q_boundary_exact.shape)
    qB[mask] += corr[mask]
    return qB


def build_surface_exchange_cache(
    rule: dict,
    trace: dict,
    conn: dict,
    face_geom: dict,
) -> dict:
    """
    Precompute static data used by the surface exchange term.

    The cache is valid as long as rule/trace/connectivity/face geometry do not
    change (typically fixed for one mesh level in a time loop).
    """
    if trace.get("trace_mode", None) != "embedded":
        raise ValueError("Surface cache currently supports embedded traces only.")

    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    inv_ws = 1.0 / ws

    ids_faces = []
    we_faces = []
    wr_faces = []
    ids_unique = []
    nfp = int(trace["nfp"])

    for face_id in (1, 2, 3):
        ids = np.asarray(trace["face_node_ids"][face_id], dtype=int).reshape(-1)
        if ids.size != nfp:
            raise ValueError("Trace face-node count must match trace['nfp'].")

        we = np.asarray(trace["face_weights"][face_id], dtype=float).reshape(-1)
        if we.size != nfp:
            raise ValueError("Trace face-weight count must match trace['nfp'].")

        ids_faces.append(ids)
        we_faces.append(we)
        wr_faces.append(we * inv_ws[ids])
        ids_unique.append(np.unique(ids).size == ids.size)

    area = np.asarray(face_geom["area"], dtype=float).reshape(-1)
    length = np.asarray(face_geom["length"], dtype=float)
    if length.shape != (area.size, 3):
        raise ValueError("face_geom['length'] must have shape (K, 3).")

    EToE = np.asarray(conn["EToE"], dtype=int)
    EToF = np.asarray(conn["EToF"], dtype=int)
    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)
    face_flip = np.asarray(conn["face_flip"], dtype=bool)

    K = int(EToE.shape[0])
    if EToE.shape != (K, 3):
        raise ValueError("conn['EToE'] must have shape (K, 3).")
    if EToF.shape != (K, 3) or is_boundary.shape != (K, 3) or face_flip.shape != (K, 3):
        raise ValueError("conn face arrays must have shape (K, 3).")

    x_face = np.asarray(face_geom["x_face"], dtype=float)
    y_face = np.asarray(face_geom["y_face"], dtype=float)
    nx = np.asarray(face_geom["nx"], dtype=float)
    ny = np.asarray(face_geom["ny"], dtype=float)
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

    ids_faces_numba = tuple(np.ascontiguousarray(ids, dtype=np.int64) for ids in ids_faces)
    wr_faces_numba = tuple(np.ascontiguousarray(wr, dtype=np.float64) for wr in wr_faces)

    for jf in range(3):
        rows = np.arange(jf, nf, 3, dtype=np.int64)
        owner_node_ids_flat_numba[rows, :] = ids_faces_numba[jf][None, :]
        owner_wratio_flat_numba[rows, :] = wr_faces_numba[jf][None, :]

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

    return {
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
        "ids_faces_numba": ids_faces_numba,
        "wr_faces_numba": wr_faces_numba,
        "area_numba": area_numba,
        "length_numba": length_numba,
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
        "nbr_elem_flat_numba": np.ascontiguousarray(nbr_elem_flat_numba, dtype=np.int64),
        "nbr_node_ids_flat_numba": np.ascontiguousarray(nbr_node_ids_flat_numba, dtype=np.int64),
        "boundary_face_idx_numba": boundary_face_idx_numba,
        "pair_face_a_numba": pair_face_a_numba,
        "pair_face_b_numba": pair_face_b_numba,
    }


def _lift_surface_penalty_to_volume(
    p: np.ndarray,
    cache: dict,
    use_numba: bool | None,
    surface_inverse_mass_T: np.ndarray | None = None,
) -> np.ndarray:
    K = int(cache["K"])
    Np = int(cache["Np"])

    if surface_inverse_mass_T is not None:
        surface_inverse_mass_t = np.asarray(surface_inverse_mass_T, dtype=float)
        if (
            surface_inverse_mass_t.ndim != 2
            or surface_inverse_mass_t.shape[0] != Np
            or surface_inverse_mass_t.shape[1] != Np
        ):
            raise ValueError("surface_inverse_mass_T must be a square (Np, Np) array.")

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
    compute_mismatches: bool = True,
    return_diagnostics: bool = True,
    use_numba: bool | None = None,
    surface_backend: str | None = None,
    surface_cache: dict | None = None,
    ndotV_precomputed: np.ndarray | None = None,
    ndotV_flat_precomputed: np.ndarray | None = None,
    surface_inverse_mass_T: np.ndarray | None = None,
    q_boundary_correction=None,
    q_boundary_correction_mode: str = "inflow",
    volume_split_cache: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Surface term using actual interior face exchange plus exact inflow boundary data.

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

    cache = build_surface_exchange_cache(rule, trace, conn, face_geom) if surface_cache is None else surface_cache

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
        if return_diagnostics:
            u_face, v_face = velocity(x_face, y_face, t)
            u_face = np.asarray(u_face, dtype=float)
            v_face = np.asarray(v_face, dtype=float)

    qB = q_boundary(x_face, y_face, t)
    qB = np.asarray(qB, dtype=float)

    if ndotV.shape != (K, 3, nfp) or qB.shape != (K, 3, nfp):
        raise ValueError("velocity/boundary callbacks must return arrays with shape (K, 3, Nfp).")

    qM_face_prefetched = None
    if q_boundary_correction is not None:
        ids_faces = cache["ids_faces"]
        qM_face_prefetched = np.empty((K, 3, nfp), dtype=float)
        for jf in range(3):
            qM_face_prefetched[:, jf, :] = q_elem[:, ids_faces[jf]]

        qB = _apply_boundary_q_correction(
            q_boundary_exact=qB,
            x_face=x_face,
            y_face=y_face,
            t=t,
            qM=qM_face_prefetched,
            ndotV=ndotV,
            is_boundary=cache["is_boundary"],
            q_boundary_correction=q_boundary_correction,
            q_boundary_correction_mode=q_boundary_correction_mode,
        )

    if backend == "face-major":
        if _should_use_numba(use_numba) and (not return_diagnostics) and (surface_inverse_mass_T is None):
            surface_rhs = np.zeros((K, Np), dtype=float)

            if ndotV_flat_precomputed is None:
                ndotV_flat = ndotV.reshape(K * 3, nfp)
                if ndotV_flat.dtype != np.float64 or (not ndotV_flat.flags.c_contiguous):
                    ndotV_flat = np.ascontiguousarray(ndotV_flat, dtype=np.float64)
            else:
                ndotV_flat = _as_c_f64(ndotV_flat_precomputed)
                if ndotV_flat.shape != (K * 3, nfp):
                    raise ValueError("ndotV_flat_precomputed must have shape (K*3, Nfp).")

            qB_flat = qB.reshape(K * 3, nfp)
            if qB_flat.dtype != np.float64 or (not qB_flat.flags.c_contiguous):
                qB_flat = np.ascontiguousarray(qB_flat, dtype=np.float64)

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
                    qB_flat=qB_flat,
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
                    qB_flat=qB_flat,
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
        qB_flat = qB.reshape(K * 3, nfp)

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
            qP_flat[boundary_flat] = qB_flat[boundary_flat]
        else:
            qP_flat = np.empty_like(qM_flat)
            qP_flat[interior_flat] = qM_flat[nbr_flat[interior_flat]]

            flip_interior = flip_flat & interior_flat
            if np.any(flip_interior):
                qP_flat[flip_interior] = qP_flat[flip_interior, ::-1]

            qP_flat[boundary_flat] = qB_flat[boundary_flat]

        p = np.minimum(ndotV_flat, 0.0) * (qM_flat - qP_flat)
        p = p.reshape(K, 3, nfp)
        surface_rhs = _lift_surface_penalty_to_volume(
            p,
            cache,
            use_numba=use_numba,
            surface_inverse_mass_T=surface_inverse_mass_T,
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
            "qP_before_boundary_fill": qP_before,
            "qP": qP_filled,
            "qB": qB,
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

    is_boundary = np.asarray(cache["is_boundary"], dtype=bool)
    qP_filled = fill_exterior_state(
        qP_exchange=qP,
        is_boundary=is_boundary,
        q_boundary_exact=qB,
    )

    # pure upwind simplification from the document
    p = upwind_penalty_simplified(
        ndotV=ndotV,
        qM=qM,
        qP=qP_filled,
    )
    surface_rhs = _lift_surface_penalty_to_volume(
        p,
        cache,
        use_numba=use_numba,
        surface_inverse_mass_T=surface_inverse_mass_T,
    )

    if not return_diagnostics:
        return surface_rhs, {}

    mismatches = interior_face_pair_mismatches(paired) if compute_mismatches else []

    diagnostics = {
        "qM": qM,
        "qP_before_boundary_fill": np.asarray(paired["uP"], dtype=float),
        "qP": qP_filled,
        "qB": qB,
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
    compute_mismatches: bool = True,
    return_diagnostics: bool = True,
    use_numba: bool | None = None,
    state_projector: np.ndarray | None = None,
    state_projector_T: np.ndarray | None = None,
    state_projector_mode: str = "both",
    surface_backend: str | None = None,
    surface_cache: dict | None = None,
    ndotV_precomputed: np.ndarray | None = None,
    ndotV_flat_precomputed: np.ndarray | None = None,
    surface_inverse_mass_T: np.ndarray | None = None,
    volume_split_cache: dict | None = None,
    q_boundary_correction=None,
    q_boundary_correction_mode: str = "inflow",
) -> tuple[np.ndarray, dict]:
    """
    Full semi-discrete RHS using actual interior face exchange.

        RHS = volume_rhs + surface_rhs
    """
    q_work = np.asarray(q_elem, dtype=float)
    projector_mat = None
    projector_mode = "both"
    if (state_projector is not None) and (state_projector_T is not None):
        raise ValueError("Provide only one of state_projector or state_projector_T.")

    if state_projector_T is not None:
        projector_mat = np.asarray(state_projector_T, dtype=float)
        if projector_mat.ndim != 2 or projector_mat.shape[0] != projector_mat.shape[1]:
            raise ValueError("state_projector_T must be a square 2D array.")
        if projector_mat.shape[0] != q_work.shape[1]:
            raise ValueError("state_projector_T size must match q_elem.shape[1].")
        projector_mode = str(state_projector_mode).strip().lower()
        if projector_mode not in ("both", "pre", "post"):
            raise ValueError("state_projector_mode must be one of: 'both', 'pre', 'post'.")
        if projector_mode in ("both", "pre"):
            q_work = q_work @ projector_mat
    elif state_projector is not None:
        projector = np.asarray(state_projector, dtype=float)
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise ValueError("state_projector must be a square 2D array.")
        if projector.shape[0] != q_work.shape[1]:
            raise ValueError("state_projector size must match q_elem.shape[1].")
        projector_mat = projector.T
        projector_mode = str(state_projector_mode).strip().lower()
        if projector_mode not in ("both", "pre", "post"):
            raise ValueError("state_projector_mode must be one of: 'both', 'pre', 'post'.")
        if projector_mode in ("both", "pre"):
            q_work = q_work @ projector_mat

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
        compute_mismatches=compute_mismatches,
        return_diagnostics=return_diagnostics,
        use_numba=use_numba,
        surface_backend=surface_backend,
        surface_cache=surface_cache,
        ndotV_precomputed=ndotV_precomputed,
        ndotV_flat_precomputed=ndotV_flat_precomputed,
        surface_inverse_mass_T=surface_inverse_mass_T,
        q_boundary_correction=q_boundary_correction,
        q_boundary_correction_mode=q_boundary_correction_mode,
    )

    if not return_diagnostics:
        total_rhs = volume_rhs
        total_rhs += surface_rhs
        if projector_mat is not None and projector_mode in ("both", "post"):
            total_rhs = total_rhs @ projector_mat
        return total_rhs, {}

    total_rhs = volume_rhs + surface_rhs
    if projector_mat is not None and projector_mode in ("both", "post"):
        total_rhs = total_rhs @ projector_mat

    diagnostics = dict(surface_diag)
    diagnostics["volume_rhs"] = volume_rhs
    diagnostics["surface_rhs"] = surface_rhs
    diagnostics["total_rhs"] = total_rhs

    return total_rhs, diagnostics