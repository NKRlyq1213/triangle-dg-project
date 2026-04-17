from __future__ import annotations

import importlib
import numpy as np

try:
    njit = importlib.import_module("numba").njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    _NUMBA_AVAILABLE = False

def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _evaluate_embedded_face_values_kernel_inplace(
        u_elem: np.ndarray,
        ids_f1: np.ndarray,
        ids_f2: np.ndarray,
        ids_f3: np.ndarray,
        out: np.ndarray,
    ) -> None:
        K = u_elem.shape[0]
        nfp = ids_f1.size

        for k in range(K):
            for i in range(nfp):
                out[k, 0, i] = u_elem[k, ids_f1[i]]
                out[k, 1, i] = u_elem[k, ids_f2[i]]
                out[k, 2, i] = u_elem[k, ids_f3[i]]


    @njit(cache=True)
    def _pair_face_traces_kernel_inplace(
        uM: np.ndarray,
        EToE: np.ndarray,
        EToF: np.ndarray,
        is_boundary: np.ndarray,
        face_flip: np.ndarray,
        boundary_fill_value: float,
        uP: np.ndarray,
    ) -> None:
        K = uM.shape[0]
        nfp = uM.shape[2]

        for k in range(K):
            for jf in range(3):
                for i in range(nfp):
                    uP[k, jf, i] = boundary_fill_value

        for k in range(K):
            for jf in range(3):
                if is_boundary[k, jf]:
                    continue

                nbr = EToE[k, jf]
                nbr_jf = EToF[k, jf] - 1

                if face_flip[k, jf]:
                    for i in range(nfp):
                        uP[k, jf, i] = uM[nbr, nbr_jf, nfp - 1 - i]
                else:
                    for i in range(nfp):
                        uP[k, jf, i] = uM[nbr, nbr_jf, i]

else:
    _evaluate_embedded_face_values_kernel_inplace = None
    _pair_face_traces_kernel_inplace = None


def evaluate_all_face_values(
    u_elem: np.ndarray,
    trace: dict,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    """
    Evaluate all local face traces for all elements.

    Parameters
    ----------
    u_elem : np.ndarray
        Element volume values.
        Shape:
            (K, Np)
    trace : dict
        Trace descriptor returned by build_trace_policy(...)

    Returns
    -------
    np.ndarray
        Shape (K, 3, Nfp)
        out[k, f-1, :] = local face values on face f of element k
    """
    u_elem = np.asarray(u_elem, dtype=float)
    if u_elem.ndim != 2:
        raise ValueError("u_elem must have shape (K, Np).")

    K = u_elem.shape[0]
    nfp = int(trace["nfp"])
    if out is None:
        out = np.zeros((K, 3, nfp), dtype=float)
    else:
        out = np.asarray(out, dtype=float)
        if out.shape != (K, 3, nfp):
            raise ValueError("out must have shape (K, 3, Nfp) matching u_elem and trace['nfp'].")

    mode = str(trace.get("trace_mode", "")).lower().strip()

    if mode == "embedded":
        ids_f1 = np.asarray(trace["face_node_ids"][1], dtype=np.int64).reshape(-1)
        ids_f2 = np.asarray(trace["face_node_ids"][2], dtype=np.int64).reshape(-1)
        ids_f3 = np.asarray(trace["face_node_ids"][3], dtype=np.int64).reshape(-1)
        if ids_f1.size != nfp:
            raise ValueError(f"Face 1 has {ids_f1.size} trace nodes, expected {nfp}.")
        if ids_f2.size != nfp:
            raise ValueError(f"Face 2 has {ids_f2.size} trace nodes, expected {nfp}.")
        if ids_f3.size != nfp:
            raise ValueError(f"Face 3 has {ids_f3.size} trace nodes, expected {nfp}.")

        if _should_use_numba(use_numba):
            u_numba = u_elem
            if u_numba.dtype != np.float64 or (not u_numba.flags.c_contiguous):
                u_numba = np.ascontiguousarray(u_numba, dtype=np.float64)

            out_numba = out
            if out_numba.dtype != np.float64 or (not out_numba.flags.c_contiguous):
                out_numba = np.ascontiguousarray(out_numba, dtype=np.float64)

            _evaluate_embedded_face_values_kernel_inplace(
                u_elem=u_numba,
                ids_f1=np.ascontiguousarray(ids_f1, dtype=np.int64),
                ids_f2=np.ascontiguousarray(ids_f2, dtype=np.int64),
                ids_f3=np.ascontiguousarray(ids_f3, dtype=np.int64),
                out=out_numba,
            )

            if out_numba is not out:
                out[...] = out_numba
            return out

        out[:, 0, :] = u_elem[:, ids_f1]
        out[:, 1, :] = u_elem[:, ids_f2]
        out[:, 2, :] = u_elem[:, ids_f3]
        return out

    if mode == "projected":
        for face_id in (1, 2, 3):
            E_edge = np.asarray(trace["face_operators"][face_id], dtype=float)
            vals = u_elem @ E_edge.T
            if vals.shape != (K, nfp):
                raise ValueError(
                    f"Projected face {face_id} produced shape {vals.shape}, expected {(K, nfp)}."
                )
            out[:, face_id - 1, :] = vals
        return out

    raise ValueError("trace['trace_mode'] must be 'embedded' or 'projected'.")


def unique_interior_face_pairs(conn: dict) -> list[tuple[int, int, int, int]]:
    """
    Return unique interior face pairs.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Each tuple is:
            (k, f, nbr, nbr_f)
        where
            k, nbr are 0-based element ids
            f, nbr_f are 1-based local face ids

        Each physical interior face appears exactly once.
    """
    EToE = np.asarray(conn["EToE"], dtype=int)
    EToF = np.asarray(conn["EToF"], dtype=int)
    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)

    if EToE.ndim != 2 or EToE.shape[1] != 3:
        raise ValueError("conn['EToE'] must have shape (K, 3).")
    if EToF.shape != EToE.shape:
        raise ValueError("conn['EToF'] must have shape (K, 3).")
    if is_boundary.shape != EToE.shape:
        raise ValueError("conn['is_boundary'] must have shape (K, 3).")

    pairs = []
    seen = set()

    K = EToE.shape[0]
    for k in range(K):
        for jf in range(3):
            f = jf + 1
            if is_boundary[k, jf]:
                continue

            nbr = int(EToE[k, jf])
            nbr_f = int(EToF[k, jf])

            key = tuple(sorted(((k, f), (nbr, nbr_f))))
            if key in seen:
                continue
            seen.add(key)

            pairs.append((k, f, nbr, nbr_f))

    return pairs


def pair_face_traces(
    u_elem: np.ndarray,
    conn: dict,
    trace: dict,
    boundary_fill_value: float = np.nan,
    out_uM: np.ndarray | None = None,
    out_uP: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> dict:
    """
    Pair local face traces with neighbor-aligned face traces.

    Parameters
    ----------
    u_elem : np.ndarray
        Shape (K, Np), volume values for each element.
    conn : dict
        Connectivity dictionary returned by build_face_connectivity(...).
    trace : dict
        Trace descriptor returned by build_trace_policy(...).
    boundary_fill_value : float
        Fill value for uP on boundary faces.

    Returns
    -------
    dict
        Contains:
            uM : np.ndarray, shape (K, 3, Nfp)
                Local face trace values
            uP : np.ndarray, shape (K, 3, Nfp)
                Neighbor-aligned face trace values
                Boundary faces are filled with boundary_fill_value
            EToE, EToF, is_boundary, face_flip
            face_t, face_weights
            nfp
            trace_mode
            table
    """
    u_elem = np.asarray(u_elem, dtype=float)
    if u_elem.ndim != 2:
        raise ValueError("u_elem must have shape (K, Np).")

    EToE = np.asarray(conn["EToE"], dtype=int)
    EToF = np.asarray(conn["EToF"], dtype=int)
    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)
    face_flip = np.asarray(conn["face_flip"], dtype=bool)

    if EToE.shape != (u_elem.shape[0], 3):
        raise ValueError("conn['EToE'] shape must match (K, 3).")
    if EToF.shape != EToE.shape:
        raise ValueError("conn['EToF'] shape must match (K, 3).")
    if is_boundary.shape != EToE.shape:
        raise ValueError("conn['is_boundary'] shape must match (K, 3).")
    if face_flip.shape != EToE.shape:
        raise ValueError("conn['face_flip'] shape must match (K, 3).")

    uM = evaluate_all_face_values(u_elem, trace, out=out_uM, use_numba=use_numba)

    if out_uP is None:
        uP = np.empty_like(uM, dtype=float)
    else:
        uP = np.asarray(out_uP, dtype=float)
        if uP.shape != uM.shape:
            raise ValueError("out_uP must have the same shape as evaluated traces (K, 3, Nfp).")

    if _should_use_numba(use_numba):
        _pair_face_traces_kernel_inplace(
            uM=uM,
            EToE=EToE,
            EToF=EToF,
            is_boundary=is_boundary,
            face_flip=face_flip,
            boundary_fill_value=float(boundary_fill_value),
            uP=uP,
        )
    else:
        uP.fill(float(boundary_fill_value))

        interior = ~is_boundary
        if np.any(interior):
            k_idx, jf_idx = np.where(interior)
            nbr = EToE[k_idx, jf_idx]
            nbr_jf = EToF[k_idx, jf_idx] - 1

            gathered = uM[nbr, nbr_jf, :]
            uP[k_idx, jf_idx, :] = gathered

            flip_mask = face_flip[k_idx, jf_idx]
            if np.any(flip_mask):
                kf = k_idx[flip_mask]
                jff = jf_idx[flip_mask]
                uP[kf, jff, :] = uP[kf, jff, ::-1]

    return {
        "uM": uM,
        "uP": uP,
        "EToE": EToE,
        "EToF": EToF,
        "is_boundary": is_boundary,
        "face_flip": face_flip,
        "face_t": trace["face_t"],
        "face_weights": trace["face_weights"],
        "nfp": int(trace["nfp"]),
        "trace_mode": trace["trace_mode"],
        "table": trace["table"],
    }


def interior_face_pair_mismatches(
    paired: dict,
) -> list[dict]:
    """
    Diagnostic helper: compute max mismatch on each unique interior face pair.

    Returns
    -------
    list[dict]
        Each item contains:
            k, f, nbr, nbr_f, max_abs_mismatch
    """
    conn = {
        "EToE": paired["EToE"],
        "EToF": paired["EToF"],
        "is_boundary": paired["is_boundary"],
    }
    uM = np.asarray(paired["uM"], dtype=float)
    uP = np.asarray(paired["uP"], dtype=float)

    out = []
    for k, f, nbr, nbr_f in unique_interior_face_pairs(conn):
        diff = uM[k, f - 1, :] - uP[k, f - 1, :]
        out.append(
            {
                "k": int(k),
                "f": int(f),
                "nbr": int(nbr),
                "nbr_f": int(nbr_f),
                "max_abs_mismatch": float(np.max(np.abs(diff))),
            }
        )
    
    return out
