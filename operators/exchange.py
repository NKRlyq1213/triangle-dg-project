from __future__ import annotations

import numpy as np

from operators.trace_policy import (
    evaluate_embedded_face_values,
    evaluate_projected_face_values,
)


def evaluate_all_face_values(
    u_elem: np.ndarray,
    trace: dict,
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
    out = np.zeros((K, 3, nfp), dtype=float)

    mode = str(trace.get("trace_mode", "")).lower().strip()

    for k in range(K):
        uk = u_elem[k]

        if mode == "embedded":
            fvals = evaluate_embedded_face_values(uk, trace)
        elif mode == "projected":
            fvals = evaluate_projected_face_values(uk, trace)
        else:
            raise ValueError("trace['trace_mode'] must be 'embedded' or 'projected'.")

        for face_id in (1, 2, 3):
            vals = np.asarray(fvals[face_id], dtype=float).reshape(-1)
            if vals.size != nfp:
                raise ValueError(
                    f"Face {face_id} of element {k} has {vals.size} values, expected {nfp}."
                )
            out[k, face_id - 1, :] = vals

    return out


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

    uM = evaluate_all_face_values(u_elem, trace)
    uP = np.full_like(uM, fill_value=boundary_fill_value, dtype=float)

    K, _, nfp = uM.shape

    for k in range(K):
        for jf in range(3):
            if is_boundary[k, jf]:
                continue

            nbr = int(EToE[k, jf])
            nbr_f = int(EToF[k, jf])  # 1-based
            vals = uM[nbr, nbr_f - 1, :].copy()

            if face_flip[k, jf]:
                vals = vals[::-1]

            uP[k, jf, :] = vals

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