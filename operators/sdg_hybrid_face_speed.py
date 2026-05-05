from __future__ import annotations

import numpy as np

from operators.sdg_edge_streamfunction_flux import reference_face_parameters


def volume_normal_flux_on_faces(op) -> np.ndarray:
    """
    Compute one-sided volume-derived normal flux:

        a_vol = n_x Fx_area + n_y Fy_area

    using each element's local outward normal.

    Shape:
        (K, 3, Nfp)
    """
    base = op.base
    conn = base.conn_seam
    ref_face = base.ref_face

    K = base.Fx_area.shape[0]
    nfp = ref_face.nfp

    a_vol = np.zeros((K, 3, nfp), dtype=float)

    for k in range(K):
        for f in range(3):
            ids = np.asarray(ref_face.face_node_ids[f], dtype=int)

            a_vol[k, f, :] = (
                conn.face_normal_x[k, f] * base.Fx_area[k, ids]
                + conn.face_normal_y[k, f] * base.Fy_area[k, ids]
            )

    return a_vol


def corner_trace_mask(op, *, corner_tol: float) -> np.ndarray:
    """
    Mask near-corner face trace nodes.

    Shape:
        (3, Nfp)

    A node is considered near-corner if:

        t <= corner_tol or t >= 1 - corner_tol.
    """
    if corner_tol < 0.0 or corner_tol > 0.5:
        raise ValueError("corner_tol must be in [0, 0.5].")

    ts = reference_face_parameters(op.base.rs_nodes, op.base.ref_face)

    nfp = op.base.ref_face.nfp
    mask = np.zeros((3, nfp), dtype=bool)

    for f in range(3):
        t = np.asarray(ts[f], dtype=float)
        mask[f, :] = (t <= corner_tol) | (t >= 1.0 - corner_tol)

    return mask


def build_hybrid_face_speed(
    op,
    fast_cache: dict,
    *,
    speed_mode: str = "corner_volavg",
    corner_tol: float = 0.10,
) -> np.ndarray:
    """
    Build a pairwise-consistent face speed array for fast RHS.

    Parameters
    ----------
    speed_mode:
        edge:
            Use original edge-streamfunction speed.
        volavg:
            Use pairwise averaged volume-derived speed on all face nodes.
        corner_volavg:
            Use pairwise averaged volume-derived speed only near corners,
            and edge-streamfunction speed elsewhere.

    Returns
    -------
    a_face:
        Local outward normal speed, shape (K, 3, Nfp), pairwise consistent.
    """
    mode = str(speed_mode).strip().lower()

    if mode not in ("edge", "volavg", "corner_volavg"):
        raise ValueError("speed_mode must be one of: edge, volavg, corner_volavg.")

    a_edge = np.asarray(fast_cache["a_face"], dtype=float)

    if mode == "edge":
        return np.ascontiguousarray(a_edge.copy(), dtype=float)

    base = op.base
    conn = base.conn_seam

    a_vol = volume_normal_flux_on_faces(op)

    a_hybrid = np.ascontiguousarray(a_edge.copy(), dtype=float)

    neighbor_elem = np.asarray(conn.neighbor_elem, dtype=int)
    neighbor_face = np.asarray(conn.neighbor_face, dtype=int)
    neighbor_face_locs = np.asarray(fast_cache["neighbor_face_locs"], dtype=int)

    K = a_hybrid.shape[0]
    nfp = a_hybrid.shape[2]

    near_corner = corner_trace_mask(op, corner_tol=corner_tol)

    for k in range(K):
        for f in range(3):
            kp = int(neighbor_elem[k, f])
            fp = int(neighbor_face[k, f])

            if kp < 0:
                # Closed sphere should have no unmatched boundary faces.
                # Keep edge speed on unmatched boundaries.
                continue

            # Count each face pair once.
            if (kp < k) or ((kp == k) and (fp < f)):
                continue

            for i in range(nfp):
                locP = int(neighbor_face_locs[k, f, i])
                if locP < 0:
                    continue

                use_volavg = mode == "volavg"

                if mode == "corner_volavg":
                    use_volavg = bool(near_corner[f, i] or near_corner[fp, locP])

                if not use_volavg:
                    continue

                aM = float(a_vol[k, f, i])

                # Neighbor local normal is opposite owner normal.
                aP_owner = -float(a_vol[kp, fp, locP])

                a_common_owner = 0.5 * (aM + aP_owner)

                # Assign pairwise-consistent local speeds.
                a_hybrid[k, f, i] = a_common_owner
                a_hybrid[kp, fp, locP] = -a_common_owner

    return np.ascontiguousarray(a_hybrid, dtype=float)


def apply_hybrid_face_speed_to_fast_cache(
    op,
    fast_cache: dict,
    *,
    speed_mode: str = "corner_volavg",
    corner_tol: float = 0.10,
) -> dict:
    """
    Mutate fast_cache['a_face'] to use hybrid face speed.

    Returns the same cache object for convenience.
    """
    fast_cache["a_face"] = build_hybrid_face_speed(
        op,
        fast_cache,
        speed_mode=speed_mode,
        corner_tol=corner_tol,
    )

    fast_cache["speed_mode"] = str(speed_mode)
    fast_cache["corner_tol"] = float(corner_tol)

    return fast_cache
