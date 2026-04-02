from __future__ import annotations

from collections import defaultdict
import numpy as np


def local_face_vertex_ids(elem_vids: np.ndarray) -> np.ndarray:
    """
    Return the 3 local faces of one triangle using the existing edge convention.

    Face convention / edge convention
    ---------------------------------
    face 1: v2 -> v3
    face 2: v3 -> v1
    face 3: v1 -> v2

    Parameters
    ----------
    elem_vids : np.ndarray
        Shape (3,), the three global vertex ids of one element.

    Returns
    -------
    np.ndarray
        Shape (3, 2), ordered local faces:
            [[v2, v3],
             [v3, v1],
             [v1, v2]]
    """
    elem_vids = np.asarray(elem_vids, dtype=int).reshape(-1)
    if elem_vids.shape[0] != 3:
        raise ValueError("elem_vids must have shape (3,).")

    v1, v2, v3 = elem_vids
    return np.array(
        [
            [v2, v3],  # face 1
            [v3, v1],  # face 2
            [v1, v2],  # face 3
        ],
        dtype=int,
    )


def all_face_vertex_ids(EToV: np.ndarray) -> np.ndarray:
    """
    Build oriented face-vertex ids for all elements.

    Parameters
    ----------
    EToV : np.ndarray
        Shape (K, 3), element-to-vertex connectivity.

    Returns
    -------
    np.ndarray
        Shape (K, 3, 2), where:
            out[k, f-1] = [va, vb]
        is the oriented global-vertex pair for face f of element k.
    """
    EToV = np.asarray(EToV, dtype=int)
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    K = EToV.shape[0]
    out = np.zeros((K, 3, 2), dtype=int)
    for k in range(K):
        out[k] = local_face_vertex_ids(EToV[k])
    return out


def face_midpoints(
    VX: np.ndarray,
    VY: np.ndarray,
    face_vertex_ids: np.ndarray,
) -> np.ndarray:
    """
    Compute physical face midpoints.

    Parameters
    ----------
    VX, VY : np.ndarray
        Global vertex coordinates, shape (Nv,)
    face_vertex_ids : np.ndarray
        Shape (K, 3, 2)

    Returns
    -------
    np.ndarray
        Shape (K, 3, 2), midpoint coordinates.
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    face_vertex_ids = np.asarray(face_vertex_ids, dtype=int)

    if face_vertex_ids.ndim != 3 or face_vertex_ids.shape[1:] != (3, 2):
        raise ValueError("face_vertex_ids must have shape (K, 3, 2).")

    K = face_vertex_ids.shape[0]
    mids = np.zeros((K, 3, 2), dtype=float)

    for k in range(K):
        for jf in range(3):
            va, vb = face_vertex_ids[k, jf]
            mids[k, jf, 0] = 0.5 * (VX[va] + VX[vb])
            mids[k, jf, 1] = 0.5 * (VY[va] + VY[vb])

    return mids


def _classify_boundary_faces_box(
    VX: np.ndarray,
    VY: np.ndarray,
    boundary_faces: np.ndarray,
    face_midpoints_xy: np.ndarray,
    tol: float = 1e-12,
) -> dict[str, list[tuple[int, int]]]:
    """
    Classify boundary faces using the global bounding box.

    Notes
    -----
    This is a geometric classification helper.
    Boundary detection itself is still purely topological.
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    boundary_faces = np.asarray(boundary_faces, dtype=int)

    xmin = float(np.min(VX))
    xmax = float(np.max(VX))
    ymin = float(np.min(VY))
    ymax = float(np.max(VY))

    groups: dict[str, list[tuple[int, int]]] = {
        "left": [],
        "right": [],
        "bottom": [],
        "top": [],
        "boundary_default": [],
    }

    for k, f in boundary_faces:
        xmid, ymid = face_midpoints_xy[k, f - 1]

        matched = False
        if abs(xmid - xmin) <= tol:
            groups["left"].append((int(k), int(f)))
            matched = True
        if abs(xmid - xmax) <= tol:
            groups["right"].append((int(k), int(f)))
            matched = True
        if abs(ymid - ymin) <= tol:
            groups["bottom"].append((int(k), int(f)))
            matched = True
        if abs(ymid - ymax) <= tol:
            groups["top"].append((int(k), int(f)))
            matched = True

        if not matched:
            groups["boundary_default"].append((int(k), int(f)))

    return groups


def build_face_connectivity(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    classify_boundary: str | None = "box",
    tol: float = 1e-12,
) -> dict:
    """
    Build triangle-face connectivity on a conforming triangular mesh.

    Outputs
    -------
    EToE : np.ndarray
        Shape (K, 3), neighbor element index (0-based), or -1 on boundary.
    EToF : np.ndarray
        Shape (K, 3), neighbor face id (1-based), or -1 on boundary.
    is_boundary : np.ndarray
        Shape (K, 3), bool mask.
    face_flip : np.ndarray
        Shape (K, 3), whether neighbor face ordering must be reversed.
    face_vertex_ids : np.ndarray
        Shape (K, 3, 2), oriented vertex ids of local faces.
    face_midpoints : np.ndarray
        Shape (K, 3, 2), physical midpoint coordinates.
    boundary_faces : np.ndarray
        Shape (Nb, 2), rows are (elem_id, face_id), face_id is 1-based.
    boundary_groups : dict[str, list[tuple[int, int]]]
        Optional boundary classification.
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)

    if VX.shape != VY.shape:
        raise ValueError("VX and VY must have the same shape.")
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    K = EToV.shape[0]

    face_vids = all_face_vertex_ids(EToV)
    mids = face_midpoints(VX, VY, face_vids)

    EToE = -np.ones((K, 3), dtype=int)
    EToF = -np.ones((K, 3), dtype=int)  # face ids are 1-based in values
    is_boundary = np.ones((K, 3), dtype=bool)
    face_flip = np.zeros((K, 3), dtype=bool)

    # key: canonical undirected face; value: list of (elem_id, face_id, va, vb)
    edge_map: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)

    for k in range(K):
        for jf in range(3):
            f = jf + 1  # 1-based face id
            va, vb = face_vids[k, jf]
            key = (min(int(va), int(vb)), max(int(va), int(vb)))
            edge_map[key].append((k, f, int(va), int(vb)))

    boundary_faces_list: list[tuple[int, int]] = []

    for key, entries in edge_map.items():
        if len(entries) == 1:
            k, f, _, _ = entries[0]
            boundary_faces_list.append((k, f))
            continue

        if len(entries) != 2:
            raise ValueError(
                f"Non-conforming or invalid mesh: face key {key} appears {len(entries)} times."
            )

        (k1, f1, va1, vb1), (k2, f2, va2, vb2) = entries

        EToE[k1, f1 - 1] = k2
        EToF[k1, f1 - 1] = f2
        is_boundary[k1, f1 - 1] = False

        EToE[k2, f2 - 1] = k1
        EToF[k2, f2 - 1] = f1
        is_boundary[k2, f2 - 1] = False

        if va1 == vb2 and vb1 == va2:
            flip = True
        elif va1 == va2 and vb1 == vb2:
            flip = False
        else:
            raise ValueError(
                "Inconsistent face pairing encountered although canonical keys match."
            )

        face_flip[k1, f1 - 1] = flip
        face_flip[k2, f2 - 1] = flip

    boundary_faces = np.asarray(boundary_faces_list, dtype=int)
    if boundary_faces.size == 0:
        boundary_faces = boundary_faces.reshape(0, 2)

    if classify_boundary is None:
        boundary_groups = {}
    elif classify_boundary == "box":
        boundary_groups = _classify_boundary_faces_box(
            VX=VX,
            VY=VY,
            boundary_faces=boundary_faces,
            face_midpoints_xy=mids,
            tol=tol,
        )
    else:
        raise ValueError("classify_boundary must be None or 'box'.")

    return {
        "EToE": EToE,
        "EToF": EToF,
        "is_boundary": is_boundary,
        "face_flip": face_flip,
        "face_vertex_ids": face_vids,
        "face_midpoints": mids,
        "boundary_faces": boundary_faces,
        "boundary_groups": boundary_groups,
    }


def validate_face_connectivity(
    EToV: np.ndarray,
    conn: dict,
) -> dict[str, int]:
    """
    Validate basic invariants of face connectivity.

    Returns
    -------
    dict[str, int]
        Summary counters:
            n_elements
            n_total_local_faces
            n_boundary_faces
            n_unique_interior_faces
    """
    EToV = np.asarray(EToV, dtype=int)
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    EToE = np.asarray(conn["EToE"], dtype=int)
    EToF = np.asarray(conn["EToF"], dtype=int)
    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)
    face_flip = np.asarray(conn["face_flip"], dtype=bool)
    face_vids = np.asarray(conn["face_vertex_ids"], dtype=int)
    boundary_faces = np.asarray(conn["boundary_faces"], dtype=int)

    K = EToV.shape[0]

    if EToE.shape != (K, 3):
        raise ValueError("EToE must have shape (K, 3).")
    if EToF.shape != (K, 3):
        raise ValueError("EToF must have shape (K, 3).")
    if is_boundary.shape != (K, 3):
        raise ValueError("is_boundary must have shape (K, 3).")
    if face_flip.shape != (K, 3):
        raise ValueError("face_flip must have shape (K, 3).")
    if face_vids.shape != (K, 3, 2):
        raise ValueError("face_vertex_ids must have shape (K, 3, 2).")

    seen_pairs = set()
    n_total_local_faces = 3 * K
    n_boundary_faces = int(np.sum(is_boundary))

    for k in range(K):
        for jf in range(3):
            f = jf + 1
            nbr = EToE[k, jf]
            nbr_f = EToF[k, jf]

            if is_boundary[k, jf]:
                if nbr != -1 or nbr_f != -1:
                    raise ValueError(f"Boundary face ({k}, {f}) should have EToE=EToF=-1.")
                continue

            if not (0 <= nbr < K):
                raise ValueError(f"Interior face ({k}, {f}) has invalid neighbor index {nbr}.")
            if nbr_f not in {1, 2, 3}:
                raise ValueError(f"Interior face ({k}, {f}) has invalid neighbor face {nbr_f}.")

            # symmetry check
            if EToE[nbr, nbr_f - 1] != k:
                raise ValueError(f"Symmetry failure in EToE at ({k}, {f}).")
            if EToF[nbr, nbr_f - 1] != f:
                raise ValueError(f"Symmetry failure in EToF at ({k}, {f}).")

            # same undirected physical edge
            va, vb = face_vids[k, jf]
            vc, vd = face_vids[nbr, nbr_f - 1]
            key1 = (min(int(va), int(vb)), max(int(va), int(vb)))
            key2 = (min(int(vc), int(vd)), max(int(vc), int(vd)))
            if key1 != key2:
                raise ValueError(f"Paired faces ({k}, {f}) and ({nbr}, {nbr_f}) do not match geometrically.")

            # flip consistency
            should_flip = (va == vd and vb == vc)
            if face_flip[k, jf] != should_flip:
                raise ValueError(f"face_flip mismatch at ({k}, {f}).")
            if face_flip[nbr, nbr_f - 1] != should_flip:
                raise ValueError(f"face_flip symmetry mismatch at ({nbr}, {nbr_f}).")

            pair_key = tuple(sorted(((k, f), (nbr, nbr_f))))
            seen_pairs.add(pair_key)

    # boundary_faces consistency
    bf_from_mask = {(int(k), int(jf + 1)) for k in range(K) for jf in range(3) if is_boundary[k, jf]}
    bf_from_array = {tuple(map(int, row)) for row in boundary_faces}
    if bf_from_mask != bf_from_array:
        raise ValueError("boundary_faces does not match is_boundary mask.")

    n_unique_interior_faces = len(seen_pairs)

    # counting identity: 3K = 2*Nint + Nbd
    if n_total_local_faces != 2 * n_unique_interior_faces + n_boundary_faces:
        raise ValueError(
            "Face counting identity failed: 3K != 2*n_unique_interior_faces + n_boundary_faces."
        )

    return {
        "n_elements": K,
        "n_total_local_faces": n_total_local_faces,
        "n_boundary_faces": n_boundary_faces,
        "n_unique_interior_faces": n_unique_interior_faces,
    }