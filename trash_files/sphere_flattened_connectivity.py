from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from geometry.sphere_flattened_mesh import (
    FlattenedSphereRawMesh,
    build_flattened_sphere_raw_mesh,
)


@dataclass(frozen=True)
class FlattenedSphereConnectivity:
    """
    Closed connectivity for the flattened 8-face sphere mesh.

    Face convention follows current triangle-dg-project convention:

        face 1: v2 -> v3
        face 2: v3 -> v1
        face 3: v1 -> v2

    EToF stores 1-based face ids.
    """

    EToE: np.ndarray          # shape (K, 3), 0-based element ids
    EToF: np.ndarray          # shape (K, 3), 1-based face ids
    face_flip: np.ndarray     # shape (K, 3), bool
    is_boundary: np.ndarray   # shape (K, 3), should be all False after gluing

    planar_boundary_faces: np.ndarray   # shape (Nb, 2), (k, face_id)
    glued_boundary_pairs: np.ndarray    # shape (Nb/2, 4), (ka, fa, kb, fb)


@dataclass(frozen=True)
class FlattenedSphereMesh:
    """
    New-mainline mesh object.

    It includes:
        - global flattened square mesh
        - face_ids for T1,...,T8
        - closed sphere connectivity after outer-boundary gluing
    """

    nsub: int
    nodes: np.ndarray
    EToV: np.ndarray
    face_ids: np.ndarray
    conn: FlattenedSphereConnectivity


# ---------------------------------------------------------------------
# Face convention
# ---------------------------------------------------------------------

def local_face_vertex_ids(elem_vids: np.ndarray) -> np.ndarray:
    """
    Return oriented face vertices using repo convention.

    face 1: v2 -> v3
    face 2: v3 -> v1
    face 3: v1 -> v2
    """
    elem_vids = np.asarray(elem_vids, dtype=int).reshape(-1)
    if elem_vids.shape[0] != 3:
        raise ValueError("elem_vids must have shape (3,).")

    v1, v2, v3 = elem_vids
    return np.asarray(
        [
            [v2, v3],
            [v3, v1],
            [v1, v2],
        ],
        dtype=int,
    )


def all_face_vertex_ids(EToV: np.ndarray) -> np.ndarray:
    """
    Shape:
        (K, 3, 2)
    """
    EToV = np.asarray(EToV, dtype=int)
    K = EToV.shape[0]

    out = np.zeros((K, 3, 2), dtype=int)
    for k in range(K):
        out[k] = local_face_vertex_ids(EToV[k])
    return out


def face_midpoints(nodes: np.ndarray, face_vids: np.ndarray) -> np.ndarray:
    """
    Compute face midpoint coordinates.

    Returns shape:
        (K, 3, 2)
    """
    nodes = np.asarray(nodes, dtype=float)
    face_vids = np.asarray(face_vids, dtype=int)

    K = face_vids.shape[0]
    mids = np.zeros((K, 3, 2), dtype=float)

    for k in range(K):
        for jf in range(3):
            va, vb = face_vids[k, jf]
            mids[k, jf] = 0.5 * (nodes[va] + nodes[vb])

    return mids


# ---------------------------------------------------------------------
# Planar connectivity
# ---------------------------------------------------------------------

def build_planar_connectivity(
    nodes: np.ndarray,
    EToV: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build connectivity from exact shared flattened-square edges.

    Returns
    -------
    EToE:
        Shape (K,3), -1 for planar boundary.
    EToF:
        Shape (K,3), 1-based face id, -1 for planar boundary.
    face_flip:
        Whether paired face node order is reversed.
    is_boundary:
        True for planar boundary before spherical gluing.
    face_vids:
        Shape (K,3,2)
    mids:
        Face midpoints.
    """
    nodes = np.asarray(nodes, dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    K = EToV.shape[0]

    face_vids = all_face_vertex_ids(EToV)
    mids = face_midpoints(nodes, face_vids)

    EToE = -np.ones((K, 3), dtype=int)
    EToF = -np.ones((K, 3), dtype=int)
    face_flip = np.zeros((K, 3), dtype=bool)
    is_boundary = np.ones((K, 3), dtype=bool)

    edge_map: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)

    for k in range(K):
        for jf in range(3):
            face_id = jf + 1
            va, vb = face_vids[k, jf]
            key = (min(int(va), int(vb)), max(int(va), int(vb)))
            edge_map[key].append((k, face_id, int(va), int(vb)))

    for key, entries in edge_map.items():
        if len(entries) == 1:
            continue

        if len(entries) != 2:
            raise ValueError(
                f"Invalid non-manifold edge {key}: appears {len(entries)} times."
            )

        (k1, f1, va1, vb1), (k2, f2, va2, vb2) = entries

        jf1 = f1 - 1
        jf2 = f2 - 1

        EToE[k1, jf1] = k2
        EToF[k1, jf1] = f2
        is_boundary[k1, jf1] = False

        EToE[k2, jf2] = k1
        EToF[k2, jf2] = f1
        is_boundary[k2, jf2] = False

        if va1 == vb2 and vb1 == va2:
            flip = True
        elif va1 == va2 and vb1 == vb2:
            flip = False
        else:
            raise RuntimeError("Canonical edge matched but oriented endpoints inconsistent.")

        face_flip[k1, jf1] = flip
        face_flip[k2, jf2] = flip

    return EToE, EToF, face_flip, is_boundary, face_vids, mids


# ---------------------------------------------------------------------
# Boundary classification and gluing
# ---------------------------------------------------------------------

def _boundary_face_array(is_boundary: np.ndarray) -> np.ndarray:
    faces = []
    K = is_boundary.shape[0]
    for k in range(K):
        for jf in range(3):
            if is_boundary[k, jf]:
                faces.append([k, jf + 1])
    return np.asarray(faces, dtype=int)


def _classify_outer_square_boundary_faces(
    nodes: np.ndarray,
    face_vids: np.ndarray,
    is_boundary: np.ndarray,
    mids: np.ndarray,
    *,
    tol: float = 1e-12,
) -> dict[str, list[tuple[int, int]]]:
    """
    Classify planar boundary faces by square side.

    Expected sides:
        right  : x =  1
        left   : x = -1
        top    : y =  1
        bottom : y = -1
    """
    groups: dict[str, list[tuple[int, int]]] = {
        "right": [],
        "left": [],
        "top": [],
        "bottom": [],
    }

    K = is_boundary.shape[0]
    for k in range(K):
        for jf in range(3):
            if not is_boundary[k, jf]:
                continue

            xmid, ymid = mids[k, jf]
            face_id = jf + 1

            if abs(xmid - 1.0) <= tol:
                groups["right"].append((k, face_id))
            elif abs(xmid + 1.0) <= tol:
                groups["left"].append((k, face_id))
            elif abs(ymid - 1.0) <= tol:
                groups["top"].append((k, face_id))
            elif abs(ymid + 1.0) <= tol:
                groups["bottom"].append((k, face_id))
            else:
                va, vb = face_vids[k, jf]
                raise ValueError(
                    "Boundary face is not on square outer boundary: "
                    f"(k={k}, face={face_id}), midpoint=({xmid},{ymid}), "
                    f"vertices={va},{vb}."
                )

    return groups


def _pair_same_side_half_edges(
    faces: list[tuple[int, int]],
    mids: np.ndarray,
    *,
    side: str,
    tol: float = 1e-12,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Pair boundary faces on the same square side.

    Sphere gluing convention copied from flattened octahedron topology:

    right / left sides:
        pair y > 0 with y < 0 using same |y|

    top / bottom sides:
        pair x > 0 with x < 0 using same |x|

    The node order should be reversed across the glued edge.
    """
    if side in {"right", "left"}:
        axis = 1  # y coordinate
    elif side in {"top", "bottom"}:
        axis = 0  # x coordinate
    else:
        raise ValueError("side must be one of right,left,top,bottom.")

    pos: list[tuple[int, int]] = []
    neg: list[tuple[int, int]] = []

    for k, face_id in faces:
        coord = float(mids[k, face_id - 1, axis])
        if coord > tol:
            pos.append((int(k), int(face_id)))
        elif coord < -tol:
            neg.append((int(k), int(face_id)))
        else:
            raise ValueError(
                f"Boundary face midpoint lies on zero split line for side={side}: "
                f"k={k}, face={face_id}, coord={coord}."
            )

    if len(pos) != len(neg):
        raise ValueError(
            f"Cannot pair side={side}: positive half has {len(pos)}, "
            f"negative half has {len(neg)}."
        )

    # Pair closest-to-zero positive with closest-to-zero negative, etc.
    pos_sorted = sorted(
        pos,
        key=lambda face: float(mids[face[0], face[1] - 1, axis]),
    )
    neg_sorted = sorted(
        neg,
        key=lambda face: -float(mids[face[0], face[1] - 1, axis]),
    )

    pairs = []
    for fa, fb in zip(pos_sorted, neg_sorted):
        ca = float(mids[fa[0], fa[1] - 1, axis])
        cb = float(mids[fb[0], fb[1] - 1, axis])

        if not np.isclose(ca, -cb, atol=tol, rtol=tol):
            raise ValueError(
                f"Gluing mismatch on side={side}: coord {ca} vs {cb}."
            )

        pairs.append((fa, fb))

    return pairs


def build_spherical_glued_connectivity(
    nodes: np.ndarray,
    EToV: np.ndarray,
    *,
    tol: float = 1e-12,
) -> FlattenedSphereConnectivity:
    """
    Build closed sphere connectivity from flattened square mesh.

    Steps:
        1. Build planar connectivity.
        2. Identify planar boundary faces on square boundary.
        3. Glue each square side half-to-half.
        4. Mark all faces as interior.
    """
    (
        EToE,
        EToF,
        face_flip,
        is_boundary,
        face_vids,
        mids,
    ) = build_planar_connectivity(nodes, EToV)

    planar_boundary_faces = _boundary_face_array(is_boundary)

    groups = _classify_outer_square_boundary_faces(
        nodes,
        face_vids,
        is_boundary,
        mids,
        tol=tol,
    )

    all_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for side in ("right", "left", "top", "bottom"):
        all_pairs.extend(
            _pair_same_side_half_edges(
                groups[side],
                mids,
                side=side,
                tol=tol,
            )
        )

    glued_rows: list[list[int]] = []

    for (ka, fa), (kb, fb) in all_pairs:
        ja = fa - 1
        jb = fb - 1

        if not is_boundary[ka, ja]:
            raise RuntimeError(f"Face ({ka},{fa}) is not boundary before gluing.")
        if not is_boundary[kb, jb]:
            raise RuntimeError(f"Face ({kb},{fb}) is not boundary before gluing.")

        EToE[ka, ja] = kb
        EToF[ka, ja] = fb
        EToE[kb, jb] = ka
        EToF[kb, jb] = fa

        # Flattened sphere gluing reverses edge-node order.
        face_flip[ka, ja] = True
        face_flip[kb, jb] = True

        is_boundary[ka, ja] = False
        is_boundary[kb, jb] = False

        glued_rows.append([ka, fa, kb, fb])

    glued_boundary_pairs = np.asarray(glued_rows, dtype=int)

    if np.any(is_boundary):
        remaining = _boundary_face_array(is_boundary)
        raise RuntimeError(
            f"Spherical gluing incomplete. Remaining boundary faces: {remaining.tolist()}"
        )

    return FlattenedSphereConnectivity(
        EToE=EToE,
        EToF=EToF,
        face_flip=face_flip,
        is_boundary=is_boundary,
        planar_boundary_faces=planar_boundary_faces,
        glued_boundary_pairs=glued_boundary_pairs,
    )


def build_flattened_sphere_mesh(nsub: int) -> FlattenedSphereMesh:
    """
    Build the new-mainline global flattened sphere mesh.

    Includes closed connectivity after spherical gluing.
    """
    raw = build_flattened_sphere_raw_mesh(nsub)
    conn = build_spherical_glued_connectivity(
        raw.nodes,
        raw.EToV,
    )

    return FlattenedSphereMesh(
        nsub=raw.nsub,
        nodes=raw.nodes,
        EToV=raw.EToV,
        face_ids=raw.face_ids,
        conn=conn,
    )


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def validate_closed_connectivity(mesh: FlattenedSphereMesh) -> dict[str, int]:
    """
    Validate basic closed connectivity invariants.

    Returns summary counters.
    """
    EToV = np.asarray(mesh.EToV, dtype=int)
    EToE = np.asarray(mesh.conn.EToE, dtype=int)
    EToF = np.asarray(mesh.conn.EToF, dtype=int)
    is_boundary = np.asarray(mesh.conn.is_boundary, dtype=bool)

    K = EToV.shape[0]

    if EToE.shape != (K, 3):
        raise ValueError("EToE must have shape (K,3).")
    if EToF.shape != (K, 3):
        raise ValueError("EToF must have shape (K,3).")
    if is_boundary.shape != (K, 3):
        raise ValueError("is_boundary must have shape (K,3).")

    if np.any(is_boundary):
        raise ValueError("Closed sphere connectivity should have no boundary faces.")

    if np.any(EToE < 0) or np.any(EToE >= K):
        raise ValueError("EToE contains invalid element ids.")

    if np.any(EToF < 1) or np.any(EToF > 3):
        raise ValueError("EToF must contain 1-based face ids in {1,2,3}.")

    for k in range(K):
        for jf in range(3):
            f = jf + 1
            nbr = int(EToE[k, jf])
            nbr_f = int(EToF[k, jf])
            nbr_jf = nbr_f - 1

            if int(EToE[nbr, nbr_jf]) != k:
                raise ValueError(
                    f"EToE symmetry failed: ({k},{f}) -> ({nbr},{nbr_f}), "
                    f"but reverse is {EToE[nbr, nbr_jf]}."
                )

            if int(EToF[nbr, nbr_jf]) != f:
                raise ValueError(
                    f"EToF symmetry failed: ({k},{f}) -> ({nbr},{nbr_f}), "
                    f"but reverse face is {EToF[nbr, nbr_jf]}."
                )

    return {
        "num_elements": int(K),
        "num_local_faces": int(3 * K),
        "num_planar_boundary_faces": int(mesh.conn.planar_boundary_faces.shape[0]),
        "num_glued_boundary_pairs": int(mesh.conn.glued_boundary_pairs.shape[0]),
    }