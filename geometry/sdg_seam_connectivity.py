from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from geometry.connectivity import build_face_connectivity
from geometry.sdg_sphere_mapping import sdg_mapping_from_xy_patch


@dataclass(frozen=True)
class SDGSeamPair:
    """
    One artificial boundary-face pair induced by the SDG square-to-sphere mapping.
    """
    elem_L: int
    face_L: int
    elem_R: int
    face_R: int
    side: str
    flip: bool
    key: tuple


def _edge_points_from_face_vids(
    VX: np.ndarray,
    VY: np.ndarray,
    face_vids: np.ndarray,
) -> np.ndarray:
    """
    Return oriented endpoint coordinates of one face.

    Output shape:
        (2,2)
    """
    va, vb = map(int, face_vids)
    return np.array(
        [
            [VX[va], VY[va]],
            [VX[vb], VY[vb]],
        ],
        dtype=float,
    )


def _classify_outer_square_side(
    pts: np.ndarray,
    tol: float = 1.0e-12,
) -> tuple[str, np.ndarray]:
    """
    Classify a boundary face on the outer square.

    Returns
    -------
    side : str
        One of 'top', 'bottom', 'left', 'right'.
    free_coords : np.ndarray
        The oriented free coordinate along the side:
            top/bottom: x coordinates
            left/right: y coordinates

    Notes
    -----
    The flat SDG square is fixed as [-1,1]^2.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (2, 2):
        raise ValueError("pts must have shape (2,2).")

    x = pts[:, 0]
    y = pts[:, 1]

    if np.all(np.abs(y - 1.0) <= tol):
        return "top", x.copy()
    if np.all(np.abs(y + 1.0) <= tol):
        return "bottom", x.copy()
    if np.all(np.abs(x - 1.0) <= tol):
        return "right", y.copy()
    if np.all(np.abs(x + 1.0) <= tol):
        return "left", y.copy()

    raise ValueError(f"Boundary face is not on the outer square: pts={pts}")


def _seam_key(
    side: str,
    free_coords: np.ndarray,
    ndigits: int = 12,
) -> tuple:
    """
    Canonical pairing key for SDG outer square seams.

    On each side, the two half-edges mirrored through the side midpoint
    map to the same sphere meridian.

    Example:
        top side segment x in [0.25, 0.50]
        pairs with top side segment x in [-0.50, -0.25].
    """
    vals = sorted([round(abs(float(v)), ndigits) for v in free_coords])
    return (side, vals[0], vals[1])


def _reflect_outer_square_point(side: str, p: np.ndarray) -> np.ndarray:
    """
    Reflection pairing on one outer side.

    top/bottom:
        (x,y) -> (-x,y)

    left/right:
        (x,y) -> (x,-y)
    """
    p = np.asarray(p, dtype=float)
    q = p.copy()

    if side in {"top", "bottom"}:
        q[0] = -q[0]
        return q

    if side in {"left", "right"}:
        q[1] = -q[1]
        return q

    raise ValueError("side must be top, bottom, left, or right.")


def _close_point(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return bool(np.linalg.norm(np.asarray(a) - np.asarray(b), ord=np.inf) <= tol)


def _compute_face_flip_by_reflection(
    side: str,
    pts_L: np.ndarray,
    pts_R: np.ndarray,
    tol: float,
) -> bool:
    """
    Determine whether neighbor face trace order must be reversed.

    flip=False means:
        reflected L endpoint 0 matches R endpoint 0,
        reflected L endpoint 1 matches R endpoint 1.

    flip=True means:
        reflected L endpoint 0 matches R endpoint 1,
        reflected L endpoint 1 matches R endpoint 0.
    """
    L0_ref = _reflect_outer_square_point(side, pts_L[0])
    L1_ref = _reflect_outer_square_point(side, pts_L[1])

    same = _close_point(L0_ref, pts_R[0], tol) and _close_point(L1_ref, pts_R[1], tol)
    cross = _close_point(L0_ref, pts_R[1], tol) and _close_point(L1_ref, pts_R[0], tol)

    if same:
        return False
    if cross:
        return True

    raise ValueError(
        "Could not determine seam face orientation.\n"
        f"side={side}\n"
        f"pts_L={pts_L}\n"
        f"pts_R={pts_R}\n"
        f"L0_ref={L0_ref}, L1_ref={L1_ref}"
    )


def build_sdg_sphere_face_connectivity(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_patch_id: np.ndarray,
    tol: float = 1.0e-12,
    ndigits: int = 12,
) -> dict:
    """
    Build SDG sphere connectivity.

    Procedure
    ---------
    1. Build ordinary flat triangular mesh connectivity using build_face_connectivity.
    2. Take the remaining outer-square boundary faces.
    3. Pair faces on each outer side by SDG seam reflection:
        top/bottom: x -> -x
        left/right: y -> -y
    4. Fill EToE/EToF/face_flip as interior-like pairs.

    Returns
    -------
    dict
        Connectivity dictionary with keys:
            EToE, EToF, is_boundary, face_flip,
            face_vertex_ids, face_midpoints,
            boundary_faces,
            flat_boundary_faces,
            sphere_seam_pairs,
            unpaired_boundary_faces
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)
    elem_patch_id = np.asarray(elem_patch_id, dtype=int)

    if VX.shape != VY.shape:
        raise ValueError("VX and VY must have same shape.")
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K,3).")
    if elem_patch_id.shape != (EToV.shape[0],):
        raise ValueError("elem_patch_id must have shape (K,).")

    base = build_face_connectivity(
        VX=VX,
        VY=VY,
        EToV=EToV,
        classify_boundary="box",
        tol=tol,
    )

    EToE = np.array(base["EToE"], dtype=int, copy=True)
    EToF = np.array(base["EToF"], dtype=int, copy=True)
    is_boundary = np.array(base["is_boundary"], dtype=bool, copy=True)
    face_flip = np.array(base["face_flip"], dtype=bool, copy=True)
    face_vids = np.array(base["face_vertex_ids"], dtype=int, copy=True)

    flat_boundary_faces = np.array(base["boundary_faces"], dtype=int, copy=True)
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for row in flat_boundary_faces:
        k, f = int(row[0]), int(row[1])
        pts = _edge_points_from_face_vids(VX, VY, face_vids[k, f - 1])
        side, free = _classify_outer_square_side(pts, tol=tol)
        key = _seam_key(side, free, ndigits=ndigits)

        groups[key].append(
            {
                "elem": k,
                "face": f,
                "side": side,
                "pts": pts,
                "key": key,
            }
        )

    seam_pairs: list[SDGSeamPair] = []
    unpaired: list[tuple[int, int]] = []

    for key, entries in groups.items():
        if len(entries) != 2:
            for e in entries:
                unpaired.append((int(e["elem"]), int(e["face"])))
            continue

        a, b = entries
        k1, f1 = int(a["elem"]), int(a["face"])
        k2, f2 = int(b["elem"]), int(b["face"])
        side = str(a["side"])

        if side != str(b["side"]):
            raise ValueError(f"Seam key grouped different sides: {a}, {b}")

        flip = _compute_face_flip_by_reflection(
            side=side,
            pts_L=np.asarray(a["pts"], dtype=float),
            pts_R=np.asarray(b["pts"], dtype=float),
            tol=max(100.0 * tol, 1.0e-12),
        )

        EToE[k1, f1 - 1] = k2
        EToF[k1, f1 - 1] = f2
        is_boundary[k1, f1 - 1] = False

        EToE[k2, f2 - 1] = k1
        EToF[k2, f2 - 1] = f1
        is_boundary[k2, f2 - 1] = False

        face_flip[k1, f1 - 1] = flip
        face_flip[k2, f2 - 1] = flip

        seam_pairs.append(
            SDGSeamPair(
                elem_L=k1,
                face_L=f1,
                elem_R=k2,
                face_R=f2,
                side=side,
                flip=bool(flip),
                key=key,
            )
        )

    remaining_boundary_faces = np.array(
        [(int(k), int(j + 1)) for k in range(EToV.shape[0]) for j in range(3) if is_boundary[k, j]],
        dtype=int,
    )
    if remaining_boundary_faces.size == 0:
        remaining_boundary_faces = remaining_boundary_faces.reshape(0, 2)

    return {
        "EToE": EToE,
        "EToF": EToF,
        "is_boundary": is_boundary,
        "face_flip": face_flip,
        "face_vertex_ids": face_vids,
        "face_midpoints": base["face_midpoints"],
        "boundary_faces": remaining_boundary_faces,
        "flat_boundary_faces": flat_boundary_faces,
        "sphere_seam_pairs": seam_pairs,
        "unpaired_boundary_faces": np.asarray(unpaired, dtype=int).reshape(-1, 2) if unpaired else np.empty((0, 2), dtype=int),
        "base_connectivity": base,
    }


def sample_face_points(
    VX: np.ndarray,
    VY: np.ndarray,
    conn: dict,
    elem: int,
    face: int,
    n_samples: int = 9,
    include_endpoints: bool = False,
) -> np.ndarray:
    """
    Sample points along one oriented face.

    Returns shape:
        (n_samples,2)
    """
    face_vids = np.asarray(conn["face_vertex_ids"], dtype=int)
    pts = _edge_points_from_face_vids(VX, VY, face_vids[int(elem), int(face) - 1])
    p0, p1 = pts

    if include_endpoints:
        t = np.linspace(0.0, 1.0, n_samples)
    else:
        t = np.linspace(0.0, 1.0, n_samples + 2)[1:-1]

    return (1.0 - t)[:, None] * p0 + t[:, None] * p1


def map_face_samples_to_sphere(
    VX: np.ndarray,
    VY: np.ndarray,
    conn: dict,
    elem_patch_id: np.ndarray,
    elem: int,
    face: int,
    n_samples: int = 9,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a face and map samples to sphere using the element's SDG patch id.

    Returns
    -------
    xy : np.ndarray
        Sample points in flat square, shape (n_samples,2).
    xyz : np.ndarray
        Sphere points, shape (n_samples,3).
    """
    xy = sample_face_points(
        VX=VX,
        VY=VY,
        conn=conn,
        elem=elem,
        face=face,
        n_samples=n_samples,
        include_endpoints=False,
    )

    pid = np.full((xy.shape[0],), int(elem_patch_id[int(elem)]), dtype=int)

    out = sdg_mapping_from_xy_patch(
        x=xy[:, 0],
        y=xy[:, 1],
        patch_id=pid,
        R=R,
        tol=tol,
    )

    xyz = np.column_stack([out.X, out.Y, out.Z])
    return xy, xyz


def seam_pair_xyz_errors(
    VX: np.ndarray,
    VY: np.ndarray,
    conn: dict,
    elem_patch_id: np.ndarray,
    n_samples: int = 9,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> np.ndarray:
    """
    Compute max sampled sphere-coordinate mismatch for every SDG seam pair.
    """
    pairs = list(conn["sphere_seam_pairs"])
    errs = np.zeros((len(pairs),), dtype=float)

    for i, pair in enumerate(pairs):
        _, xyz_L = map_face_samples_to_sphere(
            VX, VY, conn, elem_patch_id,
            elem=pair.elem_L,
            face=pair.face_L,
            n_samples=n_samples,
            R=R,
            tol=tol,
        )

        _, xyz_R = map_face_samples_to_sphere(
            VX, VY, conn, elem_patch_id,
            elem=pair.elem_R,
            face=pair.face_R,
            n_samples=n_samples,
            R=R,
            tol=tol,
        )

        if pair.flip:
            xyz_R = xyz_R[::-1]

        diff = np.linalg.norm(xyz_L - xyz_R, axis=1)
        errs[i] = float(np.nanmax(diff))

    return errs


def validate_sdg_sphere_connectivity(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_patch_id: np.ndarray,
    conn: dict,
    R: float = 1.0,
    tol: float = 1.0e-12,
    n_samples: int = 9,
) -> dict:
    """
    Validate SDG sphere connectivity.

    Unlike validate_face_connectivity for flat meshes, this allows artificial
    seam pairs whose vertex ids are different but whose mapped sphere edges
    coincide.
    """
    EToV = np.asarray(EToV, dtype=int)
    EToE = np.asarray(conn["EToE"], dtype=int)
    EToF = np.asarray(conn["EToF"], dtype=int)
    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)
    face_flip = np.asarray(conn["face_flip"], dtype=bool)

    K = EToV.shape[0]

    if EToE.shape != (K, 3):
        raise ValueError("EToE shape mismatch.")
    if EToF.shape != (K, 3):
        raise ValueError("EToF shape mismatch.")
    if is_boundary.shape != (K, 3):
        raise ValueError("is_boundary shape mismatch.")
    if face_flip.shape != (K, 3):
        raise ValueError("face_flip shape mismatch.")

    for k in range(K):
        for jf in range(3):
            f = jf + 1
            if is_boundary[k, jf]:
                if EToE[k, jf] != -1 or EToF[k, jf] != -1:
                    raise ValueError(f"Boundary face ({k},{f}) has non-boundary neighbor data.")
                continue

            nb = int(EToE[k, jf])
            nf = int(EToF[k, jf])

            if not (0 <= nb < K):
                raise ValueError(f"Invalid neighbor element at ({k},{f}): {nb}")
            if nf not in {1, 2, 3}:
                raise ValueError(f"Invalid neighbor face at ({k},{f}): {nf}")

            if int(EToE[nb, nf - 1]) != k:
                raise ValueError(f"EToE symmetry failure at ({k},{f})")
            if int(EToF[nb, nf - 1]) != f:
                raise ValueError(f"EToF symmetry failure at ({k},{f})")
            if bool(face_flip[nb, nf - 1]) != bool(face_flip[k, jf]):
                raise ValueError(f"face_flip symmetry failure at ({k},{f})")

    seam_errs = seam_pair_xyz_errors(
        VX=VX,
        VY=VY,
        conn=conn,
        elem_patch_id=elem_patch_id,
        n_samples=n_samples,
        R=R,
        tol=tol,
    )

    return {
        "n_elements": int(K),
        "n_total_local_faces": int(3 * K),
        "n_remaining_boundary_faces": int(np.sum(is_boundary)),
        "n_flat_boundary_faces_before_seam": int(np.asarray(conn["flat_boundary_faces"]).shape[0]),
        "n_sphere_seam_pairs": int(len(conn["sphere_seam_pairs"])),
        "n_unpaired_boundary_faces": int(np.asarray(conn["unpaired_boundary_faces"]).shape[0]),
        "max_seam_xyz_error": float(np.nanmax(seam_errs)) if seam_errs.size else 0.0,
        "mean_seam_xyz_error": float(np.nanmean(seam_errs)) if seam_errs.size else 0.0,
    }
