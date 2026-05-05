from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import numpy as np

from operators.sdg_surface_flux import SurfaceConnectivity


@dataclass(frozen=True)
class SphereSeamPairingInfo:
    n_original_boundary_faces: int
    n_seam_pairs: int
    n_unmatched_boundary_faces: int
    max_seam_match_error: float
    seam_pairs: tuple[tuple[int, int, int, int], ...]


def _face_xyz(cache, ref_face, k: int, f: int) -> np.ndarray:
    ids = np.asarray(ref_face.face_node_ids[f], dtype=int)
    return np.column_stack(
        [
            cache.X[k, ids],
            cache.Y[k, ids],
            cache.Z[k, ids],
        ]
    )


def _best_node_permutation(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Find permutation perm such that xyz_b[perm[i]] matches xyz_a[i].

    nfp is small for Table1 edge nodes, so brute force is robust and simple.
    """
    xyz_a = np.asarray(xyz_a, dtype=float)
    xyz_b = np.asarray(xyz_b, dtype=float)

    if xyz_a.shape != xyz_b.shape:
        raise ValueError("xyz_a and xyz_b must have the same shape.")

    nfp = xyz_a.shape[0]

    best_perm = None
    best_max = float("inf")
    best_sum = float("inf")

    for perm in permutations(range(nfp)):
        perm = np.asarray(perm, dtype=int)
        d = np.linalg.norm(xyz_a - xyz_b[perm, :], axis=1)
        dmax = float(np.max(d))
        dsum = float(np.sum(d))

        if (dmax < best_max) or (dmax == best_max and dsum < best_sum):
            best_max = dmax
            best_sum = dsum
            best_perm = perm.copy()

    if best_perm is None:
        raise RuntimeError("Failed to find node permutation.")

    return best_perm, best_max


def build_sphere_seam_connectivity(
    *,
    cache,
    ref_face,
    conn: SurfaceConnectivity,
    tol: float = 1.0e-10,
    allow_unmatched: bool = False,
) -> tuple[SurfaceConnectivity, SphereSeamPairingInfo]:
    r"""
    Convert flat-square boundary faces into sphere seam interior faces.

    Method
    ------
    Existing conn contains:
        - true flat interior faces
        - boundary faces on the square perimeter

    For each boundary face, compare its face-node sphere coordinates:

        (X, Y, Z)

    against every other boundary face. If the face-node sets match within
    tolerance, the two boundary faces are paired as a seam.

    The returned connectivity has seam faces marked as interior:
        is_boundary[k, f] = False

    and neighbor arrays filled.

    Important
    ---------
    This does not use lambda/theta or A/Ainv. It only uses physical sphere
    embedding coordinates, which avoids the pole-direction ambiguity.
    """
    neighbor_elem = np.array(conn.neighbor_elem, copy=True)
    neighbor_face = np.array(conn.neighbor_face, copy=True)
    neighbor_node_ids = np.array(conn.neighbor_node_ids, copy=True)
    is_boundary = np.array(conn.is_boundary, copy=True)

    K = is_boundary.shape[0]
    nfp = ref_face.nfp

    boundary_faces: list[tuple[int, int]] = []
    for k in range(K):
        for f in range(3):
            if bool(conn.is_boundary[k, f]):
                boundary_faces.append((k, f))

    unmatched = set(range(len(boundary_faces)))
    seam_pairs: list[tuple[int, int, int, int]] = []
    max_err = 0.0

    while unmatched:
        i = min(unmatched)
        unmatched.remove(i)

        ka, fa = boundary_faces[i]
        xyz_a = _face_xyz(cache, ref_face, ka, fa)

        best_j = None
        best_perm = None
        best_err = float("inf")

        for j in list(unmatched):
            kb, fb = boundary_faces[j]
            xyz_b = _face_xyz(cache, ref_face, kb, fb)

            perm, err = _best_node_permutation(xyz_a, xyz_b)

            if err < best_err:
                best_j = j
                best_perm = perm
                best_err = err

        if best_j is None or best_perm is None or best_err > tol:
            if allow_unmatched:
                continue
            raise ValueError(
                "Failed to find sphere seam partner for boundary face "
                f"(k={ka}, f={fa}). Best error={best_err:.6e}, tol={tol:.6e}. "
                "Try increasing --seam-tol or inspect the mapping."
            )

        unmatched.remove(best_j)

        kb, fb = boundary_faces[best_j]

        ids_a = np.asarray(ref_face.face_node_ids[fa], dtype=int)
        ids_b = np.asarray(ref_face.face_node_ids[fb], dtype=int)

        # A face order -> matching B volume node ids.
        neighbor_node_ids[ka, fa, :] = ids_b[best_perm]

        # B face order -> matching A volume node ids.
        inv_perm = np.empty_like(best_perm)
        inv_perm[best_perm] = np.arange(nfp)
        neighbor_node_ids[kb, fb, :] = ids_a[inv_perm]

        neighbor_elem[ka, fa] = kb
        neighbor_face[ka, fa] = fb
        is_boundary[ka, fa] = False

        neighbor_elem[kb, fb] = ka
        neighbor_face[kb, fb] = fa
        is_boundary[kb, fb] = False

        seam_pairs.append((ka, fa, kb, fb))
        max_err = max(max_err, float(best_err))

    new_conn = SurfaceConnectivity(
        neighbor_elem=neighbor_elem,
        neighbor_face=neighbor_face,
        neighbor_node_ids=neighbor_node_ids,
        is_boundary=is_boundary,
        face_length=np.array(conn.face_length, copy=True),
        face_normal_x=np.array(conn.face_normal_x, copy=True),
        face_normal_y=np.array(conn.face_normal_y, copy=True),
        area_flat=np.array(conn.area_flat, copy=True),
        max_match_error=max(float(conn.max_match_error), float(max_err)),
        n_internal_faces=int(conn.n_internal_faces + len(seam_pairs)),
        n_boundary_faces=int(np.count_nonzero(is_boundary)),
    )

    info = SphereSeamPairingInfo(
        n_original_boundary_faces=len(boundary_faces),
        n_seam_pairs=len(seam_pairs),
        n_unmatched_boundary_faces=int(np.count_nonzero(is_boundary)),
        max_seam_match_error=float(max_err),
        seam_pairs=tuple(seam_pairs),
    )

    return new_conn, info


def sphere_seam_connectivity_summary(
    conn: SurfaceConnectivity,
    info: SphereSeamPairingInfo,
) -> dict[str, float | int]:
    return {
        "n_original_boundary_faces": int(info.n_original_boundary_faces),
        "n_seam_pairs": int(info.n_seam_pairs),
        "n_unmatched_boundary_faces": int(info.n_unmatched_boundary_faces),
        "max_seam_match_error": float(info.max_seam_match_error),
        "n_internal_faces_after_seam": int(conn.n_internal_faces),
        "n_boundary_faces_after_seam": int(conn.n_boundary_faces),
    }
