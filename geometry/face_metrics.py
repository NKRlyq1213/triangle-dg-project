from __future__ import annotations

import numpy as np

from geometry.affine_map import map_ref_to_phys
from geometry.mesh_structured import triangle_signed_area


def _local_face_endpoints(vertices: np.ndarray, face_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the physical endpoints of one local face.

    Local face convention
    ---------------------
    face 1: v2 -> v3
    face 2: v3 -> v1
    face 3: v1 -> v2
    """
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    if face_id == 1:
        return vertices[1], vertices[2]
    if face_id == 2:
        return vertices[2], vertices[0]
    if face_id == 3:
        return vertices[0], vertices[1]

    raise ValueError("face_id must be 1, 2, or 3.")


def affine_face_geometry_from_vertices(vertices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute affine face geometry for one CCW triangle.

    Returns
    -------
    dict
        normals : (3, 2)
            outward unit normals for faces 1,2,3
        lengths : (3,)
            physical edge lengths for faces 1,2,3
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    signed_area = triangle_signed_area(vertices[0], vertices[1], vertices[2])
    if signed_area <= 0.0:
        raise ValueError("Triangle must be counterclockwise and nondegenerate.")

    normals = np.zeros((3, 2), dtype=float)
    lengths = np.zeros(3, dtype=float)

    for face_id in (1, 2, 3):
        p0, p1 = _local_face_endpoints(vertices, face_id)
        tangent = p1 - p0
        L = np.linalg.norm(tangent)
        if L <= 0.0:
            raise ValueError(f"Degenerate face detected on face {face_id}.")

        # For CCW boundary orientation, outward normal is (dy, -dx)/|e|
        nx = tangent[1] / L
        ny = -tangent[0] / L

        normals[face_id - 1, 0] = nx
        normals[face_id - 1, 1] = ny
        lengths[face_id - 1] = L

    return {
        "normals": normals,
        "lengths": lengths,
    }


def affine_face_geometry_from_mesh(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    trace: dict,
) -> dict[str, np.ndarray]:
    """
    Build face geometry for all affine triangles on the mesh.

    Parameters
    ----------
    VX, VY : np.ndarray
        Global vertex coordinates, shape (Nv,)
    EToV : np.ndarray
        Element-to-vertex connectivity, shape (K, 3)
    trace : dict
        Trace descriptor from build_trace_policy(rule) for Table 1.

    Returns
    -------
    dict
        area   : (K,)
        length : (K, 3)
        nx     : (K, 3, Nfp)
        ny     : (K, 3, Nfp)
        x_face : (K, 3, Nfp)
        y_face : (K, 3, Nfp)
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)

    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")

    if trace.get("trace_mode", None) != "embedded":
        raise ValueError("This phase-1 implementation currently expects Table 1 embedded trace.")

    K = EToV.shape[0]
    nfp = int(trace["nfp"])

    area = np.zeros(K, dtype=float)
    length = np.zeros((K, 3), dtype=float)
    nx = np.zeros((K, 3, nfp), dtype=float)
    ny = np.zeros((K, 3, nfp), dtype=float)
    x_face = np.zeros((K, 3, nfp), dtype=float)
    y_face = np.zeros((K, 3, nfp), dtype=float)

    for k in range(K):
        vids = EToV[k]
        vertices = np.column_stack([VX[vids], VY[vids]])

        signed_area = triangle_signed_area(vertices[0], vertices[1], vertices[2])
        if signed_area <= 0.0:
            raise ValueError(f"Element {k} is not CCW or is degenerate.")

        area[k] = abs(signed_area)

        gk = affine_face_geometry_from_vertices(vertices)
        normals_k = gk["normals"]
        lengths_k = gk["lengths"]

        for face_id in (1, 2, 3):
            rs_face = np.asarray(trace["face_rs"][face_id], dtype=float)
            xf, yf = map_ref_to_phys(rs_face[:, 0], rs_face[:, 1], vertices)

            x_face[k, face_id - 1, :] = xf
            y_face[k, face_id - 1, :] = yf

            length[k, face_id - 1] = lengths_k[face_id - 1]
            nx[k, face_id - 1, :] = normals_k[face_id - 1, 0]
            ny[k, face_id - 1, :] = normals_k[face_id - 1, 1]

    return {
        "area": area,
        "length": length,
        "nx": nx,
        "ny": ny,
        "x_face": x_face,
        "y_face": y_face,
    }