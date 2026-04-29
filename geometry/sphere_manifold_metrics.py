from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _apply_reference_operator(D: np.ndarray, u: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    u = np.asarray(u, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix.")
    if u.ndim != 2 or u.shape[1] != D.shape[0]:
        raise ValueError("u must have shape (K, Np), with Np matching D.")

    return u @ D.T


@dataclass(frozen=True)
class ManifoldGeometryCache:
    nodes_xyz: np.ndarray
    EToV: np.ndarray
    rs_nodes: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    J: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    nz: np.ndarray
    a1x: np.ndarray
    a1y: np.ndarray
    a1z: np.ndarray
    a2x: np.ndarray
    a2y: np.ndarray
    a2z: np.ndarray


def map_reference_nodes_to_sphere(
    nodes_xyz: np.ndarray,
    EToV: np.ndarray,
    rs_nodes: np.ndarray,
    R: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate element vertices linearly, then project each node to the sphere.
    """
    nodes_xyz = np.asarray(nodes_xyz, dtype=float)
    EToV = np.asarray(EToV, dtype=int)
    rs_nodes = np.asarray(rs_nodes, dtype=float)

    if nodes_xyz.ndim != 2 or nodes_xyz.shape[1] != 3:
        raise ValueError("nodes_xyz must have shape (Nv, 3).")
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")
    if rs_nodes.ndim != 2 or rs_nodes.shape[1] != 2:
        raise ValueError("rs_nodes must have shape (Np, 2).")
    if R <= 0.0:
        raise ValueError("R must be positive.")

    r = rs_nodes[:, 0]
    s = rs_nodes[:, 1]
    L1 = -(r + s) / 2.0
    L2 = (1.0 + r) / 2.0
    L3 = (1.0 + s) / 2.0

    tri = nodes_xyz[EToV]
    xyz = (
        L1[None, :, None] * tri[:, None, 0, :]
        + L2[None, :, None] * tri[:, None, 1, :]
        + L3[None, :, None] * tri[:, None, 2, :]
    )

    norm = np.linalg.norm(xyz, axis=2)
    if np.any(norm <= 0.0):
        raise ValueError("Reference node interpolation hit the zero vector.")

    xyz = R * xyz / norm[:, :, None]
    return xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]


def build_manifold_geometry_cache(
    nodes_xyz: np.ndarray,
    EToV: np.ndarray,
    rs_nodes: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    R: float = 1.0,
    tol: float = 1.0e-14,
) -> ManifoldGeometryCache:
    """
    Build nodal 3D manifold metrics on a projected spherical triangle mesh.
    """
    X, Y, Z = map_reference_nodes_to_sphere(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=rs_nodes,
        R=R,
    )

    xr = _apply_reference_operator(Dr, X)
    yr = _apply_reference_operator(Dr, Y)
    zr = _apply_reference_operator(Dr, Z)

    xs = _apply_reference_operator(Ds, X)
    ys = _apply_reference_operator(Ds, Y)
    zs = _apply_reference_operator(Ds, Z)

    Nx = yr * zs - zr * ys
    Ny = zr * xs - xr * zs
    Nz = xr * ys - yr * xs

    J = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
    if np.any(J <= tol):
        raise ValueError("Degenerate manifold element detected: surface J too small.")

    nx = Nx / J
    ny = Ny / J
    nz = Nz / J

    radial_dot = nx * X + ny * Y + nz * Z
    flip = radial_dot < 0.0
    nx = np.where(flip, -nx, nx)
    ny = np.where(flip, -ny, ny)
    nz = np.where(flip, -nz, nz)

    # Contravariant surface basis vectors:
    # a^1 = (a_s x n) / J, a^2 = (n x a_r) / J.
    a1x = (ys * nz - zs * ny) / J
    a1y = (zs * nx - xs * nz) / J
    a1z = (xs * ny - ys * nx) / J

    a2x = (ny * zr - nz * yr) / J
    a2y = (nz * xr - nx * zr) / J
    a2z = (nx * yr - ny * xr) / J

    return ManifoldGeometryCache(
        nodes_xyz=np.asarray(nodes_xyz, dtype=float),
        EToV=np.asarray(EToV, dtype=int),
        rs_nodes=np.asarray(rs_nodes, dtype=float),
        X=X,
        Y=Y,
        Z=Z,
        J=J,
        nx=nx,
        ny=ny,
        nz=nz,
        a1x=a1x,
        a1y=a1y,
        a1z=a1z,
        a2x=a2x,
        a2y=a2y,
        a2z=a2z,
    )
