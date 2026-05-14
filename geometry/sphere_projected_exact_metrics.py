from __future__ import annotations

import numpy as np

from geometry.sphere_manifold_metrics import (
    ManifoldGeometryCache,
    map_reference_nodes_to_sphere,
)


def build_exact_projected_sphere_geometry_cache(
    nodes_xyz: np.ndarray,
    EToV: np.ndarray,
    rs_nodes: np.ndarray,
    R: float = 1.0,
    tol: float = 1.0e-14,
) -> ManifoldGeometryCache:
    """
    Build analytical metrics for the radial projection of planar triangles to a sphere.

    This is the NumPy version of the exact metric construction used in the SBP
    manifold DG notebooks.  The returned arrays follow the repository convention:

        shape = (K, Np)

    where K is the number of elements and Np is the number of Table-1 volume nodes.
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

    X, Y, Z = map_reference_nodes_to_sphere(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=rs_nodes,
        R=R,
    )

    r = rs_nodes[:, 0]
    s = rs_nodes[:, 1]
    L1 = -(r + s) / 2.0
    L2 = (1.0 + r) / 2.0
    L3 = (1.0 + s) / 2.0

    K = EToV.shape[0]
    Np = rs_nodes.shape[0]

    J = np.zeros((K, Np), dtype=float)
    a1x = np.zeros((K, Np), dtype=float)
    a1y = np.zeros((K, Np), dtype=float)
    a1z = np.zeros((K, Np), dtype=float)
    a2x = np.zeros((K, Np), dtype=float)
    a2y = np.zeros((K, Np), dtype=float)
    a2z = np.zeros((K, Np), dtype=float)

    for k in range(K):
        v1, v2, v3 = nodes_xyz[EToV[k]]

        x_flat = L1 * v1[0] + L2 * v2[0] + L3 * v3[0]
        y_flat = L1 * v1[1] + L2 * v2[1] + L3 * v3[1]
        z_flat = L1 * v1[2] + L2 * v2[2] + L3 * v3[2]

        norm_x = np.sqrt(x_flat * x_flat + y_flat * y_flat + z_flat * z_flat)
        if np.any(norm_x <= tol):
            raise ValueError("Reference node interpolation hit the zero vector.")

        dr_x = -0.5 * v1[0] + 0.5 * v2[0]
        dr_y = -0.5 * v1[1] + 0.5 * v2[1]
        dr_z = -0.5 * v1[2] + 0.5 * v2[2]

        ds_x = -0.5 * v1[0] + 0.5 * v3[0]
        ds_y = -0.5 * v1[1] + 0.5 * v3[1]
        ds_z = -0.5 * v1[2] + 0.5 * v3[2]

        cross_x = dr_y * ds_z - dr_z * ds_y
        cross_y = dr_z * ds_x - dr_x * ds_z
        cross_z = dr_x * ds_y - dr_y * ds_x

        h_signed = v1[0] * cross_x + v1[1] * cross_y + v1[2] * cross_z
        h = float(h_signed)
        if abs(h) <= tol:
            raise ValueError("Degenerate projected triangle detected: |h| too small.")

        J[k, :] = (R * R * abs(h)) / (norm_x ** 3)
        factor = norm_x / (R * h)

        # a^1 = (|x| / (R |h|)) * (d_s x x_flat)
        a1x[k, :] = factor * (ds_y * z_flat - ds_z * y_flat)
        a1y[k, :] = factor * (ds_z * x_flat - ds_x * z_flat)
        a1z[k, :] = factor * (ds_x * y_flat - ds_y * x_flat)

        # a^2 = (|x| / (R |h|)) * (x_flat x d_r)
        a2x[k, :] = factor * (y_flat * dr_z - z_flat * dr_y)
        a2y[k, :] = factor * (z_flat * dr_x - x_flat * dr_z)
        a2z[k, :] = factor * (x_flat * dr_y - y_flat * dr_x)

    nx = X / R
    ny = Y / R
    nz = Z / R

    return ManifoldGeometryCache(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=rs_nodes,
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

