from __future__ import annotations

import numpy as np


def _apply_reference_operator(D: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Apply a reference differentiation matrix D to nodal data u.

    Supported shapes
    ----------------
    - u.shape == (Np,)
    - u.shape == (K, Np)

    Returns
    -------
    np.ndarray
        Same shape as u
    """
    D = np.asarray(D, dtype=float)
    u = np.asarray(u, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D array.")

    Np = D.shape[0]

    if u.ndim == 1:
        if u.shape[0] != Np:
            raise ValueError("For 1D input, u.shape[0] must equal D.shape[0].")
        return D @ u

    if u.ndim == 2:
        if u.shape[1] != Np:
            raise ValueError("For 2D input, u.shape[1] must equal D.shape[0].")
        # row-wise application:
        # (D @ u_k)^T = u_k^T @ D^T
        return u @ D.T

    raise ValueError("u must have shape (Np,) or (K, Np).")


def geometric_factors_2d(
    x: np.ndarray,
    y: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    tol: float = 1e-14,
) -> dict[str, np.ndarray]:
    """
    Compute geometric factors on triangles.

    Given nodal coordinates x(r,s), y(r,s), compute:
        xr, xs, yr, ys, J, rx, sx, ry, sy

    Formulas
    --------
        xr = d x / d r
        xs = d x / d s
        yr = d y / d r
        ys = d y / d s

        J  = xr*ys - xs*yr

        rx =  ys / J
        sx = -yr / J
        ry = -xs / J
        sy =  xr / J

    Parameters
    ----------
    x, y : np.ndarray
        Physical nodal coordinates.
        Supported shapes:
        - (Np,)
        - (K, Np)
    Dr, Ds : np.ndarray
        Reference differentiation matrices of shape (Np, Np)
    tol : float
        Tolerance for degenerate Jacobian detection.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        "xr", "xs", "yr", "ys", "J", "rx", "sx", "ry", "sy"
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Dr = np.asarray(Dr, dtype=float)
    Ds = np.asarray(Ds, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    xr = _apply_reference_operator(Dr, x)
    xs = _apply_reference_operator(Ds, x)
    yr = _apply_reference_operator(Dr, y)
    ys = _apply_reference_operator(Ds, y)

    J = xr * ys - xs * yr

    if np.any(np.abs(J) <= tol):
        raise ValueError("Degenerate element detected: |J| is too small.")

    rx = ys / J
    sx = -yr / J
    ry = -xs / J
    sy = xr / J

    return {
        "xr": xr,
        "xs": xs,
        "yr": yr,
        "ys": ys,
        "J": J,
        "rx": rx,
        "sx": sx,
        "ry": ry,
        "sy": sy,
    }


def physical_derivatives_2d(
    u: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    rx: np.ndarray,
    sx: np.ndarray,
    ry: np.ndarray,
    sy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute physical derivatives (ux, uy) from reference derivatives.

    Chain rule
    ----------
        ux = rx * ur + sx * us
        uy = ry * ur + sy * us

    Supported shapes
    ----------------
    u, rx, sx, ry, sy can all be shape:
    - (Np,)
    - (K, Np)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ux, uy), same shape as u
    """
    u = np.asarray(u, dtype=float)
    rx = np.asarray(rx, dtype=float)
    sx = np.asarray(sx, dtype=float)
    ry = np.asarray(ry, dtype=float)
    sy = np.asarray(sy, dtype=float)

    if not (u.shape == rx.shape == sx.shape == ry.shape == sy.shape):
        raise ValueError("u, rx, sx, ry, sy must have the same shape.")

    ur = _apply_reference_operator(Dr, u)
    us = _apply_reference_operator(Ds, u)

    ux = rx * ur + sx * us
    uy = ry * ur + sy * us
    return ux, uy


def divergence_2d(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    rx: np.ndarray,
    sx: np.ndarray,
    ry: np.ndarray,
    sy: np.ndarray,
) -> np.ndarray:
    """
    Compute div(F) = dFx/dx + dFy/dy.

    Supported shapes
    ----------------
    Fx, Fy, rx, sx, ry, sy can all be shape:
    - (Np,)
    - (K, Np)

    Returns
    -------
    np.ndarray
        Discrete divergence, same shape as Fx
    """
    Fx = np.asarray(Fx, dtype=float)
    Fy = np.asarray(Fy, dtype=float)

    if Fx.shape != Fy.shape:
        raise ValueError("Fx and Fy must have the same shape.")

    dFxdx, _ = physical_derivatives_2d(Fx, Dr, Ds, rx, sx, ry, sy)
    _, dFydy = physical_derivatives_2d(Fy, Dr, Ds, rx, sx, ry, sy)
    return dFxdx + dFydy


def affine_metric_terms_from_vertices(
    vertices: np.ndarray,
    tol: float = 1e-14,
) -> dict[str, float]:
    """
    Exact geometric factors for one straight-sided affine triangle.

    Reference triangle:
        v1 = (-1, -1)
        v2 = ( 1, -1)
        v3 = (-1,  1)

    Physical triangle vertices:
        vertices[0] = (x1, y1)
        vertices[1] = (x2, y2)
        vertices[2] = (x3, y3)

    Affine map:
        x(r,s) = -(r+s)/2 * x1 + (1+r)/2 * x2 + (1+s)/2 * x3
        y(r,s) = -(r+s)/2 * y1 + (1+r)/2 * y2 + (1+s)/2 * y3

    Therefore:
        xr = (x2 - x1)/2
        xs = (x3 - x1)/2
        yr = (y2 - y1)/2
        ys = (y3 - y1)/2
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    xr = 0.5 * (x2 - x1)
    xs = 0.5 * (x3 - x1)
    yr = 0.5 * (y2 - y1)
    ys = 0.5 * (y3 - y1)

    J = xr * ys - xs * yr
    if abs(J) <= tol:
        raise ValueError("Degenerate affine triangle: |J| is too small.")

    rx = ys / J
    sx = -yr / J
    ry = -xs / J
    sy = xr / J

    return {
        "xr": float(xr),
        "xs": float(xs),
        "yr": float(yr),
        "ys": float(ys),
        "J": float(J),
        "rx": float(rx),
        "sx": float(sx),
        "ry": float(ry),
        "sy": float(sy),
    }


def affine_geometric_factors_from_mesh(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    rs_nodes: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Exact affine geometric factors on all elements, broadcast to all table nodes.

    Parameters
    ----------
    VX, VY : np.ndarray
        Global vertex coordinates, shape (Nv,)
    EToV : np.ndarray
        Element-to-vertex connectivity, shape (K, 3)
    rs_nodes : np.ndarray
        Reference nodes (table points), shape (Np, 2)

    Returns
    -------
    dict[str, np.ndarray]
        Keys: xr, xs, yr, ys, J, rx, sx, ry, sy
        Each array has shape (K, Np)
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)
    rs_nodes = np.asarray(rs_nodes, dtype=float)

    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")
    if rs_nodes.ndim != 2 or rs_nodes.shape[1] != 2:
        raise ValueError("rs_nodes must have shape (Np, 2).")

    K = EToV.shape[0]
    Np = rs_nodes.shape[0]

    out = {
        "xr": np.zeros((K, Np), dtype=float),
        "xs": np.zeros((K, Np), dtype=float),
        "yr": np.zeros((K, Np), dtype=float),
        "ys": np.zeros((K, Np), dtype=float),
        "J":  np.zeros((K, Np), dtype=float),
        "rx": np.zeros((K, Np), dtype=float),
        "sx": np.zeros((K, Np), dtype=float),
        "ry": np.zeros((K, Np), dtype=float),
        "sy": np.zeros((K, Np), dtype=float),
    }

    for k in range(K):
        vids = EToV[k]
        vertices = np.column_stack([VX[vids], VY[vids]])
        gk = affine_metric_terms_from_vertices(vertices)

        for key in out:
            out[key][k, :] = gk[key]

    return out