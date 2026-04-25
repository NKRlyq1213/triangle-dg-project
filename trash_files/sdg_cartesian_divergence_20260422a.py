from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from data.rule_registry import load_rule
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.sphere_flattened_connectivity import build_flattened_sphere_mesh
from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)
from geometry.reference_triangle import reference_triangle_area

from geometry.sdg_velocity_global_20260422a import (
    sdg_global_contravariant_velocity_on_mesh_20260422a,
)


@dataclass(frozen=True)
class SDGCartesianDivergenceSetup20260422a:
    """
    Setup for global flattened Cartesian divergence.

    All coordinates are global flattened coordinates on [-1,1]^2.
    """

    nsub: int
    order: int
    N: int
    radius: float

    mesh: object
    rule: dict

    X: np.ndarray       # shape (K,Np)
    Y: np.ndarray       # shape (K,Np)

    Dr: np.ndarray      # shape (Np,Np)
    Ds: np.ndarray      # shape (Np,Np)

    rx: np.ndarray      # shape (K,)
    sx: np.ndarray      # shape (K,)
    ry: np.ndarray      # shape (K,)
    sy: np.ndarray      # shape (K,)
    area: np.ndarray    # shape (K,)
    J: np.ndarray       # shape (K,)


def build_reference_diff_matrices_table1(
    *,
    order: int,
    N: int | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Build reference differentiation matrices using Table1 nodes.

    Returns:
        rule, Dr, Ds

    For row-major element arrays q_elem with shape (K,Np), use:
        q_r = q_elem @ Dr.T
        q_s = q_elem @ Ds.T
    """
    if N is None:
        N = order

    rule = load_rule("table1", order)

    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])

    if V.shape[0] == V.shape[1]:
        Dr, Ds = differentiation_matrices_square(V, Vr, Vs)
    else:
        Dr, Ds = differentiation_matrices_weighted(
            V,
            Vr,
            Vs,
            ws,
            area=reference_triangle_area(),
        )

    return rule, Dr, Ds


def affine_metrics_global_xy(
    nodes: np.ndarray,
    EToV: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute affine metrics for global flattened elements.

    Reference triangle convention:
        v1 = (-1,-1)
        v2 = ( 1,-1)
        v3 = (-1, 1)

    Affine map:
        x(r,s) = x1 + 0.5(r+1)(x2-x1) + 0.5(s+1)(x3-x1)
        y(r,s) = y1 + 0.5(r+1)(y2-y1) + 0.5(s+1)(y3-y1)

    Derivative conversion:
        f_x = rx f_r + sx f_s
        f_y = ry f_r + sy f_s
    """
    nodes = np.asarray(nodes, dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    K = EToV.shape[0]

    rx = np.zeros(K, dtype=float)
    sx = np.zeros(K, dtype=float)
    ry = np.zeros(K, dtype=float)
    sy = np.zeros(K, dtype=float)
    area = np.zeros(K, dtype=float)
    J = np.zeros(K, dtype=float)

    for k in range(K):
        v1, v2, v3 = nodes[EToV[k], :]

        xr = 0.5 * (v2[0] - v1[0])
        xs = 0.5 * (v3[0] - v1[0])
        yr = 0.5 * (v2[1] - v1[1])
        ys = 0.5 * (v3[1] - v1[1])

        Jk = xr * ys - xs * yr

        if Jk <= 0.0:
            raise ValueError(f"Element {k} has non-positive Jacobian J={Jk}.")

        rx[k] = ys / Jk
        sx[k] = -yr / Jk
        ry[k] = -xs / Jk
        sy[k] = xr / Jk

        J[k] = Jk

        # Reference triangle area is 2, so physical area = 2J.
        area[k] = 2.0 * Jk

    return rx, sx, ry, sy, area, J


def build_sdg_cartesian_divergence_setup_20260422a(
    *,
    nsub: int,
    order: int = 4,
    N: int | None = None,
    radius: float = 1.0,
) -> SDGCartesianDivergenceSetup20260422a:
    """
    Build all data needed for global flattened Cartesian divergence.
    """
    if N is None:
        N = order

    mesh = build_flattened_sphere_mesh(nsub)
    rule, Dr, Ds = build_reference_diff_matrices_table1(
        order=order,
        N=N,
    )

    X, Y = map_reference_nodes_to_all_elements(
        rule["rs"],
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        mesh.EToV,
    )

    rx, sx, ry, sy, area, J = affine_metrics_global_xy(
        mesh.nodes,
        mesh.EToV,
    )

    return SDGCartesianDivergenceSetup20260422a(
        nsub=nsub,
        order=order,
        N=N,
        radius=radius,
        mesh=mesh,
        rule=rule,
        X=X,
        Y=Y,
        Dr=Dr,
        Ds=Ds,
        rx=rx,
        sx=sx,
        ry=ry,
        sy=sy,
        area=area,
        J=J,
    )


def cartesian_divergence_from_flux(
    Fx: np.ndarray,
    Fy: np.ndarray,
    setup: SDGCartesianDivergenceSetup20260422a,
) -> np.ndarray:
    """
    Compute:

        div = d_x Fx + d_y Fy

    where Fx,Fy have shape (K,Np).
    """
    Fx = np.asarray(Fx, dtype=float)
    Fy = np.asarray(Fy, dtype=float)

    if Fx.shape != setup.X.shape:
        raise ValueError("Fx shape must match setup.X shape.")
    if Fy.shape != setup.X.shape:
        raise ValueError("Fy shape must match setup.X shape.")

    Fx_r = Fx @ setup.Dr.T
    Fx_s = Fx @ setup.Ds.T

    Fy_r = Fy @ setup.Dr.T
    Fy_s = Fy @ setup.Ds.T

    div_x = setup.rx[:, None] * Fx_r + setup.sx[:, None] * Fx_s
    div_y = setup.ry[:, None] * Fy_r + setup.sy[:, None] * Fy_s

    return div_x + div_y


def sdg_cartesian_divergence_of_constant_state_20260422a(
    setup: SDGCartesianDivergenceSetup20260422a,
    *,
    u0: float = 1.0,
    alpha0: float = 0.0,
    pole_tol: float = 1e-14,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute global flattened Cartesian divergence for q=1:

        div = d_x u1 + d_y u2

    Returns:
        div, diagnostics
    """
    q = np.ones_like(setup.X)

    u1, u2, u_sph, v_sph = sdg_global_contravariant_velocity_on_mesh_20260422a(
        setup.X,
        setup.Y,
        setup.mesh.face_ids,
        radius=setup.radius,
        u0=u0,
        alpha0=alpha0,
        pole_tol=pole_tol,
    )

    Fx = u1 * q
    Fy = u2 * q

    div = cartesian_divergence_from_flux(
        Fx,
        Fy,
        setup,
    )

    diagnostics = {
        "u1": u1,
        "u2": u2,
        "u_sph": u_sph,
        "v_sph": v_sph,
        "Fx": Fx,
        "Fy": Fy,
    }

    return div, diagnostics


def integrate_global_divergence_diagnostics_20260422a(
    div: np.ndarray,
    setup: SDGCartesianDivergenceSetup20260422a,
    *,
    surface_weighted: bool = True,
) -> dict[str, float]:
    """
    Compute global diagnostics for div.

    If surface_weighted=True:
        integrate with dS = sqrtG dxdy = pi R^2 dxdy.

    If surface_weighted=False:
        integrate over flattened dxdy only.

    Outputs:
        integral, mean, L2, RMS, Linf, area_measure
    """
    div = np.asarray(div, dtype=float)

    if div.shape != setup.X.shape:
        raise ValueError("div shape must match setup.X shape.")

    ws = np.asarray(setup.rule["ws"], dtype=float).reshape(-1)

    if surface_weighted:
        measure_factor = np.pi * setup.radius**2
    else:
        measure_factor = 1.0

    total_integral = 0.0
    total_l2_sq = 0.0
    total_measure = 0.0

    for k in range(div.shape[0]):
        w = setup.area[k] * ws * measure_factor

        total_integral += float(np.dot(w, div[k, :]))
        total_l2_sq += float(np.dot(w, div[k, :] ** 2))
        total_measure += float(np.sum(w))

    l2 = float(np.sqrt(max(total_l2_sq, 0.0)))
    rms = float(np.sqrt(max(total_l2_sq, 0.0) / total_measure))

    return {
        "integral": total_integral,
        "mean": total_integral / total_measure,
        "l2": l2,
        "rms": rms,
        "linf": float(np.max(np.abs(div))),
        "measure": total_measure,
    }