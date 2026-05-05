from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from data.table1_rules import load_table1_rule
from geometry.sphere_flat_mesh import sphere_flat_square_mesh
from geometry.sphere_flat_metrics import build_sphere_flat_geometry_cache
from geometry.metrics import affine_geometric_factors_from_mesh
from operators.sdg_flattened_divergence import build_table1_reference_diff_operators
from operators.sdg_streamfunction_flux import (
    equal_area_jacobian,
    streamfunction_area_flux,
)
from operators.sdg_conservative_rhs import sdg_conservative_volume_rhs
from operators.sdg_surface_flux import (
    build_reference_face_data,
    build_surface_connectivity,
)
from operators.sdg_sphere_seam_connectivity import build_sphere_seam_connectivity
from operators.sdg_projected_surface_flux import build_projected_surface_inverse_mass_T
from operators.sdg_direct_flux_surface import (
    sdg_surface_direct_flux_rhs_projected,
    internal_direct_flux_pair_balance,
)


@dataclass
class ClosedSphereSDGOperator:
    rule: dict
    rs_nodes: np.ndarray

    VX: np.ndarray
    VY: np.ndarray
    EToV: np.ndarray
    elem_patch_id: np.ndarray

    cache: object
    geom: dict[str, np.ndarray]

    Dr: np.ndarray
    Ds: np.ndarray

    ref_face: object
    conn_flat: object
    conn_seam: object
    seam_info: object

    MinvT: np.ndarray

    Fx_area: np.ndarray
    Fy_area: np.ndarray
    psi: np.ndarray

    weights_surface: np.ndarray
    J_area: float

    N: int
    order: int
    R: float
    u0: float
    alpha0: float
    tau: float
    seam_tol: float


def build_surface_weights_from_geom(
    rule: dict,
    geom: dict[str, np.ndarray],
    *,
    R: float,
) -> np.ndarray:
    """
    Surface quadrature weights for equal-area SDG map.

        dS = J_area dx dy
        J_area = pi R^2

    On affine flat triangles:

        int_K f dxdy ≈ area_K * sum_i w_i f_i
    """
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    J_flat = np.asarray(geom["J"], dtype=float)

    if J_flat.ndim != 2:
        raise ValueError("geom['J'] must have shape (K, Np).")

    area_flat = 2.0 * J_flat[:, 0]
    J_area = equal_area_jacobian(R)

    return area_flat[:, None] * ws[None, :] * J_area


def build_closed_sphere_sdg_operator(
    *,
    nsub: int,
    order: int = 4,
    N: int | None = None,
    R: float = 1.0,
    u0: float = 1.0,
    alpha0: float = math.pi / 4.0,
    tau: float = 0.0,
    seam_tol: float = 1.0e-10,
) -> ClosedSphereSDGOperator:
    """
    Build a closed-sphere equal-area SDG operator using:

    1. Stream-function area flux:
           psi = -R Omega dot X
           Fx_area = -psi_y
           Fy_area =  psi_x

    2. Projected surface inverse-mass lifting.

    3. Pairwise direct numerical flux on all faces, including sphere seams.

    The resulting RHS is intended for:

        q_t = R(q)

    where the physical conserved mass is:

        M(q) = int_{S^2} q dS
             = sum_K sum_i weights_surface[K,i] q[K,i].
    """
    if N is None:
        N = order

    rule = load_table1_rule(order)
    rs_nodes = np.asarray(rule["rs"], dtype=float)

    VX, VY, EToV, elem_patch_id = sphere_flat_square_mesh(
        n_sub=nsub,
        R=1.0,
    )

    cache = build_sphere_flat_geometry_cache(
        rs_nodes=rs_nodes,
        VX=VX,
        VY=VY,
        EToV=EToV,
        elem_patch_id=elem_patch_id,
        R=R,
    )

    Dr, Ds = build_table1_reference_diff_operators(rule, N=N)

    geom = affine_geometric_factors_from_mesh(
        VX=VX,
        VY=VY,
        EToV=EToV,
        rs_nodes=rs_nodes,
    )

    weights_surface = build_surface_weights_from_geom(
        rule,
        geom,
        R=R,
    )

    J_area = equal_area_jacobian(R)

    Fx_area, Fy_area, psi = streamfunction_area_flux(
        cache.X,
        cache.Y,
        cache.Z,
        Dr,
        Ds,
        geom["rx"],
        geom["sx"],
        geom["ry"],
        geom["sy"],
        alpha0=alpha0,
        u0=u0,
        R=R,
    )

    ref_face = build_reference_face_data(rule)

    conn_flat = build_surface_connectivity(
        VX=VX,
        VY=VY,
        EToV=EToV,
        rs_nodes=rs_nodes,
        ref_face=ref_face,
    )

    conn_seam, seam_info = build_sphere_seam_connectivity(
        cache=cache,
        ref_face=ref_face,
        conn=conn_flat,
        tol=seam_tol,
        allow_unmatched=False,
    )

    MinvT = build_projected_surface_inverse_mass_T(
        N=N,
        rule=rule,
    )

    return ClosedSphereSDGOperator(
        rule=rule,
        rs_nodes=rs_nodes,

        VX=VX,
        VY=VY,
        EToV=EToV,
        elem_patch_id=elem_patch_id,

        cache=cache,
        geom=geom,

        Dr=Dr,
        Ds=Ds,

        ref_face=ref_face,
        conn_flat=conn_flat,
        conn_seam=conn_seam,
        seam_info=seam_info,

        MinvT=MinvT,

        Fx_area=Fx_area,
        Fy_area=Fy_area,
        psi=psi,

        weights_surface=weights_surface,
        J_area=J_area,

        N=N,
        order=order,
        R=R,
        u0=u0,
        alpha0=alpha0,
        tau=tau,
        seam_tol=seam_tol,
    )


def closed_sphere_streamfunction_rhs(
    q: np.ndarray,
    op: ClosedSphereSDGOperator,
    *,
    mask_invalid: bool = True,
) -> np.ndarray:
    """
    Full closed-sphere RHS:

        q_t = R_vol(q) + R_surf(q)

    where:

        R_vol = -1/J * div(F_area q)

        R_surf = -1/J * M^{-1} int_face fhat phi ds

    Surface flux uses pairwise direct numerical flux on flat interior faces
    and sphere seam faces.
    """
    q = np.asarray(q, dtype=float)

    if q.shape != op.Fx_area.shape:
        raise ValueError(
            f"q shape {q.shape} does not match operator node shape {op.Fx_area.shape}."
        )

    rhs_volume = sdg_conservative_volume_rhs(
        q,
        op.Fx_area,
        op.Fy_area,
        op.Dr,
        op.Ds,
        op.geom["rx"],
        op.geom["sx"],
        op.geom["ry"],
        op.geom["sy"],
        J_area=op.J_area,
        mask=op.cache.bad_mask if mask_invalid else None,
    )

    rhs_surface = sdg_surface_direct_flux_rhs_projected(
        q,
        op.Fx_area,
        op.Fy_area,
        op.rule,
        op.ref_face,
        op.conn_seam,
        N=op.N,
        J_area=op.J_area,
        tau=op.tau,
        boundary_mode="same_state",
        surface_inverse_mass_T=op.MinvT,
    )

    rhs = rhs_volume + rhs_surface

    if mask_invalid:
        rhs = np.where(op.cache.bad_mask, np.nan, rhs)

    return rhs


def mass(q: np.ndarray, op: ClosedSphereSDGOperator) -> float:
    """
    Physical mass:

        M(q) = int_{S^2} q dS.
    """
    q = np.asarray(q, dtype=float)

    good = (~op.cache.bad_mask) & np.isfinite(q) & np.isfinite(op.weights_surface)

    return float(np.sum(op.weights_surface[good] * q[good]))


def rhs_mass_residual(rhs: np.ndarray, op: ClosedSphereSDGOperator) -> float:
    """
    Mass derivative induced by RHS:

        dM/dt = int rhs dS.
    """
    rhs = np.asarray(rhs, dtype=float)

    good = (~op.cache.bad_mask) & np.isfinite(rhs) & np.isfinite(op.weights_surface)

    return float(np.sum(op.weights_surface[good] * rhs[good]))


def weighted_l2_linf(values: np.ndarray, op: ClosedSphereSDGOperator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    good = (~op.cache.bad_mask) & np.isfinite(values) & np.isfinite(op.weights_surface)

    if not np.any(good):
        return float("nan"), float("nan")

    vals = values[good]
    w = op.weights_surface[good]

    l2 = math.sqrt(max(float(np.sum(w * vals * vals)), 0.0))
    linf = float(np.max(np.abs(vals)))

    return l2, linf


def q_initial(
    case: str,
    op: ClosedSphereSDGOperator,
) -> np.ndarray:
    """
    Standard diagnostic initial states.
    """
    case = case.lower().strip()
    cache = op.cache
    R = op.R

    if case == "constant":
        return np.ones_like(cache.X)

    if case == "sphere_x":
        return cache.X / R

    if case == "sphere_y":
        return cache.Y / R

    if case == "sphere_z":
        return cache.Z / R

    if case == "flat_gaussian":
        x = cache.x_flat
        y = cache.y_flat
        x0 = 0.25
        y0 = -0.15
        sigma = 0.22
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))

    if case == "gaussian":
        # Smooth spherical Gaussian centered away from seams.
        xc = np.array([1.0, 0.0, 0.0], dtype=float)
        X = np.stack([cache.X, cache.Y, cache.Z], axis=-1)
        dot = (
            X[..., 0] * xc[0]
            + X[..., 1] * xc[1]
            + X[..., 2] * xc[2]
        ) / (R * R)
        dot = np.clip(dot, -1.0, 1.0)
        dist = R * np.arccos(dot)
        sigma = 0.35
        return np.exp(-(dist * dist) / (2.0 * sigma * sigma))

    if case == "element_jump":
        eps = 0.1
        K = cache.X.shape[0]
        sign = np.where((np.arange(K) % 2) == 0, 1.0, -1.0)
        return cache.Z / R + eps * sign[:, None]

    if case == "element_checker":
        K = cache.X.shape[0]
        sign = np.where((np.arange(K) % 2) == 0, 1.0, -1.0)
        return sign[:, None] * np.ones_like(cache.X)

    raise ValueError(
        "Unknown q case. Available: constant, sphere_x, sphere_y, sphere_z, "
        "flat_gaussian, gaussian, element_jump, element_checker."
    )


def seam_summary(op: ClosedSphereSDGOperator) -> dict[str, float | int]:
    return {
        "n_original_boundary_faces": int(op.seam_info.n_original_boundary_faces),
        "n_seam_pairs": int(op.seam_info.n_seam_pairs),
        "n_unmatched_boundary_faces": int(op.seam_info.n_unmatched_boundary_faces),
        "max_seam_match_error": float(op.seam_info.max_seam_match_error),
        "n_boundary_faces_after_seam": int(op.conn_seam.n_boundary_faces),
    }


def flux_pair_summary(
    q: np.ndarray,
    op: ClosedSphereSDGOperator,
) -> dict[str, float | int]:
    return internal_direct_flux_pair_balance(
        q,
        op.Fx_area,
        op.Fy_area,
        op.ref_face,
        op.conn_seam,
        tau=op.tau,
    )
