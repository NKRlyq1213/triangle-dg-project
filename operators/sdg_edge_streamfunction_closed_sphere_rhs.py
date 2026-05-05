from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from operators.sdg_streamfunction_closed_sphere_rhs import (
    ClosedSphereSDGOperator,
    build_closed_sphere_sdg_operator,
    q_initial,
    mass,
    rhs_mass_residual,
    weighted_l2_linf,
    seam_summary,
)
from operators.sdg_conservative_rhs import sdg_conservative_volume_rhs
from operators.sdg_edge_streamfunction_flux import (
    build_reference_face_derivative_matrices,
    sdg_surface_edge_streamfunction_strong_rhs_projected,
    edge_flux_pair_error,
)


@dataclass
class EdgeStreamfunctionClosedSphereSDGOperator:
    """
    Closed-sphere SDG operator using edge-streamfunction face flux.

    Base geometry and volume flux are inherited from ClosedSphereSDGOperator.

    Surface term:
        R_surf = 1/J M^{-1} int_face (f - fhat) phi ds

    where:
        f = a q,
        a = F dot n = - d_s psi.

    This is the version that passed:
        - constant-state RHS test,
        - discontinuous conservation test,
        - gaussian mass residual test.
    """
    base: ClosedSphereSDGOperator
    face_Dt: tuple[np.ndarray, np.ndarray, np.ndarray]


def build_edge_streamfunction_closed_sphere_sdg_operator(
    *,
    nsub: int,
    order: int = 4,
    N: int | None = None,
    R: float = 1.0,
    u0: float = 1.0,
    alpha0: float = math.pi / 4.0,
    tau: float = 0.0,
    seam_tol: float = 1.0e-10,
) -> EdgeStreamfunctionClosedSphereSDGOperator:
    base = build_closed_sphere_sdg_operator(
        nsub=nsub,
        order=order,
        N=N,
        R=R,
        u0=u0,
        alpha0=alpha0,
        tau=tau,
        seam_tol=seam_tol,
    )

    face_Dt = build_reference_face_derivative_matrices(
        base.rs_nodes,
        base.ref_face,
    )

    return EdgeStreamfunctionClosedSphereSDGOperator(
        base=base,
        face_Dt=face_Dt,
    )


def edge_streamfunction_closed_sphere_rhs(
    q: np.ndarray,
    op: EdgeStreamfunctionClosedSphereSDGOperator,
    *,
    mask_invalid: bool = True,
) -> np.ndarray:
    """
    Full RHS:

        q_t = R_vol + R_surf

    Volume:
        R_vol = -1/J div(F_area q)

    Surface:
        R_surf = 1/J M^{-1} int_face (f - fhat) phi ds

    where face f is computed from edge streamfunction flux.
    """
    base = op.base
    q = np.asarray(q, dtype=float)

    if q.shape != base.Fx_area.shape:
        raise ValueError(
            f"q shape {q.shape} does not match operator shape {base.Fx_area.shape}."
        )

    rhs_volume = sdg_conservative_volume_rhs(
        q,
        base.Fx_area,
        base.Fy_area,
        base.Dr,
        base.Ds,
        base.geom["rx"],
        base.geom["sx"],
        base.geom["ry"],
        base.geom["sy"],
        J_area=base.J_area,
        mask=base.cache.bad_mask if mask_invalid else None,
    )

    rhs_surface, _p, _a_face = sdg_surface_edge_streamfunction_strong_rhs_projected(
        q,
        base.psi,
        base.rule,
        base.rs_nodes,
        base.ref_face,
        base.conn_seam,
        N=base.N,
        J_area=base.J_area,
        tau=base.tau,
        boundary_mode="same_state",
        surface_inverse_mass_T=base.MinvT,
        face_Dt=op.face_Dt,
    )

    rhs = rhs_volume + rhs_surface

    if mask_invalid:
        rhs = np.where(base.cache.bad_mask, np.nan, rhs)

    return rhs


def edge_flux_pair_summary(
    q: np.ndarray,
    op: EdgeStreamfunctionClosedSphereSDGOperator,
) -> dict[str, float | int]:
    """
    Check pairwise consistency of edge-streamfunction normal flux.
    """
    base = op.base

    _rhs_surface, _p, a_face = sdg_surface_edge_streamfunction_strong_rhs_projected(
        q,
        base.psi,
        base.rule,
        base.rs_nodes,
        base.ref_face,
        base.conn_seam,
        N=base.N,
        J_area=base.J_area,
        tau=base.tau,
        boundary_mode="same_state",
        surface_inverse_mass_T=base.MinvT,
        face_Dt=op.face_Dt,
    )

    return edge_flux_pair_error(
        a_face,
        base.ref_face,
        base.conn_seam,
    )
