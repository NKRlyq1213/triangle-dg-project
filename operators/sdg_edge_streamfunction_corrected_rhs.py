from __future__ import annotations

import math
import numpy as np

from operators.sdg_edge_streamfunction_closed_sphere_rhs import (
    EdgeStreamfunctionClosedSphereSDGOperator,
    build_edge_streamfunction_closed_sphere_sdg_operator,
)
from operators.sdg_conservative_rhs import sdg_conservative_volume_rhs
from operators.sdg_edge_streamfunction_flux import (
    sdg_surface_edge_streamfunction_strong_rhs_projected,
)


def _element_mass_integrals(
    values: np.ndarray,
    weights_surface: np.ndarray,
    bad_mask: np.ndarray | None,
) -> np.ndarray:
    """
    Compute element-wise physical integrals:

        I_K = sum_i weights_surface[K,i] values[K,i].
    """
    values = np.asarray(values, dtype=float)
    weights_surface = np.asarray(weights_surface, dtype=float)

    if values.shape != weights_surface.shape:
        raise ValueError("values and weights_surface must have the same shape.")

    if bad_mask is None:
        good_values = np.where(np.isfinite(values), values, 0.0)
        good_weights = np.where(np.isfinite(values), weights_surface, 0.0)
    else:
        good = (~bad_mask) & np.isfinite(values)
        good_values = np.where(good, values, 0.0)
        good_weights = np.where(good, weights_surface, 0.0)

    return np.sum(good_weights * good_values, axis=1)


def _element_surface_areas(
    weights_surface: np.ndarray,
    bad_mask: np.ndarray | None,
) -> np.ndarray:
    """
    Element-wise surface area.
    """
    weights_surface = np.asarray(weights_surface, dtype=float)

    if bad_mask is None:
        good_weights = np.where(np.isfinite(weights_surface), weights_surface, 0.0)
    else:
        good = (~bad_mask) & np.isfinite(weights_surface)
        good_weights = np.where(good, weights_surface, 0.0)

    area = np.sum(good_weights, axis=1)

    if np.any(area <= 0.0):
        raise ValueError("Encountered non-positive element surface area.")

    return area


def _target_cell_mass_derivative_from_fhat(
    q: np.ndarray,
    p_face: np.ndarray,
    a_face: np.ndarray,
    op: EdgeStreamfunctionClosedSphereSDGOperator,
) -> np.ndarray:
    r"""
    Compute target cell mass derivative:

        dM_K/dt = - int_{partial K} fhat ds

    where strong penalty is:

        p = f - fhat

    hence:

        fhat = f - p = a q - p.

    Here:
        a = F dot n
    is edge-streamfunction normal flux.
    """
    base = op.base
    q = np.asarray(q, dtype=float)

    K = q.shape[0]
    target = np.zeros(K, dtype=float)

    we_all = np.asarray(base.rule["we"], dtype=float).reshape(-1)

    for k in range(K):
        for f in range(3):
            ids = np.asarray(base.ref_face.face_node_ids[f], dtype=int)
            we = we_all[ids]

            q_face = q[k, ids]
            fhat = a_face[k, f, :] * q_face - p_face[k, f, :]

            target[k] += -base.conn_seam.face_length[k, f] * float(np.sum(we * fhat))

    return target


def edge_streamfunction_closed_sphere_rhs_corrected(
    q: np.ndarray,
    op: EdgeStreamfunctionClosedSphereSDGOperator,
    *,
    mask_invalid: bool = True,
    return_info: bool = False,
):
    r"""
    Conservative-corrected edge-streamfunction RHS.

    Raw form:

        R_raw = R_vol + R_surf

    where:
        R_vol  = -1/J div(F_area q)
        R_surf =  1/J M^{-1} int_face (f - fhat) phi ds

    Correction:
        For each element K, add a constant c_K so that:

            int_K (R_raw + c_K) dS = - int_{partial K} fhat ds.

    This enforces local cell-average conservation while preserving the
    high-order shape of the raw RHS.
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

    rhs_surface, p_face, a_face = sdg_surface_edge_streamfunction_strong_rhs_projected(
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

    rhs_raw = rhs_volume + rhs_surface

    bad_mask = base.cache.bad_mask if mask_invalid else None

    current = _element_mass_integrals(
        rhs_raw,
        base.weights_surface,
        bad_mask,
    )

    target = _target_cell_mass_derivative_from_fhat(
        q,
        p_face,
        a_face,
        op,
    )

    area = _element_surface_areas(
        base.weights_surface,
        bad_mask,
    )

    c = (target - current) / area

    rhs_corr = rhs_raw + c[:, None]

    if mask_invalid:
        rhs_corr = np.where(base.cache.bad_mask, np.nan, rhs_corr)

    if not return_info:
        return rhs_corr

    info = {
        "max_abs_cell_correction": float(np.max(np.abs(c))),
        "global_current_mass_rhs": float(np.sum(current)),
        "global_target_mass_rhs": float(np.sum(target)),
        "global_correction_mass": float(np.sum(target - current)),
        "max_abs_local_mass_error_before": float(np.max(np.abs(current - target))),
        "max_abs_local_mass_error_after": float(
            np.max(
                np.abs(
                    _element_mass_integrals(rhs_corr, base.weights_surface, bad_mask)
                    - target
                )
            )
        ),
    }

    return rhs_corr, info


def build_corrected_operator(
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
    return build_edge_streamfunction_closed_sphere_sdg_operator(
        nsub=nsub,
        order=order,
        N=N,
        R=R,
        u0=u0,
        alpha0=alpha0,
        tau=tau,
        seam_tol=seam_tol,
    )
