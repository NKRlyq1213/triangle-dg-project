from __future__ import annotations

import numpy as np

from operators.sdg_edge_streamfunction_closed_sphere_rhs import (
    EdgeStreamfunctionClosedSphereSDGOperator,
    build_edge_streamfunction_closed_sphere_sdg_operator,
    edge_streamfunction_closed_sphere_rhs,
)


def sphere_area_from_weights(op: EdgeStreamfunctionClosedSphereSDGOperator) -> float:
    base = op.base
    good = (~base.cache.bad_mask) & np.isfinite(base.weights_surface)
    return float(np.sum(base.weights_surface[good]))


def global_mass_residual(rhs: np.ndarray, op: EdgeStreamfunctionClosedSphereSDGOperator) -> float:
    base = op.base
    rhs = np.asarray(rhs, dtype=float)
    good = (~base.cache.bad_mask) & np.isfinite(rhs) & np.isfinite(base.weights_surface)
    return float(np.sum(base.weights_surface[good] * rhs[good]))


def edge_streamfunction_closed_sphere_rhs_global_corrected(
    q: np.ndarray,
    op: EdgeStreamfunctionClosedSphereSDGOperator,
    *,
    mask_invalid: bool = True,
    return_info: bool = False,
):
    """
    Global mass-corrected edge-streamfunction RHS.

    Raw RHS:
        R_raw = edge-streamfunction volume + strong surface correction.

    Correction:
        R_corr = R_raw - mean_mass_residual

    where:
        mean_mass_residual = int R_raw dS / int 1 dS.

    This enforces global mass conservation while avoiding element-wise
    correction that can destroy constant-state preservation.
    """
    rhs_raw = edge_streamfunction_closed_sphere_rhs(
        q,
        op,
        mask_invalid=mask_invalid,
    )

    area = sphere_area_from_weights(op)
    I_raw = global_mass_residual(rhs_raw, op)

    c = -I_raw / area

    rhs_corr = rhs_raw + c

    if mask_invalid:
        rhs_corr = np.where(op.base.cache.bad_mask, np.nan, rhs_corr)

    if not return_info:
        return rhs_corr

    I_corr = global_mass_residual(rhs_corr, op)

    info = {
        "global_mass_residual_before": float(I_raw),
        "global_mass_residual_after": float(I_corr),
        "global_correction_constant": float(c),
        "sphere_area": float(area),
    }

    return rhs_corr, info


def build_global_corrected_operator(**kwargs) -> EdgeStreamfunctionClosedSphereSDGOperator:
    return build_edge_streamfunction_closed_sphere_sdg_operator(**kwargs)
