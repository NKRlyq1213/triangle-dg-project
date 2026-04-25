from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.sdg_sphere_mapping import (
    sdg_detA_expected,
    sdg_mapping_from_xy_patch,
    sdg_sqrtG_expected,
)


@dataclass(frozen=True)
class SphereFlatGeometryCache:
    """
    Nodal geometry cache using SDG explicit T1--T8 formulas.
    """
    x_flat: np.ndarray
    y_flat: np.ndarray
    elem_patch_id: np.ndarray
    node_patch_id: np.ndarray

    lambda_: np.ndarray
    theta: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray

    A: np.ndarray
    Ainv: np.ndarray
    detA: np.ndarray
    sqrtG: np.ndarray

    pole_mask: np.ndarray
    bad_mask: np.ndarray
    A_Ainv_err: np.ndarray
    Ainv_numpy_err: np.ndarray


def build_sphere_flat_geometry_cache(
    rs_nodes: np.ndarray,
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_patch_id: np.ndarray,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> SphereFlatGeometryCache:
    """
    Build nodal SDG mapping cache.

    Critical convention:
    --------------------
    - Flat domain remains [-1,1]^2.
    - R is only the sphere radius and the SDG metric scaling.
    - A and Ainv are SDG explicit formulas.
    - Pole limit replacement is not applied.
    """
    rs_nodes = np.asarray(rs_nodes, dtype=float)
    EToV = np.asarray(EToV, dtype=int)
    elem_patch_id = np.asarray(elem_patch_id, dtype=int)

    if rs_nodes.ndim != 2 or rs_nodes.shape[1] != 2:
        raise ValueError("rs_nodes must have shape (Np,2).")
    if elem_patch_id.shape != (EToV.shape[0],):
        raise ValueError("elem_patch_id must have shape (K,).")

    X_flat, Y_flat = map_reference_nodes_to_all_elements(
        rs_nodes=rs_nodes,
        VX=VX,
        VY=VY,
        EToV=EToV,
    )

    K, Np = X_flat.shape
    node_patch_id = np.repeat(elem_patch_id[:, None], Np, axis=1)

    out = sdg_mapping_from_xy_patch(
        x=X_flat,
        y=Y_flat,
        patch_id=node_patch_id,
        R=R,
        tol=tol,
    )

    return SphereFlatGeometryCache(
        x_flat=X_flat,
        y_flat=Y_flat,
        elem_patch_id=elem_patch_id,
        node_patch_id=node_patch_id,
        lambda_=out.lambda_,
        theta=out.theta,
        X=out.X,
        Y=out.Y,
        Z=out.Z,
        A=out.A,
        Ainv=out.Ainv,
        detA=out.detA,
        sqrtG=out.sqrtG,
        pole_mask=out.pole_mask,
        bad_mask=out.bad_mask,
        A_Ainv_err=out.A_Ainv_err,
        Ainv_numpy_err=out.Ainv_numpy_err,
    )


def geometry_diagnostics(cache: SphereFlatGeometryCache, R: float = 1.0) -> dict:
    """
    Scalar terminal diagnostics.
    """
    radial = np.sqrt(cache.X**2 + cache.Y**2 + cache.Z**2)
    radial_err = np.abs(radial - R)

    regular = ~cache.bad_mask

    det_expected = sdg_detA_expected(R)
    sqrtG_expected = sdg_sqrtG_expected(R)

    det_err = np.abs(cache.detA - det_expected)
    sqrtG_err = np.abs(cache.sqrtG - sqrtG_expected)

    def _nanmax_regular(a):
        b = np.asarray(a, dtype=float)[regular]
        b = b[np.isfinite(b)]
        return float(np.max(b)) if b.size else float("nan")

    def _nanmin_regular(a):
        b = np.asarray(a, dtype=float)[regular]
        b = b[np.isfinite(b)]
        return float(np.min(b)) if b.size else float("nan")

    return {
        "max_radial_error": float(np.nanmax(radial_err)),
        "mean_radial_error": float(np.nanmean(radial_err)),
        "detA_expected": float(det_expected),
        "detA_min_regular": _nanmin_regular(cache.detA),
        "detA_max_regular": _nanmax_regular(cache.detA),
        "detA_error_max_regular": _nanmax_regular(det_err),
        "sqrtG_expected": float(sqrtG_expected),
        "sqrtG_min_regular": _nanmin_regular(cache.sqrtG),
        "sqrtG_max_regular": _nanmax_regular(cache.sqrtG),
        "sqrtG_error_max_regular": _nanmax_regular(sqrtG_err),
        "A_Ainv_err_max_regular": _nanmax_regular(cache.A_Ainv_err),
        "Ainv_numpy_err_max_regular": _nanmax_regular(cache.Ainv_numpy_err),
        "n_total_nodes": int(cache.x_flat.size),
        "n_pole_nodes": int(np.sum(cache.pole_mask)),
        "n_bad_nodes": int(np.sum(cache.bad_mask)),
    }


def per_patch_diagnostics(cache: SphereFlatGeometryCache, R: float = 1.0) -> dict[int, dict[str, float]]:
    """
    Per-patch diagnostics for SDG mapping.
    """
    det_expected = sdg_detA_expected(R)
    sqrtG_expected = sdg_sqrtG_expected(R)

    out: dict[int, dict[str, float]] = {}
    for pid in range(1, 9):
        mask = (cache.node_patch_id == pid) & (~cache.bad_mask)

        def mx(a):
            vals = np.asarray(a, dtype=float)[mask]
            vals = vals[np.isfinite(vals)]
            return float(np.max(vals)) if vals.size else float("nan")

        out[pid] = {
            "n_regular": int(np.sum(mask)),
            "detA_error_max": mx(np.abs(cache.detA - det_expected)),
            "sqrtG_error_max": mx(np.abs(cache.sqrtG - sqrtG_expected)),
            "A_Ainv_err_max": mx(cache.A_Ainv_err),
            "Ainv_numpy_err_max": mx(cache.Ainv_numpy_err),
        }

    return out
