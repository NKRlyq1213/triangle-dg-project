from __future__ import annotations

import argparse
import math
import numpy as np

from problems.sphere_single_patch_rhs import compute_constant_state_single_patch_rhs
from geometry.sphere_metrics import sqrtG


def integrate_rhs_on_patch(
    *,
    patch_id: int,
    nsub: int,
    radius: float,
    quad_order: int,
    N: int,
    alpha0: float,
) -> dict[str, float]:
    """
    Compute patch contribution to global RHS diagnostics.

    For q = 1, this uses the already-built single-patch RHS machinery.
    The integral is weighted by the sphere surface measure:

        dS = sqrtG dx dy

    On each affine sub-element:
        dx dy ≈ area_k * sum_q ws_q
    """
    rhs, diag, setup = compute_constant_state_single_patch_rhs(
        patch_id=patch_id,
        nsub=nsub,
        radius=radius,
        quad_table="table1",
        quad_order=quad_order,
        N=N,
        alpha0=alpha0,
        tau=0.0,
    )

    rhs = np.asarray(rhs, dtype=float)
    volume_rhs = np.asarray(diag["volume_rhs"], dtype=float)
    surface_rhs = np.asarray(diag["surface_rhs"], dtype=float)

    ws = np.asarray(setup.rule["ws"], dtype=float).reshape(-1)
    elem_area = np.asarray(setup.face_geom["area"], dtype=float).reshape(-1)

    x = np.asarray(setup.X, dtype=float)
    y = np.asarray(setup.Y, dtype=float)

    J_sphere = sqrtG(
        x,
        y,
        patch_id,
        radius=radius,
    )

    integral_rhs = 0.0
    integral_volume = 0.0
    integral_surface = 0.0
    l2_rhs_sq = 0.0
    area = 0.0

    for k in range(rhs.shape[0]):
        w_phys = elem_area[k] * ws * J_sphere[k, :]

        integral_rhs += float(np.dot(w_phys, rhs[k, :]))
        integral_volume += float(np.dot(w_phys, volume_rhs[k, :]))
        integral_surface += float(np.dot(w_phys, surface_rhs[k, :]))
        l2_rhs_sq += float(np.dot(w_phys, rhs[k, :] ** 2))
        area += float(np.sum(w_phys))

    return {
        "integral_rhs": integral_rhs,
        "integral_volume": integral_volume,
        "integral_surface": integral_surface,
        "l2_rhs_sq": l2_rhs_sq,
        "linf_rhs": float(np.max(np.abs(rhs))),
        "linf_volume": float(np.max(np.abs(volume_rhs))),
        "linf_surface": float(np.max(np.abs(surface_rhs))),
        "area": area,
    }


def compute_global_constant_rhs(
    *,
    nsub: int,
    radius: float,
    quad_order: int,
    N: int,
    alpha0: float,
) -> dict[str, float]:
    """
    Aggregate RHS diagnostics over all 8 sphere patches.
    """
    total_integral_rhs = 0.0
    total_integral_volume = 0.0
    total_integral_surface = 0.0
    total_l2_rhs_sq = 0.0
    total_area = 0.0
    global_linf_rhs = 0.0
    global_linf_volume = 0.0
    global_linf_surface = 0.0

    for patch_id in range(1, 9):
        out = integrate_rhs_on_patch(
            patch_id=patch_id,
            nsub=nsub,
            radius=radius,
            quad_order=quad_order,
            N=N,
            alpha0=alpha0,
        )

        total_integral_rhs += out["integral_rhs"]
        total_integral_volume += out["integral_volume"]
        total_integral_surface += out["integral_surface"]
        total_l2_rhs_sq += out["l2_rhs_sq"]
        total_area += out["area"]

        global_linf_rhs = max(global_linf_rhs, out["linf_rhs"])
        global_linf_volume = max(global_linf_volume, out["linf_volume"])
        global_linf_surface = max(global_linf_surface, out["linf_surface"])

    exact_area = 4.0 * math.pi * radius**2

    return {
        "nsub": float(nsub),
        "alpha0": float(alpha0),
        "area_discrete": total_area,
        "area_exact": exact_area,
        "area_error": abs(total_area - exact_area),
        "integral_rhs": total_integral_rhs,
        "integral_volume": total_integral_volume,
        "integral_surface": total_integral_surface,
        "mean_rhs": total_integral_rhs / total_area,
        "l2_rhs": math.sqrt(max(total_l2_rhs_sq, 0.0)),
        "linf_rhs": global_linf_rhs,
        "linf_volume": global_linf_volume,
        "linf_surface": global_linf_surface,
    }


def run_study(
    *,
    nsubs: list[int],
    radius: float,
    quad_order: int,
    N: int,
    alpha0: float,
) -> None:
    print("=== Global 8-patch constant-state RHS diagnostic ===")
    print("q = 1")
    print("quad_table = table1")
    print(f"quad_order = {quad_order}")
    print(f"N          = {N}")
    print(f"alpha0     = {alpha0:.16e}")
    print()

    header = (
        f"{'nsub':>6s} | "
        f"{'area_err':>12s} | "
        f"{'int(rhs)':>14s} | "
        f"{'int(vol)':>14s} | "
        f"{'int(surf)':>14s} | "
        f"{'mean(rhs)':>14s} | "
        f"{'L2(rhs)':>14s} | "
        f"{'Linf(rhs)':>14s}"
    )
    print(header)
    print("-" * len(header))

    prev_l2 = None

    for nsub in nsubs:
        out = compute_global_constant_rhs(
            nsub=nsub,
            radius=radius,
            quad_order=quad_order,
            N=N,
            alpha0=alpha0,
        )

        print(
            f"{nsub:6d} | "
            f"{out['area_error']:12.3e} | "
            f"{out['integral_rhs']:14.6e} | "
            f"{out['integral_volume']:14.6e} | "
            f"{out['integral_surface']:14.6e} | "
            f"{out['mean_rhs']:14.6e} | "
            f"{out['l2_rhs']:14.6e} | "
            f"{out['linf_rhs']:14.6e}",
            end="",
        )

        if prev_l2 is not None and out["l2_rhs"] > 0.0:
            rate = math.log(prev_l2 / out["l2_rhs"]) / math.log(2.0)
            print(f"   rate(L2)={rate:7.3f}")
        else:
            print()

        prev_l2 = out["l2_rhs"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsubs", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--quad-order", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--alpha0", type=float, default=math.pi / 4.0)

    args = parser.parse_args()

    run_study(
        nsubs=args.nsubs,
        radius=args.radius,
        quad_order=args.quad_order,
        N=args.N,
        alpha0=args.alpha0,
    )


if __name__ == "__main__":
    main()