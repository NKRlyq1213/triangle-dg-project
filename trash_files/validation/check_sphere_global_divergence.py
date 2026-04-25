from __future__ import annotations

import argparse
import math
import numpy as np

from problems.sphere_single_patch_rhs import (
    compute_constant_state_single_patch_rhs,
)
from geometry.sphere_metrics_sdg_hardcoded import sqrtG_sdg_local as sqrtG


def integrate_patch_divergence(
    *,
    patch_id: int,
    nsub: int,
    radius: float,
    quad_order: int,
    N: int,
    alpha0: float,
) -> dict[str, float]:
    """
    Compute divergence diagnostics on one patch, but return only aggregate scalars.

    For q = 1:
        volume_rhs = - div_local(u1, u2)

    Since sqrtG is constant for the current equal-area mapping,
    surface divergence is represented by the local divergence.

    Integral over sphere patch:
        ∫_{S_i} div_S V dS
        = ∫_{T_i} div_local(u1,u2) sqrtG dx dy
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

    # For q=1, volume_rhs = -div.
    div_local = -np.asarray(diag["volume_rhs"], dtype=float)

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

    # ∫ div dS
    integral = 0.0

    # ∫ div^2 dS
    l2_sq = 0.0

    # ∫ 1 dS
    area = 0.0

    for k in range(div_local.shape[0]):
        w_phys = elem_area[k] * ws * J_sphere[k, :]

        integral += float(np.dot(w_phys, div_local[k, :]))
        l2_sq += float(np.dot(w_phys, div_local[k, :] ** 2))
        area += float(np.sum(w_phys))

    linf = float(np.max(np.abs(div_local)))

    return {
        "integral": integral,
        "l2_sq": l2_sq,
        "area": area,
        "linf": linf,
    }


def compute_global_divergence(
    *,
    nsub: int,
    radius: float,
    quad_order: int,
    N: int,
    alpha0: float,
) -> dict[str, float]:
    """
    Compute global divergence diagnostics over all 8 patches.

    Output does not report per-patch values.
    """
    global_integral = 0.0
    global_l2_sq = 0.0
    global_area = 0.0
    global_linf = 0.0

    for patch_id in range(1, 9):
        out = integrate_patch_divergence(
            patch_id=patch_id,
            nsub=nsub,
            radius=radius,
            quad_order=quad_order,
            N=N,
            alpha0=alpha0,
        )

        global_integral += out["integral"]
        global_l2_sq += out["l2_sq"]
        global_area += out["area"]
        global_linf = max(global_linf, out["linf"])

    exact_area = 4.0 * math.pi * radius**2

    return {
        "nsub": float(nsub),
        "alpha0": float(alpha0),
        "area_discrete": global_area,
        "area_exact": exact_area,
        "area_error": abs(global_area - exact_area),
        "integral": global_integral,
        "mean": global_integral / global_area,
        "l2": math.sqrt(max(global_l2_sq, 0.0)),
        "linf": global_linf,
    }


def run_refinement_study(
    *,
    nsubs: list[int],
    radius: float,
    quad_order: int,
    N: int,
    alpha0: float,
) -> None:
    print("=== Global divergence diagnostic on the full 8-patch sphere ===")
    print("q = 1")
    print("quad_table = table1")
    print(f"quad_order = {quad_order}")
    print(f"N          = {N}")
    print(f"alpha0     = {alpha0:.16e}")
    print()

    header = (
        f"{'nsub':>6s} | "
        f"{'area_err':>12s} | "
        f"{'int(div)':>14s} | "
        f"{'mean(div)':>14s} | "
        f"{'L2(div)':>14s} | "
        f"{'Linf(div)':>14s}"
    )
    print(header)
    print("-" * len(header))

    previous_l2 = None

    for nsub in nsubs:
        out = compute_global_divergence(
            nsub=nsub,
            radius=radius,
            quad_order=quad_order,
            N=N,
            alpha0=alpha0,
        )

        print(
            f"{nsub:6d} | "
            f"{out['area_error']:12.3e} | "
            f"{out['integral']:14.6e} | "
            f"{out['mean']:14.6e} | "
            f"{out['l2']:14.6e} | "
            f"{out['linf']:14.6e}",
            end="",
        )

        if previous_l2 is not None and out["l2"] > 0.0:
            rate = math.log(previous_l2 / out["l2"]) / math.log(2.0)
            print(f"   rate(L2)={rate:7.3f}")
        else:
            print()

        previous_l2 = out["l2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsubs",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="subdivision levels per patch",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--quad-order",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--N",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--alpha0",
        type=float,
        default=math.pi / 4.0,
        help="velocity parameter alpha0",
    )

    args = parser.parse_args()

    run_refinement_study(
        nsubs=args.nsubs,
        radius=args.radius,
        quad_order=args.quad_order,
        N=args.N,
        alpha0=args.alpha0,
    )


if __name__ == "__main__":
    main()