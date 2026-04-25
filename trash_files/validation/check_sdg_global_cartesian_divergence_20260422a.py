from __future__ import annotations

import argparse
import math
import numpy as np

from operators.sdg_cartesian_divergence_20260422a import (
    build_sdg_cartesian_divergence_setup_20260422a,
    sdg_cartesian_divergence_of_constant_state_20260422a,
    integrate_global_divergence_diagnostics_20260422a,
)
from geometry.sdg_velocity_global_20260422a import (
    sdg_velocity_roundtrip_error_20260422a,
)


def run_study(
    *,
    nsubs: list[int],
    order: int,
    N: int,
    radius: float,
    alpha0: float,
) -> None:
    print("=== SDG global flattened Cartesian divergence diagnostic ===")
    print("A/Ainv source : SDG4PDEOnSphere20260422a.pdf")
    print("domain        : global square [-1,1]^2")
    print("quadrature    : table1")
    print("q             : 1")
    print("divergence    : d_x(u1 q) + d_y(u2 q)")
    print(f"order         : {order}")
    print(f"N             : {N}")
    print(f"radius        : {radius}")
    print(f"alpha0        : {alpha0:.16e}")
    print()

    header = (
        f"{'nsub':>6s} | "
        f"{'int(div)':>14s} | "
        f"{'mean(div)':>14s} | "
        f"{'L2(div)':>14s} | "
        f"{'RMS(div)':>14s} | "
        f"{'Linf(div)':>14s} | "
        f"{'roundtrip':>12s}"
    )
    print(header)
    print("-" * len(header))

    prev_l2 = None

    for nsub in nsubs:
        setup = build_sdg_cartesian_divergence_setup_20260422a(
            nsub=nsub,
            order=order,
            N=N,
            radius=radius,
        )

        div, diag = sdg_cartesian_divergence_of_constant_state_20260422a(
            setup,
            u0=1.0,
            alpha0=alpha0,
        )

        stats = integrate_global_divergence_diagnostics_20260422a(
            div,
            setup,
            surface_weighted=True,
        )

        roundtrip = sdg_velocity_roundtrip_error_20260422a(
            setup.X,
            setup.Y,
            setup.mesh.face_ids,
            radius=radius,
            u0=1.0,
            alpha0=alpha0,
        )

        print(
            f"{nsub:6d} | "
            f"{stats['integral']:14.6e} | "
            f"{stats['mean']:14.6e} | "
            f"{stats['l2']:14.6e} | "
            f"{stats['rms']:14.6e} | "
            f"{stats['linf']:14.6e} | "
            f"{roundtrip:12.3e}",
            end="",
        )

        if prev_l2 is not None and stats["l2"] > 0.0:
            rate = math.log(prev_l2 / stats["l2"]) / math.log(2.0)
            print(f"   rate(L2)={rate:7.3f}")
        else:
            print()

        prev_l2 = stats["l2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsubs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=math.pi / 4.0)

    args = parser.parse_args()

    run_study(
        nsubs=args.nsubs,
        order=args.order,
        N=args.N,
        radius=args.radius,
        alpha0=args.alpha0,
    )


if __name__ == "__main__":
    main()