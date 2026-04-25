from __future__ import annotations

import numpy as np

from problems.sphere_single_patch_rhs import (
    compute_constant_state_single_patch_rhs,
)


def main() -> None:
    alpha0 = np.pi / 2.0
    print("=== Single-patch RHS smoke test ===")
    print("q = 1")
    print("quad_table = table1")
    print("quad_order = 4")
    print(f"alpha0 = {alpha0:.3e}")
    print()

    for patch_id in range(1, 9):
        rhs, diag, setup = compute_constant_state_single_patch_rhs(
            patch_id=patch_id,
            nsub=1,
            radius=1.0,
            quad_table="table1",
            quad_order=4,
            N=4,
            alpha0=alpha0,
            tau=0.0,
        )

        total_err = np.max(np.abs(rhs))
        volume_err = np.max(np.abs(diag["volume_rhs"]))
        surface_err = np.max(np.abs(diag["surface_rhs"]))

        print(
            f"T{patch_id}: "
            f"K={setup.EToV.shape[0]:4d}, "
            f"Np={setup.X.shape[1]:3d}, "
            f"max|volume|={volume_err:.3e}, "
            f"max|surface|={surface_err:.3e}, "
            f"max|rhs|={total_err:.3e}"
        )


if __name__ == "__main__":
    main()