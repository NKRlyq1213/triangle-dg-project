from __future__ import annotations

import numpy as np

from geometry.sphere_mesh import build_sphere_patch_mesh, mesh_size_summary
from geometry.sphere_velocity import (
    build_velocity_on_sphere_mesh,
    velocity_roundtrip_error,
)


def _print_range(name: str, arr: np.ndarray) -> None:
    print(
        f"{name:12s}: "
        f"min={np.min(arr): .6e}, "
        f"max={np.max(arr): .6e}, "
        f"mean={np.mean(arr): .6e}, "
        f"std={np.std(arr): .6e}"
    )


def main() -> None:
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=1.0,
        quad_table="table1",
        quad_order=4,
    )

    velocity = build_velocity_on_sphere_mesh(
        mesh,
        u0=1.0,
        alpha0=np.pi / 4.0,
    )

    print("=== Sphere velocity diagnostics ===")
    print("Mesh summary:")
    for k, v in mesh_size_summary(mesh).items():
        print(f"  {k:35s}: {v}")

    print()
    print("Velocity field:")
    print(f"  u0     = 1.0")
    print(f"  alpha0 = pi/4")

    print()
    print("[1] spherical components")
    _print_range("u_sph", velocity.u_sph)
    _print_range("v_sph", velocity.v_sph)

    print()
    print("[2] contravariant components")
    _print_range("u1", velocity.u1)
    _print_range("u2", velocity.u2)

    print()
    print("[3] roundtrip errors")
    errors = velocity_roundtrip_error(mesh, velocity)
    for patch_id in range(1, 9):
        print(f"  patch {patch_id}: max error = {errors[patch_id]:.3e}")

    print()
    print("[4] finite check")
    print(f"  all finite u_sph: {np.all(np.isfinite(velocity.u_sph))}")
    print(f"  all finite v_sph: {np.all(np.isfinite(velocity.v_sph))}")
    print(f"  all finite u1   : {np.all(np.isfinite(velocity.u1))}")
    print(f"  all finite u2   : {np.all(np.isfinite(velocity.u2))}")


if __name__ == "__main__":
    main()