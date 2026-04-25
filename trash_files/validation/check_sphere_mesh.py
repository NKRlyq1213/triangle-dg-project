from __future__ import annotations

import numpy as np

from geometry.sphere_mesh import (
    build_sphere_patch_mesh,
    mesh_size_summary,
)


def main() -> None:
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=1.0,
        quad_table="table1",
        quad_order=4,
    )

    summary = mesh_size_summary(mesh)

    print("=== Sphere mesh summary ===")
    for key, value in summary.items():
        print(f"{key:35s}: {value}")

    print()
    print("=== Array shapes ===")
    print(f"elem_to_patch          : {mesh.elem_to_patch.shape}")
    print(f"elem_to_local_id       : {mesh.elem_to_local_id.shape}")
    print(f"elem_vertices_local    : {mesh.elem_vertices_local.shape}")
    print(f"volume_nodes_local     : {mesh.volume_nodes_local.shape}")
    print(f"volume_lambda          : {mesh.volume_lambda.shape}")
    print(f"volume_theta           : {mesh.volume_theta.shape}")
    print(f"volume_xyz             : {mesh.volume_xyz.shape}")
    print(f"volume_sqrtG           : {mesh.volume_sqrtG.shape}")

    print()
    print("=== Radius identity ===")
    X = mesh.volume_xyz[:, :, 0]
    Y = mesh.volume_xyz[:, :, 1]
    Z = mesh.volume_xyz[:, :, 2]
    radius_err = np.max(np.abs(X**2 + Y**2 + Z**2 - mesh.radius**2))
    print(f"max |X^2+Y^2+Z^2-R^2| = {radius_err:.3e}")

    print()
    print("=== sqrtG ===")
    print(f"min sqrtG = {np.min(mesh.volume_sqrtG):.15e}")
    print(f"max sqrtG = {np.max(mesh.volume_sqrtG):.15e}")
    print(f"std sqrtG = {np.std(mesh.volume_sqrtG):.3e}")

    print()
    print("=== Patch element ranges ===")
    K_patch = mesh.nsub**2
    for patch_id in range(1, 9):
        start = (patch_id - 1) * K_patch
        stop = patch_id * K_patch - 1
        print(f"T{patch_id}: element ids [{start}, {stop}]")


if __name__ == "__main__":
    main()