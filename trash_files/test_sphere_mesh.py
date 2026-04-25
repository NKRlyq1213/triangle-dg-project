from __future__ import annotations

import numpy as np

from geometry.sphere_mesh import (
    build_sphere_patch_mesh,
    mesh_size_summary,
)
from geometry.sphere_metrics import expected_sqrtG
from geometry.sphere_patches import patch_signs


def test_sphere_mesh_shapes():
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=1.0,
        quad_table="table2",
        quad_order=4,
    )

    summary = mesh_size_summary(mesh)

    assert summary["num_patches"] == 8
    assert summary["num_elements_per_patch"] == 25
    assert summary["num_elements_total"] == 200
    assert summary["num_directed_patch_trace_pairs"] == 24

    K = summary["num_elements_total"]
    Nq = summary["num_volume_nodes_per_element"]

    assert mesh.elem_to_patch.shape == (K,)
    assert mesh.elem_to_local_id.shape == (K,)
    assert mesh.volume_nodes_local.shape == (K, Nq, 2)
    assert mesh.volume_xyz.shape == (K, Nq, 3)
    assert mesh.volume_lambda.shape == (K, Nq)
    assert mesh.volume_theta.shape == (K, Nq)
    assert mesh.volume_sqrtG.shape == (K, Nq)


def test_sphere_mesh_nodes_are_on_sphere():
    radius = 1.0
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=radius,
        quad_table="table2",
        quad_order=4,
    )

    X = mesh.volume_xyz[:, :, 0]
    Y = mesh.volume_xyz[:, :, 1]
    Z = mesh.volume_xyz[:, :, 2]

    err = np.max(np.abs(X**2 + Y**2 + Z**2 - radius**2))
    assert err < 1e-12


def test_sphere_mesh_patch_signs():
    radius = 1.0
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=radius,
        quad_table="table2",
        quad_order=4,
    )

    for patch_id in range(1, 9):
        mask = mesh.elem_to_patch == patch_id

        X = mesh.volume_xyz[mask, :, 0]
        Y = mesh.volume_xyz[mask, :, 1]
        Z = mesh.volume_xyz[mask, :, 2]

        sx, sy, sz = patch_signs(patch_id)

        assert np.all(sx * X >= -1e-12)
        assert np.all(sy * Y >= -1e-12)
        assert np.all(sz * Z >= -1e-12)


def test_sphere_mesh_sqrtG_constant():
    radius = 1.0
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=radius,
        quad_table="table2",
        quad_order=4,
    )

    err = np.max(
        np.abs(mesh.volume_sqrtG - expected_sqrtG(radius=radius))
    )

    assert err < 1e-11


def test_sphere_mesh_patch_blocks_are_contiguous():
    """
    Patch-block layout should be contiguous:

        patch 1 elements first,
        patch 2 elements second,
        ...
        patch 8 elements last.

    This is useful for vectorized patch-wise operations.
    """
    nsub = 4
    mesh = build_sphere_patch_mesh(
        nsub=nsub,
        radius=1.0,
        quad_table="table2",
        quad_order=4,
    )

    K_patch = nsub**2

    for patch_id in range(1, 9):
        start = (patch_id - 1) * K_patch
        stop = patch_id * K_patch

        assert np.all(mesh.elem_to_patch[start:stop] == patch_id)
        assert np.all(mesh.elem_to_local_id[start:stop] == np.arange(K_patch))