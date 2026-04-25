from __future__ import annotations

import numpy as np

from geometry.sphere_flattened_mesh import (
    build_flattened_sphere_raw_mesh,
    mesh_element_areas,
)


def test_flattened_mesh_element_count():
    for nsub in [1, 2, 4, 8]:
        mesh = build_flattened_sphere_raw_mesh(nsub)

        assert mesh.EToV.shape[0] == 8 * nsub**2
        assert mesh.face_ids.shape == (8 * nsub**2,)


def test_flattened_mesh_face_id_counts():
    for nsub in [1, 2, 4]:
        mesh = build_flattened_sphere_raw_mesh(nsub)

        unique, counts = np.unique(mesh.face_ids, return_counts=True)
        assert np.array_equal(unique, np.arange(1, 9))
        assert np.all(counts == nsub**2)


def test_flattened_mesh_all_elements_ccw():
    mesh = build_flattened_sphere_raw_mesh(5)
    areas = mesh_element_areas(mesh.nodes, mesh.EToV)

    assert np.all(areas > 0.0)


def test_flattened_mesh_total_flat_area_is_four():
    for nsub in [1, 2, 4, 8]:
        mesh = build_flattened_sphere_raw_mesh(nsub)
        areas = mesh_element_areas(mesh.nodes, mesh.EToV)

        assert abs(np.sum(areas) - 4.0) < 1e-12


def test_flattened_mesh_nodes_inside_square():
    mesh = build_flattened_sphere_raw_mesh(8)

    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    assert np.all(x >= -1.0 - 1e-12)
    assert np.all(x <=  1.0 + 1e-12)
    assert np.all(y >= -1.0 - 1e-12)
    assert np.all(y <=  1.0 + 1e-12)