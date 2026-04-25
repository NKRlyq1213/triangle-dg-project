from __future__ import annotations

import numpy as np

from geometry.sphere_flattened_connectivity import (
    build_flattened_sphere_mesh,
    validate_closed_connectivity,
)


def test_flattened_sphere_connectivity_is_closed():
    for nsub in [1, 2, 4, 8]:
        mesh = build_flattened_sphere_mesh(nsub)

        assert mesh.conn.is_boundary.shape == (mesh.EToV.shape[0], 3)
        assert not np.any(mesh.conn.is_boundary)


def test_flattened_sphere_connectivity_validates():
    for nsub in [1, 2, 4]:
        mesh = build_flattened_sphere_mesh(nsub)
        summary = validate_closed_connectivity(mesh)

        assert summary["num_elements"] == 8 * nsub**2
        assert summary["num_local_faces"] == 3 * 8 * nsub**2


def test_flattened_sphere_glued_boundary_pair_count():
    """
    Before spherical gluing, the square outer boundary has 8*nsub boundary edges:
        each side has 2*nsub edges
        four sides => 8*nsub

    After gluing, these become 4*nsub pairs.
    """
    for nsub in [1, 2, 4, 8]:
        mesh = build_flattened_sphere_mesh(nsub)

        assert mesh.conn.planar_boundary_faces.shape[0] == 8 * nsub
        assert mesh.conn.glued_boundary_pairs.shape[0] == 4 * nsub


def test_flattened_sphere_EToF_is_one_based():
    mesh = build_flattened_sphere_mesh(4)

    assert np.all(mesh.conn.EToF >= 1)
    assert np.all(mesh.conn.EToF <= 3)


def test_flattened_sphere_connectivity_symmetry():
    mesh = build_flattened_sphere_mesh(4)

    EToE = mesh.conn.EToE
    EToF = mesh.conn.EToF

    K = mesh.EToV.shape[0]

    for k in range(K):
        for jf in range(3):
            f = jf + 1

            nbr = int(EToE[k, jf])
            nbr_f = int(EToF[k, jf])
            nbr_jf = nbr_f - 1

            assert EToE[nbr, nbr_jf] == k
            assert EToF[nbr, nbr_jf] == f


def test_flattened_sphere_all_glued_pairs_are_flipped():
    """
    Current spherical outer-boundary gluing convention reverses face-node order.
    """
    mesh = build_flattened_sphere_mesh(4)

    for ka, fa, kb, fb in mesh.conn.glued_boundary_pairs:
        assert mesh.conn.face_flip[ka, fa - 1]
        assert mesh.conn.face_flip[kb, fb - 1]