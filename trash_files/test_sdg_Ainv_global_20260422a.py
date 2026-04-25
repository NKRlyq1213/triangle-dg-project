from __future__ import annotations

import numpy as np

from data.rule_registry import load_rule
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.sphere_flattened_connectivity import build_flattened_sphere_mesh
from geometry.sdg_Ainv_global_20260422a import (
    sdg_A_Ainv_batch_20260422a,
    sdg_sqrtG_global_20260422a,
)


def _sample_element_nodes(nsub: int = 4, order: int = 4):
    mesh = build_flattened_sphere_mesh(nsub)
    rule = load_rule("table1", order)

    X, Y = map_reference_nodes_to_all_elements(
        rule["rs"],
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        mesh.EToV,
    )

    return mesh, X, Y


def test_A_Ainv_identity_20260422a():
    mesh, X, Y = _sample_element_nodes(nsub=4, order=4)

    A, Ainv = sdg_A_Ainv_batch_20260422a(
        X,
        Y,
        mesh.face_ids,
        radius=1.0,
    )

    I1 = np.einsum("...ik,...kj->...ij", A, Ainv)
    I2 = np.einsum("...ik,...kj->...ij", Ainv, A)

    eye = np.eye(2)

    assert np.max(np.abs(I1 - eye)) < 1e-11
    assert np.max(np.abs(I2 - eye)) < 1e-11


def test_det_A_is_pi_R2_20260422a():
    mesh, X, Y = _sample_element_nodes(nsub=4, order=4)

    A, _ = sdg_A_Ainv_batch_20260422a(
        X,
        Y,
        mesh.face_ids,
        radius=1.0,
    )

    detA = np.linalg.det(A)

    assert np.max(np.abs(detA - np.pi)) < 1e-10


def test_sqrtG_is_pi_R2_20260422a():
    mesh, X, Y = _sample_element_nodes(nsub=4, order=4)

    for fid in range(1, 9):
        mask = mesh.face_ids == fid

        J = sdg_sqrtG_global_20260422a(
            X[mask],
            Y[mask],
            fid,
            radius=1.0,
        )

        assert np.max(np.abs(J - np.pi)) < 1e-12


def test_A_Ainv_finite_20260422a():
    mesh, X, Y = _sample_element_nodes(nsub=4, order=4)

    A, Ainv = sdg_A_Ainv_batch_20260422a(
        X,
        Y,
        mesh.face_ids,
        radius=1.0,
    )

    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(Ainv))