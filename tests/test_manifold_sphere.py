from __future__ import annotations

import numpy as np

from geometry.connectivity import build_face_connectivity, validate_face_connectivity
from geometry.sphere_manifold_mesh import generate_spherical_octahedron_mesh
from geometry.sphere_manifold_metrics import build_manifold_geometry_cache
from operators.manifold_rhs import (
    build_manifold_table1_k4_reference_operators,
    manifold_rhs_constant_field,
)
from problems.sphere_advection import solid_body_velocity_xyz


def _closed_connectivity(EToV: np.ndarray) -> dict:
    vertex_ids = np.arange(int(np.max(EToV)) + 1, dtype=float)
    return build_face_connectivity(
        VX=vertex_ids,
        VY=np.zeros_like(vertex_ids),
        EToV=EToV,
        classify_boundary=None,
    )


def test_spherical_octahedron_mesh_is_closed_and_on_sphere():
    R = 2.5
    for n_div in (1, 2, 4):
        nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=n_div, R=R)

        assert EToV.shape == (8 * n_div * n_div, 3)
        assert np.allclose(np.linalg.norm(nodes_xyz, axis=1), R)

        conn = _closed_connectivity(EToV)
        summary = validate_face_connectivity(EToV, conn)
        assert summary["n_boundary_faces"] == 0


def test_manifold_metrics_are_positive_and_velocity_is_tangent():
    ref_ops = build_manifold_table1_k4_reference_operators()
    nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=2, R=1.0)
    geom = build_manifold_geometry_cache(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=ref_ops.rs_nodes,
        Dr=ref_ops.Dr,
        Ds=ref_ops.Ds,
        R=1.0,
    )

    assert np.all(geom.J > 0.0)
    radial_dot_normal = geom.nx * geom.X + geom.ny * geom.Y + geom.nz * geom.Z
    assert np.all(radial_dot_normal > 0.0)

    U, V, W = solid_body_velocity_xyz(geom.X, geom.Y, geom.Z)
    tangent_dot = U * geom.X + V * geom.Y + W * geom.Z
    assert np.max(np.abs(tangent_dot)) < 1.0e-13


def test_constant_field_surface_term_is_zero_and_rhs_is_negative_divergence():
    ref_ops = build_manifold_table1_k4_reference_operators()
    nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=2, R=1.0)
    geom = build_manifold_geometry_cache(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=ref_ops.rs_nodes,
        Dr=ref_ops.Dr,
        Ds=ref_ops.Ds,
        R=1.0,
    )
    U, V, W = solid_body_velocity_xyz(geom.X, geom.Y, geom.Z)

    diag = manifold_rhs_constant_field(geom, U, V, W, ref_ops=ref_ops)
    surface = np.asarray(diag["surface_term"], dtype=float)
    rhs = np.asarray(diag["rhs"], dtype=float)
    div = np.asarray(diag["divergence"], dtype=float)

    assert np.max(np.abs(surface)) < 1.0e-14
    assert np.allclose(rhs, -div)


def test_constant_field_divergence_decreases_under_refinement():
    ref_ops = build_manifold_table1_k4_reference_operators()
    errors = []

    for n_div in (2, 4, 8):
        nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=n_div, R=1.0)
        geom = build_manifold_geometry_cache(
            nodes_xyz=nodes_xyz,
            EToV=EToV,
            rs_nodes=ref_ops.rs_nodes,
            Dr=ref_ops.Dr,
            Ds=ref_ops.Ds,
            R=1.0,
        )
        U, V, W = solid_body_velocity_xyz(geom.X, geom.Y, geom.Z)
        diag = manifold_rhs_constant_field(geom, U, V, W, ref_ops=ref_ops)
        errors.append(float(np.max(np.abs(np.asarray(diag["rhs"], dtype=float)))))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
