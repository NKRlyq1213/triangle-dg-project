from __future__ import annotations

import numpy as np
import pytest

from geometry import (
    build_octa_sphere_atlas,
    map_chart_to_sphere,
    map_chart_jacobian,
    sphere_transport_coefficients,
    tag_square_chart_faces,
    validate_elements_single_face,
    build_cross_face_edge_pairs,
)


def _sample_local_triangle(
    n: int,
    *,
    radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # Uniform sampling over x>=0, y>=0, x+y<=radius.
    r1 = rng.random(n)
    r2 = rng.random(n)
    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    return radius * r1, radius * r2


def _edge_chart_xy(edge_id: int, t: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    if edge_id == 1:
        # B -> C
        return np.zeros_like(t), radius * (1.0 - t)
    if edge_id == 2:
        # C -> A
        return radius * t, np.zeros_like(t)
    if edge_id == 3:
        # A -> B
        return radius * (1.0 - t), radius * t
    raise ValueError("edge_id must be in [1, 3].")


def _velocity_constant(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del t
    return (
        np.ones_like(X),
        2.0 * np.ones_like(Y),
        -0.5 * np.ones_like(Z),
    )


def test_atlas_shapes_and_reciprocity() -> None:
    atlas = build_octa_sphere_atlas(radius=1.0)

    assert atlas.neighbor_face.shape == (8, 3)
    assert atlas.neighbor_edge.shape == (8, 3)
    assert atlas.edge_flip.shape == (8, 3)
    assert np.all((atlas.neighbor_face >= 1) & (atlas.neighbor_face <= 8))
    assert np.all((atlas.neighbor_edge >= 1) & (atlas.neighbor_edge <= 3))

    for f in range(8):
        for e in range(3):
            g = int(atlas.neighbor_face[f, e] - 1)
            ge = int(atlas.neighbor_edge[f, e] - 1)
            assert int(atlas.neighbor_face[g, ge]) == (f + 1)
            assert int(atlas.neighbor_edge[g, ge]) == (e + 1)
            assert bool(atlas.edge_flip[g, ge]) == bool(atlas.edge_flip[f, e])


def test_equal_area_jacobian_and_total_area() -> None:
    radius = 1.0
    rng = np.random.default_rng(20260421)
    chart_area = 0.5 * radius * radius

    face_areas = []
    for face_id in range(1, 9):
        x, y = _sample_local_triangle(4000, radius=radius, rng=rng)
        _, _, _, _, _, _, G = map_chart_jacobian(face_id=face_id, x=x, y=y, radius=radius)
        assert np.all(np.isfinite(G))
        assert np.all(G > 0.0)
        assert np.allclose(G, np.pi, atol=2e-12, rtol=0.0)
        face_areas.append(float(np.mean(G) * chart_area))

    face_areas = np.asarray(face_areas, dtype=float)
    assert np.allclose(face_areas, 0.5 * np.pi * radius * radius, atol=1e-10, rtol=0.0)
    assert np.isclose(float(np.sum(face_areas)), 4.0 * np.pi * radius * radius, atol=1e-9, rtol=0.0)


def test_face_symmetry_relations() -> None:
    radius = 1.0
    rng = np.random.default_rng(20260422)
    x, y = _sample_local_triangle(2000, radius=radius, rng=rng)

    X1, Y1, Z1 = map_chart_to_sphere(face_id=1, x=x, y=y, radius=radius)
    X2, Y2, Z2 = map_chart_to_sphere(face_id=2, x=x, y=y, radius=radius)
    X3, Y3, Z3 = map_chart_to_sphere(face_id=3, x=x, y=y, radius=radius)
    X4, Y4, Z4 = map_chart_to_sphere(face_id=4, x=x, y=y, radius=radius)
    X5, Y5, Z5 = map_chart_to_sphere(face_id=5, x=x, y=y, radius=radius)
    X6, Y6, Z6 = map_chart_to_sphere(face_id=6, x=x, y=y, radius=radius)
    X7, Y7, Z7 = map_chart_to_sphere(face_id=7, x=x, y=y, radius=radius)
    X8, Y8, Z8 = map_chart_to_sphere(face_id=8, x=x, y=y, radius=radius)

    assert np.allclose(X2, -Y1, atol=1e-12, rtol=0.0)
    assert np.allclose(Y2, X1, atol=1e-12, rtol=0.0)
    assert np.allclose(Z2, Z1, atol=1e-12, rtol=0.0)

    assert np.allclose(X3, -X1, atol=1e-12, rtol=0.0)
    assert np.allclose(Y3, -Y1, atol=1e-12, rtol=0.0)
    assert np.allclose(Z3, Z1, atol=1e-12, rtol=0.0)

    assert np.allclose(X4, Y1, atol=1e-12, rtol=0.0)
    assert np.allclose(Y4, -X1, atol=1e-12, rtol=0.0)
    assert np.allclose(Z4, Z1, atol=1e-12, rtol=0.0)

    assert np.allclose(X5, X1, atol=1e-12, rtol=0.0)
    assert np.allclose(Y5, Y1, atol=1e-12, rtol=0.0)
    assert np.allclose(Z5, -Z1, atol=1e-12, rtol=0.0)

    assert np.allclose(X6, -Y1, atol=1e-12, rtol=0.0)
    assert np.allclose(Y6, X1, atol=1e-12, rtol=0.0)
    assert np.allclose(Z6, -Z1, atol=1e-12, rtol=0.0)

    assert np.allclose(X7, -X1, atol=1e-12, rtol=0.0)
    assert np.allclose(Y7, -Y1, atol=1e-12, rtol=0.0)
    assert np.allclose(Z7, -Z1, atol=1e-12, rtol=0.0)

    assert np.allclose(X8, Y1, atol=1e-12, rtol=0.0)
    assert np.allclose(Y8, -X1, atol=1e-12, rtol=0.0)
    assert np.allclose(Z8, -Z1, atol=1e-12, rtol=0.0)


def test_cross_face_edge_continuity_and_flip() -> None:
    radius = 1.0
    atlas = build_octa_sphere_atlas(radius=radius)
    pairs = build_cross_face_edge_pairs(nfp=17, atlas=atlas)
    t = np.linspace(0.0, 1.0, 17, dtype=float)

    for i in range(int(pairs["face_a"].size)):
        fa = int(pairs["face_a"][i])
        ea = int(pairs["edge_a"][i])
        fb = int(pairs["face_b"][i])
        eb = int(pairs["edge_b"][i])
        flip = bool(pairs["edge_flip"][i])

        xa, ya = _edge_chart_xy(ea, t, radius)
        tb = t[::-1] if flip else t
        xb, yb = _edge_chart_xy(eb, tb, radius)

        Xa, Ya, Za = map_chart_to_sphere(face_id=fa, x=xa, y=ya, radius=radius)
        Xb, Yb, Zb = map_chart_to_sphere(face_id=fb, x=xb, y=yb, radius=radius)

        assert np.allclose(Xa, Xb, atol=1e-12, rtol=0.0)
        assert np.allclose(Ya, Yb, atol=1e-12, rtol=0.0)
        assert np.allclose(Za, Zb, atol=1e-12, rtol=0.0)


def test_face_tagging_and_single_face_validation() -> None:
    x = np.array([0.25, -0.30, -0.20, 0.40, 0.80, -0.80, -0.70, 0.90], dtype=float)
    y = np.array([0.20, 0.20, -0.30, -0.20, 0.60, 0.40, -0.80, -0.30], dtype=float)
    face = tag_square_chart_faces(x, y, radius=1.0)
    assert np.array_equal(face, np.arange(1, 9, dtype=np.int64))

    # Valid: each row in one face.
    x_nodes_ok = np.array(
        [
            [0.20, 0.25, 0.30],    # face 1
            [-0.75, -0.60, -0.70], # face 7
        ],
        dtype=float,
    )
    y_nodes_ok = np.array(
        [
            [0.20, 0.15, 0.10],
            [-0.40, -0.55, -0.45],
        ],
        dtype=float,
    )
    elem_face = validate_elements_single_face(x_nodes_ok, y_nodes_ok, radius=1.0)
    assert np.array_equal(elem_face, np.array([1, 7], dtype=np.int64))

    # Invalid: mixed-face row.
    x_nodes_mixed = np.array([[0.2, -0.2, 0.2]], dtype=float)
    y_nodes_mixed = np.array([[0.2, 0.2, 0.1]], dtype=float)
    with pytest.raises(ValueError, match="mixed-face"):
        validate_elements_single_face(x_nodes_mixed, y_nodes_mixed, radius=1.0)

    # Invalid: seam node.
    x_nodes_seam = np.array([[0.3, 0.6, 0.1]], dtype=float)
    y_nodes_seam = np.array([[0.2, 0.4, 0.3]], dtype=float)  # second node: |x|+|y|=1
    with pytest.raises(ValueError, match="seam"):
        validate_elements_single_face(x_nodes_seam, y_nodes_seam, radius=1.0)


def test_sphere_transport_coefficients_tangent_consistency() -> None:
    radius = 1.0
    rng = np.random.default_rng(20260423)
    x, y = _sample_local_triangle(2000, radius=0.95 * radius, rng=rng)

    for face_id in range(1, 9):
        X, Y, Z, a, b, _ = sphere_transport_coefficients(
            face_id=face_id,
            x=x,
            y=y,
            velocity_sphere=_velocity_constant,
            t=0.25,
            radius=radius,
            use_numba=False,
        )
        dXdx, dXdy, dYdx, dYdy, dZdx, dZdy, _ = map_chart_jacobian(face_id, x, y, radius=radius)

        Urec_x = a * dXdx + b * dXdy
        Urec_y = a * dYdx + b * dYdy
        Urec_z = a * dZdx + b * dZdy
        Urec = np.stack([Urec_x, Urec_y, Urec_z], axis=-1)

        Ux, Uy, Uz = _velocity_constant(X, Y, Z, 0.25)
        U = np.stack([Ux, Uy, Uz], axis=-1)
        N = np.stack([X, Y, Z], axis=-1) / radius

        # Tangency check.
        tangency = np.sum(Urec * N, axis=-1)
        assert np.max(np.abs(tangency)) < 1e-10

        # Recovered vector should match explicit tangent projection.
        Ut = U - np.sum(U * N, axis=-1)[:, None] * N
        rel = np.linalg.norm(Urec - Ut, axis=-1) / np.maximum(1.0, np.linalg.norm(Ut, axis=-1))
        assert np.max(rel) < 1e-10
