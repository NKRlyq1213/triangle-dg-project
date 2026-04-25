from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from geometry import (
    map_chart_to_sphere,
    map_chart_jacobian,
    sphere_transport_coefficients,
)


def _velocity_profile(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.cos(X + 0.25 * t) + 0.1 * Y,
        np.sin(Y - 0.3 * t) - 0.2 * Z,
        0.3 * X - 0.15 * Y + 0.05 * np.cos(t) * np.ones_like(Z),
    )


def _sample_local_triangle(
    n: int,
    *,
    radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    r1 = rng.random(n)
    r2 = rng.random(n)
    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    return radius * r1, radius * r2


@pytest.mark.parametrize("face_id", (1, 2, 5, 8))
def test_map_and_jacobian_numba_matches_numpy(face_id: int) -> None:
    rng = np.random.default_rng(20260424 + face_id)
    x, y = _sample_local_triangle(3000, radius=0.95, rng=rng)

    X_np, Y_np, Z_np = map_chart_to_sphere(face_id, x, y, radius=1.0, use_numba=False)
    X_nb, Y_nb, Z_nb = map_chart_to_sphere(face_id, x, y, radius=1.0, use_numba=True)

    assert np.allclose(X_nb, X_np, atol=1e-12, rtol=0.0)
    assert np.allclose(Y_nb, Y_np, atol=1e-12, rtol=0.0)
    assert np.allclose(Z_nb, Z_np, atol=1e-12, rtol=0.0)

    jac_np = map_chart_jacobian(face_id, x, y, radius=1.0, use_numba=False)
    jac_nb = map_chart_jacobian(face_id, x, y, radius=1.0, use_numba=True)
    for arr_nb, arr_np in zip(jac_nb, jac_np):
        assert np.allclose(arr_nb, arr_np, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("face_id", (1, 3, 6))
def test_transport_coefficients_numba_matches_numpy(face_id: int) -> None:
    rng = np.random.default_rng(20260430 + face_id)
    x, y = _sample_local_triangle(2500, radius=0.9, rng=rng)

    out_np = sphere_transport_coefficients(
        face_id=face_id,
        x=x,
        y=y,
        velocity_sphere=_velocity_profile,
        t=0.375,
        radius=1.0,
        use_numba=False,
    )
    out_nb = sphere_transport_coefficients(
        face_id=face_id,
        x=x,
        y=y,
        velocity_sphere=_velocity_profile,
        t=0.375,
        radius=1.0,
        use_numba=True,
    )

    for arr_nb, arr_np in zip(out_nb, out_np):
        assert np.allclose(arr_nb, arr_np, atol=1e-11, rtol=1e-11)

