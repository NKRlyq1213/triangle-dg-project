from __future__ import annotations

import numpy as np
import pytest

from time_integration.lsrk54 import integrate_lsrk54


def test_integrate_lsrk54_applies_post_step_transform_each_step() -> None:
    q0 = np.ones((2, 3), dtype=float)

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        return np.zeros_like(q)

    def post_step_transform(t: float, q: np.ndarray) -> np.ndarray:
        return q + 1.0

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=0.0,
        tf=0.3,
        dt=0.1,
        post_step_transform=post_step_transform,
    )

    assert nsteps == 3
    assert np.isclose(tf_used, 0.3)
    assert np.allclose(qf, q0 + 3.0)


def test_integrate_lsrk54_post_step_transform_shape_validation() -> None:
    q0 = np.ones((2, 3), dtype=float)

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        return np.zeros_like(q)

    def bad_post_step_transform(t: float, q: np.ndarray) -> np.ndarray:
        return q.reshape(-1)

    with pytest.raises(ValueError, match="post_step_transform"):
        integrate_lsrk54(
            rhs=rhs,
            q0=q0,
            t0=0.0,
            tf=0.1,
            dt=0.1,
            post_step_transform=bad_post_step_transform,
        )
