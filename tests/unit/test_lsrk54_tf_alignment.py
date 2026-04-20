from __future__ import annotations

import numpy as np

from time_integration.lsrk54 import integrate_lsrk54, is_tf_reached, tf_align_tolerance


def _zero_rhs(t: float, q: np.ndarray) -> np.ndarray:
    return np.zeros_like(q)


def _linear_decay_rhs(t: float, q: np.ndarray) -> np.ndarray:
    _ = t
    return -np.asarray(q, dtype=float)


def _empirical_rates(errors: list[float]) -> list[float]:
    rates = [np.nan]
    for i in range(1, len(errors)):
        rates.append(float(np.log(errors[i - 1] / errors[i]) / np.log(2.0)))
    return rates


def test_integrate_lsrk54_fixed_dt_lands_exactly_on_tf_when_non_divisible() -> None:
    q0 = np.array([1.0], dtype=float)

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=_zero_rhs,
        q0=q0,
        t0=0.0,
        tf=1.0,
        dt=0.3,
    )

    assert np.allclose(qf, q0)
    assert tf_used == 1.0
    assert nsteps == 4


def test_integrate_lsrk54_fixed_dt_lands_exactly_on_tf_when_divisible() -> None:
    q0 = np.array([2.0, -1.0], dtype=float)

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=_zero_rhs,
        q0=q0,
        t0=0.0,
        tf=1.0,
        dt=0.25,
    )

    assert np.allclose(qf, q0)
    assert tf_used == 1.0
    assert nsteps == 4


def test_integrate_lsrk54_dt_getter_lands_exactly_on_tf() -> None:
    q0 = np.array([3.0], dtype=float)

    def dt_getter(t: float, q: np.ndarray) -> float:
        _ = (t, q)
        return 0.3

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=_zero_rhs,
        q0=q0,
        t0=0.0,
        tf=1.0,
        dt_getter=dt_getter,
    )

    assert np.allclose(qf, q0)
    assert tf_used == 1.0
    assert nsteps == 4


def test_integrate_lsrk54_near_equal_t0_tf_returns_tf_target() -> None:
    q0 = np.array([0.0], dtype=float)
    tf_target = 1.0
    t0 = tf_target - 5.0e-16

    _qf, tf_used, nsteps = integrate_lsrk54(
        rhs=_zero_rhs,
        q0=q0,
        t0=t0,
        tf=tf_target,
        dt=0.1,
    )

    assert nsteps == 0
    assert tf_used == tf_target


def test_is_tf_reached_uses_machine_scale_tolerance() -> None:
    tf = 2.0 * np.pi
    tol = tf_align_tolerance(tf)

    assert is_tf_reached(tf - 0.5 * tol, tf)
    assert not is_tf_reached(tf - 2.0 * tol, tf)


def test_integrate_lsrk54_temporal_order_preserved_with_unaligned_tf() -> None:
    tf = 2.0 * np.pi  # intentionally not aligned with dt below
    dt_list = [0.2, 0.1, 0.05, 0.025]
    q0 = np.array([1.0], dtype=float)

    errors: list[float] = []
    for dt in dt_list:
        qf, tf_used, _nsteps = integrate_lsrk54(
            rhs=_linear_decay_rhs,
            q0=q0,
            t0=0.0,
            tf=tf,
            dt=dt,
        )
        assert tf_used == tf

        q_exact = np.exp(-tf)
        errors.append(abs(float(qf[0]) - float(q_exact)))

    rates = _empirical_rates(errors)
    assert rates[-1] > 3.8, f"Final temporal order too low for unaligned tf: {rates[-1]}"
    assert rates[-2] > 3.7, f"Penultimate temporal order too low for unaligned tf: {rates[-2]}"
