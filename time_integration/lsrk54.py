from __future__ import annotations

import numpy as np


# 5-stage, 4th-order low-storage Runge-Kutta coefficients
RK4A = np.array([
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
], dtype=float)

RK4B = np.array([
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
], dtype=float)

RK4C = np.array([
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0,
], dtype=float)


def lsrk54_step(rhs, t: float, q: np.ndarray, dt: float) -> np.ndarray:
    """
    One step of 5-stage 4th-order low-storage RK.

    Parameters
    ----------
    rhs : callable
        rhs(t, q) -> dqdt, same shape as q
    t : float
        Current time
    q : np.ndarray
        Current state
    dt : float
        Time step

    Returns
    -------
    np.ndarray
        Updated state after one full LSRK54 step
    """
    q = np.asarray(q, dtype=float).copy()
    res = np.zeros_like(q)

    for s in range(5):
        t_stage = t + RK4C[s] * dt
        dqdt = np.asarray(rhs(t_stage, q), dtype=float)

        if dqdt.shape != q.shape:
            raise ValueError("rhs(t, q) must return the same shape as q.")

        res = RK4A[s] * res + dt * dqdt
        q = q + RK4B[s] * res

    return q


def integrate_lsrk54(
    rhs,
    q0: np.ndarray,
    t0: float,
    tf: float,
    dt: float | None = None,
    dt_getter=None,
    max_steps: int = 10_000_000,
) -> tuple[np.ndarray, float, int]:
    """
    Integrate q_t = rhs(t, q) from t0 to tf using repeated LSRK54 steps.

    Time-step selection
    -------------------
    Exactly one of the following must be provided:
    - dt : fixed nominal time step
    - dt_getter : callable dt_getter(t, q) -> nominal time step

    The integrator marches forward until reaching tf.
    If the next nominal step would overshoot tf, it takes a final short step:
        dt_step = min(dt_nominal, tf - t)

    Returns
    -------
    tuple
        (qf, tf_used, nsteps)
    """
    q = np.asarray(q0, dtype=float).copy()

    if tf < t0:
        raise ValueError("Require tf >= t0.")
    if (dt is None) == (dt_getter is None):
        raise ValueError("Provide exactly one of dt or dt_getter.")

    t = float(t0)
    nsteps = 0

    if np.isclose(tf, t0, atol=1e-15, rtol=1e-15):
        return q, t, nsteps

    while t < tf:
        if nsteps >= max_steps:
            raise RuntimeError("Maximum number of steps exceeded in integrate_lsrk54.")

        if dt_getter is None:
            dt_nominal = float(dt)
        else:
            dt_nominal = float(dt_getter(t, q))

        if dt_nominal <= 0.0:
            raise ValueError("Time step must be positive.")

        dt_step = min(dt_nominal, tf - t)

        q = lsrk54_step(rhs, t, q, dt_step)
        t += dt_step
        nsteps += 1

        # protect against roundoff stalling very near tf
        if abs(tf - t) <= 1e-15 * max(1.0, abs(tf)):
            t = float(tf)

    return q, t, nsteps