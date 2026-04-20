from __future__ import annotations

import numpy as np


BLOWUP_BREAK_ABS = 1000.0


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


def tf_align_tolerance(tf: float, t0: float = 0.0) -> float:
    scale = max(1.0, abs(float(tf)), abs(float(t0)))
    return 1e-15 * scale


def is_tf_reached(t: float, tf: float, *, tol: float | None = None) -> bool:
    tol_eff = tf_align_tolerance(tf, t) if tol is None else float(tol)
    return abs(float(tf) - float(t)) <= tol_eff


def _lsrk54_step_inplace(rhs, t: float, q: np.ndarray, dt: float, res: np.ndarray) -> None:
    """
    In-place 5-stage LSRK54 update on q using reusable residual buffer.

    Parameters
    ----------
    rhs : callable
        rhs(t, q) -> dqdt, same shape as q
    t : float
        Current time
    q : np.ndarray
        State, updated in place
    dt : float
        Time step
    res : np.ndarray
        Residual workspace, same shape as q
    """
    res.fill(0.0)

    for s in range(5):
        t_stage = t + RK4C[s] * dt
        dqdt = np.asarray(rhs(t_stage, q), dtype=float)

        if dqdt.shape != q.shape:
            raise ValueError("rhs(t, q) must return the same shape as q.")

        res *= RK4A[s]
        res += dt * dqdt
        q += RK4B[s] * res


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
    q_out = np.asarray(q, dtype=float).copy()
    res = np.zeros_like(q_out)
    _lsrk54_step_inplace(rhs=rhs, t=t, q=q_out, dt=dt, res=res)
    return q_out


def integrate_lsrk54(
    rhs,
    q0: np.ndarray,
    t0: float,
    tf: float,
    dt: float | None = None,
    dt_getter=None,
    post_step_transform=None,
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

    Notes
    -----
    A simple blow-up guard is applied: if max(abs(q)) > 1000
    after a step, integration stops early.

    If provided, post_step_transform is applied after each accepted full
    LSRK step as:
        q <- post_step_transform(t, q)
    where t is the updated time after the step.
    """
    q = np.asarray(q0, dtype=float).copy()
    q_shape = q.shape

    if tf < t0:
        raise ValueError("Require tf >= t0.")
    if (dt is None) == (dt_getter is None):
        raise ValueError("Provide exactly one of dt or dt_getter.")

    t = float(t0)
    tf_target = float(tf)
    tol = tf_align_tolerance(tf_target, t0)
    nsteps = 0

    if is_tf_reached(t, tf_target, tol=tol):
        return q, tf_target, nsteps

    res = np.zeros_like(q)

    if dt_getter is None:
        dt_nominal = float(dt)
        if dt_nominal <= 0.0:
            raise ValueError("Time step must be positive.")

        remaining = tf_target - t
        n_full = max(0, int(np.floor(remaining / dt_nominal)))
        while n_full > 0 and (t + n_full * dt_nominal) > (tf_target + tol):
            n_full -= 1

        if nsteps + n_full > max_steps:
            raise RuntimeError("Maximum number of steps exceeded in integrate_lsrk54.")

        t_base = t
        for i in range(n_full):
            t_step_start = t_base + i * dt_nominal
            _lsrk54_step_inplace(rhs=rhs, t=t_step_start, q=q, dt=dt_nominal, res=res)
            t = t_base + (i + 1) * dt_nominal
            # print(f"\rcurrent time = {t:.8f}", end="", flush=True)
            nsteps += 1

            if post_step_transform is not None:
                q = np.asarray(post_step_transform(t, q), dtype=float)
                if q.shape != q_shape:
                    raise ValueError("post_step_transform(t, q) must preserve state shape.")

            if np.max(np.abs(q)) > BLOWUP_BREAK_ABS:
                return q, t, nsteps

        remaining = tf_target - t
        if remaining < -tol:
            raise RuntimeError("Internal stepping overshot tf beyond tolerance.")

        if remaining > tol:
            if nsteps >= max_steps:
                raise RuntimeError("Maximum number of steps exceeded in integrate_lsrk54.")

            dt_step = remaining
            _lsrk54_step_inplace(rhs=rhs, t=t, q=q, dt=dt_step, res=res)
            t = tf_target
            nsteps += 1

            if post_step_transform is not None:
                q = np.asarray(post_step_transform(t, q), dtype=float)
                if q.shape != q_shape:
                    raise ValueError("post_step_transform(t, q) must preserve state shape.")

            if np.max(np.abs(q)) > BLOWUP_BREAK_ABS:
                return q, t, nsteps

        if is_tf_reached(t, tf_target, tol=tol):
            t = tf_target

        return q, t, nsteps

    while not is_tf_reached(t, tf_target, tol=tol):
        if nsteps >= max_steps:
            raise RuntimeError("Maximum number of steps exceeded in integrate_lsrk54.")

        dt_nominal = float(dt_getter(t, q))

        if dt_nominal <= 0.0:
            raise ValueError("Time step must be positive.")

        remaining = tf_target - t
        if remaining <= tol:
            t = tf_target
            break

        dt_step = min(dt_nominal, remaining)
        if dt_step <= 0.0:
            raise RuntimeError("Non-positive dt_step encountered before reaching tf.")

        _lsrk54_step_inplace(rhs=rhs, t=t, q=q, dt=dt_step, res=res)
        t += dt_step
        nsteps += 1

        if post_step_transform is not None:
            q = np.asarray(post_step_transform(t, q), dtype=float)
            if q.shape != q_shape:
                raise ValueError("post_step_transform(t, q) must preserve state shape.")

        if np.max(np.abs(q)) > BLOWUP_BREAK_ABS:
            break

        # protect against roundoff stalling very near tf
        if is_tf_reached(t, tf_target, tol=tol):
            t = tf_target
            break

    return q, t, nsteps