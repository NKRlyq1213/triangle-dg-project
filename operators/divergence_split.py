from __future__ import annotations

import importlib
import numpy as np

try:
    _numba = importlib.import_module("numba")
    njit = _numba.njit
    prange = _numba.prange
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False


def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


# Empirical crossover on this project/hardware is around Table1 N=4,
# K in the 1k+ range. Below this size, numpy/BLAS is often faster.
_NUMBA_SPLIT_MIN_DOF = 20000


def _apply_reference_operator(D: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Apply a reference differentiation matrix D to nodal data u.

    Supported shapes
    ----------------
    - u.shape == (Np,)
    - u.shape == (K, Np)
    """
    D = np.asarray(D, dtype=float)
    u = np.asarray(u, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D array.")

    Np = D.shape[0]

    if u.ndim == 1:
        if u.shape[0] != Np:
            raise ValueError("For 1D input, u.shape[0] must equal D.shape[0].")
        return D @ u

    if u.ndim == 2:
        if u.shape[1] != Np:
            raise ValueError("For 2D input, u.shape[1] must equal D.shape[0].")
        return u @ D.T

    raise ValueError("u must have shape (Np,) or (K, Np).")


if _NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _mapped_divergence_split_2d_kernel(
        v: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Dr: np.ndarray,
        Ds: np.ndarray,
        xr: np.ndarray,
        xs: np.ndarray,
        yr: np.ndarray,
        ys: np.ndarray,
        J: np.ndarray,
    ) -> np.ndarray:
        K = v.shape[0]
        Np = v.shape[1]
        out = np.empty((K, Np), dtype=np.float64)

        for k in prange(K):
            alpha = np.empty(Np, dtype=np.float64)
            beta = np.empty(Np, dtype=np.float64)
            alpha_v = np.empty(Np, dtype=np.float64)
            beta_v = np.empty(Np, dtype=np.float64)

            for j in range(Np):
                alpha_j = ys[k, j] * a[k, j] - xs[k, j] * b[k, j]
                beta_j = -yr[k, j] * a[k, j] + xr[k, j] * b[k, j]
                vj = v[k, j]

                alpha[j] = alpha_j
                beta[j] = beta_j
                alpha_v[j] = alpha_j * vj
                beta_v[j] = beta_j * vj

            for i in range(Np):
                vr = 0.0
                vs = 0.0
                ar = 0.0
                bs = 0.0
                Dr_alpha_v = 0.0
                Ds_beta_v = 0.0

                for j in range(Np):
                    d_r = Dr[i, j]
                    d_s = Ds[i, j]

                    vj = v[k, j]
                    vr += d_r * vj
                    vs += d_s * vj

                    ar += d_r * alpha[j]
                    bs += d_s * beta[j]

                    Dr_alpha_v += d_r * alpha_v[j]
                    Ds_beta_v += d_s * beta_v[j]

                split_r = 0.5 * (Dr_alpha_v + alpha[i] * vr + v[k, i] * ar)
                split_s = 0.5 * (Ds_beta_v + beta[i] * vs + v[k, i] * bs)
                out[k, i] = (split_r + split_s) / J[k, i]

        return out


    @njit(cache=True, parallel=True)
    def _mapped_divergence_split_2d_precomputed_kernel(
        v: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        ar: np.ndarray,
        bs: np.ndarray,
        Dr: np.ndarray,
        Ds: np.ndarray,
        J: np.ndarray,
    ) -> np.ndarray:
        K = v.shape[0]
        Np = v.shape[1]
        out = np.empty((K, Np), dtype=np.float64)

        for k in prange(K):
            alpha_v = np.empty(Np, dtype=np.float64)
            beta_v = np.empty(Np, dtype=np.float64)

            for j in range(Np):
                vj = v[k, j]
                alpha_v[j] = alpha[k, j] * vj
                beta_v[j] = beta[k, j] * vj

            for i in range(Np):
                vr = 0.0
                vs = 0.0
                Dr_alpha_v = 0.0
                Ds_beta_v = 0.0

                for j in range(Np):
                    d_r = Dr[i, j]
                    d_s = Ds[i, j]
                    vj = v[k, j]

                    vr += d_r * vj
                    vs += d_s * vj
                    Dr_alpha_v += d_r * alpha_v[j]
                    Ds_beta_v += d_s * beta_v[j]

                split_r = 0.5 * (Dr_alpha_v + alpha[k, i] * vr + v[k, i] * ar[k, i])
                split_s = 0.5 * (Ds_beta_v + beta[k, i] * vs + v[k, i] * bs[k, i])
                out[k, i] = (split_r + split_s) / J[k, i]

        return out
else:
    _mapped_divergence_split_2d_kernel = None
    _mapped_divergence_split_2d_precomputed_kernel = None


def build_mapped_divergence_split_cache_2d(
    a: np.ndarray,
    b: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    xr: np.ndarray,
    xs: np.ndarray,
    yr: np.ndarray,
    ys: np.ndarray,
    J: np.ndarray,
) -> dict:
    """
    Build time-invariant coefficients for mapped_divergence_split_2d.

    Useful when a,b and metric terms are fixed over time, allowing each RHS
    call to skip recomputing alpha/beta and their reference derivatives.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    xr = np.asarray(xr, dtype=float)
    xs = np.asarray(xs, dtype=float)
    yr = np.asarray(yr, dtype=float)
    ys = np.asarray(ys, dtype=float)
    J = np.asarray(J, dtype=float)

    if not (a.shape == b.shape == xr.shape == xs.shape == yr.shape == ys.shape == J.shape):
        raise ValueError("a, b, xr, xs, yr, ys, J must all have the same shape.")

    Dr = np.asarray(Dr, dtype=float)
    Ds = np.asarray(Ds, dtype=float)

    alpha = ys * a - xs * b
    beta = -yr * a + xr * b
    ar = _apply_reference_operator(Dr, alpha)
    bs = _apply_reference_operator(Ds, beta)

    return {
        "alpha": alpha,
        "beta": beta,
        "ar": ar,
        "bs": bs,
        "J": J,
    }


def mapped_divergence_split_2d(
    v: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    xr: np.ndarray,
    xs: np.ndarray,
    yr: np.ndarray,
    ys: np.ndarray,
    J: np.ndarray,
    use_numba: bool | None = None,
    split_cache: dict | None = None,
) -> np.ndarray:
    r"""
    Split-form mapped conservative divergence for

        F = (a v, b v)

    on the reference triangle:

        div(F)
        = 1/J [ D_r (J ∇r · F) + D_s (J ∇s · F) ]

    where
        J ∇r = (ys, -xs),
        J ∇s = (-yr, xr).

    Let
        alpha = ys * a - xs * b
        beta  = -yr * a + xr * b

    then the split form is

        div_h^split(F)
        =
        1/J * [ 1/2 ( D_r(alpha v) + alpha D_r(v) + v D_r(alpha) )
              + 1/2 ( D_s(beta  v) + beta  D_s(v) + v D_s(beta ) ) ]

    Parameters
    ----------
    v, a, b, xr, xs, yr, ys, J : np.ndarray
        Shape (Np,) or (K, Np)
    Dr, Ds : np.ndarray
        Reference differentiation matrices, shape (Np, Np)

    Returns
    -------
    np.ndarray
        Split-form divergence values, same shape as v
    """
    v = np.asarray(v, dtype=float)

    Dr = np.asarray(Dr, dtype=float)
    Ds = np.asarray(Ds, dtype=float)

    if split_cache is None:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        xr = np.asarray(xr, dtype=float)
        xs = np.asarray(xs, dtype=float)
        yr = np.asarray(yr, dtype=float)
        ys = np.asarray(ys, dtype=float)
        J_arr = np.asarray(J, dtype=float)

        if not (v.shape == a.shape == b.shape == xr.shape == xs.shape == yr.shape == ys.shape == J_arr.shape):
            raise ValueError("v, a, b, xr, xs, yr, ys, J must all have the same shape.")

        alpha = ys * a - xs * b
        beta = -yr * a + xr * b
        ar = _apply_reference_operator(Dr, alpha)
        bs = _apply_reference_operator(Ds, beta)
    else:
        alpha = np.asarray(split_cache["alpha"], dtype=float)
        beta = np.asarray(split_cache["beta"], dtype=float)
        ar = np.asarray(split_cache["ar"], dtype=float)
        bs = np.asarray(split_cache["bs"], dtype=float)
        J_arr = np.asarray(split_cache["J"], dtype=float)

        if not (v.shape == alpha.shape == beta.shape == ar.shape == bs.shape == J_arr.shape):
            raise ValueError("v and split_cache arrays must all have the same shape.")

    if (
        _should_use_numba(use_numba)
        and v.ndim == 2
        and (v.shape[0] * v.shape[1] >= _NUMBA_SPLIT_MIN_DOF)
    ):
        if Dr.ndim != 2 or Ds.ndim != 2:
            raise ValueError("Dr and Ds must be 2D matrices.")
        if Dr.shape[0] != Dr.shape[1] or Ds.shape[0] != Ds.shape[1]:
            raise ValueError("Dr and Ds must be square matrices.")
        if Dr.shape != Ds.shape:
            raise ValueError("Dr and Ds must have the same shape.")
        if v.shape[1] != Dr.shape[0]:
            raise ValueError("For 2D input, v.shape[1] must match Dr/Ds size.")

        return _mapped_divergence_split_2d_precomputed_kernel(
            v=np.ascontiguousarray(v, dtype=np.float64),
            alpha=np.ascontiguousarray(alpha, dtype=np.float64),
            beta=np.ascontiguousarray(beta, dtype=np.float64),
            ar=np.ascontiguousarray(ar, dtype=np.float64),
            bs=np.ascontiguousarray(bs, dtype=np.float64),
            Dr=np.ascontiguousarray(Dr, dtype=np.float64),
            Ds=np.ascontiguousarray(Ds, dtype=np.float64),
            J=np.ascontiguousarray(J_arr, dtype=np.float64),
        )

    vr = _apply_reference_operator(Dr, v)
    vs = _apply_reference_operator(Ds, v)

    alpha_v = alpha * v
    beta_v = beta * v

    Dr_alpha_v = _apply_reference_operator(Dr, alpha_v)
    Ds_beta_v = _apply_reference_operator(Ds, beta_v)

    split_r = 0.5 * (Dr_alpha_v + alpha * vr + v * ar)
    split_s = 0.5 * (Ds_beta_v + beta * vs + v * bs)

    return (split_r + split_s) / J_arr


def mapped_divergence_conservative_2d(
    v: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    xr: np.ndarray,
    xs: np.ndarray,
    yr: np.ndarray,
    ys: np.ndarray,
    J: np.ndarray,
) -> np.ndarray:
    r"""
    Conservative mapped divergence for

        F = (a v, b v)

    defined by

        div(F)
        = 1/J [ D_r(alpha v) + D_s(beta v) ]

    where
        alpha = ys * a - xs * b
        beta  = -yr * a + xr * b
    """
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    xr = np.asarray(xr, dtype=float)
    xs = np.asarray(xs, dtype=float)
    yr = np.asarray(yr, dtype=float)
    ys = np.asarray(ys, dtype=float)
    J = np.asarray(J, dtype=float)

    if not (v.shape == a.shape == b.shape == xr.shape == xs.shape == yr.shape == ys.shape == J.shape):
        raise ValueError("v, a, b, xr, xs, yr, ys, J must all have the same shape.")

    alpha = ys * a - xs * b
    beta = -yr * a + xr * b

    Dr_alpha_v = _apply_reference_operator(Dr, alpha * v)
    Ds_beta_v = _apply_reference_operator(Ds, beta * v)

    return (Dr_alpha_v + Ds_beta_v) / J