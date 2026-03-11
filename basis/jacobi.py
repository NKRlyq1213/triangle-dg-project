from __future__ import annotations

import numpy as np
from scipy.special import eval_jacobi, gammaln


def _validate_alpha_beta(alpha: float, beta: float) -> None:
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("alpha and beta must both be > -1.")


def _jacobi_norm_sq(n: int, alpha: float, beta: float) -> float:
    """
    Squared L2 norm of the classical Jacobi polynomial P_n^{(alpha,beta)}
    on [-1, 1] with weight (1-x)^alpha (1+x)^beta.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    _validate_alpha_beta(alpha, beta)

    log_num = (
        (alpha + beta + 1.0) * np.log(2.0)
        + gammaln(n + alpha + 1.0)
        + gammaln(n + beta + 1.0)
    )
    log_den = (
        np.log(2.0 * n + alpha + beta + 1.0)
        + gammaln(n + 1.0)
        + gammaln(n + alpha + beta + 1.0)
    )
    return float(np.exp(log_num - log_den))


def jacobi_classical(n: int, alpha: float, beta: float, x) -> np.ndarray:
    """
    Evaluate the classical Jacobi polynomial P_n^{(alpha,beta)}(x).

    Parameters
    ----------
    n : int
        Polynomial degree.
    alpha, beta : float
        Jacobi parameters, must satisfy alpha > -1 and beta > -1.
    x : array_like
        Evaluation points.

    Returns
    -------
    np.ndarray
        Values of the classical Jacobi polynomial.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    _validate_alpha_beta(alpha, beta)
    x = np.asarray(x, dtype=float)
    return np.asarray(eval_jacobi(n, alpha, beta, x), dtype=float)


def jacobi_orthonormal(n: int, alpha: float, beta: float, x) -> np.ndarray:
    """
    Evaluate the orthonormal Jacobi polynomial on [-1, 1].

    This is the classical Jacobi polynomial divided by its L2 norm.
    """
    p = jacobi_classical(n, alpha, beta, x)
    h_n = _jacobi_norm_sq(n, alpha, beta)
    return p / np.sqrt(h_n)


def grad_jacobi_classical(n: int, alpha: float, beta: float, x) -> np.ndarray:
    """
    Derivative of the classical Jacobi polynomial P_n^{(alpha,beta)}(x).

    Uses:
        d/dx P_n^{(a,b)}(x)
        = 0.5 * (n + a + b + 1) * P_{n-1}^{(a+1,b+1)}(x)
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    _validate_alpha_beta(alpha, beta)

    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.zeros_like(x, dtype=float)

    factor = 0.5 * (n + alpha + beta + 1.0)
    return factor * jacobi_classical(n - 1, alpha + 1.0, beta + 1.0, x)


def grad_jacobi_orthonormal(n: int, alpha: float, beta: float, x) -> np.ndarray:
    """
    Derivative of the orthonormal Jacobi polynomial.

    Since the normalization factor is constant in x:
        d/dx [P_n / sqrt(h_n)] = P_n' / sqrt(h_n)
    """
    dp = grad_jacobi_classical(n, alpha, beta, x)
    h_n = _jacobi_norm_sq(n, alpha, beta)
    return dp / np.sqrt(h_n)
