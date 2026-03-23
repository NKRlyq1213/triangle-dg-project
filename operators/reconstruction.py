from __future__ import annotations

import numpy as np

from operators.mass import mass_matrix_from_quadrature


def fit_modal_coefficients_square(
    u_nodes: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """
    Solve modal coefficients from nodal values using a square Vandermonde:

        V a = u

    Parameters
    ----------
    u_nodes : np.ndarray
        Nodal values of shape (n_points,).
    V : np.ndarray
        Square Vandermonde matrix of shape (n_points, n_modes).

    Returns
    -------
    np.ndarray
        Modal coefficients of shape (n_modes,).
    """
    u_nodes = np.asarray(u_nodes, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)

    if V.ndim != 2:
        raise ValueError("V must be 2D.")
    if V.shape[0] != V.shape[1]:
        raise ValueError("Square reconstruction requires V to be square.")
    if V.shape[0] != u_nodes.size:
        raise ValueError("Size mismatch between V and u_nodes.")

    return np.linalg.solve(V, u_nodes)


def fit_modal_coefficients_weighted(
    u_nodes: np.ndarray,
    V: np.ndarray,
    weights: np.ndarray,
    area: float = 1.0,
) -> np.ndarray:
    """
    Solve modal coefficients from oversampled nodal values using weighted projection:

        a = (area * V^T W V)^{-1} (area * V^T W u)

    The area factor cancels algebraically, but is kept here for clarity and
    consistency with the mass-matrix definition.

    Parameters
    ----------
    u_nodes : np.ndarray
        Nodal values of shape (n_points,).
    V : np.ndarray
        Vandermonde matrix of shape (n_points, n_modes).
    weights : np.ndarray
        Quadrature weights of shape (n_points,).
    area : float
        Geometric area factor.

    Returns
    -------
    np.ndarray
        Modal coefficients of shape (n_modes,).
    """
    u_nodes = np.asarray(u_nodes, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if V.ndim != 2:
        raise ValueError("V must be 2D.")
    if V.shape[0] != u_nodes.size:
        raise ValueError("Size mismatch between V and u_nodes.")
    if V.shape[0] != weights.size:
        raise ValueError("Size mismatch between V and weights.")

    M = mass_matrix_from_quadrature(V, weights, area=area)

    print(M.shape)
    rhs = area * (V.T @ (weights * u_nodes)) # = M a
    weights = np.diag(weights)
    print(rhs.shape)
    print(V.shape)
    print(area * V @ np.linalg.inv(M) @ (V.T @ (weights)))
    print(np.linalg.matrix_rank(area * V @ np.linalg.solve(M, V.T @ weights)))
    return np.linalg.solve(M, rhs)


def evaluate_modal_expansion(
    V_eval: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a modal expansion at target points:

        u_eval = V_eval a

    Parameters
    ----------
    V_eval : np.ndarray
        Evaluation Vandermonde matrix of shape (n_eval, n_modes).
    coeffs : np.ndarray
        Modal coefficients of shape (n_modes,).

    Returns
    -------
    np.ndarray
        Evaluated values of shape (n_eval,).
    """
    V_eval = np.asarray(V_eval, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)

    if V_eval.ndim != 2:
        raise ValueError("V_eval must be 2D.")
    if V_eval.shape[1] != coeffs.size:
        raise ValueError("V_eval.shape[1] must match len(coeffs).")

    return V_eval @ coeffs


class PolynomialReconstruction:
    """
    Lightweight helper for modal reconstruction and evaluation.
    """

    def __init__(
        self,
        V: np.ndarray,
        weights: np.ndarray | None = None,
        area: float = 1.0,
        mode: str = "weighted_projection",
    ) -> None:
        self.V = np.asarray(V, dtype=float)
        self.weights = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
        self.area = float(area)
        self.mode = mode

        if self.V.ndim != 2:
            raise ValueError("V must be 2D.")

        if self.weights is not None and self.weights.size != self.V.shape[0]:
            raise ValueError("weights size must match the number of rows in V.")

    def fit(self, u_nodes: np.ndarray) -> np.ndarray:
        u_nodes = np.asarray(u_nodes, dtype=float).reshape(-1)

        if self.mode == "square_nodal":
            return fit_modal_coefficients_square(u_nodes, self.V)

        if self.mode == "weighted_projection":
            if self.weights is None:
                raise ValueError("weights are required for weighted_projection.")
            return fit_modal_coefficients_weighted(
                u_nodes=u_nodes,
                V=self.V,
                weights=self.weights,
                area=self.area,
            )

        raise ValueError("mode must be 'square_nodal' or 'weighted_projection'.")

    def evaluate(self, V_eval: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        return evaluate_modal_expansion(V_eval, coeffs)
