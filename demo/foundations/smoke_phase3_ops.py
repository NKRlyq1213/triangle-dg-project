from __future__ import annotations

import numpy as np

from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_area
from basis.indexing import num_modes_2d
from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.mass import mass_matrix_from_quadrature
from operators.reconstruction import (
    fit_modal_coefficients_square,
    fit_modal_coefficients_weighted,
    evaluate_modal_expansion,
)
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)


def poly_u(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Test polynomial of total degree 2.
    """
    return 1.0 + 0.5 * r - 0.25 * s + 0.75 * r * s + 0.2 * r**2


def poly_ur(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 0.5 + 0.75 * s + 0.4 * r


def poly_us(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    return -0.25 + 0.75 * r


def run_square_test() -> None:
    rs_square = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [-0.5, -0.5],
    ], dtype=float)

    r_sq = rs_square[:, 0]
    s_sq = rs_square[:, 1]

    V_sq = vandermonde2d(2, r_sq, s_sq)
    Vr_sq, Vs_sq = grad_vandermonde2d(2, r_sq, s_sq)
    u_sq = poly_u(r_sq, s_sq)

    a_sq = fit_modal_coefficients_square(u_sq, V_sq)
    u_sq_rec = evaluate_modal_expansion(V_sq, a_sq)
    Dr_sq, Ds_sq = differentiation_matrices_square(V_sq, Vr_sq, Vs_sq)

    ur_num_sq = Dr_sq @ u_sq
    us_num_sq = Ds_sq @ u_sq

    print("=" * 60)
    print("Square nodal test (N=2, 6 points)")
    print("num_modes_2d(2) =", num_modes_2d(2))
    print("max reconstruction error =", np.max(np.abs(u_sq_rec - u_sq)))
    print("max Dr error =", np.max(np.abs(ur_num_sq - poly_ur(r_sq, s_sq))))
    print("max Ds error =", np.max(np.abs(us_num_sq - poly_us(r_sq, s_sq))))


def run_weighted_test(rule_order: int, N: int) -> None:
    area = reference_triangle_area()
    rule = load_table2_rule(rule_order)

    r = rule["rs"][:, 0]
    s = rule["rs"][:, 1]
    w = rule["ws"]

    V = vandermonde2d(N, r, s)
    Vr, Vs = grad_vandermonde2d(N, r, s)

    u = poly_u(r, s)

    M = mass_matrix_from_quadrature(V, w, area=area)
    a = fit_modal_coefficients_weighted(u, V, w, area=area)
    u_rec = evaluate_modal_expansion(V, a)

    Dr, Ds = differentiation_matrices_weighted(V, Vr, Vs, w, area=area)
    ur_num = Dr @ u
    us_num = Ds @ u

    print("=" * 60)
    print(f"Weighted projection test (Table 2, order {rule_order}, N={N})")
    print("num grid points =", len(w))
    print("num modes =", num_modes_2d(N))
    print("V shape =", V.shape)
    print("rank(V) =", np.linalg.matrix_rank(V))
    print("cond(M) =", np.linalg.cond(M))
    print("max reconstruction error =", np.max(np.abs(u_rec - u)))
    print("max Dr error =", np.max(np.abs(ur_num - poly_ur(r, s))))
    print("max Ds error =", np.max(np.abs(us_num - poly_us(r, s))))


def main() -> None:
    run_square_test()

    for N in [2, 3, 4]:
        run_weighted_test(rule_order=4, N=N)


if __name__ == "__main__":
    main()