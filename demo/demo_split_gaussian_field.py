from __future__ import annotations

import numpy as np

from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_area
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.metrics import affine_geometric_factors_from_mesh
from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)
from operators.split_form import split_advective_operator_2d


def build_reference_diff_operators_from_rule(rule: dict, N: int):
    rs = rule["rs"]
    w = rule["ws"]

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])

    if V.shape[0] == V.shape[1]:
        return differentiation_matrices_square(V, Vr, Vs)

    return differentiation_matrices_weighted(
        V, Vr, Vs, w, area=reference_triangle_area()
    )


def coefficient_field(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = 1.0 + 0.3 * x - 0.2 * y
    b = -0.4 + 0.1 * x + 0.25 * y
    return a, b


def gaussian_field(
    x: np.ndarray,
    y: np.ndarray,
    x0: float = 0.35,
    y0: float = 0.55,
    sigma: float = 0.18,
) -> np.ndarray:
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))


def exact_advective_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    x0: float = 0.35,
    y0: float = 0.55,
    sigma: float = 0.18,
) -> np.ndarray:
    v = gaussian_field(x, y, x0=x0, y0=y0, sigma=sigma)
    vx = -((x - x0) / sigma**2) * v
    vy = -((y - y0) / sigma**2) * v
    return a * vx + b * vy


def main():
    # table points
    rule = load_table2_rule(4)
    rs = rule["rs"]

    # reference operators on table points
    N = 4
    Dr, Ds = build_reference_diff_operators_from_rule(rule, N)

    # 8-triangle mesh
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="main")
    X, Y = map_reference_nodes_to_all_elements(rs, VX, VY, EToV)
    g = affine_geometric_factors_from_mesh(VX, VY, EToV, rs)

    # coefficient field and Gaussian initial field
    a, b = coefficient_field(X, Y)
    v = gaussian_field(X, Y)

    L_num = split_advective_operator_2d(
        v, a, b, Dr, Ds,
        g["rx"], g["sx"], g["ry"], g["sy"]
    )
    L_ex = exact_advective_gaussian(X, Y, a, b)

    err = L_num - L_ex

    max_abs = np.max(np.abs(err))
    rms = np.sqrt(np.mean(err**2))
    rel_rms = rms / np.sqrt(np.mean(L_ex**2))

    print("Gaussian split-form diagnostic")
    print("X.shape =", X.shape)
    print("max abs error =", max_abs)
    print("rms error     =", rms)
    print("relative rms  =", rel_rms)


if __name__ == "__main__":
    main()