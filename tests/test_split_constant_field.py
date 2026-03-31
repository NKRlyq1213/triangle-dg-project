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


def test_split_constant_field_is_zero():
    # table points
    rule = load_table2_rule(4)
    rs = rule["rs"]

    # reference operators built on table points
    N = 4
    Dr, Ds = build_reference_diff_operators_from_rule(rule, N)

    # 8 triangles on [0,1]^2
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="anti")
    X, Y = map_reference_nodes_to_all_elements(rs, VX, VY, EToV)
    g = affine_geometric_factors_from_mesh(VX, VY, EToV, rs)

    # coefficient field and constant initial field
    a, b = coefficient_field(X, Y)
    v = np.ones_like(X)

    Lv = split_advective_operator_2d(
        v, a, b, Dr, Ds,
        g["rx"], g["sx"], g["ry"], g["sy"]
    )

    max_err = np.max(np.abs(Lv))
    print("constant-field split operator max abs =", max_err)

    assert np.allclose(Lv, 0.0, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    test_split_constant_field_is_zero()
    print("test_split_constant_field: passed")