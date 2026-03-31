from __future__ import annotations

import numpy as np

from data.table2_rules import load_table2_rule
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.metrics import affine_geometric_factors_from_mesh, divergence_2d
from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import differentiation_matrices_square


def polynomial_flux_field(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Quadratic vector field on physical coordinates (x, y):

        Fx = x^2 + x y + 2 y
        Fy = x y + y^2 - x

    Exact divergence:
        dFx/dx + dFy/dy = (2x + y) + (x + 2y) = 3x + 3y
    """
    Fx = x**2 + x * y + 2.0 * y
    Fy = x * y + y**2 - x
    return Fx, Fy


def exact_divergence(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 3.0 * x + 3.0 * y


def test_divergence_exactness_table2_order2_square():
    """
    Use Table 2, order 2 as a P2 nodal set (6 points).
    Build exact square differentiation matrices on those table points,
    then verify multi-element divergence on the [0,1]^2 -> 8 triangles mesh.
    """
    # Table nodes on reference triangle
    rule = load_table2_rule(2)
    rs = rule["rs"]

    # Build Dr, Ds on the table points
    N = 2
    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])
    Dr, Ds = differentiation_matrices_square(V, Vr, Vs)

    # Build physical mesh: [0,1]^2 split into 8 triangles
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="anti")

    # Map the same reference table nodes to all physical elements
    X, Y = map_reference_nodes_to_all_elements(rs, VX, VY, EToV)

    # Exact affine geometric factors on all elements, broadcast to all table nodes
    g = affine_geometric_factors_from_mesh(VX, VY, EToV, rs)

    # Evaluate vector field at physical table nodes
    Fx, Fy = polynomial_flux_field(X, Y)

    # Numerical divergence
    div_num = divergence_2d(
        Fx,
        Fy,
        Dr,
        Ds,
        g["rx"],
        g["sx"],
        g["ry"],
        g["sy"],
    )

    # Exact divergence
    div_ex = exact_divergence(X, Y)

    err = div_num - div_ex
    max_err = np.max(np.abs(err))

    print("divergence exactness test")
    print("X.shape =", X.shape)
    print("Y.shape =", Y.shape)
    print("max abs error =", max_err)

    assert np.allclose(div_num, div_ex, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    test_divergence_exactness_table2_order2_square()
    print("test_divergence_exactness: passed")