from __future__ import annotations

import numpy as np

from basis.jacobi import jacobi_classical, jacobi_orthonormal
from basis.indexing import mode_indices_2d, num_modes_2d
from basis.simplex2d import simplex2d_mode, grad_simplex2d_mode
from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d


def main() -> None:
    x = np.linspace(-1.0, 1.0, 5)
    p2 = jacobi_classical(2, 0.0, 0.0, x)
    p2o = jacobi_orthonormal(2, 0.0, 0.0, x)

    print("P2 classical:", p2)
    print("P2 orthonormal:", p2o)
    print("num_modes_2d(3) =", num_modes_2d(3))
    print("mode_indices_2d(2) =", mode_indices_2d(2))

    # Six valid points in the reference triangle (r,s)
    rs = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [-0.5, -0.5],
    ], dtype=float)

    r = rs[:, 0]
    s = rs[:, 1]

    val = simplex2d_mode(1, 1, r, s)
    dr, ds = grad_simplex2d_mode(1, 1, r, s)
    V = vandermonde2d(2, r, s)
    Vr, Vs = grad_vandermonde2d(2, r, s)

    print("simplex2d_mode(1,1) shape =", val.shape)
    print("grad_simplex2d_mode(1,1) shapes =", dr.shape, ds.shape)
    print("V shape =", V.shape)
    print("Vr, Vs shapes =", Vr.shape, Vs.shape)
    print("rank(V) =", np.linalg.matrix_rank(V))
    print("cond(V) =", np.linalg.cond(V))


if __name__ == "__main__":
    main()
