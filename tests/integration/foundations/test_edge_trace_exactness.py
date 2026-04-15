from __future__ import annotations

import numpy as np

from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_area
from operators.vandermonde2d import vandermonde2d
from operators.boundary import evaluate_on_edge


def poly_u(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 1.0 + 0.5 * r - 0.25 * s + 0.75 * r * s + 0.2 * r**2


def test_edge_trace_exactness_table2_order4():
    area = reference_triangle_area()
    rule = load_table2_rule(4)

    r = rule["rs"][:, 0]
    s = rule["rs"][:, 1]
    w = rule["ws"]

    N = 4
    n_edge = 5

    V_vol = vandermonde2d(N, r, s)
    u_vol = poly_u(r, s)

    for edge_id in [1, 2, 3]:
        u_edge, rs_edge, _ = evaluate_on_edge(
            u_vol=u_vol,
            V_vol=V_vol,
            weights=w,
            N=N,
            edge_id=edge_id,
            n_edge=n_edge,
            area=area,
        )

        u_exact = poly_u(rs_edge[:, 0], rs_edge[:, 1])
        err = np.max(np.abs(u_edge - u_exact))
        assert err < 1e-12, f"edge {edge_id} trace error too large: {err}"
