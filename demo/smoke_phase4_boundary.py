from __future__ import annotations

import numpy as np

from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_area
from operators.vandermonde2d import vandermonde2d
from operators.boundary import evaluate_on_edge, evaluate_on_all_edges


def poly_u(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Degree-2 test polynomial.
    """
    return 1.0 + 0.5 * r - 0.25 * s + 0.75 * r * s + 0.2 * r**2


def main() -> None:
    area = reference_triangle_area()
    rule = load_table2_rule(4)

    r = rule["rs"][:, 0]
    s = rule["rs"][:, 1]
    w = rule["ws"]

    N = 4
    n_edge = 5

    V_vol = vandermonde2d(N, r, s)
    u_vol = poly_u(r, s)

    print("=" * 60)
    print("Single-edge test")
    u_edge, rs_edge, E_edge = evaluate_on_edge(
        u_vol=u_vol,
        V_vol=V_vol,
        weights=w,
        N=N,
        edge_id=2,
        n_edge=n_edge,
        area=area,
    )

    u_exact = poly_u(rs_edge[:, 0], rs_edge[:, 1])
    print("edge_id = 2")
    print("E_edge shape =", E_edge.shape)
    print("max edge trace error =", np.max(np.abs(u_edge - u_exact)))

    print("=" * 60)
    print("All-edge test")
    out = evaluate_on_all_edges(
        u_vol=u_vol,
        V_vol=V_vol,
        weights=w,
        N=N,
        n_edge=n_edge,
        area=area,
    )

    for edge_id, data in out.items():
        rs_e = data["rs_edge"]
        u_e = data["u_edge"]
        u_ex = poly_u(rs_e[:, 0], rs_e[:, 1])
        err = np.max(np.abs(u_e - u_ex))
        print(f"edge {edge_id}: shape={data['E_edge'].shape}, max error={err}")


if __name__ == "__main__":
    main()
