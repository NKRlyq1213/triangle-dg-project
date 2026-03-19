from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_vertices, reference_triangle_area
from geometry.sampling import dense_barycentric_lattice
from operators.vandermonde2d import vandermonde2d
from operators.reconstruction import fit_modal_coefficients_weighted, evaluate_modal_expansion
from problems.analytic_fields import ground_truth_function
from visualization.triangle_field import plot_triangle_field


def xy_to_rs(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    rs = np.empty_like(xy)
    rs[:, 0] = 2.0 * xy[:, 0] - 1.0
    rs[:, 1] = 2.0 * xy[:, 1] - 1.0
    return rs


def main() -> None:
    output_dir = Path(r"C:\Users\user\Desktop\triangle-dg-project\photo")
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = reference_triangle_vertices()
    area = reference_triangle_area()

    table_order = 4
    N = 4
    resolution = 50
    case_name = "smooth_bump"

    rule = load_table2_rule(table_order)

    xy_nodes = rule["xy"]
    rs_nodes = rule["rs"]
    w = rule["ws"]

    # nodal values from analytic field
    u_nodes = ground_truth_function(case_name, xy_nodes[:, 0], xy_nodes[:, 1])

    # build volume Vandermonde and fit modal coefficients
    V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
    coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

    # dense sampling points inside the triangle
    from geometry.display_points import build_display_points
    xy_eval = build_display_points(
        table_name="table2",
        rule=rule,
        add_vertices=True,
        add_edge_points=True,
        edge_n=table_order + 1
    )
    rs_eval = xy_to_rs(xy_eval)
    V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])
    u_eval = evaluate_modal_expansion(V_eval, coeffs)

    fig, ax = plot_triangle_field(
        xy_eval=xy_eval,
        u_eval=u_eval,
        vertices=vertices,
        nodes=xy_nodes,
        title=f"Reconstructed field: Table 2 order {table_order}, N={N}, case={case_name}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(output_dir / "reconstruct_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
