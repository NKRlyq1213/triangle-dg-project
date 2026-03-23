from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path

from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_vertices, reference_triangle_area
from geometry.display_points import build_display_points
from operators.vandermonde2d import vandermonde2d
from operators.reconstruction import fit_modal_coefficients_weighted, evaluate_modal_expansion
from problems.analytic_fields import ground_truth_function
from visualization.triangle_field import plot_triangle_field


def main() -> None:
    output_dir = Path("photo")
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = reference_triangle_vertices()
    area = reference_triangle_area()

    table_order = 4
    N = 4
    case_name = "smooth_bump"

    rule = load_table2_rule(table_order)

    rs_nodes = rule["rs"]
    w = rule["ws"]

    # nodal values from analytic field on the NEW reference triangle
    u_nodes = ground_truth_function(case_name, rs_nodes[:, 0], rs_nodes[:, 1])

    # build volume Vandermonde and fit modal coefficients
    V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
    coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

    # display/evaluation points in (r, s)
    rs_eval = build_display_points(
        table_name="table2",
        rule=rule,
        add_vertices=True,
        add_edge_points=True,
        edge_n=table_order + 1,
    )
    V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])
    u_eval = evaluate_modal_expansion(V_eval, coeffs)

    fig, ax = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=u_eval,
        vertices=vertices,
        nodes=rs_nodes,
        title=f"Reconstructed field: Table 2 order {table_order}, N={N}, case={case_name}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(output_dir / "reconstruct_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()