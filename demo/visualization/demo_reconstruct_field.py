from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule
from experiments.output_paths import photo_output_dir
from geometry.display_points import build_display_points
from geometry.reference_triangle import reference_triangle_area, reference_triangle_vertices
from operators.reconstruction import evaluate_modal_expansion, fit_modal_coefficients_weighted
from operators.vandermonde2d import vandermonde2d
from problems.analytic_fields import ground_truth_function
from visualization.triangle_field import plot_triangle_field


def _load_rule(table_name: str, order: int) -> dict:
    if table_name == "table1":
        return load_table1_rule(order)
    return load_table2_rule(order)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct and visualize a scalar field on triangle nodes.")
    parser.add_argument("--table", choices=["table1", "table2"], default="table1")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--case", type=str, default="smooth_bump")
    args = parser.parse_args()

    output_dir = photo_output_dir(__file__, "reconstruct_field")

    vertices = reference_triangle_vertices()
    area = reference_triangle_area()

    rule = _load_rule(args.table, args.order)
    rs_nodes = rule["rs"]
    w = rule["ws"]

    u_nodes = ground_truth_function(args.case, rs_nodes[:, 0], rs_nodes[:, 1])

    V = vandermonde2d(args.N, rs_nodes[:, 0], rs_nodes[:, 1])
    coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

    rs_eval = build_display_points(
        table_name=args.table,
        rule=rule,
        add_vertices=True,
        add_edge_points=True,
        edge_n=args.order + 1,
    )
    V_eval = vandermonde2d(args.N, rs_eval[:, 0], rs_eval[:, 1])
    u_eval = evaluate_modal_expansion(V_eval, coeffs)

    table_title = "Table 1" if args.table == "table1" else "Table 2"
    fig, _ = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=u_eval,
        vertices=vertices,
        nodes=rs_nodes,
        title=f"Reconstructed field: {table_title} order {args.order}, N={args.N}, case={args.case}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(
        output_dir / f"reconstruct_field_{args.table}_order{args.order}_N{args.N}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
