from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule
from experiments.output_paths import photo_output_dir
from geometry.display_points import build_display_points
from geometry.reference_triangle import reference_triangle_area, reference_triangle_vertices
from operators.reconstruction import evaluate_modal_expansion, fit_modal_coefficients_weighted
from operators.vandermonde2d import vandermonde2d
from problems.analytic_fields import ground_truth_function
from visualization.surface3d import plot_triangle_surface3d


def _load_rule(table_name: str, order: int) -> dict:
    if table_name == "table1":
        return load_table1_rule(order)
    return load_table2_rule(order)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render reconstructed and error fields as 3D surfaces.")
    parser.add_argument("--table", choices=["table1", "table2"], default="table1")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--case", type=str, default="smooth_bump")
    args = parser.parse_args()

    output_dir = photo_output_dir(__file__, "surface3d")

    vertices = reference_triangle_vertices()
    area = reference_triangle_area()

    rule = _load_rule(args.table, args.order)
    rs_nodes = rule["rs"]
    w = rule["ws"]

    u_nodes = ground_truth_function(args.case, rs_nodes[:, 0], rs_nodes[:, 1])
    V = vandermonde2d(args.N, rs_nodes[:, 0], rs_nodes[:, 1])
    coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

    add_edge_points = args.table == "table2"
    edge_n = args.order + 1 if add_edge_points else 5
    rs_eval = build_display_points(
        table_name=args.table,
        rule=rule,
        add_vertices=True,
        add_edge_points=add_edge_points,
        edge_n=edge_n,
    )
    V_eval = vandermonde2d(args.N, rs_eval[:, 0], rs_eval[:, 1])

    u_eval = evaluate_modal_expansion(V_eval, coeffs)
    u_exact = ground_truth_function(args.case, rs_eval[:, 0], rs_eval[:, 1])
    err = u_eval - u_exact

    table_title = "Table 1" if args.table == "table1" else "Table 2"

    fig1, _ = plot_triangle_surface3d(
        rs_eval=rs_eval,
        z_eval=u_eval,
        vertices=vertices,
        nodes=rs_nodes,
        title=f"3D reconstructed field: {table_title} order {args.order}, N={args.N}, case={args.case}",
        zlabel="u_recon",
    )
    fig1.savefig(
        output_dir / f"surface3d_field_{args.table}_order{args.order}_N{args.N}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig1)

    fig2, _ = plot_triangle_surface3d(
        rs_eval=rs_eval,
        z_eval=err,
        vertices=vertices,
        nodes=rs_nodes,
        title=f"3D error field: {table_title} order {args.order}, N={args.N}, case={args.case}",
        zlabel="u_recon - u_exact",
    )
    fig2.savefig(
        output_dir / f"surface3d_error_{args.table}_order{args.order}_N{args.N}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig2)

    print("max abs error =", np.max(np.abs(err)))
    print("l2-like rms error =", np.sqrt(np.mean(err ** 2)))


if __name__ == "__main__":
    main()
