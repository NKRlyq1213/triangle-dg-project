from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_table1_rule, load_table2_rule
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
    parser = argparse.ArgumentParser(description="Plot reconstruction error field for Table 1/Table 2.")
    parser.add_argument("--table", choices=["table1", "table2"], default="table1")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--case", type=str, default="smooth_bump")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[2] / "photo"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    u_exact = ground_truth_function(args.case, rs_eval[:, 0], rs_eval[:, 1])
    err = u_eval - u_exact

    table_title = "Table 1" if args.table == "table1" else "Table 2"

    fig, _ = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err,
        vertices=vertices,
        nodes=rs_eval,
        title=f"Error field: {table_title} order {args.order}, N={args.N}, case={args.case}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(
        output_dir / f"error_field_{args.table}_order{args.order}_N{args.N}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, _ = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=u_eval,
        vertices=vertices,
        nodes=rs_eval,
        title=f"Evaluated field: {table_title} order {args.order}, N={args.N}, case={args.case}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(
        output_dir / f"evaluated_field_{args.table}_order{args.order}_N{args.N}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, _ = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=u_exact,
        vertices=vertices,
        nodes=rs_eval,
        title=f"Exact field: {table_title} order {args.order}, N={args.N}, case={args.case}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(
        output_dir / f"exact_field_{args.table}_order{args.order}_N{args.N}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("--- dense grid error ---")
    print("max abs error =", np.max(np.abs(err)))
    print("l2-like rms error =", np.sqrt(np.mean(err ** 2)))

    u_reconstruct_nodes = evaluate_modal_expansion(V, coeffs)
    err_nodes = u_reconstruct_nodes - u_nodes
    print("--- table node error ---")
    print("node max abs error =", np.max(np.abs(err_nodes)))
    print("node rms error =", np.sqrt(np.mean(err_nodes ** 2)))
    weighted_orthogonality = V.T @ (w * err_nodes)
    print("||V^T W err_nodes||_inf =", np.max(np.abs(weighted_orthogonality)))


if __name__ == "__main__":
    main()