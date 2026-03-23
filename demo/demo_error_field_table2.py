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
    resolution = 60
    case_name = "smooth_bump"

    rule = load_table2_rule(table_order)

    xy_nodes = rule["xy"]
    rs_nodes = rule["rs"]
    w = rule["ws"]

    u_nodes = ground_truth_function(case_name, xy_nodes[:, 0], xy_nodes[:, 1])

    V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
    coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)
    print(V @ coeffs)
    print(u_nodes)
    
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
    u_exact = ground_truth_function(case_name, xy_eval[:, 0], xy_eval[:, 1])

    err = u_eval - u_exact

    fig, ax = plot_triangle_field(
        xy_eval=xy_eval,
        u_eval=err,
        vertices=vertices,
        nodes=xy_eval,
        title=f"Error field: Table 2 order {table_order}, N={N}, case={case_name}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(output_dir / "error_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_triangle_field(
        xy_eval=xy_eval,
        u_eval=u_eval,
        vertices=vertices,
        nodes=xy_eval,
        title=f"Evaluated field: Table 2 order {table_order}, N={N}, case={case_name}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(output_dir / "evaluated_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_triangle_field(
        xy_eval=xy_eval,
        u_eval=u_exact,
        vertices=vertices,
        nodes=xy_eval,
        title=f"Exact field: Table 2 order {table_order}, N={N}, case={case_name}",
        levels=25,
        show_nodes=True,
    )
    fig.savefig(output_dir / "exact_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("--- dense grid error ---")
    print("max abs error =", np.max(np.abs(err)))
    print("l2-like rms error =", np.sqrt(np.mean(err**2)))

    # error at table nodes only
    u_reconstruct_nodes = evaluate_modal_expansion(V, coeffs)
    err_nodes = u_reconstruct_nodes - u_nodes
    print("--- table node error ---")
    print("node max abs error =", np.max(np.abs(err_nodes)))
    print("node rms error =", np.sqrt(np.mean(err_nodes**2)))
    weighted_orthogonality = V.T @ (w * err_nodes)
    print("||V^T W err_nodes||_inf =", np.max(np.abs(weighted_orthogonality)))

if __name__ == "__main__":
    main()
