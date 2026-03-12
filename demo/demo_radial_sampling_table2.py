from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_vertices, reference_triangle_area
from geometry.sampling import centroid_star_sampling
from operators.vandermonde2d import vandermonde2d
from operators.reconstruction import fit_modal_coefficients_weighted, evaluate_modal_expansion
from problems.analytic_fields import ground_truth_function
from visualization.radial_plot import plot_radial_field, plot_radial_profile


def xy_to_rs(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    rs = np.empty_like(xy)
    rs[:, 0] = 2.0 * xy[:, 0] - 1.0
    rs[:, 1] = 2.0 * xy[:, 1] - 1.0
    return rs


def main() -> None:
    output_dir = Path(r"C:\Users\user\Desktop\triangle_dg_project\photo")
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = reference_triangle_vertices()
    area = reference_triangle_area()

    table_order = 4
    N = 4
    case_name = "poly2"

    rule = load_table2_rule(table_order)

    xy_nodes = rule["xy"]
    rs_nodes = rule["rs"]
    w = rule["ws"]

    u_nodes = ground_truth_function(case_name, xy_nodes[:, 0], xy_nodes[:, 1])

    V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
    coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

    xy_eval, theta_ids, rho = centroid_star_sampling(
        vertices=vertices,
        n_theta=120,
        n_r=50,
        include_endpoint=True,
    )
    rs_eval = xy_to_rs(xy_eval)
    V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])
    u_eval = evaluate_modal_expansion(V_eval, coeffs)

    fig1, ax1 = plot_radial_field(
        xy_eval=xy_eval,
        u_eval=u_eval,
        vertices=vertices,
        nodes=xy_nodes,
        title=f"Radial field: Table 2 order {table_order}, N={N}, case={case_name}",
    )
    fig1.savefig(output_dir / "radial_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plot_radial_profile(
        rho=rho,
        u=u_eval,
        theta_ids=theta_ids,
        n_curves_to_show=8,
        title=f"Radial profiles: Table 2 order {table_order}, N={N}, case={case_name}",
    )
    fig2.savefig(output_dir / "radial_profile_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    main()
