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
from visualization.surface3d import plot_triangle_surface3d

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
    rs_nodes = rule["rs"]
    w = rule["ws"]

    # nodal values from analytic field
    u_nodes = ground_truth_function(case_name, rs_nodes[:, 0], rs_nodes[:, 1])

    # weighted reconstruction on volume nodes
    V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
    coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

    # dense evaluation points
    from geometry.display_points import build_display_points

    rs_eval = build_display_points(
        table_name="table2",
        rule=rule,
        add_vertices=True,
        add_edge_points=True,
        edge_n=table_order + 1
    )
    V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])

    # reconstructed field and exact field
    u_eval = evaluate_modal_expansion(V_eval, coeffs)
    u_exact = ground_truth_function(case_name, rs_eval[:, 0], rs_eval[:, 1])
    err = u_eval - u_exact

    # 3D reconstructed field
    fig1, ax1 = plot_triangle_surface3d(
        rs_eval=rs_eval,
        z_eval=u_eval,
        vertices=vertices,
        nodes=rs_eval,
        title=f"3D reconstructed field: Table 2 order {table_order}, N={N}, case={case_name}",
        zlabel="u_recon",
    )
    fig1.savefig(output_dir / "surface3d_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 3D error field
    fig2, ax2 = plot_triangle_surface3d(
        rs_eval=rs_eval,
        z_eval=err,
        vertices=vertices,
        nodes=rs_eval,
        title=f"3D error field: Table 2 order {table_order}, N={N}, case={case_name}",
        zlabel="u_recon - u_exact",
    )
    fig2.savefig(output_dir / "surface3d_error_table2_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print("max abs error =", np.max(np.abs(err)))
    print("l2-like rms error =", np.sqrt(np.mean(err**2)))


if __name__ == "__main__":
    main()