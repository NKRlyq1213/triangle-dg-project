from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data.table1_rules import load_table1_rule
from geometry.reference_triangle import reference_triangle_vertices, reference_triangle_area
from geometry.sampling import dense_barycentric_lattice
from operators.vandermonde2d import vandermonde2d
from operators.reconstruction import fit_modal_coefficients_weighted, evaluate_modal_expansion
from problems.analytic_fields import ground_truth_function
from visualization.surface3d import plot_triangle_surface3d


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
    resolution = 60
    case_name = "smooth_bump"

    rule = load_table1_rule(table_order)

    xy_nodes = rule["xy"]
    rs_nodes = rule["rs"]
    w = rule["ws"]

    # nodal values from analytic field
    u_nodes = ground_truth_function(case_name, xy_nodes[:, 0], xy_nodes[:, 1])

    # weighted reconstruction on volume nodes
    V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
    coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

    # dense evaluation points
    xy_eval = dense_barycentric_lattice(vertices, resolution=resolution)
    rs_eval = xy_to_rs(xy_eval)
    V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])

    # reconstructed field and exact field
    u_eval = evaluate_modal_expansion(V_eval, coeffs)
    u_exact = ground_truth_function(case_name, xy_eval[:, 0], xy_eval[:, 1])
    err = u_eval - u_exact

    # 3D reconstructed field
    fig1, ax1 = plot_triangle_surface3d(
        xy_eval=xy_eval,
        z_eval=u_eval,
        vertices=vertices,
        nodes=xy_nodes,
        title=f"3D reconstructed field: Table 1 order {table_order}, N={N}, case={case_name}",
        zlabel="u_recon",
    )
    fig1.savefig(output_dir / "surface3d_field_table1_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 3D error field
    fig2, ax2 = plot_triangle_surface3d(
        xy_eval=xy_eval,
        z_eval=err,
        vertices=vertices,
        nodes=xy_nodes,
        title=f"3D error field: Table 1 order {table_order}, N={N}, case={case_name}",
        zlabel="u_recon - u_exact",
    )
    fig2.savefig(output_dir / "surface3d_error_table1_order4_N4.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print("max abs error =", np.max(np.abs(err)))
    print("l2-like rms error =", np.sqrt(np.mean(err**2)))


if __name__ == "__main__":
    main()