from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_vertices
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.sampling import dense_barycentric_lattice
from visualization.mesh_nodes import (
    build_reference_triangulation,
    plot_reference_rule_nodes,
    plot_physical_mesh_nodes,
    plot_physical_field_and_nodes,
)


def load_rule(table_name: str, order: int) -> dict:
    table_name = table_name.lower().strip()
    if table_name == "table1":
        return load_table1_rule(order)
    if table_name == "table2":
        return load_table2_rule(order)
    raise ValueError("table_name must be 'table1' or 'table2'.")


def initial_field(
    case_name: str,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    case = case_name.lower().strip()

    if case == "gaussian":
        x0 = 0.35
        y0 = 0.55
        sigma = 0.16
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))

    if case == "constant":
        return np.ones_like(x)

    if case == "trig":
        return np.sin(np.pi * x) * np.cos(np.pi * y)

    raise ValueError("case_name must be one of: 'gaussian', 'constant', 'trig'.")


def main():
    output_dir = Path(__file__).resolve().parents[1] / "photo"
    output_dir.mkdir(parents=True, exist_ok=True)
    # -----------------------------
    # user-configurable parameters
    # -----------------------------
    table_name = "table2"
    order = 4

    nx = 3
    ny = 3
    diagonal = "anti"

    # plot_mode: "nodes", "field_nodes", "both"
    plot_mode = "both"

    # only used when plot_mode includes field_nodes
    field_case = "gaussian"
    eval_resolution = 30
    contour_levels = 25

    # -----------------------------
    # load rule and mesh
    # -----------------------------
    rule = load_rule(table_name, order)
    rs_nodes = rule["rs"]

    VX, VY, EToV = structured_square_tri_mesh(
        nx=nx,
        ny=ny,
        diagonal=diagonal,
    )

    X_nodes, Y_nodes = map_reference_nodes_to_all_elements(rs_nodes, VX, VY, EToV)

    ref_vertices = reference_triangle_vertices()

    # reference nodes figure
    fig_ref, ax_ref = plot_reference_rule_nodes(
        rs_nodes,
        ref_vertices,
        title=f"Reference nodes: {table_name}, order {order}",
        annotate_indices=False,
    )
    fig_ref.savefig(
        output_dir / f"reference_nodes_{table_name}_order{order}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig_ref)

    # physical nodes figure
    if plot_mode in {"nodes", "both"}:
        fig_nodes, ax_nodes = plot_physical_mesh_nodes(
            VX,
            VY,
            EToV,
            X_nodes,
            Y_nodes,
            title=f"Physical mesh nodes: {table_name}, order {order}, nx={nx}, ny={ny}",
            show_element_ids=True,
            show_vertex_ids=False,
        )
        fig_nodes.savefig(
            output_dir / f"physical_nodes_{table_name}_order{order}_nx{nx}_ny{ny}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig_nodes)

    # physical field + nodes figure
    if plot_mode in {"field_nodes", "both"}:
        rs_eval = dense_barycentric_lattice(ref_vertices, resolution=eval_resolution)
        local_triangles = build_reference_triangulation(rs_eval)

        X_eval, Y_eval = map_reference_nodes_to_all_elements(rs_eval, VX, VY, EToV)
        U_eval = initial_field(field_case, X_eval, Y_eval)

        fig_field, ax_field = plot_physical_field_and_nodes(
            VX,
            VY,
            EToV,
            X_eval,
            Y_eval,
            U_eval,
            local_triangles,
            X_nodes=X_nodes,
            Y_nodes=Y_nodes,
            title=(
                f"Initial field + nodes: {field_case}, "
                f"{table_name}, order {order}, nx={nx}, ny={ny}"
            ),
            levels=contour_levels,
            show_nodes=True,
            show_mesh_edges=True,
        )
        fig_field.savefig(
            output_dir / f"initial_field_nodes_{field_case}_{table_name}_order{order}_nx{nx}_ny{ny}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig_field)


if __name__ == "__main__":
    main()