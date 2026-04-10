from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from data import load_table1_rule, load_table2_rule
from geometry.reference_triangle import reference_triangle_vertices
from visualization.node_plot import plot_nodes


def _load_rule(table_name: str, order: int) -> dict:
    if table_name == "table1":
        return load_table1_rule(order)
    return load_table2_rule(order)


def _iter_tables(table: str) -> list[str]:
    if table == "both":
        return ["table1", "table2"]
    return [table]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reference nodes for Table 1/Table 2 rules.")
    parser.add_argument(
        "--table",
        choices=["table1", "table2", "both"],
        default="both",
        help="Select which rule table to visualize.",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Quadrature orders to plot.",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[2] / "photo"
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = reference_triangle_vertices()

    for table_name in _iter_tables(args.table):
        for order in args.orders:
            rule = _load_rule(table_name, order)
            table_title = "Table 1" if table_name == "table1" else "Table 2"
            fig, _ = plot_nodes(
                rule=rule,
                vertices=vertices,
                annotate=False,
                title=f"{table_title} nodes, order {order}",
            )
            fig.savefig(
                output_dir / f"{table_name}_nodes_order_{order}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


if __name__ == "__main__":
    main()