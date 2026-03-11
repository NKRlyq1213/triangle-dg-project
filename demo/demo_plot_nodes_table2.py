from __future__ import annotations

import matplotlib.pyplot as plt

from data.table2_rules import load_table2_rule
from geometry.reference_triangle import reference_triangle_vertices
from visualization.node_plot import plot_nodes


def main() -> None:
    vertices = reference_triangle_vertices()

    for order in [1, 2, 3, 4]:
        rule = load_table2_rule(order)
        fig, ax = plot_nodes(
            rule=rule,
            vertices=vertices,
            annotate=False,
            title=f"Table 2 nodes, order {order}",
        )
        fig.savefig(f"table2_nodes_order_{order}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
