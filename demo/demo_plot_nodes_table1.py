from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path

from data.table1_rules import load_table1_rule
from geometry.reference_triangle import reference_triangle_vertices
from visualization.node_plot import plot_nodes


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "photo"
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = reference_triangle_vertices()

    for order in [1, 2, 3, 4]:
        rule = load_table1_rule(order)
        fig, ax = plot_nodes(
            rule=rule,
            vertices=vertices,
            annotate=False,
            title=f"Table 1 nodes, order {order}",
        )
        fig.savefig(output_dir / f"table1_nodes_order_{order}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
