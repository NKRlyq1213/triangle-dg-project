from __future__ import annotations

import numpy as np

from geometry.reference_triangle import reference_triangle_vertices
from data.edge_rules import all_edge_gl1d_rules


def _unique_rows(arr: np.ndarray, ndigits: int = 14) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    rounded = np.round(arr, ndigits)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx = np.sort(idx)
    return arr[idx]


def build_display_points(
    table_name: str,
    rule: dict,
    add_vertices: bool = True,
    add_edge_points: bool = True,
    edge_n: int = 5,
) -> np.ndarray:
    """
    Build display/evaluation points for plotting in the NEW reference triangle (r, s).

    Rules
    -----
    Table 1:
        display points = table nodes + vertices

    Table 2:
        display points = table nodes + vertices + GL1D edge points
    """
    table_name = table_name.lower().strip()
    pts = [np.asarray(rule["rs"], dtype=float)]

    if add_vertices:
        verts = reference_triangle_vertices()
        pts.append(verts)

    if table_name == "table2" and add_edge_points:
        edge_rules = all_edge_gl1d_rules(edge_n)
        for edge_id in [1, 2, 3]:
            pts.append(edge_rules[edge_id].rs)

    all_pts = np.vstack(pts)
    return _unique_rows(all_pts)