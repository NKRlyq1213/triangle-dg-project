from __future__ import annotations

import numpy as np


def edge_vertices(edge_id: int, vertices: np.ndarray) -> np.ndarray:
    """
    Return the two endpoint vertices of the specified edge.

    Edge convention:
        edge 1: v2 -> v3
        edge 2: v3 -> v1
        edge 3: v1 -> v2
    """
    if edge_id == 1:
        idx = [1, 2]
    elif edge_id == 2:
        idx = [2, 0]
    elif edge_id == 3:
        idx = [0, 1]
    else:
        raise ValueError("edge_id must be 1, 2, or 3.")

    return vertices[idx]


def edge_parameterization(
    edge_id: int,
    t: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Map 1D parameter t in [0, 1] to points on the specified triangle edge.
    """
    t = np.asarray(t, dtype=float)
    ev = edge_vertices(edge_id, vertices)
    p0, p1 = ev
    return (1.0 - t)[:, None] * p0 + t[:, None] * p1


def edge_length(edge_id: int, vertices: np.ndarray) -> float:
    """
    Return the edge length.
    """
    ev = edge_vertices(edge_id, vertices)
    return float(np.linalg.norm(ev[1] - ev[0]))