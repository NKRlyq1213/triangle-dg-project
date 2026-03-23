from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import roots_legendre

from geometry.reference_triangle import reference_triangle_vertices
from geometry.edges import edge_parameterization, edge_length


@dataclass
class EdgeRule:
    edge_id: int
    n_points: int
    t01: np.ndarray
    weights: np.ndarray
    rs: np.ndarray
    length: float


def gauss_legendre_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss-Legendre nodes and normalized weights on [0, 1].
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    x, w = roots_legendre(n)   # nodes/weights on [-1,1]
    t01 = 0.5 * (x + 1.0)      # map to [0,1]
    w01 = 0.5 * w              # mapped weights, sum = 1
    return np.asarray(t01, dtype=float), np.asarray(w01, dtype=float)


def edge_gl1d_rule(
    edge_id: int,
    n: int,
    vertices: np.ndarray | None = None,
) -> EdgeRule:
    """
    Build a GL1D rule on one edge of the reference triangle in (r, s).

    Edge convention:
        edge 1: v2 -> v3
        edge 2: v3 -> v1
        edge 3: v1 -> v2
    """
    if vertices is None:
        vertices = reference_triangle_vertices()

    t01, w01 = gauss_legendre_1d(n)
    rs = edge_parameterization(edge_id, t01, vertices)
    L = edge_length(edge_id, vertices)

    return EdgeRule(
        edge_id=edge_id,
        n_points=n,
        t01=t01,
        weights=w01,
        rs=rs,
        length=L,
    )


def all_edge_gl1d_rules(
    n: int,
    vertices: np.ndarray | None = None,
) -> dict[int, EdgeRule]:
    """
    Build GL1D rules on all three edges of the reference triangle.
    """
    if vertices is None:
        vertices = reference_triangle_vertices()

    return {
        1: edge_gl1d_rule(1, n, vertices),
        2: edge_gl1d_rule(2, n, vertices),
        3: edge_gl1d_rule(3, n, vertices),
    }