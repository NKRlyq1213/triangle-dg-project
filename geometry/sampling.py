from __future__ import annotations

import numpy as np

from .reference_triangle import reference_triangle_centroid
from .barycentric import barycentric_to_cartesian


def dense_barycentric_lattice(
    vertices: np.ndarray,
    resolution: int,
    boundary_mode: str = "all",
) -> np.ndarray:
    """
    Generate dense sampling points inside a triangle using a barycentric lattice.

    Parameters
    ----------
    vertices : np.ndarray
        Triangle vertices of shape (3, 2).
    resolution : int
        Barycentric lattice resolution.
    boundary_mode : str
        One of:
        - "all": include vertices and edges
        - "no_vertices": exclude only the 3 vertices
        - "interior_only": keep only strictly interior points

    Returns
    -------
    np.ndarray
        Cartesian sampling points of shape (n_points, 2).
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")
    if resolution < 1:
        raise ValueError("resolution must be >= 1")
    if boundary_mode not in {"all", "no_vertices", "interior_only", "no_top_vertex"}:
        raise ValueError("boundary_mode must be 'all', 'no_vertices', 'interior_only', or 'no_top_vertex'.")

    bary_points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j

            if boundary_mode == "all":
                keep = True

            elif boundary_mode == "no_vertices":
                # remove only the three corners:
                # (resolution,0,0), (0,resolution,0), (0,0,resolution)
                is_vertex = (
                    (i == resolution and j == 0 and k == 0) or
                    (i == 0 and j == resolution and k == 0) or
                    (i == 0 and j == 0 and k == resolution)
                )
                keep = not is_vertex
            elif boundary_mode == "no_top_vertex":
                # remove only the reference-triangle vertex v3 = (-1, 1),
                # which corresponds to barycentric coordinate (0, 0, 1)
                is_top_vertex = (i == 0 and j == 0 and k == resolution)
                keep = not is_top_vertex
            else:  # "interior_only"
                keep = (i > 0 and j > 0 and k > 0)

            if keep:
                bary = np.array([i, j, k], dtype=float) / resolution
                bary_points.append(bary)

    if not bary_points:
        raise ValueError("No points generated for this resolution/boundary_mode.")

    bary_points = np.array(bary_points, dtype=float)
    return barycentric_to_cartesian(bary_points, vertices)

def _ray_segment_intersection_distance(
    c: np.ndarray,
    d: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    tol: float = 1e-14,
) -> float | None:
    """
    Solve for rho, tau in:
        c + rho * d = p0 + tau * (p1 - p0)

    Return rho if the intersection lies on the segment and in the positive
    ray direction. Otherwise return None.
    """
    e = p1 - p0
    A = np.column_stack([d, -e])
    detA = np.linalg.det(A)

    if abs(detA) < tol:
        return None

    rhs = p0 - c
    rho, tau = np.linalg.solve(A, rhs)

    if rho <= tol:
        return None
    if tau < -tol or tau > 1.0 + tol:
        return None

    return float(rho)


def centroid_star_sampling(
    vertices: np.ndarray,
    n_theta: int,
    n_r: int,
    include_endpoint: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate centroid-centered star sampling points that fill the whole triangle.

    Parameters
    ----------
    vertices : np.ndarray
        Triangle vertices of shape (3, 2).
    n_theta : int
        Number of angular rays.
    n_r : int
        Number of radial samples on each ray.
    include_endpoint : bool
        Whether to include the boundary point on each ray.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (rs, theta_ids, radial_coordinates)
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")
    if n_theta < 3 or n_r < 2:
        raise ValueError("n_theta >= 3 and n_r >= 2 are required.")

    c = np.mean(vertices, axis=0)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    pts = []
    theta_ids = []
    rhos = []

    closed = np.vstack([vertices, vertices[0]])

    for k, theta in enumerate(thetas):
        d = np.array([np.cos(theta), np.sin(theta)], dtype=float)

        candidates = []
        for j in range(3):
            p0 = closed[j]
            p1 = closed[j + 1]
            rho = _ray_segment_intersection_distance(c, d, p0, p1)
            if rho is not None:
                candidates.append(rho)

        if not candidates:
            raise RuntimeError(f"No positive ray-edge intersection found for theta index {k}.")

        rmax = min(candidates)
        radii = np.linspace(0.0, rmax, n_r, endpoint=include_endpoint)

        for r in radii:
            pts.append(c + r * d)
            theta_ids.append(k)
            rho = 0.0 if rmax == 0.0 else r / rmax
            rhos.append(rho)

    return (
        np.array(pts, dtype=float),
        np.array(theta_ids, dtype=int),
        np.array(rhos, dtype=float),
    )