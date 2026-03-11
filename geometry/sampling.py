from __future__ import annotations

import numpy as np

from .reference_triangle import reference_triangle_centroid
from .barycentric import barycentric_to_cartesian


def dense_barycentric_lattice(
    vertices: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """
    Generate dense sampling points inside a triangle using a barycentric lattice.
    """
    if resolution < 1:
        raise ValueError("resolution must be >= 1")

    bary_points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            bary = np.array([i, j, k], dtype=float) / resolution
            bary_points.append(bary)

    bary_points = np.array(bary_points, dtype=float)
    return barycentric_to_cartesian(bary_points, vertices)


def centroid_star_sampling(
    vertices: np.ndarray,
    n_theta: int,
    n_r: int,
    include_endpoint: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate centroid-centered star sampling points that fill the whole triangle.

    For each angle theta, compute the maximum admissible radius r_max(theta)
    before hitting the triangle boundary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (xy, theta_ids, radial_coordinates)
        - xy: sampled points, shape (n_points, 2)
        - theta_ids: angle index for each point, shape (n_points,)
        - radial_coordinates: normalized radial coordinate rho in [0,1],
          shape (n_points,)
    """
    if n_theta < 3 or n_r < 2:
        raise ValueError("n_theta >= 3 and n_r >= 2 are required.")

    c = reference_triangle_centroid()
    cx, cy = float(c[0]), float(c[1])

    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    pts = []
    theta_ids = []
    rhos = []

    for k, theta in enumerate(thetas):
        ct = float(np.cos(theta))
        st = float(np.sin(theta))

        candidates = []

        # x = 0
        if ct < -1e-14:
            candidates.append((0.0 - cx) / ct)

        # y = 0
        if st < -1e-14:
            candidates.append((0.0 - cy) / st)

        # x + y = 1
        denom = ct + st
        if denom > 1e-14:
            candidates.append((1.0 - cx - cy) / denom)

        rmax = min(r for r in candidates if r > 0.0)

        rs = np.linspace(0.0, rmax, n_r, endpoint=include_endpoint)
        for j, r in enumerate(rs):
            pts.append([cx + r * ct, cy + r * st])
            theta_ids.append(k)
            rho = 0.0 if rmax == 0.0 else r / rmax
            rhos.append(rho)

    return (
        np.array(pts, dtype=float),
        np.array(theta_ids, dtype=int),
        np.array(rhos, dtype=float),
    )
