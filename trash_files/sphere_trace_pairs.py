from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from data.edge_rules import edge_gl1d_rule
from geometry.sphere_mapping import local_xy_to_sphere_xyz


@dataclass(frozen=True)
class SphereTracePair:
    """
    One shared-edge pair between two sphere patches.

    orientation:
        "same"      : node i on minus edge matches node i on plus edge
        "reversed"  : node i on minus edge matches node (N-1-i) on plus edge
    """

    patch_minus: int
    edge_minus: int
    patch_plus: int
    edge_plus: int
    orientation: str


def reference_rs_to_local_xy(rs: np.ndarray) -> np.ndarray:
    """
    Convert repo reference triangle coordinates (r,s) to local triangle (x,y).

    Repo reference triangle:
        v1 = (-1,-1)
        v2 = ( 1,-1)
        v3 = (-1, 1)

    Local triangle:
        x = (r+1)/2
        y = (s+1)/2
    """
    rs = np.asarray(rs, dtype=float)
    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs must have shape (N, 2).")

    x = 0.5 * (rs[:, 0] + 1.0)
    y = 0.5 * (rs[:, 1] + 1.0)

    return np.column_stack([x, y])


def edge_rule_local_xy(edge_id: int, n: int) -> np.ndarray:
    """
    Return GL1D edge nodes in local triangle coordinates.

    Uses repo's existing edge_gl1d_rule.
    """
    rule = edge_gl1d_rule(edge_id=edge_id, n=n)
    return reference_rs_to_local_xy(rule.rs)


def map_patch_edge_to_sphere(
    patch_id: int,
    edge_id: int,
    *,
    n: int = 12,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Map edge quadrature nodes on one patch edge to 3D sphere coordinates.

    Returns
    -------
    xyz:
        Shape (n, 3).
    """
    local_xy = edge_rule_local_xy(edge_id, n)
    x = local_xy[:, 0]
    y = local_xy[:, 1]

    X, Y, Z = local_xy_to_sphere_xyz(
        x,
        y,
        patch_id,
        radius=radius,
    )

    return np.column_stack([X, Y, Z])


def expected_sphere_trace_pairs() -> list[SphereTracePair]:
    """
    Expected one-sided trace-pair table for the 8-patch sphere.

    Repo edge convention:
        edge 1: v2 -> v3  corresponds to local x+y=1
        edge 2: v3 -> v1  corresponds to local x=0, reversed
        edge 3: v1 -> v2  corresponds to local y=0

    Pair categories:
        1. Equator upper/lower same-sector edges
        2. Upper hemisphere adjacent meridian edges
        3. Lower hemisphere adjacent meridian edges
    """
    return [
        # Equator: upper/lower patches in the same longitude sector.
        SphereTracePair(1, 1, 2, 1, "same"),
        SphereTracePair(3, 1, 4, 1, "same"),
        SphereTracePair(5, 1, 6, 1, "same"),
        SphereTracePair(7, 1, 8, 1, "same"),

        # Upper hemisphere adjacent sectors.
        SphereTracePair(1, 2, 3, 3, "reversed"),
        SphereTracePair(3, 2, 5, 3, "reversed"),
        SphereTracePair(5, 2, 7, 3, "reversed"),
        SphereTracePair(7, 2, 1, 3, "reversed"),

        # Lower hemisphere adjacent sectors.
        SphereTracePair(2, 2, 4, 3, "reversed"),
        SphereTracePair(4, 2, 6, 3, "reversed"),
        SphereTracePair(6, 2, 8, 3, "reversed"),
        SphereTracePair(8, 2, 2, 3, "reversed"),
    ]


def reverse_orientation(orientation: str) -> str:
    """
    Symmetric pair has the same orientation classification.

    If A[i] == B[i], then B[i] == A[i].
    If A[i] == B[N-1-i], then B[i] == A[N-1-i].
    """
    if orientation not in {"same", "reversed"}:
        raise ValueError("orientation must be 'same' or 'reversed'.")
    return orientation


def expected_sphere_trace_pairs_bidirectional() -> list[SphereTracePair]:
    """
    Return trace-pair table in both directions.

    Useful later for per-element trace lookup.
    """
    pairs = expected_sphere_trace_pairs()
    out = list(pairs)

    for p in pairs:
        out.append(
            SphereTracePair(
                patch_minus=p.patch_plus,
                edge_minus=p.edge_plus,
                patch_plus=p.patch_minus,
                edge_plus=p.edge_minus,
                orientation=reverse_orientation(p.orientation),
            )
        )

    return out


def classify_orientation(
    xyz_minus: np.ndarray,
    xyz_plus: np.ndarray,
) -> tuple[str, float]:
    """
    Classify whether two edge node sets match with same or reversed orientation.

    Returns
    -------
    orientation:
        "same" or "reversed".
    error:
        max pointwise distance using the chosen orientation.
    """
    xyz_minus = np.asarray(xyz_minus, dtype=float)
    xyz_plus = np.asarray(xyz_plus, dtype=float)

    if xyz_minus.shape != xyz_plus.shape:
        raise ValueError("xyz_minus and xyz_plus must have the same shape.")

    err_same = float(np.max(np.linalg.norm(xyz_minus - xyz_plus, axis=1)))
    err_rev = float(np.max(np.linalg.norm(xyz_minus - xyz_plus[::-1], axis=1)))

    if err_same <= err_rev:
        return "same", err_same
    return "reversed", err_rev


def validate_trace_pair(
    pair: SphereTracePair,
    *,
    n: int = 12,
    radius: float = 1.0,
) -> tuple[str, float]:
    """
    Validate one trace pair.

    Returns
    -------
    observed_orientation, error
    """
    xyz_minus = map_patch_edge_to_sphere(
        pair.patch_minus,
        pair.edge_minus,
        n=n,
        radius=radius,
    )
    xyz_plus = map_patch_edge_to_sphere(
        pair.patch_plus,
        pair.edge_plus,
        n=n,
        radius=radius,
    )

    return classify_orientation(xyz_minus, xyz_plus)


def infer_neighbor_for_edge(
    patch_id: int,
    edge_id: int,
    *,
    n: int = 12,
    radius: float = 1.0,
    tol: float = 1e-12,
) -> SphereTracePair:
    """
    Infer the neighboring patch edge by brute-force geometric matching.

    This is a diagnostic helper, not intended for runtime RHS assembly.
    """
    target = map_patch_edge_to_sphere(
        patch_id,
        edge_id,
        n=n,
        radius=radius,
    )

    candidates: list[tuple[float, SphereTracePair]] = []

    for q_patch in range(1, 9):
        for q_edge in (1, 2, 3):
            if q_patch == patch_id and q_edge == edge_id:
                continue

            candidate = map_patch_edge_to_sphere(
                q_patch,
                q_edge,
                n=n,
                radius=radius,
            )

            orientation, err = classify_orientation(target, candidate)
            candidates.append(
                (
                    err,
                    SphereTracePair(
                        patch_id,
                        edge_id,
                        q_patch,
                        q_edge,
                        orientation,
                    ),
                )
            )

    candidates.sort(key=lambda item: item[0])
    best_err, best_pair = candidates[0]

    if best_err > tol:
        raise RuntimeError(
            f"No matching neighbor found for patch={patch_id}, edge={edge_id}. "
            f"Best error={best_err:.3e}"
        )

    return best_pair