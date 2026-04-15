from __future__ import annotations

import numpy as np

from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.connectivity import build_face_connectivity
from geometry.mesh_structured import (
    structured_square_tri_mesh,
    validate_mesh_orientation,
)
from operators.exchange import interior_face_pair_mismatches, pair_face_traces
from operators.trace_policy import build_trace_policy


def _global_poly(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    A smooth polynomial field of total degree <= 2.
    """
    return (
        1.0
        + 2.0 * x
        - 0.5 * y
        + 0.75 * x * y
        + 0.30 * x**2
        - 0.20 * y**2
        + 0.10 * x**2 * y
        - 0.15 * x * y**2
        + 2 * x**2 * y**2
        - 0.25 * x**3 + 0.35 * y**3
        + 5.0 * x**3 * y**1
    )


def _check_all_interior_pairs_small(paired: dict, tol: float) -> None:
    items = interior_face_pair_mismatches(paired)
    assert len(items) > 0
    for item in items:
        assert item["max_abs_mismatch"] < tol, item


def test_exchange_step3_table1_embedded_pairing() -> None:
    VX, VY, EToV = structured_square_tri_mesh(
        nx=16,
        ny=16,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        diagonal="anti",
    )
    validate_mesh_orientation(VX, VY, EToV)
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)
    u_elem = _global_poly(X, Y)

    paired = pair_face_traces(u_elem, conn, trace)

    # boundary faces should remain NaN in uP
    assert np.all(np.isnan(paired["uP"][paired["is_boundary"]]))

    _check_all_interior_pairs_small(paired, tol=1e-13)


def test_exchange_step3_table2_projected_pairing() -> None:
    VX, VY, EToV = structured_square_tri_mesh(
        nx=16,
        ny=16,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        diagonal="anti",
    )
    validate_mesh_orientation(VX, VY, EToV)
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    rule = load_table2_rule(4)
    trace = build_trace_policy(rule, N=4, n_edge=5)
    
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)
    u_elem = _global_poly(X, Y)

    paired = pair_face_traces(u_elem, conn, trace)

    # boundary faces should remain NaN in uP
    assert np.all(np.isnan(paired["uP"][paired["is_boundary"]]))

    _check_all_interior_pairs_small(paired, tol=1e-12)

if __name__ == "__main__":
    test_exchange_step3_table1_embedded_pairing()
    print("test_exchange_step3_table1_embedded_pairing passed.")

    test_exchange_step3_table2_projected_pairing()
    print("test_exchange_step3_table2_projected_pairing passed.")