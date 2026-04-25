from __future__ import annotations

import numpy as np

from geometry.sphere_trace_pairs import (
    SphereTracePair,
    expected_sphere_trace_pairs,
    expected_sphere_trace_pairs_bidirectional,
    validate_trace_pair,
    infer_neighbor_for_edge,
)


def _edge_key(patch_id: int, edge_id: int) -> tuple[int, int]:
    return patch_id, edge_id


def test_expected_trace_pairs_cover_all_patch_edges_once():
    """
    One-sided trace table should cover all 8*3 patch edges exactly once.
    """
    pairs = expected_sphere_trace_pairs()

    used = []
    for p in pairs:
        used.append(_edge_key(p.patch_minus, p.edge_minus))
        used.append(_edge_key(p.patch_plus, p.edge_plus))

    assert len(used) == 8 * 3
    assert len(set(used)) == 8 * 3


def test_expected_trace_pair_orientations_are_correct():
    """
    Validate every expected trace pair geometrically on the sphere.
    """
    pairs = expected_sphere_trace_pairs()

    for p in pairs:
        observed_orientation, err = validate_trace_pair(
            p,
            n=16,
            radius=1.0,
        )

        assert observed_orientation == p.orientation
        assert err < 1e-12


def test_bidirectional_trace_pairs_are_symmetric():
    """
    Bidirectional table should have 24 directed entries.
    """
    directed = expected_sphere_trace_pairs_bidirectional()
    assert len(directed) == 24

    lookup = {
        (p.patch_minus, p.edge_minus): p
        for p in directed
    }

    assert len(lookup) == 24

    for p in directed:
        q = lookup[(p.patch_plus, p.edge_plus)]
        assert q.patch_plus == p.patch_minus
        assert q.edge_plus == p.edge_minus
        assert q.orientation == p.orientation


def test_inferred_neighbors_match_expected_table():
    """
    Brute-force geometric neighbor inference should recover the expected table.

    This protects us against manual trace-table mistakes.
    """
    directed_expected = expected_sphere_trace_pairs_bidirectional()

    expected_lookup = {
        (p.patch_minus, p.edge_minus): (
            p.patch_plus,
            p.edge_plus,
            p.orientation,
        )
        for p in directed_expected
    }

    for patch_id in range(1, 9):
        for edge_id in (1, 2, 3):
            inferred = infer_neighbor_for_edge(
                patch_id,
                edge_id,
                n=16,
                radius=1.0,
                tol=1e-12,
            )

            expected = expected_lookup[(patch_id, edge_id)]
            actual = (
                inferred.patch_plus,
                inferred.edge_plus,
                inferred.orientation,
            )

            assert actual == expected