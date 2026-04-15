from __future__ import annotations

import numpy as np

from data.table1_rules import load_table1_rule
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.mesh_structured import structured_square_tri_mesh
from operators.rhs_split_conservative_exchange import (
    _build_boundary_state_from_opposite_boundary,
    _opposite_pair_needs_flip,
    build_surface_exchange_cache,
)
from operators.trace_policy import build_trace_policy


def _flat_face_index(k: int, f: int) -> int:
    return int(k) * 3 + (int(f) - 1)


def _build_fixture() -> dict:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    VX, VY, EToV = structured_square_tri_mesh(nx=3, ny=2, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    surface_cache = build_surface_exchange_cache(
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
    )

    return {
        "rule": rule,
        "conn": conn,
        "face_geom": face_geom,
        "surface_cache": surface_cache,
    }


def test_opposite_boundary_map_pairs_faces_and_preserves_axis_coordinates() -> None:
    fx = _build_fixture()
    conn = fx["conn"]
    face_geom = fx["face_geom"]
    cache = fx["surface_cache"]

    boundary_flat = np.asarray(cache["boundary_flat"], dtype=bool)
    opposite_face_flat = np.asarray(cache["opposite_boundary_face_flat"], dtype=np.int64)
    opposite_flip_flat = np.asarray(cache["opposite_boundary_flip_flat"], dtype=bool)

    assert np.all(opposite_face_flat[boundary_flat] >= 0)
    assert np.all(opposite_face_flat[~boundary_flat] == -1)

    side_by_face: dict[int, str] = {}
    for side in ("left", "right", "bottom", "top"):
        for k, f in conn["boundary_groups"][side]:
            side_by_face[_flat_face_index(int(k), int(f))] = side

    assert len(side_by_face) == int(np.sum(boundary_flat))

    face_midpoints = np.asarray(conn["face_midpoints"], dtype=float)
    x_face = np.asarray(face_geom["x_face"], dtype=float)
    y_face = np.asarray(face_geom["y_face"], dtype=float)

    for idx in np.nonzero(boundary_flat)[0]:
        opp = int(opposite_face_flat[idx])
        assert boundary_flat[opp]
        assert int(opposite_face_flat[opp]) == int(idx)

        side = side_by_face[int(idx)]
        opp_side = side_by_face[opp]
        k, jf = divmod(int(idx), 3)
        ko, jfo = divmod(opp, 3)

        if side == "left":
            assert opp_side == "right"
            assert np.isclose(
                face_midpoints[k, jf, 1],
                face_midpoints[ko, jfo, 1],
                atol=1e-12,
                rtol=1e-12,
            )
            axis_a = y_face[k, jf, :]
            axis_b = y_face[ko, jfo, :]
        elif side == "right":
            assert opp_side == "left"
            assert np.isclose(
                face_midpoints[k, jf, 1],
                face_midpoints[ko, jfo, 1],
                atol=1e-12,
                rtol=1e-12,
            )
            axis_a = y_face[k, jf, :]
            axis_b = y_face[ko, jfo, :]
        elif side == "bottom":
            assert opp_side == "top"
            assert np.isclose(
                face_midpoints[k, jf, 0],
                face_midpoints[ko, jfo, 0],
                atol=1e-12,
                rtol=1e-12,
            )
            axis_a = x_face[k, jf, :]
            axis_b = x_face[ko, jfo, :]
        else:
            assert side == "top"
            assert opp_side == "bottom"
            assert np.isclose(
                face_midpoints[k, jf, 0],
                face_midpoints[ko, jfo, 0],
                atol=1e-12,
                rtol=1e-12,
            )
            axis_a = x_face[k, jf, :]
            axis_b = x_face[ko, jfo, :]

        d_same = float(np.max(np.abs(axis_a - axis_b)))
        d_flip = float(np.max(np.abs(axis_a - axis_b[::-1])))
        if bool(opposite_flip_flat[idx]):
            assert d_flip <= d_same + 1e-14
            assert d_flip < 1e-12
        else:
            assert d_same <= d_flip + 1e-14
            assert d_same < 1e-12


def test_opposite_boundary_state_uses_partner_face_and_flip() -> None:
    fx = _build_fixture()
    cache = fx["surface_cache"]

    K = int(cache["K"])
    Np = int(cache["Np"])
    q_elem = np.arange(K * Np, dtype=float).reshape(K, Np)

    boundary_flat = np.asarray(cache["boundary_flat"], dtype=bool)
    opposite_face_flat = np.asarray(cache["opposite_boundary_face_flat"], dtype=np.int64)
    opposite_flip_flat = np.asarray(cache["opposite_boundary_flip_flat"], dtype=bool)
    owner_elem = np.asarray(cache["owner_elem_flat_numba"], dtype=np.int64)
    owner_node_ids = np.asarray(cache["owner_node_ids_flat_numba"], dtype=np.int64)

    qB = _build_boundary_state_from_opposite_boundary(q_elem=q_elem, cache=cache)

    for idx in np.nonzero(boundary_flat)[0]:
        opp = int(opposite_face_flat[idx])
        expected = q_elem[owner_elem[opp], owner_node_ids[opp, :]]
        if bool(opposite_flip_flat[idx]):
            expected = expected[::-1]

        k, jf = divmod(int(idx), 3)
        assert np.array_equal(qB[k, jf, :], expected)


def test_opposite_pair_flip_detector_handles_same_reverse_and_tie_cases() -> None:
    x_face = np.zeros((2, 3, 4), dtype=float)
    y_face = np.zeros((2, 3, 4), dtype=float)

    y_line = np.array([0.0, 0.3, 0.7, 1.0], dtype=float)
    y_face[0, 0, :] = y_line
    y_face[1, 0, :] = y_line

    assert not _opposite_pair_needs_flip(x_face, y_face, ka=0, fa=1, kb=1, fb=1, axis=1)

    y_face[1, 0, :] = y_line[::-1]
    assert _opposite_pair_needs_flip(x_face, y_face, ka=0, fa=1, kb=1, fb=1, axis=1)

    x_face_tie = np.zeros((2, 3, 2), dtype=float)
    y_face_tie = np.zeros((2, 3, 2), dtype=float)

    x_face_tie[0, 0, :] = np.array([0.0, 1.0], dtype=float)
    y_face_tie[0, 0, :] = np.array([0.0, 0.0], dtype=float)

    x_face_tie[1, 0, :] = np.array([0.0, 1.0], dtype=float)
    y_face_tie[1, 0, :] = np.array([0.0, 0.0], dtype=float)
    assert not _opposite_pair_needs_flip(
        x_face_tie,
        y_face_tie,
        ka=0,
        fa=1,
        kb=1,
        fb=1,
        axis=0,
    )

    x_face_tie[1, 0, :] = np.array([1.0, 0.0], dtype=float)
    y_face_tie[1, 0, :] = np.array([0.0, 0.0], dtype=float)
    assert _opposite_pair_needs_flip(
        x_face_tie,
        y_face_tie,
        ka=0,
        fa=1,
        kb=1,
        fb=1,
        axis=0,
    )
