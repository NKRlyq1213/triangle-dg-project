from __future__ import annotations

import numpy as np

from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule
from operators.trace_policy import (
    build_trace_policy,
    evaluate_embedded_face_values,
    evaluate_projected_face_values,
)
from operators.vandermonde2d import vandermonde2d


def _check_face_equation(rs_face: np.ndarray, face_id: int, tol: float = 1e-12) -> None:
    r = rs_face[:, 0]
    s = rs_face[:, 1]

    if face_id == 1:
        assert np.all(np.abs(r + s) <= tol)
    elif face_id == 2:
        assert np.all(np.abs(r + 1.0) <= tol)
    elif face_id == 3:
        assert np.all(np.abs(s + 1.0) <= tol)
    else:
        raise ValueError("face_id must be 1,2,3.")


def test_table1_embedded_trace_geometry_and_weights() -> None:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    assert trace["trace_mode"] == "embedded"
    assert trace["nfp"] == 5

    for face_id in (1, 2, 3):
        rs_face = trace["face_rs"][face_id]
        t_face = trace["face_t"][face_id]
        w_face = trace["face_weights"][face_id]

        assert rs_face.shape == (5, 2)
        assert t_face.shape == (5,)
        assert w_face.shape == (5,)

        _check_face_equation(rs_face, face_id)
        assert np.all(np.diff(t_face) >= -1e-14)
        assert np.isclose(np.sum(w_face), 1.0, atol=1e-14)


def test_table1_embedded_trace_polynomial_values() -> None:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    rs_vol = rule["rs"]
    V_vol = vandermonde2d(4, rs_vol[:, 0], rs_vol[:, 1])

    # arbitrary modal polynomial of degree <= 4
    coeffs = np.linspace(0.25, 1.75, V_vol.shape[1])
    u_vol = V_vol @ coeffs

    face_values = evaluate_embedded_face_values(u_vol, trace)

    for face_id in (1, 2, 3):
        rs_face = trace["face_rs"][face_id]
        V_face = vandermonde2d(4, rs_face[:, 0], rs_face[:, 1])
        u_exact = V_face @ coeffs
        err = np.max(np.abs(face_values[face_id] - u_exact))
        assert err < 1e-13


def test_table2_projected_trace_geometry_weights_and_exactness() -> None:
    rule = load_table2_rule(4)
    trace = build_trace_policy(rule, N=4, n_edge=5)

    assert trace["trace_mode"] == "projected"
    assert trace["nfp"] == 5

    rs_vol = rule["rs"]
    V_vol = trace["V_vol"]

    coeffs = np.linspace(-0.5, 1.2, V_vol.shape[1])
    u_vol = V_vol @ coeffs

    face_values = evaluate_projected_face_values(u_vol, trace)

    for face_id in (1, 2, 3):
        rs_face = trace["face_rs"][face_id]
        t_face = trace["face_t"][face_id]
        w_face = trace["face_weights"][face_id]

        assert rs_face.shape == (5, 2)
        assert t_face.shape == (5,)
        assert w_face.shape == (5,)

        _check_face_equation(rs_face, face_id)
        assert np.all(np.diff(t_face) >= -1e-14)
        assert np.isclose(np.sum(w_face), 1.0, atol=1e-14)

        V_face = vandermonde2d(4, rs_face[:, 0], rs_face[:, 1])
        u_exact = V_face @ coeffs

        err = np.max(np.abs(face_values[face_id] - u_exact))
        assert err < 1e-12