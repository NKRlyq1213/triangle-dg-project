from __future__ import annotations

import numpy as np

from data.edge_rules import edge_gl1d_rule
from geometry.reference_triangle import reference_triangle_area
from operators.boundary import volume_to_edge_operator
from operators.vandermonde2d import vandermonde2d


def _check_rule_has_required_keys(rule: dict, required: tuple[str, ...]) -> None:
    for key in required:
        if key not in rule:
            raise KeyError(f"rule is missing required key: {key}")


def _face_mask_from_rs(
    rs: np.ndarray,
    face_id: int,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Identify which reference points lie on a given local face.

    Reference triangle
    ------------------
        v1 = (-1, -1)
        v2 = ( 1, -1)
        v3 = (-1,  1)

    Face convention
    ---------------
        face 1: v2 -> v3  => r + s = 0
        face 2: v3 -> v1  => r = -1
        face 3: v1 -> v2  => s = -1
    """
    rs = np.asarray(rs, dtype=float)
    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs must have shape (Np, 2).")

    r = rs[:, 0]
    s = rs[:, 1]

    if face_id == 1:
        return np.abs(r + s) <= tol
    if face_id == 2:
        return np.abs(r + 1.0) <= tol
    if face_id == 3:
        return np.abs(s + 1.0) <= tol

    raise ValueError("face_id must be 1, 2, or 3.")


def face_parameter_t_from_rs(
    rs_face: np.ndarray,
    face_id: int,
) -> np.ndarray:
    """
    Compute the 1D parameter t in [0, 1] along a local face, consistent with
    the existing local face orientation.

    Face convention
    ---------------
        face 1: v2 -> v3
            (r, s) = (1 - 2t, -1 + 2t)
            t = (1 - r)/2 = (1 + s)/2

        face 2: v3 -> v1
            (r, s) = (-1, 1 - 2t)
            t = (1 - s)/2

        face 3: v1 -> v2
            (r, s) = (-1 + 2t, -1)
            t = (1 + r)/2
    """
    rs_face = np.asarray(rs_face, dtype=float)
    if rs_face.ndim != 2 or rs_face.shape[1] != 2:
        raise ValueError("rs_face must have shape (Nf, 2).")

    r = rs_face[:, 0]
    s = rs_face[:, 1]

    if face_id == 1:
        return 0.5 * (1.0 - r)
    if face_id == 2:
        return 0.5 * (1.0 - s)
    if face_id == 3:
        return 0.5 * (1.0 + r)

    raise ValueError("face_id must be 1, 2, or 3.")


def _build_table1_embedded_trace(
    rule: dict,
    tol: float = 1e-12,
) -> dict:
    """
    Build embedded face-node descriptors for a Table 1 rule.

    Output
    ------
    face_node_ids : dict[int, np.ndarray]
        Indices into the volume-node array, one per face.
    face_rs : dict[int, np.ndarray]
        Face nodes in (r, s), sorted by local face orientation parameter t.
    face_t : dict[int, np.ndarray]
        Sorted t-parameter in [0,1].
    face_weights : dict[int, np.ndarray]
        Edge weights (we) sorted consistently with face_t.
    """
    _check_rule_has_required_keys(rule, ("table", "order", "rs", "we"))
    if str(rule["table"]).lower() != "table1":
        raise ValueError("This builder only accepts Table 1 rules.")

    rs = np.asarray(rule["rs"], dtype=float)
    we = np.asarray(rule["we"], dtype=float).reshape(-1)

    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rule['rs'] must have shape (Np, 2).")
    if rs.shape[0] != we.size:
        raise ValueError("rule['we'] size must match rule['rs'].")

    face_node_ids: dict[int, np.ndarray] = {}
    face_rs: dict[int, np.ndarray] = {}
    face_t: dict[int, np.ndarray] = {}
    face_weights: dict[int, np.ndarray] = {}

    counts = []

    for face_id in (1, 2, 3):
        mask = _face_mask_from_rs(rs, face_id=face_id, tol=tol)
        ids = np.where(mask)[0]

        if ids.size == 0:
            raise ValueError(f"Table 1 rule has no embedded nodes on face {face_id}.")

        rs_f = rs[ids]
        t_f = face_parameter_t_from_rs(rs_f, face_id=face_id)

        perm = np.argsort(t_f, kind="mergesort")

        ids = ids[perm]
        rs_f = rs_f[perm]
        t_f = t_f[perm]
        w_f = we[ids]

        if np.any(np.isnan(w_f)):
            raise ValueError(f"Table 1 face {face_id} contains NaN edge weights.")

        face_node_ids[face_id] = ids
        face_rs[face_id] = rs_f
        face_t[face_id] = t_f
        face_weights[face_id] = w_f
        counts.append(ids.size)

    if len(set(counts)) != 1:
        raise ValueError(f"Table 1 faces do not have equal number of trace nodes: {counts}")

    nfp = counts[0]

    return {
        "trace_mode": "embedded",
        "table": "table1",
        "order": int(rule["order"]),
        "nfp": int(nfp),
        "face_node_ids": face_node_ids,
        "face_rs": face_rs,
        "face_t": face_t,
        "face_weights": face_weights,
    }


def evaluate_embedded_face_values(
    u_vol: np.ndarray,
    trace: dict,
) -> dict[int, np.ndarray]:
    """
    Extract Table 1 embedded face values directly from volume values.
    """
    u_vol = np.asarray(u_vol, dtype=float).reshape(-1)

    if trace.get("trace_mode", None) != "embedded":
        raise ValueError("trace must be an embedded trace descriptor.")

    out: dict[int, np.ndarray] = {}
    for face_id, ids in trace["face_node_ids"].items():
        out[face_id] = u_vol[np.asarray(ids, dtype=int)]
    return out


def _build_table2_projected_trace(
    rule: dict,
    N: int,
    n_edge: int | None = None,
    area: float | None = None,
) -> dict:
    """
    Build projected face-trace descriptors for a Table 2 rule.

    Notes
    -----
    - Table 2 face nodes are NOT embedded in the volume nodes.
    - We therefore build a separate GL1D edge grid and a volume-to-edge
      evaluation/projection operator for each face.
    """
    _check_rule_has_required_keys(rule, ("table", "order", "rs", "ws"))
    if str(rule["table"]).lower() != "table2":
        raise ValueError("This builder only accepts Table 2 rules.")

    if n_edge is None:
        n_edge = int(rule["order"]) + 1
    if n_edge <= 0:
        raise ValueError("n_edge must be a positive integer.")

    if area is None:
        area = reference_triangle_area()

    rs_vol = np.asarray(rule["rs"], dtype=float)
    w_vol = np.asarray(rule["ws"], dtype=float).reshape(-1)

    if rs_vol.ndim != 2 or rs_vol.shape[1] != 2:
        raise ValueError("rule['rs'] must have shape (Np, 2).")
    if rs_vol.shape[0] != w_vol.size:
        raise ValueError("rule['ws'] size must match rule['rs'].")

    V_vol = vandermonde2d(N, rs_vol[:, 0], rs_vol[:, 1])

    face_rs: dict[int, np.ndarray] = {}
    face_t: dict[int, np.ndarray] = {}
    face_weights: dict[int, np.ndarray] = {}
    face_operators: dict[int, np.ndarray] = {}

    for face_id in (1, 2, 3):
        edge_rule = edge_gl1d_rule(edge_id=face_id, n=n_edge)
        rs_f = np.asarray(edge_rule.rs, dtype=float)
        t_f = np.asarray(edge_rule.t01, dtype=float)
        w_f = np.asarray(edge_rule.weights, dtype=float)

        V_edge = vandermonde2d(N, rs_f[:, 0], rs_f[:, 1])
        E_edge = volume_to_edge_operator(
            V_vol=V_vol,
            weights=w_vol,
            V_edge=V_edge,
            area=area,
        )

        face_rs[face_id] = rs_f
        face_t[face_id] = t_f
        face_weights[face_id] = w_f
        face_operators[face_id] = E_edge

    return {
        "trace_mode": "projected",
        "table": "table2",
        "order": int(rule["order"]),
        "N": int(N),
        "nfp": int(n_edge),
        "face_rs": face_rs,
        "face_t": face_t,
        "face_weights": face_weights,
        "face_operators": face_operators,
        "V_vol": V_vol,
    }


def evaluate_projected_face_values(
    u_vol: np.ndarray,
    trace: dict,
) -> dict[int, np.ndarray]:
    """
    Evaluate Table 2 projected face values from volume values.
    """
    u_vol = np.asarray(u_vol, dtype=float).reshape(-1)

    if trace.get("trace_mode", None) != "projected":
        raise ValueError("trace must be a projected trace descriptor.")

    out: dict[int, np.ndarray] = {}
    for face_id, E_edge in trace["face_operators"].items():
        out[face_id] = np.asarray(E_edge, dtype=float) @ u_vol
    return out


def build_trace_policy(
    rule: dict,
    N: int | None = None,
    n_edge: int | None = None,
    area: float | None = None,
    tol: float = 1e-12,
) -> dict:
    """
    Unified entry point for trace-policy construction.

    Table 1
    -------
    Returns embedded face-node descriptors.
    N is ignored.

    Table 2
    -------
    Returns projected face-trace descriptors.
    N is required because the volume-to-edge operator depends on the modal basis order.
    """
    table = str(rule.get("table", "")).lower().strip()

    if table == "table1":
        return _build_table1_embedded_trace(rule, tol=tol)

    if table == "table2":
        if N is None:
            raise ValueError("For Table 2 projected trace, N must be provided.")
        return _build_table2_projected_trace(rule, N=N, n_edge=n_edge, area=area)

    raise ValueError("rule['table'] must be either 'table1' or 'table2'.")