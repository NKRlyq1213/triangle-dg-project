from __future__ import annotations

import itertools
import numpy as np

from geometry.reference_triangle import reference_triangle_vertices
from geometry.barycentric import barycentric_to_cartesian


TABLE2_RAW: dict[int, list[dict[str, float | str]]] = {
    1: [
        {"sym": "S3", "b1": 0.1666666666666666, "b2": 0.1666666666666666, "ws": 0.3333333333333333},
    ],
    2: [
        {"sym": "S3", "b1": 0.09157621350977067, "b2": 0.09157621350977067, "ws": 0.1099517436553218},
        {"sym": "S3", "b1": 0.4459484909159648, "b2": 0.4459484909159648, "ws": 0.2233815896780115},
    ],
    3: [
        {"sym": "S3", "b1": 0.2194299825497830, "b2": 0.2194299825497830, "ws": 0.1713331241529809},
        {"sym": "S3", "b1": 0.4801379641122150, "b2": 0.4801379641122150, "ws": 0.08073108959303095},
        {"sym": "S6", "b1": 0.1416190159239682, "b2": 0.0193717243612408, "ws": 0.04063455979366068},
    ],
    4: [
        {"sym": "S6", "b1": 0.7284923929554044, "b2": 0.2631128296346379, "ws": 0.02723031417443505},
        {"sym": "S3", "b1": 0.4592925882927232, "b2": 0.4592925882927232, "ws": 0.09509163426728455},
        {"sym": "S3", "b1": 0.1705693077517602, "b2": 0.1705693077517602, "ws": 0.1032173705347182},
        {"sym": "S3", "b1": 0.05054722831703096, "b2": 0.05054722831703096, "ws": 0.03245849762319804},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.1443156076777874},
    ],
}


def _expected_count(sym: str) -> int:
    lookup = {"S1": 1, "S3": 3, "S6": 6}
    if sym not in lookup:
        raise ValueError(f"Unknown symmetry label: {sym}")
    return lookup[sym]


def _unique_permutations(values: tuple[float, float, float], ndigits: int = 15) -> np.ndarray:
    perms = set()
    for p in itertools.permutations(values):
        perms.add(tuple(round(v, ndigits) for v in p))
    arr = np.array(sorted(perms), dtype=float)
    return arr


def _expand_row(sym: str, b1: float, b2: float) -> np.ndarray:
    b3 = 1.0 - b1 - b2
    bary = _unique_permutations((b1, b2, b3))
    expected = _expected_count(sym)
    if bary.shape[0] != expected:
        raise ValueError(
            f"Symmetry expansion mismatch: sym={sym}, got {bary.shape[0]}, expected {expected}."
        )
    return bary


def _rule_to_xy_rs(bary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = reference_triangle_vertices()
    xy = barycentric_to_cartesian(bary, verts)

    # (xi, eta) -> (r, s)
    rs = np.empty_like(xy)
    rs[:, 0] = 2.0 * xy[:, 0] - 1.0
    rs[:, 1] = 2.0 * xy[:, 1] - 1.0
    return xy, rs


def load_table2_rule(order: int) -> dict:
    if order not in TABLE2_RAW:
        raise KeyError(f"Order {order} is not available in TABLE2_RAW.")

    bary_all = []
    ws_all = []

    for row in TABLE2_RAW[order]:
        sym = str(row["sym"])
        b1 = float(row["b1"])
        b2 = float(row["b2"])
        ws = float(row["ws"])

        bary_row = _expand_row(sym, b1, b2)
        bary_all.append(bary_row)
        ws_all.append(np.full(bary_row.shape[0], ws, dtype=float))

    bary = np.vstack(bary_all)
    ws = np.concatenate(ws_all)
    xy, rs = _rule_to_xy_rs(bary)

    return {
        "table": "table2",
        "order": order,
        "bary": bary,
        "xy": xy,
        "rs": rs,
        "ws": ws,
        "we": None,
    }
