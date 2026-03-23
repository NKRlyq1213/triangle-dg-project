from __future__ import annotations

import itertools
import numpy as np

from geometry.barycentric import raw_barycentric_to_reference_rs


TABLE1_RAW: dict[int, list[dict[str, float | str | None]]] = {
    1: [
        {"sym": "S6", "b1": 0.2113248654051871, "b2": 0.0, "ws": 0.16666666666666667, "we": 0.5000000000000000},
    ],
    2: [
        {"sym": "S6", "b1": 0.1127016653792583, "b2": 0.0, "ws": 0.04166666666666666, "we": 0.2777777777777777},
        {"sym": "S3", "b1": 0.5000000000000000, "b2": 0.0, "ws": 0.09999999999999999, "we": 0.4444444444444444},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.45000000000000000, "we": None},
    ],
    3: [
        {"sym": "S6", "b1": 0.06943184420297367, "b2": 0.0, "ws": 0.01509901487256561, "we": 0.1739274225687269},
        {"sym": "S6", "b1": 0.3300094782075718, "b2": 0.0, "ws": 0.04045654068298990, "we": 0.3260725774312731},
        {"sym": "S6", "b1": 0.5841571139756568, "b2": 0.1870738791912763, "ws": 0.11111111111111111, "we": None},
    ],
    4: [
        {"sym": "S6", "b1": 0.04691007703066797, "b2": 0.0, "ws": 0.006601315081001592, "we": 0.1184634425280944},
        {"sym": "S6", "b1": 0.2307653449471584, "b2": 0.0, "ws": 0.02053045968042892, "we": 0.2393143352496833},
        {"sym": "S3", "b1": 0.5000000000000000, "b2": 0.0, "ws": 0.01853708483394990, "we": 0.2844444444444446},
        {"sym": "S3", "b1": 0.1394337314154536, "b2": 0.1394337314154536, "ws": 0.10542932962084440, "we": None},
        {"sym": "S3", "b1": 0.4384239524408185, "b2": 0.4384239524408185, "ws": 0.12473673228977350, "we": None},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.09109991119771331, "we": None},
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


def load_table1_rule(order: int) -> dict:
    """
    Load Table 1 quadrature data and return points directly in the NEW
    reference triangle coordinates (r, s).

    Notes
    -----
    - 'bary' is kept in the raw literature convention.
    - 'rs' is the actual point set used by the code base.
    """
    if order not in TABLE1_RAW:
        raise KeyError(f"Order {order} is not available in TABLE1_RAW.")

    bary_all = []
    ws_all = []
    we_all = []

    for row in TABLE1_RAW[order]:
        sym = str(row["sym"])
        b1 = float(row["b1"])
        b2 = float(row["b2"])
        ws = float(row["ws"])
        we = np.nan if row["we"] is None else float(row["we"])

        bary_row = _expand_row(sym, b1, b2)
        bary_all.append(bary_row)
        ws_all.append(np.full(bary_row.shape[0], ws, dtype=float))
        we_all.append(np.full(bary_row.shape[0], we, dtype=float))

    bary = np.vstack(bary_all)
    ws = np.concatenate(ws_all)
    we = np.concatenate(we_all)
    rs = raw_barycentric_to_reference_rs(bary)

    edge_mask = ~np.isnan(we)

    return {
        "table": "table1",
        "order": order,
        "bary": bary,
        "rs": rs,
        "ws": ws,
        "we": we,
        "edge_mask": edge_mask,
    }
