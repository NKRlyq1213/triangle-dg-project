from __future__ import annotations

import numpy as np

from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule


def summarize_rule(rule: dict) -> None:
    print("=" * 60)
    print(f"table = {rule['table']}, order = {rule['order']}")
    print("num points =", len(rule["ws"]))
    print("sum(ws)    =", np.sum(rule["ws"]))
    print("xy shape   =", rule["xy"].shape)
    print("rs shape   =", rule["rs"].shape)

    if rule["we"] is not None:
        edge_mask = rule["edge_mask"]
        print("num edge-marked points =", int(np.sum(edge_mask)))
        print("unique edge weights    =", np.unique(rule["we"][edge_mask]))
    else:
        print("we = None (use separate GL1D rule)")


def main() -> None:
    for order in [1, 2, 3, 4]:
        summarize_rule(load_table1_rule(order))

    for order in [1, 2, 3, 4]:
        summarize_rule(load_table2_rule(order))


if __name__ == "__main__":
    main()
