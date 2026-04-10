from __future__ import annotations

import numpy as np

from data.edge_rules import gauss_legendre_1d, all_edge_gl1d_rules


def main() -> None:
    t01, w01 = gauss_legendre_1d(4)
    print("GL1D nodes on [0,1]:", t01)
    print("GL1D weights on [0,1]:", w01)
    print("sum(weights) =", np.sum(w01))

    rules = all_edge_gl1d_rules(4)
    for edge_id, rule in rules.items():
        print("-" * 50)
        print(f"edge {edge_id}")
        print("length =", rule.length)
        print("xy shape =", rule.xy.shape)
        print("first point =", rule.xy[0])
        print("last point  =", rule.xy[-1])


if __name__ == "__main__":
    main()