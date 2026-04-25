from __future__ import annotations

from geometry.sphere_trace_pairs import (
    expected_sphere_trace_pairs,
    expected_sphere_trace_pairs_bidirectional,
    validate_trace_pair,
    infer_neighbor_for_edge,
)


def main() -> None:
    radius = 1.0
    n = 16

    print("=== Sphere trace-pair diagnostics ===")
    print(f"radius = {radius}")
    print(f"edge nodes per edge = {n}")
    print()

    print("[1] expected one-sided trace pairs")
    for p in expected_sphere_trace_pairs():
        observed_orientation, err = validate_trace_pair(
            p,
            n=n,
            radius=radius,
        )

        ok_orientation = observed_orientation == p.orientation
        ok_error = err < 1e-12

        status = "OK" if (ok_orientation and ok_error) else "FAIL"

        print(
            f"  T{p.patch_minus}.e{p.edge_minus} "
            f"<-> T{p.patch_plus}.e{p.edge_plus} | "
            f"expected={p.orientation:8s}, "
            f"observed={observed_orientation:8s}, "
            f"err={err:.3e} | {status}"
        )

    print()
    print("[2] brute-force inferred neighbor for every directed edge")

    for patch_id in range(1, 9):
        for edge_id in (1, 2, 3):
            p = infer_neighbor_for_edge(
                patch_id,
                edge_id,
                n=n,
                radius=radius,
                tol=1e-12,
            )

            print(
                f"  T{patch_id}.e{edge_id} -> "
                f"T{p.patch_plus}.e{p.edge_plus}, "
                f"orientation={p.orientation}"
            )

    print()
    print("[3] directed table size")
    directed = expected_sphere_trace_pairs_bidirectional()
    print(f"  directed entries = {len(directed)}")
    print("  expected         = 24")


if __name__ == "__main__":
    main()