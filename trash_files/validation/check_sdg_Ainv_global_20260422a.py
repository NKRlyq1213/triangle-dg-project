from __future__ import annotations

import argparse
import numpy as np

from data.rule_registry import load_rule
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.sphere_flattened_connectivity import build_flattened_sphere_mesh
from geometry.sdg_Ainv_global_20260422a import sdg_A_Ainv_batch_20260422a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsub", type=int, default=4)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--radius", type=float, default=1.0)
    args = parser.parse_args()

    mesh = build_flattened_sphere_mesh(args.nsub)
    rule = load_rule("table1", args.order)

    X, Y = map_reference_nodes_to_all_elements(
        rule["rs"],
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        mesh.EToV,
    )

    A, Ainv = sdg_A_Ainv_batch_20260422a(
        X,
        Y,
        mesh.face_ids,
        radius=args.radius,
    )

    I1 = np.einsum("...ik,...kj->...ij", A, Ainv)
    I2 = np.einsum("...ik,...kj->...ij", Ainv, A)
    eye = np.eye(2)

    detA = np.linalg.det(A)
    expected_det = np.pi * args.radius**2

    print("=== SDG A/Ainv diagnostic based on SDG4PDEOnSphere20260422a.pdf ===")
    print(f"nsub        : {args.nsub}")
    print(f"table       : table1")
    print(f"order       : {args.order}")
    print(f"radius      : {args.radius}")
    print()

    print("Shapes:")
    print(f"  X,Y       : {X.shape}")
    print(f"  A         : {A.shape}")
    print(f"  Ainv      : {Ainv.shape}")
    print()

    print("Identity checks:")
    print(f"  max |A Ainv - I|      = {np.max(np.abs(I1 - eye)):.3e}")
    print(f"  max |Ainv A - I|      = {np.max(np.abs(I2 - eye)):.3e}")
    print()

    print("Determinant checks:")
    print(f"  min det(A)            = {np.min(detA):.15e}")
    print(f"  max det(A)            = {np.max(detA):.15e}")
    print(f"  expected pi R^2       = {expected_det:.15e}")
    print(f"  max |det(A)-piR^2|    = {np.max(np.abs(detA - expected_det)):.3e}")
    print()

    print("Finite checks:")
    print(f"  all finite A          = {np.all(np.isfinite(A))}")
    print(f"  all finite Ainv       = {np.all(np.isfinite(Ainv))}")


if __name__ == "__main__":
    main()