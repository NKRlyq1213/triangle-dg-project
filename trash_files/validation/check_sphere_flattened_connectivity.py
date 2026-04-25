from __future__ import annotations

import argparse
import numpy as np

from geometry.sphere_flattened_connectivity import (
    build_flattened_sphere_mesh,
    validate_closed_connectivity,
)
from geometry.sphere_flattened_mesh import mesh_element_areas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsub", type=int, default=4)
    args = parser.parse_args()

    mesh = build_flattened_sphere_mesh(args.nsub)
    summary = validate_closed_connectivity(mesh)

    areas = mesh_element_areas(mesh.nodes, mesh.EToV)

    print("=== Flattened sphere mesh/connectivity diagnostic ===")
    print(f"nsub                  : {mesh.nsub}")
    print(f"num nodes             : {mesh.nodes.shape[0]}")
    print(f"num elements          : {mesh.EToV.shape[0]}")
    print(f"expected elements     : {8 * mesh.nsub**2}")
    print(f"min element area      : {np.min(areas):.6e}")
    print(f"max element area      : {np.max(areas):.6e}")
    print(f"total flat area       : {np.sum(areas):.15e}")
    print()

    print("Face id counts:")
    unique, counts = np.unique(mesh.face_ids, return_counts=True)
    for fid, count in zip(unique, counts):
        print(f"  T{fid}: {count}")
    print()

    print("Connectivity summary:")
    for key, value in summary.items():
        print(f"  {key:30s}: {value}")

    print()
    print("Boundary/gluing:")
    print(f"  planar boundary faces : {mesh.conn.planar_boundary_faces.shape[0]}")
    print(f"  glued boundary pairs  : {mesh.conn.glued_boundary_pairs.shape[0]}")
    print(f"  remaining boundaries  : {int(np.sum(mesh.conn.is_boundary))}")

    print()
    print("First few glued pairs:")
    for row in mesh.conn.glued_boundary_pairs[: min(10, len(mesh.conn.glued_boundary_pairs))]:
        ka, fa, kb, fb = row
        print(f"  ({ka}, face {fa}) <-> ({kb}, face {fb})")


if __name__ == "__main__":
    main()