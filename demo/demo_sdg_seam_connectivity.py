from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.sphere_flat_mesh import mesh_summary, sphere_flat_square_mesh
from geometry.sdg_seam_connectivity import (
    build_sdg_sphere_face_connectivity,
    validate_sdg_sphere_connectivity,
)
from visualization.sdg_seam_diagnostics import write_all_sdg_seam_diagnostics


def parse_args():
    p = argparse.ArgumentParser(description="SDG sphere seam connectivity diagnostics.")
    p.add_argument("--n-sub", type=int, default=2, help="Subdivisions per macro triangle.")
    p.add_argument("--R", type=float, default=1.0, help="Sphere radius. Flat square is fixed [-1,1]^2.")
    p.add_argument("--n-samples", type=int, default=9, help="Samples per seam face for xyz mismatch check.")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Flat square remains [-1,1]^2.
    VX, VY, EToV, elem_patch_id = sphere_flat_square_mesh(
        n_sub=args.n_sub,
        R=1.0,
    )

    conn = build_sdg_sphere_face_connectivity(
        VX=VX,
        VY=VY,
        EToV=EToV,
        elem_patch_id=elem_patch_id,
    )

    diagnostics = validate_sdg_sphere_connectivity(
        VX=VX,
        VY=VY,
        EToV=EToV,
        elem_patch_id=elem_patch_id,
        conn=conn,
        R=args.R,
        n_samples=args.n_samples,
    )

    outdir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "outputs" / "sdg_seam_connectivity" / f"nsub{args.n_sub}_R{args.R:g}"
    )

    paths = write_all_sdg_seam_diagnostics(
        VX=VX,
        VY=VY,
        EToV=EToV,
        elem_patch_id=elem_patch_id,
        conn=conn,
        output_dir=outdir,
        R=args.R,
        n_samples=args.n_samples,
    )

    print("\n=== Mesh summary ===")
    for k, v in mesh_summary(VX, VY, EToV, elem_patch_id).items():
        print(f"{k}: {v}")

    print("\n=== SDG seam connectivity diagnostics ===")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")

    expected_seam_pairs = 4 * args.n_sub
    print("\n=== Expected counts ===")
    print(f"expected sphere seam pairs = 4*n_sub = {expected_seam_pairs}")
    print("expected remaining boundary faces = 0")
    print("expected max_seam_xyz_error ≈ machine precision")

    print("\n=== Output figures ===")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
