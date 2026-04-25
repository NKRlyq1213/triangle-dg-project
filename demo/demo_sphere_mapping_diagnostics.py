from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Allow running as: python demo/demo_sphere_mapping_diagnostics.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.table1_rules import load_table1_rule
from geometry.sphere_flat_mesh import mesh_summary, sphere_flat_square_mesh
from geometry.sphere_flat_metrics import build_sphere_flat_geometry_cache, geometry_diagnostics
from problems.sphere_advection import flattened_velocity_from_cache
from visualization.sphere_mapping_diagnostics import write_all_mapping_diagnostics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnostics for flattened-square equal-area sphere mapping."
    )
    p.add_argument("--n-sub", type=int, default=2, help="Subdivisions per macro triangle.")
    p.add_argument("--order", type=int, default=3, help="Table1 order, available: 1,2,3,4.")
    p.add_argument("--R", type=float, default=1.0, help="Sphere radius. The flattened square remains [-1,1]^2.")
    p.add_argument("--u0", type=float, default=1.0, help="Velocity scale.")
    p.add_argument("--alpha0", type=float, default=0.0, help="Velocity tilt angle in radians.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: outputs/sphere_mapping/order{order}_nsub{n_sub}",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rule = load_table1_rule(args.order)
    rs_nodes = rule["rs"]

    VX, VY, EToV, elem_patch_id = sphere_flat_square_mesh(
        n_sub=args.n_sub,
        R=1.0,
    )

    cache = build_sphere_flat_geometry_cache(
        rs_nodes=rs_nodes,
        VX=VX,
        VY=VY,
        EToV=EToV,
        elem_patch_id=elem_patch_id,
        R=args.R,
    )

    u1, u2, u_sph, v_sph = flattened_velocity_from_cache(
        cache,
        u0=args.u0,
        alpha0=args.alpha0,
    )

    outdir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "outputs" / "sphere_mapping" / f"order{args.order}_nsub{args.n_sub}"
    )

    paths = write_all_mapping_diagnostics(
        VX=VX,
        VY=VY,
        EToV=EToV,
        elem_patch_id=elem_patch_id,
        cache=cache,
        u1=u1,
        u2=u2,
        u_sph=u_sph,
        v_sph=v_sph,
        output_dir=outdir,
        R=args.R,
    )

    print("\n=== Mesh summary ===")
    for k, v in mesh_summary(VX, VY, EToV, elem_patch_id).items():
        print(f"{k}: {v}")

    print("\n=== Geometry diagnostics ===")
    for k, v in geometry_diagnostics(cache, R=args.R).items():
        print(f"{k}: {v}")

    print("\n=== Output figures ===")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

