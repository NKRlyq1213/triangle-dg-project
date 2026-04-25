from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.table1_rules import load_table1_rule
from geometry.sphere_flat_mesh import mesh_summary, sphere_flat_square_mesh
from geometry.sphere_flat_metrics import (
    build_sphere_flat_geometry_cache,
    geometry_diagnostics,
)
from operators.sdg_flattened_divergence import (
    build_table1_reference_diff_operators,
    divergence_stats_by_patch,
    sdg_flattened_cartesian_divergence,
)
from problems.sphere_advection import flattened_velocity_from_cache
from visualization.sdg_divergence_diagnostics import write_all_sdg_divergence_diagnostics


def parse_args():
    p = argparse.ArgumentParser(description="Volume-only SDG flattened Cartesian divergence diagnostic.")
    p.add_argument("--n-sub", type=int, default=2, help="Subdivisions per macro triangle.")
    p.add_argument("--order", type=int, default=3, help="Table1 order: 1,2,3,4.")
    p.add_argument("--N", type=int, default=None, help="Polynomial degree. Default: N=order.")
    p.add_argument("--R", type=float, default=1.0, help="Sphere radius. Flat square remains [-1,1]^2.")
    p.add_argument("--u0", type=float, default=1.0)
    p.add_argument("--alpha0", type=float, default=np.pi / 4.0)
    p.add_argument(
        "--q-case",
        type=str,
        default="constant",
        choices=["constant", "gaussian_flat", "cosine_bell_flat", "sinxy_flat", "sphere_z"],
    )
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def q_initial(case: str, cache, R: float = 1.0) -> np.ndarray:
    x = cache.x_flat
    y = cache.y_flat

    if case == "constant":
        return np.ones_like(x)

    if case == "gaussian_flat":
        x0 = 0.25
        y0 = -0.15
        sigma = 0.22
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))

    if case == "cosine_bell_flat":
        x0 = 0.35
        y0 = -0.20
        radius = 0.45
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        q = np.zeros_like(x)
        inside = r <= radius
        q[inside] = 0.5 * (1.0 + np.cos(np.pi * r[inside] / radius))
        return q

    if case == "sinxy_flat":
        return np.sin(np.pi * x) * np.cos(np.pi * y)

    if case == "sphere_z":
        return cache.Z / R

    raise ValueError(f"Unknown q case: {case}")


def _finite_stats(name: str, arr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    arr = np.asarray(arr, dtype=float)
    if mask is None:
        vals = arr[np.isfinite(arr)]
    else:
        vals = arr[(~mask) & np.isfinite(arr)]

    if vals.size == 0:
        return {
            "name": name,
            "n": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "linf": float("nan"),
        }

    return {
        "name": name,
        "n": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "linf": float(np.max(np.abs(vals))),
    }


def main():
    args = parse_args()
    N = args.order if args.N is None else args.N

    rule = load_table1_rule(args.order)
    rs_nodes = np.asarray(rule["rs"], dtype=float)

    # Flat square remains [-1,1]^2.
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

    Dr, Ds = build_table1_reference_diff_operators(rule, N=N)

    u1, u2, u_sph, v_sph = flattened_velocity_from_cache(
        cache,
        u0=args.u0,
        alpha0=args.alpha0,
    )

    q = q_initial(args.q_case, cache, R=args.R)

    div = sdg_flattened_cartesian_divergence(
        q=q,
        u1=u1,
        u2=u2,
        Dr=Dr,
        Ds=Ds,
        VX=VX,
        VY=VY,
        EToV=EToV,
        rs_nodes=rs_nodes,
        mask=cache.bad_mask,
    )

    outdir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "outputs" / "sdg_flattened_divergence" / (
            f"order{args.order}_N{N}_nsub{args.n_sub}_R{args.R:g}_{args.q_case}"
        )
    )

    paths = write_all_sdg_divergence_diagnostics(
        VX=VX,
        VY=VY,
        EToV=EToV,
        cache=cache,
        q=q,
        u1=u1,
        u2=u2,
        div=div,
        output_dir=outdir,
    )

    print("\n=== Mesh summary ===")
    for k, v in mesh_summary(VX, VY, EToV, elem_patch_id).items():
        print(f"{k}: {v}")

    print("\n=== SDG geometry diagnostics ===")
    for k, v in geometry_diagnostics(cache, R=args.R).items():
        print(f"{k}: {v}")

    print("\n=== Differentiation setup ===")
    print(f"Table1 order: {args.order}")
    print(f"Polynomial degree N: {N}")
    print(f"Np Table1 nodes: {rs_nodes.shape[0]}")
    print(f"Dr shape: {Dr.shape}")
    print(f"Ds shape: {Ds.shape}")

    print("\n=== Global field stats, excluding bad nodes ===")
    for item in [
        _finite_stats("q", q, cache.bad_mask),
        _finite_stats("u1", u1, cache.bad_mask),
        _finite_stats("u2", u2, cache.bad_mask),
        _finite_stats("div_flat", div, cache.bad_mask),
        _finite_stats("rhs=-div_flat", -div, cache.bad_mask),
    ]:
        print(item)

    print("\n=== Per-patch divergence stats ===")
    pp = divergence_stats_by_patch(div, cache.node_patch_id, mask=cache.bad_mask)
    for pid, d in pp.items():
        print(f"T{pid}: {d}")

    print("\n=== Output figures ===")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
