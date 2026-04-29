from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from geometry.sphere_manifold_mesh import (
    generate_spherical_octahedron_mesh,
    spherical_mesh_hmin,
)
from geometry.sphere_manifold_metrics import build_manifold_geometry_cache
from operators.manifold_rhs import (
    build_manifold_table1_k4_reference_operators,
    manifold_rhs_constant_field,
)
from problems.sphere_advection import solid_body_velocity_xyz


@dataclass(frozen=True)
class ManifoldDivHConvergenceConfig:
    mesh_levels: tuple[int, ...] = (2, 4, 8, 16, 32)
    R: float = 1.0
    u0: float = 1.0
    alpha0: float = -math.pi / 4.0
    verbose: bool = True


def compute_convergence_rate(errors: list[float], hs: list[float]) -> list[float]:
    rates = [math.nan]
    for i in range(1, len(errors)):
        e0 = errors[i - 1]
        e1 = errors[i]
        h0 = hs[i - 1]
        h1 = hs[i]
        if e0 <= 0.0 or e1 <= 0.0 or h0 <= 0.0 or h1 <= 0.0:
            rates.append(math.nan)
        else:
            rates.append(math.log(e0 / e1) / math.log(h0 / h1))
    return rates


def manifold_weighted_norms(
    values: np.ndarray,
    J: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    J = np.asarray(J, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if values.ndim != 2:
        raise ValueError("values must have shape (K, Np).")
    if J.shape != values.shape:
        raise ValueError("J must match values.")
    if weights.shape != (values.shape[1],):
        raise ValueError("weights must have shape (Np,).")

    good = np.isfinite(values) & np.isfinite(J)
    l2_sq = 0.0
    linf = 0.0

    for k in range(values.shape[0]):
        g = good[k]
        if not np.any(g):
            continue
        vk = values[k, g]
        wk = weights[g]
        jk = J[k, g]
        l2_sq += float(np.dot(wk, jk * vk * vk))
        linf = max(linf, float(np.max(np.abs(vk))))

    return {"L2": math.sqrt(l2_sq), "Linf": linf}


def manifold_weighted_mass(
    values: np.ndarray,
    J: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    J = np.asarray(J, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if values.ndim != 2:
        raise ValueError("values must have shape (K, Np).")
    if J.shape != values.shape:
        raise ValueError("J must match values.")
    if weights.shape != (values.shape[1],):
        raise ValueError("weights must have shape (Np,).")

    good = np.isfinite(values) & np.isfinite(J)
    mass = 0.0

    for k in range(values.shape[0]):
        g = good[k]
        if not np.any(g):
            continue
        vk = values[k, g]
        wk = weights[g]
        jk = J[k, g]
        mass += float(np.dot(wk, jk * vk))

    return mass


def run_manifold_div_h_convergence(
    config: ManifoldDivHConvergenceConfig,
) -> list[dict]:
    ref_ops = build_manifold_table1_k4_reference_operators()
    results: list[dict] = []

    for n_div in config.mesh_levels:
        t0 = perf_counter()
        nodes_xyz, EToV = generate_spherical_octahedron_mesh(
            n_div=n_div,
            R=config.R,
        )
        geom = build_manifold_geometry_cache(
            nodes_xyz=nodes_xyz,
            EToV=EToV,
            rs_nodes=ref_ops.rs_nodes,
            Dr=ref_ops.Dr,
            Ds=ref_ops.Ds,
            R=config.R,
        )

        U, V, W = solid_body_velocity_xyz(
            geom.X,
            geom.Y,
            geom.Z,
            u0=config.u0,
            alpha0=config.alpha0,
        )
        diag = manifold_rhs_constant_field(geom, U, V, W, ref_ops=ref_ops)
        rhs = np.asarray(diag["rhs"], dtype=float)
        div = np.asarray(diag["divergence"], dtype=float)

        rhs_norms = manifold_weighted_norms(rhs, geom.J, ref_ops.weights_2d)
        div_norms = manifold_weighted_norms(div, geom.J, ref_ops.weights_2d)
        h = spherical_mesh_hmin(nodes_xyz, EToV)

        row = {
            "n_div": int(n_div),
            "K": int(EToV.shape[0]),
            "Nv": int(nodes_xyz.shape[0]),
            "Np": int(ref_ops.rs_nodes.shape[0]),
            "total_dof": int(EToV.shape[0] * ref_ops.rs_nodes.shape[0]),
            "h": float(h),
            "L2": rhs_norms["L2"],
            "Linf": rhs_norms["Linf"],
            "L2_divergence": div_norms["L2"],
            "Linf_divergence": div_norms["Linf"],
            "max_surface_abs": float(diag["max_surface_abs"]),
            "elapsed_sec": float(perf_counter() - t0),
        }
        results.append(row)

        if config.verbose:
            print(
                f"[manifold div h-study] n_div={n_div:3d} | "
                f"K={row['K']:6d} | h={row['h']:.6e} | "
                f"L2={row['L2']:.6e} | Linf={row['Linf']:.6e} | "
                f"surface={row['max_surface_abs']:.3e} | "
                f"time={row['elapsed_sec']:.2f}s"
            )

    hs = [r["h"] for r in results]
    l2_rates = compute_convergence_rate([r["L2"] for r in results], hs)
    linf_rates = compute_convergence_rate([r["Linf"] for r in results], hs)

    for row, r_l2, r_linf in zip(results, l2_rates, linf_rates):
        row["rate_L2"] = r_l2
        row["rate_Linf"] = r_linf

    return results


def print_results_table(results: list[dict]) -> None:
    header = (
        f"{'n_div':>6s} {'K':>9s} {'h':>12s} "
        f"{'L2':>14s} {'rate':>8s} "
        f"{'Linf':>14s} {'rate':>8s} {'surf':>12s}"
    )
    print(header)
    print("-" * len(header))

    def fmt_rate(v):
        return "   -   " if not np.isfinite(v) else f"{v:8.3f}"

    for row in results:
        print(
            f"{row['n_div']:6d} {row['K']:9d} {row['h']:12.4e} "
            f"{row['L2']:14.6e} {fmt_rate(row['rate_L2'])} "
            f"{row['Linf']:14.6e} {fmt_rate(row['rate_Linf'])} "
            f"{row['max_surface_abs']:12.4e}"
        )


def save_results_csv(results: list[dict], filepath: str | Path) -> None:
    if not results:
        raise ValueError("results is empty.")

    with Path(filepath).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
