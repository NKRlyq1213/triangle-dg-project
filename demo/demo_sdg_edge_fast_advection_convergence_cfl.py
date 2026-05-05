from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from time_integration.CFL import mesh_min_altitude, cfl_dt_from_h

from operators.sdg_edge_streamfunction_closed_sphere_rhs import (
    build_edge_streamfunction_closed_sphere_sdg_operator,
    edge_flux_pair_summary,
)
from operators.sdg_edge_streamfunction_fast_rhs import (
    build_fast_edge_rhs_cache,
    edge_streamfunction_closed_sphere_rhs_global_corrected_fast,
)
from operators.sdg_streamfunction_closed_sphere_rhs import (
    mass,
    rhs_mass_residual,
    weighted_l2_linf,
    seam_summary,
)


RK4A = np.array(
    [
        0.0,
        -567301805773.0 / 1357537059087.0,
        -2404267990393.0 / 2016746695238.0,
        -3550918686646.0 / 2091501179385.0,
        -1275806237668.0 / 842570457699.0,
    ],
    dtype=float,
)

RK4B = np.array(
    [
        1432997174477.0 / 9575080441755.0,
        5161836677717.0 / 13612068292357.0,
        1720146321549.0 / 2090206949498.0,
        3134564353537.0 / 4481467310338.0,
        2277821191437.0 / 14882151754819.0,
    ],
    dtype=float,
)


def parse_alpha_expr(expr: str) -> float:
    s = expr.strip().lower().replace(" ", "")
    if s == "0":
        return 0.0
    if s == "pi":
        return math.pi
    if s == "-pi":
        return -math.pi
    if s.startswith("pi/"):
        return math.pi / float(s.split("/", 1)[1])
    if s.startswith("-pi/"):
        return -math.pi / float(s.split("/", 1)[1])
    return float(s)


def parse_alpha_list(text: str) -> list[float]:
    return [parse_alpha_expr(part) for part in text.split(",") if part.strip()]


def solid_body_omega(alpha0: float, u0: float) -> np.ndarray:
    return np.array(
        [-math.sin(alpha0) * u0, 0.0, math.cos(alpha0) * u0],
        dtype=float,
    )


def rotate_points_by_axis_angle(X, Y, Z, *, axis: np.ndarray, angle: float):
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        return np.array(X, copy=True), np.array(Y, copy=True), np.array(Z, copy=True)

    k = axis / norm
    kx, ky, kz = k

    c = math.cos(angle)
    s = math.sin(angle)

    dot = kx * X + ky * Y + kz * Z

    cross_x = ky * Z - kz * Y
    cross_y = kz * X - kx * Z
    cross_z = kx * Y - ky * X

    Xr = X * c + cross_x * s + kx * dot * (1.0 - c)
    Yr = Y * c + cross_y * s + ky * dot * (1.0 - c)
    Zr = Z * c + cross_z * s + kz * dot * (1.0 - c)

    return Xr, Yr, Zr


def backward_characteristic_points(op, t: float):
    base = op.base
    omega = solid_body_omega(base.alpha0, base.u0)
    omega_norm = float(np.linalg.norm(omega))

    if omega_norm == 0.0:
        return base.cache.X, base.cache.Y, base.cache.Z

    axis = omega / omega_norm
    angle = -omega_norm * float(t)

    return rotate_points_by_axis_angle(
        base.cache.X,
        base.cache.Y,
        base.cache.Z,
        axis=axis,
        angle=angle,
    )


def spherical_gaussian_from_points(X, Y, Z, *, R: float, sigma: float, center: np.ndarray):
    center = np.asarray(center, dtype=float)
    center = center / np.linalg.norm(center) * R

    dot = (X * center[0] + Y * center[1] + Z * center[2]) / (R * R)
    dot = np.clip(dot, -1.0, 1.0)

    dist = R * np.arccos(dot)

    return np.exp(-(dist * dist) / (2.0 * sigma * sigma))


def exact_state(case: str, op, t: float, *, sigma: float = 0.35):
    case = case.lower().strip()
    base = op.base

    X0, Y0, Z0 = backward_characteristic_points(op, t)

    if case == "constant":
        return np.ones_like(X0)

    if case == "sphere_x":
        return X0 / base.R

    if case == "sphere_y":
        return Y0 / base.R

    if case == "sphere_z":
        return Z0 / base.R

    if case == "gaussian":
        return spherical_gaussian_from_points(
            X0,
            Y0,
            Z0,
            R=base.R,
            sigma=sigma,
            center=np.array([1.0, 0.0, 0.0], dtype=float),
        )

    raise ValueError("Supported q_case: constant, sphere_x, sphere_y, sphere_z, gaussian.")


def flattened_vmax(op) -> float:
    """
    Estimate vmax in flattened equal-area coordinates.

    PDE:
        q_t + (1/J) div(F_area q) = 0

    so the effective flat velocity is:
        u_eff = F_area / J.
    """
    base = op.base
    ux = base.Fx_area / base.J_area
    uy = base.Fy_area / base.J_area
    return float(np.nanmax(np.sqrt(ux * ux + uy * uy)))


def choose_cfl_dt_and_steps(op, *, cfl: float, final_time: float):
    base = op.base

    hmin = mesh_min_altitude(
        base.VX,
        base.VY,
        base.EToV,
    )

    vmax = flattened_vmax(op)

    dt_nominal = cfl_dt_from_h(
        cfl=float(cfl),
        h=float(hmin),
        N=int(base.N),
        vmax=float(vmax),
    )

    nsteps = int(math.ceil(float(final_time) / dt_nominal))
    nsteps = max(nsteps, 1)

    dt_eff = float(final_time) / nsteps

    return float(dt_eff), int(nsteps), float(dt_nominal), float(hmin), float(vmax)


def run_lsrk_fast(q0, op, cache, *, dt: float, steps: int):
    q = np.array(q0, dtype=float, copy=True)
    res = np.zeros_like(q)

    max_stage_mass_rhs_abs = 0.0
    max_stage_rhs_l2 = 0.0
    max_stage_corr = 0.0

    t0 = time.perf_counter()

    for _ in range(steps):
        for a, b in zip(RK4A, RK4B):
            rhs, info = edge_streamfunction_closed_sphere_rhs_global_corrected_fast(
                q,
                op,
                fast_cache=cache,
                return_info=True,
            )

            m_rhs = rhs_mass_residual(rhs, op.base)
            rhs_l2, _ = weighted_l2_linf(rhs, op.base)

            max_stage_mass_rhs_abs = max(max_stage_mass_rhs_abs, abs(m_rhs))
            max_stage_rhs_l2 = max(max_stage_rhs_l2, rhs_l2)
            max_stage_corr = max(max_stage_corr, abs(info["global_correction_constant"]))

            res = a * res + dt * rhs
            q = q + b * res

    t1 = time.perf_counter()

    return {
        "q": q,
        "elapsed_s": float(t1 - t0),
        "max_stage_mass_rhs_abs": float(max_stage_mass_rhs_abs),
        "max_stage_rhs_l2": float(max_stage_rhs_l2),
        "max_stage_global_correction": float(max_stage_corr),
    }


def error_stats(q_num, q_ex, op):
    err = q_num - q_ex
    l2, linf = weighted_l2_linf(err, op.base)
    q_l2, _ = weighted_l2_linf(q_ex, op.base)
    rel_l2 = l2 / max(q_l2, 1.0e-300)
    return l2, linf, rel_l2


def run_one_case(
    *,
    nsub: int,
    order: int,
    N: int,
    R: float,
    u0: float,
    alpha0: float,
    tau: float,
    q_case: str,
    seam_tol: float,
    final_time: float,
    cfl: float,
    sigma: float,
):
    op = build_edge_streamfunction_closed_sphere_sdg_operator(
        nsub=nsub,
        order=order,
        N=N,
        R=R,
        u0=u0,
        alpha0=alpha0,
        tau=tau,
        seam_tol=seam_tol,
    )

    cache = build_fast_edge_rhs_cache(op)

    dt, steps, dt_nominal, hmin, vmax = choose_cfl_dt_and_steps(
        op,
        cfl=cfl,
        final_time=final_time,
    )

    q0 = exact_state(q_case, op, 0.0, sigma=sigma)
    q_exact_T = exact_state(q_case, op, final_time, sigma=sigma)

    # compile / warm-up
    _rhs0, info0 = edge_streamfunction_closed_sphere_rhs_global_corrected_fast(
        q0,
        op,
        fast_cache=cache,
        return_info=True,
    )

    m0 = mass(q0, op.base)
    mex = mass(q_exact_T, op.base)

    out = run_lsrk_fast(
        q0,
        op,
        cache,
        dt=dt,
        steps=steps,
    )

    qT = out["q"]
    mT = mass(qT, op.base)

    l2_err, linf_err, rel_l2_err = error_stats(qT, q_exact_T, op)

    seam = seam_summary(op.base)
    pair = edge_flux_pair_summary(q0, op)

    return {
        "nsub": int(nsub),
        "hmin": float(hmin),
        "h_proxy": float(1.0 / nsub),
        "order": int(order),
        "N": int(N),
        "R": float(R),
        "u0": float(u0),
        "vmax": float(vmax),
        "alpha0": float(alpha0),
        "alpha0_over_pi": float(alpha0 / math.pi),
        "tau": float(tau),
        "cfl": float(cfl),
        "q_case": q_case,
        "sigma": float(sigma),

        "steps": int(steps),
        "dt": float(dt),
        "dt_nominal": float(dt_nominal),
        "final_time": float(final_time),
        "elapsed_s": float(out["elapsed_s"]),

        "L2_error": float(l2_err),
        "Linf_error": float(linf_err),
        "relative_L2_error": float(rel_l2_err),

        "mass0": float(m0),
        "mass_exact_T": float(mex),
        "massT": float(mT),
        "mass_drift": float(mT - m0),
        "exact_mass_drift": float(mex - m0),

        "max_stage_mass_rhs_abs": float(out["max_stage_mass_rhs_abs"]),
        "max_stage_rhs_L2": float(out["max_stage_rhs_l2"]),
        "max_stage_global_correction": float(out["max_stage_global_correction"]),

        "rhs0_global_correction": float(info0["global_correction_constant"]),

        "n_seam_pairs": int(seam["n_seam_pairs"]),
        "n_unmatched_boundary_faces": int(seam["n_unmatched_boundary_faces"]),
        "max_seam_match_error": float(seam["max_seam_match_error"]),
        "edge_pair_error_initial": float(pair["edge_flux_pair_max_error"]),

        "L2_rate": float("nan"),
        "Linf_rate": float("nan"),
        "relative_L2_rate": float("nan"),
    }


def add_rates(rows: list[dict]) -> None:
    groups = {}
    for r in rows:
        key = (r["alpha0"], r["q_case"], r["cfl"], r["final_time"])
        groups.setdefault(key, []).append(r)

    for group in groups.values():
        group.sort(key=lambda x: x["nsub"])
        prev = None
        for r in group:
            if prev is not None:
                ratio = float(r["nsub"]) / float(prev["nsub"])

                if r["L2_error"] > 0.0 and prev["L2_error"] > 0.0:
                    r["L2_rate"] = math.log(prev["L2_error"] / r["L2_error"]) / math.log(ratio)

                if r["Linf_error"] > 0.0 and prev["Linf_error"] > 0.0:
                    r["Linf_rate"] = math.log(prev["Linf_error"] / r["Linf_error"]) / math.log(ratio)

                if r["relative_L2_error"] > 0.0 and prev["relative_L2_error"] > 0.0:
                    r["relative_L2_rate"] = math.log(prev["relative_L2_error"] / r["relative_L2_error"]) / math.log(ratio)

            prev = r


def write_csv(rows, path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'nsub':>6s} "
        f"{'alpha/pi':>10s} "
        f"{'cfl':>7s} "
        f"{'steps':>7s} "
        f"{'dt':>10s} "
        f"{'L2_err':>12s} "
        f"{'rate':>7s} "
        f"{'Linf_err':>12s} "
        f"{'rate':>7s} "
        f"{'massDrift':>12s} "
        f"{'time_s':>9s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['cfl']:7.3f} "
            f"{r['steps']:7d} "
            f"{r['dt']:10.4e} "
            f"{r['L2_error']:12.4e} "
            f"{r['L2_rate']:7.3f} "
            f"{r['Linf_error']:12.4e} "
            f"{r['Linf_rate']:7.3f} "
            f"{r['mass_drift']:12.4e} "
            f"{r['elapsed_s']:9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFL-controlled fast edge-streamfunction advection convergence test."
    )

    parser.add_argument("--nsubs", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--seam-tol", type=float, default=1.0e-10)

    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--cfl", type=float, default=0.5)

    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--alphas", type=str, default="pi/4")

    parser.add_argument(
        "--q-case",
        type=str,
        default="gaussian",
        choices=["constant", "sphere_x", "sphere_y", "sphere_z", "gaussian"],
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_edge_fast_advection_convergence_cfl"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG edge-streamfunction CFL advection convergence ===")
    print(f"order      = {args.order}")
    print(f"N          = {N}")
    print(f"R          = {args.R}")
    print(f"u0         = {args.u0}")
    print(f"tau        = {args.tau}")
    print(f"q_case     = {args.q_case}")
    print(f"sigma      = {args.sigma}")
    print(f"final_time = {args.final_time}")
    print(f"cfl        = {args.cfl}")
    print(f"alphas     = {alphas}")
    print()

    rows = []
    for alpha0 in alphas:
        for nsub in args.nsubs:
            rows.append(
                run_one_case(
                    nsub=nsub,
                    order=args.order,
                    N=N,
                    R=args.R,
                    u0=args.u0,
                    alpha0=alpha0,
                    tau=args.tau,
                    q_case=args.q_case,
                    seam_tol=args.seam_tol,
                    final_time=args.final_time,
                    cfl=args.cfl,
                    sigma=args.sigma,
                )
            )

    add_rates(rows)
    print_table(rows)

    out = Path(args.output_dir)
    csv_path = out / (
        f"edge_fast_advection_convergence_cfl_{args.q_case}_T{args.final_time:g}_cfl{args.cfl:g}_tau{args.tau:g}.csv"
    )
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
