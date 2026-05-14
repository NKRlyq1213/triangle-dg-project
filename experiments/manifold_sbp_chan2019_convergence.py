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
from geometry.sphere_projected_exact_metrics import build_exact_projected_sphere_geometry_cache
from operators.manifold_rhs import build_manifold_face_connectivity
from operators.manifold_sbp_chan2019 import (
    build_manifold_sbp_chan_table1_k4_reference_operators,
    constant_state_rhs_diagnostic,
    manifold_sbp_chan2019_rhs_exchange,
)
from problems.sphere_advection import (
    constant_field_xyz,
    exact_gaussian_bell_xyz,
    solid_body_velocity_xyz,
)
from time_integration.CFL import cfl_dt_from_h
from time_integration.lsrk54 import integrate_lsrk54, is_tf_reached

from experiments.manifold_div_h_convergence import (
    compute_convergence_rate,
    manifold_weighted_mass,
    manifold_weighted_norms,
)


_VALID_FIELD_CASES = {"gaussian", "constant"}
_VALID_FLUX_TYPES = {"upwind", "central", "lax_friedrichs"}


@dataclass(frozen=True)
class ManifoldSBPChan2019Config:
    mesh_levels: tuple[int, ...] = (2, 4, 8)
    R: float = 1.0
    u0: float = 1.0
    alpha0: float = math.pi / 4.0
    cfl: float = 0.5
    tf: float = 1.0
    N: int = 4
    gaussian_width: float = 1.0 / math.sqrt(10.0)
    field_case: str = "gaussian"
    constant_value: float = 1.0
    flux_type: str = "upwind"
    alpha_lf: float = 1.0
    use_numba: bool | None = True
    record_history: bool = False
    verbose: bool = True


def _validate_config(config: ManifoldSBPChan2019Config) -> None:
    if not config.mesh_levels:
        raise ValueError("mesh_levels must not be empty.")
    if any(int(level) < 1 for level in config.mesh_levels):
        raise ValueError("mesh_levels must contain positive integers.")
    if config.R <= 0.0:
        raise ValueError("R must be positive.")
    if config.cfl <= 0.0:
        raise ValueError("cfl must be positive.")
    if config.tf < 0.0:
        raise ValueError("tf must be non-negative.")
    if config.N != 4:
        raise ValueError("This implementation is fixed to Table1 k=4 / N=4.")
    if config.gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be positive.")
    if str(config.field_case).lower() not in _VALID_FIELD_CASES:
        raise ValueError("field_case must be one of: gaussian, constant.")
    if str(config.flux_type).lower() not in _VALID_FLUX_TYPES:
        raise ValueError("flux_type must be one of: upwind, central, lax_friedrichs.")
    if config.alpha_lf <= 0.0:
        raise ValueError("alpha_lf must be positive.")


def _field_case(config: ManifoldSBPChan2019Config) -> str:
    return str(config.field_case).strip().lower()


def _flux_type(config: ManifoldSBPChan2019Config) -> str:
    return str(config.flux_type).strip().lower()


def _max_speed(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> float:
    return float(np.max(np.sqrt(U * U + V * V + W * W)))


def _make_reference_field_getter(geom, config: ManifoldSBPChan2019Config):
    field_case = _field_case(config)
    if field_case == "gaussian":
        center_xyz = (0.0, config.R, 0.0)

        def q_ref(t: float) -> np.ndarray:
            return exact_gaussian_bell_xyz(
                geom.X,
                geom.Y,
                geom.Z,
                t=float(t),
                u0=config.u0,
                R=config.R,
                alpha0=config.alpha0,
                width=config.gaussian_width,
                center_xyz=center_xyz,
            )

        return q_ref(0.0), q_ref

    def q_ref(t: float) -> np.ndarray:
        del t
        return constant_field_xyz(geom.X, geom.Y, geom.Z, value=config.constant_value)

    return q_ref(0.0), q_ref


def _mass_relative_error(mass: float, mass0: float) -> float:
    if abs(mass0) <= np.finfo(float).tiny:
        return math.nan
    return float(mass - mass0) / float(mass0)


def _record_error(q, t, q_ref_getter, geom, weights_2d) -> tuple[np.ndarray, dict[str, float]]:
    q_ref = np.asarray(q_ref_getter(float(t)), dtype=float)
    err = np.asarray(q, dtype=float) - q_ref
    norms = manifold_weighted_norms(err, geom.J, weights_2d)
    norms["max_abs_error"] = float(np.max(np.abs(err)))
    return q_ref, norms


def _run_one_level(
    config: ManifoldSBPChan2019Config,
    n_div: int,
    ref_ops,
) -> dict:
    t_start = perf_counter()

    nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=n_div, R=config.R)
    geom = build_exact_projected_sphere_geometry_cache(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=ref_ops.rs_nodes,
        R=config.R,
    )
    conn = build_manifold_face_connectivity(EToV)

    U, V, W = solid_body_velocity_xyz(
        geom.X,
        geom.Y,
        geom.Z,
        u0=config.u0,
        alpha0=config.alpha0,
    )
    velocity_xyz = (U, V, W)
    vmax = _max_speed(U, V, W)
    h = spherical_mesh_hmin(nodes_xyz, EToV)
    dt = cfl_dt_from_h(cfl=config.cfl, h=h, N=config.N + 1, vmax=vmax)

    q0, q_ref_getter = _make_reference_field_getter(geom, config)
    mass0 = manifold_weighted_mass(q0, geom.J, ref_ops.weights_2d)

    if config.verbose:
        diag = constant_state_rhs_diagnostic(
            geom=geom,
            velocity_xyz=velocity_xyz,
            ref_ops=ref_ops,
            conn=conn,
            flux_type=config.flux_type,
            use_numba=config.use_numba,
        )
        print(
            f"[SBP Chan2019] n_div={n_div} constant RHS Linf="
            f"{float(diag['max_rhs_abs']):.6e}, massRHS={float(diag['weighted_rhs_mass']):.6e}"
        )

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        return manifold_sbp_chan2019_rhs_exchange(
            q=q,
            geom=geom,
            velocity_xyz=velocity_xyz,
            ref_ops=ref_ops,
            conn=conn,
            flux_type=config.flux_type,
            alpha_lf=config.alpha_lf,
            global_vmax=vmax,
            t=t,
            use_numba=config.use_numba,
        )

    step_ids: list[int] = []
    times: list[float] = []
    l2_errors: list[float] = []
    linf_errors: list[float] = []
    masses: list[float] = []
    mass_errors: list[float] = []
    mass_rel_errors: list[float] = []

    if config.record_history:
        _, norms0 = _record_error(q0, 0.0, q_ref_getter, geom, ref_ops.weights_2d)
        step_ids.append(0)
        times.append(0.0)
        l2_errors.append(float(norms0["L2"]))
        linf_errors.append(float(norms0["Linf"]))
        masses.append(float(mass0))
        mass_errors.append(0.0)
        mass_rel_errors.append(0.0)

    def _post_step_transform(t_step: float, q_step: np.ndarray) -> np.ndarray:
        _, norms_step = _record_error(q_step, t_step, q_ref_getter, geom, ref_ops.weights_2d)
        mass_step = float(manifold_weighted_mass(q_step, geom.J, ref_ops.weights_2d))
        step_ids.append(len(step_ids))
        times.append(float(t_step))
        l2_errors.append(float(norms_step["L2"]))
        linf_errors.append(float(norms_step["Linf"]))
        masses.append(mass_step)
        mass_errors.append(float(mass_step - mass0))
        mass_rel_errors.append(float(_mass_relative_error(mass_step, mass0)))
        return q_step

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=0.0,
        tf=config.tf,
        dt=dt,
        post_step_transform=_post_step_transform if config.record_history else None,
    )

    _, norms_final = _record_error(qf, tf_used, q_ref_getter, geom, ref_ops.weights_2d)
    mass_final = float(manifold_weighted_mass(qf, geom.J, ref_ops.weights_2d))

    row = {
        "n_div": int(n_div),
        "K": int(EToV.shape[0]),
        "Nv": int(nodes_xyz.shape[0]),
        "Np": int(ref_ops.rs_nodes.shape[0]),
        "total_dof": int(EToV.shape[0] * ref_ops.rs_nodes.shape[0]),
        "h": float(h),
        "dt": float(dt),
        "tf_target": float(config.tf),
        "tf": float(tf_used),
        "reached_tf": bool(is_tf_reached(tf_used, config.tf)),
        "nsteps": int(nsteps),
        "field_case": _field_case(config),
        "flux_type": _flux_type(config),
        "alpha_lf": float(config.alpha_lf),
        "mass0": float(mass0),
        "mass": float(mass_final),
        "mass_error": float(mass_final - mass0),
        "mass_rel_error": float(_mass_relative_error(mass_final, mass0)),
        "L2_error": float(norms_final["L2"]),
        "Linf_error": float(norms_final["Linf"]),
        "max_abs_error": float(norms_final["max_abs_error"]),
        "elapsed_sec": float(perf_counter() - t_start),
    }

    if config.record_history:
        row["history"] = {
            "mesh_level": int(n_div),
            "h": float(h),
            "field_case": _field_case(config),
            "flux_type": _flux_type(config),
            "alpha_lf": float(config.alpha_lf),
            "mass0": float(mass0),
            "step_ids": np.asarray(step_ids, dtype=int),
            "times": np.asarray(times, dtype=float),
            "l2": np.asarray(l2_errors, dtype=float),
            "linf": np.asarray(linf_errors, dtype=float),
            "mass": np.asarray(masses, dtype=float),
            "mass_error": np.asarray(mass_errors, dtype=float),
            "mass_rel_error": np.asarray(mass_rel_errors, dtype=float),
            "reached_tf": bool(row["reached_tf"]),
            "tf_used": float(tf_used),
            "nsteps": int(nsteps),
        }

    if config.verbose:
        print(
            f"[SBP Chan2019] field={row['field_case']:>8s} flux={row['flux_type']:>14s} "
            f"n_div={row['n_div']:3d} K={row['K']:6d} h={row['h']:.6e} "
            f"dt={row['dt']:.3e} steps={row['nsteps']:6d} "
            f"L2={row['L2_error']:.6e} Linf={row['Linf_error']:.6e} "
            f"mass_err={row['mass_error']:.3e} time={row['elapsed_sec']:.2f}s"
        )

    return row


def run_manifold_sbp_chan2019_convergence(
    config: ManifoldSBPChan2019Config,
) -> list[dict]:
    _validate_config(config)
    ref_ops = build_manifold_sbp_chan_table1_k4_reference_operators()
    results = [_run_one_level(config, int(level), ref_ops) for level in config.mesh_levels]

    hs = [float(row["h"]) for row in results]
    l2_rates = compute_convergence_rate([float(row["L2_error"]) for row in results], hs)
    linf_rates = compute_convergence_rate([float(row["Linf_error"]) for row in results], hs)

    for row, rate_l2, rate_linf in zip(results, l2_rates, linf_rates):
        row["rate_L2"] = float(rate_l2)
        row["rate_Linf"] = float(rate_linf)

    return results


def extract_time_histories(results: list[dict]) -> list[dict]:
    return [row["history"] for row in results if isinstance(row.get("history"), dict)]


def print_results_table(results: list[dict]) -> None:
    header = (
        f"{'field':>10s} {'flux':>14s} {'n_div':>6s} {'K':>9s} "
        f"{'h':>12s} {'dt':>12s} {'steps':>8s} "
        f"{'L2':>14s} {'rate':>8s} {'Linf':>14s} {'rate':>8s} {'mass_err':>12s}"
    )
    print(header)
    print("-" * len(header))

    def fmt_rate(value: float) -> str:
        return "   -   " if not np.isfinite(value) else f"{value:8.3f}"

    for row in results:
        print(
            f"{str(row['field_case']):>10s} {str(row['flux_type']):>14s} "
            f"{row['n_div']:6d} {row['K']:9d} {row['h']:12.4e} "
            f"{row['dt']:12.4e} {row['nsteps']:8d} "
            f"{row['L2_error']:14.6e} {fmt_rate(float(row['rate_L2']))} "
            f"{row['Linf_error']:14.6e} {fmt_rate(float(row['rate_Linf']))} "
            f"{row['mass_error']:12.4e}"
        )


def _summary_fieldnames(results: list[dict]) -> list[str]:
    return [
        key for key, value in results[0].items()
        if not isinstance(value, (dict, list, tuple, np.ndarray))
    ]


def save_results_csv(results: list[dict], filepath: str | Path) -> None:
    if not results:
        raise ValueError("results is empty.")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _summary_fieldnames(results)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row[name] for name in fieldnames})


def save_time_history_csv(results: list[dict], filepath: str | Path) -> None:
    histories = extract_time_histories(results)
    if not histories:
        raise ValueError("results does not contain recorded histories.")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mesh_level",
        "h",
        "field_case",
        "flux_type",
        "alpha_lf",
        "mass0",
        "step_index",
        "time",
        "L2_error",
        "Linf_error",
        "mass",
        "mass_error",
        "mass_rel_error",
        "reached_tf",
        "tf_used",
        "nsteps",
    ]
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for history in histories:
            times = np.asarray(history["times"], dtype=float)
            for i in range(times.size):
                writer.writerow(
                    {
                        "mesh_level": int(history["mesh_level"]),
                        "h": float(history["h"]),
                        "field_case": str(history["field_case"]),
                        "flux_type": str(history["flux_type"]),
                        "alpha_lf": float(history["alpha_lf"]),
                        "mass0": float(history["mass0"]),
                        "step_index": int(history["step_ids"][i]),
                        "time": float(times[i]),
                        "L2_error": float(history["l2"][i]),
                        "Linf_error": float(history["linf"][i]),
                        "mass": float(history["mass"][i]),
                        "mass_error": float(history["mass_error"][i]),
                        "mass_rel_error": float(history["mass_rel_error"][i]),
                        "reached_tf": bool(history["reached_tf"]),
                        "tf_used": float(history["tf_used"]),
                        "nsteps": int(history["nsteps"]),
                    }
                )

