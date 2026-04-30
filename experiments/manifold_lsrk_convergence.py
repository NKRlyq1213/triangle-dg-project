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
    build_manifold_exchange_cache,
    build_manifold_table1_k4_reference_operators,
    manifold_rhs_exchange,
)
from problems.sphere_advection import (
    constant_field_xyz,
    exact_gaussian_bell_xyz,
    solid_body_velocity_xyz,
)
from time_integration.CFL import cfl_dt_from_h
from time_integration.lsrk54 import integrate_lsrk54, is_tf_reached, tf_align_tolerance

from experiments.manifold_div_h_convergence import (
    compute_convergence_rate,
    manifold_weighted_norms,
    manifold_weighted_mass,
)


_VALID_FIELD_CASES = {"gaussian", "constant"}
_VALID_INITIAL_PRESETS = {
    "custom",
    "equator",
    "equator_x",
    "equator_y",
    "north_pole",
    "south_pole",
}
_VALID_FLUX_TYPES = {"upwind", "central", "lax_friedrichs"}
_VALID_VOLUME_FORMS = {"split", "conservative"}
_VALID_SURFACE_ASSEMBLIES = {"local_side", "single_edge"}


@dataclass(frozen=True)
class ManifoldLSRKConvergenceConfig:
    mesh_levels: tuple[int, ...] = (2, 4, 8,)
    R: float = 1.0
    u0: float = 1.0
    alpha0: float = -math.pi / 4.0
    center_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0)
    initial_preset: str = "custom"
    cfl: float = 0.25
    tf: float = 1.0
    N: int = 4
    gaussian_width: float = 1.0 / math.sqrt(10.0)
    field_case: str = "gaussian"
    constant_value: float = 1.0
    flux_type: str = "upwind"
    alpha_lf: float = 1.0
    volume_form: str = "split"
    surface_assembly: str = "local_side"
    use_numba: bool | None = True
    record_history: bool = False
    record_step_snapshots: bool = False
    snapshot_times: tuple[float, ...] = ()
    verbose: bool = True


def _validate_config(config: ManifoldLSRKConvergenceConfig) -> None:
    if len(config.mesh_levels) == 0:
        raise ValueError("mesh_levels must not be empty.")
    if any(int(n) < 1 for n in config.mesh_levels):
        raise ValueError("All mesh_levels must be positive.")
    if config.R <= 0.0:
        raise ValueError("R must be positive.")
    if config.cfl <= 0.0:
        raise ValueError("cfl must be positive.")
    if config.tf < 0.0:
        raise ValueError("tf must be non-negative.")
    if config.N != 4:
        raise ValueError("Manifold LSRK study is fixed to Table1 k=4 and N=4.")
    if config.gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be positive.")
    flux_type = _normalize_flux_type(config.flux_type)
    volume_form = _normalize_volume_form(config.volume_form)
    surface_assembly = _normalize_surface_assembly(config.surface_assembly)
    center_xyz = np.asarray(config.center_xyz, dtype=float).reshape(-1)
    if center_xyz.shape != (3,):
        raise ValueError("center_xyz must contain exactly three values.")
    if not np.all(np.isfinite(center_xyz)):
        raise ValueError("center_xyz must be finite.")
    field_case = str(config.field_case).strip().lower()
    if field_case not in _VALID_FIELD_CASES:
        raise ValueError("field_case must be one of: gaussian, constant.")
    initial_preset = _normalize_initial_preset(config.initial_preset)
    if initial_preset not in _VALID_INITIAL_PRESETS:
        raise ValueError(
            "initial_preset must be one of: custom, equator, equator_x, equator_y, north_pole, south_pole."
        )
    if initial_preset == "custom" and float(np.linalg.norm(center_xyz)) <= 0.0:
        raise ValueError("center_xyz must not be the zero vector when initial_preset='custom'.")
    if not np.isfinite(float(config.constant_value)):
        raise ValueError("constant_value must be finite.")
    if config.alpha_lf <= 0.0:
        raise ValueError("alpha_lf must be positive.")
    if flux_type == "lax_friedrichs" and config.alpha_lf < 1.0:
        raise ValueError("alpha_lf must be at least 1.0 for lax_friedrichs flux.")
    if volume_form == "conservative" and surface_assembly == "local_side":
        pass
    for t_snap in config.snapshot_times:
        if float(t_snap) < 0.0 or float(t_snap) > float(config.tf):
            raise ValueError("snapshot_times must satisfy 0 <= t <= tf.")


def _normalize_field_case(field_case: str) -> str:
    return str(field_case).strip().lower()


def _normalize_flux_type(flux_type: str) -> str:
    flux = str(flux_type).strip().lower()
    if flux not in _VALID_FLUX_TYPES:
        raise ValueError("flux_type must be one of: upwind, central, lax_friedrichs.")
    return flux


def _normalize_volume_form(volume_form: str) -> str:
    val = str(volume_form).strip().lower()
    if val not in _VALID_VOLUME_FORMS:
        raise ValueError("volume_form must be one of: split, conservative.")
    return val


def _normalize_surface_assembly(surface_assembly: str) -> str:
    val = str(surface_assembly).strip().lower()
    if val not in _VALID_SURFACE_ASSEMBLIES:
        raise ValueError("surface_assembly must be one of: local_side, single_edge.")
    return val


def _normalize_snapshot_times(snapshot_times: tuple[float, ...]) -> tuple[float, ...]:
    if not snapshot_times:
        return ()
    return tuple(sorted({float(t) for t in snapshot_times}))


def _normalize_initial_preset(initial_preset: str) -> str:
    return str(initial_preset).strip().lower()


def _resolve_initial_center(config: ManifoldLSRKConvergenceConfig) -> tuple[float, float, float]:
    preset = _normalize_initial_preset(config.initial_preset)
    R = float(config.R)
    if preset == "custom":
        center = np.asarray(config.center_xyz, dtype=float).reshape(3)
        center_norm = float(np.linalg.norm(center))
        if center_norm == 0.0:
            raise ValueError("center_xyz must not be the zero vector when initial_preset='custom'.")
        center = R * center / center_norm
        return float(center[0]), float(center[1]), float(center[2])
    if preset in {"equator", "equator_y"}:
        return 0.0, R, 0.0
    if preset == "equator_x":
        return R, 0.0, 0.0
    if preset == "north_pole":
        return 0.0, 0.0, R
    if preset == "south_pole":
        return 0.0, 0.0, -R
    raise ValueError(f"Unsupported initial preset: {preset}")


def _max_speed(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> float:
    return float(np.max(np.sqrt(U * U + V * V + W * W)))


def _dt_speed_scale(config: ManifoldLSRKConvergenceConfig) -> float:
    flux_type = _normalize_flux_type(config.flux_type)
    if flux_type == "lax_friedrichs":
        return float(config.alpha_lf)
    return 1.0


def _field_case_error_label(field_case: str) -> str:
    if field_case == "gaussian":
        return "exact_gaussian"
    return "constant_drift"


def _make_reference_field_getter(geom, config: ManifoldLSRKConvergenceConfig):
    field_case = _normalize_field_case(config.field_case)
    center_xyz = _resolve_initial_center(config)
    if field_case == "gaussian":
        q0 = exact_gaussian_bell_xyz(
            geom.X,
            geom.Y,
            geom.Z,
            t=0.0,
            u0=config.u0,
            R=config.R,
            alpha0=config.alpha0,
            width=config.gaussian_width,
            center_xyz=center_xyz,
        )

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

        return q0, q_ref

    q0 = constant_field_xyz(
        geom.X,
        geom.Y,
        geom.Z,
        value=config.constant_value,
    )

    def q_ref(t: float) -> np.ndarray:
        del t
        return constant_field_xyz(
            geom.X,
            geom.Y,
            geom.Z,
            value=config.constant_value,
        )

    return q0, q_ref


def _compute_error_state(
    q: np.ndarray,
    t: float,
    q_ref_getter,
    geom,
    weights_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    q_ref = np.asarray(q_ref_getter(float(t)), dtype=float)
    err = np.asarray(q, dtype=float) - q_ref
    norms = manifold_weighted_norms(err, geom.J, weights_2d)
    norms["max_abs_error"] = float(np.max(np.abs(err)))
    return q_ref, err, norms


def _mass_relative_error(mass: float, mass0: float) -> float:
    if abs(float(mass0)) <= np.finfo(float).tiny:
        return math.nan
    return float(mass - mass0) / float(mass0)


def _assign_legacy_aliases(row: dict) -> None:
    field_case = str(row["field_case"])
    if field_case == "gaussian":
        row["gaussian_L2_error"] = float(row["L2_error"])
        row["gaussian_Linf_error"] = float(row["Linf_error"])
        row["rate_gaussian_L2"] = float(row["rate_L2"])
        row["rate_gaussian_Linf"] = float(row["rate_Linf"])
        row["const_L2_drift"] = math.nan
        row["const_Linf_drift"] = math.nan
        row["const_max_drift"] = math.nan
        row["rate_const_L2"] = math.nan
        row["rate_const_Linf"] = math.nan
    else:
        row["gaussian_L2_error"] = math.nan
        row["gaussian_Linf_error"] = math.nan
        row["rate_gaussian_L2"] = math.nan
        row["rate_gaussian_Linf"] = math.nan
        row["const_L2_drift"] = float(row["L2_error"])
        row["const_Linf_drift"] = float(row["Linf_error"])
        row["const_max_drift"] = float(row["max_abs_error"])
        row["rate_const_L2"] = float(row["rate_L2"])
        row["rate_const_Linf"] = float(row["rate_Linf"])


def _build_summary_row(
    *,
    config: ManifoldLSRKConvergenceConfig,
    field_case: str,
    n_div: int,
    EToV: np.ndarray,
    nodes_xyz: np.ndarray,
    ref_ops,
    h: float,
    vmax: float,
    dt: float,
    tf_used: float,
    nsteps: int,
    norms: dict[str, float],
    mass0: float,
    mass_final: float,
    elapsed_sec: float,
) -> dict:
    center_x, center_y, center_z = _resolve_initial_center(config)
    return {
        "n_div": int(n_div),
        "K": int(EToV.shape[0]),
        "Nv": int(nodes_xyz.shape[0]),
        "Np": int(ref_ops.rs_nodes.shape[0]),
        "total_dof": int(EToV.shape[0] * ref_ops.rs_nodes.shape[0]),
        "h": float(h),
        "hmin": float(h),
        "vmax": float(vmax),
        "cfl": float(config.cfl),
        "dt": float(dt),
        "flux_type": _normalize_flux_type(config.flux_type),
        "alpha_lf": float(config.alpha_lf),
        "tf_target": float(config.tf),
        "tf": float(tf_used),
        "reached_tf": bool(is_tf_reached(tf_used, config.tf)),
        "initial_preset": _normalize_initial_preset(config.initial_preset),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_z": float(center_z),
        "mass0": float(mass0),
        "mass": float(mass_final),
        "mass_error": float(mass_final - mass0),
        "mass_rel_error": float(_mass_relative_error(mass_final, mass0)),
        "nsteps": int(nsteps),
        "field_case": field_case,
        "constant_value": float(config.constant_value),
        "error_reference": _field_case_error_label(field_case),
        "L2_error": float(norms["L2"]),
        "Linf_error": float(norms["Linf"]),
        "max_abs_error": float(norms["max_abs_error"]),
        "elapsed_sec": float(elapsed_sec),
    }


def _run_one_level(
    config: ManifoldLSRKConvergenceConfig,
    n_div: int,
    ref_ops,
) -> dict:
    t0 = perf_counter()
    field_case = _normalize_field_case(config.field_case)
    snapshot_targets = _normalize_snapshot_times(config.snapshot_times)

    nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=n_div, R=config.R)
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
    exchange_cache = build_manifold_exchange_cache(
        EToV=EToV,
        ref_ops=ref_ops,
        geom=geom,
        U=U,
        V=V,
        W=W,
        use_numba=config.use_numba,
    )

    h = spherical_mesh_hmin(nodes_xyz, EToV)
    vmax = _max_speed(U, V, W)
    flux_speed_scale = _dt_speed_scale(config)
    # Match the notebook-scale dt = CFL * h / (vmax * (k + 1)^2) while reusing
    # the shared helper, whose denominator is its N argument squared.
    dt = cfl_dt_from_h(cfl=config.cfl, h=h, N=config.N + 1, vmax=vmax * flux_speed_scale)
    q0, q_ref_getter = _make_reference_field_getter(geom, config)
    mass0 = float(manifold_weighted_mass(q0, geom.J, ref_ops.weights_2d))
    velocity_xyz = (U, V, W)

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        return manifold_rhs_exchange(
            q=q,
            geom=geom,
            velocity_xyz=velocity_xyz,
            exchange_cache=exchange_cache,
            ref_ops=ref_ops,
            flux_type=_normalize_flux_type(config.flux_type),
            alpha_lf=config.alpha_lf,
            volume_form=_normalize_volume_form(config.volume_form),
            surface_assembly=_normalize_surface_assembly(config.surface_assembly),
            t=t,
            use_numba=config.use_numba,
        )

    histories_enabled = bool(config.record_history)
    snapshots_enabled = len(snapshot_targets) > 0
    tol = tf_align_tolerance(float(config.tf), 0.0)
    times: list[float] = []
    step_ids: list[int] = []
    l2_errors: list[float] = []
    linf_errors: list[float] = []
    max_abs_q: list[float] = []
    masses: list[float] = []
    mass_errors: list[float] = []
    mass_rel_errors: list[float] = []
    snapshots: list[dict] = []
    step_snapshots: list[dict] = []
    next_snapshot = 0
    step_snapshots_enabled = bool(config.record_step_snapshots)

    q_ref0, err0, norms0 = _compute_error_state(q0, 0.0, q_ref_getter, geom, ref_ops.weights_2d)
    mass_rel0 = 0.0 if abs(mass0) > np.finfo(float).tiny else math.nan
    if histories_enabled:
        step_ids.append(0)
        times.append(0.0)
        l2_errors.append(float(norms0["L2"]))
        linf_errors.append(float(norms0["Linf"]))
        max_abs_q.append(float(np.max(np.abs(q0))))
        masses.append(float(mass0))
        mass_errors.append(0.0)
        mass_rel_errors.append(mass_rel0)
    if step_snapshots_enabled:
        step_snapshots.append(
            {
                "step_index": 0,
                "time_actual": 0.0,
                "q": np.array(q0, copy=True),
                "q_ref": np.array(q_ref0, copy=True),
                "error": np.array(err0, copy=True),
                "mass": float(mass0),
                "mass_error": 0.0,
                "mass_rel_error": mass_rel0,
            }
        )

    while snapshots_enabled and next_snapshot < len(snapshot_targets) and snapshot_targets[next_snapshot] <= tol:
        snapshots.append(
            {
                "time_requested": float(snapshot_targets[next_snapshot]),
                "time_actual": 0.0,
                "q": np.array(q0, copy=True),
                "q_ref": np.array(q_ref0, copy=True),
                "error": np.array(err0, copy=True),
                "mass": float(mass0),
                "mass_error": 0.0,
                "mass_rel_error": mass_rel0,
            }
        )
        next_snapshot += 1

    def _record_step(t_step: float, q_step: np.ndarray) -> np.ndarray:
        nonlocal next_snapshot
        q_ref_step, err_step, norms_step = _compute_error_state(
            q_step,
            t_step,
            q_ref_getter,
            geom,
            ref_ops.weights_2d,
        )
        mass_step = float(manifold_weighted_mass(q_step, geom.J, ref_ops.weights_2d))
        mass_error_step = float(mass_step - mass0)
        mass_rel_error_step = _mass_relative_error(mass_step, mass0)
        if histories_enabled:
            step_ids.append(len(step_ids))
            times.append(float(t_step))
            l2_errors.append(float(norms_step["L2"]))
            linf_errors.append(float(norms_step["Linf"]))
            max_abs_q.append(float(np.max(np.abs(q_step))))
            masses.append(mass_step)
            mass_errors.append(mass_error_step)
            mass_rel_errors.append(mass_rel_error_step)
        current_step_index = len(times) - 1 if histories_enabled else len(step_snapshots)
        if step_snapshots_enabled:
            step_snapshots.append(
                {
                    "step_index": int(current_step_index),
                    "time_actual": float(t_step),
                    "q": np.array(q_step, copy=True),
                    "q_ref": np.array(q_ref_step, copy=True),
                    "error": np.array(err_step, copy=True),
                    "mass": float(mass_step),
                    "mass_error": mass_error_step,
                    "mass_rel_error": mass_rel_error_step,
                }
            )
        while snapshots_enabled and next_snapshot < len(snapshot_targets):
            t_req = snapshot_targets[next_snapshot]
            if float(t_step) + tol < t_req:
                break
            snapshots.append(
                {
                    "time_requested": float(t_req),
                    "time_actual": float(t_step),
                    "q": np.array(q_step, copy=True),
                    "q_ref": np.array(q_ref_step, copy=True),
                    "error": np.array(err_step, copy=True),
                    "mass": float(mass_step),
                    "mass_error": mass_error_step,
                    "mass_rel_error": mass_rel_error_step,
                }
            )
            next_snapshot += 1
        return q_step

    post_step_transform = _record_step if (histories_enabled or snapshots_enabled) else None
    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=0.0,
        tf=config.tf,
        dt=dt,
        post_step_transform=post_step_transform,
    )
    _q_ref_final, _err_final, norms_final = _compute_error_state(
        qf,
        tf_used,
        q_ref_getter,
        geom,
        ref_ops.weights_2d,
    )
    mass_final = float(manifold_weighted_mass(qf, geom.J, ref_ops.weights_2d))
    if config.verbose:
        print(
            f"Finished integration: tf_used={tf_used:.6e}, "
            f"nsteps={nsteps}, elapsed_sec={perf_counter() - t0:.2f}s"
        )
    row = _build_summary_row(
        config=config,
        field_case=field_case,
        n_div=n_div,
        EToV=EToV,
        nodes_xyz=nodes_xyz,
        ref_ops=ref_ops,
        h=h,
        vmax=vmax,
        dt=dt,
        tf_used=tf_used,
        nsteps=nsteps,
        norms=norms_final,
        mass0=mass0,
        mass_final=mass_final,
        elapsed_sec=perf_counter() - t0,
    )

    if histories_enabled:
        row["history"] = {
            "mesh_level": int(n_div),
            "field_case": field_case,
            "flux_type": _normalize_flux_type(config.flux_type),
            "alpha_lf": float(config.alpha_lf),
            "error_reference": row["error_reference"],
            "initial_preset": row["initial_preset"],
            "center_x": float(row["center_x"]),
            "center_y": float(row["center_y"]),
            "center_z": float(row["center_z"]),
            "flux_type": str(row["flux_type"]),
            "alpha_lf": float(row["alpha_lf"]),
            "mass0": float(row["mass0"]),
            "mass": float(row["mass"]),
            "mass_error": float(row["mass_error"]),
            "mass_rel_error": float(row["mass_rel_error"]),
            "h": float(h),
            "step_ids": np.asarray(step_ids, dtype=int),
            "times": np.asarray(times, dtype=float),
            "l2": np.asarray(l2_errors, dtype=float),
            "linf": np.asarray(linf_errors, dtype=float),
            "max_abs_q": np.asarray(max_abs_q, dtype=float),
            "mass": np.asarray(masses, dtype=float),
            "mass_error": np.asarray(mass_errors, dtype=float),
            "mass_rel_error": np.asarray(mass_rel_errors, dtype=float),
            "reached_tf": bool(row["reached_tf"]),
            "tf_used": float(tf_used),
            "nsteps": int(nsteps),
        }
    if snapshots_enabled:
        row["snapshots"] = snapshots
    if step_snapshots_enabled:
        row["step_snapshots"] = step_snapshots
    if snapshots_enabled or step_snapshots_enabled:
        row["artifacts"] = {
            "geom": geom,
            "nodes_xyz": np.asarray(nodes_xyz, dtype=float),
            "EToV": np.asarray(EToV, dtype=int),
        }
    
    print(f"Completed level n_div={n_div} in {perf_counter() - t0:.2f}s.")
    return row


def run_manifold_lsrk_convergence(
    config: ManifoldLSRKConvergenceConfig,
) -> list[dict]:
    _validate_config(config)
    ref_ops = build_manifold_table1_k4_reference_operators()
    results = [_run_one_level(config, int(n), ref_ops) for n in config.mesh_levels]

    hs = [float(r["h"]) for r in results]
    l2_rates = compute_convergence_rate([float(r["L2_error"]) for r in results], hs)
    linf_rates = compute_convergence_rate([float(r["Linf_error"]) for r in results], hs)

    for row, rl2, rlinf in zip(results, l2_rates, linf_rates):
        row["rate_L2"] = float(rl2)
        row["rate_Linf"] = float(rlinf)
        _assign_legacy_aliases(row)

        if config.verbose:
            center_label = (
                f"preset={row['initial_preset']} "
                f"center=({row['center_x']:.3f},{row['center_y']:.3f},{row['center_z']:.3f})"
            )
            print(
            f"[manifold LSRK] field={row['field_case']:>8s} | "
                f"flux={row['flux_type']:>14s} | "
                f"n_div={row['n_div']:3d} | K={row['K']:6d} | "
                f"h={row['h']:.6e} | dt={row['dt']:.3e} | "
                f"steps={row['nsteps']:6d} | "
                f"L2={row['L2_error']:.6e} | "
                f"Linf={row['Linf_error']:.6e} | "
                f"mass_err={row['mass_error']:.3e} | "
                f"time={row['elapsed_sec']:.2f}s | {center_label}"
            )

    return results


def extract_time_histories(results: list[dict]) -> list[dict]:
    histories: list[dict] = []
    for row in results:
        history = row.get("history")
        if isinstance(history, dict):
            histories.append(history)
    return histories


def print_results_table(results: list[dict]) -> None:
    header = (
        f"{'field':>10s} {'flux':>14s} {'n_div':>6s} {'K':>9s} {'h':>12s} {'dt':>12s} {'steps':>8s} "
        f"{'L2':>14s} {'rate':>8s} {'Linf':>14s} {'rate':>8s}"
    )
    print(header)
    print("-" * len(header))

    def fmt_rate(v: float) -> str:
        return "   -   " if not np.isfinite(v) else f"{v:8.3f}"

    for row in results:
        print(
            f"{str(row['field_case']):>10s} {str(row['flux_type']):>14s} {row['n_div']:6d} {row['K']:9d} {row['h']:12.4e} "
            f"{row['dt']:12.4e} {row['nsteps']:8d} "
            f"{row['L2_error']:14.6e} {fmt_rate(float(row['rate_L2']))} "
            f"{row['Linf_error']:14.6e} {fmt_rate(float(row['rate_Linf']))}"
        )


def _summary_fieldnames(results: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    for key, value in results[0].items():
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            continue
        fieldnames.append(key)
    return fieldnames


def save_results_csv(results: list[dict], filepath: str | Path) -> None:
    if not results:
        raise ValueError("results is empty.")

    fieldnames = _summary_fieldnames(results)
    with Path(filepath).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row[name] for name in fieldnames})


def save_time_history_csv(results: list[dict], filepath: str | Path) -> None:
    histories = extract_time_histories(results)
    if not histories:
        raise ValueError("results does not contain any recorded histories.")

    fieldnames = [
        "mesh_level",
        "h",
        "field_case",
        "flux_type",
        "alpha_lf",
        "initial_preset",
        "center_x",
        "center_y",
        "center_z",
        "mass0",
        "mass",
        "mass_error",
        "mass_rel_error",
        "error_reference",
        "step_index",
        "time",
        "L2_error",
        "Linf_error",
        "max_abs_q",
        "reached_tf",
        "tf_used",
        "nsteps",
    ]
    with Path(filepath).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for history in histories:
            times = np.asarray(history["times"], dtype=float)
            step_ids = np.asarray(history["step_ids"], dtype=int)
            l2 = np.asarray(history["l2"], dtype=float)
            linf = np.asarray(history["linf"], dtype=float)
            qmax = np.asarray(history["max_abs_q"], dtype=float)
            for i in range(times.size):
                writer.writerow(
                    {
                        "mesh_level": int(history["mesh_level"]),
                        "h": float(history["h"]),
                        "field_case": str(history["field_case"]),
                        "flux_type": str(history["flux_type"]),
                        "alpha_lf": float(history["alpha_lf"]),
                        "initial_preset": str(history["initial_preset"]),
                        "center_x": float(history["center_x"]),
                        "center_y": float(history["center_y"]),
                        "center_z": float(history["center_z"]),
                        "mass0": float(history["mass0"]),
                        "mass": float(history["mass"][i]),
                        "mass_error": float(history["mass_error"][i]),
                        "mass_rel_error": float(history["mass_rel_error"][i]),
                        "error_reference": str(history["error_reference"]),
                        "step_index": int(step_ids[i]),
                        "time": float(times[i]),
                        "L2_error": float(l2[i]),
                        "Linf_error": float(linf[i]),
                        "max_abs_q": float(qmax[i]),
                        "reached_tf": bool(history["reached_tf"]),
                        "tf_used": float(history["tf_used"]),
                        "nsteps": int(history["nsteps"]),
                    }
                )
