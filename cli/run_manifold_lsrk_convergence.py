from __future__ import annotations

import argparse
import math

import numpy as np

from experiments.manifold_lsrk_convergence import (
    ManifoldLSRKConvergenceConfig,
    extract_time_histories,
    print_results_table,
    run_manifold_lsrk_convergence,
    save_results_csv,
    save_time_history_csv,
)
from experiments.output_paths import experiments_output_dir
from visualization.manifold_diagnostics import (
    plot_manifold_l2_error_vs_time,
    plot_manifold_mass_error_vs_time,
    plot_manifold_lsrk_convergence,
)


def _parse_levels(text: str) -> tuple[int, ...]:
    levels = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not levels:
        raise argparse.ArgumentTypeError("mesh levels must not be empty.")
    return levels


def _parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("center-xyz must contain exactly three comma-separated values.")
    try:
        x, y, z = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("center-xyz must contain valid floats.") from exc
    return x, y, z


def _float_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(float(value)).replace(".", "p")


def _float3_slug(values: tuple[float, float, float]) -> str:
    return "_".join(f"{axis}{_float_slug(value)}" for axis, value in zip(("cx", "cy", "cz"), values, strict=True))


def _normalize_center_xyz(R: float, center_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    center = np.asarray(center_xyz, dtype=float).reshape(3)
    norm = float(np.linalg.norm(center))
    if norm == 0.0:
        raise ValueError("center-xyz must not be the zero vector when initial-preset=custom.")
    center = float(R) * center / norm
    return float(center[0]), float(center[1]), float(center[2])


def _field_slug(
    field_case: str,
    constant_value: float,
    initial_preset: str,
    center_xyz: tuple[float, float, float],
    R: float,
) -> str:
    field_case = str(field_case).strip().lower()
    if field_case == "constant":
        return f"constant_v{_float_slug(constant_value)}"
    initial_preset = str(initial_preset).strip().lower()
    if initial_preset != "custom":
        return f"{field_case}_{initial_preset}"
    return f"{field_case}_{_float3_slug(_normalize_center_xyz(R, center_xyz))}"


def _flux_slug(flux_type: str, alpha_lf: float) -> str:
    flux = str(flux_type).strip().lower()
    if flux == "upwind" and float(alpha_lf) == 1.0:
        return ""
    if flux == "upwind":
        return f"upwind_a{_float_slug(alpha_lf)}"
    if flux == "central":
        return "central"
    if flux == "lax_friedrichs":
        return f"lf_a{_float_slug(alpha_lf)}"
    raise ValueError("flux_type must be one of: upwind, central, lax_friedrichs.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Table1 k=4 3D manifold sphere LSRK time-stepping study.",
    )
    parser.add_argument("--mesh-levels", type=_parse_levels, default=(2, 4, 8))
    parser.add_argument("--tf", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=1.0)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=-math.pi / 4.0)
    parser.add_argument("--center-xyz", type=_parse_xyz, default=(1.0, 0.0, 0.0))
    parser.add_argument(
        "--initial-preset",
        choices=("custom", "equator", "equator_x", "equator_y", "north_pole", "south_pole"),
        default="custom",
    )
    parser.add_argument("--gaussian-width", type=float, default=1.0 / math.sqrt(10.0))
    parser.add_argument(
        "--field-case",
        choices=("gaussian", "constant"),
        default="gaussian",
    )
    parser.add_argument(
        "--flux-type",
        choices=("upwind", "central", "lax_friedrichs"),
        default="upwind",
    )
    parser.add_argument("--alpha-lf", type=float, default=1.0)
    parser.add_argument("--constant-value", type=float, default=1.0)
    parser.add_argument("--use-numba", action="store_true")
    args = parser.parse_args()

    output_dir = experiments_output_dir(__file__, "manifold_lsrk_convergence")
    config = ManifoldLSRKConvergenceConfig(
        mesh_levels=args.mesh_levels,
        R=args.R,
        u0=args.u0,
        alpha0=args.alpha0,
        center_xyz=args.center_xyz,
        initial_preset=args.initial_preset,
        cfl=args.cfl,
        tf=args.tf,
        gaussian_width=args.gaussian_width,
        field_case=args.field_case,
        flux_type=args.flux_type,
        alpha_lf=args.alpha_lf,
        constant_value=args.constant_value,
        record_history=True,
        verbose=True,
        use_numba=args.use_numba,
    )

    results = run_manifold_lsrk_convergence(config)
    histories = extract_time_histories(results)
    slug = _field_slug(args.field_case, args.constant_value, args.initial_preset, args.center_xyz, args.R)
    flux_slug = _flux_slug(args.flux_type, args.alpha_lf)
    if flux_slug:
        slug = f"{slug}_{flux_slug}"

    csv_path = output_dir / f"manifold_lsrk_convergence_table1_k4_{slug}.csv"
    fig_path = output_dir / f"manifold_lsrk_convergence_table1_k4_{slug}.png"
    history_csv_path = output_dir / f"manifold_lsrk_error_vs_time_table1_k4_{slug}.csv"
    history_fig_path = output_dir / f"manifold_lsrk_error_vs_time_table1_k4_{slug}.png"
    mass_fig_path = output_dir / f"manifold_lsrk_mass_error_vs_time_table1_k4_{slug}.png"

    print()
    print_results_table(results)

    save_results_csv(results, csv_path)
    plot_manifold_lsrk_convergence(results, fig_path)
    save_time_history_csv(results, history_csv_path)
    plot_manifold_l2_error_vs_time(histories, history_fig_path)
    plot_manifold_mass_error_vs_time(histories, mass_fig_path)

    print()
    print("[OK] wrote " + str(csv_path))
    print("[OK] wrote " + str(fig_path))
    print("[OK] wrote " + str(history_csv_path))
    print("[OK] wrote " + str(history_fig_path))
    print("[OK] wrote " + str(mass_fig_path))


if __name__ == "__main__":
    main()
