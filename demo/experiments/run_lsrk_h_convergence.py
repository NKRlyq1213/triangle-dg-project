from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from experiments.lsrk_h_convergence import (
    LSRKHConvergenceConfig,
    run_lsrk_h_convergence,
    run_lsrk_h_convergence_compare_qb_correction,
    print_results_table,
    save_results_csv,
)


def _tf_label(tf: float) -> str:
    if float(tf).is_integer():
        return str(int(tf))
    return str(tf).replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LSRK h-convergence study with quick/full presets.",
    )
    parser.add_argument(
        "--preset",
        choices=("quick", "full"),
        default="quick",
        help="quick: faster sanity run, full: complete run",
    )
    parser.add_argument(
        "--qb-correction",
        choices=("off", "on", "compare"),
        default=None,
        help="qB mode: off=baseline, on=RK-stage qB correction only, compare=run baseline and correction",
    )
    parser.add_argument(
        "--compare-qb-correction",
        action="store_true",
        help="legacy alias of --qb-correction compare",
    )
    parser.add_argument(
        "--qb-correction-only",
        action="store_true",
        help="legacy alias of --qb-correction on",
    )
    parser.add_argument(
        "--surface-inverse-mass-mode",
        choices=("diagonal", "projected"),
        default="diagonal",
        help="surface lifting inverse mass: diagonal (default) or projected",
    )
    parser.add_argument(
        "--test-function",
        choices=("sin2pi_x", "sin2pi_y", "sin2pi_xy"),
        default="sin2pi_x",
        help="transport exact-profile mode",
    )
    parser.add_argument(
        "--physical-boundary-mode",
        choices=("exact_qb", "opposite_boundary"),
        default="exact_qb",
        help="physical boundary exterior-state mode",
    )
    parser.add_argument(
        "--interior-trace-mode",
        choices=("exchange", "exact_trace"),
        default="exchange",
        help="interior-face mode: exchange uses connectivity, exact_trace uses exact exterior trace on all faces",
    )
    parser.add_argument(
        "--state-projection",
        choices=("off", "on"),
        default="off",
        help="toggle polynomial state projection P on nodal state",
    )
    parser.add_argument(
        "--projection-mode",
        choices=("pre", "post", "both"),
        default="post",
        help="state projection application point inside RHS",
    )
    parser.add_argument(
        "--projection-frequency",
        choices=("rhs", "step"),
        default="step",
        help="apply state projection at each RHS call or each RK step",
    )
    return parser.parse_args()


def _print_compare_summary(
    baseline: list[dict],
    corrected: list[dict],
    *,
    tf: float,
) -> None:
    print()
    print(f"[compare] tf={tf:g} | L2_ratio=baseline/corrected | time_ratio=corrected/baseline")
    print(f"{'n':>6s} {'L2_ratio':>12s} {'time_ratio':>12s}")
    print("-" * 34)

    corr_by_n = {int(r["nx"]): r for r in corrected}
    for row_b in baseline:
        n = int(row_b["nx"])
        row_c = corr_by_n.get(n)
        if row_c is None:
            continue

        l2_b = float(row_b["L2_error"])
        l2_c = float(row_c["L2_error"])
        t_b = float(row_b["elapsed_sec"])
        t_c = float(row_c["elapsed_sec"])

        l2_ratio = (l2_b / l2_c) if (l2_c > 0.0 and l2_b > 0.0) else float("nan")
        time_ratio = (t_c / t_b) if t_b > 0.0 else float("nan")

        print(f"{n:6d} {l2_ratio:12.4f} {time_ratio:12.4f}")


def _build_config(
    preset: str,
    surface_inverse_mass_mode: str,
    *,
    state_projection: str,
    projection_mode: str,
    projection_frequency: str,
    test_function_mode: str,
    physical_boundary_mode: str,
    interior_trace_mode: str,
) -> LSRKHConvergenceConfig:
    projection_enabled = str(state_projection).strip().lower() == "on"

    if preset == "quick":
        return LSRKHConvergenceConfig(
            table_name="table1",
            order=4,
            N=4,
            diagonal="anti",
            mesh_levels=(1, 2, 4, 8, 16),
            cfl=1.0,
            tf_values=(np.pi*2,),
            tau=0.0,
            use_numba=True,
            enforce_polynomial_projection=projection_enabled,
            projection_mode=projection_mode,
            projection_frequency=projection_frequency,
            surface_inverse_mass_mode=surface_inverse_mass_mode,
            surface_backend="face-major",
            interior_trace_mode=interior_trace_mode,
            test_function_mode=test_function_mode,
            physical_boundary_mode=physical_boundary_mode,
            use_surface_cache=True,
            verbose=False,
        )

    return LSRKHConvergenceConfig(
        table_name="table1",
        order=4,
        N=4,
        diagonal="anti",
        mesh_levels=(1, 2, 4, 8, 16, 32),
        cfl=1.0,
        tf_values=(1.0,),
        tau=0.0,
        enforce_polynomial_projection=projection_enabled,
        use_numba=True,
        projection_mode=projection_mode,
        projection_frequency=projection_frequency,
        surface_inverse_mass_mode=surface_inverse_mass_mode,
        surface_backend="face-major",
        interior_trace_mode=interior_trace_mode,
        test_function_mode=test_function_mode,
        physical_boundary_mode=physical_boundary_mode,
        use_surface_cache=True,
        verbose=True,
    )


def _resolve_qb_mode(args: argparse.Namespace) -> str:
    qb_mode = None if args.qb_correction is None else str(args.qb_correction).strip().lower()
    legacy_compare = bool(args.compare_qb_correction)
    legacy_on = bool(args.qb_correction_only)

    if legacy_compare and legacy_on:
        raise ValueError("Use only one of --compare-qb-correction or --qb-correction-only.")

    if qb_mode is not None and (legacy_compare or legacy_on):
        raise ValueError(
            "Use either --qb-correction or legacy flags (--compare-qb-correction/--qb-correction-only), not both."
        )

    if qb_mode is not None:
        return qb_mode
    if legacy_compare:
        return "compare"
    if legacy_on:
        return "on"
    return "off"


def main() -> None:
    args = _parse_args()
    qb_mode = _resolve_qb_mode(args)

    output_dir = Path(__file__).resolve().parents[2] / "experiments_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        args.preset,
        args.surface_inverse_mass_mode,
        state_projection=args.state_projection,
        projection_mode=args.projection_mode,
        projection_frequency=args.projection_frequency,
        test_function_mode=args.test_function,
        physical_boundary_mode=args.physical_boundary_mode,
        interior_trace_mode=args.interior_trace_mode,
    )

    if qb_mode == "on":
        config = replace(
            config,
            use_sinx_rk_stage_boundary_correction=True,
            q_boundary_correction=None,
        )
    print(f"[run] preset={args.preset}")
    print(f"[run] surface_inverse_mass_mode={config.surface_inverse_mass_mode}")
    print(f"[run] test_function_mode={config.test_function_mode}")
    print(f"[run] physical_boundary_mode={config.physical_boundary_mode}")
    print(f"[run] interior_trace_mode={config.interior_trace_mode}")
    print(f"[run] qb_correction={qb_mode}")
    print(f"[run] state_projection={'on' if config.enforce_polynomial_projection else 'off'}")
    print(f"[run] projection_mode={config.projection_mode}")
    print(f"[run] projection_frequency={config.projection_frequency}")

    if qb_mode == "compare":
        print("[run] compare_qb_correction=on")
        compared = run_lsrk_h_convergence_compare_qb_correction(config)

        baseline_map = compared["baseline"]
        corrected_map = compared["rk_stage_correction"]

        for tf, baseline_results in baseline_map.items():
            corrected_results = corrected_map[tf]

            print()
            print_results_table(
                baseline_results,
                title=(
                    f"LSRK h-convergence (sinx) | baseline | "
                    f"trace={config.interior_trace_mode} | tf={tf:g}, CFL={config.cfl:g}"
                ),
            )
            print()
            print_results_table(
                corrected_results,
                title=(
                    f"LSRK h-convergence (sinx) | rk-stage qB correction | "
                    f"trace={config.interior_trace_mode} | tf={tf:g}, CFL={config.cfl:g}"
                ),
            )

            _print_compare_summary(baseline_results, corrected_results, tf=tf)

            base_name = (
                f"lsrk_h_convergence_sinx_tf{_tf_label(tf)}_"
                f"table1_order{config.order}_N{config.N}_{config.diagonal}"
            )
            if str(config.interior_trace_mode).strip().lower() != "exchange":
                base_name += f"_{str(config.interior_trace_mode).strip().lower()}"
            if args.preset != "full":
                base_name += f"_{args.preset}"

            csv_baseline = output_dir / f"{base_name}_baseline.csv"
            csv_corrected = output_dir / f"{base_name}_rkstage_qb.csv"
            save_results_csv(baseline_results, str(csv_baseline))
            save_results_csv(corrected_results, str(csv_corrected))

            print()
            print("[OK] wrote " + str(csv_baseline))
            print("[OK] wrote " + str(csv_corrected))
        return

    if qb_mode == "on":
        print("[run] qb_correction_only=on")
        all_results = run_lsrk_h_convergence(config)

        for tf, results in all_results.items():
            print()
            print_results_table(
                results,
                title=(
                    f"LSRK h-convergence (sinx) | rk-stage qB correction | "
                    f"trace={config.interior_trace_mode} | tf={tf:g}, CFL={config.cfl:g}"
                ),
            )

            csv_name = (
                f"lsrk_h_convergence_sinx_tf{_tf_label(tf)}_"
                f"table1_order{config.order}_N{config.N}_{config.diagonal}_rkstage_qb_only.csv"
            )
            if str(config.interior_trace_mode).strip().lower() != "exchange":
                csv_name = csv_name.replace(
                    ".csv",
                    f"_{str(config.interior_trace_mode).strip().lower()}.csv",
                )
            if args.preset != "full":
                csv_name = csv_name.replace(".csv", f"_{args.preset}.csv")

            csv_path = output_dir / csv_name
            save_results_csv(results, str(csv_path))
            print()
            print("[OK] wrote " + str(csv_path))
        return

    all_results = run_lsrk_h_convergence(config)

    for tf, results in all_results.items():
        print()
        print_results_table(
            results,
            title=(
                f"LSRK h-convergence (sinx) | trace={config.interior_trace_mode} | "
                f"tf={tf:g}, CFL={config.cfl:g}"
            ),
        )

        csv_name = (
            f"lsrk_h_convergence_sinx_tf{_tf_label(tf)}_"
            f"table1_order{config.order}_N{config.N}_{config.diagonal}.csv"
        )
        if str(config.interior_trace_mode).strip().lower() != "exchange":
            csv_name = csv_name.replace(
                ".csv",
                f"_{str(config.interior_trace_mode).strip().lower()}.csv",
            )
        if args.preset != "full":
            csv_name = csv_name.replace(".csv", f"_{args.preset}.csv")

        csv_path = output_dir / csv_name
        save_results_csv(results, str(csv_path))
        print()
        print("[OK] wrote " + str(csv_path))


if __name__ == "__main__":
    main()
