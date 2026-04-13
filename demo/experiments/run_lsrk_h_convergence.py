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
        "--compare-qb-correction",
        action="store_true",
        help="run baseline and notebook-style RK-stage qB correction side-by-side",
    )
    parser.add_argument(
        "--qb-correction-only",
        action="store_true",
        help="run only notebook-style RK-stage qB correction (no baseline)",
    )
    parser.add_argument(
        "--surface-inverse-mass-mode",
        choices=("diagonal", "projected"),
        default="projected",
        help="surface lifting inverse mass: projected (default) or diagonal",
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
            tf_values=(np.pi * 2,),
            tau=0.0,
            use_numba=True,
            enforce_polynomial_projection=projection_enabled,
            projection_mode=projection_mode,
            projection_frequency=projection_frequency,
            surface_inverse_mass_mode=surface_inverse_mass_mode,
            surface_backend="face-major",
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
        tf_values=(np.pi * 2,),
        tau=0.0,
        enforce_polynomial_projection=projection_enabled,
        use_numba=True,
        projection_mode=projection_mode,
        projection_frequency=projection_frequency,
        surface_inverse_mass_mode=surface_inverse_mass_mode,
        surface_backend="face-major",
        use_surface_cache=True,
        verbose=True,
    )


def main() -> None:
    args = _parse_args()

    if args.compare_qb_correction and args.qb_correction_only:
        raise ValueError("Use only one of --compare-qb-correction or --qb-correction-only.")

    output_dir = Path(__file__).resolve().parents[2] / "experiments_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        args.preset,
        args.surface_inverse_mass_mode,
        state_projection=args.state_projection,
        projection_mode=args.projection_mode,
        projection_frequency=args.projection_frequency,
    )

    if args.qb_correction_only:
        config = replace(
            config,
            use_sinx_rk_stage_boundary_correction=True,
            q_boundary_correction=None,
        )
    print(f"[run] preset={args.preset}")
    print(f"[run] surface_inverse_mass_mode={config.surface_inverse_mass_mode}")
    print(f"[run] state_projection={'on' if config.enforce_polynomial_projection else 'off'}")
    print(f"[run] projection_mode={config.projection_mode}")
    print(f"[run] projection_frequency={config.projection_frequency}")

    if args.compare_qb_correction:
        print("[run] compare_qb_correction=on")
        compared = run_lsrk_h_convergence_compare_qb_correction(config)

        baseline_map = compared["baseline"]
        corrected_map = compared["rk_stage_correction"]

        for tf, baseline_results in baseline_map.items():
            corrected_results = corrected_map[tf]

            print()
            print_results_table(
                baseline_results,
                title=f"LSRK h-convergence (sinx) | baseline | tf={tf:g}, CFL={config.cfl:g}",
            )
            print()
            print_results_table(
                corrected_results,
                title=f"LSRK h-convergence (sinx) | rk-stage qB correction | tf={tf:g}, CFL={config.cfl:g}",
            )

            _print_compare_summary(baseline_results, corrected_results, tf=tf)

            base_name = (
                f"lsrk_h_convergence_sinx_tf{_tf_label(tf)}_"
                f"table1_order{config.order}_N{config.N}_{config.diagonal}"
            )
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

    if args.qb_correction_only:
        print("[run] qb_correction_only=on")
        all_results = run_lsrk_h_convergence(config)

        for tf, results in all_results.items():
            print()
            print_results_table(
                results,
                title=f"LSRK h-convergence (sinx) | rk-stage qB correction | tf={tf:g}, CFL={config.cfl:g}",
            )

            csv_name = (
                f"lsrk_h_convergence_sinx_tf{_tf_label(tf)}_"
                f"table1_order{config.order}_N{config.N}_{config.diagonal}_rkstage_qb_only.csv"
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
        print_results_table(results, title=f"LSRK h-convergence (sinx) | tf={tf:g}, CFL={config.cfl:g}")

        csv_name = (
            f"lsrk_h_convergence_sinx_tf{_tf_label(tf)}_"
            f"table1_order{config.order}_N{config.N}_{config.diagonal}.csv"
        )
        if args.preset != "full":
            csv_name = csv_name.replace(".csv", f"_{args.preset}.csv")

        csv_path = output_dir / csv_name
        save_results_csv(results, str(csv_path))
        print()
        print("[OK] wrote " + str(csv_path))


if __name__ == "__main__":
    main()
