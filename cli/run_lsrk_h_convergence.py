from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from experiments.lsrk_h_convergence import (
    LSRKHConvergenceConfig,
    _resolve_test_function_spec,
    compute_convergence_rate,
    print_results_table,
    resolve_effective_taus,
    run_lsrk_study,
    save_results_csv,
)
from experiments.output_paths import experiments_output_dir


def _tf_label(tf: float) -> str:
    if float(tf).is_integer():
        return str(int(tf))
    return str(tf).replace(".", "p")


def _tau_label(tau: float) -> str:
    if float(tau).is_integer():
        return str(int(tau))
    return str(tau).replace(".", "p")


def _test_function_slug(mode: str) -> str:
    return _resolve_test_function_spec(mode).mode


def _tf_values_label(tf_values: tuple[float, ...]) -> str:
    labels = [_tf_label(tf) for tf in sorted({float(tf) for tf in tf_values})]
    return "-".join(labels)


def _output_stem(config: LSRKHConvergenceConfig, tf: float) -> str:
    tau_interior, tau_qb = resolve_effective_taus(
        tau=config.tau,
        tau_interior=config.tau_interior,
        tau_qb=config.tau_qb,
    )
    return (
        f"lsrk_h_convergence_{_test_function_slug(config.test_function_mode)}_tf{_tf_label(tf)}_"
        f"table1_order{config.order}_N{config.N}_{config.diagonal}"
        f"_face{str(config.face_order_mode).strip().lower()}"
        f"_{str(config.surface_inverse_mass_mode).strip().lower()}"
        f"_{str(config.physical_boundary_mode).strip().lower()}"
        f"_taui{_tau_label(tau_interior)}"
        f"_tauqb{_tau_label(tau_qb)}"
    )


def _time_scan_output_stem(
    config: LSRKHConvergenceConfig,
    *,
    variant_slug: str,
    preset: str,
) -> str:
    tau_interior, tau_qb = resolve_effective_taus(
        tau=config.tau,
        tau_interior=config.tau_interior,
        tau_qb=config.tau_qb,
    )

    stem = (
        f"lsrk_h_conv_time_cli_{_test_function_slug(config.test_function_mode)}"
        f"_tf{_tf_values_label(config.tf_values)}"
        f"_table1_order{config.order}_N{config.N}_{config.diagonal}"
        f"_face{str(config.face_order_mode).strip().lower()}"
        f"_{str(config.surface_inverse_mass_mode).strip().lower()}"
        f"_{str(config.physical_boundary_mode).strip().lower()}"
        f"_taui{_tau_label(tau_interior)}"
        f"_tauqb{_tau_label(tau_qb)}"
        f"_{variant_slug}"
    )

    trace_mode = str(config.interior_trace_mode).strip().lower()
    if trace_mode != "exchange":
        stem += f"_{trace_mode}"
    if preset != "full":
        stem += f"_{preset}"
    return stem


def _flatten_results_by_tf(results_by_tf: dict[float, list[dict]]) -> list[dict]:
    flat_rows: list[dict] = []
    for tf in sorted(float(k) for k in results_by_tf.keys()):
        rows = sorted(results_by_tf[tf], key=lambda row: int(row["nx"]))
        for row in rows:
            merged = {"tf_scan": float(tf)}
            for key, value in row.items():
                merged[key] = value
            flat_rows.append(merged)
    return flat_rows


def _build_time_scan_summary_rows(results_by_tf: dict[float, list[dict]]) -> list[dict]:
    summary_rows: list[dict] = []
    for tf in sorted(float(k) for k in results_by_tf.keys()):
        rows = sorted(results_by_tf[tf], key=lambda row: int(row["nx"]))
        if len(rows) == 0:
            continue

        reached_tf_all = all(bool(row.get("reached_tf", False)) for row in rows)
        finest_row = max(rows, key=lambda row: int(row["nx"]))
        total_elapsed_sec = float(sum(float(row.get("elapsed_sec", 0.0)) for row in rows))

        l2_finest = float(finest_row["L2_error"])
        linf_finest = float(finest_row["Linf_error"])
        if not reached_tf_all:
            l2_finest = float(finest_row["L2_error_at_stop"])
            linf_finest = float(finest_row["Linf_error_at_stop"])

        p_l2_last = float("nan")
        p_l2_avg = float("nan")
        p_linf_last = float("nan")
        p_linf_avg = float("nan")

        if len(rows) < 2:
            rate_status = "insufficient_mesh_levels"
        elif not reached_tf_all:
            rate_status = "strict_final_time_not_met"
        else:
            h_values = [float(row["h"]) for row in rows]
            l2_values = [float(row["L2_error"]) for row in rows]
            linf_values = [float(row["Linf_error"]) for row in rows]
            l2_rates = compute_convergence_rate(l2_values, h_values)
            linf_rates = compute_convergence_rate(linf_values, h_values)

            finite_l2 = [float(v) for v in l2_rates if np.isfinite(float(v))]
            finite_linf = [float(v) for v in linf_rates if np.isfinite(float(v))]
            if len(finite_l2) == 0 or len(finite_linf) == 0:
                rate_status = "rate_not_finite"
            else:
                rate_status = "ok"
                p_l2_last = float(finite_l2[-1])
                p_l2_avg = float(np.mean(finite_l2))
                p_linf_last = float(finite_linf[-1])
                p_linf_avg = float(np.mean(finite_linf))

        summary_rows.append(
            {
                "tf": float(tf),
                "reached_tf_all": bool(reached_tf_all),
                "num_mesh_levels": int(len(rows)),
                "finest_n": int(finest_row["nx"]),
                "L2_error_finest": float(l2_finest),
                "Linf_error_finest": float(linf_finest),
                "p_L2_last": float(p_l2_last),
                "p_L2_avg": float(p_l2_avg),
                "p_Linf_last": float(p_linf_last),
                "p_Linf_avg": float(p_linf_avg),
                "total_elapsed_sec": float(total_elapsed_sec),
                "rate_status": str(rate_status),
            }
        )

    return summary_rows


def _save_time_scan_summary_csv(path: Path, rows: list[dict]) -> None:
    if len(rows) == 0:
        raise ValueError("time-scan summary rows are empty.")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_scan_rate(v: float) -> str:
    return "   -   " if not np.isfinite(float(v)) else f"{float(v):7.3f}"


def _print_time_scan_summary(rows: list[dict], *, variant_label: str) -> None:
    print()
    print(f"[time-cli] tf-scan summary | variant={variant_label}")
    header = (
        f"{'tf':>10s} {'all_tf':>8s} {'nmax':>6s} "
        f"{'L2_finest':>14s} {'pL2_last':>9s} {'pL2_avg':>9s} "
        f"{'Linf_finest':>14s} {'pLinf_last':>11s} {'pLinf_avg':>10s} "
        f"{'time(s)':>9s} {'status':>24s}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{float(row['tf']):10.6f} {str(bool(row['reached_tf_all'])):>8s} {int(row['finest_n']):6d} "
            f"{float(row['L2_error_finest']):14.6e} {_fmt_scan_rate(float(row['p_L2_last'])):>9s} {_fmt_scan_rate(float(row['p_L2_avg'])):>9s} "
            f"{float(row['Linf_error_finest']):14.6e} {_fmt_scan_rate(float(row['p_Linf_last'])):>11s} {_fmt_scan_rate(float(row['p_Linf_avg'])):>10s} "
            f"{float(row['total_elapsed_sec']):9.2f} {str(row['rate_status']):>24s}"
        )


def _save_time_scan_plot(
    path: Path,
    rows: list[dict],
    *,
    variant_label: str,
    test_function_label: str,
) -> None:
    if len(rows) == 0:
        raise ValueError("time-scan summary rows are empty.")

    tfs = np.asarray([float(row["tf"]) for row in rows], dtype=float)
    l2_finest = np.asarray([float(row["L2_error_finest"]) for row in rows], dtype=float)
    linf_finest = np.asarray([float(row["Linf_error_finest"]) for row in rows], dtype=float)
    p_l2_last = np.asarray([float(row["p_L2_last"]) for row in rows], dtype=float)
    p_linf_last = np.asarray([float(row["p_Linf_last"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), constrained_layout=True)

    ax_rate = axes[0]
    rate_l2_mask = np.isfinite(p_l2_last)
    rate_linf_mask = np.isfinite(p_linf_last)
    if np.any(rate_l2_mask):
        ax_rate.plot(
            tfs[rate_l2_mask],
            p_l2_last[rate_l2_mask],
            marker="o",
            linewidth=1.8,
            label="p_L2(last)",
        )
    if np.any(rate_linf_mask):
        ax_rate.plot(
            tfs[rate_linf_mask],
            p_linf_last[rate_linf_mask],
            marker="s",
            linewidth=1.8,
            label="p_Linf(last)",
        )
    if (not np.any(rate_l2_mask)) and (not np.any(rate_linf_mask)):
        ax_rate.text(0.5, 0.5, "rates unavailable", transform=ax_rate.transAxes, ha="center", va="center")
    ax_rate.set_xlabel("tf")
    ax_rate.set_ylabel("rate")
    ax_rate.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_rate.legend(loc="best")

    ax_err = axes[1]
    l2_mask = np.isfinite(l2_finest) & (l2_finest > 0.0)
    linf_mask = np.isfinite(linf_finest) & (linf_finest > 0.0)
    if np.any(l2_mask):
        ax_err.semilogy(
            tfs[l2_mask],
            l2_finest[l2_mask],
            marker="o",
            linewidth=1.8,
            label="L2(finest n)",
        )
    if np.any(linf_mask):
        ax_err.semilogy(
            tfs[linf_mask],
            linf_finest[linf_mask],
            marker="s",
            linewidth=1.8,
            label="Linf(finest n)",
        )
    ax_err.set_xlabel("tf")
    ax_err.set_ylabel("error")
    ax_err.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_err.legend(loc="best")

    fig.suptitle(f"LSRK h-convergence time scan ({test_function_label}) | {variant_label}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _emit_time_cli_outputs(
    *,
    output_dir: Path,
    config: LSRKHConvergenceConfig,
    preset: str,
    results_by_tf: dict[float, list[dict]],
    variant_slug: str,
    variant_label: str,
    test_function_label: str,
) -> None:
    summary_rows = _build_time_scan_summary_rows(results_by_tf)
    _print_time_scan_summary(summary_rows, variant_label=variant_label)

    stem = _time_scan_output_stem(
        config,
        variant_slug=variant_slug,
        preset=preset,
    )
    merged_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    plot_path = output_dir / f"{stem}.png"

    merged_rows = _flatten_results_by_tf(results_by_tf)
    save_results_csv(merged_rows, str(merged_path))
    _save_time_scan_summary_csv(summary_path, summary_rows)
    _save_time_scan_plot(
        plot_path,
        summary_rows,
        variant_label=variant_label,
        test_function_label=test_function_label,
    )

    print("[OK] wrote " + str(merged_path))
    print("[OK] wrote " + str(summary_path))
    print("[OK] wrote " + str(plot_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LSRK h-convergence study with quick/full presets.",
    )
    parser.add_argument(
        "--preset",
        choices=("quick", "full", "upstream-pbc"),
        default="quick",
        help="quick: faster sanity run, full: complete run, upstream-pbc: align with the upstream PBC notebook setup",
    )
    parser.add_argument(
        "--qb-correction",
        choices=("off", "on", "compare"),
        default=None,
        help="exact-source mode: off=baseline, on=RK-stage exact-source correction only, compare=run baseline and correction",
    )
    parser.add_argument(
        "--compare-qb-correction",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qb-correction-only",
        action="store_true",
        help=argparse.SUPPRESS,
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
        choices=("exact_qb", "opposite_boundary", "periodic_vmap"),
        default="exact_qb",
        help="boundary exterior-state mode; periodic_vmap reproduces the upstream notebook's coordinate-matched periodic trace overwrite",
    )
    parser.add_argument(
        "--interior-trace-mode",
        choices=("exchange", "exact_trace"),
        default="exchange",
        help="interior-face mode: exchange uses connectivity, exact_trace uses exact exterior trace on interior faces",
    )
    parser.add_argument(
        "--face-order-mode",
        choices=("triangle", "simplex"),
        default="triangle",
        help="surface face-index convention: triangle(default) or simplex",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="shared surface numerical flux parameter; used for both tau-interior and tau-qb unless overridden",
    )
    parser.add_argument(
        "--tau-interior",
        type=float,
        default=None,
        help="surface numerical flux parameter for interior faces and non-exact_qb exterior traces",
    )
    parser.add_argument(
        "--tau-qb",
        type=float,
        default=None,
        help="surface numerical flux parameter for physical-boundary-mode=exact_qb faces only",
    )
    parser.add_argument(
        "--diagonal",
        choices=("main", "anti"),
        default=None,
        help="override the preset triangle split orientation",
    )
    parser.add_argument(
        "--mesh-levels",
        type=int,
        nargs="+",
        default=None,
        help="override the preset mesh subdivision list, for example: --mesh-levels 2 4 8 16 32",
    )
    parser.add_argument(
        "--tf-values",
        type=float,
        nargs="+",
        default=None,
        help="override the preset final times, for example: --tf-values 10",
    )
    parser.add_argument(
        "--use-numba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable numba acceleration (default: enabled)",
    )
    parser.add_argument(
        "--time-cli",
        action="store_true",
        help=(
            "enable time-scan outputs across tf-values: terminal summary, merged CSV, "
            "summary CSV, and PNG"
        ),
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
    tau: float,
    tau_interior: float | None,
    tau_qb: float | None,
    test_function_mode: str,
    physical_boundary_mode: str,
    interior_trace_mode: str,
    face_order_mode: str,
    use_numba: bool,
) -> LSRKHConvergenceConfig:
    if preset == "upstream-pbc":
        return LSRKHConvergenceConfig(
            table_name="table1",
            order=4,
            N=4,
            diagonal="anti",
            mesh_levels=(1, 2, 4, 8, 16, 32, 64),
            cfl=0.05,
            tf_values=(1.0,),
            tau=float(tau),
            tau_interior=None if tau_interior is None else float(tau_interior),
            tau_qb=None if tau_qb is None else float(tau_qb),
            use_numba=bool(use_numba),
            surface_inverse_mass_mode=surface_inverse_mass_mode,
            surface_backend="face-major",
            interior_trace_mode=interior_trace_mode,
            face_order_mode=face_order_mode,
            test_function_mode=test_function_mode,
            physical_boundary_mode=physical_boundary_mode,
            use_surface_cache=True,
            verbose=True,
        )

    if preset == "quick":
        return LSRKHConvergenceConfig(
            table_name="table1",
            order=4,
            N=4,
            diagonal="anti",
            mesh_levels=(1, 2, 4, 8, 16),
            cfl=0.05,
            tf_values=(1.0,),
            tau=float(tau),
            tau_interior=None if tau_interior is None else float(tau_interior),
            tau_qb=None if tau_qb is None else float(tau_qb),
            use_numba=bool(use_numba),
            surface_inverse_mass_mode=surface_inverse_mass_mode,
            surface_backend="face-major",
            interior_trace_mode=interior_trace_mode,
            face_order_mode=face_order_mode,
            test_function_mode=test_function_mode,
            physical_boundary_mode=physical_boundary_mode,
            use_surface_cache=True,
            verbose=True,
        )

    return LSRKHConvergenceConfig(
        table_name="table1",
        order=4,
        N=4,
        diagonal="anti",
        mesh_levels=(1, 2, 4, 8, 16, 32),
        cfl=1.0,
        tf_values=(np.pi*3,),
        tau=float(tau),
        tau_interior=None if tau_interior is None else float(tau_interior),
        tau_qb=None if tau_qb is None else float(tau_qb),
        use_numba=bool(use_numba),
        surface_inverse_mass_mode=surface_inverse_mass_mode,
        surface_backend="face-major",
        interior_trace_mode=interior_trace_mode,
        face_order_mode=face_order_mode,
        test_function_mode=test_function_mode,
        physical_boundary_mode=physical_boundary_mode,
        use_surface_cache=True,
        verbose=True,
    )


def _apply_config_overrides(
    config: LSRKHConvergenceConfig,
    args: argparse.Namespace,
) -> tuple[LSRKHConvergenceConfig, bool]:
    overridden = False
    out = config

    if args.diagonal is not None:
        out = replace(out, diagonal=str(args.diagonal).strip().lower())
        overridden = True

    if args.mesh_levels is not None:
        mesh_levels = tuple(int(n) for n in args.mesh_levels)
        if len(mesh_levels) == 0:
            raise ValueError("--mesh-levels must provide at least one subdivision level.")
        out = replace(out, mesh_levels=mesh_levels)
        overridden = True

    if args.tf_values is not None:
        tf_values = tuple(float(tf) for tf in args.tf_values)
        if len(tf_values) == 0:
            raise ValueError("--tf-values must provide at least one final time.")
        out = replace(out, tf_values=tf_values)
        overridden = True

    return out, overridden


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

    output_dir = experiments_output_dir(__file__, "lsrk_h_convergence")

    config = _build_config(
        args.preset,
        args.surface_inverse_mass_mode,
        tau=args.tau,
        tau_interior=args.tau_interior,
        tau_qb=args.tau_qb,
        test_function_mode=args.test_function,
        physical_boundary_mode=args.physical_boundary_mode,
        interior_trace_mode=args.interior_trace_mode,
        face_order_mode=args.face_order_mode,
        use_numba=bool(args.use_numba),
    )
    config, has_overrides = _apply_config_overrides(config, args)
    tau_interior_eff, tau_qb_eff = resolve_effective_taus(
        tau=config.tau,
        tau_interior=config.tau_interior,
        tau_qb=config.tau_qb,
    )

    study = run_lsrk_study(config, qb_mode=qb_mode)

    test_function_label = _test_function_slug(config.test_function_mode)
    print(f"[run] preset={args.preset}")
    if has_overrides:
        print("[run] preset_overrides=on")
    print(f"[run] surface_inverse_mass_mode={config.surface_inverse_mass_mode}")
    print(f"[run] use_numba={bool(config.use_numba)}")
    print(f"[run] test_function_mode={config.test_function_mode}")
    print(f"[run] physical_boundary_mode={config.physical_boundary_mode}")
    print(f"[run] interior_trace_mode={config.interior_trace_mode}")
    print(f"[run] face_order_mode={config.face_order_mode}")
    print(f"[run] diagonal={config.diagonal}")
    print(f"[run] mesh_levels={config.mesh_levels}")
    print(f"[run] tf_values={config.tf_values}")
    print(f"[run] tau={config.tau:g}")
    print(f"[run] tau_interior={tau_interior_eff:g}")
    print(f"[run] tau_qb={tau_qb_eff:g}")
    print("[run] tau_role=penalty uses tau_interior on interior/non-exact_qb faces and tau_qb on physical-boundary-mode=exact_qb faces")
    print(f"[run] qb_correction={qb_mode}")

    if qb_mode == "compare":
        print("[run] compare_qb_correction=on")
        baseline_map = study["baseline"]
        corrected_map = study["rk_stage_correction"]

        for tf, baseline_results in baseline_map.items():
            corrected_results = corrected_map[tf]

            print()
            print_results_table(
                baseline_results,
                title=(
                    f"LSRK h-convergence ({test_function_label}) | baseline | "
                    f"trace={config.interior_trace_mode} | tf={tf:g}, CFL={config.cfl:g}, "
                    f"tau_i={tau_interior_eff:g}, tau_qb={tau_qb_eff:g}"
                ),
            )
            print()
            print_results_table(
                corrected_results,
                title=(
                    f"LSRK h-convergence ({test_function_label}) | rk-stage exact-source correction | "
                    f"trace={config.interior_trace_mode} | tf={tf:g}, CFL={config.cfl:g}, "
                    f"tau_i={tau_interior_eff:g}, tau_qb={tau_qb_eff:g}"
                ),
            )

            _print_compare_summary(baseline_results, corrected_results, tf=tf)

            base_name = _output_stem(config, tf)
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

        if args.time_cli:
            print()
            _emit_time_cli_outputs(
                output_dir=output_dir,
                config=config,
                preset=str(args.preset),
                results_by_tf=baseline_map,
                variant_slug="baseline",
                variant_label="baseline",
                test_function_label=test_function_label,
            )
            print()
            _emit_time_cli_outputs(
                output_dir=output_dir,
                config=config,
                preset=str(args.preset),
                results_by_tf=corrected_map,
                variant_slug="rkstage_qb",
                variant_label="rk-stage exact-source correction",
                test_function_label=test_function_label,
            )
        return

    if qb_mode == "on":
        print("[run] qb_correction_only=on")
        all_results = study["rk_stage_correction"]

        for tf, results in all_results.items():
            print()
            print_results_table(
                results,
                title=(
                    f"LSRK h-convergence ({test_function_label}) | rk-stage exact-source correction | "
                    f"trace={config.interior_trace_mode} | tf={tf:g}, CFL={config.cfl:g}, "
                    f"tau_i={tau_interior_eff:g}, tau_qb={tau_qb_eff:g}"
                ),
            )

            csv_name = _output_stem(config, tf) + "_rkstage_qb_only.csv"
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

        if args.time_cli:
            print()
            _emit_time_cli_outputs(
                output_dir=output_dir,
                config=config,
                preset=str(args.preset),
                results_by_tf=all_results,
                variant_slug="rkstage_qb_only",
                variant_label="rk-stage exact-source correction",
                test_function_label=test_function_label,
            )
        return

    all_results = study["baseline"]

    for tf, results in all_results.items():
        print()
        print_results_table(
            results,
            title=(
                f"LSRK h-convergence ({test_function_label}) | trace={config.interior_trace_mode} | "
                f"tf={tf:g}, CFL={config.cfl:g}, tau_i={tau_interior_eff:g}, tau_qb={tau_qb_eff:g}"
            ),
        )

        csv_name = _output_stem(config, tf) + ".csv"
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

    if args.time_cli:
        print()
        _emit_time_cli_outputs(
            output_dir=output_dir,
            config=config,
            preset=str(args.preset),
            results_by_tf=all_results,
            variant_slug="baseline",
            variant_label="baseline",
            test_function_label=test_function_label,
        )


if __name__ == "__main__":
    main()
