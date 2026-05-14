from __future__ import annotations

import argparse
import math

from experiments.manifold_sbp_chan2019_convergence import (
    ManifoldSBPChan2019Config,
    extract_time_histories,
    print_results_table,
    run_manifold_sbp_chan2019_convergence,
    save_results_csv,
    save_time_history_csv,
)
from experiments.output_paths import experiments_output_dir
from visualization.manifold_diagnostics import (
    plot_manifold_l2_error_vs_time,
    plot_manifold_lsrk_convergence,
    plot_manifold_mass_error_vs_time,
)


def _parse_levels(text: str) -> tuple[int, ...]:
    levels = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not levels:
        raise argparse.ArgumentTypeError("mesh levels must not be empty.")
    return levels


def _float_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(float(value)).replace(".", "p").replace("-", "m")


def _slug(args: argparse.Namespace) -> str:
    base = f"{args.field_case}_{args.flux_type}_tf{_float_slug(args.tf)}_cfl{_float_slug(args.cfl)}"
    if args.flux_type == "lax_friedrichs":
        base += f"_a{_float_slug(args.alpha_lf)}"
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Table1 k=4 Chan-style SBP manifold DG convergence study.",
    )
    parser.add_argument("--mesh-levels", type=_parse_levels, default=(2, 4, 8))
    parser.add_argument("--tf", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=math.pi / 4.0)
    parser.add_argument("--gaussian-width", type=float, default=1.0 / math.sqrt(10.0))
    parser.add_argument("--field-case", choices=("gaussian", "constant"), default="gaussian")
    parser.add_argument(
        "--flux-type",
        choices=("upwind", "central", "lax_friedrichs"),
        default="upwind",
    )
    parser.add_argument("--alpha-lf", type=float, default=1.0)
    parser.add_argument("--constant-value", type=float, default=1.0)
    parser.add_argument("--use-numba", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action="store_true", help="Save optional diagnostic figures.")
    args = parser.parse_args()

    output_dir = experiments_output_dir(__file__, "manifold_sbp_chan2019_convergence")

    config = ManifoldSBPChan2019Config(
        mesh_levels=args.mesh_levels,
        R=args.R,
        u0=args.u0,
        alpha0=args.alpha0,
        cfl=args.cfl,
        tf=args.tf,
        gaussian_width=args.gaussian_width,
        field_case=args.field_case,
        flux_type=args.flux_type,
        alpha_lf=args.alpha_lf,
        constant_value=args.constant_value,
        use_numba=args.use_numba,
        record_history=args.record_history,
        verbose=True,
    )

    results = run_manifold_sbp_chan2019_convergence(config)
    histories = extract_time_histories(results)
    slug = _slug(args)

    csv_path = output_dir / f"manifold_sbp_chan2019_table1_k4_{slug}.csv"
    history_csv_path = output_dir / f"manifold_sbp_chan2019_time_history_table1_k4_{slug}.csv"

    print()
    print_results_table(results)
    save_results_csv(results, csv_path)
    print()
    print("[OK] wrote " + str(csv_path))

    if histories:
        save_time_history_csv(results, history_csv_path)
        print("[OK] wrote " + str(history_csv_path))

    if args.plot:
        fig_path = output_dir / f"manifold_sbp_chan2019_convergence_table1_k4_{slug}.png"
        error_fig_path = output_dir / f"manifold_sbp_chan2019_error_vs_time_table1_k4_{slug}.png"
        mass_fig_path = output_dir / f"manifold_sbp_chan2019_mass_vs_time_table1_k4_{slug}.png"
        plot_manifold_lsrk_convergence(results, fig_path)
        plot_manifold_l2_error_vs_time(histories, error_fig_path)
        plot_manifold_mass_error_vs_time(histories, mass_fig_path)
        print("[OK] wrote " + str(fig_path))
        print("[OK] wrote " + str(error_fig_path))
        print("[OK] wrote " + str(mass_fig_path))


if __name__ == "__main__":
    main()

