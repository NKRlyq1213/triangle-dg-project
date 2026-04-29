from __future__ import annotations

import argparse
import math

from experiments.manifold_div_h_convergence import (
    ManifoldDivHConvergenceConfig,
    print_results_table,
    run_manifold_div_h_convergence,
    save_results_csv,
)
from experiments.output_paths import experiments_output_dir
from visualization.manifold_diagnostics import plot_manifold_convergence


def _parse_levels(text: str) -> tuple[int, ...]:
    levels = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not levels:
        raise argparse.ArgumentTypeError("mesh levels must not be empty.")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Table1 k=4 3D manifold sphere constant-divergence h-study."
    )
    parser.add_argument("--mesh-levels", type=_parse_levels, default=(2, 4, 8, 16, 32))
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=-math.pi / 4.0)
    args = parser.parse_args()

    output_dir = experiments_output_dir(__file__, "manifold_div_h_convergence")
    config = ManifoldDivHConvergenceConfig(
        mesh_levels=args.mesh_levels,
        R=args.R,
        u0=args.u0,
        alpha0=args.alpha0,
        verbose=True,
    )

    results = run_manifold_div_h_convergence(config)
    csv_path = output_dir / "manifold_div_h_convergence_table1_k4.csv"
    fig_path = output_dir / "manifold_div_h_convergence_table1_k4.png"

    print()
    print_results_table(results)

    save_results_csv(results, csv_path)
    plot_manifold_convergence(results, fig_path)

    print()
    print("[OK] wrote " + str(csv_path))
    print("[OK] wrote " + str(fig_path))


if __name__ == "__main__":
    main()
