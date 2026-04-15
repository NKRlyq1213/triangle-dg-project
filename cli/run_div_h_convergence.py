from __future__ import annotations

from experiments.div_h_convergence import (
    DivHConvergenceConfig,
    print_results_table,
    run_div_h_convergence,
    save_results_csv,
)
from experiments.output_paths import experiments_output_dir


def main() -> None:
    output_dir = experiments_output_dir(__file__, "div_h_convergence")

    config = DivHConvergenceConfig(
        table_name="table1",
        order=4,
        N=4,
        diagonal="anti",
        mesh_levels=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        eval_resolution=12,
        field_case="gaussian",
        x0=0.35,
        y0=0.55,
        sigma=0.16,
        coeff_case="constant_one",
        verbose=True,
    )

    results = run_div_h_convergence(config)
    csv_path = output_dir / "div_h_convergence_table1_order4_N4_anti.csv"

    print()
    print_results_table(results)

    save_results_csv(results, str(csv_path))
    print()
    print("[OK] wrote " + str(csv_path))


if __name__ == "__main__":
    main()
