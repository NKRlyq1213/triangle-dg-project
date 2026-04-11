from __future__ import annotations

from pathlib import Path

from experiments.lsrk_h_convergence import (
    LSRKHConvergenceConfig,
    run_lsrk_h_convergence,
    print_results_table,
    save_results_csv,
)


def _tf_label(tf: float) -> str:
    if float(tf).is_integer():
        return str(int(tf))
    return str(tf).replace(".", "p")


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "experiments_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = LSRKHConvergenceConfig(
        table_name="table1",
        order=4,
        N=4,
        diagonal="anti",
        mesh_levels=(1, 2, 4, 8, 16, 32),
        cfl=1.0,
        tf_values=(1.0, 2.0),
        tau=0.0,
        use_numba=True,
        verbose=True,
    )

    all_results = run_lsrk_h_convergence(config)

    for tf, results in all_results.items():
        print()
        print_results_table(results, title=f"LSRK h-convergence (sinx) | tf={tf:g}, CFL={config.cfl:g}")

        csv_name = (
            f"lsrk_h_convergence_sinx_tf{_tf_label(tf)}_"
            f"table1_order{config.order}_N{config.N}_{config.diagonal}.csv"
        )
        csv_path = output_dir / csv_name
        save_results_csv(results, str(csv_path))
        print()
        print("[OK] wrote " + str(csv_path))


if __name__ == "__main__":
    main()
