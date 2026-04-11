from __future__ import annotations

from pathlib import Path

from experiments.rhs_exchange_benchmark import (
    RHSExchangeBenchmarkConfig,
    run_rhs_exchange_benchmark,
    print_results_table,
    save_results_csv,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "experiments_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RHSExchangeBenchmarkConfig(
        order=4,
        N=4,
        diagonal="anti",
        mesh_levels=(8, 16, 32),
        repeats=80,
        warmup=8,
        backends=("numpy", "auto", "numba"),
        modes=("full", "perf"),
        verbose=True,
    )

    results = run_rhs_exchange_benchmark(config)

    print()
    print_results_table(results)

    csv_path = output_dir / "rhs_exchange_benchmark_table1_order4_N4_anti.csv"
    save_results_csv(results, str(csv_path))

    print()
    print("[OK] wrote " + str(csv_path))


if __name__ == "__main__":
    main()
