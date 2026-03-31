from __future__ import annotations

from experiments.field_h_convergence import (
    FieldHConvergenceConfig,
    run_field_h_convergence,
    print_results_table,
    save_results_csv,
)
from pathlib import Path

def main():
    output_dir = Path(r"C:\Users\user\Desktop\triangle-dg-project\experiments_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = FieldHConvergenceConfig(
        table_name="table2",
        order=4,
        N=4,
        diagonal="anti",
        mesh_levels=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        eval_resolution=12,
        x0=0.5,
        y0=0.5,
        sigma=0.16,
        verbose=True,
    )

    results = run_field_h_convergence(config)

    csv_path = output_dir / "field_h_convergence_table2_order4_N4_anti.csv"

    print()
    print_results_table(results)

    save_results_csv(results, str(csv_path))
    print()
    print("[OK] wrote field_h_convergence_table2_order4_N4_anti.csv")


if __name__ == "__main__":
    main()