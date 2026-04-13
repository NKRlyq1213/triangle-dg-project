# triangle_dg_project

Triangle-domain polynomial reconstruction and DG-oriented operators on the reference triangle.

## Overview

This repository provides reusable numerical components for triangle-based DG workflows.
It focuses on basis construction, quadrature handling, differentiation/reconstruction
operators, geometric mappings, exchange-based RHS assembly, and experiment scripts.

Current state is a DG-ready foundation, not yet a complete production solver.

## Installation

Recommended Python version:

- Python 3.10 or later

Core dependencies:

- numpy>=1.26
- scipy>=1.11
- matplotlib>=3.8
- pytest>=7.4

Install dependencies:

```bash
pip install -r requirements.txt
```

Editable install:

```bash
pip install -e .
```

Optional performance extra (Numba):

```bash
pip install -e .[perf]
```

## Repository Structure

```text
basis/
    Jacobi and simplex basis utilities, mode indexing.

data/
    Table-1/Table-2 quadrature rules and edge rules.

geometry/
    Reference triangle, affine mapping, metrics, mesh connectivity,
    sampling, and display-point helpers.

operators/
    Vandermonde, mass, differentiation, reconstruction,
    trace/exchange, and conservative split-form RHS operators.

problems/
    Analytic fields and initial-condition wrappers.

time_integration/
    LSRK54 integrator and CFL utilities.

visualization/
    Plotting utilities for nodes, scalar fields, radial/3D diagnostics.

experiments/
    Experiment backends for h-convergence and RHS benchmarks.

demo/
    foundations/
        smoke checks and connectivity/trace/exchange demos.
    visualization/
        plotting and reconstruction demos.
    advanced/
        differentiation and split diagnostics.
    experiments/
        executable experiment runners.

tests/
    unit/
    integration/
    regression/
    test_*.py at tests root (core exactness and functional checks)
```

## Quick Start

Load a quadrature rule and build Vandermonde:

```python
from data import load_table2_rule
from operators import vandermonde2d

rule = load_table2_rule(4)
rs = rule["rs"]
V = vandermonde2d(4, rs[:, 0], rs[:, 1])
```

Weighted modal reconstruction:

```python
from data import load_table2_rule
from operators import vandermonde2d, fit_modal_coefficients_weighted
from problems import ground_truth_function
from geometry import reference_triangle_area

rule = load_table2_rule(4)
rs = rule["rs"]
w = rule["ws"]
u = ground_truth_function("smooth_bump", rs[:, 0], rs[:, 1])
V = vandermonde2d(4, rs[:, 0], rs[:, 1])

coeffs = fit_modal_coefficients_weighted(
    u_nodes=u,
    V=V,
    weights=w,
    area=reference_triangle_area(),
)
```

## Demo Commands

Run commands from the project root.

### Foundations

```bash
python -m demo.foundations.smoke_phase1
python -m demo.foundations.smoke_phase2_edge
python -m demo.foundations.smoke_phase3_ops
python -m demo.foundations.smoke_phase4_boundary
python -m demo.foundations.smoke_table_rules
python -m demo.foundations.demo_connectivity_step1
python -m demo.foundations.demo_trace_policy_step2
python -m demo.foundations.demo_exchange_step3
python -m demo.foundations.demo_compare_sampling
```

### Visualization

```bash
python -m demo.visualization.demo_plot_nodes --table both --orders 1 2 3 4
python -m demo.visualization.demo_reconstruct_field --table table1 --order 4 --N 4 --case smooth_bump
python -m demo.visualization.demo_error_field --table table2 --order 4 --N 4 --case smooth_bump
python -m demo.visualization.demo_3d --table table1 --order 4 --N 4 --case smooth_bump
python -m demo.visualization.demo_plot_mesh_nodes
python -m demo.visualization.demo_radial_sampling_table2
```

### Advanced

```bash
python -m demo.advanced.demo_differentiation_weighted
python -m demo.advanced.demo_split_gaussian_field
```

### Experiments

```bash
python -m demo.experiments.run_field_h_convergence
python -m demo.experiments.run_div_h_convergence
python -m demo.experiments.run_rhs_exchange_benchmark
python -m demo.experiments.run_lsrk_h_convergence --preset quick
python -m demo.experiments.run_lsrk_h_convergence --preset full
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction compare
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction on
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction on --surface-inverse-mass-mode projected
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction on --state-projection on --projection-mode both --projection-frequency rhs
```

## LSRK qB Correction Workflow

The LSRK runner supports three user-facing modes:

- Baseline:

```bash
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction off
```

- Compare baseline vs correction:

```bash
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction compare
```

- Correction-only:

```bash
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction on
```

Flag notes:

- `--preset quick`: mesh_levels=(1,2,4,8,16), tf_values=(2*pi,)
- `--preset full`: mesh_levels=(1,2,4,8,16,32), tf_values=(2*pi,)
- `--qb-correction`: `off` (baseline), `on` (rk-stage correction only), `compare` (run both)
- legacy aliases are still supported: `--compare-qb-correction` and `--qb-correction-only`
- `--surface-inverse-mass-mode`: `projected` (default), `diagonal`
- `--state-projection`: `off` (default), `on`
- `--projection-mode`: `pre`, `post`, `both` (active when `--state-projection on`)
- `--projection-frequency`: `rhs`, `step` (active when `--state-projection on`)

Relevant `LSRKHConvergenceConfig` knobs in the backend:

- `enforce_polynomial_projection`: enable/disable nodal state projection
- `projection_mode`: `both`, `pre`, `post`
- `projection_frequency`: `rhs`, `step`
- `surface_inverse_mass_mode`: `projected`, `diagonal`
- `surface_backend`: currently configured as `face-major` in the runner presets
- `use_sinx_rk_stage_boundary_correction`: built-in notebook-style stage correction
- `q_boundary_correction`: custom callback hook
- `q_boundary_correction_mode`: `inflow`, `boundary`, `all`

## Output Files

Generated outputs are written from project root:

- Figures: `photo/`
- CSV tables: `experiments_outputs/`

Common CSV names:

- `field_h_convergence_table1_order4_N4_anti.csv`
- `div_h_convergence_table1_order4_N4_anti.csv`
- `rhs_exchange_benchmark_table1_order4_N4_anti.csv`
- `lsrk_h_convergence_sinx_tf{tf}_table1_order4_N4_anti.csv`

LSRK mode-specific suffixes:

- Quick preset baseline: `_quick`
- Compare mode outputs: `_baseline`, `_rkstage_qb` (and `_quick_...` variants)
- Correction-only outputs: `_rkstage_qb_only` (or `_rkstage_qb_only_quick`)

LSRK CSV rows include mesh/time/error metrics and mode metadata such as:

- `projection_enabled`
- `projection_mode`, `projection_frequency`
- `surface_inverse_mass_mode`
- `q_boundary_correction_kind`, `q_boundary_correction_mode`

## Testing and Validation

Typical commands:

```bash
pytest tests/unit
pytest tests/integration
pytest tests/regression
pytest tests
```

If you want routine runs without LSRK-focused cases:

```bash
pytest tests -k "not lsrk"
```

## Current Scope

Implemented:

- basis and quadrature utilities
- Vandermonde, mass, differentiation, reconstruction operators
- affine geometry and connectivity helpers
- exchange-based split-form RHS building blocks
- LSRK experiment workflows and visualization demos

Still evolving:

- full multi-physics production solver pipeline
- broader long-horizon regression coverage for all workflows
- finalized and stable public API boundaries across all helper modules

## Development Principles

1. Correctness before optimization
2. Exactness and consistency checks before large demos
3. Geometry/metric consistency before mapped-operator usage
4. Clear module interfaces before solver-level expansion
