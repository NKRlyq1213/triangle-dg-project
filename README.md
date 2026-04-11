# triangle_dg_project

Triangle-domain polynomial reconstruction and DG-oriented operators on the reference triangle.

## Overview

This repository provides numerical building blocks for triangle-based polynomial approximation and DG-oriented operator construction.

Current scope focuses on basis functions, quadrature handling, reconstruction, differentiation, affine mappings, edge evaluation operators, and visualization utilities on the reference triangle.

At this stage, the project is a numerical foundation for later DG development, not a full production DG solver.

## Features

- Triangle quadrature handling from Table 1 and Table 2 datasets
- Symmetry-orbit expansion into full quadrature point sets
- Classical and orthonormal Jacobi polynomials
- 2D simplex basis functions on the reference triangle
- Mode indexing utilities
- Vandermonde and gradient Vandermonde matrices
- Quadrature-based mass matrices
- Square nodal and weighted modal reconstruction
- Reference-space differentiation operators
- Mapped conservative and split divergence operators
- Edge evaluation and volume-to-edge trace operators
- Reference-to-physical affine mapping utilities
- Visualization tools for nodes, fields, and radial diagnostics

## Repository Structure

```text
data/
    Quadrature tables, rule loaders, and edge rules.

basis/
    Jacobi polynomials, simplex basis functions, and mode indexing.

geometry/
    Reference triangle utilities, barycentric mappings, mesh helpers,
    affine maps, geometric factors, sampling, and display-point helpers.

operators/
    Vandermonde construction, mass matrices, differentiation operators,
    reconstruction routines, boundary and edge operators, and split-form tools.

problems/
    Analytic test fields and initial-condition wrappers.

visualization/
    Plotting utilities for nodes, scalar fields, and radial diagnostics.

demo/
    foundations/
        Smoke checks and connectivity/trace/exchange step demos.
    visualization/
        Node, reconstruction, error, 3D, mesh-node, and radial demos.
    advanced/
        Differentiation and split-form diagnostics.
    experiments/
        h-convergence execution scripts.

tests/
    unit/
        Geometry and core component tests.
    integration/
        Multi-module operator and RHS consistency tests.
    regression/
        LSRK regression tests (kept but currently deferred in routine validation).
    test_*.py (root stubs)
        Placeholder empty test files intentionally kept at tests root.
```

## Installation

Recommended Python version:

- Python 3.10 or later

Minimal dependencies:

- numpy>=1.26
- scipy>=1.11
- matplotlib>=3.8
- pytest>=7.4

Install with:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

Optional performance extra (Numba backend for RHS exchange kernels):

```bash
pip install -e .[perf]
```

## Quick Examples

### Load a quadrature rule

```python
from data import load_table2_rule

rule = load_table2_rule(4)
rs = rule["rs"]
weights = rule["ws"]
```

### Build a Vandermonde matrix

```python
from operators import vandermonde2d

V = vandermonde2d(4, rs[:, 0], rs[:, 1])
```

### Weighted modal reconstruction

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

## Demo Run Guide

Use module execution from project root:

### Foundations

```bash
python -m demo.foundations.smoke_phase1
python -m demo.foundations.smoke_table_rules
python -m demo.foundations.demo_connectivity_step1
python -m demo.foundations.demo_trace_policy_step2
python -m demo.foundations.demo_exchange_step3
```

### Visualization

```bash
python -m demo.visualization.demo_plot_nodes --table both
python -m demo.visualization.demo_reconstruct_field --table table1
python -m demo.visualization.demo_error_field --table table2
python -m demo.visualization.demo_3d --table table1
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
python -m demo.experiments.run_lsrk_h_convergence --preset quick --compare-qb-correction
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction-only
```

### LSRK qB Boundary Correction Modes

The LSRK experiment runner now supports boundary-trace correction workflows
for qB in the exchange path.

- Baseline (no correction):

```bash
python -m demo.experiments.run_lsrk_h_convergence --preset quick
```

- Side-by-side compare (baseline vs RK-stage qB correction):

```bash
python -m demo.experiments.run_lsrk_h_convergence --preset quick --compare-qb-correction
```

- Correction-only run (only RK-stage qB correction):

```bash
python -m demo.experiments.run_lsrk_h_convergence --preset quick --qb-correction-only
```

Implementation notes:

- Built-in notebook-style stage correction can be enabled through
    `use_sinx_rk_stage_boundary_correction=True`.
- Custom correction callbacks are supported through `q_boundary_correction`.
- Correction application mode is controlled by `q_boundary_correction_mode`
    with values: `inflow`, `boundary`, `all`.
- Compare mode writes separate baseline and corrected CSV outputs.

## Demo Migration Map

| Old script | New command |
|---|---|
| demo_plot_nodes_table1.py / demo_plot_nodes_table2.py | python -m demo.visualization.demo_plot_nodes --table table1 or --table table2 |
| demo_reconstruct_field_table1.py / demo_reconstruct_field_table2.py | python -m demo.visualization.demo_reconstruct_field --table table1 or --table table2 |
| demo_error_field_table1.py / demo_error_field_table2.py | python -m demo.visualization.demo_error_field --table table1 or --table table2 |
| demo_3d_table1.py / demo_3d_table2.py | python -m demo.visualization.demo_3d --table table1 or --table table2 |

No compatibility wrappers are kept for old root-level table-specific script names.

## Testing

Current routine validation commands:

```bash
pytest tests/unit
pytest tests/integration
pytest tests -k "not lsrk"
```

Notes:

- LSRK regression tests are preserved under tests/regression but are not part of this validation round.
- Empty placeholder test files remain in tests root by design and are not moved.

## Output Location

Generated figures are written under:

```text
photo/
```

Convergence CSV outputs are written under:

```text
experiments_outputs/
```

Both are resolved from project root.

## Current Scope

Already implemented:

- basis construction
- quadrature loading
- Vandermonde operators
- weighted reconstruction
- differentiation operators
- affine geometry utilities
- edge evaluation tools
- visualization support

Not yet fully developed:

- a complete DG time-marching solver
- full interface flux coupling on general multi-element meshes
- broad automated regression coverage across all modules
- a finalized public API for every helper routine

## Recommended Next Development Steps

1. Expand exactness and regression coverage for currently empty test stubs
2. Finalize stable public APIs across modules
3. Add numerical flux coupling across mesh interfaces
4. Build end-to-end DG examples on multi-element triangular meshes

## Development Principles

1. Correctness before optimization
2. Exactness tests before large demos
3. Geometry consistency before mapped-operator use
4. Clear module interfaces before solver-level expansion
