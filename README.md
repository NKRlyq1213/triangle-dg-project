# triangle_dg_project

Triangle-domain polynomial reconstruction and DG-oriented operators on the reference triangle.

## Overview

This repository provides the numerical building blocks needed for triangle-based polynomial approximation and DG-oriented operator construction. The current implementation focuses on basis functions, quadrature handling, reconstruction, differentiation, affine mappings, edge evaluation operators, and visualization tools on the reference triangle.

At its current stage, the project should be viewed as a **numerical foundation** for later discontinuous Galerkin development rather than a complete production DG solver.

## Features

- Triangle quadrature handling from Table 1 / Table 2 datasets
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
    reconstruction routines, boundary / edge operators, and split-form tools.

problems/
    Analytic test fields and initial-condition wrappers.

visualization/
    Plotting utilities for nodes, scalar fields, and radial diagnostics.

utils/
    Small helper checks.

demo/
    Example scripts for reconstruction and visualization.

tests/
    Test directory for numerical verification.
```

## Installation

Recommended Python version:

- Python 3.10 or later

Minimal dependencies:

- `numpy>=1.26`
- `scipy>=1.11`
- `matplotlib>=3.8`
- `pytest>=7.4`

Install with:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
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

## Demo Scripts

Typical demo runs:

```bash
python demo/demo_reconstruct_field_table1.py
python demo/demo_error_field_table2.py
python demo/demo_radial_sampling_table2.py
```

## Output Location

Demo figures are saved under:

```text
photo/
```

in the project root directory.

This keeps generated figures inside the repository and avoids machine-specific desktop paths.

## Current Scope

The repository currently provides the approximation and operator layer for triangle-based DG-style work.

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

1. Strengthen exactness and regression tests
2. Finalize stable public APIs across modules
3. Add numerical flux coupling across mesh interfaces
4. Build end-to-end DG examples on multi-element triangular meshes

## Project Positioning

This code base is best suited for:

- numerical experiments on triangle-based polynomial approximation
- DG pre-development and operator verification
- reconstruction and differentiation studies
- geometry/operator prototyping before full solver assembly

## Development Principles

When extending this project, keep the following priorities:

1. correctness before optimization
2. exactness tests before large demos
3. geometry consistency before mapped-operator use
4. clear module interfaces before solver-level expansion