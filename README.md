# triangle_dg_project

Triangle-domain polynomial reconstruction and DG-ready operators.

## Current goals

- Handle Table 1 / Table 2 nodes and weights
- Build Jacobi / simplex basis on triangle
- Construct Vandermonde and gradient Vandermonde matrices
- Support weighted polynomial reconstruction
- Build differentiation matrices
- Visualize node distributions and reconstructed fields inside a triangle

## Project structure

- `data/`: quadrature rules and edge rules
- `basis/`: Jacobi and simplex basis
- `geometry/`: triangle geometry and sampling
- `operators/`: Vandermonde, reconstruction, differentiation, boundary
- `problems/`: initial conditions and analytic test fields
- `visualization/`: plotting utilities
- `tests/`: unit tests

## Setup

```bash
pip install -r requirements.txt
```
## Notes
Table 2 uses separate GL1D edge rules for boundary extension.---

## `data/__init__.py`

```python
from .rule_registry import load_rule

__all__ = ["load_rule"]