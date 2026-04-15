# triangle_dg_project

Triangle-domain polynomial reconstruction and DG-oriented operators on the reference triangle.

三角形參考域上的多項式重建與 DG 導向運算元實作。

## Overview / 概覽

This repository provides reusable numerical components for triangle-based DG workflows.
It focuses on basis construction, quadrature handling, differentiation and reconstruction
operators, geometric mappings, exchange-based RHS assembly, and experiment runners.

Current state is a DG-ready foundation, not yet a complete production solver.

本專案提供可重用的三角形 DG 工作流程數值元件，重點在基底、積分規則、微分/重建運算元、
幾何映射、exchange-based RHS 組裝與實驗執行工具。

目前定位為 DG-ready foundation，尚未是完整的 production solver。

## Installation / 安裝

Recommended Python version:

- Python 3.10 or later

建議 Python 版本：

- Python 3.10 以上

Runtime dependencies:

- numpy>=1.26
- scipy>=1.11
- matplotlib>=3.8

執行期依賴如下：

- numpy>=1.26
- scipy>=1.11
- matplotlib>=3.8

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Editable install:

```bash
pip install -e .
```

Development extras:

```bash
pip install -e .[dev]
pip install -r requirements-dev.txt
```

Optional performance extra (Numba):

```bash
pip install -e .[perf]
pip install -e .[dev,perf]
```

## Repository Structure / 專案結構

```text
basis/              Basis construction and mode indexing
cli/                Canonical experiment entry points
data/               Table-1/Table-2 quadrature rules and edge rules
demo/               Foundations, visualization, and advanced demos
docs/               Project notes and experiment CLI docs
experiments/        Experiment backends and output-path helpers
geometry/           Reference triangle, affine maps, metrics, connectivity
operators/          Vandermonde, mass, differentiation, trace, RHS assembly
problems/           Analytic fields and initial-condition helpers
time_integration/   CFL utilities and LSRK54 integrator
tests/              unit, integration, and regression coverage
visualization/      Plotting helpers for scalar fields and diagnostics
```

上面列出目前核心模組分工；若你要擴充功能，建議優先在 `experiments/`、`operators/`、`tests/`
三個區塊保持同步演進。

## Quick Start / 快速開始

Load a quadrature rule and build a Vandermonde matrix:

```python
from data import load_table2_rule
from operators import vandermonde2d

rule = load_table2_rule(4)
rs = rule["rs"]
V = vandermonde2d(4, rs[:, 0], rs[:, 1])
```

載入積分規則並建立 Vandermonde 矩陣。

Weighted modal reconstruction:

```python
from data import load_table2_rule
from geometry import reference_triangle_area
from operators import fit_modal_coefficients_weighted, vandermonde2d
from problems import ground_truth_function

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

以上示範以加權最小平方法進行 modal reconstruction。

## Commands / 指令

Run commands from the project root.

請在專案根目錄執行以下指令。

### Foundations / 基礎範例

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

### Visualization / 視覺化

```bash
python -m demo.visualization.demo_plot_nodes --table both --orders 1 2 3 4
python -m demo.visualization.demo_reconstruct_field --table table1 --order 4 --N 4 --case smooth_bump
python -m demo.visualization.demo_error_field --table table2 --order 4 --N 4 --case smooth_bump
python -m demo.visualization.demo_3d --table table1 --order 4 --N 4 --case smooth_bump
python -m demo.visualization.demo_plot_mesh_nodes
python -m demo.visualization.demo_radial_sampling_table2
```

### Advanced / 進階

```bash
python -m demo.advanced.demo_differentiation_weighted
python -m demo.advanced.demo_split_gaussian_field
```

### Experiments (Canonical CLI) / 實驗（標準 CLI 入口）

```bash
python -m cli.run_field_h_convergence
python -m cli.run_div_h_convergence
python -m cli.run_rhs_exchange_benchmark
python -m cli.run_lsrk_h_convergence --preset quick
python -m cli.plot_lsrk_error_vs_time --mesh-levels 8 16 32 --tf 6.283185307179586 --test-function sin2pi_xy --qb-correction on
```

Legacy compatibility wrappers under `demo.experiments.*` are still available,
but `cli.*` is the canonical entry point going forward.

`demo.experiments.*` 仍可使用，但其角色是相容層；建議新流程以 `cli.*` 為主。

## LSRK CLI Highlights / LSRK CLI 重點

`cli.run_lsrk_h_convergence` visible parameters:

- `--preset {quick,full}` (default `quick`)
- `--qb-correction {off,on,compare}` (default `off`)
- `--surface-inverse-mass-mode {diagonal,projected}`
- `--test-function {sin2pi_x,sin2pi_y,sin2pi_xy}`
- `--physical-boundary-mode {exact_qb,opposite_boundary}`
- `--interior-trace-mode {exchange,exact_trace}`
- `--tau FLOAT` controls the surface numerical flux; `tau=0` is pure upwind and larger `tau` reduces the `|n·V|(qM-qP)` penalty

`cli.plot_lsrk_error_vs_time` visible parameters:

- `--mesh-level INT` or `--mesh-levels INT [INT ...]`
- `--tf FLOAT`, `--cfl FLOAT`
- `--test-function {sin2pi_x,sin2pi_y,sin2pi_xy}`
- `--physical-boundary-mode {exact_qb,opposite_boundary}`
- `--interior-trace-mode {exchange,exact_trace}`
- `--qb-correction {off,on}`
- `--surface-inverse-mass-mode {diagonal,projected}`
- `--tau FLOAT` controls the surface numerical flux; `tau=0` is pure upwind and larger `tau` reduces the `|n·V|(qM-qP)` penalty
- `--output PATH`

Current constraints:

- `interior-trace-mode=exact_trace` supports only `physical-boundary-mode=exact_qb`
- `interior-trace-mode=exact_trace` does not support `surface-inverse-mass-mode=projected`
- `physical-boundary-mode=opposite_boundary` disables qB correction

LSRK 指令常用參數與限制如上，完整細節可參考 [docs/experiments.md](docs/experiments.md)。

## Outputs / 輸出

Experiment outputs are managed under `experiments_outputs/`.
Canonical experiment runs write into producer-specific subdirectories, while
temporary analysis, profiling, or preserved legacy artifacts should go under
`experiments_outputs/scratch/`.

實驗輸出統一放在 `experiments_outputs/`，正式結果依實驗類型分組；
暫時分析圖、profiling、smoke artifacts 與保留的 legacy 檔案則放在
`experiments_outputs/scratch/`。

Canonical grouped directories:

- `experiments_outputs/field_h_convergence/`
- `experiments_outputs/div_h_convergence/`
- `experiments_outputs/rhs_exchange_benchmark/`
- `experiments_outputs/lsrk_h_convergence/`
- `experiments_outputs/lsrk_error_vs_time/`
- `experiments_outputs/scratch/`

Demo figures are written under `photo/`.
Both top-level figure files and grouped folders can appear, for example:

- `photo/table1_nodes_order_4.png`
- `photo/demo_connectivity_step1_2x2.png`
- `photo/differentiation_photo/`

詳細實驗 CLI 參數、命名規則與限制條件請見 [docs/experiments.md](docs/experiments.md)。

## Testing / 測試

Typical commands:

```bash
pytest tests
pytest tests/unit
pytest tests/integration
pytest tests/integration/lsrk
pytest tests/integration/rhs
pytest tests/regression
```

常見測試流程會先跑 integration 子集合，再做完整 `pytest tests` 回歸。

## Current Scope / 目前範圍

Implemented:

- basis and quadrature utilities
- Vandermonde, mass, differentiation, reconstruction operators
- affine geometry and connectivity helpers
- exchange-based split-form RHS building blocks
- LSRK experiment workflows and visualization demos

目前已實作：

- 基底與積分規則工具
- Vandermonde、mass、微分與重建運算元
- 仿射幾何與連通性工具
- exchange-based split-form RHS 元件
- LSRK 實驗流程與視覺化範例

Still evolving:

- full multi-physics production solver pipeline
- broader long-horizon regression coverage for all workflows
- finalized and stable public API boundaries across helper modules

仍在演進：

- 多物理 production solver 完整流程
- 更完整的長時程回歸覆蓋
- helper 模組公開 API 邊界的最終穩定化
