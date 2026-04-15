# Experiments / 實驗說明

This document tracks canonical experiment entry points, output locations, and
the current LSRK-related CLI surface.

本文件整理目前標準實驗入口、輸出位置，以及 LSRK 相關 CLI 介面。

## Canonical Entry Points / 標準入口

Use the `cli` package for experiment runs.

請使用 `cli` 套件作為實驗執行入口。

- `python -m cli.run_field_h_convergence`
  Output: `experiments_outputs/field_h_convergence/`
- `python -m cli.run_div_h_convergence`
  Output: `experiments_outputs/div_h_convergence/`
- `python -m cli.run_rhs_exchange_benchmark`
  Output: `experiments_outputs/rhs_exchange_benchmark/`
- `python -m cli.run_lsrk_h_convergence`
  Output: `experiments_outputs/lsrk_h_convergence/`
- `python -m cli.plot_lsrk_error_vs_time`
  Output: `experiments_outputs/lsrk_error_vs_time/`

Legacy wrappers under `demo.experiments.*` still forward to the same
implementations, but they are compatibility shims.

`demo.experiments.*` 仍可使用，但僅作為相容層；正式流程建議使用 `cli.*`。

## LSRK h-Convergence CLI / LSRK h-收斂 CLI

Command:

```bash
python -m cli.run_lsrk_h_convergence
```

Visible parameters:

- `--preset {quick,full}`
  Default: `quick`
- `--qb-correction {off,on,compare}`
  Default: `off`
- `--surface-inverse-mass-mode {diagonal,projected}`
  Default: `diagonal`
- `--test-function {sin2pi_x,sin2pi_y,sin2pi_xy}`
  Default: `sin2pi_x`
- `--physical-boundary-mode {exact_qb,opposite_boundary}`
  Default: `exact_qb`
- `--interior-trace-mode {exchange,exact_trace}`
  Default: `exchange`
- `--tau FLOAT`
  Default: `0.0`

可見參數如上；若未指定，會採用 Default 值。

Hidden compatibility parameters:

- `--compare-qb-correction`
- `--qb-correction-only`

Examples:

```bash
python -m cli.run_lsrk_h_convergence --preset quick
python -m cli.run_lsrk_h_convergence --preset quick --qb-correction compare
python -m cli.run_lsrk_h_convergence --preset quick --qb-correction on
python -m cli.run_lsrk_h_convergence --preset quick --test-function sin2pi_xy
python -m cli.run_lsrk_h_convergence --preset quick --interior-trace-mode exact_trace
python -m cli.run_lsrk_h_convergence --preset quick --tau 0.4
```

Behavior notes:

- `quick` uses `mesh_levels=(1,2,4,8,16)` and `tf=2*pi`
- `full` uses `mesh_levels=(1,2,4,8,16,32)` and `tf=1`
- `compare` writes separate baseline and corrected CSVs
- `on` writes only the corrected run
- `off` writes only the baseline run
- `tau` is the surface numerical-flux parameter; `tau=0` is pure upwind, and larger `tau` reduces the `|n·V|(qM-qP)` penalty

## LSRK Error-vs-Time CLI / LSRK 誤差-時間 CLI

Command:

```bash
python -m cli.plot_lsrk_error_vs_time
```

Visible parameters:

- `--mesh-level INT`
  Default: `8`
- `--mesh-levels INT [INT ...]`
- `--tf FLOAT`
  Default: `1.0`
- `--cfl FLOAT`
  Default: `1.0`
- `--test-function {sin2pi_x,sin2pi_y,sin2pi_xy}`
  Default: `sin2pi_x`
- `--physical-boundary-mode {exact_qb,opposite_boundary}`
  Default: `exact_qb`
- `--interior-trace-mode {exchange,exact_trace}`
  Default: `exchange`
- `--qb-correction {off,on}`
  Default: `off`
- `--surface-inverse-mass-mode {diagonal,projected}`
  Default: `diagonal`
- `--tau FLOAT`
  Default: `0.0`
- `--output PATH`
  Default: auto-generated path under `experiments_outputs/lsrk_error_vs_time/`

Hidden internal parameters:

- `--q-boundary-correction-mode {inflow,boundary,all}`
  Default: `all`
- `--use-numba` / `--no-use-numba`
  Default: enabled

Examples:

```bash
python -m cli.plot_lsrk_error_vs_time --mesh-level 8 --tf 1.0
python -m cli.plot_lsrk_error_vs_time --mesh-levels 8 16 32 --tf 6.283185307179586 --test-function sin2pi_xy --qb-correction on
python -m cli.plot_lsrk_error_vs_time --mesh-level 16 --interior-trace-mode exact_trace --physical-boundary-mode exact_qb
python -m cli.plot_lsrk_error_vs_time --mesh-level 16 --tau 0.4
```

`tau` controls the surface numerical flux via
`f* = 1/2[(n·V)qM + (n·V)qP] + (1-tau)/2 |n·V| (qM - qP)`;
`tau=0` is pure upwind, and larger `tau` reduces the upwind penalty.

## Current Constraints / 目前限制

- `interior-trace-mode=exact_trace` currently supports only `physical-boundary-mode=exact_qb`
- `interior-trace-mode=exact_trace` does not support `surface-inverse-mass-mode=projected`
- `physical-boundary-mode=opposite_boundary` disables qB correction even if `--qb-correction on` is requested

以上限制為目前實作行為，若違反會在執行時拋出錯誤或忽略校正。

## Output Naming / 輸出命名

Canonical CLI outputs are grouped by producer under `experiments_outputs/`.
Ad hoc profiling, scratch, or preserved legacy artifacts should be written under
`experiments_outputs/scratch/`.

標準 CLI 輸出會依實驗類型寫入 `experiments_outputs/` 的子資料夾；
臨時 profiling、檢查圖、分析檔或保留的 legacy artifacts 則集中放在
`experiments_outputs/scratch/`。

LSRK h-convergence CSVs:

- Base stem: `lsrk_h_convergence_{test_function}_tf{tf}_table1_order{order}_N{N}_{diagonal}_tau{tau}`
- Compare mode suffixes: `_baseline.csv`, `_rkstage_qb.csv`
- Correction-only suffix: `_rkstage_qb_only.csv`
- Quick runs append `_quick`
- Non-exchange trace runs append `_{interior_trace_mode}`

LSRK error-vs-time outputs:

- Stored under `experiments_outputs/lsrk_error_vs_time/`
- A `.png` plot and matching `.csv` are written with the same stem
- The auto-generated stem includes `tf`, mesh levels, test function, boundary mode,
  surface inverse mass mode, `tau`, qB mode, and optional trace mode

Current canonical layout:

- `experiments_outputs/field_h_convergence/`
- `experiments_outputs/div_h_convergence/`
- `experiments_outputs/rhs_exchange_benchmark/`
- `experiments_outputs/lsrk_h_convergence/`
- `experiments_outputs/lsrk_error_vs_time/`
- `experiments_outputs/scratch/`

## Demo Output Layout / Demo 輸出布局

Demo figures are written under `photo/`.
Both top-level figure files and grouped folders can appear.

Demo 圖像輸出位於 `photo/`，目前可同時看到根目錄檔案與子資料夾。

Examples:

- `photo/table1_nodes_order_4.png`
- `photo/demo_connectivity_step1_2x2.png`
- `photo/demo_exchange_step3.png`
- `photo/differentiation_photo/`
