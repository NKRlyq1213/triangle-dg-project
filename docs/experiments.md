# Experiments / 實驗說明

This document tracks canonical experiment entry points, output locations, and
the current LSRK-related CLI contract.

本文件整理目前標準實驗入口、輸出位置，以及 LSRK 相關 CLI 契約
（參數、互動規則、限制條件、輸出行為）。

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

## LSRK CLI Decision Flow / LSRK CLI 決策流程

1. Choose your goal / 先選目標
   - `python -m cli.run_lsrk_h_convergence`: h-convergence table (CSV-first)
   - `python -m cli.plot_lsrk_error_vs_time`: time-history curves (PNG+CSV)
2. Choose trace geometry behavior / 再選 trace 幾何行為
   - `--physical-boundary-mode {exact_qb,opposite_boundary,periodic_vmap}`
   - `--interior-trace-mode {exchange,exact_trace}`
   - `--face-order-mode {triangle,simplex,simplex_strict}`
3. Choose lifting and penalties / 再選 lifting 與數值耗散
   - `--surface-inverse-mass-mode {diagonal,projected}`
   - `--tau`, `--tau-interior`, `--tau-qb`
4. Choose correction mode / 最後選 correction
   - h-convergence: `--qb-correction {off,on,compare}`
   - error-vs-time: `--qb-correction {off,on}`

If the selected combination violates a runtime constraint, the CLI raises a
`ValueError` with an explicit reason.

若參數組合違反目前限制，CLI 會直接丟出 `ValueError` 並附上原因。

## LSRK h-Convergence CLI / LSRK h-收斂 CLI

Command:

```bash
python -m cli.run_lsrk_h_convergence
```

### Parameters (visible) / 可見參數

- `--preset {quick,full,upstream-pbc}`
  Effective default: `quick`
- `--qb-correction {off,on,compare}`
  Effective default: `off`
- `--surface-inverse-mass-mode {diagonal,projected}`
  Default: `diagonal`
- `--test-function {sin2pi_x,sin2pi_y,sin2pi_xy}`
  Default: `sin2pi_x`
- `--physical-boundary-mode {exact_qb,opposite_boundary,periodic_vmap}`
  Default: `exact_qb`
- `--interior-trace-mode {exchange,exact_trace}`
  Default: `exchange`
- `--face-order-mode {triangle,simplex,simplex_strict}`
  Default: `triangle`
- `--tau FLOAT`
  Default: `0.0`
- `--tau-interior FLOAT`
  Default: unset; falls back to `--tau`
- `--tau-qb FLOAT`
  Default: unset; falls back to `--tau`
- `--diagonal {main,anti}`
  Default: unset; if set, overrides preset diagonal
- `--mesh-levels INT [INT ...]`
  Default: unset; if set, overrides preset mesh levels
- `--tf-values FLOAT [FLOAT ...]`
  Default: unset; if set, overrides preset final times
- `--use-numba` / `--no-use-numba`
  Default: enabled (`--use-numba`)

Hidden compatibility flags (legacy, not recommended for new runs):

- `--compare-qb-correction`
- `--qb-correction-only`

隱藏旗標僅保留相容用途，不建議新流程使用。

### Runtime Logic / 執行邏輯

- `preset` picks baseline mesh/tf/diagonal; current built-ins:
  - `quick`: `mesh_levels=(1,2,4,8,16)`, `tf_values=(3*pi,)`, `diagonal=anti`
  - `full`: `mesh_levels=(1,2,4,8,16,32)`, `tf_values=(3*pi,)`, `diagonal=anti`
  - `upstream-pbc`: `mesh_levels=(1,2,4,8,16,32,64)`, `tf_values=(3*pi,)`, `diagonal=anti`
- Override priority is explicit:
  - `--mesh-levels`, `--tf-values`, `--diagonal` replace preset values
  - If you do not pass them, preset values remain active
- Execution backend:
  - default uses numba fast path (`--use-numba`)
  - `--no-use-numba` forces numpy fallback
- Tau role split is fixed:
  - `tau_interior` is used on interior and non-`exact_qb` exterior traces
  - `tau_qb` is used only on `physical-boundary-mode=exact_qb` faces
- `qb-correction` behavior:
  - `off`: baseline only
  - `on`: corrected only (`rk-stage exact-source correction`)
  - `compare`: baseline and corrected are both run and exported

## LSRK Error-vs-Time CLI / LSRK 誤差-時間 CLI

Command:

```bash
python -m cli.plot_lsrk_error_vs_time
```

### Parameters (visible) / 可見參數

- `--mesh-level INT`
  Default: `8`
- `--mesh-levels INT [INT ...]`
  Optional multi-run overlay; takes priority when provided
- `--tf FLOAT`
  Default: `1.0`
- `--cfl FLOAT`
  Default: `1.0`
- `--diagonal {anti,main}`
  Default: `anti`
- `--h-definition {min-altitude,min-edge}`
  Default: `min-altitude`
- `--dt-cfl-mode {n2,nplus1-squared}`
  Default: `n2`
- `--test-function {sin2pi_x,sin2pi_y,sin2pi_xy}`
  Default: `sin2pi_x`
- `--physical-boundary-mode {exact_qb,opposite_boundary,periodic_vmap}`
  Default: `exact_qb`
- `--face-order-mode {triangle,simplex,simplex_strict}`
  Default: `triangle`
- `--interior-trace-mode {exchange,exact_trace}`
  Default: `exchange`
- `--qb-correction {off,on}`
  Default: `off`
- `--surface-inverse-mass-mode {diagonal,projected}`
  Default: `diagonal`
- `--tau FLOAT`
  Default: `0.0`
- `--tau-interior FLOAT`
  Default: unset; falls back to `--tau`
- `--tau-qb FLOAT`
  Default: unset; falls back to `--tau`
- `--output PATH`
  Default: auto-generated stem under `experiments_outputs/lsrk_error_vs_time/`

Hidden internal flags:

- `--q-boundary-correction-mode {inflow,boundary,all}` (default `all`)
- `--use-numba` / `--no-use-numba` (default enabled)

### Runtime Logic / 執行邏輯

- Mesh-level selection:
  - if `--mesh-levels` is provided, it is used
  - otherwise, CLI uses `--mesh-level`
  - duplicated levels are de-duplicated while preserving order
- CFL timestep control:
  - `--h-definition=min-altitude|min-edge` controls which element size is used for `h`
  - `--dt-cfl-mode=n2|nplus1-squared` controls denominator scaling (`N^2` vs `(N+1)^2`)
- Output path:
  - no `--output`: auto-generate `.png` and matching `.csv` in canonical output dir
  - relative `--output`: resolved from project root

## Compatibility Matrix / 相容性矩陣

The following are enforced by runtime checks:

| Condition | Status | Reason |
| --- | --- | --- |
| `interior-trace-mode=exact_trace` + `face-order-mode=simplex` | Invalid | `simplex` currently supports `exchange` only |
| `interior-trace-mode=exact_trace` + `face-order-mode=simplex_strict` | Invalid | `simplex_strict` currently supports `exchange` only |
| `face-order-mode=simplex_strict` + `surface-inverse-mass-mode=diagonal` | Invalid | `simplex_strict` requires `projected` |
| `interior-trace-mode=exact_trace` + `surface-inverse-mass-mode=projected` | Invalid | projected lifting not supported with exact interior trace |
| `qb-correction=on/compare` + no exact source | Invalid | requires `interior-trace-mode=exact_trace` or `physical-boundary-mode=exact_qb` |

## Scenario-Driven Examples / 情境導向範例

### 1) Quick sanity / 快速健檢

```bash
python -m cli.run_lsrk_h_convergence --preset quick
```

### 2) Full convergence / 完整收斂

```bash
python -m cli.run_lsrk_h_convergence --preset full
```

### 3) Upstream PBC alignment / 對齊 upstream PBC 風格

```bash
python -m cli.run_lsrk_h_convergence --preset upstream-pbc --physical-boundary-mode periodic_vmap --face-order-mode simplex
```

### 4) Enable periodic coordinate mapping / 開啟 periodic_vmap

```bash
python -m cli.run_lsrk_h_convergence --preset quick --physical-boundary-mode periodic_vmap
```

### 5) `simplex_strict + projected` / 嚴格 simplex 面序

```bash
python -m cli.run_lsrk_h_convergence --preset quick --face-order-mode simplex_strict --surface-inverse-mass-mode projected --interior-trace-mode exchange
```

### 6) Compare baseline vs correction / 比較 baseline 與修正

```bash
python -m cli.run_lsrk_h_convergence --preset quick --physical-boundary-mode exact_qb --qb-correction compare
```

### 7) Multi-mesh error-vs-time overlay / 多網格誤差隨時間曲線

```bash
python -m cli.plot_lsrk_error_vs_time --mesh-levels 8 16 32 --tf 1.0 --test-function sin2pi_xy --physical-boundary-mode periodic_vmap --face-order-mode simplex
```

### Expected invalid examples / 預期失敗範例

```bash
# invalid: simplex_strict requires projected
python -m cli.plot_lsrk_error_vs_time --face-order-mode simplex_strict --surface-inverse-mass-mode diagonal

# invalid: correction needs at least one exact source
python -m cli.plot_lsrk_error_vs_time --physical-boundary-mode opposite_boundary --interior-trace-mode exchange --qb-correction on
```

## Output Naming / 輸出命名

Canonical CLI outputs are grouped by producer under `experiments_outputs/`.
Ad hoc profiling, scratch, or preserved legacy artifacts should be written under
`experiments_outputs/scratch/`.

標準 CLI 輸出會依實驗類型寫入 `experiments_outputs/` 的子資料夾；
臨時 profiling、檢查圖、分析檔或保留的 legacy artifacts 則集中放在
`experiments_outputs/scratch/`。

### LSRK h-convergence CSV stems

Base stem:

`lsrk_h_convergence_{test_function}_tf{tf}_table1_order{order}_N{N}_{diagonal}_face{face_order_mode}_{surface_inverse_mass_mode}_{physical_boundary_mode}_taui{tau_interior}_tauqb{tau_qb}`

Suffix rules:

- `qb-correction=off`: `.csv`
- `qb-correction=on`: `_rkstage_qb_only.csv`
- `qb-correction=compare`: `_baseline.csv` and `_rkstage_qb.csv`
- `interior-trace-mode!=exchange`: append `_{interior_trace_mode}`
- `preset!=full`: append `_{preset}`

### LSRK error-vs-time outputs

- Stored under `experiments_outputs/lsrk_error_vs_time/`
- Auto mode writes a `.png` and matching `.csv` with same stem
- Auto stem includes:
  `tf`, mesh-level tags, test function, boundary mode,
  surface inverse-mass mode, diagonal, face order mode, `h` tag, `dt` CFL tag,
  `tau_interior`, `tau_qb`, and qB mode
- If `interior-trace-mode!=exchange`, stem appends `_{interior_trace_mode}`

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
