# Before 與 After 公式逐符號說明

對應主報告: [diagnostics/results/Final_Acceptance_Report.md](diagnostics/results/Final_Acceptance_Report.md)

本文專門解釋主報告第 1.2 與 1.3 的數學式中，每一個符號在程式中的產生方式與數學邏輯。

## 0. 版本邊界先說清楚

本文中的 Before 指的是:

- 已完成 face_order 對齊（拓樸配對問題已修）
- 但尚未啟用 simplex_strict 的 strict lift normalization（lift_scale）

也就是「post face-order fix, pre simplex_strict lift-scale fix」階段。

After 指的是 simplex_strict 啟用後的最終路徑。

## 1. 兩個核心公式（重述）

Before（主報告 1.2）:

$$
\mathcal{I}^{tri}_K = E^T\bigl[L_f\,\hat w_i\,p_{f,i}\bigr],
$$

$$
\mathrm{RHS}^{tri}_{surf}
= \frac{1}{A_K}\,\mathcal{I}^{tri}_K\,M^{-1}_{proj}.
$$

After（主報告 1.3，simplex 參考等價）:

$$
\mathcal{I}^{sx}_K = E^T\bigl[J_{s,f}\,w_i^{[-1,1]}\,p_{f,i}\bigr],
$$

$$
\mathrm{RHS}^{sx}_{surf}
= \frac{1}{J_K}\,M^{-1}_{proj}\,\mathcal{I}^{sx}_K.
$$

simplex_strict 在 triangle 端落地為:

$$
\mathrm{RHS}^{strict}_{surf}
= \underbrace{\mathrm{lift\_scale}}_{=A_{ref}=2}
\cdot
\frac{1}{A_K}\,\mathcal{I}^{tri}_K\,M^{-1}_{proj}.
$$

## 2. 符號總表（每一個符號如何產生）

| 符號 | 程式來源 | 產生方式 | 數學角色 |
|---|---|---|---|
| $K$ | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1061) | cache 中元素數 | 元素索引範圍 |
| $f$ | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1102) | 迴圈 face 0,1,2 | 本地面索引 |
| $i$ | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1105) | face 上節點索引 | 面積分節點 |
| $q^-_{f,i}$（qM） | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1491) | pair_face_traces 產生內側 trace | 面內側狀態 |
| $q^+_{f,i}$（qP） | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1507) | fill_exterior_state 合併鄰居/邊界 | 面外側狀態 |
| $\mathbf{n}_{f,i}$ | [triangle-dg-project/geometry/face_metrics.py](triangle-dg-project/geometry/face_metrics.py#L146) | face 法向複製到 Nfp 節點 | 通量方向 |
| $\mathbf{v}_{f,i}$ | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1264) | velocity(x_face,y_face,t) | 對流速度 |
| $\mathbf{n}\cdot\mathbf{v}$（ndotV） | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1267) | nx*u + ny*v | 入/出流判定 |
| $p_{f,i}$ | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L470) | numerical_flux_penalty，tau=0 時退化為 min(ndotV,0)(qM-qP) | Upwind 懲罰 |
| $L_f$ | [triangle-dg-project/geometry/face_metrics.py](triangle-dg-project/geometry/face_metrics.py#L145) | 每面邊長 | 面積分幾何尺度 |
| $A_K$ | [triangle-dg-project/geometry/face_metrics.py](triangle-dg-project/geometry/face_metrics.py#L132) | 元素面積 | 體積尺度 |
| $\hat w_i$ | [triangle-dg-project/operators/trace_policy.py](triangle-dg-project/operators/trace_policy.py#L153) | Table1 的 face_weights（由 rule[we] 取出並排序） | triangle 端面積分權重 |
| $E$ | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L930) | E_face_matrix（face 節點散佈到 volume 節點） | 面到體投影 |
| $\mathcal{I}^{tri}_K$ | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1101) | 累積 face_contrib 到 surface_integral | 面積分向量 |
| $M^{-1}_{proj}$ | [triangle-dg-project/experiments/lsrk_h_convergence.py](triangle-dg-project/experiments/lsrk_h_convergence.py#L374) | 由 projector 與 1/ws 建立 projected inverse mass | 體積 lifting 算子 |
| $J_{s,f}$ | [triangle-dg-project/tests/integration/rhs/test_simplex_strict_dynamic_parity.py](triangle-dg-project/tests/integration/rhs/test_simplex_strict_dynamic_parity.py#L521) | simplex 參考中用 j_face_expanded（=L_f/2） | simplex 面 Jacobian |
| $w_i^{[-1,1]}$ | [triangle-dg-project/tests/integration/rhs/test_simplex_strict_dynamic_parity.py](triangle-dg-project/tests/integration/rhs/test_simplex_strict_dynamic_parity.py#L520) | simplex 參考的 weights_1d（定義域 [-1,1]） | simplex 面權重 |
| $J_K$ | [triangle-dg-project/tests/integration/rhs/test_simplex_strict_dynamic_parity.py](triangle-dg-project/tests/integration/rhs/test_simplex_strict_dynamic_parity.py#L524) | simplex 幾何雅可比 | simplex 體積尺度 |
| $A_{ref}$ | [triangle-dg-project/geometry/reference_triangle.py](triangle-dg-project/geometry/reference_triangle.py#L30) | reference_triangle_area() 回傳 2 | 參考三角形尺度常數 |
| lift_scale | [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L81) | simplex_strict 時設為 reference_triangle_area() | strict normalization 因子 |

## 3. Before 公式逐步展開（程式對照）

Step 1. 取得 trace 狀態

- qM 來自 interior pairing。
- qP 由 interior exchange + boundary exact fill 組成。

對照: [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1491)

Step 2. 計算通量懲罰

$$
p_{f,i}=\min(\mathbf{n}\cdot\mathbf{v},0)(q^-_{f,i}-q^+_{f,i}).
$$

對照: [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L470)

Step 3. 每一面加權

$$
\text{face\_contrib}_{f,i}=L_f\,\hat w_i\,p_{f,i}.
$$

對照: [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1105)

Step 4. 用 $E^T$ 散佈到體節點（形成 $\mathcal{I}^{tri}_K$）

對照: [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1101)

Step 5. projected lifting + 幾何除法

$$
\mathrm{RHS}^{tri}_{surf}=\frac{1}{A_K}\,\mathcal{I}^{tri}_K\,M^{-1}_{proj}.
$$

對照: [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1114)

## 4. After 公式逐步展開（simplex_strict 對齊邏輯）

Step 1. strict 模式啟動時先求 lift_scale

- simplex_strict 才會啟用。
- 值為 $A_{ref}=2$。

對照: [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L81)

Step 2. strict 路徑改用矩陣式組裝

$$
\text{scaled\_penalty}=p\cdot \hat w \cdot L,
\quad
\mathcal{I}=E^T\text{scaled\_penalty}.
$$

對照: [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1086)

Step 3. 仍先做 triangle 幾何除法，再乘 lift_scale

$$
\mathrm{RHS}^{strict}_{surf}=\left(\frac{1}{A_K}\,\mathcal{I}\,M^{-1}_{proj}\right)\cdot A_{ref}.
$$

對照: [triangle-dg-project/operators/rhs_split_conservative_exchange.py](triangle-dg-project/operators/rhs_split_conservative_exchange.py#L1091)

Step 4. 與 simplex 參考公式的等價關係

simplex 參考 evaluator:

$$
\mathcal{I}^{sx}=E^T\left[J_{s,f}\,w_i^{[-1,1]}\,p\right],
\quad
\mathrm{RHS}^{sx}=\frac{1}{J_K}M^{-1}_{proj}\mathcal{I}^{sx}.
$$

對照: [triangle-dg-project/tests/integration/rhs/test_simplex_strict_dynamic_parity.py](triangle-dg-project/tests/integration/rhs/test_simplex_strict_dynamic_parity.py#L519)

利用

$$
J_{s,f}=\frac{L_f}{2},\quad w_i^{[-1,1]}\approx 2\hat w_i,\quad J_K=\frac{A_K}{2}
$$

可得

$$
\frac{1}{J_K}=\frac{2}{A_K},
\quad
J_{s,f}w_i^{[-1,1]}\approx L_f\hat w_i
\Rightarrow
\mathrm{RHS}^{sx}\approx 2\,\mathrm{RHS}^{tri}.
$$

這就是為什麼 strict 需要 lift_scale = 2 才能在 generic state 下保持同構。

## 5. 符號背後的數學邏輯（一句話版）

- $p$: 決定入流方向下應該被懲罰的 jump。
- $L_f\hat w_i$ 或 $J_{s,f}w_i^{[-1,1]}$: 面積分離散化權重（面幾何乘積分權重）。
- $E^T$: 把面上的懲罰注入到體節點自由度。
- $M^{-1}_{proj}$: 在 polynomial 子空間上做穩定 lifting（不是直接點值倒質量）。
- $1/A_K$ 或 $1/J_K$: 把 surface integral 轉回體積尺度。
- lift_scale: 用來消除 two-convention（$A_K$ vs $J_K$）差異的常數橋樑。

## 6. 與主報告搭配閱讀建議

- 先看主報告結論: [diagnostics/results/Final_Acceptance_Report.md](diagnostics/results/Final_Acceptance_Report.md)
- 再看本文件第 2 節符號總表。
- 最後看第 3 與第 4 節，逐步把程式語句映射到公式。

以上即是 1.2/1.3 兩個公式中每個符號的完整來源與邏輯。