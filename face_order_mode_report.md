# `face-order-mode {triangle, simplex, simplex_strict}` 技術判定報告

## 1. 問題背景

本報告整理本次對話中，針對 GitHub repo `NKRlyq1213/triangle-dg-project` 與相關資料來源所做的技術判定，目的在於回答以下問題：

1. `triangle`、`simplex`、`simplex_strict` 三種 `face-order-mode` 的數值／幾何意義是什麼。
2. 為何 `simplex_strict` 需要額外乘上 2 倍縮放。
3. 從目前 repo、主要資料來源 `SDG4PDEOnSphere20260416.pdf`，以及其他 DG / simplex 文獻交叉驗證後，哪一個模式才是正確的。

本報告特別區分兩件事：

- **face ordering / face labeling**：三條邊如何編號與排列。
- **surface lifting normalization**：表面項 lift 回體積未知量時，幾何尺度如何進入。

這兩者在本問題中不能混為一談。

---

## 2. 使用的資料來源

### 2.1 專案 repo

- `experiments/lsrk_h_convergence.py`：定義 `face_order_mode` 的可選值與模式限制。fileciteturn6file0
- `operators/rhs_split_conservative_exchange.py`：三種模式的核心實作，包括 face permutation 與 `simplex_strict` 的 2 倍 lifting scale。fileciteturn9file0
- `operators/trace_policy.py`：repo 原生 triangle face convention 與 embedded face-node 排列方式。fileciteturn16file0
- `geometry/reference_triangle.py`：reference triangle 的頂點與面積，面積為 `2.0`。fileciteturn17file0
- `operators/mass.py`：mass matrix 的 quadrature form。fileciteturn19file0

### 2.2 主要資料來源

- `SDG4PDEOnSphere20260416.pdf`：本次判定的主資料來源。文件中給出 simplex quadrature 與 DG lifting 的定義。fileciteturn22file0 fileciteturn22file1

### 2.3 其他文獻

- *Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications*：標準 nodal DG on triangles 的 face mask / edge mass / lifting 架構。fileciteturn25file0 fileciteturn25file5
- Chen–Shu review：simplex SBP / entropy-stable DG 的 reference simplex 與 boundary operator 寫法。fileciteturn25file9 fileciteturn25file14
- Zhang–Cui–Liu quadrature paper：simplex quadrature 的標準尺度寫法。fileciteturn25file6

---

## 3. repo 目前對三種模式的明確定義

### 3.1 `triangle`

`triangle` 是 repo 的預設模式。`LSRKHConvergenceConfig` 中，`face_order_mode` 預設為 `"triangle"`，且 mode 合法值為 `triangle`, `simplex`, `simplex_strict`。fileciteturn6file0

repo 原生的 local face convention 在 `trace_policy.py` 中明確寫成：

- face 1: `v2 -> v3`
- face 2: `v3 -> v1`
- face 3: `v1 -> v2` fileciteturn16file0

而 reference triangle 頂點為：

- `v1 = (-1,-1)`
- `v2 = ( 1,-1)`
- `v3 = (-1, 1)`，且其面積為 `2.0`。fileciteturn17file0

因此，`triangle` 的意義是：

> 使用 repo 原生的 triangle local face numbering 與 face-node 排列。

---

### 3.2 `simplex`

在 `rhs_split_conservative_exchange.py` 中，repo 直接寫出 triangle 與 simplex 兩套 face ordering 的對應：

- triangle order `B = [v2v3, v3v1, v1v2]`
- simplex order `A = [v1v2, v2v3, v3v1]`
- 對應 permutation `face_perm_new_to_old = [2,0,1]`。fileciteturn9file0

而且這個 permutation 不是只改 trace label，而是連動重排：

- `length`
- `EToE`
- `EToF`
- `is_boundary`
- `face_flip`
- `x_face`, `y_face`
- `nx`, `ny` fileciteturn9file0

因此，`simplex` 的意義是：

> 把 repo 原生 triangle face convention 重新標成 simplex 文獻較直觀的 face ordering，但維持整套幾何與連通資料一致。

這表示 `simplex` **沒有改 PDE，也沒有改 flux family**；它只是 face-label convention 的一致重編號。

---

### 3.3 `simplex_strict`

`simplex_strict` 與 `simplex` 使用相同的 simplex-like face permutation，但多了一個額外條件：

- 若 mode 是 `simplex_strict`，則 `_resolve_surface_lift_scale(...)` 回傳 `reference_triangle_area()`。fileciteturn9file0
- `reference_triangle_area()` 的值為 `2.0`。fileciteturn17file0

也就是：

> `simplex_strict = simplex ordering + 額外乘上 2 倍 lifting scale`

此外，repo 還強制限制：

- `simplex` / `simplex_strict` 只支援 `interior_trace_mode='exchange'`
- `simplex_strict` 必須搭配 `surface_inverse_mass_mode='projected'`。fileciteturn6file0

因此 `simplex_strict` 並不是單純「更嚴格的排序」，而是：

> 一種綁定 projected inverse-mass lifting 的額外 normalization 模式。

---

## 4. 主要資料來源 `SDG4PDEOnSphere20260416.pdf` 的要求

### 4.1 quadrature 的尺度寫法

文件把 simplex 體積與邊界積分寫成：

\[
\int_T f(\xi)\,d\xi = |T| \sum_i f(\xi_i) w_i^s,
\qquad
\int_{\partial T^\gamma} f(\xi)\,d\xi = |\partial T^\gamma| \sum_i f(\xi_i^\gamma) w_i^e.
\]

這表示：

- `w_i^s`, `w_i^e` 是 **normalized quadrature weights**
- 真正幾何尺度由 `|T|` 與 `|∂T^γ|` 額外乘上。fileciteturn22file0

---

### 4.2 DG surface lifting 的尺度寫法

文件中的半離散 DG formula 把 surface 項寫成：

\[
(|T|)^{-1} W^{-1}E^T W^e p.
\]

也就是說，**surface lifting 前面應帶有明確的 `1/|T|`**。fileciteturn22file1

這一點對本題最重要，因為它直接影響 `simplex_strict` 多乘一個 `|T_ref|=2` 是否合理。

---

### 4.3 boundary DOF 的排列要求

文件將未知向量寫成：

\[
q = [q_1,\dots,q_{3(n+1)}, q_{3(n+1)+1},\dots,q_N]^T,
\]

前面 `3(n+1)` 個就是 boundary points；相應的選取矩陣為

\[
E=[I_{3(n+1)\times 3(n+1)} \mid 0].
\]

這表示：

> boundary DOFs 必須先被整理成三條邊的 block 結構。fileciteturn22file1

文件在這裡要求的是 **邊分塊(blocked by faces)**，但並沒有給出足夠資訊說「只能使用 triangle ordering，不可用 simplex ordering」。

因此，從主資料來源可推出：

- **face ordering 需要一致，但不是唯一固定的物理選擇**
- 真正不可違反的是 **surface lifting 的尺度**

---

## 5. 其他文獻的交叉驗證

### 5.1 Hesthaven–Warburton：face 編號是 convention，不是 physics

在標準 nodal DG triangle 寫法中：

- face nodes 由 `Fmask1`, `Fmask2`, `Fmask3` 組成
- `Fmask = [fmask1; fmask2; fmask3]'`
- 每一欄就是一條 face 的 local nodes。fileciteturn25file5

同時 surface lifting 採用：

\[
\mathrm{LIFT} = M^{-1}E,
\]

而 edge mass matrix 已包含沿 edge 的 Jacobian factor：

\[
M_1 = J_1 (V_{1D}V_{1D}^T)^{-1}.
\] fileciteturn25file0

這代表：

- 文獻允許自定 local face numbering，只要整套一致即可
- 幾何尺度已經透過 edge mass / mapping 進入，不需要最後再乘一個 reference-area 因子

因此，這篇文獻支持：

- `triangle` 與 `simplex` 可以是等價正確模式
- `simplex_strict` 的額外 `×2` 沒有標準 DG 根據

---

### 5.2 Chen–Shu：simplex SBP 的尺度由 reference operators 與 boundary matrix 決定

在 simplex SBP / entropy stable DG review 中，作者明確指出：

- local matrices 來自 reference simplex，再藉 affine map 轉到實體元素。fileciteturn25file14
- 對 collocated surface nodes，boundary matrix 可寫成

\[
E_\kappa^\gamma = (R_\kappa^\gamma)^T B^\gamma R_\kappa^\gamma.
\] fileciteturn25file14

也就是：

- 體積與邊界的尺度由 quadrature / boundary matrix / affine mapping 給出
- 沒有任何地方說「因為用了 simplex ordering，所以 surface lift 最後還要乘 reference triangle area」

因此，這篇文獻也不支持 `simplex_strict` 的 2 倍補償。

---

### 5.3 Zhang–Cui–Liu：simplex quadrature 的標準形式本來就只乘一次 `|T|`

該文獻對 simplex quadrature 寫成：

\[
\int_T f(x)\,dx \approx |T|\sum_{i=1}^n f(p_i)w_i.
\]

即：

- weights 是 normalized
- 幾何尺度只出現一次 `|T|`。fileciteturn25file6

這與主資料來源 `SDG4PDEOnSphere20260416.pdf` 完全一致。fileciteturn22file0

因此，從 quadrature 角度看，也沒有文獻支持在 lifting 最後再額外乘上一次 `|T_ref|=2`。

---

## 6. 為什麼 `simplex_strict` 是錯的

### 6.1 repo 自己的 projected lifting 已經含有 area normalization

repo 中 projected inverse-mass 的建法為：

\[
M = |T| V^T W V,
\qquad
P = V M^{-1} (|T| V^T).
\]

之後在 `_lift_surface_penalty_to_volume(...)` 裡，surface contribution 會再做 `/ area`。fileciteturn6file0 fileciteturn19file0 fileciteturn9file0

這與主資料來源的

\[
(|T|)^{-1}W^{-1}E^TW^e p
\]

方向一致：**最後應該保留一個 `1/|T|` 因子**。fileciteturn22file1

---

### 6.2 `simplex_strict` 會把這個 `1/|T|` 抵消掉

但 `simplex_strict` 又額外把 `lift_scale` 設成 `reference_triangle_area() = 2.0`。fileciteturn9file0 fileciteturn17file0

因此對目前 reference triangle 而言：

- 原本 projected lifting 裡已有 `/ area`
- `simplex_strict` 再乘一個 `area`
- 結果就是把理論上應有的 `1/|T|` 抵消掉

這與主資料來源和其他文獻的寫法都不一致。

因此可以判定：

> `simplex_strict` 的額外 `×2` 是多餘的 normalization，沒有資料來源支持。

---

## 7. `triangle` 與 `simplex` 哪一個更「對」

這裡需要分成兩個層次。

### 7.1 若問「理論上誰正確」

答案是：

> **`triangle` 與 `simplex` 都正確。**

理由：

- face numbering 在標準 DG 與 simplex SBP 文獻中本來就是 local convention
- 只要 face permutation 連同 `length`, `EToF`, `face_flip`, `normals`, `boundary flags` 等資料一起一致重排，就不會改變方法本身。fileciteturn25file5 fileciteturn9file0

---

### 7.2 若問「哪一個最貼近 simplex 文獻的命名與直覺」

答案是：

> **`simplex`**

理由：

- `simplex` 把 face ordering 重排成 `[v1v2, v2v3, v3v1]`，這和 simplex 文獻的命名習慣更接近。fileciteturn9file0
- `triangle` 只是 repo 原生 Hesthaven-style face numbering 的等價版本。fileciteturn16file0

因此：

- **最自然的「文獻對應」選擇：`simplex`**
- **數值上等價正確：`triangle` 與 `simplex`**

---

## 8. 兩個互斥假設的判定

### 假設 A：`triangle`、`simplex` 錯，`simplex_strict` 對

這個假設若要成立，必須有文獻支持：

- projected surface lifting 在標準 simplex DG/SBP 寫法下，最後還需要再乘上一個 reference triangle area。

目前使用的文獻中：

- 主資料來源 `SDG4PDEOnSphere20260416.pdf` 不支持。fileciteturn22file1
- Hesthaven–Warburton 不支持。fileciteturn25file0
- Chen–Shu review 不支持。fileciteturn25file14
- Zhang–Cui–Liu quadrature 也不支持。fileciteturn25file6

因此：

> **假設 A 不成立。**

---

### 假設 B：`simplex_strict` 錯，`triangle`、`simplex` 對

這個假設被所有資料來源一致支持：

1. face ordering 是 convention，不是 physics。fileciteturn25file5
2. lifting 的幾何尺度由 mass/boundary/quadrature/affine mapping 決定，而不是最後再乘 `|T_ref|`。fileciteturn25file0 fileciteturn25file14
3. simplex quadrature 的 area factor 只出現一次。fileciteturn25file6 fileciteturn22file0
4. `simplex_strict` 額外乘 `2` 會破壞主資料來源中的 `1/|T|` 尺度。fileciteturn22file1 fileciteturn9file0

因此：

> **假設 B 成立。**

---

## 9. 最終結論

### 9.1 最終技術判定

- **`triangle`：正確**
- **`simplex`：正確**
- **`simplex_strict`：錯誤**

### 9.2 若只選一個最貼近資料來源／文獻 simplex 記號的模式

- **建議選 `simplex`**

### 9.3 若從目前 repo 的工程角度給建議

- 把 `simplex_strict` 視為應淘汰或至少應清楚註解的兼容模式
- 主要保留 `triangle` 與 `simplex`
- 若後續要對外說明，建議以 `simplex` 當文獻對應名稱，以 `triangle` 視為等價的 internal ordering 版本

---

## 10. 已知、未知、風險

### 已知

- repo 對三種模式的實作差異已可從程式碼直接讀出。fileciteturn6file0 fileciteturn9file0
- 主資料來源與其他文獻一致支持：surface lifting 不應額外乘 reference triangle area。fileciteturn22file1 fileciteturn25file0 fileciteturn25file14

### 未知

- 主資料來源沒有把三條邊的編號精確定成唯一不可替代的 `face 1/2/3` 順序，因此不能僅憑文件把 `triangle` 判成理論錯。

### 風險

- 若把 `simplex_strict` 的「乘 2 後某些案例看起來更穩定」誤解為理論正確，會把 normalization 問題錯當成 face ordering 問題。
- `simplex_strict` 若在某些測試中表現較好，更可能表示其他地方仍有 boundary / projected lifting / exact-state correction 的尺度不一致，而不是 `simplex_strict` 本身正確。

---

## 11. 建議的後續驗證工作

1. 針對同一組測試案例，直接比較 `triangle` 與 `simplex` 的：
   - `surface_rhs`
   - `total_rhs`
   - interior face mismatch
   驗證兩者是否只差 permutation。

2. 將 `simplex_strict` 從主要收斂／誤差研究中移除，避免把錯誤 normalization 當成候選正解。

3. 若仍想保留 `simplex_strict`，建議在 README 與 CLI help 中明確寫成：
   - 它不是一般正解
   - 它只是特定 projected lifting normalization 的實驗模式

---

## 12. 一句話總結

> 綜合 repo、主資料來源 `SDG4PDEOnSphere20260416.pdf`，以及其他標準 DG / simplex 文獻，`triangle` 與 `simplex` 是數值上等價且正確的 face-order conventions，而 `simplex_strict` 因為對 projected surface lifting 額外乘上 reference triangle area `2`，不符合標準的 lifting 尺度，應判為錯誤模式。fileciteturn9file0 fileciteturn22file1 fileciteturn25file0
