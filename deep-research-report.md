# NKRlyq1213 triangle-dg-project 與球面簡報 DG 文件的一致性與穩定性分析

## 執行摘要

本次檢閱的核心結論有五點。

第一，儲存庫 `NKRlyq1213/triangle-dg-project` 的空間離散主體，**大致上有對上** `SDG4PDEOnSphere20260416.pdf` 中的三角形 nodal DG 架構：它確實有三角形基底、Vandermonde、質量矩陣、加權微分矩陣、邊界/trace 運算子、mapped split-form 體積項，以及以 `p = n·(f-f*)` 為核心的面通量抬升；這些對應到 PDF 的式 (2)–(3)、(8)、(14)–(18)、(22)、(27)、(34)–(45)。但它**不是逐式照抄**：程式把 PDF 的抽象矩陣寫成較標準的 Dubiner/weighted-projection 形式，而且在 interior face 上還額外提供 `exchange` 與 `exact_trace` 兩種 trace 模式，這是 PDF 沒明寫的擴充。fileciteturn52file0 fileciteturn40file0 fileciteturn41file0 fileciteturn42file0 fileciteturn43file0 fileciteturn44file0 fileciteturn38file0 fileciteturn39file0

第二，PDF 的 simplex basis 公式本身很可能有排版或索引錯置；repo 內 `simplex2d_mode` 用的是標準 Dubiner 形式  
\[
\psi_{ij}(r,s)=\sqrt{2}\,P_i^{(0,0)}(a)\,P_j^{(2i+1,0)}(b)\,(1-b)^i,
\]
這和 DG/高階三角形元素文獻中的標準寫法一致；而 PDF 第 2–3 頁把 Jacobi 指數與 \((1-b)\) 次方寫成了看起來互換的版本，和它後面示例也不完全自洽。也就是說，**repo 的基底實作比 PDF 公式更可信**。fileciteturn52file0 fileciteturn40file0 citeturn14search1turn14search12turn15search12

第三，`DGLG_A_central_flux.ipynb` 中的「boundary exact-value correction」本質上不是單純把 \(g(t_s)\) 直接塞到每個 stage，而是**把邊界精確值及其時間導數，和主解一起用同一組 low-storage RK stage 遞推**。repo 的 `qb-correction` 文件說明也明確寫成「evolves exact-source traces via RK-stage correction instead of reapplying exact values at every stage」，因此從設計理念上看，**repo 的修正方向是對的，而且確實是在對應 notebook 的 stage-consistent boundary forcing**；不過它只能修正「exact-source faces 的 stage-time mismatch」，**不能修補 interior flux 本身耗散不足**。fileciteturn54file0 fileciteturn53file0 fileciteturn38file0 fileciteturn39file0

第四，`tau != 0` 時會爆炸，最關鍵的原因不是「LF 太強」而是**目前 repo 根本沒有實作 PDF 式 (52) 的 Lax–Friedrichs flux**。repo 現在實作的是  
\[
f^*=\frac{(n\!\cdot\!V)q^-+(n\!\cdot\!V)q^+}{2}
+\frac{1-\tau}{2}|n\!\cdot\!V|(q^- - q^+),
\]
這只是在 **upwind**（\(\tau=0\)）和 **central**（\(\tau=1\)）之間線性調弱耗散；它對應的是 \(C=(1-\tau)|n\!\cdot\!V|\)，**不是** PDF 式 (52) 的 \(C=\alpha \max_T |V|\)（\(\alpha\ge 1\)）那種 LF 懲罰。也就是說，`tau` 越大，耗散越少；若你以為自己在「加 LF」，其實程式是在**減黏性**，爆炸正是合理結果。fileciteturn52file0 fileciteturn54file0 fileciteturn38file0 fileciteturn39file0

第五，三個測試函數裡 `sin(2π(x+y−2t))` 在 exact-value correction 後還會炸，最合理的解釋是：它是真正的二維斜向傳播，對目前 `diagonal="anti"` 的三角網格來說，**最強的法向通量正好落在 anti-diagonal 內部面**，而 boundary exact-value correction 只管外邊界，不會替 interior face 增加耗散；再加上目前 `tau` 家族只會減耗散而不會變成 LF，於是這個 case 仍然最容易把 fully discrete 的不穩定模式放大。相對地，`sin(2π(x−t))` 與 `sin(2π(y−t))` 比較接近 quasi-1D 平移，邊界 stage mismatch 一旦修正，症狀會顯著改善。fileciteturn45file0 fileciteturn47file0 fileciteturn54file0 fileciteturn52file0

## 研究範圍與證據基礎

我先用已啟用的 `entity["company","GitHub","developer platform"]` connector 檢閱該 repo，再回到上傳的 PDF 與 notebook 做理論對照。repo 的 README 與 `docs/experiments.md` 已經把這次分析最重要的控制參數說得很清楚：LSRK 實驗目前公開了 `--test-function {sin2pi_x,sin2pi_y,sin2pi_xy}`、`--qb-correction`、`--physical-boundary-mode`、`--interior-trace-mode`、`--tau`、`--tau-interior`、`--tau-qb`；而且文件直接寫明「`qb-correction` 會用 RK-stage correction 演化 exact-source traces，而不是每個 stage 重新套 exact value」；同一份文件還直接給出現行 penalty 公式  
\[
f^*=\tfrac12[(n\!\cdot\!V)q_M + (n\!\cdot\!V)q_P] + \tfrac{1-\tau}{2}|n\!\cdot\!V|(q_M-q_P),
\]
這一句幾乎已經把問題的核心講完了。fileciteturn53file0 fileciteturn54file0

PDF 方面，真正有負載力的內容集中在四段：三角形 quadrature 與邊界 quadrature 的式 (2)–(3)；Vandermonde、質量矩陣、加權微分矩陣的式 (8)、(11)、(14)–(15)；split-form advection DG 的式 (22)、(27)；以及數值通量與穩定性分析的式 (34)–(52)。最後，notebook 雖然是 1D 變係數中央通量實驗，但它對「邊界精確值必須以同一組 low-storage RK 係數 stage-by-stage 演化」這件事給了非常直接的原型。fileciteturn52file0

我另外補查了兩組一手或高權威參考：一組是 entity["organization","NASA","us space agency"] 的 Carpenter–Kennedy 1994 技術報告，用來確認 5-stage 2N-storage RK54 的來源；另一組是 DG / simplicial basis 的標準參考，用來核對三角 Dubiner basis 的正確形式。這些不是為了替代 repo 與 PDF，而是用來判斷「PDF 自己的公式」與「repo 的實作」哪一方比較像標準文獻。citeturn12view0turn14search1turn14search12turn15search12turn10view0

## PDF 與程式實作的對照

下面先給最重要的對照表。repo connector 在這次回合回傳的是**函式級內容**而非逐行號碼，所以 GitHub 程式我只能精準到「檔案 / 函式作用域」；上傳 notebook 則可給 cell 內行號。這是本次證據鏈唯一的形式限制，但不影響方法對照。fileciteturn38file0 fileciteturn39file0

| PDF / notebook 對象 | 理論內容 | repo / notebook 對應位置 | 結論 |
|---|---|---|---|
| PDF 式 (2)–(3) | 三角形面積與邊線積分 quadrature | `operators/trace_policy.py`：`_build_table1_embedded_trace`、`_build_table2_projected_trace`；`operators/boundary.py`：`volume_to_edge_operator` | **匹配**。Table 1 直接抽 embedded face nodes；Table 2 以加權投影建 face operator。fileciteturn52file0 fileciteturn48file0 fileciteturn49file0 |
| PDF simplex basis 段落 | 三角形正交基底 | `basis/simplex2d.py`：`simplex2d_mode`, `grad_simplex2d_mode` | **概念匹配，但 PDF 公式疑似誤植**；repo 使用標準 Dubiner basis。fileciteturn52file0 fileciteturn40file0 citeturn14search1turn14search12turn15search12 |
| PDF 式 (8)、(11) | \(M=|T|V^T W V\)，係數回推 | `operators/mass.py`：`mass_matrix_from_quadrature` | **匹配**。fileciteturn52file0 fileciteturn42file0 |
| PDF 式 (14)–(15) | \(D_\xi=|T|V_\xi M^{-1}V^TW\)，\(D_\eta\) 同理 | `operators/differentiation.py`：`differentiation_matrices_weighted` | **匹配**。另有 `square` 版本供方陣節點情形。fileciteturn52file0 fileciteturn43file0 |
| PDF 式 (22)、(27) | split-form advection DG | `operators/divergence_split.py`：`mapped_divergence_split_2d`；`operators/rhs_split_conservative_exchange.py`、`operators/rhs_split_conservative_exact_trace.py` | **匹配**，但 repo 用 mapped \((r,s)\) 係數 \(\alpha,\beta\) 的更一般寫法。fileciteturn52file0 fileciteturn44file0 fileciteturn38file0 fileciteturn39file0 |
| PDF 式 (34)–(45) | \(p=n\cdot(f-f^*)\) 與邊界抬升 | `upwind_flux_and_penalty`、`surface_term_from_exchange`、`surface_term_from_exact_trace` | **匹配**。面項確實是先算 \(p\)，再用逆質量/抬升灌回 volume DOF。fileciteturn52file0 fileciteturn38file0 fileciteturn39file0 |
| PDF 式 (48)–(52) | 中央 / 上風 / LF 介面懲罰 \(C\) | `upwind_flux_and_penalty` + docs 中 `tau` 說明 | **部分匹配**：只實作了 \(C=(1-\tau)|n\cdot V|\) 的 upwind–central 連續族，**沒有**真正的 LF \(C=\alpha \max_T|V|\)。fileciteturn52file0 fileciteturn54file0 fileciteturn38file0 |
| notebook cell 0 行 109–114, 295–307 | 邊界精確值以同一 RK stage 遞推 | repo `qb-correction` 說明；面算子內 `_apply_exact_source_q_correction` 掛點 | **設計方向匹配**。repo 說明明確指出它是在做 RK-stage exact-source trace correction。fileciteturn54file0 fileciteturn38file0 fileciteturn39file0 |

這個表裡最重要的不是「完全 matching」的那些欄，而是兩個**不完全 matching**：

一個是 simplex basis。PDF 的第 2–3 頁把三角形基底寫成了不太像標準 Dubiner basis 的形式，而 repo 的 `simplex2d.py` 則正好是教科書/社群最常用的實作。就數值軟體角度，我會把 repo 視為正確、把 PDF 視為有公式排版風險。fileciteturn52file0 fileciteturn40file0 citeturn14search1turn14search12turn15search12

另一個是 LF flux。PDF 在式 (52) 非常清楚地把 LF 定義為 \(C=\alpha \max_T |V|\), \(\alpha\ge 1\)；repo 文件則同樣清楚地寫著現況仍是  
\[
f^*=\tfrac12[(n\!\cdot\!V)q_M+(n\!\cdot\!V)q_P]+\tfrac{1-\tau}{2}|n\!\cdot\!V|(q_M-q_P).
\]
所以如果你的研究問題是「repo 有沒有按 PDF 做 LF？」答案是**沒有**；它目前做的是「把 upwind penalty 乘上一個 \(1-\tau\) 的縮放」，只是一種 reduced-upwind / central interpolation。fileciteturn52file0 fileciteturn54file0 fileciteturn38file0

## 邊界精確值修正與 notebook 的一致性

先說 notebook 在做什麼。`DGLG_A_central_flux.ipynb` 的 cell 0 把邊界值 \(g(t)\)、一階導數 \(g_t\)、二階導數 \(g_{tt}\) 打包成 `g_array`，再用 `Flux_bd(g,t,pa) = [g_t, g_{tt}, g_{ttt}]` 當成一個小型 ODE，在主解同一個 low-storage RK54 迴圈裡同步更新。關鍵不在於 `g` 是 exact，而在於它在**stage 內部**也維持和主解完全一致的更新節奏；真正送進 RHS 的不是 \(g(t_s)\) 的即時重取，而是 `g_array[0]` 這個經過同樣 stage 演化的邊界狀態。notebook 的關鍵程式位置是 cell 0 第 109–114 行與第 295–307 行；而面通量真正使用該 stage 邊界值的地方，是 cell 0 第 246–256 行。這個結構非常清楚。  

接著看 repo。repo 的 README 與 `docs/experiments.md` 都把 `qb-correction` 說成「evolves exact-source traces via RK-stage correction instead of reapplying exact values at every stage」，而且限制條件也寫得很準：只有在 `interior-trace-mode=exact_trace` 或 `physical-boundary-mode=exact_qb` 時，這個修正才有 exact source 可修。再加上 `rhs_split_conservative_exchange.py` / `rhs_split_conservative_exact_trace.py` 都明確保留 `q_boundary_correction` 與 `_apply_exact_source_q_correction` 的掛接位置，表示這不是 README 的口號，而是實際進入表面通量組裝路徑的功能。fileciteturn54file0 fileciteturn53file0 fileciteturn38file0 fileciteturn39file0

因此，對問題「boundary exact-value correction 是否有依 notebook 正確實作」我的判斷是：

- **若問題限定為**：「它是不是在修正 RK stage 中，exact boundary data 與內部 low-storage stage history 不一致的缺陷？」  
  答案是**是**；這個邏輯和 notebook 相符。fileciteturn54file0 fileciteturn38file0 fileciteturn39file0

- **若問題擴大成**：「它能不能保證整個 fully discrete 解在 `tau!=0` 時都穩定？」  
  答案是**不能**；因為 notebook 型修正只作用在 exact-source boundary trace，並不會替 interior face 產生新的數值耗散。fileciteturn54file0 fileciteturn52file0

這一點可以用流程圖看得更清楚：

```mermaid
flowchart LR
    A[主解 q^n] --> B[LSRK stage s]
    C[exact boundary source g, g_t, g_tt, g_ttt] --> D[用同一組 A/B/C 係數更新 boundary stage state]
    D --> E[qP_exact_corrected]
    B --> F[qM on faces]
    F --> G[數值通量 f*]
    E --> G
    G --> H[p = n·(f-f*)]
    H --> I[face lift / inverse mass]
    I --> J[stage RHS]
    J --> K[q 的下一個 stage]
```

這個修正的數學意義也可以說得更嚴謹一點。邊界資料若每個 stage 都直接用 \(g(t+c_s\Delta t)\) 重取，而主解 \(q\) 卻是經 low-storage accumulator 累積而成，那麼外力/邊界面在 stage 層級上其實不共享同一個 B-series 歷史；這會把一個額外的 time-discretization forcing defect 注入 inflow face。notebook 的做法，正是把 \(g\) 本身也變成一個同階 RK ODE 變數，讓兩邊的 stage history 對齊。repo 的 `qb-correction` 文件描述與掛點正是在做這件事。fileciteturn54file0

## 為什麼 tau 不等於零時會爆炸

這一節是整份報告最核心的診斷。

先把 repo 的現行通量寫成數學式。`upwind_flux_and_penalty` 寫的是  
\[
f = (n\!\cdot\!V)\,q_M,
\qquad
f^* = \frac{(n\!\cdot\!V)q_M + (n\!\cdot\!V)q_P}{2}
      + \frac{1-\tau}{2}|n\!\cdot\!V|(q_M-q_P),
\]
\[
p = f-f^*.
\]
這和 PDF 的邊界/介面能量分析放在一起看，可以直接得到：若在 interior face 上兩側法向互為相反，則總介面耗散正比於  
\[
-(1-\tau)|n\!\cdot\!V|\,[q]^2.
\]
所以 **\(\tau\) 越大，耗散越小；\(\tau=1\) 時就是 central flux，耗散為零。** 這跟「LF 會更穩」剛好相反。fileciteturn38file0 fileciteturn39file0 fileciteturn52file0

也因此，repo 目前的 `tau` 不是 PDF 式 (52) 的 LF 參數。PDF 的 LF 是  
\[
C=\alpha\max_T |V|,\qquad \alpha\ge 1,
\]
也就是當你懷疑 fully discrete 穩定性不夠時，**把懲罰加大**。但 repo 的 `tau` 家族是  
\[
C_{\text{repo}}=(1-\tau)|n\!\cdot\!V|,
\]
其作用是**把 upwind penalty 往 0 縮**。因此問題其實不是「為什麼 LF 還會炸」，而是「為什麼把耗散削弱了之後會炸」；這一點完全符合數學預期。fileciteturn54file0 fileciteturn52file0

這個結論還可以和 PDF 最後一頁的話精準對上。PDF 自己就說：即便 semi-discrete 層級 \(C>0\) 有穩定性，到了 fully discrete 層級也**可能仍不穩定**，因為耗散不夠；必要時要提高 \(\alpha\)，代價是更小時間步。repo 目前做的卻是把 \(C\) 從 upwind 的 \(|n\!\cdot\!V|\) 再往下縮，當然更容易碰到 fully discrete 放大。fileciteturn52file0

再進一步看三個測試函數。repo 的 structured mesh 預設是 `diagonal="anti"`，也就是每個矩形切成一條由左上到右下的內部對角面。這條 anti-diagonal 的單位法向量可取為 \((\pm1,\pm1)/\sqrt2\)。因此三個速度方向在這種內部面上的 \(|n\!\cdot\!V|\) 是：

| 測試型態 | 可視為速度方向 | anti-diagonal face 上 \( |n\!\cdot\!V| \) |
|---|---:|---:|
| \(\sin(2\pi(x-t))\) | \(V=(1,0)\) | \(1/\sqrt2 \approx 0.707\) |
| \(\sin(2\pi(y-t))\) | \(V=(0,1)\) | \(1/\sqrt2 \approx 0.707\) |
| \(\sin(2\pi(x+y-2t))\) | \(V=(1,1)\) | \(\sqrt2 \approx 1.414\) |

也就是說，`sin2pi_xy` 在這些**最重要的 interior diagonal faces** 上，面法向通量正好最大；一旦 `tau` 把耗散削弱，它是第一個把 underdamped interior mode 放大的 case。這是理論上非常合理、而且和 repo mesh 幾何完全一致的結果。fileciteturn45file0 fileciteturn47file0

如果把這件事寫成最小的兩元素介面模型，取一個內部面跳躍 \([q]=q^- - q^+\)，則現行 repo 通量在該面提供的能量耗散就是  
\[
\mathcal D_\Gamma = -(1-\tau)|n\!\cdot\!V|[q]^2.
\]
對 `sin2pi_xy` 在 anti-diagonal 內部面上，\(|n\!\cdot\!V|=\sqrt2\)，所以同樣的 \(\tau\) 會比 x-only 或 y-only case 更早把耗散吃光。例子如下：

| \(\tau\) | 現行 \(C_{\text{repo}}=(1-\tau)\sqrt2\) | 對 jump \( [q]=2 \) 的耗散 \( -(1-\tau)\sqrt2\,[q]^2 \) |
|---:|---:|---:|
| 0.0 | 1.414 | -5.657 |
| 0.5 | 0.707 | -2.828 |
| 0.9 | 0.141 | -0.566 |
| 1.0 | 0 | 0 |

這個表本身就已經說明：你把 `tau` 從 0 調大，不是在「啟用 LF」，而是在**系統性地拿掉內部面耗散**。

最後還有 fully discrete 的角度。repo / notebook 所用的 5-stage 2N-storage RK 係數出自 Carpenter–Kennedy 1994；這類 RK 法對純虛特徵值的穩定範圍有限，中央通量或弱耗散通量會把半離散譜推向虛軸附近，而 upwind/LF 則會把譜往左半平面拉。換句話說，當 `tau` 讓 \(C\) 下降時，你同時失去的是「面耗散」與「時間離散可承受的實部裕量」。這就是為什麼 PDF 最後要特別提醒 fully discrete 可能仍不穩，並建議用更大的 \(\alpha\) 做 LF。citeturn12view0turn11search0turn52file0

## Lax–Friedrichs 實作修正方案

這裡我給出的是**可直接落地**的修正方案，不是抽象建議。

### 修正原則

目前 repo 把「flux 類型」和「耗散大小」綁在同一個 `tau` 上。這會讓語意混亂，也直接阻礙 LF。正確做法應該把它拆成兩件事：

- `flux_mode`：`{"central","upwind","lf","tau_blend"}`  
- `C_face`：真正進入 PDF 式 (48)–(52) 的懲罰係數

也就是說，數值通量統一寫成  
\[
f^*=\frac{(n\!\cdot\!V)q_M+(n\!\cdot\!V)q_P}{2}+\frac{C_{\text{face}}}{2}(q_M-q_P),
\]
其中：

- `central`：\(C_{\text{face}}=0\)
- `upwind`：\(C_{\text{face}}=|n\!\cdot\!V|\)
- `tau_blend`：\(C_{\text{face}}=(1-\tau)|n\!\cdot\!V|\)  
  這是保留現況的 backward-compatible 模式
- `lf`：\(C_{\text{face}}=\alpha\,V_{\max,\text{ref}}\)，\(\alpha\ge 1\)  
  這才是 PDF 的 LF

這樣做以後，`tau` 不再假冒 LF，LF 也終於能真的比 upwind 更黏。fileciteturn52file0 fileciteturn54file0

### 建議 patch

下列 patch 的重點只有三個：  
一，新增通量模式；二，把 `tau` 退回成可選的 blend；三，為 LF 額外引入 `lf_alpha` 與 `c_face`。這樣不會破壞既有 `tau=0` 的 upwind 路徑，也能最小成本地接上 LF。

```diff
diff --git a/operators/rhs_split_conservative_exchange.py b/operators/rhs_split_conservative_exchange.py
@@
-def upwind_flux_and_penalty(ndotV, qM, qP, tau: float | np.ndarray = 0.0):
+def numerical_flux_and_penalty(
+    ndotV,
+    qM,
+    qP,
+    *,
+    flux_mode: str = "upwind",
+    tau: float | np.ndarray | None = None,
+    c_face: float | np.ndarray | None = None,
+):
@@
-    f = ndotV * qM
-    fstar = 0.5 * (ndotV * qM + ndotV * qP) + 0.5 * (1.0 - tau) * np.abs(ndotV) * (qM - qP)
+    f = ndotV * qM
+
+    if c_face is None:
+        mode = str(flux_mode).strip().lower()
+        if mode == "central":
+            c_face = np.zeros_like(ndotV)
+        elif mode == "upwind":
+            c_face = np.abs(ndotV)
+        elif mode == "tau_blend":
+            if tau is None:
+                tau = 0.0
+            c_face = (1.0 - np.asarray(tau, dtype=float)) * np.abs(ndotV)
+        elif mode == "lf":
+            raise ValueError("flux_mode='lf' requires c_face to be precomputed.")
+        else:
+            raise ValueError("Unknown flux_mode.")
+
+    c_face = np.asarray(c_face, dtype=float)
+    if c_face.shape != ndotV.shape:
+        raise ValueError("c_face must have the same shape as ndotV.")
+
+    fstar = 0.5 * (ndotV * qM + ndotV * qP) + 0.5 * c_face * (qM - qP)
     p = f - fstar
     return f, fstar, p
```

```diff
diff --git a/operators/rhs_split_conservative_exchange.py b/operators/rhs_split_conservative_exchange.py
@@
-def surface_term_from_exchange(..., tau=0.0, tau_interior=None, tau_qb=None, ...):
+def surface_term_from_exchange(
+    ...,
+    flux_mode: str = "upwind",
+    tau=0.0,
+    tau_interior=None,
+    tau_qb=None,
+    lf_alpha: float = 1.0,
+    ...
+):
@@
-    tau_interior_eff, tau_qb_eff, tau_face = build_face_tau_array(...)
-    f, fstar, p = upwind_flux_and_penalty(ndotV, qM, qP, tau=tau_face)
+    tau_interior_eff, tau_qb_eff, tau_face = build_face_tau_array(...)
+
+    if flux_mode == "lf":
+        speed_face = np.sqrt(u_face**2 + v_face**2)
+        # 最簡單穩妥版：全域/當前 face 最大速率
+        c_face = lf_alpha * np.max(speed_face) * np.ones_like(ndotV)
+    elif flux_mode == "tau_blend":
+        c_face = (1.0 - tau_face) * np.abs(ndotV)
+    elif flux_mode == "upwind":
+        c_face = np.abs(ndotV)
+    elif flux_mode == "central":
+        c_face = np.zeros_like(ndotV)
+    else:
+        raise ValueError("Unknown flux_mode.")
+
+    f, fstar, p = numerical_flux_and_penalty(
+        ndotV, qM, qP,
+        flux_mode=flux_mode,
+        tau=tau_face,
+        c_face=c_face,
+    )
```

```diff
diff --git a/operators/rhs_split_conservative_exact_trace.py b/operators/rhs_split_conservative_exact_trace.py
@@
-    f, fstar, p = upwind_flux_and_penalty(ndotV, qM, qP, tau=tau_face)
+    # 與 exchange 版本一致：支援 central / upwind / tau_blend / lf
+    ...
```

同樣的修正也要同步進 `rhs_split_conservative_exact_trace.py`；否則 `exchange` 與 `exact_trace` 兩條路徑會在相同參數下給出不同物理語意。這是必要條件，不是附帶條件。fileciteturn38file0 fileciteturn39file0

### 測試計畫

我建議測三層。

第一層是**單元測試**。檢查 `central`, `upwind`, `tau_blend`, `lf` 四種 `c_face` 是否真的對應到預期公式，且 `lf_alpha>1` 時 \(C_{\text{face}}\ge |n\!\cdot\!V|\)。這層最便宜，但能避免你把 LF 又實作成 reduced-upwind。fileciteturn52file0

第二層是**兩元素介面能量測試**。用兩個元素、一個 interior face、常速率 \(V\) 與固定跳躍 \([q]\)，直接驗證總介面耗散是否回到  
\[
-(C_{\text{face}})[q]^2.
\]
這個測試最能保證你的實作和 PDF 式 (49)–(52) 是同一件事。fileciteturn52file0

第三層是**LSRK 回歸測試**。直接跑 repo 已公開的三個 test functions：

- `sin2pi_x`
- `sin2pi_y`
- `sin2pi_xy`

至少固定 `mesh_level=8,16`，比對 `upwind`、`tau_blend(tau=0.5)`、`lf(lf_alpha=1.0,1.2,1.5)`。預期結果應是：

- `upwind`：有界
- `tau_blend(tau=0.5)`：某些組合仍可能長時間成長
- `lf(alpha>=1.0)`：應比 `tau_blend` 明顯穩
- `sin2pi_xy` 對 `lf_alpha` 最敏感；若 `alpha=1.0` 仍太弱，可按 PDF 提示往上加。fileciteturn54file0 fileciteturn52file0

## 為什麼 sin(2π(x+y−2t)) 在 exact-value correction 後仍然會炸

這個問題必須把「邊界問題」和「內部面問題」分開看。

若是 `sin(2π(x−t))` 或 `sin(2π(y−t))`，解基本上只沿一個座標方向平移。這時候最顯著的誤差來源通常是 inflow boundary trace 的 stage mismatch；notebook 型的 `qb-correction` 把這個缺陷修掉後，解往往就不再被持續注入錯誤外力。因此你會看到修正很有效。fileciteturn54file0

但 `sin(2π(x+y−2t))` 完全不同。它不是單向平移，而是沿 \((1,1)\) 方向斜著穿網格。對 repo 預設的 anti-diagonal 網格切法來說，這個方向在 interior diagonal face 上的法向通量最大，代表整個解最依賴**內部面耗散是否足夠**。而 `qb-correction` 再怎麼精準，都只會修 exterior exact-source trace；它不會平空在 interior face 上生成 LF 耗散。因此，只要現行 exterior-corrected / interior-underdamped 的機制不改，`sin2pi_xy` 仍然最可能爆掉。fileciteturn45file0 fileciteturn47file0 fileciteturn54file0

還有一個更細的觀點。repo 除了 `physical-boundary-mode=exact_qb` 外，還有 `interior-trace-mode={exchange,exact_trace}`。這表示內部面的 \(q^+\) 可以是真實鄰居 trace，也可以是 exact trace。若你用的是 `exchange`，那 interior instability 幾乎完全取決於面通量耗散；若你用的是 `exact_trace`，雖然 interior side 的 trace data 更乾淨，但面通量若仍採中央或弱耗散，也只是把「錯誤的來源」從 exchange mismatch 改成「中央/弱耗散的 fully discrete 放大」，不會自動穩。這也是為什麼 exact-value correction 不是萬靈丹。fileciteturn53file0 fileciteturn54file0 fileciteturn39file0

我對這個 case 的修補優先順序會是：

1. **先把 LF 真正做對**，不要再用 `tau` 假扮 LF。  
2. 對 `sin2pi_xy` 先測 `flux_mode="lf", alpha=1.0`。  
3. 若 `tf=2π` 仍有成長，再測 `alpha=1.2`、`1.5`。  
4. 若仍有問題，才再檢查 CFL 是否必須隨 `alpha` 重新縮放。  

這個順序是因為 PDF 自己就已經告訴你：fully discrete 若不穩，首先要調的是 \(C\)；不是先懷疑 boundary correction。fileciteturn52file0

## 最小可重現測試、預期結果與可交付項

這裡整理成你可以直接放進 repo issue / PR / 實驗紀錄的形式。

### 最小可重現測試

**測試一：確認現行 `tau` 不是 LF**

設定一個 face，令 \(n\!\cdot\!V=0.8\)、\(q_M=1\)、\(q_P=0\)。

- 現行 repo  
  \[
  C_{\text{repo}}=(1-\tau)|n\!\cdot\!V|
  \]
  若 \(\tau=0.5\)，得到 \(C=0.4\)

- 真 LF（PDF 式 (52)）  
  \[
  C=\alpha \max_T |V|
  \]
  若 \(\max_T|V|=1\)、\(\alpha=1.2\)，得到 \(C=1.2\)

兩者方向完全相反：一個在減耗散，一個在加耗散。若此測試不過，代表 LF 修正仍未實現。fileciteturn52file0 fileciteturn54file0

**測試二：兩元素介面能量測試**

兩元素共一 interior face，給 \(q^- = 1\)、\(q^+ = -1\)。應驗證：

- `central`：總面耗散 \(=0\)
- `upwind`：總面耗散 \(=-|n\!\cdot\!V|[q]^2\)
- `lf(alpha)`：總面耗散 \(=-\alpha V_{\max}[q]^2\)

這個測試可以直接檢查你是否真的回到 PDF 的式 (49)–(52)。fileciteturn52file0

**測試三：repo 三個公開 test function**

固定：

- `mesh_levels = 8,16`
- `tf = 2*pi`
- `physical-boundary-mode = exact_qb`
- `qb-correction = on`

分別比較：

- `flux_mode=upwind`
- `flux_mode=tau_blend, tau=0.5`
- `flux_mode=lf, alpha=1.0`
- `flux_mode=lf, alpha=1.2`

預期：

- `sin2pi_x`, `sin2pi_y`：`qb-correction` 應顯著減少長時間誤差漂移
- `sin2pi_xy`：`tau_blend` 可能仍不穩；真正的 `lf` 應有明顯改善，且 `alpha` 越大越穩，但需更小 dt。fileciteturn54file0 fileciteturn52file0

### 建議的診斷表

你可以把每次跑出的 CSV 收斂成下面這種表。repo CLI 已經支援時間誤差圖與區分 baseline / corrected 輸出，所以這張表是能落地的。fileciteturn53file0 fileciteturn54file0

| test function | flux mode | 參數 | qb-correction | 預期現象 |
|---|---|---|---|---|
| `sin2pi_x` | upwind | – | off | 可跑，但長時間誤差可能受邊界 stage mismatch 汙染 |
| `sin2pi_x` | upwind | – | on | 誤差明顯下降或至少不再單調漂移 |
| `sin2pi_y` | upwind | – | on | 與上類似 |
| `sin2pi_xy` | tau_blend | `tau=0.5` | on | 仍可能成長甚至 blow-up |
| `sin2pi_xy` | lf | `alpha=1.0` | on | 應比 `tau_blend` 穩 |
| `sin2pi_xy` | lf | `alpha=1.2~1.5` | on | 最穩，但時間步需更保守 |

### 最終判斷

如果把使用者要求濃縮成一句話，那就是：

**repo 的 DG 主骨架大致符合 PDF；boundary exact-value correction 的設計也確實對應 notebook 的 RK-stage 同步修正；真正的失配點在於目前 `tau` 不是 LF，而只是把上風耗散往中央通量方向削弱，所以 `tau!=0` 爆炸不是反常，而是程式語意與 PDF 式 (52) 不一致後的必然結果。** `sin(2π(x+y−2t))` 在修正後仍炸，則是因為它最依賴 interior diagonal face 的耗散，而 qb-correction 只修邊界、完全不修內部面。fileciteturn52file0 fileciteturn54file0 fileciteturn38file0 fileciteturn39file0 fileciteturn45file0 fileciteturn47file0