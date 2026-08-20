# ESSGNN 判讀重審 — §2.5 vs Appendix C

**日期：2026-08-19 · 這一輪只研究，未修改任何 code / docs / tests。**

權威來源：`docs/paper/metafind_source/2methdology.tex`、`docs/paper/metafind_source/appendix.tex`、
`docs/paper/egnn_source/sections/model.tex`、`metafind/vendor/egnn_clean.py`。

本文推翻先前登記於 `docs/audit/C_PAPER_CONTRADICTIONS.md#c1` 與 U-26 的判讀。
推翻的理由與證據逐條列於下。**尚未套用到任何檔案。**

---

## 0. 一句話結論

先前判讀認為論文定義了**兩個 architecture family**（`sec25_two_mlp` / `appendix_shared_msg`），
需要在其中擇一。**該分類無論文證據支持。**

論文只描述一套架構，但該描述含有數處錯誤，且與附錄的證明前提衝突。

---

## 1. QUESTION A — 是否存在兩個 architecture family

### 判定：**UNSUPPORTED**

### 先前的理由，以及它為何無效

先前理由（`metafind/models/essgnn.py:90-91`）：

> The two ESSGNNs MetaFind describes. Not a style choice —
> different parameter counts, different gradient paths, different functions.

以及（`docs/audit/E_GRAPH_REVALIDATION.md:173`）：

> C1 two ESSGNNs | suspected | `VERIFIED` — §2.5 has `f_h`/`f_x`, the appendix has `φ_e`/`φ_x`/`φ_h`

**兩條都無效。**

- 「different parameter counts / gradient paths」描述的是**我們實作的性質**，不是論文的性質。
  這是 OBSERVED IMPLEMENTATION 被當成 PAPER FACT。
- `E_GRAPH_REVALIDATION.md:173` 標記 `VERIFIED`，而它給出的理由**就是符號差異本身** ——
  即「附錄有 `m_ij`，正文沒有」。該論證不成立（見下）。

### 為何符號差異推不出架構差異

§2.5 的 `f_h(d, h_i, h_j, e)`、`f_x(d, h_i, h_j, e)` 是**函數名稱**，
論文只說 "approximate using multilayer perceptrons (MLPs)"，
**未規定內部結構**。`f_h` 完全可能內含 `φ_h ∘ φ_e` 這樣的 factorization。

### 支持 UNSUPPORTED 的三項證據

| 證據 | 出處 |
|---|---|
| 附錄從未給 `φ_e` 的 codomain，也未給 `m_ij` 的維度 | `appendix.tex` 全文唯一的 `\mathbb{R}` 為 `g ∈ R^3`、`Q ∈ R^{3×3}` |
| 論文從未斷言 `θ_h ∩ θ_x = ∅` | `2methdology.tex:54` 僅寫 "parameterized by θ_h and θ_x, respectively" |
| 沒有維度限制即沒有瓶頸 | 附錄讀法下 `θ_h = (θ_φe, θ_φh)`、`θ_x = (θ_φe, θ_φx)`，兩者**本就重疊** |

### 一項必須避免的過度聲稱

本輪曾以 `φ_e = identity` 做建構示範，得到位元級 0 差異
（`f_h` 與 `f_x` 輸出最大差皆為 `0.000e+00`）。

**該示範證明的比「兩者 function class 相同」少。** 它只證明*存在*一個附錄形式的實例
等於給定的 §2.5 實例。在**固定有限寬度／深度**下（程式中 `φ_e` 輸出寬度為
`hidden_dim=128`，即真實瓶頸），兩者不必然等價。

**安全的陳述只有一句：**

> 論文沒有足夠證據支持 shared-vs-independent 是兩個 MetaFind architecture family。

---

## 2. QUESTION B — 完整 layer update 是否相同

### 判定：**PROVABLY DIFFERENT**（layer-for-layer）

### 承重理由：一個上標，與 `m_ij` 無關

```
§2.5 Eq.(3)   f_x( d_ij^l , h_i^(l+1) , h_j^(l+1) , e_ij )      <- l+1
附錄 Eq.(13)  φ_x( m_ij ) ,  m_ij = φ_e( h_i^(l) , h_j^(l) , ... )   <- l
```

因 `h_i^(l+1) = h_i^l + Σ_{k∈N(i)} f_h(..., h_k^l, ...)`，
§2.5 中邊 `(i,j)` 的 coordinate gate 依賴 `{h_k^l : k ∈ N(i) ∪ N(j)}`；
附錄的 gate 僅依賴 `{h_i^l, h_j^l, x_i^l, x_j^l, e_ij}`。

`h_k^l`（`k ∉ {i,j}`）**不在 `φ_e` 的參數列中**，故任何 `(φ_e, φ_h, φ_x)`
皆無法重現 §2.5 的映射（在 `f_h` 對其 `h_j` 槽非常數的前提下）。

### 數值反例（已實測）

路徑圖 `0—1—2`，點 0 與點 2 不相鄰。擾動 `h_2`，量測點 0 的更新座標變化：

| 形式 | Δx₀ |
|---|---|
| 附錄式（`ESSGCLShared`） | **`0.000000e+00`**（位元級精確 0） |
| §2.5 式（`ESSGCL`） | `4.191246e-02` |

（float64；coordinate gate 末層初始化調至 `std=0.5` 以排除數值雜訊。）

### 第二個獨立差異：`f_x` 的值域

見 §4.2。

### 第三個差異：鄰域寫法

逐字查證，**兩式皆一致地不同**，非隨手筆誤：

```
2methdology.tex   \sum_{j \in \mathcal{N}(i)}    出現 2 次（Eq.2, Eq.3）
appendix.tex      \sum_{j \ne i}                 出現 2 次（Eq.13, Eq.14）
```

**不列為矛盾。** EGNN `model.tex:46` 明文：

> we choose to aggregate messages from all other nodes `j ≠ i`, but we could limit
> the message exchange to a given neighborhood `j ∈ N(i)` if desired **in both equations**

實測規模差距（12,000 間房）：

```
現有 sem_edge_ids（N(i)）         4,128,637
完全圖 j≠i 所需                  39,919,647     9.7 倍
```

### A 與 B 可同時成立

`A = UNSUPPORTED` 與 `B = PROVABLY DIFFERENT` 不衝突。
**B 的成立完全不依賴 A。**

---

## 3. 撤回：`r` vs `r²`

### 判定：**ACCEPT 撤回**

先前將 `‖x_i − x_j‖₂`（§2.5）vs `‖x_i − x_j‖²`（附錄）列為
「不同 function class」。**該分類不精確。**

`r ↦ r²` 在 `[0, ∞)` 上為雙射，故對 unrestricted learnable functions，
給定任意 `f(r)`，令 `g(u) := f(√u)` 即有 `g(r²) = f(r)`。實測差 `0.000e+00`。

### 重新分類

| 層級 | 判定 |
|---|---|
| function-spec | **不是**核心 architecture distinction |
| training / numerical | **可能有影響**，且已量測 |

實測（400 間房、137,700 條真實 `phys_edge`）：

| | 中位數 | p99 | max | max/median |
|---|---|---|---|---|
| `r` | 1.2548 m | 4.8562 | 8.3675 | 6.7× |
| `r²` | 1.5745 | 23.5829 | 70.0154 | **44.5×** |

梯度行為：

```
d/dx |x|  @ [1e-4, 1e-2, 1.0, 5.0] -> [1.0,    1.0,  1.0,  1.0 ]   常數
d/dx x²   @ [1e-4, 1e-2, 1.0, 5.0] -> [0.0002, 0.02, 2.0,  10.0]   跨 5e4
```

另：`‖·‖₂` 在 0 不可微（實測 383 條邊 `r < 0.1 m`）；`‖·‖²` 處處平滑。

**不再與 `coord_feat` 置於同一層級。**

---

## 4. 真正成立的 paper contradictions

### 4.1 `h⁰` — PAPER TYPE / DIMENSION INCONSISTENCY

`appendix.tex:29` 明文前提：

> We begin by assuming that `h⁰` is invariant to SE(3) transformations on `x`

`2methdology.tex:44` 明文定義：

> `h_i^(0) = Concat(x_i, t_i)`

**三個獨立破口：**

1. **SE(3) 前提破。** `Concat(Qx+g, t) ≠ Concat(x, t)`，**連純平移都不成立**。
   後果：`appendix.tex:56` 將 `Q` 提出 `Σ (Qx_i − Qx_j)·φ_x(m_ij)` 的步驟失效，
   因 `φ_x(m_ij)` 不再 invariant。歸納法基底崩潰，所有 `h^l` 皆非 invariant。

2. **殘差寬度不閉合。** `h⁰ ∈ R^(d+3)`、`f_h → R^d`，`h⁰ + f_h(...)` 在 `l=0` 未定義。

3. **參數寬度不閉合。** 第 0 層餵給 `f_h` 的實際寬度為
   `1 + (d+3) + (d+3) + e = 2d+7+e`，但宣告 domain 為 `R^(2d+1+e)`。

**不替作者重新定義 `d`。三個破口皆照原文記錄。**

### 4.2 `f_x → R³` — GENUINE CONTRADICTION

`2methdology.tex:54` 逐字：`f_x: \mathbb{R}^{(2d + 1 + e)} \to \mathbb{R}^{3}`。
**PAPER FACT，非轉檔錯誤。**

與論文自身的 SE(3)-equivariance 宣稱及附錄證明衝突：
`(x_i − x_j)·φ_x(m_ij)` 要能提出 `Q`，`φ_x` 必須為 invariant scalar。
`R³` 唯一維度自洽的讀法是 Hadamard，而 `Q(a ⊙ b) ≠ (Qa) ⊙ b`。

實測：

| gate 型別 | `‖Q·f(x) − f(Qx)‖∞` |
|---|---|
| 純量（附錄 / EGNN） | `2.22e-16` ✅ |
| `R³`（§2.5 字面） | **`4.31e-01`** ❌ |

UPSTREAM FACT：
- EGNN `model.tex:44`：`φ_x: R^nf → R^1` … "outputs a scalar value"
- 參考實作 `metafind/vendor/egnn_clean.py:65`：`nn.Linear(hidden_nf, 1, bias=False)`

**這不是 implementation choice。**

### 4.3 `h^(l+1)` 是否為 typo — **UNKNOWN**

支持 typo 的**弱**證據：EGNN 使用 `h^l`；附錄使用 `h^l`；
論文稱 ESSGNN "extends the EGNN formulation"。

**但第三條被削弱**（見 §6.1）：附錄的 h-update 本身即非 EGNN 形式。

**維持 `PLAUSIBLE TYPO` + `UNKNOWN`。復現不得逕行修改。**

### 4.4 不列為矛盾者

- `N(i)` vs `j≠i` — EGNN 明文兩者皆可，是選項非矛盾
- 附錄稱 `Q` 為 orthogonal（含鏡射，O(3)），§2.5 稱 `R ∈ SO(3)`，兩處皆標 SE(3)
  — 標籤鬆散，證明對 O(3) 亦成立，**數學無誤**

---

## 5. `e_ij` equivariance premise

**必須分開陳述：**

**PAPER PREMISE**（`appendix.tex:29`）：
> `e_ij` "derived solely from object-level textual descriptions and thus
> **independent of spatial coordinates**"

**OBSERVED RUNTIME**（實測追蹤）：

```
procthor_object_text.json        {'Apple_24': 'a apple'}       僅類別名
      ↓
build_relation_prompt()          metafind/data/semantic_edges.py:151-168
                                 僅吃 desc_i, desc_j；含 assert 擋空間詞
      ↓
Qwen → 關係句 → frozen CLIP ViT-B/32 → e_ij (4242, 512)
```

cache key = `sha256(desc_i, desc_j, prompt_version, llm_model, encoder_version)`
—— `x` 不在任何一環。

實測 4,242 條產出句子：

| 檢查 | 結果 |
|---|---|
| 含阿拉伯數字 | **0 / 4242** |
| 含空間詞 | 958 / 4242（near 458、next to 392、beside 54、left 17、above 7） |

**判定：premise 守住。**

那 958 條不違反前提 —— 判準為**函數依賴**而非字面。
"under" 由**類別對**生成；SE(3) 變換下 `desc_i`／`desc_j` 不變 → 句子不變 → `e_ij` 不變。

**但這是 OBSERVED RUNTIME，不是機制保證。**
`assert` 僅擋 prompt，不擋 LLM 輸出。本次為 0 是結果，非保證。

---

## 6. 現行文件未記載的補充發現

### 6.1 附錄的 h-update 不是 EGNN 的

```
EGNN     h_i^{l+1} = φ_h( h_i^l , m_i )           先聚合再變換
MetaFind h_i^{l+1} = h_i^l + Σ_j φ_h(m_ij)        逐邊變換再加總
附錄
```

**「附錄 = EGNN，故繼承 upstream 權威」此論證不成立。**
§2.5 的 Eq.(2) 反而與附錄的 h-update 同型；**兩者皆偏離 EGNN**。

### 6.2 EGNN 的正規化常數在兩處皆缺席

EGNN `model.tex:15` 有 `C = 1/(M-1)`；MetaFind §2.5 與附錄皆無。
兩邊一致省略，故非兩者間的差異，但復現時須記為 DEVIATION。

### 6.3 混淆的源頭

`docs/graph/02_BUILD_STEPS.md:854` 登記 U-26 時**將兩個問題綁在一起**：

> U-26 | `f_h`／`f_x` 是否共用一條訊息，**以及 `f_x` 看到的是 `h^{l+1}` 還是 `h^l`**

而 `docs/graph/graph_spec.yaml:602` 又寫：

> coord_feat (U-26, f_x sees h^(l+1) or h^(l))

**同一編號，兩個意義。** 程式後續拆成兩個旋鈕，並**選錯主軸**。

### 6.4 一次誤讀的更正

本輪一度報告「§2.5 版等變性壞掉，誤差 `7.748e+20`」。**該結論錯誤。**
那是**絕對誤差**，當時 `|x|` 已達 `1.555e+35`。相對誤差為 `1.145e-16`，
與附錄版同等。**§2.5 形式（在 scalar gate + invariant `h⁰` 下）是完全等變的。**

### 6.5 穩定性觀察 — OBSERVED DATA + STABILITY CONCERN

§2.5 形式存在正回饋：`h` 增長 → gate 輸出增大 → 座標增大 → 平方距離增大 → `h` 更增長。

| 形式 | 4 層後 `|x|` |
|---|---|
| 附錄 | `~1.2e+02` |
| §2.5 | `~1.6e+35` |

**不得用於解決 paper ambiguity。** 單一 seed、單一設定，且受 initialization、
MLP 深度、hidden 寬度、gate scale、distance 表示、層數、normalization 影響。
相對等變誤差兩者皆 `~1e-16`，**故此為穩定性問題，非正確性問題。**

---

## 7. 現行實作審查

| 位置 | 內容 | 分類 |
|---|---|---|
| `metafind/models/essgnn.py:90-91` | 「different parameter counts, different gradient paths」 | **STALE ASSUMPTION** |
| `docs/audit/E_GRAPH_REVALIDATION.md:173` | 標 `VERIFIED`，理由為符號差異本身 | **STALE ASSUMPTION（最嚴重）** |
| `docs/audit/C_PAPER_CONTRADICTIONS.md:25-77` | C1「two different ESSGNNs — STRUCTURAL, blocking」 | **STALE ASSUMPTION** |
| `docs/graph/00_FINDINGS.md:1091` | 「正文與 Appendix C 描述的是兩個不同的架構」 | **STALE ASSUMPTION** |
| `docs/graph/README.md:17` | 「U-26 的 ESSGNN 架構也已選定（附錄的 shared-message 版）」 | **STALE ASSUMPTION** |
| `metafind/models/resolve_stage2.py:93-131` | ARCH_DECISIONS 將 family 當論文分叉並選 primary | **STALE ASSUMPTION** |
| `metafind/models/essgnn.py:191-195` | `__post_init__` 由 family 推導 `coord_feat` | **RESEARCH-SIGNIFICANT BUG** |
| `metafind/models/essgnn.py:491-503` | family 強制 `coord_feat`，否則 raise | **RESEARCH-SIGNIFICANT BUG** |
| `tests/test_resolve_stage2.py:179` | 硬斷言 `== "appendix_shared_msg"` | **STALE ASSUMPTION** |
| `tests/test_essgnn.py:21-28` | `FAMILY` = primary、`TWO_MLP` = competing hypothesis | **DOCUMENTATION-ONLY ISSUE** |
| `metafind/models/essgnn.py:298-445` | 兩個 layer class 皆實作且可執行 | **VALID IMPLEMENTATION ABLATION** |
| `outputs/essgnn_arch_protocol.json` | 缺 `architecture_family`、`coord_feat="updated"`、標 `resolved` | **RESEARCH-SIGNIFICANT BUG（獨立問題）** |

### 為何前兩個標記為 RESEARCH-SIGNIFICANT BUG

```python
# metafind/models/essgnn.py:191-195
if self.coord_feat is None:
    self.coord_feat = ("current" if self.architecture_family == "appendix_shared_msg"
                       else "updated")
```

**有論文依據的軸（`coord_feat`）成為無證據的軸（`architecture_family`）的附屬品**，
且 `essgnn.py:498` 禁止拆開。

**後果：無法在固定 sharing 結構下單獨測試 `coord_feat`** —— 而那正是唯一需要測試的維度。

### 三個變數的混淆（實測）

| family | 參數 | h→gate 線性層數 | `coord_feat` |
|---|---|---|---|
| `appendix_shared_msg` | 164,737 | 4 | `current` |
| `sec25_two_mlp` | 213,761 | 2 | `updated` |

**僅第三欄有論文依據。** 深度 4 vs 2 論文未提，為實作引入。
直接對比兩者，**任何差異皆無法歸因至單一因素**
（違反 `.claude/rules/experiments.md` §2）。

---

## 8. 建議的新實驗軸

### 主軸：`coord_feat`（唯一 paper-conflicted 維度）

```
current  =  gate 讀 h^(l)      （附錄 Eq.13）
updated  =  gate 讀 h^(l+1)    （§2.5 Eq.3）
```

**用詞：`paper-conflicted`，非 `paper-backed`。**
兩值皆為論文所寫，論文自身衝突。我們不是在論文值與替代方案間選擇，
而是在論文的兩處陳述間選擇。

### 必須固定的條件

sharing structure、MLP 深度、hidden 寬度、參數量、initialization、
distance 表示、層數。

### 次要 ablation

`shared-message` vs `independent heads` **保留**，
但標記為 **`IMPLEMENTATION ABLATION`**，
**不是** `METAFIND PAPER ARCHITECTURE FAMILY`。

---

## 9. 待修改清單（尚未執行）

| 檔案:行 | 目的 |
|---|---|
| `docs/graph/02_BUILD_STEPS.md:854` | **源頭**：U-26 拆成兩個編號（sharing 一個、coord_feat 一個） |
| `docs/audit/E_GRAPH_REVALIDATION.md:173` | 撤銷 `VERIFIED` —— 其理由為符號差異本身 |
| `docs/audit/C_PAPER_CONTRADICTIONS.md:25-77` | C1 從「two different ESSGNNs」降級為 notation difference |
| `docs/graph/00_FINDINGS.md:1091` | 同上 |
| `docs/graph/README.md:17` | U-26 結論措辭 |
| `metafind/models/essgnn.py:90-91` | 刪除無效理由 |
| `metafind/models/essgnn.py:151-165` | `architecture_family` 改標 implementation ablation，移除 "primary" |
| `metafind/models/essgnn.py:191-195` | 解除 `coord_feat` 對 family 的推導 |
| `metafind/models/essgnn.py:491-503` | 解除 family 對 `coord_feat` 的強制，使兩軸正交 |
| `metafind/models/resolve_stage2.py:93-131` | 重寫 C1 決策理由；主軸改為 `coord_feat` |
| `tests/test_resolve_stage2.py:179` | 移除對 family 的硬斷言 |
| `tests/test_essgnn.py:21-28` | primary / competing hypothesis 措辭 |
| `outputs/essgnn_arch_protocol.json` | **獨立問題**：缺欄位、建不出模型 |

---

## 10. 重現本文結論的方式

所有數值皆在記憶體中計算，未寫入 repo。重現方式：

```bash
# A：建構示範（φ_e = identity 下兩式輸出相同）
# B：路徑圖 0—1—2 擾動 h_2，量 Δx_0
# C：R^3 Hadamard gate 的旋轉等變誤差
# D：真實 ProcTHOR phys_edge 的 r 與 r² 分布
# 見本次 session 逐項腳本；均為 float64、單檔可重跑。
```

環境：`/home/kyzen/miniconda3/envs/MetaFind/bin/python`，
GPU `NVIDIA GeForce RTX 5090 (32,607 MiB)`，
git branch `main`。

---

## 11. 未解與限制

1. 僅證明 **layer-for-layer** 非等價。未證明「§2.5 疊 `L` 層 ≠ 附錄疊 `L'` 層」。
2. 曾以「coordinate channel lags one layer」描述差異 —— **該說法已棄用**，
   易講反。正式描述應為：§2.5 具 **intra-layer feature-to-coordinate coupling**
   （先更新 `h`，再於同層以 `h^(l+1)` 計算 coordinate gate）；
   附錄的 gate 使用由 **current `h^(l)`** 建立的 message。
3. `h^(l+1)` 是否為 typo：**UNKNOWN**，不得逕行修復。
4. 穩定性觀察為單一 seed、單一設定，不可用於解決 paper ambiguity。
