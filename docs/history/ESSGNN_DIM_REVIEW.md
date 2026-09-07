# ESSGNN 維度重建 — 審查回覆 v3

**作者**：Claude (MASTER)，MetaFindV1
**版本**：v3，2026-08-27。v1 = `cdf826d`，v2 = `961ffa6`，兩版皆保留可對照。
**本輪未修改任何程式碼。**

---

## v3 撤回紀錄

外部審查駁回 v2 的十八處。**全部查證成立，全部修正。**
其中兩項是實質新發現，不是措辭問題：

### 新發現一 — Candidate D 的 φ_h 寫錯了

我把 D 命名為「Appendix-consistent」，卻寫了 **原始 EGNN 的節點更新式**。

`docs/paper/metafind_source/appendix.tex` 逐字：

```
m_ij      = φ_e(h_i^l, h_j^l, ||x_i^l - x_j^l||², e_ij)
x_i^{l+1} = x_i^l + Σ_{j≠i} (x_i^l - x_j^l) · φ_x(m_ij)
h_i^{l+1} = h_i^l + Σ_{j≠i} φ_h(m_ij)
```

**φ_h 只吃 `m_ij`。** 殘差在外面，逐邊求和。
我寫的 `φ_h(h_i, m_i)`（先聚合再連同 h_i 一起吃）是 EGNN `model.tex:12` 式(6) 的形式。
**MetaFind 在這裡本來就改了 EGNN，而我把它改回去了。**

### 新發現二 — 我提議重開一個已經裁決的問題

v2 說「第一順位：送使用者裁決 D 還是 A」。**那本身違反 Rule 13。**

```
docs/audit/C_PAPER_CONTRADICTIONS.md:55
  Registered as U-26. DECIDED 2026-08-17: `appendix_shared_msg` is primary.

workflow/DECISION_LEDGER.md:564
  U-26 的 decided_by 是 "user + external review" —— 「是 USER 換句話說」
  （十個 RESOLVED 條目中，只有 U-20 沒有 USER）
```

**架構家族早就定案，而且有使用者參與。** 本輪推翻的是
`d=1280` 無條件、C 零發明、比例推因果、QM9-128 lineage —— **沒有一項觸及家族選擇**。
Rule 13：既有 ledger 已解決的，不得重開，除非新證據實質牴觸。**沒有牴觸。**

所以 D 不是「一個待選的第四讀法」，**D 就是現行決策**，我的工作是把它寫對，而我寫錯了。

### 其餘十六處

| # | v2 寫的 | 改成 |
|---|---|---|
| 1 | §0 末尾殘留「論文只強制 d = Fusion 寬度」、Q5「走字面讀法 d 沒有自由度」 | 全部補上「若 Pooling 保寬」條件 |
| 2 | Q1「h 每層寬度…**PAPER FACT**」（與標籤總表的 INFERENCE 打架） | **INFERENCE**，受 §2.5 維度一致性強制 |
| 3 | Q2「必須憑空插入**兩個** Linear」 | 只有 input mapping 是必要的；輸出端若 readout 不保寬，readout 自己就能負責 128→1280 |
| 4 | Q3 開頭仍寫「GPT 引用了錯誤的檔案」 | 刪除。`egnn_clean.py` 是有效的 UPSTREAM-OFFICIAL-IMPL 證據，錯的是權威等級 |
| 5 | Q4「原文用字指向相反方向」 | 降為：對 wrapper projection **既未正面支持、也未正面排除** |
| 6 | Q6「MetaFind 沒有這一層（**PAPER FACT，以缺席論**）」 | **PAPER SILENCE**。違反 Rule 2 的 evidence-of-absence 條款 |
| 7 | Q7 標題「edge 要不要投影 —— **不要**」 | 改為 candidate，不是結論 |
| 8 | Q8「GPT 的方案讓失衡嚴重約**五倍**」 | 刪除。表改名為「第一層輸入維度／weight-column 配置」，因果結論全部移除 |
| 9 | Q9「稀釋**是真的**」 | 刪除。改為未量測。且 0.065%→0.39% 約 6 倍，不宜稱「一個數量級」 |
| 10 | Q10 標題稱 BERT 是「保幾何做法」 | 改名。保幾何的效果依賴已撤回的 Q8 論證 |
| 11 | Q11 兩處「以缺席論 PAPER FACT」；表格說 node_dec「MetaFind 有對應」 | 改 PAPER SILENCE；改「無明確對應」。`node_dec` 降為 QM9 experiment-specific precedent |
| 12 | Candidate A 未標 Pooling 保寬前提 | 補上 |
| 13 | Candidate B 稱「最忠實 EGNN lineage」 | 改名 **QM9-wrapper-inspired** |
| 14 | Candidate C 稱「工程上最合理（我的建議）」「發明數量 0」 | 兩個欄位直接刪除，改名 **Heterogeneous-encoder candidate** |
| 15 | Candidate D 的 φ_h | 見新發現一 |
| 16 | Candidate D「hidden d 由兩個 OPEN 決定」太模糊 | 拆成 D_h / D_e / D_m / D_layout / D_fusion 五個獨立寬度 |
| 17 | 裁決順序第一項是「內部讀法」 | 見新發現二。改為 Step 0 先查既有決策 |
| 18 | 順序漏了 `use_io_projections` 與 edge 維度契約 | 補入 |

**v2 未動搖、v3 保留的唯一核心**：128 不是 MetaFind PAPER FACT，
「QM9 用 128 → MetaFind 用 128」的 lineage 不成立。

---

## 1. 論文的維度代數：什麼被強制、什麼沒有

### 無條件成立

| 陳述 | 依據 | 標籤 |
|---|---|---|
| `x_i ∈ R^3` | `2methdology.tex` 明寫 | **METAFIND PAPER FACT** |
| `t_i ∈ R^d`（符號，**數值從未給**） | 同上 | **PAPER FACT**（符號）／**UNKNOWN**（數值） |
| `e` 是與 `d` 分開定義的邊寬度符號 | `f_h : R^(2d+1+e) → R^d`＋「e denotes the dimension of the semantic edge embedding」 | **METAFIND PAPER FACT** |
| `dim(e_layout) = dim(Fusion)` | Eq.7 是加法 | **INFERENCE（無條件）** |
| §2.5 的 h 在 l ≥ 1 各層等寬 | `f_h` 與 `f_x` 兩條型別簽章＋殘差 | **INFERENCE**，受維度一致性強制 |

### 有條件、或根本不成立

| 陳述 | 為什麼 | 標籤 |
|---|---|---|
| `d = dim(Fusion)` | **只有在 Pooling 保寬時**。`Pooling` 未命名，一個帶投影的 attention/set readout 同樣符合字面而不保寬 | **INFERENCE（條件式）** |
| `Pooling` 是什麼運算 | 論文只寫 `Pooling({h_i^(L)})` | **PAPER AMBIGUITY** |
| `h^(0)` 到底是什麼 | §2.5 寫 `Concat(x_i,t_i) ∈ R^(d+3)`，與 `f_h` 期望的 `R^d` 不相容；Appendix 又假設 `h^0` 對 x 的變換不變，與把原始座標塞進 `h^0` 衝突 | **PAPER 內部矛盾**（Rule 8） |
| 論文沒寫 node input projection | 缺席 | **PAPER SILENCE**（不是「證據顯示不存在」） |
| 論文沒寫 edge projection | 缺席。型別簽章只證明「不要求 e = d」 | **PAPER SILENCE** |
| 論文沒寫 e_layout 的正規化 | 缺席 | **PAPER SILENCE** |
| 論文沒寫 pre-pool / post-pool MLP | 缺席 | **PAPER SILENCE** |

**Rule 2：缺席不得轉成「證據顯示不存在」。以上四條 SILENCE 全部只能導出
IMPLEMENTATION CHOICE CANDIDATE，不能導出 PAPER FACT。**

---

## 2. 架構家族：已裁決，不重開

```
U-26   DECIDED 2026-08-17   appendix_shared_msg is primary
       decided_by: user + external review
       C_PAPER_CONTRADICTIONS.md:55 · DECISION_LEDGER.md:564
       程式碼已實作：essgnn.py:107,169
```

### 現行決策的正確形式（v3 修正）

```
m_ij      = φ_e(h_i^l, h_j^l, ||x_i^l - x_j^l||², e_ij)
x_i^{l+1} = x_i^l + Σ_{j≠i} (x_i^l - x_j^l) · φ_x(m_ij)
h_i^{l+1} = h_i^l + Σ_{j≠i} φ_h(m_ij)
```

三個必須記住的細節，**每一個都是 MetaFind 對 EGNN 的改動**：

| 項目 | EGNN `model.tex:12` | MetaFind `appendix.tex` |
|---|---|---|
| 節點更新 | `h_i^{l+1} = φ_h(h_i^l, m_i)`，先聚合再一起吃 | `h_i^{l+1} = h_i^l + Σ_j φ_h(m_ij)`，逐邊吃、殘差在外 |
| φ_x 輸入 | `φ_x(m_ij)` | 同（Appendix 相同；**§2.5 不同**，吃原始 tuple） |
| 求和範圍 | `j ≠ i`（並註明可改 `j ∈ N(i)`） | Appendix `j ≠ i`；**§2.5 是 `j ∈ N(i)`** |
| 距離 | `‖·‖²` 平方 | Appendix `‖·‖²`；**§2.5 是 `‖·‖_2` 不平方** |

§2.5 與 Appendix 在後三列全部不一致 —— 這是 U-26 之外另外登記的矛盾群。

### 非現行家族（保留供對照，不是待選項）

- **§2.5 message-function family**（`sec25_two_mlp`）：`f_h` 與 `f_x` 各吃原始 tuple。
  採用它還要另外裁決 Concat 矛盾、h⁰ 不變性矛盾、f_x 值域矛盾、距離平方矛盾。
- **QM9-wrapper-inspired**（v2 叫「最忠實 EGNN lineage」，**已改名**）：
  input projection ＋ hidden 128 ＋ `node_dec` ＋ sum ＋ 發明的 `128→1280`。
  其中 128、sum、`node_dec` 全部是 QM9 experiment-specific，
  `graph_dec` 輸出是純量、屬 task head、MetaFind 無對應物。**provenance 上離 MetaFind 最遠。**

---

## 3. 真正還開著的：維度契約

家族定了，維度沒定。Appendix 的約束比 §2.5 **少**，所以自由度反而更多。

**不要再用單一個 `d`。至少五個獨立寬度：**

```
D_h        節點狀態寬度        h_i^l
D_e        邊嵌入寬度          e_ij
D_m        訊息寬度            m_ij = φ_e(...)
D_layout   佈局輸出寬度        e_layout = Pooling({h^(L)})
D_fusion   Fusion 輸出寬度     我們的實作是 1280（OBSERVED IMPLEMENTATION，論文沒寫）
```

Appendix 明確強制的只有兩條：

```
dim(φ_h 的輸出) = D_h        因為要與 h_i^l 殘差相加
D_layout = D_fusion          因為 Eq.7 是加法
```

**Appendix 沒有強制的**：`D_m = D_h`、`D_e = D_h`、`D_h = D_layout`。
最後一條要成立，需要「Pooling 保寬」這個未經論文支持的假設。

### 待決事項（皆為 IMPLEMENTATION CHOICE CANDIDATE，需使用者核可）

| 項目 | 內容 | 現況 |
|---|---|---|
| `use_io_projections` | node 進 ESSGNN 前、e_layout 出去前，要不要 adapter | 程式碼中是**必填無預設**旗標（`essgnn.py:180`） |
| Pooling 運算 | sum / mean / max / attention / set readout —— 「不保寬」不是一種架構，要指名 | `sum` 有既有決策，重開與否須先查 ledger |
| `layer_sharing` | independent / shared，參數量差約 L 倍 | `essgnn.py:200` 預設 independent |
| edge 維度契約 | 編碼器是誰、`D_e` 多少、要不要 edge adapter | 全開 |
| `D_h` 數值 | 128 / 256 / 512 / 1280 / 其他 | **前四項定了才有合法上游** |

---

## 4. 那張輸入維度表（因果結論已移除）

**表名：第一層輸入維度／weight-column 配置。**

| 組態 | 輸入寬 | node pair | edge | distance |
|---|---|---|---|---|
| D_h=1280, D_e=1280 | 3841 | 67% | 33% | 0.026% |
| D_h=128, D_e=1280 | 1537 | 17% | 83% | 0.065% |
| D_h=1280, D_e=768 | 3329 | 77% | 23% | 0.030% |
| （對照）EGNN QM9 | 257 | 99.6% | 0% | 0.39% |

**這張表只能導出一句話**：

> `edge_attr` 在該組態下佔第一層輸入維度的 N%，意即第一層為 edge block 配置較多 weight columns。

**不能導出**：edge 壓過幾何、幾何被稀釋、失衡若干倍。
第一層是 `y = W_h h_i + W_h' h_j + w_d d_ij + W_e e_ij`；
一個純量配大 `w_d` 可以主導，1280 維配小 `W_e` 可以被壓成零。

要判定實際貢獻必須量測 block-wise `‖W_h h_i‖ / ‖W_h' h_j‖ / ‖w_d d_ij‖ / ‖W_e e_ij‖`
與 block-wise 梯度範數。**在量到之前，任何 domination 說法都是 HYPOTHESIS。**

---

## 5. 參數量（限定條件已標明）

v2 這張表未標前提，被誤讀成 ESSGNN 的通用成本。**限定 `sec25_two_mlp` ＋ `independent` ＋ 7 層**：

| D_h | D_e | 訊息輸入寬 | 約參數量 |
|---|---|---|---|
| 1280 | 1280 | 3841 | ≈ 80M |
| 1280 | 768 | 3329 | ≈ 71M |
| 512 | 1280 | 2305 | ≈ 20M |
| 128 | 1280 | 1537 | ≈ 3.2M |

換成 `appendix_shared_msg`（現行家族）或 `layer_sharing=shared`，同寬度的成本大幅改變 ——
共享單層在 1280/1280 下約 11–13M 而非 80M。**四種組合最多差約 7 倍。**

因此「71–80M 對 1,467 個目標過重」在架構家族與層共享定案前**不是固定成本**，
不能拿來當 `D_h` 的工程理由。

（Stage 2 只訓 ESSGNN + Fusion + λ；語料為場景圖 12,000（n07）、獨立可檢索資產 1,467（n07b）。）

---

## 6. 上游證據（本輪未動搖的部分）

### EGNN 的 in/out projection 屬什麼等級

| 來源 | 有無 | 內容 |
|---|---|---|
| EGNN 論文 `model.tex:12` 式(3)–(6) | 無 | `h^{l+1}, x^{l+1} = EGCL[h^l, x^l, E]`；`model.tex:6` 定義 `h_i ∈ R^nf`，全程同寬 |
| `qm9/models.py:53` | 只有 in | `self.embedding`。**沒有 `embedding_out`** |
| `n_body_system/model.py:53,78` | 只有 in | 同上 |
| `models/egnn_clean/egnn_clean.py:133-134` | in + out | `embedding_in` / `embedding_out` |

`egnn_clean.py` **是官方 repo 的一部分，是有效的 UPSTREAM-OFFICIAL-IMPL 證據** ——
它證明官方實作曾使用 in/out projection。

它**不能**證明 EGCL intrinsic architecture 有 projection（論文式(3)–(6)沒有），
也**不能**證明 QM9 用了 `embedding_out`（`qm9/models.py:53` 只有 in）。

正確等級：**EGNN 實作慣例**，不是 EGNN UPSTREAM ARCHITECTURE FACT。

### QM9 的 readout

EGNN 論文 `appendix.tex:135` 逐字：

> Our EGNN consists of 7 layers. ... the output of our EGNN h^L is forwarded through a **two layers MLP that acts node-wise**, a **sum pooling** operation and **another two layers MLP** that maps the averaged embedding to the predicted property value ... The number of hidden features for all model hidden layers is **128**.

| EGNN QM9 元件 | MetaFind |
|---|---|
| `node_dec`（hidden→hidden，含 Swish） | **無明確對應** |
| `sum pooling` | MetaFind 只寫 `Pooling`，未命名 |
| `graph_dec`（hidden→1，純量） | **無明確對應，task-specific** |

`node_dec` 出現在 **QM9 implementation details 節**，不是 EGCL intrinsic mechanics。
標籤：**EGNN EXPERIMENT-SPECIFIC PRECEDENT**，不是「EGNN lineage 支持」。

同節的 `7 layers` 與 `hidden 128` 亦然。

### MetaFind 對文字編碼器說了什麼

原文：`encoded into dense vectors using a frozen text encoder (e.g. CLIP or BERT)`

| 主張 | 標籤 |
|---|---|
| 「BERT」是論文明列的編碼器例子 | **METAFIND PAPER FACT** |
| BERT-base 而非 large、768 維、某種句向量抽法（[CLS]／mean-token／pooler）、是否正規化、最大長度 | **IMPLEMENTATION CHOICE**（論文一項都沒指定） |
| node 與 edge 使用不同編碼器 | **IMPLEMENTATION CHOICE**（論文未禁止 ≠ 論文授權） |

### e_layout 的正規化

MetaFind 原文：
`This residual design allows layout reasoning to enhance retrieval without disrupting the original embedding space.`

這是**目標陳述**，不是機制。分兩層看：

- **不得無聲採用某個正規化然後宣稱忠實復現** —— 成立且合規（Rule 2 / Rule 16）
- **「所以不該用正規化」** —— 太強。正確狀態是 PAPER SILENCE → 上游查找 → 仍未解 →
  IMPLEMENTATION CHOICE → **等使用者決定**

且必須分清兩件事（`metafind/models/losses.py:166-167` 已 `F.normalize` 兩側）：

```
(1) e_layout 這條分支自己正規化     ← 論文沉默，未決
(2) Fusion + λ·e_layout 之後的最終查詢向量正規化   ← 程式碼已經在做
```

「目前不新增分支層級正規化」可當最小發明 baseline，
但標 **candidate / pending**，不可標成論文規定的行為。

比正規化更關鍵的是 **λ 的初始化**：Flamingo `content.tex:187-189` 的 `tanh(α)`、α 初始 0，
效果正是「初始輸出與凍結模型完全相同」= without disrupting 的字面實現。已列未決。

---

## 7. 裁決順序（v3 改寫）

```
Step 0   先查既有決策                                    ← v2 漏掉，導致差點違反 Rule 13
         架構家族 = U-26，2026-08-17，USER-backed，有效 → appendix_shared_msg，不重開
         Pooling = sum 亦有既有決策 → 重開與否須先查 ledger 與新證據，不得默默改 mean

Step 1   維度介面契約  use_io_projections
         node 進 ESSGNN 前、e_layout 出去前要不要 adapter
         這一項決定 D_h 能不能與 node/Fusion 寬度脫鉤，是所有寬度問題的前提

Step 2   Pooling / readout 運算子
         要指名 sum / mean / max / attention / set readout
         「不保寬」不是一種架構

Step 3   layer_sharing  independent / shared        ← 與 Step 2 可並行
         若寬度要用參數預算當理由，此項須先定，它把參數量改掉約 L 倍

Step 4   edge 維度契約  編碼器是誰 / D_e 多少 / 要不要 edge adapter
         它直接改 message MLP 的參數量，同樣須排在寬度之前

Step 5   D_h 數值
         前四步定了才有合法上游。128 目前只是 QM9 experiment-specific candidate
```

**在 Step 1–4 定案前，`D_h` 不動，任何候選都不寫進協定。**

## 8. 我不再提出的

v2 推薦 Candidate C，靠三根支柱：零發明、幾何不易被壓、71M 過重。
**三根倒了三根。推薦整個撤回。**

本輪唯一未動搖的結論：
**128 不是 MetaFind PAPER FACT，「QM9 128 → MetaFind 128」的 lineage 不成立。**
