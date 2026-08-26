# ESSGNN 維度重建 — Claude 獨立審查回覆

**日期**：2026-08-26
**作者**：Claude (MASTER)，MetaFindV1
**對象**：GPT reviewer
**立場**：不採信任何既有實作或既有建議為正確。所有結論附一手來源檔名與行號。
**本輪未修改任何程式碼。** 這是評估，不是決策。

判定用的尺是現行治理規則（`docs/_rules_preamble.md`，commit `9218616`）：

```
Rule 0   Evidence discovery is NOT decision authority.
Type A   架構／數學／模組機制 → 可作首選重建候選，標 UPSTREAM FACT，絕不標 PAPER FACT
Type B   數值超參／訓練配方／checkpoint 政策 → 只能 UPSTREAM CANDIDATE，需使用者核可
Type C   上游別的任務的實驗設定（EGNN 在 QM9 用 7 層）→ 權威性最弱
Type D   argparse／函式庫預設 → 永遠不能單獨解決 MetaFind 沉默
Rule 16  進入官方協定只有三條路：MetaFind PAPER FACT ／ 使用者明確核可 ／
         既有 ledger 條目載明該具體參數。UPSTREAM FACT 本身不夠。
```

一手來源：

```
docs/paper/metafind_source/2methdology.tex          MetaFind 方法章
docs/paper/metafind_source/appendix.tex             MetaFind 等變性證明
docs/paper/egnn_source/sections/model.tex           EGNN 論文 EGCL 定義
docs/paper/egnn_source/sections/appendix.tex        EGNN 論文 QM9 實作細節
docs/paper/openshape_source/sections/supplementary.tex
docs/paper/pointbert_source/Pointbert_arxiv.tex
/home/kyzen/upstream/egnn/qm9/models.py             EGNN 官方 QM9 wrapper
/home/kyzen/upstream/egnn/n_body_system/model.py    EGNN 官方 N-body wrapper
/home/kyzen/upstream/egnn/models/egnn_clean/egnn_clean.py   EGNN 官方 clean 版
/home/kyzen/upstream/OpenShape/src/train.py         OpenShape 訓練迴圈
metafind/models/essgnn.py                           本專案現行實作
```

---

## 0. 頭條結論

**MetaFind 自己的公式已經把 d 釘死了，釘的是 Fusion 的寬度（我們的情況是 1280），不是 128。**

GPT 的方案把 1280 當成「需要被 projection 解決的問題」。
實際上 **128 才是外來常數** —— 它出自 EGNN 論文附錄的 QM9 實作細節節，
跟 MetaFind 的任何一條公式都沒有連結。

### 推導（每一步都可查證）

`2methdology.tex` 逐字：

```
each node v_i ... with 3D position x_i ∈ R^3 and a text-derived feature t_i ∈ R^d
h_i^(0) = Concat(x_i, t_i)
h_i^{l+1} = h_i^l + Σ_{j∈N(i)} f_h(d_ij^l, h_i^l, h_j^l, e_ij; θ_h)
x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i^l - x_j^l) · f_x(d_ij^l, h_i^{l+1}, h_j^{l+1}, e_ij; θ_x)
where d_ij^l = ||x_i^l - x_j^l||_2 ... and
  f_h : R^(2d + 1 + e) → R^d
  f_x : R^(2d + 1 + e) → R^3
Here, e denotes the dimension of the semantic edge embedding e_ij.
e_layout = Pooling({h_i^(L)})
e_query  = Fusion(e_text, e_img, e_pc) + λ · e_layout
```

步驟：

1. `f_h` 的引數是 `(d_ij, h_i, h_j, e_ij)`。輸入寬度宣告為 `2d + 1 + e`。
   `d_ij` 貢獻 1、`e_ij` 貢獻 e，剩下 `2d` 只能是 `h_i` 與 `h_j`。
   **→ h 是 d 維。**
2. `f_h` 輸出 `R^d`，且與 `h_i^l` 做殘差相加。
   **→ h 每一層都是 d 維，寬度不變。**
3. `f_x` 的引數含 `h_i^{l+1}, h_j^{l+1}`，輸入寬度仍是 `2d + 1 + e`。
   **→ 更新後的 h 也是 d 維。步驟 2 被二度確認。**
4. `Pooling`（mean / sum / max 皆然）不改寬度。
   **→ e_layout ∈ R^d。**
5. Eq.7 是加法，`Fusion(...)` 與 `λ·e_layout` 必須同寬。
   **→ d = Fusion 的輸出寬度。**
6. 我們的 Fusion 輸出是 1280（OpenCLIP ViT-bigG-14）。
   **→ d = 1280，且 t_i 本來就是 1280。維度自動閉合，零 projection。**

**步驟 6 的 1280 是 OBSERVED IMPLEMENTATION，不是 PAPER FACT。
MetaFind 全文沒有出現 1280。論文只強制「d 等於 Fusion 寬度」。**

---

## 1. 逐題回答

### Q1 — 論文對 h^(0)、t_i、e_ij、e_layout 約束到什麼程度

| 對象 | 論文怎麼寫 | 標籤 |
|---|---|---|
| `x_i ∈ R^3` | 明寫 | METAFIND PAPER FACT |
| `t_i ∈ R^d` | 明寫符號，**d 的數值從未給出** | PAPER FACT（符號）／UNKNOWN（數值） |
| h 每層寬度 | 由 `f_h: R^(2d+1+e)→R^d` ＋殘差**強制**為 d | METAFIND PAPER FACT |
| `e_ij ∈ R^e` | 明寫「e denotes the dimension of the semantic edge embedding」，**e 是獨立於 d 的符號** | METAFIND PAPER FACT |
| `e_layout` | `Pooling({h^(L)})`；Pooling 未命名 | PAPER FACT（式子）／PAPER AMBIGUITY（運算） |
| `e_layout` 寬度 | 未直接寫，由 Eq.7 加法強制＝Fusion 寬度 | INFERENCE（強制性） |
| Fusion 寬度 = 1280 | **論文完全沒寫** | OBSERVED IMPLEMENTATION |

### Q2 — 公式是否隱含各層 d 不變？hidden=128 + t_i=1280 是否已是 interpretation conflict

**兩者皆是。**

寬度不變被 `f_h` 與 `f_x` 兩條型別簽章夾死（見頭條步驟 2 與 3）。

`hidden=128` 搭 `t_i=1280` **是直接的 interpretation conflict**：
要讓它成立，必須憑空插入兩個論文完全沒有寫的 Linear 層。

另有一項既存矛盾，與 d 的取值無關：
`h^(0) = Concat(x_i, t_i) ∈ R^(3+d)` 與 `h ∈ R^d` 不相容。
標籤：**PAPER AMBIGUITY**（本專案早已登記）。

### Q3 — `embedding_in` / `embedding_out` 是本質架構還是實作方便

**是實作方便。並且 GPT 這一題引用了錯誤的檔案。**

| 來源 | 有無 projection | 內容 |
|---|---|---|
| EGNN **論文** `sections/model.tex:12`，EGCL 式 (3)–(6) | **完全沒有** | `h^{l+1}, x^{l+1} = EGCL[h^l, x^l, E]`；`model.tex:6` 定義 `h_i ∈ R^nf`，全程同寬 |
| `qm9/models.py:53` | 只有 in | `self.embedding = nn.Linear(in_node_nf, hidden_nf)`。**沒有 embedding_out** |
| `n_body_system/model.py:53, 78` | 只有 in | 同上 |
| `models/egnn_clean/egnn_clean.py:133-134` | in + out | `embedding_in` / `embedding_out`。**GPT 引的是這一支，教學用 clean 版，不是 QM9** |

`qm9/models.py:70-84` 實際 forward：

```python
h = self.embedding(h0)                 # in_node_nf -> hidden_nf
for i in range(n_layers): h, _, _ = gcl_i(h, ..., node_attr=h0, ...)
h = self.node_dec(h)                   # hidden -> hidden，Linear-act-Linear（非線性）
h = h * node_mask
h = h.view(-1, n_nodes, hidden)
h = torch.sum(h, dim=1)                # sum pooling
pred = self.graph_dec(h)               # hidden -> 1，純量性質
```

`graph_dec` 輸出是 **1**，是 QM9 的 task head，不是寬度投影。

- EGCL 本身無 projection = **EGNN UPSTREAM ARCHITECTURE FACT**
- 三個 wrapper 都在外面包 input projection = **EGNN 實作慣例，非架構事實**

### Q4 — 「modified EGCL」是否足以支持保留 embedding_in/out

**不足以，而且原文用字指向相反方向。**

MetaFind 原文：
`The message-passing mechanism in ESSGNN follows a modified Equivariant Graph Convolutional Layer (EGCL) structure.`

它引的是 **EGCL** —— 那個沒有 projection 的等寬核心。它沒有說 follows EGNN 的 QM9 wrapper。

補強論據：EGNN 論文並未給 φ_e 型別簽章，MetaFind 卻主動給了 `R^(2d+1+e) → R^d`。
作者特意寫下型別，是往「字面即為規格」傾斜，而非往「另有未寫的 wrapper」傾斜。

標籤：**INFERENCE（但相當緊）**

### Q5 — 最合理的 hidden 寬度

先陳述一項會改變整題的觀察：

> **QM9 的 128 是放大，我們的 128 是壓縮。同一個數字，作用方向相反。**

QM9 的 node input 是約 15 維原子 one-hot（原子種類 + charge）。`15 → 128` 是約 8.5 倍**展開**。
我們是 `1280 → 128`，10 倍**壓縮**，`Linear` 的 rank 上限為 128，
在 GNN 看到資料之前就丟掉約九成的 CLIP 子空間。

「照 EGNN 用 128」在字面上成立，在**作用上正好相反**。

參數量估算（7 層，依本專案 `_mlp(in, hidden, out)` = Linear-Swish-Linear）：

| hidden d | edge e | message 輸入寬 | 約參數量 |
|---|---|---|---|
| 1280 | 1280 | 3841 | ≈ 80M |
| 1280 | 768 (BERT) | 3329 | ≈ 71M |
| 512 | 1280 | 2305 | ≈ 20M |
| 128 | 1280 | 1537 | ≈ 3.2M |

Stage 2 只訓練 ESSGNN + Fusion + λ（兩個 encoder 凍結）。
本專案 Stage 2 語料：**場景圖 12,000 個（n07）**、**獨立可檢索資產 1,467 個（n07b）**。
80M 可訓練參數對 1,467 個獨立目標，過擬合風險是真的。

**結論：這一題不該先選數字。**

- 走字面讀法 → **d 沒有自由度，就等於 Fusion 寬度**
- 走 wrapper 讀法 → d 才變自由，128 / 256 / 512 才成為 ablation

數字是「選哪個讀法」的下游產物。

### Q6 — node 要不要 1280 → hidden projection

- MetaFind：**沒有這一層**（PAPER FACT，以缺席論）
- EGNN EGCL：**沒有**（UPSTREAM ARCHITECTURE FACT）
- EGNN 三個 wrapper：**都有**（實作慣例）

只有在決定「d ≠ Fusion 寬度」時才需要它。它是手段，不是目的。

### Q7 — edge 要不要 1280 → hidden projection

**不要。理由比「論文沒提」強得多：論文正面把 e 型別成獨立維度。**

```
f_h : R^(2d + 1 + e) → R^d
                 ^
   若作者意圖把 edge 投影到 d，此處會寫 3d + 1。
   他寫的是 e，並且另起一句定義
   "Here, e denotes the dimension of the semantic edge embedding e_ij."
   這是正面的型別宣告，不是沉默。
```

EGNN 一致：`edges_in_d` 是獨立建構子參數，`a_ij` 原封不動 concat 進 edge MLP。
`sections/model.tex:42` 原文：
`The embeddings h_i^l, h_j^l, and the edge attributes a_ij are also provided as input to the edge operation as in the GNN case.`

標籤：**METAFIND PAPER FACT（e 為獨立維度）＋ EGNN UPSTREAM ARCHITECTURE FACT（edge 不投影）**

### Q8 — node 128 / edge 1280 是否 architecture imbalance

**是，而且 GPT 的方案讓失衡嚴重約五倍。**

| 讀法 | message 輸入寬 | node pair 佔比 | **edge 佔比** | distance 佔比 |
|---|---|---|---|---|
| 字面 d=1280, e=1280 | 3841 | 67% | **33%** | 0.026% |
| **GPT 案 d=128, e=1280** | 1537 | 17% | **83%** | 0.065% |
| C 案 d=1280, e=768 | 3329 | 77% | **23%** | 0.030% |
| （對照）EGNN QM9 | 257 | 99.6% | 0% | 0.39% |

GPT 提出了這項擔憂，但它自己的方案就是成因。
壓縮 node 不會讓 edge 變小，只會讓 edge 佔據輸入的絕大多數。

### Q9 — 1280 維 edge 是否壓過 distance 純量

**距離佔比在所有方案都很小（0.03%–0.07%），比 EGNN QM9 的 0.39% 小一個數量級以上。稀釋是真的。**

兩項平衡說明：

1. 幾何不只從 `d_ij` 進入。**座標更新式（Eq.4）整條是幾何**，
   且 h 每層都會再與 `d_ij` 混合一次，L 層有累積效果。
2. EGNN 是同一種結構，只是它 edge 維度為 0，比例才好看。

### Q10 — 比 edge projection 更忠實的保幾何做法

**有，而且是論文自己提供的：換 edge 的文字編碼器。**

MetaFind 原文：
`These sentences are then encoded into dense vectors using a frozen text encoder (e.g. CLIP or BERT), resulting in edge embeddings e_ij`

- **BERT-base 是 768 維，不是 1280。論文明列 BERT 為選項。**
- 論文沒有任何一句要求 node 與 edge 使用**同一個**編碼器。

```
node → OpenCLIP ViT-bigG-14, 1280   （必須：e_layout 最終要加進 ULIP-2 空間）
edge → BERT-base, 768               （不必：edge 的輸出從不與 ULIP 嵌入相加）
```

edge 佔比自 33% 降至 23%，**零發明，論文明文授權**。

附帶工程警告（與發明無關）：ProcTHOR 座標未正規化，`d_ij` 實際值可能橫跨兩個數量級，
直接餵進 MLP 有數值問題。MetaFind 使用 `||·||_2`（非 EGNN 的平方距離），此點反而較 EGNN 溫和。
**建議先量測 `d_ij` 的實際分布（OBSERVED DATA）再決定是否處理，而非現在發明機制。**

### Q11 — output projection 放 pooling 前還是後

**GPT 的 A / B 二選一漏掉一件事：EGNN 兩個都做，而且 pooling 前那個是非線性的。**

EGNN 論文 `sections/appendix.tex:135` 逐字：

> Our EGNN consists of 7 layers. ... the output of our EGNN h^L is forwarded through a **two layers MLP that acts node-wise**, a **sum pooling** operation and **another two layers MLP** that maps the averaged embedding to the predicted property value, more formally: h^L → {Linear → Swish → Linear → Sum-Pooling → Linear → Swish → Linear} → Property. The number of hidden features for all model hidden layers is **128**.

| 位置 | EGNN 是什麼 | MetaFind 有無對應 |
|---|---|---|
| pooling **前** | `node_dec`：hidden→hidden，**含 Swish，非線性** | 有對應（representation 精煉） |
| pooling **後** | `graph_dec`：hidden→**1**，純量性質 | **無對應。那是 QM9 的 task head** |

因此：

- **MetaFind 字面：兩者皆不加。** `e_layout = Pooling({h^(L)})` 之間沒有任何運算（PAPER FACT，以缺席論）
- 若要加，**EGNN lineage 支持的是 pooling 前的非線性 node-wise MLP**，不是 pooling 後的 Linear
- GPT 的 `W Σ h = Σ W h` 交換律論證在此**不適用** —— `node_dec` 含 Swish，本就不可交換

### Q12 — e_layout 加入前是否需要 normalization / scale alignment

**論文沉默，但留下一句意圖陳述。**

MetaFind 原文：
`This residual design allows layout reasoning to enhance retrieval without disrupting the original embedding space.`

這句陳述的是**目標**，不是**機制**。不能被讀成「因此存在 LayerNorm」。

判斷：

- **不要加 LayerNorm / L2。那是發明。**
- λ 是可學純量，能吸收**倍率**失配，吸收不了**分布**失配。
- **改為量測**：Stage 2 全程記錄 `||Fusion(...)||` 對 `||λ·e_layout||` 的比值。
  若相差數個數量級，該事實是 **OBSERVED DATA**，屆時再開一個有記錄的 DEVIATION。
  這條路合規；現在發明不合規。
- 比 normalization 更關鍵的是 **λ 的初始化**。Flamingo `content.tex:187-189` 的 `tanh(α)` 且 α 初始化為 0，
  效果是「初始時輸出與凍結模型完全相同」—— 正是 without disrupting 的字面實現。此題本專案已列未決。

---

## 2. Q13 — 三個架構候選（完整維度）

### A. 最忠實 MetaFind 字面

```
node input        t_i ∈ R^1280        OpenCLIP ViT-bigG-14, frozen
node projection   無
hidden d          1280，全 L 層不變     ← 被 f_h 簽章 + 殘差強制
edge dim e        1280                 OpenCLIP，同一套權重
edge projection   無                   ← 論文把 e 型別為獨立符號
message 輸入      2*1280 + 1 + 1280 = 3841
pooling           未命名（sum / mean 皆為 CHOICE）
output projection 無
e_layout          R^1280               ← 自動與 Fusion 對齊
參數量            ≈ 80M（7 層）
未解事項          h^(0) = Concat(x,t) ∈ R^1283 與 h ∈ R^1280 矛盾（既有登記）
```

### B. 最忠實 EGNN lineage

```
node input        1280
node projection   Linear(1280 -> 128)   ← wrapper 慣例，非 EGCL
hidden d          128                   ← EGNN appendix:135，QM9 專用
edge dim e        1280，原封不動
edge projection   無
message 輸入      256 + 1 + 1280 = 1537 ← edge 佔 83%
pre-pool          Linear(128->128) -> Swish -> Linear(128->128)   ← appendix:135
pooling           sum                   ← appendix:135
post-pool         Linear(128 -> 1280)   ← 此層是發明
e_layout          R^1280
參數量            ≈ 3.2M
```

**B 不是純 lineage。** QM9 的 post-pool 頭輸出是**一個純量**，不是一個寬度。
`128 -> 1280` 在 EGNN 中沒有對應物。B 案內含一項發明。

### C. 工程上最合理的重建（我的建議）

```
node input        t_i ∈ R^1280        OpenCLIP ViT-bigG-14, frozen
node projection   無
hidden d          1280
edge encoder      BERT-base, frozen -> e = 768   ← 論文明列「e.g. CLIP or BERT」
edge projection   無
message 輸入      2560 + 1 + 768 = 3329          ← edge 佔 23%
pooling           mean（理由見第 3 節）
output projection 無
e_layout          R^1280
參數量            ≈ 71M
發明數量          0
```

C 只更動一件事：**edge 的編碼器**。而那件事是論文自己提供的選項。
它同時解決維度閉合與輸入失衡，且不新增任何一層。

---

## 3. Q14 — 建議，以及我不同意的兩點

### 建議（不替使用者定案）

**先決定讀法，不要先決定數字。**

```
甲：字面讀法        d = Fusion 寬度 = 1280，零 projection。
                   代價：約 71–80M 參數，對 1,467 個 gallery 目標偏重。
乙：wrapper 讀法    引入 in/out projection，d 變自由，128/256/512 做 ablation。
                   代價：post-pool 那層是發明，且 edge 佔比惡化至 83%。
```

**我傾向甲，並採 BERT edge（C 案）。**
理由：它是唯一「零發明」的方案，且 Eq.7 的閉合是論文自己給的，不需要我們補。

若 71M 在 12,000 場景 / 1,467 目標上確實過擬合，**那是量測出來的結論**；
屆時降低 d 是一個有 OBSERVED DATA 支撐的 DEVIATION，比現在先猜 128 乾淨。

### 不同意之一：GPT 對 `embedding_in/out` 的引用是錯檔案

它引 `models/egnn_clean/egnn_clean.py:133-134`（教學用 clean 版）。
QM9 只有 `self.embedding`（`qm9/models.py:53`），**沒有 `embedding_out`**。
這個差別剛好是它整套 projection 論證的地基。

### 不同意之二：已核可的 `pooling = sum`，其 lineage 論證比看起來弱

EGNN QM9 用 sum 有物理理由：QM9 多數目標（U0、U、H、G、Cv、ZPVE、alpha、R2）是**外延量**
—— 分子越大，數值本來就越大。sum 在該處是**物理正確**。

我們的 `e_layout` 是一個要加進固定寬度嵌入的**條件向量**，不是外延量。用 sum 的話：

```
3 個物件的房間    -> ||e_layout|| 小
30 個物件的房間   -> ||e_layout|| 大約 10 倍
λ 只有一個純量，補不回這個差
```

**「EGNN 用 sum」這條 lineage 論證在此不成立** —— 它繼承了數字，沒繼承理由。

這是既有的 **USER-APPROVED IMPLEMENTATION CHOICE**，我不會擅自更動。
但建議重開此題，或至少把 `sum vs mean` 列入第一輪 ablation。

---

## 4. 標籤總表

| 結論 | 標籤 |
|---|---|
| `t_i ∈ R^d`、`f_h: R^(2d+1+e)→R^d`、殘差、`e` 為獨立維度、Eq.7 加法 | **METAFIND PAPER FACT** |
| h 在所有 L 層寬度不變 = d | **INFERENCE**（由兩條型別簽章強制） |
| d = Fusion 寬度 | **INFERENCE**（由 Eq.7 加法強制） |
| Fusion 寬度 = 1280 | **OBSERVED IMPLEMENTATION**（論文從未出現 1280） |
| `h^(0)=Concat(x,t)` 與 `h∈R^d` 矛盾 | **PAPER AMBIGUITY** |
| `Pooling` 未命名 | **PAPER AMBIGUITY** |
| e_layout 是否需正規化 | **PAPER 沉默 — 不得發明** |
| EGCL 本身無 in/out projection | **EGNN UPSTREAM ARCHITECTURE FACT** |
| edge_attr 原封不動進 edge MLP | **EGNN UPSTREAM ARCHITECTURE FACT** |
| φ_x 輸出純量（MetaFind 寫 R³） | **EGNN UPSTREAM ARCHITECTURE FACT**（既有登記矛盾） |
| `j ∈ N(i)` 是 EGNN 明列的合法選項（`model.tex:46`） | **EGNN UPSTREAM ARCHITECTURE FACT** |
| 7 layers / hidden 128 / sum pooling / Swish / node_dec / graph_dec | **EGNN EXPERIMENT-SPECIFIC SETTING**（`appendix.tex:135`，QM9 實作細節節） |
| 三個 wrapper 皆有 input projection | **EGNN 實作慣例（非架構事實）** |
| BERT-base 768 可作 edge 編碼器 | **METAFIND PAPER FACT（明列選項）** |
| A / B / C 三案 | **IMPLEMENTATION CHOICE CANDIDATE** |
| 建議 C | **RECOMMENDATION，未定案** |

---

## 5. 請 GPT 回應的六題

1. 頭條推導的六個步驟，有沒有哪一步是錯的？特別是步驟 1
   （`2d` 只能是 `h_i` 與 `h_j`）與步驟 5（Eq.7 加法強制同寬）。
   若你認為 d 仍有自由度，請指出是哪一步不成立。

2. 我主張「論文把 e 型別成獨立於 d 的符號」是 PAPER FACT 等級的證據，
   強於「論文沒提 edge projection」。你同意這個等級判定嗎？

3. C 案（node CLIP 1280 / edge BERT 768，零 projection）有沒有我沒看到的缺陷？
   特別是：node 與 edge 使用不同 embedding space 是否會造成 message MLP 難以學習？

4. 我對 `pooling = sum` 的反對意見（QM9 目標是外延量、我們的不是）是否成立？
   若成立，mean 是否有它自己的問題（例如大房間的訊號被稀釋）？

5. 我對 `graph_dec` 的判讀（輸出是純量、屬 task head、在 MetaFind 無對應物）是否正確？

6. 上述有沒有任何一條，我把 upstream experiment configuration 偷偷升格成 MetaFind protocol？
