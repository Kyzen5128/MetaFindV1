# ESSGNN 維度重建 — Claude 獨立審查回覆

**日期**：2026-08-26
**作者**：Claude (MASTER)，MetaFindV1
**對象**：GPT reviewer
**立場**：不採信任何既有實作或既有建議為正確。所有結論附一手來源檔名與行號。
**本輪未修改任何程式碼。** 這是評估，不是決策。

---

## v2 撤回紀錄（2026-08-27，外部審查後）

外部審查（GPT）逐條駁回下列九處。**我查證後全部確認成立，以下全部撤回或降級。**
原文保留在 commit `cdf826d`，本版是修正版。

| # | 我原本寫的 | 為什麼錯 | 現在改成 |
|---|---|---|---|
| 1 | 「論文已把 d 釘死」 | Step 4 偷加了「Pooling 保寬」的假設。`Pooling` 未命名，論文沒有排除非保寬的 readout | **條件式**：若 Pooling 是保寬聚合，則 d = Fusion 寬度 |
| 2 | Candidate A =「最忠實 MetaFind 字面」 | A 其實混了兩個互相矛盾的 formulation：用 §2.5 的型別簽章，卻用 Appendix 的 `h^0=t_i` 解掉 Concat 矛盾。真正的 §2.5 字面是 **dimensional-invalid** | A 重新命名，並補上第四讀法 **D（Appendix-consistent）** |
| 3 | 漏了第四讀法 | 而且 **本專案自己的程式碼早就有它，還標成 primary**（`essgnn.py:107,169` `appendix_shared_msg <- primary`）。我沒讀自己的程式碼就寫了候選清單 | 補列 D |
| 4 | 「BERT-base 768 = METAFIND PAPER FACT」 | 論文只寫 `e.g. CLIP or BERT`。base/large、768、[CLS] 還是 mean-token、要不要正規化，一個都沒指定 | 「BERT 是論文明列的例子」=PAPER FACT；「BERT-base 768」=IMPLEMENTATION CHOICE |
| 5 | 「edge 不投影 = PAPER FACT」 | 型別簽章只證明「e 不被要求等於 d」。它沒有排除文字編碼器輸出到 `e_ij` 之間存在 adapter。這是把「沒有證據」轉成「證據顯示沒有」 | 降為 INFERENCE（極強）＋ literal-reading candidate |
| 6 | Candidate C「發明數量 0」 | C 自己選了 mean pooling（論文未命名）、選了 BERT-base、選了 node/edge 異質編碼器。三項都是 paper-silent choice | 改成「不新增 I/O projection 層，但含多項 paper-silent 選擇」 |
| 7 | 用維度佔比證明 edge 壓過 geometry（83% vs 0.065%） | 算術對，推論錯。**輸入維度佔比 ≠ 神經貢獻佔比**。一個純量配大權重可以主導；1280 維配小權重可以被壓成零 | 降為 hypothesis，需量測 block-wise ‖W·x‖ 或梯度範數才能成立 |
| 8 | 「QM9 用 sum 是因為目標是外延量，所以物理正確」＋「30 個物件 ‖e_layout‖ 大 10 倍」 | EGNN 從未說過這個理由；QM9 目標也含 HOMO/LUMO/gap/μ 等非外延量。而且隨機方向下 ‖Σh‖ 更接近 √N 而非 N | 降為 INFERENCE。**對 sum 的核心質疑（不能靠 lineage 就決定）仍然成立** |
| 9 | 「GPT 引錯檔案」 | `egnn_clean.py` 是官方 repo 的一部分，能證明官方實作曾用 in/out projection。真正的問題是**權威等級**判定，不是檔案不相關 | 改成：它證明的是實作慣例，不是 EGCL intrinsic architecture |

**唯一被確認正確且未動搖的核心**：128 不是 MetaFind PAPER FACT，
且「QM9 128 → MetaFind 128」的 lineage 不成立。這一條外部審查同意。

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

**（v2 修正）若 `Pooling` 是保寬聚合，則 MetaFind 的公式把 d 綁在 Fusion 的寬度上（我們的情況是 1280）。
這是一個條件結論，不是無條件的 PAPER FACT —— `Pooling` 未命名，論文沒有排除非保寬的 readout。**

無論條件成立與否，**128 都不是 MetaFind PAPER FACT** —— 它出自 EGNN 論文附錄的 QM9 實作細節節，
跟 MetaFind 的任何一條公式都沒有連結。這一條外部審查同意。

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
4. **（v2：這一步是假設，不是論文強制）** 若 `Pooling` 是保寬聚合（mean / sum / max 皆然），
   則 e_layout ∈ R^d。**論文只寫 `Pooling(...)`，未命名。** 一個帶投影的 attention readout
   或 set-transformer readout 同樣符合這個字面，而它們不保寬。
   **→ 條件式：e_layout ∈ R^d。**
5. Eq.7 是加法，`Fusion(...)` 與 `λ·e_layout` 必須同寬。
   **→ dim(e_layout) = dim(Fusion)。這一步無條件成立。**
6. 我們的 Fusion 輸出是 1280（OpenCLIP ViT-bigG-14）。
   **→ 搭配步驟 4 的假設才得到 d = 1280。** 若 Pooling 不保寬，d 與 Fusion 寬度脫鉤，
   步驟 1–3 的等寬性仍成立，但 d 的**數值**回到自由。

**（v2）步驟 1–3 還有一個層 0 的問題**：它們描述的是 l ≥ 1 的遞迴，在該範圍成立。
但 `h^(0) = Concat(x_i, t_i) ∈ R^(d+3)` 與 `h ∈ R^d` 不相容，
且 Appendix 又假設 `h^0` 對 x 的 SE(3) 變換不變 —— 把原始座標 concat 進 `h^0` 與此衝突。
**這是論文內部矛盾，依 Rule 8 不得默默選一邊。**

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
| `d` 的數值 | **只有在 Pooling 保寬時**才被綁到 Fusion 寬度 | **INFERENCE（條件式，v2 降級）** |
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

參數量估算。**（v2 重大修正）原本這張表只算了一種架構，卻寫得像 ESSGNN 的通用成本。**
本專案 `essgnn.py:107` 有兩個架構家族、`:200` 有 independent / shared 兩種層共享方式，
四種組合的參數量最多相差 **約 7 倍**。下表限定
**`sec25_two_mlp` ＋ `layer_sharing=independent` ＋ 7 層**：

| hidden d | edge e | message 輸入寬 | 約參數量 |
|---|---|---|---|
| 1280 | 1280 | 3841 | ≈ 80M |
| 1280 | 768 (BERT) | 3329 | ≈ 71M |
| 512 | 1280 | 2305 | ≈ 20M |
| 128 | 1280 | 1537 | ≈ 3.2M |

換成 `appendix_shared_msg` 或 `layer_sharing=shared`，同一組寬度的成本會大幅改變
（共享單層在 1280/1280 下約 11–13M，而非 80M）。
**所以「71–80M 對 1,467 個目標過重」這個工程論證，在架構家族與層共享方式定案前不是固定成本。**

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

**（v2 修正）標籤拆開，原本的合併標籤太強：**

| 主張 | 標籤 |
|---|---|
| `e` 是與 `d` 分開定義的符號 | **METAFIND PAPER FACT** |
| 公式不要求 `e = d` | **INFERENCE（極強）** |
| 論文沒寫 edge projection | **PAPER SILENCE** |
| 因此 V1 不加 edge projection | **IMPLEMENTATION CHOICE CANDIDATE（literal-reading）** |

**不可寫成「edge 不投影 = PAPER FACT」。** 型別簽章只描述「進入 message function 時的 edge 寬度」，
沒有排除文字編碼器輸出到 `e_ij` 之間存在 adapter。把 PAPER SILENCE 轉成
「證據顯示不存在」違反 Rule 2 的 evidence-of-absence 條款。

### Q8 — node 128 / edge 1280 是否 architecture imbalance

**是，而且 GPT 的方案讓失衡嚴重約五倍。**

| 讀法 | message 輸入寬 | node pair 佔比 | **edge 佔比** | distance 佔比 |
|---|---|---|---|---|
| 字面 d=1280, e=1280 | 3841 | 67% | **33%** | 0.026% |
| **GPT 案 d=128, e=1280** | 1537 | 17% | **83%** | 0.065% |
| C 案 d=1280, e=768 | 3329 | 77% | **23%** | 0.030% |
| （對照）EGNN QM9 | 257 | 99.6% | 0% | 0.39% |

這張表顯示的是**第一層線性的參數欄位佔比**。壓縮 node 不會讓 edge 變小。

**（v2 重大修正）但「佔比大」不等於「主導」。** 第一層是
`y = W_h h_i + W_h' h_j + w_d d_ij + W_e e_ij`。
一個純量配上很大的 `w_d` 可以有巨大影響；1280 個 edge 維度也可以被 `W_e` 幾乎壓成零。

所以本表能證明的是 **edge block 佔第一層較多參數容量**，
**不能證明** edge 在訓練/推論中實際壓過幾何。
要證明後者必須量測 block-wise `‖W_h h_i‖ / ‖w_d d_ij‖ / ‖W_e e_ij‖`
或 block-wise 梯度範數。**在量到之前，這只是 hypothesis。**

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

edge 的參數欄位佔比自 33% 降至 23%。

**（v2 修正）「零發明」是錯的，已撤回。** 論文只寫 `e.g. CLIP or BERT` ——
它沒有指定 base 還是 large、沒有指定 768、沒有指定句向量怎麼抽（`[CLS]` / mean-token / pooler）、
沒有指定要不要正規化、也沒有「授權」node 與 edge 使用不同編碼器（它只是沒有禁止）。

正確標籤：
- 「BERT 是論文明列的例子」= **METAFIND PAPER FACT**
- 「BERT-base、768 維、某種句向量抽法」= **IMPLEMENTATION CHOICE**
- 「node 與 edge 用不同編碼器」= **IMPLEMENTATION CHOICE**（論文未禁止 ≠ 論文授權）

而且「因此幾何更不容易被壓過」依賴 Q8 那個已被降級的佔比論證，同樣只是 hypothesis。

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

- **（v2 拆成兩層，原本的單句太強）**
  - 作為「不得無聲採用」：**成立且合規**。論文沒寫分支正規化，就不能偷塞一個然後宣稱忠實復現。
  - 作為「科學結論：所以不該用正規化」：**太強，已撤回**。正確狀態是
    PAPER 沉默 → 上游查找 → 仍未解 → IMPLEMENTATION CHOICE → **等使用者決定**。
- **還要分清楚兩件不同的事**（`metafind/models/losses.py:166-167` 已 `F.normalize(query)` 與
  `F.normalize(gallery)`）：
  - (1) `e_layout` 這條分支自己正規化 —— 論文沉默，未決
  - (2) `Fusion + λ·e_layout` 之後的最終查詢向量正規化 —— **程式碼已經在做**
  這兩件不是同一件事。目前「不新增分支層級正規化」可以當最小發明 baseline，
  但要標 **candidate / pending decision**，不可標成論文規定的行為。
- λ 是可學純量，能吸收**倍率**失配，吸收不了**分布**失配。
- **改為量測**：Stage 2 全程記錄 `||Fusion(...)||` 對 `||λ·e_layout||` 的比值。
  若相差數個數量級，該事實是 **OBSERVED DATA**，屆時再開一個有記錄的 DEVIATION。
  這條路合規；現在發明不合規。
- 比 normalization 更關鍵的是 **λ 的初始化**。Flamingo `content.tex:187-189` 的 `tanh(α)` 且 α 初始化為 0，
  效果是「初始時輸出與凍結模型完全相同」—— 正是 without disrupting 的字面實現。此題本專案已列未決。

---

## 2. Q13 — 三個架構候選（完整維度）

### A. §2.5 型別簽章 ＋ Appendix 的 h⁰（**v2 改名**）

**原本叫「最忠實 MetaFind 字面」，那是錯的。**
真正的 §2.5 字面（`h^(0)=Concat(x_i,t_i)` ∈ R^(d+3)，而 `f_h` 期望每個 h 都是 d 維）
**在維度上不成立**。A 是先用 Appendix 的 `h^0 = t_i` 解掉那個矛盾，再套 §2.5 的型別簽章 ——
也就是說 **A 跨了兩個互相矛盾的 formulation**，它不是單一讀法。

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

C 只更動一件事：**edge 的編碼器**。
**（v2）但 C 不是「零發明」** —— 見 Q10 的修正。它不新增 I/O projection 層，
但含 mean pooling、BERT-base 768、node/edge 異質編碼器等多項 paper-silent 選擇。

### D. Appendix-consistent / proof-consistent 讀法（**v2 新增，外部審查提出**）

```
x_i               獨立的座標串流，不進 h
h_i^(0)           = t_i
message           phi_e(h_i, h_j, distance, e_ij) -> m_ij
coordinate head   phi_x(m_ij)
feature head      phi_h(h_i, m_i)
Pooling           保持 OPEN
edge encoder / e  保持 OPEN
hidden d          由上述兩個 OPEN 決定，不預先綁定
```

**D 不是 A。** A 用 §2.5 的兩個獨立 MLP（`f_h` 與 `f_x` 各自吃原始 tuple），
D 用 Appendix 的兩階段結構（先算 `m_ij`，再分給 `phi_x` 與 `phi_h`）。

**而且 D 早就在本專案的程式碼裡，還被標成 primary：**

```
essgnn.py:107  ArchFamily = Literal["appendix_shared_msg", "sec25_two_mlp"]
essgnn.py:169  appendix_shared_msg  phi_e -> m_ij -> {phi_x, phi_h}   <- primary
essgnn.py:170  sec25_two_mlp        f_h and f_x, each on the raw tuple
```

**我寫候選清單時沒有讀自己的程式碼。** 這是本輪我最該記下的一筆：
外部審查靠讀 repo 找到了一個讀法，而那個讀法是我們自己的預設。

---

## 3. Q14 — 建議，以及我不同意的兩點

### 建議（不替使用者定案）

**（v2 改寫）原本的甲／乙二選一問錯了問題，已撤回。**

真正待決的不是 hidden 寬度，而是 **MetaFind 內部要採哪一個讀法** ——
因為 §2.5 的字面在維度上不成立，Appendix 又給了另一套彼此不相容的假設。
依 Rule 8，論文內部矛盾不得默默選一邊。讀法定了，維度才有合法的上游。

```
讀法 D  Appendix-consistent   h^0 = t_i，兩階段 phi_e -> {phi_x, phi_h}，座標獨立串流
                              本專案程式碼的 primary（essgnn.py:169）
讀法 A  §2.5 型別簽章         f_h / f_x 各吃原始 tuple，但 h^0 仍須向 Appendix 借
```

讀法定案後，才輪到這兩項（各自獨立，不可綁在一起問）：

```
Pooling 是否保寬   決定 d 是否被綁到 Fusion 寬度
層共享方式         independent / shared，參數量差約 7 倍
```

**我不再推薦特定寬度。** 上一版推薦 C 的三根支柱裡，
「零發明」已撤回、「幾何不易被壓過」降為未量測的 hypothesis、
「71M 過重」在架構家族未定前不是固定成本。三根倒了三根。

唯一站得住的仍然是：**128 不是 MetaFind PAPER FACT，
且「QM9 128 → MetaFind 128」的 lineage 不成立。**

### 不同意之一：`embedding_in/out` 的**權威等級**被抬高了（v2 改寫）

**原本寫「GPT 引錯檔案」，那句太強，已撤回。**
`models/egnn_clean/egnn_clean.py:133-134` 是官方 repo 的一部分，
它確實證明**官方 EGNN 實作曾使用 in/out projection**。這是有效證據。

真正的問題是它被當成什麼等級的證據。它**不能**證明：

- EGCL intrinsic architecture 必須有 projection —— 論文 `model.tex:12` 的式 (3)–(6) 沒有
- QM9 用了 `embedding_out` —— `qm9/models.py:53` 只有 `self.embedding`，沒有 out

所以正確分類是 **EGNN 實作慣例**，不是 **EGNN UPSTREAM ARCHITECTURE FACT**。

### 不同意之二：已核可的 `pooling = sum`，其 lineage 論證比看起來弱

**（v2 降級）** 我原本寫「EGNN 用 sum 是因為 QM9 目標是外延量，所以物理正確」。
**EGNN 從未說過這個理由**，而且 QM9 目標也包含 HOMO、LUMO、gap、偶極矩 μ 等**非外延量**。
所以那是我的 **INFERENCE**，不是 EGNN upstream fact，已降級。

我們的 `e_layout` 是一個要加進固定寬度嵌入的**條件向量**，不是外延量。用 sum 的話：

```
3 個物件的房間    -> ||e_layout|| 較小
30 個物件的房間   -> ||e_layout|| 較大
λ 只有一個純量，補不回隨圖大小變動的倍率
```

**（v2）「大約 10 倍」已撤回。** 向量和的範數不是線性於 N：同方向時接近 N，
隨機方向時更接近 √N，相消時更小；而且 h_i 本身已經依賴鄰居數。
正確的說法只到「隨圖大小變動」，不到具體倍數。

**mean 也不是免費的答案。** 它換來圖大小不變性，代價是物件數量資訊被弱化、
少數關鍵物件的貢獻隨 N 增大被稀釋、大房間與小房間若組成相似則 global average 可能過於相近。
**所以這題的正解是 `sum vs mean` 消融，不是宣稱 mean 比較正確。**

**「EGNN 用 sum」這條 lineage 論證在此不成立** —— 它繼承了數字，沒繼承理由。

這是既有的 **USER-APPROVED IMPLEMENTATION CHOICE**，我不會擅自更動。
但建議重開此題，或至少把 `sum vs mean` 列入第一輪 ablation。

---

## 4. 標籤總表

| 結論 | 標籤 |
|---|---|
| `t_i ∈ R^d`、`f_h: R^(2d+1+e)→R^d`、殘差、`e` 為獨立維度、Eq.7 加法 | **METAFIND PAPER FACT** |
| h 在 l ≥ 1 的所有層寬度不變 = d | **INFERENCE**（由兩條型別簽章強制） |
| `dim(e_layout) = dim(Fusion)` | **INFERENCE**（由 Eq.7 加法強制，無條件） |
| d = Fusion 寬度 | **INFERENCE（條件式，v2 降級）** — 僅在 Pooling 保寬時成立 |
| Pooling 是否保寬 | **PAPER AMBIGUITY**（未命名，v2 新增） |
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
| 「BERT」是論文明列的編碼器例子 | **METAFIND PAPER FACT** |
| BERT-base、768 維、某種句向量抽法 | **IMPLEMENTATION CHOICE（v2 修正，原標 PAPER FACT 是錯的）** |
| node 與 edge 使用不同編碼器 | **IMPLEMENTATION CHOICE**（論文未禁止 ≠ 論文授權） |
| edge 佔第一層較多參數欄位 | **OBSERVED（算術）** |
| edge 在訓練中實際壓過幾何 | **HYPOTHESIS，未量測（v2 降級）** |
| QM9 用 sum 是因為目標外延 | **INFERENCE（v2 降級，EGNN 未如此陳述）** |
| `egnn_clean.py` 的 in/out projection | **EGNN 實作慣例（v2 修正，非 ARCHITECTURE FACT）** |
| A / B / C / D 四案 | **IMPLEMENTATION CHOICE CANDIDATE** |
| 建議 | **v2 撤回。真正待決的是「採哪一個 MetaFind 內部讀法」，寬度是下游** |

---

## 5. 外部審查已完成（v2）

上一版的六題已由外部審查逐條回覆，**九處全部駁回成立**，見本檔開頭的 v2 撤回紀錄。
外部審查同時聲明它無法直接讀 `/home/kyzen/upstream/OpenShape/*` 這個本機鏡像，
並且**沒有把我對該鏡像的描述當成證據** —— 本輪結論不依賴它。

### 現在該由使用者裁決的

```
第一順位   採哪一個 MetaFind 內部讀法（D Appendix-consistent / A §2.5 型別簽章）
           依 Rule 8，論文內部矛盾不得默默選一邊
第二順位   Pooling 是否保寬（決定 d 會不會被綁到 Fusion 寬度）
第三順位   層共享方式 independent / shared（參數量差約 7 倍）
最後       hidden 寬度的數值 —— 前三項定了才有合法上游
```

**在讀法定案前，不得更動 hidden 寬度，也不得把任何候選寫進協定。**
