# Retrieval Evaluation Definition — what Table 1 actually measures (2026-09-04)

Read-only audit. No code, split, gallery, scorer or test seal was touched. Sources: A0
`docs/paper/metafind_source/metafind_arxiv_v1.html` (offsets from the plain-text dump); Text2Shape
`docs/paper/text2shape_source/text2shape_arxiv_v1.html` + `/home/kyzen/upstream/text2shape` @ 2f62ebc;
this repo at `6ae3953`.

Labels: PAPER FACT · UPSTREAM FACT · OBSERVED CODE · PAPER-CONSTRAINED INFERENCE · STANDARD RETRIEVAL PRACTICE ·
IMPLEMENTATION CHOICE · UNRESOLVED · RETRACTED.

---

# 1. 一句話結論

**MetaFind Table 1 的 Text Only 是「文字丟進 query 塔（另外兩格放 mask token）→ 去找 gallery 塔用 T+I+PC 三模態編好的資產向量」，
正解是同一個 UID。** 是 §六 的選項 **D**，不是 T→T、T→PC、T→I。證據等級：gallery 端是 PAPER FACT，query 端「只給文字」是
PAPER-CONSTRAINED INFERENCE（§5）。它不是 pairwise cross-modal retrieval；把它讀成 T→PC 是概念錯誤（§10）。
**我們的程式碼做的正是 D**（§9），錯的不是程式，是過去幾份解釋裡拿 T→PC 的幾何去解釋 D 的數字。

---

# 2. Retrieval taxonomy

| 類型 | 定義 | 回答的問題 |
|---|---|---|
| A within-modal（T→T、I→I、PC→PC） | query 與 gallery 同一種編碼器、同一種模態 | 同模態空間裡分不分得開 |
| B pairwise cross-modal（T→I、T→PC、I→PC 及反向） | 一種模態的編碼器輸出去找另一種模態的編碼器輸出；正解 = 成對的同一資產 | 兩個模態有沒有對齊在同一空間 |
| C multimodal asset retrieval（MetaFind） | query = 任意模態子集經 query 塔；gallery = 三模態齊全經 gallery 塔，預先算好 | 「使用者給什麼都能找到那個資產」 |

A、B 是 STANDARD RETRIEVAL PRACTICE。C 是 MetaFind 自己定義的（§5）。三者的畫廊向量**不是同一種東西**，數字不可互比。

---

# 3. Text2Shape protocol（UPSTREAM FACT，除非另標）

| 項 | 內容 | 來源 |
|---|---|---|
| 資料切分 | "We create train/val/test splits by random subsets of 80%/10%/10% … For all experiments described here, we present results on the test split." | 論文 §5 @29246 |
| 正解 | "a retrieval is considered correct if the retrieved item (text description or shape) belongs to the same instance as the query … we categorize a retrieved sentence to be correct only if it came from the same model" | 論文 App. C.1 |
| Text-to-shape | query = 一句測試描述經 text encoder；gallery = 測試集形狀經 shape encoder；正解 = 該描述所屬的 model_id（一個） | 論文 §5.1「retrieve descriptions and shapes belonging to the same configuration … ground truth association for one shape」；Table 1 / Fig. 4 |
| Shape-to-text | 反向；一個形狀有多句描述 → **多個正解**（`num_relevant` = 同 label 的 gallery 列數） | 論文 App. C；`compute_pr_at_k` |
| Text-to-text / Shape-to-shape | 同一池子最近鄰，**排除自己**（`fit_eq_query` → 多取一個鄰居再刪掉 self） | `eval_text_encoder.py:95-134` |
| Query 是否在 gallery 裡 | 同模態時在（已排除 self）；跨模態時 query 是描述、gallery 是形狀，不同物件 | 同上 |
| 相似度 | 程式只接受 `metric='cosine'`，而該路徑是 `np.dot(query, fit.T)`，檔案印 "Using unnormalized cosine distance" → **未正規化內積**；訓練腳本開 `--lba_unnormalize` | `eval_text_encoder.py:109,162-181`；README |
| 前 k 名 | `np.argpartition` 取 top-k 再排序；同分順序任意 | `eval_text_encoder.py:112-117` |
| 指標 | `precision@k`、`recall@k`（命中數／正解數）、`recall_rate@k`（前 k 有無正解，論文寫 **RR@k**）、`NDCG@k`，k = 1..20；論文報 RR@1、RR@5、NDCG@5 | `compute_pr_at_k`；論文 Table 1/5/7 |
| 沒有的 | **沒有 MRR、沒有 reciprocal rank**。論文的 "RR" = recall rate，不是 reciprocal rank | 全文 0 命中 |
| 釋出的評估入口 | 只有單一池子的版本（一個 embeddings pickle，自己找自己）；text→shape 的入口沒有獨立腳本，`LBASolver` 以 `lba_test_mode text/shape` 各存一份 embedding | OBSERVED CODE；跨模態入口 UNRESOLVED |

```
Text-to-shape (Text2Shape)          positive = shape whose model_id == the caption's model_id
caption_i ─ Text Encoder ─ q_i ──┐
                                 ├─ dot(q_i, g_j)  → top-k  → RR@k / NDCG@k
shape_1..N ─ Shape Encoder ─ g_j ┘          (query set = test captions, gallery = test shapes)
```

這是 **TYPE B**（pairwise cross-modal），單一正解、exact instance、test→test。

---

# 4. Standard 3-modal pairwise diagnostic（建議，不是 Table 1）

命名 **PAIRWISE CROSS-MODAL DIAGNOSTIC**。六格（T→I、I→T、T→PC、PC→T、I→PC、PC→I）＋三格 within（T→T、I→I、PC→PC），
全部用**釋出的 ULIP-2 單塔輸出**（不經 Fusion），正解同 UID，test→test，cosine 與 raw dot 各報一次。

目的：(1) ULIP-2 三模態是否真的對齊（文字↔點雲、影像↔點雲各多強）；(2) 哪一對最弱；(3) 單模態本身的分辨力；
(4) 把「骨幹沒對齊」與「Fusion 學壞」分開；(5) PC→PC 是不是 identity shortcut（same-record 下必為 100）。
它**不能**標成 Table 1 重現：Table 1 的畫廊是 Fusion 過的三模態向量，這裡的畫廊是單模態向量。

已有的零成本量測（`output/look/exp_ulip_row_category_query.json`，dev_val 4,569，舊 split）可以直接改標籤成這一組的一部分：
own text→pc R@1 18.9、own view→pc 69.9、pc→pc 100.0（cosine）。其餘六格待跑（不用訓練，十分鐘）。

---

# 5. MetaFind Table 1 formal definition

對測試資產 A_i = (text_i, image_i, pc_i)：

```
g_j = f_gallery(text_j, image_j, pc_j)             for every gallery asset j      (precomputed, fixed)
q_i^c = f_query(subset_c(text_i, image_i, pc_i))   c ∈ {T, I, PC, T+I, T+PC, I+PC, T+I+PC}, absent slots = mask token
rank_i^c = 1 + #{ j ≠ i : sim(q_i^c, g_j) ≥ sim(q_i^c, g_i) }
R@k^c = mean_i [rank_i^c ≤ k]
```

| 句 | 原文 | 支持什麼 | 標籤 |
|---|---|---|---|
| §2.1 Eq. 1 | "retrieves the asset A* from a pre-encoded asset database 𝒜: A* = argmax sim(f_query(Q), f_gallery(A))" | 兩座不同的函數；gallery 端輸入是**資產 A**，不是某一模態 | PAPER FACT |
| §2.2 @11627 | "The gallery encoder precomputes embeddings for assets using three available modalities, which are then stored" | 畫廊 = 三模態編的向量 | PAPER FACT |
| §2.4 @16412 | "The gallery encoder is modality-complete and frozen after pretraining, while the query encoder remains flexible: It accepts any subset of modalities" | 畫廊永遠三模態；query 任意子集 | PAPER FACT |
| §2.6 @22143 | "each modality in the query has a 30% probability of being independently masked. Rather than zero-padding, we apply masked embeddings" | 缺席模態放 mask 向量（訓練期的機制，評估用固定子集時同一個機制） | PAPER FACT（機制）；「評估時用同一機制」PAPER-CONSTRAINED INFERENCE |
| §2.7 @25400 | "At inference time, all gallery asset embeddings are precomputed and cached … Given an input query—which may consist of any combination of text, image, point cloud … the query encoder generates a … embedding used to identify the most contextually suitable asset from the gallery" | 七個條件都是「子集 → 同一份畫廊」 | PAPER FACT |
| §3.2 @32900 | "All methods are evaluated under seven query conditions: text-only, image-only, point cloud-only, …" | 條件是 **query** 的條件，畫廊不變 | PAPER FACT |
| §3.2 F7 | "our dual-tower framework introduces more cross-modality retrieval, which results in lower accuracy under the 'PC only'" | PC-only 的 query 向量 ≠ 畫廊向量，因為畫廊是三模態融合的 | PAPER FACT（結論）；支持 D | 

**有沒有哪一句反對 §五的候選？沒有。** 唯一沒明寫的是「同一資產的 text_i／image_i／pc_i」——論文說訓練資料 "each asset has full modality
inputs"，評估時 query 用哪一份觀測沒寫（UNRESOLVED，見 SPLIT_RETRIEVAL_FRESH_AUDIT §8）。§五就是最小假設的讀法。

**§六的答案：D。** A（T→T）被 §2.2 排除（畫廊不是文字向量）；B（T→PC）被 §2.2/§2.4 排除（畫廊不是點雲向量，是三模態融合）；
C（T→I）同理。

---

# 6. Baseline formal definition

原文（§3.1 @31900，PAPER FACT）："we extend each baseline by adding a simple mean pooling layer to aggregate available modalities, and use
these fused embeddings to retrieve from a pre-encoded gallery." 加 F7："their 'PC only' performance reflects retrieval using **identical
embeddings for both query and gallery**".

| 候選 | 與 F7 相容？ | 判定 |
|---|---|---|
| B1 gallery = PC（shape）向量 | PC-only 的 query = 該點雲的向量 = 畫廊那一列 → 「identical」成立 | **PAPER-CONSTRAINED INFERENCE，最一致** |
| B2 gallery = T/I/PC 平均 | PC-only 的 query = PC 向量 ≠ 三模態平均 → 不 identical（除非平均被 PC 主導到幾乎相等） | 與 F7 字面衝突；只在數值上可能近似 |
| B3 baseline 原生 shape embedding | 對 ULIP／OpenShape／Uni3D 就是 PC 向量 = B1 | 同 B1 |

所以基線列的 Text Only 是 **T→PC（TYPE B）**，Full 是 mean(T,I,PC)→PC。基線與 MetaFind 列**測的不是同一種畫廊**：這正是 F7 要解釋的事，
也是為什麼基線 PC-only 97.9 而 MetaFind 75.1。這一點是「用 paper 語意判斷」，沒有動用我們的數字。

---

# 7. Split protocol

| 協定 | query | gallery | 用途 | 文獻 | 標籤 |
|---|---|---|---|---|---|
| A test→test | test | test | **final benchmark** | Text2Shape "results on the test split"；OpenShape／ULIP-2 zero-shot 也是整個測試集自成池子 | STANDARD RETRIEVAL PRACTICE；MetaFind 未寫 → PAPER-CONSTRAINED INFERENCE |
| B test→full | test | 全語料 | final 的另一讀法（"asset database"） | 無文獻先例 | UNRESOLVED，並報 |
| A20 test→holdout | test | val+test 9,138 | 論文 20% 大小的畫廊 | — | IMPLEMENTATION CHOICE（D-3b） |
| C val→val | val | val | **model selection** | Text2Shape 的 val | STANDARD RETRIEVAL PRACTICE |
| D val→train+val | val | 41,123 | **diagnostic only** | 無 | IMPLEMENTATION CHOICE；論文無任何支持 |

「dev query → train gallery」在文獻裡找不到先例；D 只能是 DIAGNOSTIC，這與 2026-08-27 協定裡 `reported: false` 一致。

---

# 8. Metrics

單一正解、exact instance 下（`rank_i` 為正解名次）：

| 名稱 | 定義 | 單一正解時 |
|---|---|---|
| R@K（MetaFind）＝ Hit@K ＝ Recall Rate@K（Text2Shape 的 RR@K） | mean[rank_i ≤ K] | **三者相同** |
| Recall@K | mean[命中數 / 正解數] | 正解數 = 1 → 等於 R@K |
| Precision@K | mean[命中數 / K] | = R@K / K |
| RR（reciprocal rank）／MRR | 1/rank_i 的平均 | 不同的量；Text2Shape **沒有**它 |
| NDCG@K | (1/log2(rank_i+1))·[rank_i ≤ K] 的平均（單一正解時 ideal = 1） | 介於 R@1 與 R@K 之間 |

同分：我們算輸（保守）；Text2Shape 的 argpartition 任意。MetaFind 未寫（UNRESOLVED）。
MetaFind 的 "top-k retrieval accuracy (R@1, R@5)"（§3.1 Metrics，PAPER FACT）= 上表第一列，PAPER-CONSTRAINED INFERENCE。
`metafind/eval/text2shape_eval.py` 已把上游的 RR@k／NDCG@5 逐字搬來並在每格旁邊印出，unit 向量上與我們的 R@k 相等（`tests/test_text2shape_eval.py`）。

---

# 9. Current repo audit（OBSERVED CODE，`6ae3953`）

| 格 | Q | G |
|---|---|---|
| Text only | `model.query({text_i, image_i, pc_i}, present=[1,0,0])` → Fusion 只看 text slot，另兩個 slot 是可學 mask token（`retrieval.py:90` `condition_mask`；`run_retrieval.py:838-843`） | `model.gallery({text_j, image_j, pc_j})`，缺一即 raise（`dual_tower.py:176`）；整個畫廊**只編一次**，七個條件共用（`run_retrieval.py:773-774, 838`） |
| Image only | present=[0,1,0] | 同上 |
| PC only | present=[0,0,1] | 同上 |
| T+I | present=[1,1,0] | 同上 |
| T+PC | present=[1,0,1] | 同上 |
| I+PC | present=[0,1,1] | 同上 |
| Full | present=[1,1,1] | 同上 |

- 正解 = 同 UID（`targets = col[uid]`，`run_retrieval.py:1173`）；負例 = 畫廊其餘全部；rank 含同分（`retrieval.py:107-220`）；float64 cosine。
- 訓練端與評估端走同一個 `split_embeds` 接縫（`stage1.py:950`），dev-val 選模用同一函數（`evaluate_dev_val`）。
- **結論：程式做的是 Q subset → full multimodal gallery，也就是 §5 的 D。沒有做錯。**
- 兩個與 §5 不同、但都是有紀錄的 IMPLEMENTATION CHOICE：(i) query 的 image 是 12 視角之 1，畫廊是 12 視角平均（P1，`--query-image-policy single_view`）；
  (ii) 部分 arm（P5／P8／P9／P10／P12）用 `--query-observation` / `--query-partner` / `--query-text-override` 換掉 query 端輸入，那些是 sensitivity。

---

# 10. Mismatch list

| # | 現況 | paper-faithful 定義 | 差在哪 | 標籤 |
|---|---|---|---|---|
| M1 | 評估器 = D（Q subset → G(TIP)） | D | 無差 | OBSERVED CODE ✅ |
| M2 | 解釋 MetaFind 格數字時用了 T→PC 的幾何（cos(text, pc) 0.29 等）與「ULIP 列的形狀」 | MetaFind 列的畫廊是 Fusion 向量，不是 pc 向量 | **概念混用**：拿 TYPE B 的直覺講 TYPE C | RETRACTED（作為解釋） |
| M3 | 基線探針（P0-4）試了 gallery=pc 與 gallery=mean3 | B1（pc）與 F7 一致 | B2 只當敏感度 | PAPER-CONSTRAINED INFERENCE |
| M4 | query image = 單視角、gallery = 12 視角平均 | 論文未寫 | 未必錯，但不是 same-record 字面 | IMPLEMENTATION CHOICE |
| M5 | P5／P8：query 三模態換第二份觀測 | 論文無此要求 | 是 robustness，不是 Table 1 | RETRACTED as protocol（前一份審計 §8） |
| M6 | P9：query 換同類別另一資產 | 正解定義變成類別 | 不是 exact-instance | DIAGNOSTIC only |
| M7 | D 協定曾被稱「主要對照」 | final = test→test | — | 已改口徑 |
| M8 | scorer cosine | 論文未定義；Text2Shape 是 raw dot | 兩種都印 | UNRESOLVED |
| M9 | 尚無 PAIRWISE CROSS-MODAL DIAGNOSTIC 的完整九格 | — | 骨幹對齊與 Fusion 未分開驗 | 建議 §11 |

**這個概念混用影響了哪些？** Stage 1 的解釋（M2）、Table 1 讀法（把 MetaFind text 13.8 跟 ULIP text 0.1 當同一種任務比）、基線重現（B1/B2 之爭其實由 F7 定）、
P1／P5／P8 的結論（它們的**數字**沒錯，錯的是「用擾動 query 追 T→PC 式直覺」的動機）、畫廊構法（MetaFind 列的畫廊必須是 Fusion 向量，
沒有第二種）、scorer（M8 獨立）、split（不受影響）。

---

# 11. Recommended next step（只建議）

1. 跑 PAIRWISE CROSS-MODAL DIAGNOSTIC 九格（釋出 ULIP-2、零訓練、test→test 需 ✅ 或先用 val→val），標名不標 Table 1。
2. Table 1 主線維持 D 讀法（現行程式）；等 P1s／P13 出來後，用 val→val 選定主線，再一次性 `--unseal` 跑 A／A20／B，同時報 text2shape RR@k／NDCG@5。
3. 說明文件與簡報裡所有「T→PC」式的解釋改寫成「子集 → 三模態畫廊」。

---

# 12. Retrieval taxonomy 總表

| Evaluation | Query | Gallery | Positive | Purpose | MetaFind Table 1? | Evidence |
|---|---|---|---|---|---|---|
| T→T | text enc | text enc | same asset | within-modal | No | §2.2 畫廊是三模態融合向量 |
| I→I | image enc | image enc | same asset | within-modal | No | 同上 |
| PC→PC | pc enc | pc enc | same asset | identity／alignment diagnostic | No（但基線列的 PC-only ≈ 這個，F7） | F7 |
| T→I / I→T | text | image | same asset | cross-modal alignment | No | §2.2 |
| T→PC / PC→T | text | pc | same asset | cross-modal alignment | No for MetaFind；**Yes for the baseline rows**（T→PC = 基線 Text Only） | §2.2；F7＋§3.1 mean pooling |
| I→PC / PC→I | image | pc | same asset | cross-modal alignment | No / baseline Image Only | 同上 |
| T→G(TIP) | query tower(text) | gallery tower(T,I,PC) | same UID | MetaFind retrieval | **Yes** | §2.2、§2.4、§2.7、§3.2 |
| I→G(TIP) | query tower(image) | 同 | same UID | MetaFind | **Yes** | 同上 |
| PC→G(TIP) | query tower(pc) | 同 | same UID | MetaFind | **Yes** | 同上 + F7 |
| T+I→G(TIP) | | 同 | | MetaFind | **Yes** | §2.4 "any subset" |
| T+PC→G(TIP) | | 同 | | MetaFind | **Yes** | 同上 |
| I+PC→G(TIP) | | 同 | | MetaFind | **Yes** | 同上 |
| TIP→G(TIP) | query tower(all) | 同 | same UID | MetaFind | **Yes** | 同上 |

---

**核心問題的答案：MetaFind Table 1 的 Text Only，是文字經 query 塔去找「三模態一起編好的資產向量」，正解是同一個資產。**
根據：§2.1 Eq. 1（找的是資產 A）、§2.2（畫廊用三模態預先編碼）、§2.4（畫廊 modality-complete、query 任意子集）、§2.7（推論時畫廊快取、query 任意組合）、
§3.2（七個是 query 條件）。不是 T→PC；T→PC 是基線列的定義（§6）。

---

## 附錄 A. Text2Shape 全文＋程式碼完整讀後補正（2026-09-04 19:5x，Kyzen 問「你有完整看過嗎」）

之前 §3 是用關鍵字抽段落寫的；這次把論文（正文＋附錄 A–D）與 `lib/lba.py`、`lib/solver_encoder.py`、
`lib/data_process_encoder.py`、`lib/losses.py`、`lib/layers.py`、`lib/config.py`、`tools/eval/eval_text_encoder.py` 全部讀完。補正與新增：

| # | 項 | 內容 | 標籤 |
|---|---|---|---|
| A1 | ShapeNet 上的絕對數字**非常低** | Table 7（ShapeNet，test 10% ≈ 1.5K 形狀、≈7.5K 句）：Full-MM **text-to-shape RR@1 0.40 / RR@5 2.37 / NDCG@5 1.35**；text-to-text 1.74 / 6.05；shape-to-text 0.83 / 3.37；Random 0.11 / 0.35。作者自己說「the absolute numbers do not entirely capture the performance」。→ exact-instance 的 text→shape 在千級畫廊就是個位數以下，MetaFind 基線列的 0.1–7（畫廊 ~9.6K）與此同量級。 | UPSTREAM FACT |
| A2 | 釋出的評估器是**混合池** | `LBASolver.get_outputs_dict` 把 test 的 shape embedding 與 text embedding **接成同一個池子**，`compute_metrics` 對這個池做「自己找自己、排除 self」的最近鄰；正解 = 同 model_id，所以一句 query 的正解**同時包含**該形狀與它的其他 4 句描述。訓練時每次 validation 用的分數是這個混合池的 **precision@5**（`solver_encoder.py:124`），連續 5 次不進步就停（early stopping，`:134-136`）。 | OBSERVED CODE |
| A3 | 純 text→shape 的入口**沒有釋出** | `compute_pr_at_k` 有 `fit_labels` 參數可以做 query≠fit，但 `compute_metrics` 永遠傳同一個矩陣當 fit 與 query。Table 7 三欄分開的數字，用釋出腳本跑不出來，要自己接。§3 表最後一列的 UNRESOLVED 改成：**確認沒有**。 | OBSERVED CODE |
| A4 | 訓練 batch 構造 | 每 batch **100 個不同形狀 × 每形狀 2 句**（App. A；`config.py` BATCH_SIZE 100、N_CAPTIONS_PER_MODEL 2；`LBADataProcess.run`）。同一 batch 內 model_id 不重複。 | UPSTREAM FACT |
| A5 | 損失 | L = TST(walker + 0.25·visit) + STS(同) + metric_tt + 2·metric_st（`lba.py:231-232`，`METRIC_MULTIPLIER 1`）。metric loss 是 Song et al. lifted-structure 的平滑版，相似度用點積除以 128、margin 1（`INVERTED_LOSS`）。**向量不做單位化**（README 訓練指令 `--lba_unnormalize`；論文 §4.4「embeddings are not restricted to have unit norm」），超過 norm 10 才罰（`MAX_NORM 10`，權重 2）。 | UPSTREAM FACT |
| A6 | 編碼器 | 文字：詞向量隨機初始化 + 4 層 1D conv + GRU(256) + fc → **128 維**；形狀：3D CNN on 32³ RGBA 體素 → 128 維。**沒有任何預訓練**（§1「use no pre-training」）。 | UPSTREAM FACT |
| A7 | 資料 | ShapeNet 椅+桌 15,038 個、75,344 句（每個 5 句，AMT 眾包，看旋轉動畫描述）；spaCy 小寫化＋lemmatize；>96 詞的句子丟掉；詞頻 ≤2 → UNK，詞彙 3,588。 | UPSTREAM FACT |
| A8 | 對 MetaFind 的啟示 | Text2Shape 的「同一形狀的其他描述也算正解」在 MetaFind 不存在（一資產一份描述），所以 MetaFind 的 text-only 比 Text2Shape 的 text-to-text 更嚴。它的 RR@k 定義、exact-instance 正解、test→test 池，與我們一致；它的 scorer（raw dot、不單位化）與我們不同，已並列印出。 | PAPER-CONSTRAINED INFERENCE |
