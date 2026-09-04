# Split & Retrieval-Protocol Fresh Audit — 2026-09-04

Static audit. Nothing under `data/`, `splits.py`, `stage1_protocol.json`, the towers, the
scorer or the test seal was modified. Every "paper says" carries a location in
`docs/paper/metafind_source/metafind_arxiv_v1.html` (A0); `.tex` line numbers are given only
to locate text. Tree at audit time: `eae4af6`.

Evidence labels used: PAPER FACT · PAPER FIGURE FACT · UPSTREAM FACT · OBSERVED IMPLEMENTATION ·
OBSERVED DATA · PAPER-CONSTRAINED INFERENCE · IMPLEMENTATION CHOICE · UNRESOLVED · RETRACTED ·
INVALID EXPERIMENT · DIRECT DEVIATION.

---

# 1. Executive Summary

**我們目前的 split 有沒有證據問題？** 沒有「錯」，但有一件事以前沒寫清楚：

- 80/20 是 PAPER FACT（§3.1）。**上游沒有任何 Objaverse-LVIS 內部的 train/test split 可以沿用**
  （§4 逐一查過 Objaverse 官方套件、allenai/objaverse HF、ULIP-2 的 `lvis.json`、OpenShape 的
  `meta_data/split/lvis.json`、Uni3D 的 `lvis_testset.txt`）。上游把**整個 LVIS 當 zero-shot 測試集**，
  訓練時反而把它排除。所以 GPT 的 H1「作者沿用上游 split」對物件層是 **不成立**（候選 B 排除）。
  作者必然自己切；隨機或按規則、seed 為何，**UNRESOLVED**。
- 70/10/20 裡的 「10」不是論文的，是 **DIRECT DEVIATION D-3**，Kyzen 2026-08-27 核可
  （`metafind/data/splits.py:102-131`）。**80/10/10 當天就被提出並被 Kyzen 否決**，理由記在同一段：
  它把論文的 20% 測試集拆掉一半，最終畫廊變 10%。這次 H2 是同一個提案再提一次。
- 兩者都沒有論文原文能裁決；差別只在「哪一個更少動到論文固定的三個數字（80 訓、20 測、測試不參與選模）」。
  S1（現行）三個都不動；S2（80/10/10）動了「20 測」。

**我們目前的 Table 1 retrieval protocol 有沒有證據問題？** 骨架沒有，宣稱方式有：

- 骨架（同 UID 為正解、七個固定條件、eval 不做隨機遮罩、畫廊三模態齊、R@1/R@5、float64 cosine、同分算輸）
  逐條對過 A0，全部是 PAPER FACT 或 PAPER-CONSTRAINED INFERENCE，程式碼與之一致（§7）。
- 問題在：(1) 論文**沒寫** query pool 與 gallery pool；A（test→test）與 B（test→full）兩個都是我們的讀法，
  一直並報是對的。(2) 過去把 **D（dev_val→train）當論文對照**在講，這是錯的：D 的畫廊是訓練集，
  `reported: false` 從 2026-08-27 起就寫在協定裡。(3) 「query 用第二份觀測」在 A0 裡**沒有任何一句**支持，
  正文層級降為 **RETRACTED INFERENCE**；它只剩 Table 1 基線數字給的間接約束（§8）。

**目前最可能需要重做的是什麼？** 不是 split，也不是評估器。是：

1. **Q/G tower 的讀法**（§9）：Kyzen 2026-09-01 曾裁 A（兩份 ULIP-2），同日被 Codex 的第三讀法打開
   （DL-070 「reopened, not resolved」）；Figure 1 原圖印 `ULIP-2 (Shared)`。程式只實作 shared backbone，
   `fully_separate` 會被 trainer 拒跑。**這是唯一一條「論文有兩種可讀、程式只能跑一種」的項目。**
2. 把 P8／P9／P10／P12／縱圖／perturbation 這一整批從「追 Table 1」重新標回 sensitivity／diagnostic（§10）。
3. 最終數字只能來自 A／B，需要 `--unseal`，而且只做一次（§7）。

---

# 2. Paper Facts

每條附 A0 位置（純文字化後的字元偏移，`tools` 內 `dump_paper.py` 可重現）與 `.tex` 行。

| # | 項目 | 原文 | 位置 | 標籤 |
|---|---|---|---|---|
| F1 | 80/20 | "In both datasets, we allocate 80% of the data for training and reserve the remaining 20% for testing." | A0 @30215，§3.1；`3experiments.tex` Datasets 段 | PAPER FACT |
| F2 | 語料 | "the annotated Objaverse-LVIS dataset containing 48K unique 3D assets" / "approximately 48,000 distinct 3D assets. Each asset is rendered from 11 orthogonal viewpoints and annotated using GPT-4o." | A0 @30215（§3.1）、@13900（§2.3） | PAPER FACT（數字是約數；釋出的 manifest 是 46,052，OBSERVED DATA） |
| F3 | 檢索任務 | "A* = argmax_{A∈𝒜} sim(f_query(Q), f_gallery(A))" ；"retrieves the asset A* from a pre-encoded asset database 𝒜" | A0 @10800，Eq. 1，§2.1 | PAPER FACT |
| F4 | 七種條件 | "All methods are evaluated under seven query conditions: text-only, image-only, point cloud-only, text+image, text+point cloud, image+point cloud, and full (text+image+point cloud)." | A0 @32900，§3.2 | PAPER FACT |
| F5 | 畫廊預先編碼、三模態齊 | "The gallery encoder precomputes embeddings for assets using three available modalities, which are then stored for efficient retrieval." / "At inference time, all gallery asset embeddings are precomputed and cached" / "The gallery encoder is modality-complete" | A0 @11627（§2.2）、@25400（§2.7）、@16412（§2.4） | PAPER FACT |
| F6 | 指標 | "standard retrieval metrics, including top-k retrieval accuracy (R@1, R@5)" | A0 @32227，§3.1 Metrics | PAPER FACT |
| F7 | 基線 PC-only 的解釋 | "since other models do not adopt a dual-tower design, their 'PC only' performance reflects retrieval using identical embeddings for both query and gallery, leading to inflated accuracy. In contrast, our dual-tower framework introduces more cross-modality retrieval, which results in lower accuracy under the 'PC only'." | A0 @33565，§3.2 | PAPER FACT |
| F8 | 雙塔用語 | "MetaFind employs a dual-tower architecture with separate encoders for the query and gallery. Each tower leverages ULIP-2 to independently encode available modalities" / "a dual-tower retrieval framework consisting of a query encoder and a gallery encoder, both leveraging the ULIP-2 embedding backbone" | A0 @16077（§2.4）、@11627（§2.2） | PAPER FACT |
| F9 | Figure 1 | 圖框標籤 `ULIP-2 (Shared)`；caption "both the user query and candidate assets are encoded using the ULIP-2 backbone … each 3D asset in the repository is pre-encoded independently by ULIP-2 into a fixed vector" | `MetaFind.drawio.png`（DL-090 讀圖）；A0 @3980 | PAPER FIGURE FACT |
| F10 | Stage 1 遮罩 | "each modality in the query has a 30% probability of being independently masked. Rather than zero-padding, we apply masked embeddings" — 寫在 "Stage 1: Cross-Modal Alignment Pretraining … are trained" 段 | A0 @22143，§2.6 | PAPER FACT（訓練期） |
| F11 | Stage 1 訓什麼 | "both query and gallery encoders are trained on large-scale object-level data from Objaverse-LVIS, where each asset has full modality inputs" | A0 @22143 | PAPER FACT |
| F12 | Stage 2 凍結 | "Only the query-side fusion layer and the ESSGNN module are updated during this stage; the gallery encoder is frozen" | A0 @24239，§2.6 | PAPER FACT |
| F13 | 基線怎麼評 | "we extend each baseline by adding a simple mean pooling layer to aggregate available modalities, and use these fused embeddings to retrieve from a pre-encoded gallery" | A0 @31900，§3.1 | PAPER FACT |
| F14 | τ | "The temperature is 0.5 for all experiments." | A0 @31900 | PAPER FACT |
| F15 | Table 1 MetaFind w/o ESSGNN R@1/R@5 | 13.8/23.1 · 11.7/19.2 · 75.1/78.0 · 17.2/21.8 · 44.5/71.3 · 45.8/73.1 · 51.7/76.5 | A0 @36000 | PAPER FACT |

論文**沒有**出現的字（A0 全文搜尋 0 命中）："validation"、"epoch"、"learning rate"、"cosine"、"dot product"、"split file"。

---

# 3. Paper Unknowns

| # | 未知 | 論文查到什麼 | 標籤 |
|---|---|---|---|
| U1 | train/test UID 如何產生 | 只有 F1 一句；沒有 seed、沒有規則、沒有檔案 | UNRESOLVED |
| U2 | Table 1 的 query pool | "reserve 20% for testing" 沒接到 Table 1；未寫 query = 測試集 | PAPER-CONSTRAINED INFERENCE：query = 20% test（`splits.py` 模組說明第 U-09 段已記錄此假設） |
| U3 | gallery pool / size | F3 說 "pre-encoded asset database"，F5 說 "all gallery asset embeddings"；沒說是 20% 還是 100% | UNRESOLVED（A/B 並報） |
| U4 | q_text 字串 | §2.3 只說 GPT-4o 產生「structured text descriptions」；query 端字串形式未寫 | UNRESOLVED |
| U5 | q_img 視角／聚合 | "11 orthogonal viewpoints" 是資料描述；query 用幾張、怎麼聚合未寫 | UNRESOLVED |
| U6 | q_pc 來源 | 未寫是否重取樣、部分掃描、canonical | UNRESOLVED |
| U7 | sim(·,·) | Eq. 1 "denotes the similarity function"，未定義 | UNRESOLVED（現行 cosine = IMPLEMENTATION CHOICE，`stage1_protocol.similarity`） |
| U8 | Q/G 觀測是否同一份 | F11 "each asset has full modality inputs"（訓練資料）；沒有任何一句說 query 觀測與 gallery 不同 | UNRESOLVED；「必為第二份」→ RETRACTED（§8） |
| U9 | tower 參數是否共用 | F8 "separate encoders" vs F9 `ULIP-2 (Shared)` | UNRESOLVED（§9） |
| U10 | 正解定義 | 未寫 "same asset"；Eq. 5 以 (Q, A) 成對、F7 談 "identical embeddings for both query and gallery" | PAPER-CONSTRAINED INFERENCE：exact-instance |
| U11 | Eval 時是否遮罩 | F10 寫在訓練段；F4 列的是固定條件 | PAPER-CONSTRAINED INFERENCE：eval 不隨機遮罩 |
| U12 | Table 1 的 R@k 分母 | 未寫 n_query、是否去重、同分怎麼算 | UNRESOLVED |

---

# 4. Upstream Dataset Split Audit

**問題：Objaverse-LVIS 有沒有官方 train/test split？答：沒有。** 逐一：

| 來源 | 查到的東西 | 結論 | 標籤 |
|---|---|---|---|
| `objaverse` 官方套件（`site-packages/objaverse/__init__.py`） | 只有 `load_lvis_annotations()`：回傳 **category → uid list**（1,156 類）。原始碼 "train"/"test" 0 命中；"split" 1 命中是 `str.split("/")` | 無 split | UPSTREAM FACT（OBSERVED） |
| `allenai/objaverse` HF 樹（curl `api/datasets/allenai/objaverse/tree/main`，2026-09-04） | `glbs/ metadata/ lvis-annotations.json.gz object-paths.json.gz README.md` | 無 split 檔 | UPSTREAM FACT（OBSERVED） |
| ULIP-2 `SFXX/ulip` HF 樹 | 頂層 `ULIP-1/ ULIP-2/ ULIP_Objaverse_Triplets/ ULIP_Shapenet_Triplets/`；`ULIP-2/` 下 `modelnet40_10k_colored_pc/ objaverse_lvis/ pretrained_models/` | 無 split 檔；`objaverse_lvis/` 是 160 個 shard（本機 `/mnt/data1/kyzen/ulip2_objaverse_lvis/ULIP-2/objaverse_lvis/000-000.tar.gz … `，共 160，無其他檔） | UPSTREAM FACT（OBSERVED） |
| ULIP-2 `lvis.json`（我們 manifest 的來源，`download.py:61-79` 從 `ULIP_Objaverse_Triplets/lvis.json` 抓） | dict，**46,052** 筆 uid → `000-009/<uid>.npy`；沒有 split 欄位 | 無 split | OBSERVED DATA |
| ULIP-2 `objaverse_lvis_metadata.json` | keys `value_to_key_mapping`(46,207 uid→類別)、`all_keys`(1,156)、`key_to_id`；"train" 命中全是類別名（`bullet train`、`train (railroad vehicle)`…） | 無 split | OBSERVED DATA |
| ULIP 官方碼 `data/dataset_3d.py:456-533` `Objaverse_Lvis_Colored` | 讀 `lvis.json` 全部 → `file_list`；用於 `--evaluate_3d_ulip2 --validate_dataset_name=objaverse_lvis_colored`（`scripts/test_ulip2_pointbert_objaverse_lvis.sh:11`）；`main.py:367` 用 `lvis_metadata['all_keys']` 當 zero-shot 類別 | **整個 LVIS = 測試集**，zero-shot 分類 | UPSTREAM FACT |
| ULIP-2 論文（`docs/paper/ulip2_source/ulip2_arxiv_v4.html` @25705） | "Objaverse-LVIS is a subset of the Objaverse dataset with human-verified category labels. It has ∼46k samples spanning ∼1.2k categories, which is suitable for more challenging open-world zero-shot 3d shape classification." Table 1 有 "Objaverse(no LVIS) + ShapeNet" 與 "Objaverse + ShapeNet" 兩列（46.3 vs 50.6） | 用整個 LVIS 測；訓練有含／不含 LVIS 兩種，但**不在 LVIS 內部切** | UPSTREAM FACT |
| OpenShape 官方碼 `src/configs/train.yaml:61-63` + `src/data.py:333-344` + `README.md:112` | `objaverse_lvis.split: meta_data/split/lvis.json` 是**一個清單**，`ObjaverseLVIS` 全讀；`train_no_lvis.json` = 四資料集去掉 LVIS 形狀；`test_objaverse_lvis()` 做 zero-shot 分類，`best_lvis_acc` 用來存 checkpoint | 整個 LVIS = 測試集；訓練排除 | UPSTREAM FACT |
| OpenShape 論文（`openshape_arxiv_v2.html` @3990、@25738、@27040） | "We exclude shapes in Objaverse-LVIS during training"；"comprises 46,832 shapes among 1,156 LVIS categories"；"Ensembled (no LVIS) … excludes all shapes from the Objaverse-LVIS subset" | 同上 | UPSTREAM FACT |
| Uni3D 官方碼 `data/Objaverse_lvis_openshape.yaml` | `PC_PATH: ./data/test_datasets/objaverse_lvis/lvis_testset.txt` | 檔名就叫 testset，整個 LVIS | UPSTREAM FACT |
| ProcTHOR-10K（場景層） | `prior.load_dataset("procthor-10k")` 有官方 `train/val/test`（`download.py:216`；本機 `datasets/procthor-10k/{train,val,test}.jsonl`） | **場景層有官方 split**，但論文 F1 說兩個資料集都 80/20 → 與官方 10k/1k/1k 不同 | UPSTREAM FACT；已列 UNRESOLVED（`STAGE1_RESOLUTION_PLAN §2`） |

**未查到／查不到：** MetaFind 作者的 split 檔（NeurIPS 頁只有 Paper + Bibtex，DL-070；OpenReview 被擋，arXiv 只有 v1）。

**§3 候選裁決：**
- 候選 B（沿用上游 split）：**排除**。上游沒有 LVIS 內部 split 可沿用；上游全把 LVIS 當測試集。
- 候選 A（自己隨機 80/20）與 C（自己建固定 manifest 未公開）：**無法區分**。兩者對我們的意義相同：
  作者的 UID 集合不可得，我們的 test 20% 與作者的 test 20% **必然是不同集合**，這件事本身已是可比較性的上限。
- 裁決：**D（證據不足以在 A/C 之間唯一決定），但 B 已可排除。** 標籤 UNRESOLVED（U1）。

---

# 5. Current Repo Split Audit

```
46,052   Objaverse-LVIS manifest（ULIP-2 lvis.json；download.py）                 OBSERVED DATA
46,024   n04 實際渲染出來                                                            OBSERVED DATA
45,692   admitted corpus = 46,024 − 311（n05 quarantine）− 21（人工剔除）             OBSERVED DATA
         splits.json: admitted_total 45,692                                          OBSERVED DATA
   │
   ├── split_assets(seed 20260816, 0.8)  ── sorted → shuffle → round(0.8·N)          OBSERVED IMPLEMENTATION
   │      train 36,554  /  test 9,138                                                OBSERVED DATA
   │
   └── split_dev(train, seed 20260827, 0.125)                                        OBSERVED IMPLEMENTATION
          dev_train 31,985  /  dev_val 4,569
          等效 70 / 10 / 20 of corpus
```

| 設計 | 來源 | 標籤 |
|---|---|---|
| A. 80/20 | F1 | PAPER FACT（比例）；seed、排序後洗牌、四捨五入切點 = IMPLEMENTATION CHOICE（`splits.py:154-166`） |
| B. 從 80% 再切 0.125 → 10% corpus 當 dev_val | 不是論文（論文沒有 "validation"）；不是上游（上游沒有 split）；是 **DIRECT DEVIATION D-3，Kyzen 2026-08-27 核可**（`splits.py:102-131` 逐字記錄，含被否決的 80/10/10） | DIRECT DEVIATION（ML hygiene，使用者核可） |
| C. test 20% 完全 sealed | 論文只說 "reserve … for testing"；「選模不能碰測試集」是 D-3 的推論加 Codex 2026-08-30 的 `check_seal`（`run_retrieval.py:477`） | PAPER-CONSTRAINED INFERENCE + IMPLEMENTATION CHOICE（機制） |
| D. C 協定 dev_val→dev_val | checkpoint selection；`reported: false`（`splits.py:236`） | IMPLEMENTATION CHOICE（選模） |
| D. D 協定 dev_val→train | diagnostic；畫廊全是訓練過的資產；`reported: false`（`splits.py:263`） | IMPLEMENTATION CHOICE（診斷）；**不得報成論文數字** |

---

# 6. Candidate Split Protocols

| | S1 現行 70/10/20 | S2 80/10/10 | S3 80/20 無 val | S4 沿用上游 split |
|---|---|---|---|---|
| 符合論文明文？ | 80 訓、20 測、測試不動：**三個都符合**；10% val 論文沒提 | 80 訓符合；**20 測被拆成 10**，最終 test 只剩論文一半 | **最貼字面** | 不可行：物件層上游沒有 split（§4） |
| 符合上游？ | 上游沒有可比 protocol | 同左 | 同左 | — |
| 適合 strict reproduction？ | 訓練集比論文少 10 個百分點（31,985 vs 36,554）；這是唯一與論文不同處 | 訓練量與論文相同；測試集規模與論文不同 | 訓練量、測試規模都同論文；**但沒有任何 model-selection 依據** | — |
| 汙染論文測試集？ | 否 | val 與 final test 不同資產，final test 本身沒被選模碰；但「論文保留的 20%」有一半被拿去選模 | 否（若真的完全不看 test） | — |
| 適合選超參／選 checkpoint？ | 是（C 協定） | 是 | **否**；需要事先寫死 recipe（epoch 數、lr）。ULIP-2 上游 recipe 存在（`main.py` 250 epochs），MetaFind 自己的 recipe 論文沒寫 | — |
| final 還能稱 held-out？ | 是 | 是（但只有 10%） | 是 | — |
| Table 1 在哪算？ | test 9,138（A）／full 45,692（B） | final test ≈4,569（若 test→test）；**畫廊只有論文一半**，R@k 不可直接比 | test 9,138／full | — |

**S2 特別討論（GPT 要求）：** 這 10% val 來自論文保留的測試分布——「分布」相同這點對 S1 也成立（隨機切，同分布）。
真正的差別是 **final 畫廊大小**：test→test 下 S2 的畫廊是 4,569，S1 是 9,138，論文（若 20% 且 ≈48K）約 9,600。
R@1 隨畫廊變小而變高（D vs C 已量到 8 倍畫廊差 20 個點以上），所以 S2 在 test→test 下的數字**系統性偏高**，
且不是 "方便" 能解釋的偏差，是 benchmark 被改了。S2 的唯一好處是訓練量回到 80%；
但 P1e25 已量到多訓練不改形狀（§5a），訓練量差 12.5% 不是目前的瓶頸。

**裁決：S1 保留。** 沒有證據要求改；改了會讓所有既有 arm 失去可比性；80/10/10 已在 2026-08-27 被 Kyzen 否決一次。
若要更貼字面，可行的是 **S3 當最終跑法**（用 S1 選好 recipe 後，用全 80% 重訓一次，固定 epoch，只跑一次 A/B）；
這是「先選再重訓」的標準做法，不動 split 檔，只多一個 final run。標籤 IMPLEMENTATION CHOICE，需 Kyzen ✅。

---

# 7. Candidate Retrieval Protocols

先重建定義（§七）：對測試資產 A_i、條件 c ∈ 七種：
`q = f_query(subset_c(A_i 的觀測))`，`g_j = f_gallery(T_j, I_j, P_j)` 對畫廊全體，`rank = 1 + #{j ≠ i : sim(q, g_j) ≥ sim(q, g_i)}`，
R@k = mean[rank ≤ k]。

| 項 | 論文 | 我們 | 標籤 |
|---|---|---|---|
| A. 正解 = 同 UID | 未明寫；Eq. 5 的 (Q, A) 成對、F7 的 "identical embeddings for both query and gallery" 只在 instance-level 才說得通 | `targets = col[uid]`（`run_retrieval.py:1173`） | PAPER-CONSTRAINED INFERENCE，程式一致 |
| B. 負例 = 畫廊其餘全部 | Eq. 1 argmax over 𝒜 | 全畫廊 GEMM，同分算輸（`retrieval.py:107-220`） | PAPER-CONSTRAINED INFERENCE，程式一致（同分算輸是我們保守的選擇） |
| C. 畫廊三模態齊 | F5 | `GalleryTower.forward` 缺一即 raise；`encode_pools` 畫廊只編一次（`run_retrieval.py:768-774`） | PAPER FACT，程式一致 |
| D. eval 時 30% 遮罩？ | F10 在訓練段；F4 七個固定條件 | `condition_mask` 固定（`retrieval.py:90-105`）；`sample_modality_mask` 只在 `stage1.py:2999` 訓練迴圈 | PAPER-CONSTRAINED INFERENCE：**不遮**，程式一致 |
| E. instance 還是 category | 論文沒有 category-level 的任何字；Table 1 基線 PC-only 97.9–99.0 只有 instance-level 才有這個量級 | instance | PAPER-CONSTRAINED INFERENCE |
| F. R@k 成功條件 | "top-k retrieval accuracy" | rank ≤ k，rank 含同分 | PAPER FACT（名稱）；同分規則 IMPLEMENTATION CHOICE |

**Pool 比較：**

| 協定 | query | gallery | 含訓練資產？ | 可選模？ | 可報 Table 1？ | 與論文的關係 | 優 | 缺 |
|---|---|---|---|---|---|---|---|---|
| P-A test→test | 9,138 | 9,138 | 否 | 否（sealed） | **是** | F1 的 20%；畫廊≈論文 20% 若作者也如此 | 最保守；不含訓練資產 | 若作者畫廊是全集，數字偏高 |
| P-B test→full | 9,138 | 45,692 | 是（36,554） | 否（sealed） | **是** | F3 "asset database"、F5 "all gallery asset embeddings" 讀成全庫 | 貼近「檢索整個倉庫」的應用敘事 | 畫廊含訓練資產，數字偏低；作者未說 |
| P-C train+test | 同 P-B | 同 P-B | 同 | 同 | 同 | 數學上 = P-B | provenance 寫法不同而已 | — |
| P-D dev_val→dev_val（C） | 4,569 | 4,569 | 否 | **是** | 否 | 無 | 選 checkpoint | 畫廊小，數字偏高；null model 也能過（DL-044） |
| P-E dev_val→train（D） | 4,569 | 36,554 | 是（全部） | 診斷 | 否 | 無 | 8× 畫廊、看訓練效果 | 畫廊全是訓練過的資產 |
| P-F 80/10/10 val→val；final test→test | 4,569 / 4,569 | 4,569 / 4,569 | 否 | 是 | final 是，但畫廊是論文一半 | 動了 F1 的 20% | 訓練量回 80% | benchmark 規模改變（§6） |
| P-G 80/10/10 final test→full | 4,569 | 45,692 | 是 | — | 是 | 同 P-B 但 query 少一半 | — | 同 P-F |

**裁決：** query = test 是 PAPER-CONSTRAINED INFERENCE；gallery 範圍 UNRESOLVED，**A 為主、B 並報**（與 2026-08-27 的 U-09 決定相同）。
C 只選模，D 只診斷；兩者的數字以後只能標 "dev"，不得與 F15 並排寫成「對照論文」。
A/B 需 `--unseal`，且**只在 Stage 1 定案後跑一次**。

---

# 8. Query Construction Audit

**「second observation 有沒有 paper evidence？」——沒有。** A0 全文沒有任何一句要求：query 用另一份觀測、
query 圖片必須 held-out、query 點雲必須重取樣、query 文字必須另一段描述。§2.6 F11 說訓練資料是 "each asset has
full modality inputs"，§2.3 說每資產 11 視角 + GPT-4o 註解——這些都是**資料集描述**，不是 query 構法。

因此：
- 「論文的 query 一定是另一份 observation」→ **RETRACTED INFERENCE**（2026-09-01 E1 協定與 2026-09-04 P8 都是在這個
  未被支持的前提下設計的；它們的結果仍是有效量測，但只是 sensitivity）。
- 論文明確定義的只有 **modality subset**（F4）。

**但有兩件間接證據要保留，標清楚等級：**

1. **Table 1 基線列（PAPER FACT 的數字 + 我們的量測）**：八個基線 text-only 0.1–6.9、image-only 0.1–2.3。
   用資產自己的（gallery 那份）文字查自己的 PC 畫廊或三模態畫廊，我們量到 text 18.9–71.4、image 69.9–92.3（畫廊 4,569，pc／三模態平均兩種畫廊）
   （`exp_ulip_row_category_query.json`，釋出 ULIP-2、零訓練）。**沒有一種 same-record 算法給得出 0.1。**
   這是 PAPER-CONSTRAINED INFERENCE：基線的 q_text／q_img 不是 gallery 存的那份向量。它約束的是**基線**的評估，
   對 MetaFind 列只能推「同一套 harness」（F4 "All methods are evaluated under…"），不能直接推 MetaFind 的 q_text。
2. **NeurIPS 2025 海報（PAPER FIGURE FACT，`docs/reference/metafind_neurips2025_poster.png`）**：畫廊文字顯示為整包
   annotation JSON；query 文字顯示 `Platform Bed (size: …)`。這是圖示，不是協定；它支持「query 文字**形式**與畫廊不同」，
   不支持「query 是第二份觀測」。

**分層結論：**

| 層 | 內容 | 標籤 |
|---|---|---|
| 論文明定 | query modality subset ∈ 七種 | PAPER FACT |
| 論文未定 | q_text 字串、q_img 視角／張數、q_pc 是否重取樣、Q/G 是否同一份、scorer | UNRESOLVED |
| 主線讀法 | **same-record**：q_text／q_img／q_pc 就是該資產的那份紀錄（P1 的圖片用 12 張裡的 1 張，畫廊用 12 張平均，這一點已是偏離 literal same-record 的 IMPLEMENTATION CHOICE） | PAPER-CONSTRAINED INFERENCE（最少假設） |
| 假設 | 「q_text 是短式（類別＋尺寸）」——海報支持形式差異 | 假設，待 P12（text serialization arm） |
| 假設 | 「q_img／q_pc 是第二份觀測」 | RETRACTED as protocol；保留為 robustness sensitivity |

---

# 9. Query/Gallery Tower Separation Audit

**論文三句（全 PAPER FACT）：** F8 "separate encoders for the query and gallery. Each tower leverages ULIP-2 to
independently encode"；F9 圖框 `ULIP-2 (Shared)`；F12 Stage 2 "gallery encoder is frozen"、F11 Stage 1 "both … are trained"。
外加 §2.4 前一句："While prior works typically align 3D encoders to a fixed CLIP embedding space by freezing pretrained
text and image encoders, our MetaFind framework adopts a more flexible dual-tower design."

```
候選 A  shared ULIP-2 ──┬── Fusion_Q          （現行 stage1_protocol.tower_sharing = shared_backbone_separate_fusion）
                        └── Fusion_G
候選 B  ULIP-2_Q ── Fusion_Q ；ULIP-2_G ── Fusion_G   （兩份 backbone；程式 `fully_separate`，trainer 拒跑 stage1.py:2563）
候選 C  = B 但兩份 backbone 由同一 checkpoint 初始化   （實務上 B 只能這樣做，C ≡ B）
候選 D  shared ULIP-2 ── 一個 Fusion 兩塔共用          （`fully_shared`；freeze_gallery 拒跑，因 F12 無法同時成立）
```

| 證據 | 支持 | 反對 | 備註 |
|---|---|---|---|
| F8 "separate encoders … Each tower leverages ULIP-2 to independently encode" | B/C；也可讀成 A（"encoder" = 整座塔含 fusion，"leverages ULIP-2" = 共用 backbone） | D | 字面不足以裁 |
| F9 `ULIP-2 (Shared)` | A/D | B/C | 圖是 OBSERVED DATA（原圖讀出，DL-090） |
| F12 gallery frozen / query flexible（Stage 2） | 排除 D（一份 fusion 無法一凍一動）；A、B 都可 | — | Kyzen 2026-09-01 用這句裁 A（DL-068 的「A」= 本表 B）；Codex 指出 A（shared backbone + 兩個 fusion）同樣滿足這句 → **reopened**（DL-070） |
| §2.4 "prior works … freezing pretrained text and image encoders … ours more flexible" + §3.4 "full encoder fine-tuning yields better performance by allowing earlier layers to adapt" | 至少有一座塔的 encoder 在 Stage 1 有更新；若兩塔各自更新且輸入相同，B 才有「兩份不同權重」 | — | 不區分 A/B |
| F7（Table 1 那句） | 見下 | | |

**F7 到底支持哪一種？** 原文對照的是 "identical embeddings for both query and gallery"（基線）與 "our dual-tower framework
introduces more cross-modality retrieval"（MetaFind）。主詞是 **embedding** 與 **encoder 架構**，不是 observation。
所以 F7 支持的是「**同一份 PC 輸入 → 走不同的 query/gallery encoder → 得到不同 embedding**」（讀法 1），
而不是「query 與 gallery 用不同的觀測」（讀法 2）。這一句對 A/B 不區分：A 之下 PC-only 的 query 走
`Fusion_Q(mask, mask, p)`、gallery 走 `Fusion_G(t, i, p)`，embedding 已經不同（P1 量到 66.6/86.1 對論文 75.1，DL-090）。
"cross-modality" 三個字在 A 之下也成立：query 只有 PC，gallery 是三模態融合。

**裁決：U-16 仍 UNRESOLVED，且是唯一「論文可雙讀、程式只能跑一讀」的項目。**
- 現行 A 有 F9 撐；B 有 Kyzen 2026-09-01 的裁決但已被自己接受 reopen。
- 程式狀態（OBSERVED IMPLEMENTATION）：`dual_tower.py` 不含 backbone；`stage1.py` 只建一個 `ULIPBackbone`；
  `fully_separate` 與 `shared_backbone_separate_fusion` 建出**同一個模型**（DL-068 量過參數張量身分）；trainer 對
  `fully_separate` 直接 `SystemExit`。實作 B 需要：第二份 Point-BERT、checkpoint 帶兩份 backbone、評估器兩路編碼。
  CLIP 文字／影像塔凍結且走 n06 快取，B 只會分開 **點雲路徑**。
- **這次審計不動架構**（§十二禁令）。要不要開 B 的 arm（P13）由 Kyzen 決定；預期見 §11。

---

# 10. Reclassification of Existing Experiments

分類：A = paper reproduction candidate；B = sensitivity arm；C = diagnostic only；D = invalid；E = still unresolved。
數字皆為 D 協定 R@1（dev_val→train，`reported: false`），只用來說明分類，不對照論文。

| 實驗 | 動了什麼 | 舊定位 | 新分類 | 理由 |
|---|---|---|---|---|
| pilot10b same_record（v2_cm 長文、12 視角平均、raw） | 三模態全同一份 | 主線候選 | **A**（literal same-record 的第一個實現）+ **E**（text 58.0 過高） | 沒有論文句子反對它；數字問題在文字字串化與 tower，不在 protocol |
| P1（attrs_v1、query 單視角、prefusion L2） | 文字改填表句；query 圖片 1/12 | 主線 | **A**（現行主線）；「query 單視角」這一項本身是 IMPLEMENTATION CHOICE，未被論文要求 | 仍是 same-record 讀法 |
| P3（12 view tokens） | Fusion 輸入粒度 | arm | **B**（C4 sensitivity） | 與 P1 同族 |
| P4（一份 Fusion 共用） | tower 候選 D | arm | **C**（D 已被 F12 排除，只能當診斷） | freeze_gallery 拒跑 → Stage 2 不可用 |
| P5（desc_v1；query = 第二名描述 + 重取樣 pc + 單視角） | 三模態都第二份 | 「independent-observation stress test」 | **B**（robustness） | 前提已 RETRACTED（§8） |
| P6（每步隨機視角） | 訓練期視角 | arm | **B** | 同族 |
| P7（prefusion L2 關） | C8 | arm | **B** | 同族，非決定軸 |
| query second observation（E1 協定、query pack） | 第二份觀測 | 主線假設 | **B** + 前提 **RETRACTED** | §8 |
| PC resample / nocolor / jitter / sparse / half scan（`exp_query_pc_observation`） | q_pc 擾動 | 追 Table 1 | **B**（robustness）；**不得再用來追 Table 1**（§十二） | 論文無此要求 |
| P8（第二描述 + held-out 視角 + 半掃描 pc） | 三模態擾動 + lr 1e-4 | 曾被寫成主線方向 | **B**；**撤回「主線」標記** | 同上 |
| P9（同類別另一資產的文字／圖片，訓練＋測試） | 換資產 | 假設檢驗 | **C**（診斷：模型學會忽略 text/image） | 任務定義已非 exact-instance |
| P1 + partner query at test（5i） | 測試時換資產 | 「找到了」 | **C**；「形狀翻過來」是 diagnostic 觀察，不是 protocol 候選 | 換了資產就不是 F7 的 "identical" 對照 |
| P10（gallery 描述句／query 欄位句） | 文字形式 | 排隊中 | **B**（text serialization sensitivity，same-record） | 對應 U4 |
| P12（gallery JSON／query 類別＋尺寸，照海報） | 文字形式 | 排隊中 | **B**（text serialization；海報支持形式差異） | 對應 U4；若形狀翻轉也只升到 E，不能直接升 A |
| 縱圖 / ULIP-2 npz 文字（thumbnail、name、blip、msft） | q_img／q_text 換來源 | 探針 | **B** | 資料來源與論文的 11 視角 + GPT-4o 不同 |
| Protocol C | dev→dev | 選模 | **C**（選模） | `reported: false` |
| Protocol D | dev→train | 「主要對照」 | **C**（診斷）；**撤回「主要對照」** | 畫廊全是訓練資產 |
| raw-dot scorer probe（P0-C） | sim 定義 | 探針 | **C**（U7 的診斷：dot 給 PC-only 97.9） | 不能因為對上 97.9 就升 A |
| ULIP baseline PC-only 97.9 probe / ULIP row 重現（5f、5g） | 基線構法 | 探針 | **C**（約束基線 harness） | 對 MetaFind 列只是間接 |
| 早期壞探針（46） | — | INVALID | **D** | 已撤，`test_probe_gallery_parity.py` 守住 |
| lr sweep、P1e25、AMP | 超參／效率 | — | **C**（訓練工程） | 與 protocol 無關 |
| U-16 tower 分離 | 架構 | — | **E**（唯一的 P0 未決） | §9 |

---

# 11. Recommended Mainline

只在證據夠的地方寫；其餘 UNRESOLVED。

| 項 | 建議 | 依據 | 標籤 |
|---|---|---|---|
| split | **S1 保留**（70/10/20，seed 不動，test 不解封） | §4 無上游 split；§6 S2 改 benchmark；2026-08-27 已裁 | 維持 DIRECT DEVIATION D-3 |
| final 訓練量 | S3 式「選好 recipe 後用全 80% 重訓一次」——**可選**，需 ✅ | 補回 12.5% 訓練量，不動 split 檔 | IMPLEMENTATION CHOICE（待 ✅） |
| query pool | test 20% | F1 | PAPER-CONSTRAINED INFERENCE |
| gallery pool | **A 主、B 並報** | U3 無法裁 | UNRESOLVED（並報是對的處理） |
| 正解 | 同 UID | F7 + Eq. 5 | PAPER-CONSTRAINED INFERENCE |
| eval 遮罩 | 無 | F4 | PAPER-CONSTRAINED INFERENCE |
| Q/G 觀測 | **same-record 為主線**；P8 類降 sensitivity | §8 | PAPER-CONSTRAINED INFERENCE |
| q_text | UNRESOLVED；P12（海報形式）是下一個該看的 arm，但它是 text serialization 不是 second observation | §8 | UNRESOLVED |
| q_img | UNRESOLVED（單視角 vs 平均都是我們的） | U5 | UNRESOLVED |
| q_pc | same-record（重取樣類實驗只作 robustness） | U6 | PAPER-CONSTRAINED INFERENCE |
| tower | **UNRESOLVED**；若要開 B（兩份 Point-BERT）需 Kyzen ✅，且先做零訓練探針（DL-068 提的「一份 checkpoint 載兩份、擾動一份、評分」） | §9 | UNRESOLVED |
| scorer | cosine 維持；dot 為 diagnostic | U7 | IMPLEMENTATION CHOICE |

**對 P13 的預測（寫在跑之前，避免事後解釋）：** 兩份 ULIP-2 由同一 checkpoint 出發，對同一份輸入初始輸出完全相同，
Eq. 5 在 same-record 下的固定點仍是「兩塔一致」；預期 full 仍 > 95。若實測掉到論文量級（~52），§9 的 A 讀法就要撤。

---

# 12. Author Questions

只能靠作者回答的：

1. Objaverse-LVIS 的 80/20 是怎麼切的（隨機 seed？按類別分層？有沒有公開 UID 清單）？
2. Table 1 的 query pool 是不是 20% test 全體？n_query 多少？
3. Table 1 的 gallery 是 20% test（≈9.6K）還是全部 48K？
4. q_text 的字串：GPT-4o 的完整結構化描述？類別＋尺寸的短句（海報所示）？還是 JSON？
5. q_img：11 視角中的一張？哪一張？還是多視角聚合？聚合方式？
6. q_pc：與畫廊同一份點雲，還是重取樣／部分？
7. query 與 gallery 的三模態觀測是否同一份紀錄？
8. Query encoder 與 gallery encoder 的 ULIP-2（Point-BERT 與 CLIP 塔）是否共用參數？Stage 1 更新到哪些層？
9. sim(·,·) 是 cosine 還是未正規化點積？基線的 PC-only 97.9（非 100）是怎麼產生的？
10. 基線列的 q_text／q_img 是什麼（它們的 0.1 用資產自己的文字重現不出來）？

---

# 13. 總表

| 問題 | 現行做法 | 候選做法 | 證據等級 | 目前裁決 | 是否要改 |
|---|---|---|---|---|---|
| train/test split source | 自切，seed 20260816 | 沿用上游 | 上游無 split（UPSTREAM FACT） | 候選 B 排除；A/C 不可分 | 否 |
| 70/10/20 | D-3，Kyzen 2026-08-27 | — | DIRECT DEVIATION | 保留 | 否 |
| 80/10/10 | — | S2 | 無論文支持；2026-08-27 已否決；改 benchmark 規模 | 不採 | 否 |
| query split | test 20% | — | PAPER-CONSTRAINED INFERENCE | 維持 | 否 |
| gallery split | A test / B full 並報 | 只報一個 | UNRESOLVED | 並報，A 主 | 否（口徑改：D 不再當論文對照） |
| positive target | 同 UID | category-level | PAPER-CONSTRAINED INFERENCE | 同 UID | 否 |
| eval masking | 無 | 30% 隨機 | PAPER-CONSTRAINED INFERENCE | 無 | 否 |
| query text source | 資產自己的填表句（attrs_v1） | 海報短句 / JSON / 描述 | UNRESOLVED | same-record 主線；形式待 P10/P12 | 待證據 |
| query image source | 12 視角之 1 | 平均 / 縱圖 | UNRESOLVED（單視角是我們的） | 維持，標 IMPLEMENTATION CHOICE | 待證據 |
| query pc source | 同一份 | 重取樣 / 部分 | PAPER-CONSTRAINED INFERENCE | 同一份 | 否 |
| second observation | E1／P5／P8 曾當主線 | — | 無 paper evidence → RETRACTED | 降為 sensitivity | 是（分類與文件） |
| query/gallery tower sharing | shared backbone + 2 fusion | 兩份 ULIP-2 | UNRESOLVED（F8 vs F9；DL-068 裁 A 後 reopened） | **P0 未決** | 待 Kyzen ✅ 才開 arm |
| scorer | cosine | dot | UNRESOLVED（U7） | cosine；dot 為診斷 | 否 |
| Protocol C | 選模 | — | IMPLEMENTATION CHOICE | 選模 only | 否 |
| Protocol D | 曾寫「主要對照」 | — | IMPLEMENTATION CHOICE | 診斷 only | 是（口徑） |
| P8 | 曾寫「主線方向」 | — | 前提 RETRACTED | sensitivity | 是（分類） |

---

## 附：這次審計檢查過的檔案

A0 `metafind_arxiv_v1.html`（全文純文字化搜尋）；`2methdology.tex`/`3experiments.tex` 只作定位。
上游：`objaverse` 套件原始碼、HF `allenai/objaverse` / `SFXX/ulip` 樹（curl 2026-09-04）、
`/mnt/data1/kyzen/ulip2_objaverse_lvis/ULIP-2/objaverse_lvis/`（160 shard，無其他檔）、
`upstream/ULIP/{data/dataset_3d.py,data/Objaverse_Lvis_Colored.yaml,scripts/test_ulip2_pointbert_objaverse_lvis.sh,main.py}`、
`upstream/OpenShape_code/{README.md,src/configs/train.yaml,src/data.py,src/train.py}`、
`upstream/Uni3D/data/Objaverse_lvis_openshape.yaml`、`ulip2_arxiv_v4.html`、`openshape_arxiv_v2.html`。
Repo：`metafind/data/splits.py`、`download.py`、`metafind/models/dual_tower.py`、`metafind/train/stage1.py:2555-2575`、
`metafind/eval/retrieval.py`、`run_retrieval.py`、`data/outputs/{splits.json,stage1_protocol.json,eval_protocols.json}`、
`workflow/DECISION_LEDGER.md`（DL-046/068/070/090）、`docs/audit/STAGE1_FRESH_AUDIT_20260903.md §C`、
`workflow/STAGE1_RESOLUTION_PLAN_20260903.md`、`output/look/ARMS_TABLE.md`。

---

## 14. Ruling after the audit（Kyzen，2026-09-04 16:0x，逐字「80/10/10 就這樣拆」）

§6 的顧慮已呈報，Kyzen 讀後重申採 **S2 = 80/10/10**。照 §6 的定義實作，記為 **DIRECT DEVIATION D-3b**：

- 論文的 80/20（seed 20260816）**逐位元不動**：train 36,554 與之前完全相同；論文的 20%（9,138，現稱 `holdout`）
  用 seed 20260904 對半切成 `val` 4,569（選模）／`test` 4,569（最終，封印）。
- 舊檔複製保留：`outputs/splits_70_10_20_seed20260816_dev20260827.json`、`outputs/eval_protocols_70_10_20.json`。
- 協定：A test→test 4,569；**A20 test→holdout 9,138（論文 20% 的畫廊大小，reported；註記 val 那一半參與過選模）**；
  B test→full 45,692；C val→val 4,569（選模）；D val→train 36,554（診斷）。`check_seal` 把 `holdout` 一併封印。
- `dev_train`／`dev_val` 以**別名**保留（= train／val），trainer 與探針不用改路徑。
- 可比性：70/10/20 下訓練的所有 checkpoint（P1…P12）訓練池是 31,985，新制是 36,554；它們的 C/D 查詢集也不同。
  新制的主線對照是 **P1s**（P1 配方重跑），與 P13（兩份點雲路徑，Kyzen ✅ 同日）排在同一條鏈。
- 資料夾視圖：`outputs/split_dirs/{train,val,test,holdout}/{annotations,pointclouds,embeddings}/`（符號連結，
  `tools/materialize_split_dirs.py` 產生）；純 uid 清單在 `outputs/split_lists/*.txt`。
- 到本次為止，**沒有任何數字來自 test**：所有 Table 1 對照都是 C/D（dev_val 4,569 查詢），`--unseal` 從未給過。
