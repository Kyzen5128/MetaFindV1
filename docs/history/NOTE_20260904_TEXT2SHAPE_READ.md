# Text2Shape 讀後筆記 —— 論文全文＋官方程式碼（2026-09-04）

來源：`docs/paper/text2shape_source/text2shape_arxiv_v1.html`（Chen, Choy, Savva, Chang, Funkhouser, Savarese；arXiv 1803.08495v1，2018-03-22）
與 `/home/kyzen/upstream/text2shape` @ 2f62ebc（MIT）。讀過的檔：`lib/lba.py`、`lib/solver_encoder.py`、`lib/data_process_encoder.py`、
`lib/losses.py`（`build_lba_loss`）、`lib/layers.py`（`smoothed_metric_loss`）、`lib/config.py`、`tools/eval/eval_text_encoder.py`、`README.md`。
標籤：UPSTREAM PAPER ／ UPSTREAM CODE；兩者衝突處明標。它是 MetaFind Table 1 基線 SCA3D／Parts2Words 這一支的祖先，也是「RR@k」這個寫法的出處。

---

## 1. 它要做什麼

兩個任務：(i) **text-to-shape retrieval**：一句話找形狀；(ii) **text-to-shape generation**：一句話生成上色體素（CWGAN）。
本筆記只管 (i)，因為 MetaFind 只引用它的檢索評估。

核心主張：不用細粒度類別／屬性標籤，只用「一句描述 ↔ 一個形狀實例」的配對，把文字和形狀學進同一個空間，
讓語意相近的描述和形狀自己靠在一起（§1、§4）。

## 2. 資料（UPSTREAM PAPER §3、App. B）

| 項 | 內容 |
|---|---|
| ShapeNet | 椅子 6,591 ＋ 桌子 8,447 ＝ **15,038** 個 CAD 模型；**75,344** 句描述（平均每個 5 句），AMT 眾包，工人看**旋轉動畫**寫「顏色、形狀、材質、外觀」 |
| 形狀表示 | 上色體素 32³ × 4 通道（RGB＋佔據），256³ 取樣後低通濾波再降取樣；solid 體素 |
| 文字前處理 | 小寫、spaCy tokenize＋lemmatize、LanguageTool 拼字修正；>96 詞的句子丟掉（16 句）；詞頻 ≤2 → UNK；詞彙 3,588 |
| Primitives | 合成資料：6 種基本形 × 14 色 × 9 尺寸 = 756 種設定，每種 10 個擾動樣本 = 7,560 形狀；模板句 191,850 句 |
| 切分 | **80/10/10 隨機**（train/val/test），兩個資料集都是；「所有結果都報 test split」（§6） |

## 3. 模型（UPSTREAM PAPER App. A Table 3；UPSTREAM CODE）

- 文字編碼器：詞向量隨機初始化可學 → conv1..4（128/128/256/256，kernel 3）→ GRU(256) → fc 256 → fc **128**。
- 形狀編碼器：3D CNN，conv 64/128/256（stride 2）→ avg pool → fc **128**。
- **沒有任何預訓練**（§1「use no pre-training」；§2 特別對比 Reed et al. 需要預訓練＋細類別標籤）。
- 向量**不做單位化**（§4.4「not restricted to have unit norm」）；README 訓練指令 `--lba_unnormalize` → `LBA.NORMALIZE=False`；
  超過 norm 10 才罰（`MAX_NORM 10`，權重 2；`lba.py:234-240`）。

## 4. 訓練（UPSTREAM PAPER §4、App. A；UPSTREAM CODE）

### 4.1 每個 batch 長什麼樣
- **100 個不同形狀，每個形狀抽 2 句描述**（App. A「100 unique shapes … two matching captions per shape」；
  `config.py` `BATCH_SIZE 100`、`N_CAPTIONS_PER_MODEL 2`；`LBADataProcess.run` 保證 batch 內 model_id 不重複）。
- label = 第幾個形狀（0..99，每個重複兩次），這是唯一的監督：**同形狀的兩句是正例，其餘全是負例**。

### 4.2 損失（`lba.py:159-241`、`losses.py:141-190`、`layers.py:148-`）
```
L = L_TST + L_STS + γ (L_ML^TT + L_ML^TS)                        (論文 Eq. 3)
L_TST = walker(text→shape→text) + 0.25 · visit                    learning by association（Haeusser et al.）
L_STS = walker(shape→text→shape) + 0.25 · visit
L_ML  = lifted-structure 平滑版（Song et al.），相似度 = 點積/128，margin 1；
        程式：metric_tt × 1 ＋ metric_st × 2                       （`METRIC_MULTIPLIER 1`，`lba.py:231-232`）
```
- **walker**：P^{TS} = softmax(T·Sᵀ)，P^{ST} = softmax(S·Tᵀ)，來回機率 P^{TST} = P^{TS}P^{ST}，目標是「回到同形狀的那兩句」均勻分布，取 cross-entropy。
- **visit**：每個形狀被走到的機率要均勻（熵最大化），避免大家都擠到少數形狀。
- MM = TST + STS 都用；TST = 只有一個方向。論文 Table 1/5/6/7 顯示 MM 明顯較好。
- 相似度全程是**點積**（§4.3「we define similarity between embeddings with the dot product」）。

### 4.3 優化與停止（UPSTREAM CODE）
- README 指令：`--learning_rate 2e-4 --num_epochs 100 --decay_steps 2500 --batch_size 100 --visit_weight 0.25 --lba_mode MM --lba_unnormalize`。
- **每次驗證**：把 val 的形狀向量與文字向量**接成一個池**，自己找自己（排除 self），算 precision@5（`solver_encoder.py:96-142`）；
  連續 5 次驗證都沒超過歷史最佳就**early stop**（`:134-136`）。

## 5. 檢索評估（UPSTREAM PAPER §6.1、App. C；UPSTREAM CODE `eval_text_encoder.py`）

### 5.1 定義
- **正解 = 同一個 model_id**（App. C.1：「a retrieval is considered correct if the retrieved item (text description or shape)
  belongs to the same instance as the query」）。語意一樣但不同模型 → 算錯。作者明說絕對數字因此偏低。
- 指標：**RR@k** = recall rate = 前 k 名裡至少一個正解（§6.1）；**NDCG@k**；程式另算 precision@k、recall@k（命中／正解數），k = 1..20。
  **沒有 MRR、沒有 reciprocal rank**（全文 0 命中）。RR 不是 reciprocal rank。
- 池子：test split；同模態時排除自己（`_compute_nearest_neighbors_cosine` 的 `fit_eq_query` 多取一個再刪 self）。
- 相似度：程式只接受 `metric='cosine'`，但那條路徑是 `np.dot(query, fit.T)` 且印「Using unnormalized cosine distance」→ **未正規化內積**；
  `TextEncoderSolver` 寫 `minkowski`（L2）但實際會被 `compute_nearest_neighbors` 拒絕（`raise ValueError('Use cosine distance.')`）。
- 前 k 名用 `np.argpartition`，同分順序任意。

### 5.2 釋出的評估器到底算哪一種
- `tools/eval/eval_text_encoder.py main`：吃**一個** pickle（`text_embeddings.p` 或 `shape_embeddings.p`），池 = 該 pickle 全部，fit == query。
  → 只能做 **text-to-text** 或 **shape-to-shape**。
- 訓練中的驗證（`LBASolver.get_outputs_dict`）與 `save_outputs` 的 `text_and_shape_embeddings.p`：**形狀＋文字混合池**，
  query 是池裡每一個向量；一句 query 的正解 = 同形狀的形狀向量**和**其他描述。
- **純 text→shape（query=文字、fit=形狀）沒有釋出入口**：`compute_pr_at_k(fit_labels=...)` 有這個能力，但 `compute_metrics` 永遠傳同一個矩陣。
  論文 Table 6/7 分開的三欄，得自己接。

### 5.3 數字（UPSTREAM PAPER）

| 資料集 | 任務 | Full-MM RR@1 / RR@5 / NDCG@5 | Random |
|---|---|---|---|
| Primitives（Table 1/6） | text→shape | 95.07 / 99.08 / 95.51 | 0.24 / 0.76 / 0.27 |
| Primitives（Table 6） | shape→text | 93.47 / 93.47 / 93.47 | 2.8 / 6.53 / 1.84 |
| Primitives（Table 5） | text→text ／ shape→shape | 100 ／ 96.13 / 99.20 / 95.59 | — |
| **ShapeNet（Table 7）** | **text→shape** | **0.40 / 2.37 / 1.35** | 0.11 / 0.35 / 0.23 |
| ShapeNet | text→text | 1.74 / 6.05 / 1.43 | 0.08 / 0.38 / 0.08 |
| ShapeNet | shape→text | 0.83 / 3.37 / 0.73 | 0.07 / 0.34 / 0.06 |

Primitives 的「正解」是同一種**設定**（class），一種設定有 10 個形狀、~255 句，所以正解很多；ShapeNet 是 exact instance，所以掉到個位數以下。

## 6. 對 MetaFind 的意義

| 項 | Text2Shape | MetaFind / 我們 | 判定 |
|---|---|---|---|
| 正解 | 同一 model_id | 同一 UID | 一致（PAPER-CONSTRAINED INFERENCE） |
| R@k 定義 | RR@k = 前 k 名有正解 | R@k 同 | 一致；我們的評估器已逐字並印 RR@k／NDCG@5（`metafind/eval/text2shape_eval.py`） |
| 池 | test → test（10%） | A test→test；A20 test→holdout 9,138；B test→全庫 | 一致的是 A |
| 一資產幾份文字 | 5 句，正解多個 | 1 份，正解 1 個 | MetaFind 更嚴 |
| 相似度 | raw dot（不單位化） | cosine（並印 raw dot） | 不同；U7 未決 |
| 訓練 | 從零、association＋metric、100×2 batch | 凍結 CLIP，Point-BERT＋Fusion，InfoNCE | 完全不同族，不可抄數值 |
| 千級畫廊的 text→shape 量級 | RR@1 0.4 | 基線列 0.1–7、MetaFind 13.8（畫廊 ~9.6K） | 基線列數字不是異常 |

一句話：Text2Shape 給的是「**評估的定義**」——同實例、RR@k、test 池、內積——不是訓練配方。它的釋出程式只有同池版本，跨模態要自己接；
我們已經接了（`text2shape_metrics`：fit = 畫廊、query = 查詢、每列一類）。

---

## 7. 逐字讀完後的補充（2026-09-04 19:5x；再讀 `losses.py`、`layers.py`、`config.py`、`main.py`、`solver.py`、`network.py`、`models/lba_models.py`、`run_lba_encoder.sh`、README 全文）

**訓練細節（UPSTREAM CODE，補 §4）**
- 優化器 Adam；lr 預設 1e-4（`config.py:84`），README 的 ShapeNet 指令用 **2e-4**；指數衰減 0.95、每 2,500 步一階（staircase）（`network.py:10-15`）。
- 100 個 epoch（`NUM_EPOCHS 100`）；每 2,500 步存 checkpoint 並驗證一次（`CKPT_FREQ = VALIDATION_FREQ = 2500`）；驗證最多取 20,000 個樣本（`solver.py:125`）。
- 訓練隨機種子 123（`config.py:28`、`main.py:251`）；資料佇列 3 個 worker。
- `--lba_only` = 關掉度量損失（只剩 association）；`--metric_learning_only` = 關掉 association（只剩度量）。論文 Table 1 的 LBA-MM／ML 就是這兩個開關。
- walker loss 的目標分布 `p_target`：同 label 的欄位均勻、其他 0（`losses.py:99-101`）；`p_aba = softmax(TSᵀ)·softmax(STᵀ)`（`:110-112`）；visit loss = 每欄平均訪問機率對均勻分布的交叉熵（`:28-45`）。STS 方向的 label 是 `range(BATCH_SIZE)`，即每個形狀自成一類（`:168`）。
- 度量損失（`layers.py:148-201`）：`INVERTED_LOSS=True` 時 D = 點積/128，J_ij = log Σ exp(1 + D_ik) − D_ij，取 max(0,·)² 的平均再除 2；正對永遠是「相鄰兩列」（batch 排成 caption_1_1, caption_1_2, …）。text-shape 版是把偶數列換成該形狀的向量再算同一個式子（`lba.py:214-226`），權重 ×2。
- `LBA.DIST_TYPE` 還有 `mahalanobis` 選項（可學矩陣 A，定期投影成 PSD），README 沒用，論文沒提。
- 早停：`TextEncoderSolver.validate` 保留最近 5 次驗證的 precision@5，若全部都高於本次就終止（`solver_encoder.py:134-136`）。

**評估細節（UPSTREAM CODE，補 §5）**
- 測試模式強制 `BATCH_SIZE 1`、`N_CAPTIONS_PER_MODEL 1`（`main.py:335-339`）。`--lba_test_mode shape` 時每個形狀只餵它的**第一句**當文字（`data_process_encoder.py:451-452`）；`text` 模式餵每一句，重複句子去重（`:394-402`）。
- 訓練中驗證用的是 **val** split（`--val_split` 預設 val）；README 產 embedding 的指令沒帶 `--val_split test`，要自己給才會在 test 上算。
- `eval_text_encoder.compute_metrics` 用 20 個鄰居算 k = 1..20，但只在單一池上。`TextEncoderSolver` 寫 metric='minkowski' 會被 `compute_nearest_neighbors` 拒絕（`raise ValueError('Use cosine distance.')`）—— 這條路徑只有非 LBA 的舊 text encoder 才走到。

**模型（補 §3）**：`LBA1` = `CNNRNNTextEncoder` + `ShapeEncoder`（`models/lba_models.py:5-10`）；`--lba_unnormalize` 拿掉輸出的 L2 normalize（`lba.py:92-108`）。

**沒讀的（與檢索無關）**：`cwgan.py`、`t2s_*_component.py`、`classifier*.py`、`data_process_gan.py`、`render.py`、`nrrd_rw.py`、`encoder_component.py` 的層定義（論文 Table 3 已列）。
