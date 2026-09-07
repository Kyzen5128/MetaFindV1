# 筆記 2026-09-04：ULIP-2 有沒有被動、Stage 1 一輪總結、Stage 2 初步結果

> Kyzen 的問題（2026-09-04）：「只要有關融合全部跑偏，你都是比較高……ULIP 是將 3 模態拉近，你怎麼感覺是 3 個的準度相加？你是不是有動 ulip2 的設計？」
> 這份筆記先回答這題，再把 Stage 1 每個版本的訓練與評估數字、Stage 2 初步結果、選定版本、待驗證項目寫齊。
> 數字全部來自 repo 內的紀錄檔；出處寫在每一節。

---

## 一、ULIP-2 沒有被動；分數高的原因在 query 觀測，不在 backbone

### 1. 程式碼比對（OBSERVED IMPLEMENTATION）

- `metafind/vendor/ulip/` 的 17 個檔案與上游 `/home/kyzen/upstream/ULIP` 逐 byte 相同（`ulipdiff.py` 比對，2026-09-04）。
- `metafind/compat/ulip_patch.py` 只做三件相容的事，沒有改任何數學：
  - `torch._six` 在新版 PyTorch 不存在，補一個轉接。
  - `pointnet2_ops` 的 CUDA 擴充編不起來，`misc.fps` 換成純 torch 的貪婪最遠點取樣，同一個演算法、同樣從索引 0 起算。
  - `knn_cuda` 沒有裝，放一個會直接報錯的替身（我們的路徑不會呼叫它）。
- 我們載入的權重是官方釋出的 `ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt`，Point-BERT 18 層，全程凍結。

### 2. 設計對照：ULIP-2 論文 vs 我們

| 項目 | ULIP-2（arXiv 2305.08275 v4） | 我們 |
|---|---|---|
| 文字／影像編碼器 | OpenCLIP ViT-G/14，凍結 | 同一個，凍結 |
| 3D 編碼器 | Point-BERT，ULIP-2 預訓練時訓練 | 載入釋出權重，凍結 |
| 3D 輸入 | 10k 點 xyzrgb（OpenShape 前處理） | 10k 點 xyzrgb |
| 訓練損失 | P2I + P2T 雙向對比，可學 τ | 我們不訓練 ULIP-2；Stage 1 只訓 Fusion |
| 相似度 | cosine | cosine（float64） |
| 論文有沒有「融合」 | 沒有，三個模態各自一個向量 | Fusion 是 MetaFind 加的，不屬於 ULIP-2 |

ULIP-2 論文沒有任何檢索指標，也沒有多模態融合；MetaFind Table 1 的 ULIP 列是 MetaFind 自己定的協定。

### 3. 零樣本檢驗：釋出權重在我們的資料上重現論文數字（OBSERVED DATA）

跑法完全照 ULIP 官方 `main.py`（第 41 行預設模板 `modelnet40_64`；第 372–405 行：每類 64 個模板句 → 正規化 → 平均 → 再正規化；點雲正規化；cosine 取 top-1 / top-5）。

| | top-1 | top-5 |
|---|---|---|
| ULIP-2 論文 Table 10（Point-BERT, 10k xyzrgb, ViT-G, Objaverse-LVIS） | 50.6 | 79.1 |
| 我們（dev_val 4,569 朵，1,156 個 LVIS 類別全放進候選） | **50.9** | **79.3** |

工具：`tools/probes/ulip2_zero_shot_lvis.py`；輸出 `output/look/exp_ulip2_zero_shot_lvis.json`。
差異說明：論文用整個 Objaverse-LVIS（約 46K 朵），我們用 dev_val 的 4,569 朵；類別集合同樣是 1,156 類。

結論：backbone、點雲前處理、文字塔三者合起來就是論文的 ULIP-2。分數高不是這裡來的。

### 4. 那為什麼融合格像「相加」？——因為 query 的點雲跟 gallery 的點雲幾乎是同一份

不是相加，是「pc 一個人就決定了名次，text／image 只能整體拉低分數、翻不動排序」。

- 釋出的 ULIP-2、不訓練、query 直接用可用模態的平均，對 36,554 個 gallery 的 R@1：pc 100.0、text+pc 98.6～99.3、full 96.6～98.0，不管文字用哪一種（類別名／填表／描述／完整模板）都一樣（`output/look/ARMS_TABLE.md` ULIP row hypothesis 節）。
- 數學：q = (p + t)/|p + t|，gallery 就是自己的 p 時，自己的分數是 1 + p·t，別人是 p·p_j + t·p_j。除非文字 t 偏愛別人超過 pc 本身的差距 1 − p·p_j，否則名次不會翻。所以「沒資訊的文字」只會把全體分數往下拉，排序不動。
- 配對相似度（釋出編碼器）：第二份文字 cos 0.80、第二份影像 0.93、重取樣點雲 0.99（`output/look/exp_observation_geometry.json`）。點雲換成重取樣的第二份，text+pc 就從 98.6 掉到 96.8；full 從 96.6 掉到 94.6。
- 論文的形狀（text+pc 44.5 < pc 75.1）需要 query 的 pc 跟 gallery 的 pc 離得遠，或文字系統性地誤導。我們的 query 觀測太像 gallery，這是資料層的問題（怎麼取得 query 的文字／影像／點雲），Kyzen 已裁定資料集調整延後。

評語分類：第 1–3 節 OBSERVED；第 4 節的解釋是 INFERENCE，數字是 OBSERVED DATA。

---

## 二、Stage 1 一輪總結

### 1. 共同的訓練設定（所有 arm 相同；`data/outputs/stage1_hyperparameters.json`）

| 項目 | 值 | 來源 |
|---|---|---|
| 訓練／驗證 | dev_train 31,985 ／ dev_val 4,569（P5 因描述缺漏 31,970） | OBSERVED DATA |
| 優化器 | AdamW，lr 5e-4，weight decay 0.1，betas (0.9, 0.98) | UPSTREAM（ULIP）候選，Kyzen 核可的 pilot 值 |
| 排程 | warmup 1 epoch 從 1e-6，cosine 到 1e-5 | 同上 |
| batch | 64 | 同上 |
| 損失 | Eq. 5 單向 InfoNCE，τ = 0.5 固定 | PAPER FACT |
| 遮罩 | query 每模態獨立 30% 遮掉、換 mask token | PAPER FACT |
| epoch | 10（pilot ladder 5→10→25 的第二階） | IMPLEMENTATION CHOICE |
| 選 checkpoint | dev_val 七格 R@1 平均最高的 epoch（每個 arm 都是 epoch 9） | IMPLEMENTATION CHOICE |
| seed | 20260816 | |

### 2. 每個版本：內容、訓練終點、驗證

訓練終點 loss = 最後一個 epoch 記到的 batch loss（`train_stage1.jsonl`）；驗證 = dev_val 七格 R@1 平均（gallery 4,569）。

| 版本 | 跟前一版差在哪 | 終點 loss | dev_val mean R@1 | 結果 |
|---|---|---|---|---|
| pilot10b | 舊構法：query = gallery 同一份紀錄、v2_cm 長模板、12 視角平均、原始輸入 | 2.34 | 0.947 | 全部 90+，形狀 0.57，最遠 |
| P1 | 填表文字 attrs_v1 + query 單一視角 + 融合前 L2 | 2.47 | 0.795 | 主線 |
| P4 | P1 + 一份共用 Fusion（論文說兩份） | 2.41 | 0.746 | pc 掉到 52，其餘同族 |
| P3 | P1 + 12 個視角 token 進 Fusion（不平均） | 2.49 | 0.767 | 同族 |
| P5 | 描述文字 desc_v1；query = 另一段描述 + 重取樣點雲 + 單視角 | 2.44 | 0.839 | 各模態都高，融合更接近 pc |
| P7 | P1 但關掉融合前 L2 | 2.41 | 0.788 | 同族 |
| P6 | P1 但每步隨機抽一個視角 | 2.48 | 0.795 | 同族 |

### 3. 最後評估：對 Table 1（w/o ESSGNN）的 14 格

D 協定：4,569 dev_val query 對 36,554 train gallery（`output/look/ARMS_TABLE.md`）。R@1：

| 版本 | text | image | pc | T+I | T+PC | I+PC | full | level | shape |
|---|---|---|---|---|---|---|---|---|---|
| **論文** | **13.8** | **11.7** | **75.1** | **17.2** | **44.5** | **45.8** | **51.7** | 0 | 0 |
| P1 | 11.6 | 29.7 | 66.6 | 67.5 | 95.6 | 77.8 | 98.1 | 0.59 | 0.41 |
| P3 | 10.4 | 24.6 | 61.1 | 59.7 | 94.6 | 72.7 | 98.0 | 0.56 | 0.40 |
| P4 | 12.0 | 25.0 | 52.3 | 58.5 | 94.4 | 65.8 | 98.0 | 0.54 | 0.41 |
| P5 | 14.3 | 49.4 | 88.5 | 59.5 | 92.8 | 94.0 | 95.5 | 0.66 | 0.40 |
| P6 | 12.8 | 28.8 | 67.3 | 67.1 | 96.9 | 77.3 | 98.2 | 0.58 | 0.40 |
| P7 | 9.5 | 36.1 | 69.1 | 64.6 | 95.6 | 81.7 | 98.5 | 0.61 | 0.44 |
| pilot10b | 58.0 | 84.6 | 78.8 | 96.5 | 99.6 | 94.1 | 100.0 | 0.91 | 0.57 |

level = 14 格 |ln(我們/論文)| 的平均；shape = 扣掉整體高低後只看七格的相對形狀。
R@5 與 C 協定（4,569 對 4,569）在同一檔。

### 4. 讀法與裁決

- 七個 arm 的 shape 都在 0.40～0.44，舊構法 0.57。改 Fusion 一份／兩份、token／平均、正規化開／關、視角固定／隨機，形狀都不動：**架構軸沒有一個能改變形狀**。
- 單模態格（text 11.6、pc 66.6）已經貼近論文；跑偏的是所有含 pc 的融合格，原因就是第一節第 4 點：query 的 pc 與 gallery 的 pc 幾乎同一份。
- **選定 P1** 為主線與 Stage 2 的父 checkpoint：只動一個變數的乾淨版本，text／pc 最接近論文。Kyzen 可改。
- 記錄為 UNRESOLVED、不阻擋 Stage 2：query 觀測分布（資料層）。

---

## 三、Stage 2 初步結果

### 1. 資料與配方

| 項目 | 值 |
|---|---|
| 場景 | ProcTHOR-10k train 的前 1,500 間（全量 9,600 間尚未跑） |
| 樣本 | 99,945 個（房間內的物件，每個當一次 query） |
| gallery | 1,439 個 ProcTHOR 資產，由 P1 編碼 |
| 凍結 | gallery 塔、ULIP 編碼器（PAPER FACT） |
| λ 初值 | 0.1 × median‖Fusion‖ = 93.46 |
| 場景 dropout | 每 batch 30%（PAPER FACT） |
| 配方 | 先導：Stage 1 配方（平坦 5e-4）；S2-C／S2-D：lr 5e-5、warmup 10%、cosine 到 1e-6、1 epoch（`workflow/stage2_hyperparameters_ft_lr5e-5.json`） |
| 小批次 | 少於 8 個樣本的尾巴 batch 丟掉（303 個，583 樣本） |

### 2. 四次跑的讀數

「ProcTHOR」= 在 1,500 間場景裡找精確資產的 R@1；「w/ ESSGNN C」= Table 1 w/ ESSGNN 列，Stage 2 query 頭疊在 P1 上、layout 關閉、C 協定，七格 R@1。

| arm | query 給什麼 | 配方 | 終點 loss | ProcTHOR S1 / S2-off / S2-on | w/ ESSGNN C（text/image/pc/T+I/T+PC/I+PC/full） |
|---|---|---|---|---|---|
| 先導 1 | 全部 T/I/P | 平坦 5e-4 | — | — | INVALID：builder 漏帶 prefusion_norm、尾巴 batch 退化 |
| 先導 2 | 全部 T/I/P | 平坦 5e-4 | — | 82.4 / 24.2 / 23.5 | 10.1 / 15.2 / 49.2 / 22.2 / 56.4 / 50.5 / 58.9 |
| S2-C | 只給文字（Figure 1 的 query 形式） | 5e-5 warmup cosine | 1.08 | 10.3 / 12.4 / 12.2 | 22.5 / 37.7 / 74.3 / 62.8 / 86.1 / 80.1 / 88.2 |
| S2-D | 全部 T/I/P | 5e-5 warmup cosine | 1.08 | 82.4 / 36.8 / 32.8 | 24.9 / 36.1 / 71.2 / 58.2 / 80.9 / 73.0 / 80.1 |

P1 父在 C 協定：34.7 / 56.9 / 86.1 / 87.6 / 99.0 / 92.7 / 99.7。
論文 w/ ÷ w/o = 0.82～0.93；S2-C ÷ 父 = 0.65～0.88；S2-D ÷ 父 = 0.63～0.83。

### 3. 讀法與選定

- λ 三次都不動（93.46 → 93.43）、S2-on ≈ S2-off：layout 在「精確找同一個資產」這件事上沒得幫忙。這跟論文 w/ < w/o 方向一致。
- 掉多少由配方決定：平坦 5e-4 把 layout-free 頭打壞（÷父 0.25～0.59）；5e-5 warmup cosine 把損傷縮到論文的量級。
- **S2-C 為 Stage 2 主線候選**（不是定案）：query 只給文字最貼 Figure 1，數字最接近論文的 w/ 列。

### 4. 待驗證與待決定

待驗證（有 arm 或排程）：
1. 全量 9,600 間 ProcTHOR 跑一次 S2-C。
2. iterative-prefix 對 leave-one-out（正文 vs 附錄 C 的矛盾，目前走附錄版）。
3. ESSGNN 正文字面版（距離、層選擇、串接三處）。
4. pooling／λ₀ 掃描。
5. Stage 1 延後項：Transformer 內部掃描、CLIP 最後 N 層解凍、三個 seed。

待 Kyzen 決定：
1. ProcTHOR 切分：現在 80/20（9,600/2,400，論文寫法）還是官方 10k/1k/1k。
2. 是否用 GPT-4o 當場景層評審。
3. 最終評估是否 `--unseal`。
4. P1 是否維持為 Stage 2 父。
5. Gemma 對 GPT-4o 的標註比較需要 GPT-4o。

---

## 出處

- ULIP-2 論文：`docs/paper/ulip2_source/ulip2_arxiv_v4.html`（Table 10；§3 架構）。
- ULIP 官方零樣本程式：`/home/kyzen/upstream/ULIP/main.py` 第 41、362–405 行。
- 零樣本結果：`output/look/exp_ulip2_zero_shot_lvis.json`。
- Stage 1 14 格與 ULIP row：`output/look/ARMS_TABLE.md`；幾何：`output/look/exp_observation_geometry.json`。
- 訓練紀錄：各 overlay 的 `outputs/logs/train_stage1.jsonl`、`train_stage1_dev_val.jsonl`。
- Stage 1 審計：`docs/audit/STAGE1_FRESH_AUDIT_20260903.md`；Stage 2 審計：`docs/audit/STAGE2_FRESH_AUDIT_20260904.md`；裁決：`workflow/STAGE1_RESOLUTION_PLAN_20260903.md` §5–6。
- Stage 2 arm 產物：`metafind_data_attrs/outputs/checkpoints/stage2_arms/`。
