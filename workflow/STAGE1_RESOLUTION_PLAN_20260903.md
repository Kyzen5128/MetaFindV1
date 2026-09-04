# Stage 1 逐項裁決計畫（2026-09-03 晚間）

Kyzen：「這些評估你都應該重視……逐步不排除及修正。」

最高權威：`docs/paper/metafind_source/metafind_arxiv_v1.html`（`arXiv:2510.04057v1`，檔頭核對過）。

五個標籤，照 Kyzen 的整理稿：**PAPER FACT / PAPER FIGURE FACT / PAPER-CONSTRAINED INFERENCE / IMPLEMENTATION CHOICE / UNRESOLVED**；特殊標記 **DIRECT DEVIATION / PAPER-INTERNAL CONTRADICTION / INVALID EXPERIMENT**。

## 0. 裁決規則

- 每個 UNRESOLVED 項目對應**至少一個實驗 arm**；沒有 arm 的項目不准寫成「已排除」。
- 每個 arm 用**同一套評估器**（`metafind.eval.run_retrieval`，float64 cosine，同分算輸）跑協定 C（4,569）與 D（36,554），**七個條件 × R@1 + R@5 = 14 格**一起看，不單看 text R@1。
- 一個 arm 只動一個變數（相對於它的父 arm）；動兩個的要有理由寫在表裡。
- 「比較接近論文」只能寫成「14 格距離變小」，不能寫成「找到論文 protocol」。
- 每個 arm 有自己的資料根目錄（`/home/kyzen/metafind_data_<tag>`），原資料不動。

## 1. 已有 arm 的項目

| # | 項目（整理稿編號） | 標籤 | 父 arm → 子 arm | 狀態 | 目前讀數（協定 D，R@1） |
|---|---|---|---|---|---|
| P0-1 | 文字字串化（15） | IMPLEMENTATION CHOICE，高敏感 | pilot10b（v2_cm 長描述）→ **P1**（attrs_v1 純填表）→ **P5**（desc_v1 一句描述） | P1 ✅、P5 排隊 | text 58.0 → 11.6（論文 13.8） |
| P0-2 | 11 視角 → e_img（14） | UNRESOLVED | 12 視角平均（pilot10b）→ **P1** query 單一視角 → **P3** 12 視角當 12 token → **P6** 每步隨機視角（ULIP-2 自己的做法） | ✅ 全部 | image 84.6 → 29.7（P1）→ 24.6（P3）→ 28.8（P6）；論文 11.7。三種視角構法同族 |
| P0-3 | Q/G 同一份 vs 獨立觀測（17） | UNRESOLVED；「三個都同一份」為 **EXPERIMENTALLY DISFAVORED**（實測 full 0.9998、cos 0.9989；Eq. 5 只是動機，不是定理——GPT 修正，採納） | P1（只有影像獨立）→ **P5**（三個都「第二份」：第二名描述、重取樣點雲、單一視角） | ✅ P5 是 **independent-observation stress test**，不是 reproduction candidate（GPT 命名，採納） | P5 D：14.3/49.4/88.5/59.5/92.8/94.0/95.5。text 對上了（14.3 vs 13.8），但 **geometry 探針證明「第二份」在 embedding 空間裡幾乎不獨立**：釋出編碼器下配對 cos text 0.80 / image 0.93 / pc 0.99，單模態 raw 檢索 R@1 text 50.4 / image 89.0 / pc 97.7。重取樣點雲只是換了 bytes，沒換難度。交互比 T+PC/PC 1.05、Full/PC 1.08（論文 0.59 / 0.69）—— 加模態仍不會變差 |
| P0-4 | ULIP baseline 畫廊構法（24） | PAPER-CONSTRAINED INFERENCE | B1 純 PC 畫廊 vs B2 三模態平均畫廊；raw mean vs L2 mean；文字四種；影像三種 | ✅ 全量完 | 只有 B1 重現「加了 text/image 反而變差」的**單調形狀**；但 T+PC 98.7 對論文 33.9 —— query 點雲同一份時無法壓下來 |
| P0-5 | 評估畫廊範圍（25） | UNRESOLVED | 評估器四協定：A test→test 9,138、B test→full 45,692、C dev→dev 4,569、D dev→train 36,554 | ✅ 機制齊；A/B 只在最終上鎖時 `--unseal` | 每個 arm 同時報 C、D |
| P0-6 | 進 Fusion 前正規化（C8） | IMPLEMENTATION CHOICE | P1（開）→ **P7**（關，其餘同 P1，唯一對照） | ✅ | P7 D：9.5/36.1/69.1/64.6/95.6/81.7/98.5 對 P1 11.6/29.7/66.6/67.5/95.6/77.8/98.1 —— 同族（shape 0.44 vs 0.41）；正規化不是決定性的軸，主線保留開 |
| — | Fusion 一份或兩份（3 / C2） | PAPER FIGURE FACT（圖畫一個）vs 正文 "separate encoders" | P1（兩份）→ **P4**（一份共用） | ✅ | P4 與 P1 七格同形（text 12.0 vs 11.6、pc 52.3 vs 66.6）→ 不是拉開差距的軸 |
| — | Fusion 輸入粒度（13 / C4） | UNRESOLVED：Eq. 6 寫每模態一支，Figure 1 畫 K 支 | P1（一支）→ **P3**（12 token） | ✅ | P3 D：10.4/24.6/61.1/59.7/94.6/72.7/98.0，與 P1 同族（shape 0.41 vs 0.41）→ 不是拉開差距的軸；主線維持每模態一支 |
| — | 11 vs 12 視角（44） | DIRECT DEVIATION | 11-of-12 兩邊都換（評估敏感度） | ✅ | 七格不動（57.8/84.7/78.4/96.3/99.6/94.2/100.0 對 12 視角 58.0/84.6/78.8/96.5/99.6/94.1/100.0）→ 不重渲染 |
| — | ULIP 只餵類別（16） | UNRESOLVED（論文兩列都沒寫餵什麼） | 評估敏感度：類別 / 填表 / 描述 / 完整 | ✅ | text 3.8 / 4.7 / 24.1 / 24.5（論文 0.1）；T+PC 全 98.7（論文 33.9）→ 文字解釋一格、解釋不了形狀 |
| — | 早期壞探針（46） | INVALID EXPERIMENT | 撤回；`tests/test_probe_gallery_parity.py` 守住 | ✅ | 四個結論全在正式評估器上重測 |

## 2. 還沒有 arm 的項目（依序補）

| # | 項目 | 標籤 | 要做的 arm | 排程 |
|---|---|---|---|---|
| P0-4′ | ULIP baseline 的 query 是不是第二份觀測 | 兩個候選並列（GPT 修正，採納）：(a) query pc / image 與 gallery 不同；(b) pc 同一份，但論文的 text / image 觀測或 scorer 足以翻掉 pc 排名 | ULIP 探針加 query pack | ✅ 量完：pack 當 query（重取樣 pc、第二名描述、單視角）→ 24.0/46.1/**97.7**/48.9/96.8/96.6/94.6。pc-only 正好落在論文的 97.9（同一份時是 100.0），但 T+PC 仍 96.8 —— **換點雲不會讓 T+PC 掉**，(a) 單獨不夠 |
| P0-A/B/C/D | ULIP 列：scorer 2×2、影像單視角 vs 平均、PC-only 校準、margin decomposition | GPT 提出，採納；零訓練 | `tools/probes/exp_ulip_scorer_margin.py` | ✅ **P0-C 命中**：query pc 同一份時 **raw dot 給 PC-only 97.9**（cosine 100.0），正好是論文的 97.9 —— scorer = 點積是一個站得住的讀法。但 dot 下 T+PC 仍 94.3（論文 33.9）。**P0-D**：pc 同一份時，我們的文字只能翻掉 0.7% 查詢、影像 0.6–0.9%、兩者合計 ≤ 2.1%（pc top-1 margin 中位 0.215）→ 我們的 T/I 觀測沒有能力製造論文的形狀；論文的 query 文字／影像對自己資產的辨識力必須遠低於我們的，或 query 的 pc/image 與 gallery 不同。兩個候選仍並列。完整 16 列在 `output/look/exp_ulip_scorer_margin.json` |
| P0-3′ | 三個模態各自獨立的貢獻 | UNRESOLVED | P5 若不等於論文：拆成「只有文字獨立」「只有點雲獨立」兩個 arm | 看 P5 結果 |
| P1-7 | Transformer 內部（12） | IMPLEMENTATION CHOICE × 6 | 在最好的觀測構法上掃：層數 1/2/4、讀出 mean/CLS、缺席 slot 排除 | 觀測構法定案後 |
| P1-8 | Stage 1 訓練範圍（20–22） | UNRESOLVED（§3.4 "earlier layers adapt"） | CLIP 文字／影像塔最後 N 層解凍（全解凍 AdamW 狀態 ~30 GB，卡不下） | 觀測構法定案後；需先量記憶體 |
| P1-9 | Stage 2 leave-one-out vs iterative-prefix（43） | UNRESOLVED | 兩個 arm | Stage 1 定案後 |
| — | ProcTHOR 家具數：論文 "more than 3,000 unique assets" | **論文數字沒有上游支持**：ProcTHOR 論文摘要 / §3 / 附錄 B.8.1 皆為 **1,633 assets, 108 types**；生成器 repo `asset-database.json` 1,653 / 109；ProcTHOR-10K 的 12,000 間房實際用到 **1,467**（含門窗 1,528）；AI2-THOR 官網 iTHOR "over 2000 unique objects"、RoboTHOR "600+"。3,000 對不到任何一個。資料下載完整（10,000 / 1,000 / 1,000 與 ProcTHOR 論文 §4 一致），從資料榨不出 3,000 | 記錄為 PAPER STATEMENT WITHOUT UPSTREAM SUPPORT；用 1,467 | — |
| — | ProcTHOR 切分：論文 80/20 vs 官方 10k/1k/1k | UNRESOLVED | Kyzen 決定：照官方或重切 | Stage 2 前 |
| P1-10 | Stage 2 query 構法（42） | UNRESOLVED | 完整 T/I/P vs 文字為主 | Stage 1 定案後 |
| P1-11 | ESSGNN 正文 vs 附錄（40） | PAPER-INTERNAL CONTRADICTION | Method-literal / Proof-consistent 兩個 arm | Stage 1 定案後 |
| P1-12 | ESSGNN pooling / λ₀（41） | UNRESOLVED | sum/mean/attention × λ₀ | Stage 1 定案後 |
| — | Gemma vs GPT-4o（45） | DIRECT DEVIATION | 300–500 筆 GPT-4o 對照 | 需要 GPT-4o 存取；Kyzen 提供才做 |
| — | τ 是否可學（10） | PAPER-CONSTRAINED INFERENCE：「is 0.5 for all experiments」→ 固定 | 不做 arm；若做，標 ablation | — |

## 3. 已定的 PAPER FACT（不做 arm）

雙塔（1）、兩塔都用 ULIP-2（2）、`ULIP-2 (Shared)` 圖示（3）、w/o ESSGNN = Stage 1（4）、gallery 三模態齊（5）、query 任意子集（6）、30% 獨立遮罩（7）、不補零（8）、Eq. 5 單向（9）、τ = 0.5（10）、Transformer fusion（11）、兩塔都訓（19）、full encoder fine-tuning 優於 fuser-only（20）、Stage 2 Eq. 6–8、λ 可學、只訓 query fusion + ESSGNN、30% 場景 dropout、**w/ ESSGNN 的 Table 1 在「無 layout、ESSGNN 關閉」下評**（§3.2 原文："evaluating on Objaverse-LVIS (which lacks layout and disables ESSGNN)"；同段："Using the Stage-1 head reproduces the 'w/o ESSGNN'"）。

## 4. 執行順序（自動鏈）

```
P1 ✅ → P4 ✅ → P3 ✅ → P5 ✅ → geometry 探針 ✅ → P7 ✅ → P6 ✅ → ULIP scorer/margin 探針 ✅ → **Stage 1 架構與評估方式定案（見 §5）→ 進 Stage 2**
```

所有 arm 的 14 格表：`output/look/ARMS_TABLE.md`（`tools/fingerprint.py` 產生，含到論文列的距離）。

## 5. Stage 1 裁決（2026-09-04 凌晨）

**架構與評估方式：確認無誤。** 七個 retrain arm（pilot10b、P1、P3、P4、P5、P6、P7）在同一套評估器上的 14 格 shape 全在 0.40–0.44（舊構法 0.57）；改 Fusion 一份／兩份、token／平均、正規化開／關、視角固定／隨機，七格都是同一族。所有 PAPER FACT 項目逐項對過（§3）。**沒有任何一個架構軸能改變形狀。**

**差距的來源：query 觀測與 gallery 太像。** 三個「第二份」在釋出編碼器下配對 cos 0.80 / 0.93 / 0.99；pc 同一份時我們的 T/I 只能翻掉 ≤ 2% 的查詢（論文形狀需要約 2/3）。這是**資料層**的問題（query 的文字／影像／點雲怎麼來），Kyzen 已裁定資料集調整延後。記錄為 UNRESOLVED（P0-3、P0-4′），不阻擋 Stage 2。

**一個新的 PAPER-CONSTRAINED INFERENCE**：scorer 用 raw dot 時 PC-only 正好 97.9（cosine 100.0）；記錄，不改主線（cosine 是 Stage 1/2 一致的選擇；改 scorer 要連訓練一起改，另開 arm）。

**Stage 2 的 Stage 1 父 checkpoint：P1**（attrs_v1 + 單一視角 + pre-fusion L2；text 11.6 / pc 66.6 最接近論文，一次只動一個變數的乾淨主線；MASTER 選定，Kyzen 可改）。


### 5a. 代數不是原因（2026-09-04 07:55，P1e25）

Kyzen 問「會不會是訓練不夠」。P1 設定完全不變，只把 epochs 10 改 25（cosine 也拉到 25）。最佳 epoch 24，dev_val 平均 R@1 0.778（10 代版 0.795）。

| arm | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| 論文 | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| P1（10 代）D | 11.6 | 29.7 | 66.6 | 67.5 | 95.6 | 77.8 | 98.1 |
| P1e25（25 代）D | 12.1 | 26.1 | 65.4 | 61.6 | 95.7 | 75.7 | 97.8 |

形狀不變，含 pc 的融合格仍 95+。裁決：代數定 10；250 不跑（論文沒寫代數，250 是 ULIP main.py:23 的預設）。
出處：`/home/kyzen/metafind_data_attrs/outputs/checkpoints/pilotP1e25_20260904/`、`outputs/eval/eval_pilotP1e25_20260904/`、`outputs/logs/chain_P1e25.log`。


### 5b. 學習率掃描（2026-09-04 10:30；P1 設定不變，只換 lr，各 10 代）

選法事先寫死：dev_val 七格 R@1 平均最高者勝（不看論文、不看測試集）。

| lr | best epoch | dev_val mean R@1 | D：text / image / pc / T+I / T+PC / I+PC / full |
|---|---|---|---|
| 1e-4 | 9 | **0.830** | 11.1 / 40.5 / 92.1 / 63.6 / 98.5 / 92.9 / 99.0 |
| 5e-4（P1） | 9 | 0.795 | 11.6 / 29.7 / 66.6 / 67.5 / 95.6 / 77.8 / 98.1 |
| 1e-3 | 9 | 0.748 | 9.1 / 27.1 / 43.8 / 59.2 / 93.0 / 70.6 / 97.7 |
| 3e-3（ULIP 官方值，batch 512） | 9 | 0.531 | 21.5 / 8.4 / **0.0** / 51.4 / 48.7 / 16.1 / 90.4（Point-BERT 崩掉） |
| 論文 | | | 13.8 / 11.7 / 75.1 / 17.2 / 44.5 / 45.8 / 51.7 |

裁決：**lr = 1e-4**（dev_val 規則）。附註：lr 越低 pc 越接近釋出 ULIP-2（92.1，離論文 75.1 更遠），形狀四個 lr 全部相同（full > pc），所以 lr 不是形狀的來源；3e-3 在單卡 batch 64 下把點雲塔訓壞，ULIP 官方 3e-3 是 8 卡 batch 512 的值。
出處：`metafind_data_attrs/outputs/checkpoints/pilotP1_lr{1e-4,1e-3,3e-3}_20260904/`、`outputs/eval/eval_pilotP1_lr*`、`outputs/logs/chain_lr_sweep.log`。

P8 已用 lr 1e-4 開跑（query 三模態全換第二份觀測；image 改 held_out_view，把 query 那張從 gallery 平均拿掉）：`outputs/logs/chain_P8.log`。


### 5c. P8：query 三模態全換第二份觀測（2026-09-04 11:50）— 形狀仍未翻

設定：desc_v1 另一段描述（query pack）、held_out_view（query 那張從 gallery 12 張平均拿掉）、點雲單邊掃描（`--query-pc-perturb half`）、lr 1e-4、10 代。最佳第 9 代，dev_val 平均 0.829。

| | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| 論文 | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| P1（5e-4）D | 11.6 | 29.7 | 66.6 | 67.5 | 95.6 | 77.8 | 98.1 |
| P8 D | 19.2 | 55.4 | 69.5 | 66.3 | 81.0 | 86.5 | 90.7 |

讀法：融合格有下來（T+PC 95.6 → 81.0，full 98.1 → 90.7），pc 69.5 接近論文 75.1；但 full 仍高於 pc，image 反而升到 55.4（模型學會對 held-out 視角）。「同一物件的不同觀測」不足以做出論文形狀。

**新假設（由論文 ULIP 列與 Figure 1 推得）**：ULIP 列 text 0.1、image 0.1 是亂猜水準 → 論文 query 的文字、圖片不指向該實例，只指向類別（Figure 1 的文字 query 是 `Platform Bed {size:…}`）。只有點雲是該資產自己的。若成立，T+PC 從 97.9 掉到 33.9 就有解釋：文字／圖片把 query 拉向同類別的別的資產。
檢驗（不訓練）：`tools/probes/exp_ulip_row_category_query.py`，query 文字 = 類別名、圖片 = 同類別另一個資產的一張視角、pc = 自己的，用釋出 ULIP-2 直接平均對 ULIP 列。排在 AMP 量測之後自動跑。


### 5d. AMP bf16 量測（2026-09-04 12:05；P1 配方，lr 1e-4，100 步，RTX 5090 32 GB）

| | 峰值記憶體（allocated） | 每步 | 100 步 loss | dev_val 平均 |
|---|---|---|---|---|
| float32（現行） | 23.8 GiB | 0.74 s | 2.649 | 0.643 |
| bf16 autocast | 17.3 GiB | 0.50 s | 2.656 | 0.639 |

省 27% 記憶體、快 1.5 倍，100 步內 loss／dev_val 差在雜訊內。ULIP 官方訓練本來就開 amp.autocast（main.py），所以開 bf16 是向上游靠，不是偏離。`--amp bf16` 已進 arm hash；預設仍 off，等 Stage 1 形狀定案再切主線。
出處：`metafind_data_attrs/outputs/logs/chain_amp.log`、`checkpoints/smoke_amp_{off,bf16}_20260904/`（debug run，非科學結果）。


### 5e. 類別層級 query 假設：解釋得了 0.1，解釋不了 33.9（2026-09-04 12:10）

釋出 ULIP-2、直接平均、gallery = dev_val 4,569 個 pc 向量。query 文字換類別名、圖片換同類別另一個資產的一張視角、pc 用自己的：

| variant | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| 論文 ULIP 列 | 0.1 | 0.1 | 97.9 | 0.0 | 33.9 | 22.6 | 6.4 |
| 自己的文字／自己的視角／自己的 pc | 18.9 | 69.9 | 100.0 | 62.0 | 99.9 | 99.9 | 99.8 |
| 類別名／自己的視角／自己的 pc | 9.8 | 69.9 | 100.0 | 60.0 | 100.0 | 99.9 | 99.8 |
| 類別名／別的資產視角／自己的 pc | 9.8 | 4.7 | 100.0 | 6.5 | 100.0 | 99.9 | 99.6 |
| 類別名／別的資產視角／重取樣 pc | 9.8 | 4.7 | 99.7 | 6.5 | 99.6 | 99.6 | 99.2 |

讀法：類別層級的文字與圖片確實把單模態格拉到接近論文（9.8、4.7 對 0.1），**但含 pc 的融合格一格都沒掉**（T+PC 100、full 99.6）。原因同 §5 的幾何：只要 query 的 pc 是那個資產的（cos ≈ 1，最像的別人 0.59），任何文字／圖片平均進去都翻不了名次。論文 ULIP 列 pc 97.9 卻 T+PC 33.9，在「單位向量平均」下**沒有任何 query 觀測做得出來**。

裁決：Stage 1 的形狀追逐到此停止。論文有寫的全部照做（§3、§5 表）；論文沒寫的 query 取得方式，用五種假設（同一份、不同觀測、破壞點雲、三模態第二觀測、類別層級）逐一檢驗，沒有一種能同時給出 pc ≈ 75～98 與 T+PC ≈ 34～45。記為 **UNRESOLVED：論文未指定的 query 協定**。下一步由 Kyzen 決定：問作者、或接受此差異進 Stage 2。
出處：`tools/probes/exp_ulip_row_category_query.py`、`output/look/exp_ulip_row_category_query.json`。


### 5f. 論文形狀第一次出現：gallery = 三模態平均 ＋ query 文字／圖片來自同類別的另一個資產（2026-09-04 12:40）

根據論文兩句話重讀 ULIP 列的協定：§3.1「a simple mean pooling layer to aggregate available modalities, and use these fused embeddings to retrieve from a **pre-encoded gallery**」；§3.2 基線的 PC-only「retrieval using **identical embeddings for both query and gallery**」。gallery 不必是 pc 向量，可以是資產三模態的平均（fused embedding）。釋出 ULIP-2、不訓練、dev_val 4,569：

| gallery | query | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|---|
| 論文 ULIP 列 | | 0.1 | 0.1 | 97.9 | 0.0 | 33.9 | 22.6 | 6.4 |
| pc | 自己的三份 | 18.9 | 69.9 | 100.0 | 62.0 | 99.9 | 99.9 | 99.8 |
| mean3(raw) | 自己的三份 | 71.4 | 92.3 | 99.3 | 99.0 | 99.9 | 99.6 | 100.0 |
| mean3(L2) | 同類別另一資產的文字、視角 + 自己的 pc，raw mean | 8.9 | 4.2 | 99.8 | 4.4 | 94.0 | 72.5 | 34.6 |
| **mean3(raw)** | **同類別另一資產的文字、視角 + 自己的 pc，raw mean** | **8.8** | **4.2** | **99.3** | **4.4** | **87.7** | **55.3** | **24.0** |

**這是第一次在任何設定下出現論文的順序**：pc ≫ T+PC > I+PC > full ≫ text ≈ image。機制：gallery 是三模態融合向量，query 帶進來的「別的資產」的文字／圖片會精準對上那個資產在 gallery 裡的文字／圖片分量，把 query 拉向它；pc 分量只佔三分之一，壓不住。
差距仍在（T+PC 87.7 對 33.9、full 24.0 對 6.4）：候選解釋是「別的資產」不限同類別（任意資產）、gallery 更大（論文 test 9,138）。正在跑任意資產變體。
含意：MetaFind 自己那列（text 13.8、T+I 17.2 > 兩個單模態、full < pc）也與「query 的文字／圖片不是該資產的、pc 是」一致。若成立，Stage 1 的 query 構法要改成跨資產的文字／圖片（資料集改動，待 Kyzen）。
出處：`tools/probes/exp_ulip_row_category_query.py`、`output/look/exp_ulip_row_category_query.json`、`data/outputs/logs/exp_ulip_row_category_query2.log`。


### 5g. 任意資產的文字／圖片 + 自己的 pc：ULIP 列七格全部對上量級（2026-09-04 12:55）

gallery = mean3(raw)，query raw mean，dev_val 4,569：

| query 文字／圖片來源 | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| 論文 ULIP 列 | 0.1 | 0.1 | 97.9 | 0.0 | 33.9 | 22.6 | 6.4 |
| 同類別另一資產 | 8.8 | 4.2 | 99.3 | 4.4 | 87.7 | 55.3 | 24.0 |
| **任意資產（不限類別）** | **0.0** | **0.0** | **99.3** | **0.0** | **64.9** | **36.7** | **0.2** |
| 文字同類別、圖片任意 | 8.8 | 0.0 | 99.3 | 2.5 | 87.7 | 36.7 | 79.5 |

讀法：ULIP 列的 0.1 / 0.1 / 0.0 三格只有「文字、圖片與目標無關」做得出來；T+PC 64.9 對 33.9、I+PC 36.7 對 22.6 是同一量級（gallery 4,569 對論文 9,138，更大的 gallery 會再壓低）。**ULIP 列的協定至此可重現到量級**：gallery = 三模態 raw 平均、query 的文字／圖片不是目標資產的、pc 是。
對 MetaFind 自己那列的含意：text 13.8 / image 11.7 不是 0，所以 MetaFind 的 query 文字／圖片帶有目標的類別資訊 → 「同類別另一資產」版本（8.8 / 4.2 的量級）；訓練後可望到 13.8 / 11.7；含 pc 的融合格會被別的資產拉低 → full < pc。這就是 P9 的構法。
兩列的 query 來源不一定相同（基線與 MetaFind 各有評估管線），論文沒寫；記 INFERENCE。
出處：`data/outputs/logs/exp_ulip_row_category_query3.log`、`output/look/exp_ulip_row_category_query.json`。


### 5h. P9：訓練與評估都用「同類別另一資產的文字／圖片 + 自己的 pc」（2026-09-04 13:43）— 模型學會忽略文字圖片

Kyzen ✅ 後開跑。P1 構法、lr 1e-4、10 代、`--query-partner same_category`。最佳第 8 代，dev_val 平均 0.583。

| | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| 論文 | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| P9 D | 0.9 | 0.4 | 90.7 | 0.9 | 93.6 | 92.1 | 93.4 |

讀法：訓練時文字／圖片對「找哪一個」沒有資訊，模型就把它們**整個丟掉**，只靠 pc：單模態 text／image 掉到接近 0（論文 13.8 / 11.7），含 pc 的四格全等於 pc（93）。論文的形狀（full < pc，text 13.8）要求模型**有在用**文字圖片、而且被它們拉偏；一個在「文字圖片無資訊」下訓出來的對比模型不會這樣。
→ 新假設：**訓練用資產自己的文字／圖片（有資訊），測試用別的資產的（誤導）**。訓練讓 Fusion 學會信任文字圖片，測試時它們就把 query 拉走。不用訓練即可驗：拿 P1（訓練用自己的文字／單張圖／自己的 pc）在測試時改用 partner query 評 C、D（`eval_P1_testpartner_20260904`，跑中）。
出處：`metafind_data_attrs/outputs/checkpoints/pilotP9_partner_same_category_lr1e-4_20260904b/`、`outputs/eval/eval_pilotP9_*`、`outputs/logs/chain_P9.log`。


### 5i. 找到了：訓練用自己的紀錄、測試用別人的文字／圖片，MetaFind 列的形狀翻過來（2026-09-04 13:51）

P1 checkpoint 不動（訓練：自己的文字、單張圖、自己的 pc），評估時 query 文字／圖片改成同類別另一資產的（`--query-partner same_category`），pc 自己的：

| | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| 論文 | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| P1，測試 partner，D | 1.0 | 0.4 | 66.6 | 0.5 | **53.6** | **50.5** | **28.9** |
| P1，測試 partner，C | 5.3 | 1.4 | 86.2 | 1.1 | 74.7 | 75.1 | 50.4 |
| P1，測試自己的（原本）D | 11.6 | 29.7 | 66.6 | 67.5 | 95.6 | 77.8 | 98.1 |

**第一次在訓練過的 MetaFind 模型上出現論文的順序**：pc > T+PC ≈ I+PC > full。T+PC 53.6 對 44.5、I+PC 50.5 對 45.8，同一量級；full 28.9 對 51.7。機制：模型在訓練時學會信任文字／圖片（因為它們是目標的），測試時它們換成別人的，就把 query 拉走。P9（訓練時也用別人的）反而讓模型丟掉文字圖片，證明論文不是那樣訓的。
還差的：text 1.0 / image 0.4 對論文 13.8 / 11.7 → 論文 query 的文字／圖片帶有目標的**類別層級**資訊但不指向實例（Figure 1 的 `Platform Bed {size:…}` 正是這種：目標的類別＋尺寸欄位，沒有描述）。下一步（不訓練）：測試時 query 文字改成目標自己的「類別＋尺寸」填表句、圖片維持同類別別人的，看 text 單格會不會升到 13.8 而 full 仍 < pc。
裁決：**融合格偏高的原因確定**——訓練協定（兩塔都讀目標自己的紀錄）與論文一致，差的是**評估時 query 的文字／圖片來源**。這是評估協定，不是模型。
出處：`metafind_data_attrs/outputs/eval/eval_P1_testpartner_20260904/`、`outputs/logs/eval_P1_testpartner.log`。


### 5j. 測試時 query 文字改「目標的類別＋尺寸」（Figure 1 格式），圖片用同類別參考圖（2026-09-04 14:25）

P1 不動，D 協定，pc = 目標自己的：

| query（文字 | 圖片） | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|---|
| 論文 | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| 自己的填表句 | 自己的圖（原評估，對照） | 11.6 | 29.7 | 66.6 | 67.5 | 95.6 | 77.8 | 98.1 |
| 別人的填表句 | 別人的圖（5i，對照） | 1.0 | 0.4 | 66.6 | 0.5 | 53.6 | 50.5 | 28.8 |
| **目標「{類別} {size: w x l x h cm}」 | 別人的圖** | 3.3 | 0.4 | 66.6 | 1.1 | 74.9 | 50.5 | **55.2** |
| 目標「{類別}」 | 別人的圖 | 1.7 | 0.4 | 66.6 | 1.2 | 70.2 | 50.5 | 50.4 |
| 目標「{類別} {size}」 | 自己的圖 | 3.3 | 29.7 | 66.6 | 42.4 | 74.9 | 77.8 | 83.4 |

讀法：full 55.2 對 51.7、I+PC 50.5 對 45.8 已在量級內；T+PC 74.9 對 44.5 仍高（目標自己的類別＋尺寸句跟 gallery 存的填表句仍相近，拉不夠）；text 3.3 對 13.8、image 0.4 對 11.7、T+I 1.1 對 17.2 仍低。兩種構法各對一半：「別人的文字」把 T+PC 拉到 53.6 但 text 只剩 1.0；「目標的類別句」text 3.3 但 T+PC 74.9。
論文的 query 文字要同時滿足「單獨有 13.8」與「加進 pc 掉到 44.5」，圖片要「單獨 11.7」但「加進 pc 掉到 45.8」：都是**關於目標、但不是 gallery 那份**的觀測。候選：Objaverse 原始 metadata 的資產名稱（Figure 1 的「Platform Bed」像 Sketchfab 名稱）＋尺寸；圖片用 Objaverse 官方縱圖（不同渲染管線的目標本人）。兩者都是「目標的、但跟 gallery 存的不同」。是否取得為 Kyzen 決定（縱圖需下載）。
出處：`tools/probes/exp_type_level_query.py`、`output/look/exp_type_level_query.json`。


### 5k. OpenShape 官方程式碼怎麼餵、怎麼檢索（2026-09-04 15:10，UPSTREAM FACT）

Clone：`/home/kyzen/upstream/OpenShape_code`（abe5aa4）、`/home/kyzen/upstream/openshape-demo`（HF Space）、`/home/kyzen/upstream/openshape-demo-support`（HF repo；`retrieval.py` 在 commit 70dbc29）。

訓練（`src/data.py:68-103`、`src/configs/train.yaml:35-49`）：文字每步從三種來源隨機挑一（metadata 名稱／標籤、BLIP 或 Azure 對圖生成的短句、檢索文字），特徵用 ULIP 64 模板平均；圖片 50% 用 **Sketchfab 縱圖**特徵、50% 用一張渲染圖特徵；點雲 10k、y-up、正規化、增強＋隨機 z 旋轉、50% 顏色塗灰 0.4；損失 pc↔text、pc↔image 雙向對比，logit scale 可學（`train.py:60-124`）；CLIP ViT-bigG 特徵事先算好凍結。
檢索 demo（`retrieval.py`）：gallery = 全 Objaverse 的 OpenShape **形狀向量**（`objaverse.pt`：`us`、`feats`）；query = CLIP 文字特徵／CLIP 圖片特徵／OpenShape 點雲特徵，單模態 cosine 排序。**沒有多模態融合**。
`objaverse_meta.json`（HF dataset `OpenShape/openshape-objaverse-embeddings`）每個 uid 有：`name`、`tags`、`cats`、`desc`、**`img`（Sketchfab 縱圖 URL）**、glb 路徑。這就是「關於目標、但不是我們 gallery 那份」的文字與圖片來源；MetaFind 的渲染照 OpenShape，Figure 1 的「Platform Bed」像 Sketchfab 名稱。
含意：Table 1 的 OpenShape 列與 ULIP 列一樣，是 MetaFind 自己加 mean pooling 算的；query 的文字／圖片最可能是 metadata 名稱與縱圖。


### 5l. 測試時 q_text 改 Sketchfab 名稱（P1 不動；2026-09-04 15:30）

| q_text ｜ q_image（pc 自己的） | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| 論文 | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| Sketchfab 名稱 ｜ 參考圖 | 0.8 | 0.4 | 66.6 | 0.6 | 57.7 | 50.5 | 42.5 |
| 名稱＋尺寸 ｜ 參考圖 | 1.5 | 0.4 | 66.6 | 1.1 | 60.6 | 50.5 | 46.4 |
| 名稱＋標籤 ｜ 參考圖 | 0.7 | 0.4 | 66.6 | 0.7 | 61.0 | 50.5 | 42.3 |
| 作者描述或名稱 ｜ 參考圖 | 0.6 | 0.4 | 66.6 | 0.6 | 55.4 | 50.5 | 41.2 |
| 名稱＋尺寸 ｜ 自己的圖 | 1.5 | 29.7 | 66.6 | 28.7 | 60.6 | 77.8 | 74.0 |

讀法：含 pc 的融合格全落在論文量級（T+PC 55～61 對 44.5；full 41～46 對 51.7）；但 text 單格 0.6～1.5 對 13.8。P1 的 Fusion 是用填表句訓的，沒見過 Sketchfab 名稱，所以把名稱當雜訊。論文 text 13.8 表示它的模型**認得**這種文字 → 訓練時 q_text 就是這類文字。
→ P10a：訓練與評估都用 q_text = Sketchfab 名稱（`--query-text-override`），q_image 維持單張自己的圖（縱圖待下載），q_pc 自己的；lr 1e-4、10 代。預期 text 單格升、T+PC 掉。名稱向量快取：`data/outputs/_probe/text_override/sketchfab_name{,_size}.npz`（凍結 CLIP，全 45,692）。
出處：`output/look/exp_type_level_query_sketchfab.json`、`data/outputs/logs/exp_type_level_query2.log`。


### 5m. Kyzen 的讀法：gallery = 完整描述、query = 欄位句（2026-09-04 14:35）

先在 P5（gallery 文字 = 描述 desc_v1）上不訓練直接換 q_text = 欄位填表句（attrs_v1 向量）：

| q_text ｜ q_image（pc 自己的；gallery 描述） | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| 論文 | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| 自己的描述 ｜ 自己的圖（P5 對照） | 17.8 | 49.4 | 89.2 | 65.0 | 96.3 | 94.7 | 97.9 |
| **欄位句 ｜ 參考圖** | 0.7 | 0.3 | 89.2 | 0.5 | 86.8 | 62.9 | 62.0 |
| 欄位句 ｜ 自己的圖 | 0.7 | 49.4 | 89.2 | 42.6 | 86.8 | 94.7 | 93.9 |

讀法：P5 的 Fusion 只看過描述，欄位句對它是雜訊（text 0.7），跟 5l 的 Sketchfab 名稱同一現象。**測試時換 q_text 無法判斷這個讀法**，要訓練時就用「gallery 描述、query 欄位句」讓 Fusion 學會跨寫法對應。→ P10：gallery 文字 = 描述（desc overlay）、q_text = 欄位句（attrs_v1 向量，`--query-text-override`）、q_image 單張自己的圖、q_pc 自己的；lr 1e-4、10 代。這是 Kyzen 提出的構法，也是 Figure 1（query 欄位）＋ Figure 2（gallery 描述）最直接的讀法。
出處：`output/look/exp_type_level_query_P5desc.json`、`data/outputs/logs/exp_type_level_query_P5.log`。

### 5n. OpenShape 論文的檢索怎麼做（arXiv 2305.10764v2，UPSTREAM FACT；2026-09-04 15:20）

§4.4 Cross-Modal Applications：「we retrieve 3D shapes … by calculating the cosine similarity between input embedding(s) and 3D shape embeddings and performing kNN」。輸入是**單一**圖片、文字或點雲；全部是**定性圖例**（Figure 11、12、14、15），**沒有 R@k**、沒有測試集。兩個點雲同時查詢的做法是 argmax_i min(h_i·h_a, h_i·h_b)，取「對兩者都近」，**不是平均**。附錄 6.1：圖片查詢的輸入圖來自 unsplash.com（真實照片，不是渲染圖）。附錄 6.3.1：Objaverse 的 raw text = 該 shape 的 **name**（Sketchfab 名稱），再經 GPT-4 過濾、BLIP／Azure 描述、LAION 檢索文字補強。
含意：(1) MetaFind Table 1 的 OpenShape 列不可能來自 OpenShape 官方碼，是 MetaFind 自己加 mean pooling 算的；(2) 上游做圖片查詢時用真實照片而非渲染圖，與「MetaFind 的 q_image 不是渲染圖」一致；(3) 上游的 raw text 就是名稱，支持 Figure 1「Platform Bed」= 名稱的讀法。
檔案：論文 HTML 放在 docs/paper 下的 openshape_source 目錄。


### 5o. 網路查證：NeurIPS 2025 海報（2026-09-04 15:40，PAPER FIGURE FACT）

來源：https://neurips.cc/media/PosterPDFs/NeurIPS%202025/115513.png（存檔 `docs/reference/metafind_neurips2025_poster.png`）。OpenReview 論壇與 PDF 被驗證頁擋住，抓不到審稿意見；arXiv 只有 v1；沒有程式碼連結。

兩個新事實：
1. **gallery 文字是標註 JSON 本身**。海報右上「Structured Detailed Description」印的是 `{"annotations": {"category": "robot", "synset": "robot.n.01", "width": 30, "length": 30, "height": 40, "volume": 36000, "mass": 2.5, "description": "A small cubic-shaped robot …", "materials": ["metal","glass","plastic"], "onCeiling": false, "onWall": false, "onFloor": true, "onObject": true}}`。這串超過 CLIP 的 77 token（我們的資產約 120 token），CLIP 會截尾。
2. **query 文字是 `Platform Bed (size: ……)`**（框架圖），只有類別＋尺寸。q_image 畫的是一張床的渲染圖，q_pc 一朵點雲。

Table 1 全表（海報版，多了幾列基線）：

| Method | Text | Image | PC | T+I | T+PC | I+PC | T+I+PC |
|---|---|---|---|---|---|---|---|
| ULIP | 0.1/0.9 | 0.1/1.3 | 97.9/99.4 | 0/0.3 | 33.9/58 | 22.6/41.6 | 6.4/15.9 |
| OpenShape | 0.6/1.7 | 0.3/1.1 | 98.4/99.7 | 0/0.5 | 35.1/61.4 | 25.0/44.3 | 7.0/17.2 |
| SCA3D | 6.9/10.4 | – | 98.1/99.3 | – | 39.7/65.2 | – | – |
| Uni3DL | 4.5/9.2 | – | 98.5/99.8 | – | 37.4/63.9 | – | – |
| Uni3D | 1.7/3.9 | 1.2/2.5 | 98.3/99.4 | 0.5/1.1 | 36.3/63.6 | 26.1/44.8 | 8.2/19.1 |
| OmniBind (Base/Large/Full) | 1.2–5.3 | 0.6–2.3 | 98.2–99.0 | 0–0.5 | 34.0–37.5 | 21.5–27.5 | 5.5–11.9 |
| MetaFind w/o ESSGNN | 13.8/23.1 | 11.7/19.2 | 75.1/78.0 | 17.2/21.8 | 44.5/71.3 | 45.8/73.1 | 51.7/76.5 |
| MetaFind w/ ESSGNN | 11.3/21.5 | 10.5/15.9 | 63.2/66.5 | 15.9/20.3 | 41.2/68.8 | 42.0/70.4 | 48.2/74.9 |

讀法：八個基線形狀一模一樣（text/image ≈ 0–7、pc ≈ 98、T+PC ≈ 34–40、I+PC ≈ 22–28、full ≈ 5–12），與 5g「任意資產的文字／圖片 + 自己的 pc、gallery 三模態平均」的重現（0.0/0.0/99.3/0.0/64.9/36.7/0.2）同型。文字專用模型（SCA3D 6.9、Uni3DL 4.5）text 單格較高 → query 文字帶類別資訊。
→ P12（排在 P10、縱圖重評之後）：gallery 文字 = 標註 JSON（`figure2_json` 模板，截 77 token 並記錄）、q_text = `{category} {size: w x l x h cm}`、q_image 單張自己的圖、q_pc 自己的、lr 1e-4、10 代。overlay：`/home/kyzen/metafind_data_json`。

## 6. Stage 2（2026-09-04 凌晨開跑）

審計：`docs/audit/STAGE2_FRESH_AUDIT_20260904.md`。Eq. 6/7/8 逐項一致；場景 dropout、凍結範圍、τ 全對；正文 vs 附錄三處矛盾走附錄版（Eq. 4 才成立）。

| arm | 內容 | 狀態 | 讀數 |
|---|---|---|---|
| 先導 1 | P1 父、完整 T/I/P query、Stage 1 配方（平坦 5e-4） | INVALID | builder 漏帶 prefusion_norm（F1）；批次尾巴退化（F2） |
| 先導 2 | 同上，builder 修好 | ✅ | ProcTHOR S1 82.4 / S2-off 24.2 / S2-on 23.5；λ 93.5→93.1 不學；w/ ESSGNN C 10.1/15.2/49.2/22.2/56.4/50.5/58.9（父 34.7/56.9/86.1/87.6/99.0/92.7/99.7）。結論：完整 T/I/P 的 query 讓 layout 無事可做；平坦 lr 把 layout-free 頭訓壞 |
| S2-C | query 只給文字（Figure 1 的 query 形式）+ 5e-5 warmup 10% cosine | ✅ | ProcTHOR S1 10.3 / S2-off 12.4 / S2-on 12.2；w/ ESSGNN C 22.5/37.7/74.3/62.8/86.1/80.1/88.2（÷父 0.65…0.88，論文 0.82…0.93）→ **Stage 2 主線候選** |
| S2-D | 完整 T/I/P + 5e-5 warmup cosine（只隔離配方） | ✅ | ProcTHOR S1 82.4 / S2-off 36.8 / S2-on 32.8；w/ ESSGNN C 24.9/36.1/71.2/58.2/80.9/73.0/80.1。配方讓損傷從 −40 縮到 −25；λ 仍不動 |

Stage 2 的 UNRESOLVED（各有 arm 或待排）：query 構法（S2-C / S2-D / stage1 遮罩）、leave-one-out vs iterative-prefix、正文字面版 ESSGNN、pooling / λ₀、ProcTHOR 切分（Kyzen 定）。
