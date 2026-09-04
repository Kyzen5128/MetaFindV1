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

## 6. Stage 2（2026-09-04 凌晨開跑）

審計：`docs/audit/STAGE2_FRESH_AUDIT_20260904.md`。Eq. 6/7/8 逐項一致；場景 dropout、凍結範圍、τ 全對；正文 vs 附錄三處矛盾走附錄版（Eq. 4 才成立）。

| arm | 內容 | 狀態 | 讀數 |
|---|---|---|---|
| 先導 1 | P1 父、完整 T/I/P query、Stage 1 配方（平坦 5e-4） | INVALID | builder 漏帶 prefusion_norm（F1）；批次尾巴退化（F2） |
| 先導 2 | 同上，builder 修好 | ✅ | ProcTHOR S1 82.4 / S2-off 24.2 / S2-on 23.5；λ 93.5→93.1 不學；w/ ESSGNN C 10.1/15.2/49.2/22.2/56.4/50.5/58.9（父 34.7/56.9/86.1/87.6/99.0/92.7/99.7）。結論：完整 T/I/P 的 query 讓 layout 無事可做；平坦 lr 把 layout-free 頭訓壞 |
| S2-C | query 只給文字（Figure 1 的 query 形式）+ 5e-5 warmup 10% cosine | ✅ | ProcTHOR S1 10.3 / S2-off 12.4 / S2-on 12.2；w/ ESSGNN C 22.5/37.7/74.3/62.8/86.1/80.1/88.2（÷父 0.65…0.88，論文 0.82…0.93）→ **Stage 2 主線候選** |
| S2-D | 完整 T/I/P + 5e-5 warmup cosine（只隔離配方） | ✅ | ProcTHOR S1 82.4 / S2-off 36.8 / S2-on 32.8；w/ ESSGNN C 24.9/36.1/71.2/58.2/80.9/73.0/80.1。配方讓損傷從 −40 縮到 −25；λ 仍不動 |

Stage 2 的 UNRESOLVED（各有 arm 或待排）：query 構法（S2-C / S2-D / stage1 遮罩）、leave-one-out vs iterative-prefix、正文字面版 ESSGNN、pooling / λ₀、ProcTHOR 切分（Kyzen 定）。
