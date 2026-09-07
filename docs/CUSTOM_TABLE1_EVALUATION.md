# 自訂七模態檢索評估

本流程以固定物件清單評估「能否找回同一個 3D 資產」，輸出兩張七欄表。
它沿用 MetaFind 的七種 query 條件與 R@1／R@5，但 **不宣稱已恢復作者 Table 1 未公開的完整評估設定**。

## 依據與範圍

- **PAPER FACT**：MetaFind 採 T、I、P、T+I、T+P、I+P、T+I+P，並報告 R@1／R@5。baseline 另加 mean pooling。來源：`docs/paper/metafind_source/3experiments.tex:15–24`。
- **IMPLEMENTATION CHOICE**：下列觀測分離、候選清單、逐模態正規化 mean baseline 與 exact-UID 正解是本評估的明確規則。
- **UNKNOWN**：作者實際 query 配對、baseline gallery 構成及正規化細節。這份規格不替這些未知項目補寫論文事實。
- **OBSERVED IMPLEMENTATION**：實作入口為 `metafind.eval.custom_protocol`、`metafind.eval.custom_table1`，模型載入在 `metafind.eval.custom_models`。不覆寫原本的 `eval_protocols.json`、promoted gallery 或訓練流程。

## 兩個評估版本

| 版本 | Query | Gallery | 可解釋的結果 |
|---|---|---|---|
| `same_observation` | canonical 文字、全部快取視角的特徵平均、canonical 點雲 | 同一套觀測；各方法用自己的 gallery encoder | 相同觀測下的資產辨識能力 |
| `different_observations` | 外部提供的不同描述、一個固定視角、重新採樣的點雲 | canonical 文字、排除該資產保留視角後的圖像特徵平均、canonical 點雲 | 換觀測後能否仍找回同一資產 |

每個 gallery 資產都預先固定 `k = uid_seed(uid) % n_views`，query 只取第 k 個視角，gallery 取其餘視角。
11 views 為 1／10；12 views 為 1／11。所有候選資產都採同一規則，gallery 不依當前 query 改動，因此仍可預先編碼。
子集圖像特徵採 float32 算術平均，不先逐視角正規化。
兩個版本均從相同的快取 `views` 矩陣轉 float32 後求平均。舊 n06 對平均向量與逐視角向量分別量化為 float16，故此處不直接讀取其 `image` 平均欄位，也不宣稱與舊評估逐位元相同。

不同文字必須在實際載入的 CLIP tokenizer 下得到不同 token IDs；只有完整字串不同、差異被截斷到 context 外，不算不同文字。
canonical 全文取自 embedding sidecar，並與凍結的 serializer／annotation 核對；執行時以實際 backbone 重新編碼文字。
不同描述不自動代表獨立標註或統計獨立，描述來源須記在 provenance。

點雲由同一 UID 的 mesh，使用 `uid_seed(uid) + 1000003` 重採樣 10,000 個 xyzrgb 點。
記錄 mesh hash、sampler 版本及原始碼 hash。歷史 canonical 點雲若缺乏 mesh／sampler 身分，不能宣稱兩份點雲「只有 seed 不同」。
本流程不以其他物件的文字／圖片冒充該 UID 的 query。

## 模型列

1. **`ulip2_available_mean`**：官方 ULIP-2 預訓練權重。每個可用模態先 L2 normalize，取 mean，再 L2 normalize。缺模態不加入向量、不加入 learned mask token。gallery 同樣融合完整 T/I/P，使用自己的預訓練 PC encoder。
2. **`stage1`**：還原 checkpoint 內記錄的架構、初始化權重、point encoder 與 fusion；保留 checkpoint 本身的缺模態處理與融合正規化設定。若訓練時有獨立 query point encoder，評估也使用它。
3. **`stage2_layout_off`**（提供 Stage 2 record 時）：檢查父 Stage 1 checkpoint 與實際權重 hash，只覆蓋 query fusion；`layout=None`。與 Stage 1 使用相同 gallery 向量，並執行不變性檢查。

Stage 1 相對 mean 的提升包含整體訓練效果，可能也包含 point encoder 微調，不能單獨歸因於 fusion。
Stage 2 layout 關閉這一列不測量場景品質或 ESSGNN 的 layout 效益。

v1 支援 frozen CLIP、mean 圖像快取、每個 tower 一個 image token。trainable CLIP、多 image tokens、錯誤父 checkpoint、缺少必要身分的 legacy Stage 2、會連帶修改 gallery 的 tied fusion 都直接拒絕。

## 固定清單與計分

- query／gallery 清單都不得為空或重複；每個 query UID 必須存在 gallery。
- 同一個 protocol 的兩個版本、所有模型、七種模態都使用完全相同的清單與順序。缺資料直接失敗，不在每列自行刪除 query，也不退回 canonical 觀測。
- 每個 gallery 始終完整提供 T/I/P；七種條件只改變 query 可見模態。
- 正解由 `query_uid → gallery column` 映射，不假設相似度矩陣的對角線就是正解。
- 使用既有 `normalize_for_scoring` 與 `score_streaming`：float64 cosine；同分採對模型不利的 rank。
- `R@k = 100 × 命中前 k 名的 query 數 / 全部 query 數`。這是同資產命中率；不是類別分類，也不是語義替代品適合度。
- JSON 另記融合相對其可用單一模態最佳結果、以及相對 mean baseline 的 R@1 提升（percentage points）。
- 高分不作失敗條件。若簡單 baseline 已飽和，應如實說明此設定辨別融合優勢的能力有限。

預設結果標示為 `development_comparison`，不認證獨立測試集。已用來選 checkpoint／改規則的資料不可再稱未見測試集。
要提出獨立泛化結論，應先固定完整規格、只用驗證資料選模型，再在未參與選擇的資產上執行；不要在看完分數後重選 query 或候選數量。

## 執行方式

先準備三個 JSON 檔。UID 必須是實際資料中的 UID；以下只展示格式：

`query_uids.json`：

```json
["asset_a", "asset_b"]
```

`gallery_uids.json`：

```json
["asset_b", "asset_c", "asset_a"]
```

`query_texts.json`：

```json
{
  "asset_a": "A separately written description of asset A.",
  "asset_b": "A separately written description of asset B."
}
```

query 文字檔必須恰好覆蓋 query 清單，不接受多餘／缺少 UID。請使用能描述該資產的真實文字；範例字串不可拿去產正式結果。
可另提供 `provenance.json`，記錄 split 原始檔案位置／hash、資料版本、文字由誰或哪個模型產生、prompt／版本、是否曾用於模型選擇。
CLI 會自動 hash 三個輸入檔及提供的 provenance 檔。

以下路徑是操作模板；`METAFIND_DATA` 必須指向要評估的完整資料根目錄，checkpoint 必須與其 encoding 設定相容。

```bash
export METAFIND_DATA=/path/to/evaluation_data
PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python

# 1. CPU 準備不同 query 點雲，凍結所有輸入與視角規則。
$PY -m metafind.eval.custom_protocol \
  --query-uids /path/to/query_uids.json \
  --gallery-uids /path/to/gallery_uids.json \
  --query-texts /path/to/query_texts.json \
  --provenance /path/to/provenance.json \
  --out-dir /path/to/custom_protocol_run01

# 2. 核對來源、設定與 checkpoint；不建立大型模型，不產分數。
$PY -m metafind.eval.custom_table1 \
  --protocol /path/to/custom_protocol_run01/protocol.json \
  --stage1-record /path/to/stage1_best_ckpt.json \
  --stage2-record /path/to/variant_ckpts.json \
  --stage2-variant full \
  --out-dir /path/to/custom_eval_run01 \
  --check-only

# 3. 有可用運算資源時執行相同參數，移除 --check-only。
$PY -m metafind.eval.custom_table1 \
  --protocol /path/to/custom_protocol_run01/protocol.json \
  --stage1-record /path/to/stage1_best_ckpt.json \
  --stage2-record /path/to/variant_ckpts.json \
  --stage2-variant full \
  --out-dir /path/to/custom_eval_run01 \
  --device cuda --batch-size 16 --seed 20260907
```

只評估 mean／Stage 1 時可省略 `--stage2-record`。CLI 預設裝置是 CPU，不會自動占用 GPU。
輸出目錄必須全新；失敗的目錄保留供檢查，重跑換新名稱。
這些命令不修改 canonical 資料、訓練 checkpoint 或執行中的 annotation／training 工作。

## 產物與可驗證範圍

- `protocol.json`：完整來源快照、UID 清單、query 文字、視角規則及 query 點雲 hash。
- `provenance.json`：checkpoint／protocol／程式碼身分、指令、seed、環境與裝置。
- `results.json`：兩個版本的所有模型與七種模態 R@1／R@5、提升幅度與驗證狀態。
- `table.md`：兩張可讀的七欄表。
- 每模型／版本／模態的 `.jsonl`：逐 query 的 target UID、target column、top1 UID、rank、tie count。
- `failure.json`：模型評估中途失敗原因；不會把部分結果冒充完整表格。

圖像特徵取自逐檔 hash 核對的原有 cache，producer sidecars 一併保存。若歷史 cache 沒有記錄 OpenCLIP 權重 bytes，來源限制保留在結果中；目前模型的初始化身分不能追溯證明歷史 cache 的權重身分。
文字 token 差異必須等實際 backbone tokenizer 載入才驗證，因此 `--check-only` 不宣稱已通過此項。

實作測試使用合成資料、真實小型 fusion 模型與替代 backbone，驗證 UID 配對、兩個觀測版本、七模態遮罩、checkpoint 還原與錯誤拒絕。
這些測試不等於已執行真實 ULIP-2／Objaverse 評估；正式數字須由上述資料與 checkpoint 命令另外產生。

2026-09-07 最初實作驗證：新增 98 個 CPU 測試；連同當時既有回歸共 **1,236 passed，68.56 秒**。命令：

```bash
/home/kyzen/miniconda3/envs/MetaFind/bin/python -m pytest tests -q \
  --ignore=tests/gpu/test_cuda_smoke.py \
  --ignore=tests/hooks/test_research_authority_guard.py
```

GPU smoke 與 Claude-only hook 測試未執行。共 69 個 warnings，包含 transformer 設定、刻意測試的 temperature 偏差、套件 deprecated API，以及一項 multiprocessing QueueFeederThread 清理警告；此次未修改這些既有子系統。
逐檔身分與驗證摘要另存於 `docs/audit/custom_table1_20260907_checks.json`。

2026-09-07 接續已有真實 ULIP-2／既有 Stage 1 checkpoint 的 CPU smoke：固定 2 query／6 gallery、兩觀測與七模態共 28 組完成，模型／loader／scorer 沒有替換；影像使用原有 per-view cache。這是小樣本執行驗證，分數不代表模型品質或論文復現。舊 Stage 2 record 缺 embedded metadata，預檢按設計拒絕，未納入成功 run。當輪完整回歸 **1,552 passed**，命令／scope／真模型 artifacts 見 [執行路徑審查](history/REPRODUCTION_RUNTIME_REVIEW_20260907.md)。

2026-09-08 再以新產生的 Stage 1／Stage 2 單步診斷 checkpoint 跑通三種模型列：**42 組、84 筆 query records**，CPU 51.32 秒，Stage 2 gallery 不變性通過，未放寬 legacy 檢查。沿用 2×6 protocol，其中一個 query 曾參與新 parent 的 selection，所以只能作執行診斷。完整回歸與真訓練／評估證據見 [接續審查](history/REPRODUCTION_TRAINING_REVIEW_20260908.md)。
