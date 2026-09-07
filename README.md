# MetaFindV1

復現 *MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation*。
論文內容權威是作者的 [arXiv TeX source](docs/paper/metafind_source/)。本 repo 包含資料處理、Stage 1／Stage 2 訓練、gallery 建立及檢索評估；程式可執行、測試通過與論文數值復現，是需要分別驗證的狀態。

## 從這裡開始

| 要找的內容 | 入口 |
|---|---|
| 資料從哪裡來、產物給誰讀、更新後哪些 cache 需重新核對 | [資料流與產物交接](docs/DATA_FLOW.md) |
| 本 checkout 的實際程序、指定 corpus 的檔案盤點 | `bash tools/status.sh --data /absolute/path/to/data`；加 `--json` 保存快照 |
| 同一觀測／不同觀測的七模態檢索；mean、Stage 1、Stage 2-off | [自訂評估規格與命令](docs/CUSTOM_TABLE1_EVALUATION.md) |
| 從論文到訓練／評估的審查、修正及未完成項目 | [最新資料邊界與評分交接](docs/audit/REPRODUCTION_CORPUS_REVIEW_20260908.md)、[真實場景驗證](docs/history/REPRODUCTION_SCENE_REVIEW_20260908.md)、[真實訓練審查](docs/history/REPRODUCTION_TRAINING_REVIEW_20260908.md) |
| Algorithm 1 的逐步取回／圖更新，與完整 Table 2 的界線 | [場景檢索核心](docs/SCENE_COMPOSITION.md) |
| I-Design → 原始查詢 → 檢索 → 實際 GLB 放置／渲染 | [planner 輸入](docs/IDesign_INPUTS.md)、[原始查詢交接](docs/RAW_SCENE_INPUTS.md)、[Blender 放置](docs/SCENE_PLACEMENT.md) |
| 外部場景評分綁定、四維 1–5 分與缺分／失敗的分母 | [場景評分格式與命令](docs/SCENE_SCORES.md)；部分 n17／n20，未執行裁判。 |
| CPU、GPU、hook 測試分組、環境要求與驗證界線 | [測試導覽](tests/README.md) |
| 測試整理、失效程式刪除、文件搬移與 outputs 清理 | [本輪整理紀錄](docs/audit/REPRODUCTION_CORPUS_REVIEW_20260908.md)、[先前測試整理](docs/history/CLEANUP_REPORT_20260907.md)、[文件導覽](docs/README.md) |
| 2026-09-07 修正的程式問題與當次驗證 | [程式修正紀錄](docs/history/CODE_REPAIR_REPORT_20260907.md) |
| 論文逐條公式與證據 | [原始公式清單](docs/audit/A_FORMULA_INVENTORY.md)、[本輪公式／梯度審查](docs/audit/formula_review_20260907.md)，回查 [paper source](docs/paper/metafind_source/) |
| 衍生規格、gate 與已記錄決策 | [graph 文件導覽](docs/graph/README.md)、[Decision Ledger](workflow/DECISION_LEDGER.md) |

自訂評估是 **IMPLEMENTATION CHOICE**：固定同一批 query/gallery UID，比較觀測構法。它不補定作者未交代的 Table 1 query 細節，也不把其分數標成論文復現結果。歷史 `TABLE1_REPORT_*`、`NOTE_*`、audit 與 handoff 須連同原始 artifact、實驗配置及後續更正閱讀。

## 環境與資料位置

新環境可依序執行：

```bash
# 如需外部資料根目錄，先 export METAFIND_DATA=/absolute/path/to/data。
bash setup/01_storage.sh
bash setup/02_conda_env.sh
conda activate MetaFind
python -m metafind.paths
python setup/03_verify_env.py
```

`METAFIND_DATA` 預設為 `<repo>/data`，必須在啟動 Python 前設定。已有環境應先核對列出的資料位置，再選擇需要的建置步驟；`01_storage.sh` 會建立目錄並設定 repo 的 data 連結。環境驗證加 `--full` 會包含約 10GB 的 ViT-bigG-14 權重下載；環境驗證不代表資料前處理已完成。

目前標註使用的 paper corpus 可唯讀觀察：

```bash
bash tools/status.sh --data /home/kyzen/metafind/metafind_data_paper
```

status 以 `/proc` 核對 cwd 為本 checkout 的 Python module／shell chain，列出各程序初始環境宣告的 corpus；不把繼承的 `METAFIND_REPO` 當 Python import 來源。從同一 corpus 的 live annotation 推斷 v9／v10；沒有 live annotation 時可明確加 `--prompt-mode figure2_v10`，否則只列契約分布。它不啟動模型、不使用 GPU、不重建 index；檔案存在或契約標記相符只表示盤點結果，不代表完成、來源驗證或 gate 通過。

一般 CPU 測試可使用以下命令；GPU／hook 另行執行，詳細需求見 [測試導覽](tests/README.md)。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYOPENGL_PLATFORM=egl \
LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
python -m pytest tests -q -ra -p no:cacheprovider \
  --ignore=tests/gpu --ignore=tests/hooks
```

上例的 Mesa vendor JSON 是本機已驗證路徑；其他機器需依實際軟體 OpenGL 安裝設定。只隱藏 CUDA 不足以隔離舊 PyRender 的 EGL backend，詳見 [測試導覽](tests/README.md)。

## 程式與文件目錄

```text
metafind/
  data/       GLB／點雲／render／annotation、文字與影像 cache、splits、場景圖
  models/     ULIP-2 封裝、fusion、DualTower、ESSGNN、protocol 解析
  train/      Stage 1、Stage 2、gallery index 建立與 promotion
  eval/       retrieval、diagnostics、自訂觀測 protocol 與七模態評估
  scene/      planner／原始查詢交接、逐步場景檢索、語義準備、GLB 放置與 CPU 渲染
  gates/      明訂的驗證與 promotion gate
  compat/     上游 runtime 相容修補
  vendor/     保留原始上游程式與各自授權
tools/        資料準備、實驗工具與指定流程的 shell 串接
setup/        環境建置與驗證
tests/        data／models／train／eval／pipeline／gpu／hooks 七組
docs/         資料流、操作規格、paper source、衍生 audit 與歷史報告
workflow/     決策、任務與工作狀態紀錄
data/         預設資料根目錄，亦可能是外部資料的連結
```

## 實驗身分與證據

- **PAPER FACT** 從 [MetaFind source](docs/paper/metafind_source/) 核對。其他論文與 upstream 實作只在 MetaFind 明確繼承的範圍內適用。
- **OBSERVED IMPLEMENTATION／DATA** 需記錄資料根目錄、UID 清單、檔案 hashes、文字模板、view／sampler 版本、checkpoint 與實際 forward 設定。
- **IMPLEMENTATION CHOICE／DEVIATION** 以該次 resolved protocol、決策與 run record 為準。例如 CLIP 訓練範圍、標註模型、可用模態及選模集合，不能只從「Stage 1／Stage 2」名稱判定。
- **UNKNOWN／INFERENCE** 保留其界線。相同 UID 不足以證明 train/test contamination；某批融合分數偏高也不足以確定唯一成因。CPU 測試、G4、數值 parity 各自只證明其檢查範圍。

目前 Stage 1 trainer 的預設路徑訓練 point encoder 與 fusion，text/image cache 使用 frozen CLIP；另有 `fuser_only` 等設定。這是目前實作，不是 MetaFind 已逐模組指定凍結策略的宣告。具體 checkpoint 是否符合研究決策，須核對其 record，不能沿用舊 README 的固定偏離數量或永久啟用／停用狀態。

## 已登記的偏離議題

以下是 [graph registry](docs/graph/graph_spec.yaml) 的完整議題索引，保留追溯性；不表示全部仍啟用，也不把歷史 corpus 數或模型名稱當成目前狀態。每次實驗應回查相應決策與實際 run record。

| id | 已登記範圍 |
|---|---|
| **D-1** | CLIP 訓練範圍與條件式偏離判讀 |
| **D-2** | 資產標註模型 |
| **D-3** | 基線重跑範圍 |
| **D-4** | 人工場景評分範圍 |
| **D-5** | I-Design 規劃模型替換 |
| **D-6** | I-Design 行為修改 |
| **D-7** | JSON decoding 約束 |
| **D-8** | 場景評分模型 |
| **D-9** | 類別標籤參與標註 |
| **D-10** | 對比學習負樣本／batch 範圍 |
| **D-11** | 渲染背景與影像處理 |
| **D-12** | 點雲顏色與 `COLOR_0` 處理 |
| **D-13** | 取得、解析與 admitted 語料數差異 |
| **D-14** | ESSGNN 初始節點表示的原文歧義與已選解讀 |

## 授權

`metafind/vendor/` 下的第三方程式各自沿用原授權，見 [vendor 說明](metafind/vendor/README.md)（ULIP：BSD-3-Clause；EGNN：MIT）。
