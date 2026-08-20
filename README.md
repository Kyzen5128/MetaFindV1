# MetaFind 復現

復現 *MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation*
論文權威是作者的 arXiv TeX source：[`docs/paper/metafind_source/`](docs/paper/metafind_source/)
（公式逐條清單見 [`docs/audit/A_FORMULA_INVENTORY.md`](docs/audit/A_FORMULA_INVENTORY.md)）。
PDF 轉出的 Markdown 副本已刪除 —— 轉檔會把 LaTeX 反斜線當成 C 跳脫字元，
留著就是留一套會靜默出錯的競爭權威。

單張 24GB 以上的 GPU（本機實測為 RTX 5090 32GB）。**Stage 1 訓練 PointBERT + fusion**，只有 ViT-bigG-14 凍結；本地 Qwen 取代 GPT-4o／GPT-4。

## 快速開始

```bash
bash setup/01_storage.sh          # 建資料目錄（預設 <repo>/data，可用 METAFIND_DATA 覆寫）
bash setup/02_conda_env.sh        # 建 conda 環境 MetaFind
conda activate MetaFind
python setup/03_verify_env.py     # 驗證環境（加 --full 會下載 10GB 的 ViT-bigG-14）
python -m pytest tests/ -q        # 單元測試
```

## 目錄結構

```
metafind/            我們寫的程式
  models/            ESSGNN、模態融合、雙塔對比 loss、雙塔模型、ULIP-2 backbone 封裝
  data/              資料抓取、完整性驗證、ProcTHOR 場景圖
  compat/            ULIP 在現代 PyTorch 上的 runtime 修補（不改上游原始碼）
  vendor/            上游第三方原始碼（ULIP、EGNN）→ 見 vendor/README.md
setup/               環境建置與驗證
tests/               單元測試
docs/
  paper/             論文：*.gz 是 arXiv 原始壓縮檔，*_source/ 是解壓後的 TeX（唯一權威）
  audit/             公式稽核 A–F → 見 audit/A_FORMULA_INVENTORY.md
  graph/             設計文件 → 見 graph/README.md
data/                資料根目錄，或指向 METAFIND_DATA 的 symlink（不進 git）
scratch/             參考用的雜項腳本
```

## 設計文件

`docs/graph/` 是用 graph-engineering 方法產出的完整規格：

| 檔案 | 內容 |
|---|---|
| [`00_FINDINGS.md`](docs/graph/00_FINDINGS.md) | 實際檢查論文與程式碼後的硬事實，**包含論文的多處自相矛盾** |
| [`01_GRAPH_SPEC.md`](docs/graph/01_GRAPH_SPEC.md) | 節點、state、邊、路由、迴圈、失敗政策、gate、可觀測性 |
| [`02_BUILD_STEPS.md`](docs/graph/02_BUILD_STEPS.md) | 逐步驟做什麼、每步的通過條件 |
| `*.yaml` | 結構化規格（可程式化檢查） |

## 已知偏離論文之處

**正式偏離六項（D-2…D-7）＋條件式一項（D-1，已判定 `resolved_inactive`）**，編號以
[`docs/graph/graph_spec.yaml`](docs/graph/graph_spec.yaml) 為準：D-2…D-7 在
`boundary.deviations`，D-1 在 `boundary.conditional_deviations`，
`active_if: paper_clip_train_scope == 'trainable' AND actual_clip_train_scope == 'frozen'`。

| id | 內容 |
|---|---|
| **D-1** *(條件式・`resolved_inactive`)* | ViT-bigG-14 的 CLIP 側保持凍結。**U-34 已於 2026-08-16 判定為 `frozen`**，故 `paper = actual = frozen`、`active_if` 為 false，**不列為 active deviation**。判讀依據：MetaFind 明確建立於 ULIP-2；ULIP-2 §3.3 明文 "freeze it during pre-training"；MetaFind 全文未逐 module 聲明改變此策略。**不得寫成「MetaFind 明文說 OpenCLIP frozen」**。 規則保留供日後重開 |
| **D-2** | Qwen2.5-VL 取代 **GPT-4o**（資產標註與場景評分） |
| **D-3** | 不重跑 6 個 baseline |
| **D-4** | 不做人工評分 |
| **D-5** | I-Design 中所有設為 `gpt-4`／`gpt-4-1106-preview` 的 LLM 路徑改導向 `qwen2.5-7b-instruct` |
| **D-6** | 對 I-Design 的**行為性**修改（patch 02／03）：偏離的是**公開實作**，不是「論文所做的事」 |
| **D-7** | I-Design 的 **JSON-constrained decoding 未重現**。補充材料 §7：*"All agents utilize GPT-4's JSON mode to restrict outputs exclusively to valid JSON"*，而我們的 vLLM 沒開任何 guided decoding。**與 D-5 不同**——D-5 是誰回答，D-7 是回答受不受結構約束。Qwen 因此**可能吐出結構上不合法的 JSON，GPT-4 在那個模式下不可能**，那會落進 Engineer 的 schema 驗證重試迴圈。分開編號是因為兩者可獨立修復：開了 guided JSON 就能退掉 D-7，D-5 原封不動 |

> **兩次更正記在這裡。**
> 早期這張表寫「D2＝frozen backbone 全部預先快取、訓練只在 1280-d 向量上」——
> 那正是 Table 3 的 `Train fuser only`（8.7 vs Full 11.4），與 §2.6 相反。
>
> 後來 D-1 被寫成「已確立的偏離」，理由是 ULIP-2 公開程式沒有凍 CLIP。
> **那個推論已撤回**：ULIP-2 §3.3 明文 "freeze it during the pre-training"，
> 公開程式沒設 `requires_grad=False` 是它自己對不上論文，不是設計如此。
> D-1 之後改為條件式，繫於 **U-34**；U-34 已於 2026-08-16 判定為 `frozen`，D-1 因此不啟用。

論文自身的矛盾（F 系列、RA 系列）另見 [`docs/graph/00_FINDINGS.md`](docs/graph/00_FINDINGS.md)
與 [`01_GRAPH_SPEC.md` §11](docs/graph/01_GRAPH_SPEC.md)。

## 授權

MetaFind 復現程式碼見本 repo。`metafind/vendor/` 下的第三方程式碼各自沿用原授權
（ULIP: BSD-3-Clause，EGNN: MIT）。
