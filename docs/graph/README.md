# MetaFind 復現 — 設計文件

依 `graph-engineering` 方法產出，用於在單張 RTX 4090 上復現
[MetaFind](../metafind_paper.md)。

**2026-08-15 全面改寫。** 先前的草稿有六個會實際改變實驗結果的錯誤（最嚴重的一個
把論文列為較差的 ablation 裝成了主線），全部依論文原文逐條核對修正。
修正清單見 [`01_GRAPH_SPEC.md` §16](01_GRAPH_SPEC.md)。

## 三類內容全程分開標示

文件裡最容易出事的，就是把這三種東西混在一起：

| 標記 | 意思 |
|---|---|
| **[論文]** | 原文明確規定，附引文 |
| **[未定]** | 論文沒說，我們選了一個並記錄（累積 10 條，U-01…U-10） |
| **[偏離]** | 與論文不同，必須在報告聲明（4 條，D-1…D-4） |

先前的草稿沒有分開，結果出現「我自己加的參數被當成論文真值」這種事。

## 閱讀順序

| # | 檔案 | 內容 |
|---|---|---|
| 1 | [`00_FINDINGS.md`](00_FINDINGS.md) | 實際檢查論文與程式碼後的硬事實，**含論文的多處自相矛盾** |
| 2 | [`01_GRAPH_SPEC.md`](01_GRAPH_SPEC.md) | 完整規格：分類、目標、state、節點、邊、路由、迴圈、失敗、驗證、gate、可觀測性、風險、修正紀錄 |
| 3 | [`02_BUILD_STEPS.md`](02_BUILD_STEPS.md) | 逐步驟建置流程，每步標明論文怎麼說、我們怎麼做 |
| 4 | [`graph_spec.yaml`](graph_spec.yaml) | 機器可讀：35 個 state channel、39 條邊、11 組 join policy、10 個決策點、3 個 cycle |
| 5 | [`node_registry.yaml`](node_registry.yaml) | 27 個節點 + 4 個 subgraph，含逐節點 failure policy 與 rollback |
| 6 | [`validation_plan.yaml`](validation_plan.yaml) | 43 個 L1、15 個 L2、5 個 gate、3 個 Required Audit |

## 一頁摘要

`hierarchical DAG + stateful + parallel`，主線零回邊；3 個 cycle 全封在 subgraph 內。
`control_authority: A1` ／ `execution_mode: probabilistic` ／ `topology_class: workflow`
—— Qwen 出現三次但**從不決定路由**，只產生 payload。

### 論文只要兩個資料集

| | 內容 | 我們怎麼取得 |
|---|---|---|
| **Objaverse-LVIS** | manifest 實際 46,052 個資產（論文說「約 48,000」） | 下載 GLB **原始 mesh 並保留**；點雲與 11 視角渲染圖都從 mesh 產生 |
| **ProcTHOR-10K** | 10,000 train + 1,000 val + 1,000 test | JSONL |

**不下載**：ULIP-2 預先取樣的點雲（185 GB）、ULIP-2 的渲染圖（474 GB，而且不是論文要的
11 正交視角）、ShapeNet triplets（409 GB）。

### 四項偏離

| id | 偏離 | 影響 |
|---|---|---|
| **D-1** | ViT-bigG-14 凍結（2.5B 參數在 24GB 上無法訓練） | 「entire encoder」在我們的設定下指 3D encoder + fusion（RA-3 記錄） |
| **D-2** | Qwen2.5-VL 取代 GPT-4o | Table 2 絕對數字與論文不可比，只剩方向性 |
| **D-3** | 不重跑 6 個 baseline | 只能與論文公佈值比較，並註明協定不同 |
| **D-4** | 不做人工評分 | Table 2 人工欄判 `INSUFFICIENT_EVIDENCE` |

### 論文自身的三個矛盾

都不設 gate，改用 **Required Audit**（必跑、必留紀錄、**永不阻斷**）——
因為設成 gate 之後，唯一「讓它變綠」的方法就是放寬判準，那等於沒有檢查。

| id | 矛盾 | 預期 |
|---|---|---|
| **RA-1** | §2.5 的 `h⁰ = Concat(x,t)` vs Appendix C 的「`h⁰` 對 SE(3) 不變」前提 | **失敗** |
| **RA-2** | §2.5 的 `f_x → ℝ³` vs 證明需要 `φ_x` 為純量才能提出 `Q` | **失敗** |
| **RA-3** | §3.4 的「fine-tune entire encoder」 vs 單卡 24GB | **不可行**，縮小 claim |

### 五個 gate

`G1` 來源有效（G-INVALID）→ `G2` 點雲分布（G-INVALID）→ `G3` 語料有效（G-INVALID）
→ `G4` gallery 凍結（G-CONTAM）→ `G5` 報告發布（G-IRREVERSIBLE）

58 個測試對 5 個 gate。被降級的 gate 候選有 5 個，都寫明不符四判準的哪一條。

### 最大的未解項

**U-08：Stage 2 的訓練樣本怎麼從 ProcTHOR 建構，論文完全沒有定義。**
哪個物件當目標、「current scene」是什麼、一間房產幾筆樣本、負樣本怎麼取 —— 全部未定。
我們採用的協定明列在 `02_BUILD_STEPS`，但那是**選擇**，不是論文規定。

**R-01：I-Design 尚未驗證能否執行。** Table 2 全部與 Table 3 的場景欄全部依賴它。
查它很便宜，應該盡早做。
