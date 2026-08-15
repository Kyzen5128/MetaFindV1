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
| **[未定]** | 論文沒說，我們選了一個並記錄（累積 35 條，其中 **U-08a／U-08b／U-18／U-21 為阻斷級**） |
| **[偏離]** | 與論文不同，必須在報告聲明（6 條，D-1…D-6） |

先前的草稿沒有分開，結果出現「我自己加的參數被當成論文真值」這種事。

## 閱讀順序

### 權威順序（衝突時以上位者為準）

```
Level 0 — MetaFind 本身
  docs/metafind_paper.md（含 Appendix）

Level 1 — 相依元件的官方實作（證據，不是論文真值）
  salesforce/ULIP        ULIP-2 backbone 實際怎麼寫
  vgsatorras/egnn        EGNN 參考實作
  atcelen/IDesign        I-Design 公開實作

Level 2 — 我們的復現決策
  02_BUILD_STEPS.md      最新決策，人類可讀
  01_GRAPH_SPEC.md       graph 設計、U／D／RA 登記表
  graph_spec.yaml
  node_registry.yaml     機器可讀契約
  validation_plan.yaml

Level 3 — 實作
  metafind/、tools/、setup/

00_FINDINGS.md           實測事實（F 系列）與決策紀錄
```

> **Level 1 只能回答「這個相依元件官方怎麼做」，不能自動補上 MetaFind 沒寫的部分。**
>
> 例：官方 EGNN 用 `‖·‖²` —— 這**支持** U-17 選平方，但**推不出**
> 「MetaFind 主文其實也想寫平方」。同理，官方 EGNN 有 `embedding_in`／`embedding_out`
> （U-33）、官方 ULIP-2 只對 CLIP 呼叫 `eval()` 而**沒有**設 `requires_grad=False`。
> 這些是證據，不是真值 —— 兩者混淆正是這個專案反覆在防的錯誤。

`00_FINDINGS` 排最後是因為它的 **D 系列是決策，會隨新事實改變**（D1/D2 已於
2026-08-15 大幅修正）。它的 F 系列是實測、可信；D 系列與上位文件衝突時以上位者為準。
它**不再使用 `U-nn` 編號** —— 它曾有自己一套與登記表同編號、不同意義的 UNKNOWN。

### 一致性靠檢查，不靠讀

```bash
python3 tools/check_graph.py      # 1,000+ 項結構檢查
```

這六份文件互相引用的東西太多，人工同步一定會漏。檢查器涵蓋：
邊的端點與 channel、主線無環、可達性、**channel 的 writers／readers 必須與節點的
`reads`／`writes` 對稱**、**gate 判準提到的 channel 必須在它的 `reads` 裡**、
execution order 必須符合 dependency DAG、join policy 的 group 必須有對應的邊、
**`U-nn` 編號跨文件唯一**、以及各文件宣稱的數量與實際相符。

每一項都是因為抓到過真實錯誤才存在的 —— 例如「gate 判準 vs `reads`」就是
G1 宣稱檢查 ProcTHOR 卻看不到它那個 bug 的來源。

| # | 檔案 | 內容 |
|---|---|---|
| 1 | [`02_BUILD_STEPS.md`](02_BUILD_STEPS.md) | **從這裡開始**。逐步驟建置流程，每步標明論文怎麼說、我們怎麼做 |
| 2 | [`01_GRAPH_SPEC.md`](01_GRAPH_SPEC.md) | 完整規格：分類、目標、state、節點、邊、路由、迴圈、失敗、驗證、gate、風險、修正紀錄 |
| 3 | [`00_FINDINGS.md`](00_FINDINGS.md) | 實測事實（F 系列）與架構決策（D 系列），**含論文的多處自相矛盾** |
| 4 | [`graph_spec.yaml`](graph_spec.yaml) | 機器可讀：45 個 state channel、51 條邊、16 組 join policy、11 個決策點、3 個 cycle、UNKNOWN 登記表 |
| 5 | [`node_registry.yaml`](node_registry.yaml) | 33 個節點 + 4 個 subgraph，含逐節點 failure policy 與 rollback |
| 6 | [`validation_plan.yaml`](validation_plan.yaml) | 54 個 L1、17 個 L2、7 個 gate、4 個 Required Audit |

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

### 六項偏離

| id | 偏離 | 影響 |
|---|---|---|
| **D-1** | ViT-bigG-14 凍結（2.5B 參數在 24GB 上無法訓練） | 「entire encoder」在我們的設定下指 3D encoder + fusion（RA-3 記錄） |
| **D-2** | Qwen2.5-VL 取代 GPT-4o | **Table 1 與 Table 2 都受影響** —— 它不只換裁判，也換掉 46,052 筆標註（文字塔的訓練資料）。SC-1 因此只報告差距、不設門檻 |
| **D-3** | 不重跑 6 個 baseline | 只能與論文公佈值比較，並註明協定不同 |
| **D-4** | 不做人工評分 | Table 2 人工欄判 `INSUFFICIENT_EVIDENCE` |
| **D-5** | I-Design 中所有設為 `gpt-4`／`gpt-4-1106-preview` 的 LLM 路徑改導向 `qwen2.5-7b-instruct` | **與 D-2 不同**（那是 GPT-4o／標註與評分）。換規劃器改變**場景本身** → Table 2 全部與 Table 3 場景欄位移；**Table 1 不受影響**。做法是 patch `filter_dict`，**沒有別名** |
| **D-6** | 對 I-Design 的**行為性**修改（patch 02／03）：佈局引用正規化、丟棄懸空引用、合併重複 id、修正迴圈上限、重試換 seed、耗盡放棄場景 | 改的是管線**產出什麼**，不只是誰產出。**偏離的是公開實作** —— 論文作者的整合程式從未公開，不能斷言他們沒做類似修改 |

### 論文自身的四個矛盾

都不設 gate，改用 **Required Audit**（必跑、必留紀錄、**永不阻斷**）——
因為設成 gate 之後，唯一「讓它變綠」的方法就是放寬判準，那等於沒有檢查。

| id | 矛盾 | 預期 |
|---|---|---|
| **RA-1** | §2.5 的 `h⁰ = Concat(x,t)` vs Appendix C 的「`h⁰` 對 SE(3) 不變」前提 | **失敗** |
| **RA-2** | §2.5 的 `f_x → ℝ³` vs 證明需要 `φ_x` 為純量才能提出 `Q` | **失敗** |
| **RA-3** | §3.4 的「fine-tune entire encoder」 vs 單卡 24GB | **不可行**，縮小 claim |
| **RA-4** | §2.5／§3.4 宣稱 ESSGNN 解決 GAT 對 translation **與 scaling** 的敏感，但 SE(3) **不含縮放** | **量測，不預測** |

**RA-4 是逐字重讀才發現的。** §2.5 的動機是
"GATs are highly sensitive to global translation **and scaling**"，§3.4 也重複一次。
但論文證的是 SE(3) 等變，**縮放不在 SE(3) 裡** —— 所以**沒有結構性保證**：
`x → s·x` 時 `‖x_i − x_j‖²` 變成 `s²‖x_i − x_j‖²`，每條訊息都會變。

**但「沒有保證」不等於「一定做不到」** —— MLP 仍可能學到在訓練範圍內對尺度不敏感的行為。
RA-4 **量測 `e_layout` 實際移動多少，不預測結果**。
RA-1／RA-2 是式子寫錯，RA-4 是 claim 開太大；三者都只記錄、不阻斷。

### 七個 gate

```
G1 來源有效 → G2 點雲健全 → G3 物件語料 → G4 gallery 凍結 → G5 報告發布
                                  ├→ G6 Stage 2 就緒（協定 + 場景語料）
                                  └→ G7 場景合成協定
```

`G6` 與 `G7` 是**決定尚未做出**時的擋板，未決回傳 `BLOCKED_EVIDENCE`(rc=3) 而非 FAIL ——
沒有東西壞掉，只是決定還沒做，上游照常跑。

- **G6**：`stage2_protocol`（U-08a／U-08b）**或 `essgnn_edge_protocol`（U-29／U-30／U-19）**未 `resolved`、或 `scene_splits` 有洩漏之前，Stage 2 不准訓練。
- **G7**：`composition_protocol.status` 未 `resolved`（U-18／U-21）之前，不准合成場景。**Table 1 不經過它。**

71 個測試對 7 個 gate。被降級的 gate 候選有 5 個，都寫明不符四判準的哪一條。

`G2` 這一輪**縮小了判準**：它原本要求自取樣點雲必須與 ULIP 官方釋出的點雲一致，
但論文從未說 MetaFind 沿用 ULIP 預取樣的點雲，而 Stage 1 本來就會 fine-tune point encoder。
那個判準在檢驗一個論文沒有主張的命題，降為 `L2-PC-ULIP-REF` 診斷。
**它不是因為過不了才降級 —— 它從來沒跑過。**

### 四個阻斷級的未解項

**U-08a：Stage 2 的正樣本是哪一個 gallery 條目？**

Eq.7a/7b 需要一個 positive。目標是 **ProcTHOR 物件**，gallery 是 **Objaverse-LVIS**，
而兩者的識別碼**實測交集為 0**：

```
ProcTHOR assetId : Countertop_I_8x2, Fridge_19 ...   995 個
Objaverse uid    : 867dfc95e96a4987...            46,052 個
交集             : 0
```

論文完全沒有提到這個對應關係。**沒有它，loss 的正樣本不存在。**

**U-08b：目標物件的 text / image / point cloud 從哪來？**
ProcTHOR 只提供 metadata 與座標，沒有渲染圖也沒有點雲，所以 Eq.6 的三個模態沒有來源。

**這兩個決定之前不要實作 Stage 2** —— 而且現在由 `G6_stage2_ready` 這道 gate
強制，不靠自律。

**U-18：Algorithm 1 第 7 行「放進場景、更新場景圖」到底產生什麼？**
下一輪立刻要 `ESSGNN(G)`，需要新節點的 `t_i`、位置、朝向、尺度、物理邊、語意邊。
論文一項都沒定義，而這個選擇會改變後面每一次檢索。

**U-21：Algorithm 1 的 `G_0` 與 `{Q_1..Q_N}` 從哪來？**

論文這裡自己說了兩件不見得相容的事：

```
§3.1  Scene-level layout-aware retrieval is conducted on ProcTHOR-10K
§3.3  evaluate ... on the scene generation pipeline of I-Design
      on a set of 200 randomly sampled scenes
```

讀法 A：200 個場景抽自 ProcTHOR-10K，再送進 I-Design 的管線。
讀法 B：200 個場景由 I-Design 從文字生成，ProcTHOR 只用於 Stage 2 訓練。

**上一輪我把讀法 B 當成事實寫死了，那是過度修正。** 兩種讀法都還開著。
但無論哪一種，graph 原本**根本沒有評估輸入這條 channel**，
而 `n16` 讀的是 ProcTHOR 房屋 —— 讀法 B 下那是錯的資料集，
讀法 A 下那仍是錯的東西（Algorithm 1 要的是生成請求，不是完成的佈局）。
**Table 2 的資料流從來沒有閉合。** 現在由 `G7_composition_protocol` 擋著。

Stage 1 與 Table 1 不經過 G6/G7，可以照常進行。

### Stage 1 也還沒鎖住的四件事

這一輪逐項對論文後新發現，都會直接改變 Table 1：

| id | 未定 | 論文怎麼說 |
|---|---|---|
| **U-13** | Full model 用哪一種 fusion | §2.4 列了五種、沒說是哪個。Table 3 只排除 Mean 與 MLPs |
| **U-14** | 11 張渲染圖怎麼變成一個 `e_image` | 完全沒說。影響 Table 1 七個條件中的四個 |
| **U-15** | 結構化標註怎麼序列化成 encoder 的輸入字串 | 只給欄位，沒給格式 |
| **U-16** | query / gallery 兩塔是否共享權重 | 說「dedicated query encoder」、說兩者都訓練，但沒說共享關係 |

### 其他重大未解項

**R-01：已部分實測。** I-Design **裝得起來**（README 要的 MinkowskiEngine／dgl／torch 1.12
都不需要，`requirements.txt` 的 `ag2==0.2.0` 在 PyPI 上不存在），
`create_initial_design` **會成功**，但 Qwen2.5-7B 跑 5 次**0 個場景完成**，每次失敗路徑不同。

**沒有基準，所以不能斷定那是缺陷** —— I-Design 沒用原版規劃器在本機跑過，論文也沒說那是什麼。
`setup/patches/` 的三個 patch 沒有論文依據，其中兩個會改變場景與完成率。

**U-17：ESSGNN 用 `d` 還是 `d²`。** §2.5 寫 `‖x_i − x_j‖₂`，Appendix C 用 `‖·‖²`。
兩者都是 SE(3) 不變、都不破壞證明，但**餵進 MLP 的數值不同，訓練結果就不同**。
實作依 Appendix C 與原始 EGNN 用平方，記錄為選擇。

**U-22：論文沒有公佈任何訓練超參數。** optimizer、learning rate、batch size、
epochs、weight decay、scheduler、`τ`、`λ` 初值、ESSGNN 的 `L` 與 hidden 寬度、pooling ——
一個都沒有。全部列在 `01_GRAPH_SPEC.md` §15，否則最後對不上時，
分不清是模型沒復現還是 recipe 不同。
