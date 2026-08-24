# MetaFind 復現 — 設計文件

依 `graph-engineering` 方法產出，用於在單張 RTX 4090（此處的 `RTX 4090` 為前一台機器；本機實測為 RTX 5090 32GB，凡以 24GB 為前提的可行性判斷都要重新量測） 上復現
[MetaFind](../paper/metafind_source/)（作者 arXiv TeX）。

**2026-08-15 全面改寫。** 先前的草稿有六個會實際改變實驗結果的錯誤（最嚴重的一個
把論文列為較差的 ablation 裝成了主線），全部依論文原文逐條核對修正。
修正清單見 [`01_GRAPH_SPEC.md` §16](01_GRAPH_SPEC.md)。

## 三類內容全程分開標示

文件裡最容易出事的，就是把這三種東西混在一起：

| 標記 | 意思 |
|---|---|
| **[論文]** | 原文明確規定，附引文 |
| **[未定]** | 論文沒說，我們選了一個並記錄。U registry 共 42 項，其中 32 項 unresolved、10 項 resolved（U-08／08a／08b／08d／08e、U-18、U-20、U-21、U-34 於 2026-08-16；**U-26 於 2026-08-17**）。**先前四個會擋住 Stage 2／Table 2 的項目已全部判定，U-26 的 ESSGNN 架構也已選定（附錄的 shared-message 版，標為 `[INFERENCE]`，2.5 版留作對照假設）**；新增 **U-08c**（ProcTHOR 資產數四個數字互相矛盾，一律從安裝的 build 現場推導）|
| **[偏離]** | 與論文不同，必須在報告聲明（**12 條 D-2…D-13**。條件式的 **D-1** 已於 2026-08-16 判定**不啟用**） |

先前的草稿沒有分開，結果出現「我自己加的參數被當成論文真值」這種事。

## 閱讀順序

### 權威順序（衝突時以上位者為準）

**公式權威：arXiv TeX source > 已發表 PDF > 轉檔 Markdown。**

> **命名提醒：本文件與程式裡的「Eq. 7a／7b」是我們的簡稱，不是論文的編號。**
> 作者 TeX 只有**一個**編號 (7)，同一式裡並列 q2g 與 g2q；7a/7b 的拆分是
> PDF 轉檔造成的。當內部簡稱用沒問題，但**不得寫成「論文 Eq. 7a」**。
PDF 轉出的 Markdown 副本**已刪除**。轉檔器把 LaTeX 反斜線當成 C 跳脫字元，
`\frac` 變成換頁符加 rac，`\neq` 變成**真正的換行** —— 換行是合法字元，
控制字元普查抓不到，所以它撐過了兩輪「已修好」。留著一份會靜默出錯的副本，
就是留一套競爭權威。要看散文請直接讀 `docs/paper/*_source/*.tex`。

```
Level 0 — MetaFind 本身（作者 arXiv TeX source）
  docs/paper/metafind_source/   neurips_2025.tex + 2methdology / 3experiments
                                / appendix / 4backgound（見 SOURCE_MANIFEST.json）
  公式逐條清單：docs/audit/A_FORMULA_INVENTORY.md

Level 1 — 相依元件的「原論文」（同樣以 TeX source 為準）
  docs/paper/ulip2_source/      main.tex — 明文 "freeze it during the pre-training"
  docs/paper/egnn_source/       sections/model.tex — φ_x → R^1「outputs a scalar value」
  docs/paper/idesign_source/    main.tex + supplementary.tex（含 60 條 prompt 清單）

Level 2 — 相依元件的官方「實作」（證據，且低於它自己的論文）
  salesforce/ULIP        注意：ULIP-2 factory 未設 requires_grad=False，與其論文不一致
  vgsatorras/egnn        EGNN 參考實作
  atcelen/IDesign        I-Design 公開實作

Level 3 — 我們的復現決策
  02_BUILD_STEPS.md      最新決策，人類可讀
  01_GRAPH_SPEC.md       graph 設計、U／D／RA 登記表
  graph_spec.yaml
  node_registry.yaml     機器可讀契約
  validation_plan.yaml

Level 4 — 我們的實作
  metafind/、tools/、setup/

00_FINDINGS.md           實測事實（F 系列）與決策紀錄
```

> **兩條規則。**
>
> **一、Level 1／2 只能回答「這個相依元件怎麼定義／怎麼實作」，不能自動補上 MetaFind 沒寫的部分。**
>
> **二、相依元件的「論文」高於它自己的「實作」。** 這一層是後來才加的，
> 而它的缺席正是 D-1 出錯的原因 —— 當時拿 ULIP-2 **程式**沒有 `requires_grad=False`
> 去論證「凍結是我們的偏離」，但 ULIP-2 **論文** §3.3 明文凍結。
> 兩者不一致時，論文說的是設計，程式只能說「公開版本長這樣」。
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
| 4 | [`graph_spec.yaml`](graph_spec.yaml) | 機器可讀：56 個 state channel、69 條邊、16 組 join policy、11 個決策點、3 個 cycle、UNKNOWN 登記表 |
| 5 | [`node_registry.yaml`](node_registry.yaml) | 38 個節點 + 4 個 subgraph，含逐節點 failure policy 與 rollback |
| 6 | [`validation_plan.yaml`](validation_plan.yaml) | 64 個 L1、18 個 L2、7 個 gate、4 個 Required Audit |

## 一頁摘要

`hierarchical DAG + stateful + parallel`，主線零回邊；3 個 cycle 全封在 subgraph 內。
`control_authority: A1` ／ `execution_mode: probabilistic` ／ `topology_class: workflow`
—— Qwen 出現四次（資產標註、語意邊、場景評分、I-Design 規劃）但**從不決定路由**，只產生 payload。

### 論文只要兩個資料集

| | 內容 | 我們怎麼取得 |
|---|---|---|
| **Objaverse-LVIS** | manifest 實際 46,052 個資產（論文說「約 48,000」） | 下載 GLB **原始 mesh 並保留**；點雲與 11 視角渲染圖都從 mesh 產生 |
| **ProcTHOR-10K** | 10,000 train + 1,000 val + 1,000 test | JSONL |

**不下載**：ULIP-2 預先取樣的點雲（185 GB）、ULIP-2 的渲染圖（474 GB，而且不是論文要的
11 正交視角）、ShapeNet triplets（409 GB）。

### 十二項偏離（D-2…D-13）＋一項條件式（D-1，已判定不啟用）

| id | 偏離 | 影響 |
|---|---|---|
| **D-1** *(條件式・**不啟用**)* | ViT-bigG-14 的 CLIP 側保持凍結。`active_if: paper_clip_train_scope == 'trainable' AND actual_clip_train_scope == 'frozen'`；實際兩者皆為 `frozen`，條件不成立 | **U-34 已於 2026-08-16 判定為 `frozen`，D-1 因此不啟用。** 理由不是「4090 塞不下所以偏離」，而是：MetaFind 明確建立於 ULIP-2，ULIP-2 §3.3 明文凍結 OpenCLIP，而 MetaFind 全文從未聲明改變此策略。§2.6「Both query and gallery encoders are trained」講的是**塔**（point encoder／projection／fuser 本來就在 optimizer 裡），§3.4「entire encoder」對比的是 fuser-only ablation，§2.4「gallery frozen after pretraining」與 §2.6 是 Stage 1／Stage 2 的界線，不是矛盾。**不得寫成「論文明文說 CLIP 凍結」** —— 論文沒有這句。若日後取得官方 code 或作者回覆證實 optimizer 更新到 OpenCLIP，重開 U-34 並啟用 D-1。 RA-3 仍照跑，量的是**另一個讀法**在本機是否可執行，只記錄不阻斷 |
| **D-2** | Qwen3.8-27B 取代 GPT-4o（**資產標註 n05**）。使用者決定 U-6，2026-08-21 | **主要影響 Table 1** —— 45,952 筆標註是文字塔的訓練資料。SC-1 只報告差距、不設門檻。標註另錨定於 Objaverse-LVIS 真值類別，該錨定本身也是已記錄偏離。**GPT-4o 可用性為 UNRESOLVED，非已證實不可得** |
| **D-8** | Qwen2.5-VL 取代 GPT-4o（**場景評分 n17**）。2026-08-21 由 D-2 拆出 | **影響 Table 2**。裁判不是 GPT-4o，SC-3 僅保留方向性結論 |
| **D-3** | 不重跑 6 個 baseline | 只能與論文公佈值比較，並註明協定不同 |
| **D-4** | 不做人工評分 | Table 2 人工欄判 `INSUFFICIENT_EVIDENCE` |
| **D-5** | I-Design 中所有設為 `gpt-4`／`gpt-4-1106-preview` 的 LLM 路徑改導向 `qwen2.5-7b-instruct` | **與 D-2／D-8 不同**（那是 GPT-4o／標註與評分）。換規劃器改變**場景本身** → Table 2 全部與 Table 3 場景欄位移；**Table 1 不受影響**。做法是 patch `filter_dict`，**沒有別名** |
| **D-6** | 對 I-Design 的**行為性**修改（patch 02／03）：佈局引用正規化、丟棄懸空引用、合併重複 id、修正迴圈上限、重試換 seed、耗盡放棄場景 | 改的是管線**產出什麼**，不只是誰產出。**偏離的是公開實作** —— 論文作者的整合程式從未公開，不能斷言他們沒做類似修改 |
| **D-7** | I-Design 的 **JSON-constrained decoding 未重現**。補充材料 §7：*"All agents utilize GPT-4's JSON mode to restrict outputs exclusively to valid JSON"*，而我們的 vLLM 沒開任何 guided decoding。**與 D-5 不同**——D-5 是誰回答，D-7 是回答受不受結構約束。Qwen 因此**可能吐出結構上不合法的 JSON，GPT-4 在那個模式下不可能**，那會落進 Engineer 的 schema 驗證重試迴圈。分開編號是因為兩者可獨立修復：開了 guided JSON 就能退掉 D-7，D-5 原封不動  影響同 D-5：Table 2 全部與 Table 3 場景欄；Table 1 不受影響 |
| **D-9** | **n05 把 Objaverse-LVIS 真值類別餵給標註器當錨定身分**（`DL-007`），模型只能向下細化、不能橫向替換 | **論文是讓 VLM「產生」類別**（`2methdology.tex:28`、`neurips_2025.tex:100`、Figure 2 caption）。**餵標籤進去是departure，永遠不得寫成 paper-faithful。** 影響每一筆標註、每一個 Stage 1 文字向量、Table 1 每一個文字條件欄。**未解：`D0-010` 的證據稽核從未做過**，設計是靠批准通過的；`U-AB` 要求該稽核（ULIP2 `W-6`）必須在全量標註前完成 |
| **D-10** | **Stage 1 對比負樣本只有單卡 batch**，上游是 8 卡 all-gather 的 512（`F-N10-1`） | UPSTREAM FACT：`upstream/ULIP models/losses.py:38-40` 呼叫 `all_gather_batch`。**梯度累積補不回來** —— 每個 micro-batch 仍各自形成對比矩陣。負樣本數是對比目標的一階項，是 Table 1 落差的候選解釋。實際值在 batch size 定下來後**量測**，不是選的 |
| **D-11** | **n04 渲染背景為白色**，ULIP-2 官方渲染是黑色（`U-W`，USER 決定 2026-08-22） | 量測而非假設：對 ULIP-2 自己的 `image_feat`，全部 286 個重疊資產，白 R@1 **97.2%**／matched 0.9141／gap 0.3689，黑 95.8%／0.8783／0.3406，n=100 重現同號同量級。`S-5` 是本里程碑自選的判準，**判準不能贏的時候算、輸的時候不算**。影響 n04 語料與其所有影像向量；**不影響與論文的可比性**（論文未提背景） |
| **D-12** | **`COLOR_0` 從 `texture` 類別撤回**，牴觸 glTF 2.0（`R-12`） | glTF 2.0 定義 `COLOR_0` 為 base colour 的線性乘子，而 base colour = `baseColorFactor × baseColorTexture`，**涵蓋 texture 類**。全量測（n=37，該類與 ULIP 的完整重疊）：調變使 **37/37 變暗**，平均亮度 −0.2076，對 ULIP 自有點雲的 cosine 0.9005 → 0.8980。`SAMPLER_VERSION 6`，影響約 995 個資產的 rgb 通道。⚠️ **16/37 落在雜訊內**（37 次擲幣為 18.5±3），未做配對顯著性檢定；`R-12` 自己寫「變暗是確定的，『因此更差』不是」。此撤回**靠的是 `R-11` 的預設對齊上游規則，不是顯著量測**，而 `R-8` 已確立上游**根本沒有發布點雲上色程序** —— 只有產物，沒有行為可對齊。**永遠不得寫成「ULIP-2 就是這樣做的」** |
| **D-13** | **語料 46,052，論文說「約 48,000」**（`U-01`，2026-08-22 登記） | **不可避免，不是選的** —— 可取得的 Objaverse-LVIS manifest 就是 46,052 個 uid，且**全部都成功解析**（GLB 覆蓋率 100.00%，兩向差集皆為 0）。論文那份 48K 拿不到。`paper-reproduction.md` §9 要求不可避免的差異也要登記。**影響每一個分母**：依論文 80/20 切分，少約 1,558 個訓練資產與約 390 個測試資產 —— 這**不只是評估集不同**，因為 ULIP-2 是在 Objaverse-LVIS 上「評估」，MetaFind 是在上面「訓練」（`2methdology.tex:75`、`3experiments.tex:24`，皆 PAPER FACT）。依 `O-2` 當成 Table 1 的限制帶著；登記不代表重開該決定。manifest sha256 已補記於 `graph_spec.yaml`（`U-01` 自己寫的 resolution 從來沒被執行過） |
| **D-14** | **ESSGNN 用 `h⁰ = t_i`；論文 §2.5 字面是 `h_i^(0) = Concat(x_i, t_i)`**（`h0_mode="semantic"`，2026-08-22 登記） | 程式自己的註解就寫「**CONTRADICTS 2.5's literal**」。依 `C2` 採附錄 C 的前提，字面讀法保留為 `RA-1`。影響每個節點的初始狀態 → `e_layout` → Stage 2 → Table 2，而且它正是等變性測試拿來做負向注入的開關（`test_essgnn.py:138`）。**不是靜默風險**：`from_protocol` 不管協定寫什麼都強制它（Master 用敵意協定實測），兩個測試也在斷言。缺的只是登記。⚠️ `PRIMARY_INTERPRETATION` 另外三個值**不是偏離**，它們**遵循**論文：`coords_agg="sum"` 對應 `2methdology.tex:51-52` 的 `\sum`（偏離的是參考 EGNN 的 mean 預設，而依 `U-O` 論文有講就聽論文）、`edge_proj_dim=None`、`normalize_coord_diff=False`。**這三個「打開」才是偏離。** |

### 論文自身的四個矛盾

都不設 gate，改用 **Required Audit**（必跑、必留紀錄、**永不阻斷**）——
因為設成 gate 之後，唯一「讓它變綠」的方法就是放寬判準，那等於沒有檢查。

| id | 矛盾 | 預期 |
|---|---|---|
| **RA-1** | §2.5 的 `h⁰ = Concat(x,t)` vs Appendix C 的「`h⁰` 對 SE(3) 不變」前提 | **失敗** |
| **RA-2** | §2.5 的 `f_x → ℝ³` vs 證明需要 `φ_x` 為純量才能提出 `Q` | **失敗** |
| **RA-3** | **U-34 `trainable` 讀法的可行性稽核** —— `train_scope=full`（梯度真的到 ViT-bigG）在單卡 24GB 上跑不跑得動 | 跑不動只證明**那個讀法**在本機不可行，**不證明論文要求那個讀法**；凍結那條有 ULIP-2 §3.3 直接支持 |
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

82 個檢查（64 個 L1 ＋ 18 個 L2）對 7 個 gate。被降級的 gate 候選有 5 個，都寫明不符四判準的哪一條。

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

### 實作狀態 —— 31 個非 gate 節點裡，**有程式的是 17 個**

規格完整不等於管線存在。這張表是為了讓讀者不會把前者讀成後者
（第十九輪剛因為同一個理由修過 `L1-STAGE1-PROTOCOL-APPLIED` 的措辭）。

| 節點 | 狀態 | 程式 |
|---|---|---|
| `n01_env_bootstrap` | **可執行** | `setup/01_storage.sh`、`02_conda_env.sh`、`03_verify_env.py`（10/10 通過，含 AI2-THOR headless 渲染與 procthor-10k 載入） |
| `n02_download` | ✅ **完成** | `metafind/data/download.py`。46,052 個 GLB（351 GB）、0 失敗 |
| `n03_sample_pointclouds` | ✅ **完成** | `metafind/data/pointclouds.py`。46,052 朵點雲（5.6 GB）、**0 隔離**；顏色對照官方 ULIP 雲平均差 0.0021；19 條測試 |
| `n04_render_views` | ✅ **完成** | `metafind/data/renders.py`。45,955 個資產（7.3 GB）、**隔離率 0.21%**（G3 門檻 2%）；11 張視圖全相異且無空白；11 條測試 |
| `n05_annotate` | **可執行** | `metafind/data/annotate.py`（schema／prompt，27 條測試）＋ `annotate_run.py`（Qwen2.5-VL 生成與 C1 修復迴圈）。全量尚未跑完 |
| `n07_scene_graphs` | ✅ **完成** | `metafind/data/scene_graphs.py`。12,000 間房、0 隔離、房間對應 100%；support 邊來自 ProcTHOR 的 children 樹，座標保留原始值；16 條測試，兩條負向注入實測會失敗 |
| `n08_semantic_edges` | **可執行** | `metafind/data/semantic_edges.py`（key／prompt／驗證，30 條測試）＋ `semantic_edges_run.py`（Qwen 關係句 ＋ 凍結 CLIP 文字編碼器）。**實測 410 萬條語意邊只有 4,242 組唯一描述配對（快取命中 99.90%）**；三條負向注入實測會失敗。LLM 階段須等 n05 讓出 GPU |
| `n09c_build_scene_splits` | ✅ **完成** | `metafind/data/scene_splits.py`。9,600 train／2,400 test（80/20，seed 20260816）、**洩漏 0**；13 條測試，負向注入實測會失敗。語意邊覆蓋率待 n08 |
| `n05b_resolve_stage1_encoding` | **可執行** | `metafind/models/resolve_stage1.py`。釘死 U-15 文字模板（golden-string 測試）、U-14 取 11 視圖平均、U-11 learned token；**U-34 已判定 `frozen`**，連同 basis 與 confidence 一併記錄，主線再無執行期歧義。25 條測試 |
| `n07b_procthor_asset_modalities` | ✅ **完成** | `metafind/data/procthor_modalities.py`。**1,467 / 1,467、隔離 0、28 個無點雲**（透明材質，F26）。相機協定**由 `renders.py` import 而非抄寫**；點雲以 AI2-THOR 自報的 bounding box 驗證反投影，判準需**比例與絕對誤差同時超標**（單一判準兩個方向都誤報過，F26）|
| `n11b_stage2_gallery_index` | **只有規格** | U-08a 判定後新增：用凍結的 Stage 1 塔編碼 ProcTHOR 資產，Stage 2 專屬索引 |
| `n15a_resolve_eval_scene_protocol` | **只有規格** | U-27：Table 2 的 200 個 I-Design 請求，**仍需人決定** |
| `n09_build_splits` | **可執行** | `metafind/data/splits.py`。物件 80/20 ＋ **U-09 的兩種評估協定並行**（gallery=test 與 gallery=full），gallery_size 由切分推導、不寫死；18 條測試 |
| `n06_encode_text_image` | **可執行** | `metafind/data/encode_text_image.py`。凍結 bigG 編碼文字與 11 視角；**11 個 per-view 向量全部保留**（只存聚合後的會把 U-14 烤死在 46,052 個檔案裡）；token 數實測不假設 |
| `n10_train_stage1` | **可執行** | `metafind/train/stage1.py`。凍結 CLIP 的向量走 n06 快取、**點雲即時編碼**（PointBERT 在 optimizer 裡，快取等於變成 fuser-only ablation）；checkpoint 只存 requires_grad 參數（F27）；17 條測試 |
| `n11` ＋ `n12` ＋ `n11b` | **可執行** | `metafind/train/gallery_index.py`。三個節點同一支程式，因為它們是同一個操作套在不同語料上，而**不能漂移的正是編碼器** —— 拆開會有三份「載入、凍結、雜湊」，而那個雜湊就是重點。Stage 1 的 Objaverse 索引與 Stage 2 的 ProcTHOR 索引**永不合併** |
| `n09b_resolve_stage2_protocol` | **可執行** | `metafind/models/resolve_stage2.py`。把 U-08a/b/d/e 與四個 ESSGNN 選擇寫成 n13 讀得到的形式，並在寫入前**用 `ESSGNNConfig.from_protocol` 驗一遍** —— 一個 Literal 打錯要在一秒內失敗，不是等 Stage 1 訓練完 |
| 其餘 **十四個**節點 | **只有規格** | 無 |

另有兩塊不對應任何節點、但已可執行：

| 元件 | 用途 |
|---|---|
| `tools/check_graph.py` | 六份規格文件的一致性檢查（跑一次就知道項數，此處不寫死） |
| `setup/04_idesign_env.sh` ＋ `tools/idesign_generate.py` ＋ 三個 patch | I-Design 場景生成，R-01 的量測對象。**目前 0 個場景完成**（見 **F18**） |

`metafind/models/`（`essgnn` / `dual_tower` / `fusion` / `losses` / `ulip_backbone` /
`stage1_config`）是 `n10`／`n13` **會用到的元件**，不是那兩個節點本身 ——
`n10_train_stage1` 與 `n13_train_stage2` 都還沒有 trainer。
492 個測試函式涵蓋六個模型模組、取樣器、渲染器、標註 schema、場景圖建構、語意邊、場景切分、Stage 1 編碼協定與 ProcTHOR 資產模態（pytest 參數化後展開成 614 個 case），**沒有一條涵蓋任何節點的執行**。

---

### 其他重大未解項

**R-01：已部分實測。** I-Design **裝得起來**（README 要的 MinkowskiEngine／dgl／torch 1.12
都不需要，`requirements.txt` 的 `ag2==0.2.0` 在 PyPI 上不存在），
`create_initial_design` **會成功**，但 Qwen2.5-7B 跑 5 次**0 個場景完成**，每次失敗路徑不同。

**沒有基準，所以不能斷定那是缺陷** —— I-Design 沒用原版規劃器在本機跑過。
`setup/patches/` 的三個 patch 沒有論文依據，其中兩個會改變場景與完成率。

**[讀 I-Design 原論文後修正]** §5.2 把「物件多、房間小就可能放不下」列為**第一項已知限制**，
而先前的 smoke 設定（15 件放進 16 m²）正好落在那個區間。仍然沒有完成率基準，
但結論從「有東西壞了」改成「可能就是論文描述的行為」。另外兩件事：prompt 論文
Table 4／5 其實有給，先前那兩條是我編的（已換成原文）；`JSON mode` 我們沒有繼承，
那是 D-5 一個先前沒記下的差異。詳見 **F18**。

**U-17：ESSGNN 用 `d` 還是 `d²`。** §2.5 寫 `‖x_i − x_j‖₂`，Appendix C 用 `‖·‖²`。
兩者都是 SE(3) 不變、都不破壞證明，但**餵進 MLP 的數值不同，訓練結果就不同**。
實作依 Appendix C 與原始 EGNN 用平方，記錄為選擇。

**U-22：論文沒有公佈任何訓練超參數。** optimizer、learning rate、batch size、
epochs、weight decay、scheduler、`τ`、`λ` 初值、ESSGNN 的 `L` 與 hidden 寬度、pooling ——
一個都沒有。全部列在 `01_GRAPH_SPEC.md` §15，否則最後對不上時，
分不清是模型沒復現還是 recipe 不同。
