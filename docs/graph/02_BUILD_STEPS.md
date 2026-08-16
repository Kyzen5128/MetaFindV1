# MetaFind 復現 — 建置流程

> 2026-08-15 全面改寫。前一版有六個會**實際改變實驗結果**的錯誤，逐條修正見末尾「修正紀錄」。
> 每一步都標明：論文原文怎麼說 / 論文沒說什麼 / 我們怎麼做。
>
> **三種東西必須分開，不可混為一談：**
> **[論文]** 原文明確規定 ｜ **[未定]** 論文沒說，我們選了一個並記錄 ｜ **[偏離]** 與論文不同，必須在報告中聲明

**正式偏離六項（D-2…D-7）＋條件式一項（D-1）**，編號與 `README.md`、
`graph_spec.yaml` 一致，不得另編。D-1 放在 `boundary.conditional_deviations`，
`active_if: paper_clip_train_scope == 'trainable' AND actual_clip_train_scope == 'frozen'`。

> **舊條件是反的。** 先前寫 `clip_train_scope == 'trainable'` —— 那會在 run
> **確實訓了 CLIP、根本沒有偏離**的時候標記 D-1 為 active，
> 而 D-1 真正描述的狀態（論文要訓、我們凍了）**完全無法表示**。
> U-34 現在拆成兩個欄位：`paper_` 是我們對論文的解讀，`actual_` 是這次實驗怎麼跑，
> **只有 `actual_` 會分支 graph、只有它會到 backbone**。

| id | 偏離 |
|---|---|
| **D-1** *(條件式・`resolved_inactive`)* | ViT-bigG-14 的 CLIP 側保持凍結。**U-34 已於 2026-08-16 判定為 `frozen`**，故 `paper = actual = frozen`、`active_if` 為 false，**不列為 active deviation**。規則保留供日後重開 |
| **D-2** | Qwen2.5-VL 取代 **GPT-4o**（資產標註與場景評分） |
| **D-3** | 不重跑 6 個 baseline |
| **D-4** | 不做人工評分 |
| **D-5** | I-Design 中**所有**設為 `gpt-4`／`gpt-4-1106-preview` 的 LLM 路徑改導向 `qwen2.5-7b-instruct` |
| **D-6** | 對 I-Design 的**行為性**修改（patch 02／03） |
| **D-7** | I-Design 的 **JSON-constrained decoding 未重現**。補充材料 §7：*"All agents utilize GPT-4's JSON mode to restrict outputs exclusively to valid JSON"*，而我們的 vLLM 沒開任何 guided decoding。**與 D-5 不同**——D-5 是誰回答，D-7 是回答受不受結構約束。Qwen 因此**可能吐出結構上不合法的 JSON，GPT-4 在那個模式下不可能**，那會落進 Engineer 的 schema 驗證重試迴圈。分開編號是因為兩者可獨立修復：開了 guided JSON 就能退掉 D-7，D-5 原封不動 |

**D-6 改的是「產出什麼」，不是「誰產出」。** patch 02／03 會正規化佈局引用、
丟棄懸空引用、合併重複 id、給修正迴圈上限、每次重試換 seed、耗盡時放棄場景 ——
每一項都改變場景內容、哪些場景存活、以及完成率，因此 **Table 2 全部與 Table 3 場景欄都受影響**。

> **邊界要講清楚：** 這些修改不存在於**目前公開的** `atcelen/IDesign`，
> 但那**不等於**論文作者跑的是未修改版 —— 他們的整合程式與 I-Design 設定從未公開。
> 誠實的說法是**我們偏離的是公開實作，不是「論文所做的事」**。

（先前這裡只寫「五個規劃 agent」；公開 repo 至少六個 AssistantAgent 角色，
所以改成描述 patch 實際做的事，不數 agent。）

**D-5 與 D-2 是兩件事。** D-2 換的是 GPT-4o（標註／評分），D-5 換的是 GPT-4（I-Design 規劃器）。
換規劃器會**改變場景本身**，Table 2 全部與 Table 3 場景欄一起位移；Table 1 完全不受影響。
做法是**直接 patch I-Design 的 `filter_dict`**（`setup/patches/idesign-01`），
模型名從頭到尾都是 `qwen2.5-7b-instruct`，**沒有用別名**。

本文先前把「GLB 不刪」編成 D-1、把 ViT-bigG 凍結編成 D-3，與上述兩份文件錯位，已更正。
**「保留 GLB」和「不提供 caption fallback」都不是偏離** —— 論文沒有相反規定，
它們只是工程決定，不進偏離清單，否則報告的 deviation traceability 會亂掉。

---

## 資料與路徑

```
data/ -> /mnt/data1/kyzen/MetaFind/
├── datasets/          原始資料，只讀
│   ├── objaverse-lvis/   manifest + 46,052 個 GLB
│   └── procthor-10k/     train/val/test.jsonl
├── models/            預訓練權重，不由我們訓練
│   ├── ulip2/            ULIP-2 checkpoint
│   └── hf-cache/         ViT-bigG-14、Qwen2.5-VL
└── outputs/           全部可從上面重新生成
```

路徑集中在 `metafind/paths.py`，模組內不寫死。

### 論文只要兩個資料集

**[論文 §2.3]**
> Object-Level Dataset: **Objaverse-LVIS** ... approximately 48,000 distinct 3D assets
> Scene-Level Dataset: **ProcTHOR** ... over 10,000 generated houses

**[未定 U-01]** 論文說「approximately 48,000」，釋出的 manifest 是 **46,052**。
程式一律用 `len(manifest)`，**任何地方都不得寫死 48000**。manifest 的 sha256 記入報告。

**不下載**：ULIP-2 預先取樣的點雲（185 GB）、ULIP-2 的渲染圖（474 GB）、ShapeNet triplets（409 GB）。
前兩者我們有 mesh 可以自己產，而且 ULIP 的渲染圖不是論文要的 11 正交視角。

---

## Phase 0 — 環境與資料取得

### Step 0.1　環境

```bash
bash setup/02_conda_env.sh && conda activate MetaFind
python setup/03_verify_env.py --full
```

驗 9 項：torch/CUDA、compat shim、純 torch FPS、vendored EGNN 未被竄改、EGNN forward shape、
SE(3) 等變性 smoke、determinism、儲存區與快取落點、ULIP-2 建模且 `pc_projection` 為 `(768, 1280)`。

**G1 的判準也包含 ProcTHOR 三個 split 齊全** —— 先前它的判準文字提到 ProcTHOR，
但那條 channel 根本沒進 graph，所以 gate 看不到它：Objaverse 齊全而 ProcTHOR 不存在時會 PASS。

### Step 0.2　下載

```bash
python -m metafind.data.download                    # 全部
python -m metafind.data.download --only glbs        # 只抓 mesh（最慢，~216 GB）
```

| 項目 | 大小 | 用途 |
|---|---|---|
| Objaverse-LVIS manifest | 13 MB | 定義 46,052 個資產 |
| ProcTHOR-10K → `procthor_dataset` | 395 MB | **必須進 graph state**，否則 G1 無從檢查它 |
| Objaverse-LVIS GLB | ~216 GB | **保留不刪**，見下 |
| ULIP-2 checkpoint | 384 MB | PointBERT／`pc_projection` 的**初始權重**（Stage 1 會繼續訓練它們），以及凍結的 CLIP 側 |
| ViT-bigG-14 | 9.5 GB | ULIP-2 的 text/image 編碼器 |
| Qwen2.5-VL-7B | 16.6 GB | 取代 GPT-4o |

**GLB 不刪除。** 前一版設計「渲完就刪」是錯的：
Algorithm 1 的 iterative composition 需要**真實幾何**才能放進場景，只有 embedding 不夠。
（這**不是偏離** —— 論文沒說要刪，這只是一個工程決定。先前誤編為「偏離 D-1」，
與 README／`graph_spec.yaml` 的 D-1 = ViT-bigG 凍結衝突，已更正。）

**[偏離 D-2] Qwen2.5-VL 取代 GPT-4o**（§2.3 明寫 GPT-4o）。
標註是文字塔的訓練資料，換標註器等於換文字分布 → 每筆標註記錄 `annotator_model`，
報告中列為偏離。

---

## Phase 1 — 資料處理

### Step 1.1　點雲取樣

**[論文]** 沒有規定點雲怎麼產生，只說資產是 3D assets。

**[未定 U-02]** ULIP-2 checkpoint 是在**它自己取樣的點雲**上訓練的（10,000 點、xyz+rgb）。
我們從 mesh 自行取樣，取樣方式若不一致，embedding 會偏離訓練分布。

**做法**：`G2_pc_sanity` 擋的是**結構有效性** —— 形狀 `(10000, 6)`、數值有限、
`pc_norm` 後質心≈0 半徑≈1、非退化，且自取樣雲能在 1,000 資產的探針集裡檢索回自己。
這些不成立時，每個 embedding 都是錯的，而且不會報錯，所以是 gate。

與 **ULIP 官方點雲**的比較保留為 `L2-PC-ULIP-REF` 診斷，**不擋**。
理由：論文從未說 MetaFind 沿用 ULIP 預取樣的點雲，而 §2.6 的 Stage 1 會 fine-tune
point encoder，encoder 本來就能適應我們的取樣 —— 「和官方雲不一樣」推不出「復現無效」。
**這項不是因為過不了才降級，它從來沒跑過**；是因為測錯命題才降級。
若日後找到作者明確說沿用 ULIP 前處理，它就變回 gate。

前處理必須完全複製 ULIP 的 `pc_norm`：質心置中、除以最大半徑。
checkpoint 是在這樣的點雲上訓練的，餵原始座標不會報錯，只會讓每個 embedding 偏離分布。

### Step 1.2　渲染 11 視角

**[論文 §2.3]**
> Each asset is rendered from **11 orthogonal viewpoints**

**[未定 U-03a]** 「orthogonal」不可能指 11 個互相正交的方向（三維最多 3 軸 6 向），
所以**正交投影**是合理的讀法 —— 但論文並沒有寫 "orthographic camera"，作者也可能只是
用詞不精確地指「分散的多視角」。**這不是已解決，是選擇**，而且它會改變 image embedding 的分布。
兩種投影都保留，選擇記入每筆 sidecar。

**[未定 U-03]** 相機擺位論文沒說。11 不對應任何標準配置（立方體 6 面、二十面體 12 頂點），
ULIP 自己的慣例是 30 個方位角，也對不上。預設 Fibonacci lattice（任意 N 都近似均勻），
軸對齊版本保留為選項，選擇記入每筆 sidecar。

**[未定 U-04]** 解析度論文沒說。用 224px 對齊 ULIP-2 慣例與 image tower 輸入。

**[未定 U-14] 11 張圖怎麼變成一個 `e_image`，論文完全沒說。**
Eq.6 只吃一個 `e_image`，但 §2.3 只說 render 11 views，中間的規則是空的：

```
11 張渲染圖  →  ???  →  e_image (1280-d)
```

候選：隨機取 1 張／固定取 1 張／11 個 embedding 平均／取 max／學一個 multi-view fusion。
**Table 1 七個條件裡有四個（Image Only、T+I、I+PC、T+I+PC）直接取決於這個選擇。**
真正要防的失誤不是「選錯」，而是**訓練時與評估時選得不一樣** —— `L1-IMAGE-AGGREGATION`
要求規則寫在 run config 裡、兩邊一致、並隨 embedding 一起記錄。

渲染前 mesh 置中、縮放到單位球，否則 image tower 學到的是**建模單位**而非形狀。
代價是絕對尺度歸零（見 Step 1.3）。

實測 31 ms/資產，46,052 個約 0.4 小時；瓶頸是下載不是渲染。

### Step 1.3　結構化標註

**[論文 §2.3]**
> annotations provide rich textual descriptions detailing attributes such as
> **object category, size dimensions, materials, and placement constraints**

> **注意「such as」。** 論文列的是**例子**，不是 schema。以下四項全部是
> **我們的實作契約，不是論文要求**：

| 我們規定 | 論文有沒有說 |
|---|---|
| 剛好這四個欄位 | 沒有 —— 原文是 "attributes **such as**" |
| 四個欄位全部必填 | 沒有 |
| `placement_constraints` 用封閉詞彙表 | 沒有 |
| `dimensions` 以公尺為單位 | 沒有 |

保留這些規定是對的 —— 沒有 schema 就沒有可驗證的產物，而 `placement_constraints`
是讓 layout-aware 檢索成立的訊號，也是 ULIP-2 現成 caption 沒有、因此**不能拿來替代**的原因。
但它們必須以 **[未定／實作選擇]** 的身分出現在報告裡，不能寫成「論文要求」。

**[未定 U-15] 標註怎麼變成 text encoder 的輸入字串，論文完全沒說。**
是 `"wooden dining chair"`？是 `Category: chair. Material: wood. ...` 這種帶標籤的多行紀錄？
還是直接餵 JSON？**這會直接改變每一個 text embedding**，因而改變 Table 1 的
Text Only、T+I、T+PC、T+I+PC 四欄。模板必須釘死並加 golden-string 測試（`L1-TEXT-SERIALIZATION`）。

**不提供 fallback 到 ULIP-2 captions。** 前一版把它設成便宜的預設分支是錯的：
那會讓實驗變成別的實驗。若真的要用，整份結果標 `DEGRADED`，不得當成主線復現。
（同樣**不是偏離** —— 論文沒說可以退回 caption，這是我們拒絕一條捷徑。）

**[本實作的限制 F13]** 我們在渲染前做了 unit-sphere 正規化（否則 image tower 學到的是建模單位），
代價是絕對尺度歸零 —— 1.8 m 桌子與 0.1 m 杯子正規化後同大小。
因此**在本實作中**，標註模型只看渲染圖，`size dimensions` 只能是類別先驗。

**這是我們的前處理造成的，不是論文的性質。** 論文沒有說 GPT-4o 只看得到渲染圖，
也沒有說渲染前要正規化 —— 它可能同時提供了 mesh 的尺寸 metadata。不能反推
「論文的 dimensions 也是猜的」。
prompt 明說渲染圖是 scale-normalised；同時把 mesh 的真實 `extents_m` 記進 sidecar，
讓模型的估計可稽核。

Schema 失敗走 bounded 修復迴圈（錯誤訊息餵回去，最多 2 次），
格式問題（fences、前後贅字、數字字串）直接接受，不浪費修復額度。

### Step 1.4　場景圖抽取

**[論文 §2.5]**
> nodes represent objects with 3D position $x_i$ and a text-derived feature $t_i$
> Spatial edges are extracted from physical layout constraints (e.g., adjacency, support)
> semantic edges are generated by prompting an LLM with **object descriptions**

**物理邊**：ProcTHOR 的 `children` 樹本身就是支撐關係（`Apple_24` 是 `Countertop` 的 child），
正是論文舉的「cup on table」，直接讀取而非用幾何推導。

**[未定 U-05] adjacency 的判準論文沒給**（沒有半徑、沒有鄰居數）。
房間平均 69 個物件、最多 245，全連接會是 6 萬條邊。預設 kNN（k=8）讓 degree 不隨房間大小暴增，
參數記入產物。

**語意邊 —— 這裡有兩個先前的錯誤。**

**其一，cache key。** 論文說是 **object descriptions**（Appendix C：「derived solely from
**object-level textual descriptions**」），不是 category。先前用
`(category_a, category_b)` 會把 `office chair + desk`、`dining chair + dining table`
壓成同一條關係。

**其二，描述從哪來。** 先前讓場景圖去讀 `annotations` —— 那是 **Objaverse-LVIS 的
資產標註**。但實測：

```
ProcTHOR assetId : Countertop_I_8x2, Fridge_19, Houseplant_11   (995 個)
Objaverse uid    : 867dfc95e96a4987...                          (46,052 個)
交集             : 0
```

**兩個命名空間完全不相交。** ProcTHOR 的節點特徵 `t_i` 必須來自 ProcTHOR
自己的 semantic metadata（§2.3 說它有提供），拆成獨立的 `procthor_object_text` channel。
**[未定 U-12]** metadata 怎麼變成句子，論文沒說。

cache key **改為**：
```
key = sha256(desc_i, desc_j, prompt_version, llm_model, text_encoder_version)
```
只有描述完全相同才重用。

**[未定 U-06] 語意邊要對哪些物件對，論文沒說。**
§2.3 只寫「prompting an LLM on object pairs」—— 全部對？只有物理鄰居？某個 kNN？
這直接影響 ESSGNN 的輸入。選一個並記錄，且列為報告中的未定項。

**[未定 U-20] 句子再由哪個 encoder 變成 `t_i`，論文也沒說。**
§2.5 只寫 `t_i ∈ ℝ^d`、稱它 "a text-derived feature"。文中確實提到一個
「frozen text encoder (e.g., CLIP or BERT)」，但那句話講的是**語意邊** `e_ij`，
不是 `t_i` —— 而且連那句都是 "e.g."。兩者是否同一個 encoder，論文從未說明。
這決定了 `d` 的值，也決定 `f_h : ℝ^(2d+1+e) → ℝ^d` 的實際寬度。記錄選擇與 `d`。

（順帶：Abstract 說 ESSGNN 捕捉 "spatial relationships and **object appearance
features**"，但 §2.5 說 `t_i` 是 **text-derived**。兩句不一致，也支持
「`t_i` 到底怎麼來」確實沒鎖。）

**[未定 U-19] 邊是有向還是無向，論文沒說。**
§2.3 只講有 physical 與 semantic 兩種邊，沒說方向性，也沒說
`relation(A, B)` 是否等於 `relation(B, A)`。這會改變 message passing ——
有向的 support 邊與對稱的 support 邊，`h` 的更新結果不同。
現行 `L1-SCENE-SUPPORT` 斷言「杯在桌下 → 雙向 support 邊」，
**那是我們的慣例，不是論文要求**，測試留著是為了鎖住慣例不漂移。

**座標不正規化。** 場景座標保持原始世界座標。

理由不是「正規化會破壞等變性」—— 那在數學上不成立，單純置中 `x'_i = x_i − x̄`
不會讓 EGNN 變成非等變。真正的理由是：**論文明確以 unnormalized、未對齊的開放世界座標
為前提**（§2.5 "large and often unnormalized coordinate systems, with no guarantee that
scenes are aligned or centered"）。預先置中等於把它想解決的那個全域平移先消掉，
之後就測不到論文宣稱的能力。

測試釘住：整棟房子平移 100 公尺，座標必須跟著平移、邊結構必須不變。

### Step 1.5　資料劃分

**[論文 §3.1]**
> We allocate **80% of the data for training and reserve 20% for testing**

物件級 80/20（`n09_build_splits`）、房屋級 80/20（`n09c_build_scene_splits`，依 `house_id` 切，
不是依 room 或 object）。

**[前一版 dependency 錯誤] 這兩件事拆成兩個節點。**
先前由同一個 `n09` 同時做，於是它必須讀 `scene_graphs`、`procthor_object_text`、
`sem_edge_cache` —— 等於把 **Qwen 對 ProcTHOR 產語意邊**放進了 Stage 1 的關鍵路徑。
但 §2.6 的 Stage 1 是 **Objaverse-LVIS 上的物件級預訓練**，完全不需要 ProcTHOR。
ProcTHOR 分支的任何故障都會停掉一個不依賴它的訓練。兩條分支現在直到 `G6` 才匯流。

**[未定 U-09] 論文沒說 Table 1 的 query 就是那 20%。**
§3.1 只有「80% 訓練、20% 測試」這一句。gallery 範圍我們跑兩個協定都報，
但 **query = test split 是我們的假設**，同樣要標成假設，不能寫成論文規定。

**[前一版錯誤] 不再強制「同一 asset 不得同時出現在 train 與 test 房屋」。**
論文沒有這個要求，而 ProcTHOR 本來就是 12,000 間房共用約 1,467 個資產庫 —— 強制 disjoint
會改變 ProcTHOR 原本的場景分布。房屋集合不相交是必要的；資產集合是否相交列為**額外協定**，
不是必要驗證。

**[未定 U-07]** ProcTHOR 官方自帶 10k/1k/1k 三個 split，與論文的 80/20 不一致。
兩種切法都記錄，主線用論文的 80/20。

---

## Phase 2 — 訓練

### Step 2.1　Stage 1：跨模態對齊

**[論文 §2.6]**
> **Both query and gallery encoders are trained** on large-scale object-level data
> from Objaverse-LVIS ... each modality in the query has a **30% probability of being
> independently masked**. Rather than zero-padding, we apply **masked embeddings**

**[論文 §3.4]**
> **Fine-tuning the entire encoder outperformed training the fuser only**

**[前一版最嚴重的錯誤]** 我原本設計成「凍結 backbone、預先算好 embedding、只訓練 head」。
那正是 Table 3 的 `Train fuser only` 那一列 —— **論文明確報告它較差（8.7 vs 11.4）**。
把它當主線等於一開始就跑錯實驗。

**改為三個等級。第二個是本復現選定的判讀與主線執行方式。**
MetaFind 未逐 module 明說誰訓練；本復現依其對 ULIP-2 的繼承關係，將 U-34 判讀為 `frozen`（2026-08-16，confidence: moderate）。第三個等級不是並列的另一種讀法，而是 RA-3 的 alternative 稽核對象：

| 等級 | 訓練什麼 | 4090 可行 | 定位 |
|---|---|---|---|
| `fuser_only` | 只有 fusion 層 | ✅ | **Table 3 的 ablation 列** |
| `point_encoder+fuser` | PointBERT (32.5M) + fusion + 投影 | ✅ | **主線** |
| `full` | 再加 ViT-bigG-14 (2.5B) | ❓ **未量測** | `actual=trainable` 的執行對象，由 **RA-3** 量測 |

**[D-1 —— 條件式偏離，已判定不啟用]** ViT-bigG-14 的 text/image 端保持凍結。

**U-34 已於 2026-08-16 判定為 `frozen`，D-1 因此不啟用。** 理由不是「4090 塞不下所以偏離」，而是：MetaFind 明確建立於 ULIP-2，ULIP-2 §3.3 明文凍結 OpenCLIP，而 MetaFind 全文從未聲明改變此策略。§2.6「Both query and gallery encoders are trained」講的是**塔**（point encoder／projection／fuser 本來就在 optimizer 裡），§3.4「entire encoder」對比的是 fuser-only ablation，§2.4「gallery frozen after pretraining」與 §2.6 是 Stage 1／Stage 2 的界線，不是矛盾。**不得寫成「論文明文說 CLIP 凍結」** —— 論文沒有這句。若日後取得官方 code 或作者回覆證實 optimizer 更新到 OpenCLIP，重開 U-34 並啟用 D-1。

**ULIP-2 論文明文凍結 CLIP。** §3.3：

> We adopt the largest version of encoders from OpenCLIP (**ViT-G/14**) [13] for most
> of our experiments and **freeze it during pre-training**. The feature space, already
> **pre-aligned** by OpenCLIP, serves as the target space where we aim to integrate the
> 3D modality. ... We extract the image feature and text feature **based on the frozen
> encoders**. We train the 3D encoder $E_P$ to align the 3D feature with the image and
> text features.

**[更正]** 先前這裡引的是「based on the **pre-aligned and frozen image encoder and
text encoder** in OpenCLIP」，那**不是原文**——它把上面兩句併成一句。原文是分開的
「already pre-aligned by OpenCLIP」與「based on the frozen encoders」。
凍結這件事沒有變，**但這份文件的整套紀律就是不准把改寫當引文**，
而 D-1 的撤回正是建立在這句話上。

（論文寫 **ViT-G/14**，公開程式載入的是 **ViT-bigG-14**。兩者不是同一個 open_clip
權重名稱，本文其餘各處講的都是**實際會載入的** ViT-bigG-14。這個差異出在 ULIP-2
自己身上，不是我們的選擇，也不影響「凍結」這句話。）

目標函數也只訓 3D encoder。所以**主線的凍結有 ULIP-2 論文的直接支持**。

> **先前這裡寫反了，記下來。** 我曾用「公開程式沒有 `requires_grad = False`」
> 論證「凍結是我們的偏離」。對**程式**的觀察沒錯 ——
> `ULIP2_PointBERT_Colored` 確實只呼叫 `eval()`，而 `main.py` 的 optimizer 是
> `if not p.requires_grad: continue`，所以那些參數會被收進去。
> 但**拿實作去論證設計是錯的**：同一份檔案裡 ULIP-1 的五個 factory
> （`ULIP_PN_SSG`、`ULIP_PointBERT` …）**都有**明確凍結，只有 ULIP-2 的沒有，
> 那比較像是它自己對不上自己的論文。
> **這正是這個專案一直在防的錯誤，只是這次是我犯的。**

至於 MetaFind 是否要求解凍 CLIP —— **U-34，2026-08-16 判定為 `frozen`**。
§2.6 寫 "Both query and gallery encoders are trained"、§3.4 寫 "fine-tuning the
**entire** encoder"、§2.4 還特地與「凍結 text/image encoder 的既有做法」對比；
但它**從未逐個 module 說誰訓練**。這三句都不足以推出 CLIP 被訓練：§2.6 講的是**塔**
（PointBERT／projection／fusion 本來就在 optimizer 裡，兩種讀法下都成立），
§3.4 對比的是 fuser-only ablation（光是 point encoder 可訓練就造成這個對比），
§2.4 與 §2.6 的差異是 Stage 1／Stage 2 的界線。
**判讀依據**：MetaFind 明確建立於 ULIP-2；ULIP-2 §3.3 明文 "freeze it during pre-training"；MetaFind 全文未逐 module 聲明改變此策略。**不得寫成「MetaFind 明文說 OpenCLIP frozen」** —— 論文沒有這句。重開條件：取得官方 code 或作者回覆，證實 optimizer 更新到 OpenCLIP 參數。

報告中須聲明「entire encoder」在我們的設定下指 3D encoder + fusion，不含 CLIP。

**只有「點雲」的 embedding 快取限定 `fuser_only` 那一列。**
text／image 走凍結的 ViT-bigG-14 時本來就該快取——那是 `actual=frozen` 之下的正確做法，
不是妥協，也不預設 D-1 成立。`actual=trainable` 時**不得快取**，
而 Stage 1 之後由 `n10b_post_stage1_encode` 用訓練後的 encoder 重編（見 `post_stage1_embeddings`）。

```
text / image   凍結 → 主線就該快取，省掉每個 epoch 重跑 2.5B 參數
point cloud    主線可訓練 → 不可快取
```

理由是機制而非慣例：**embedding 快取按定義就是「某個不再更新的網路」的輸出**。
在主線上快取點雲 embedding，等於把 point encoder 凍住，
那就是 `fuser_only` ablation ——不管 `train_scope` 寫什麼。
`L1-STAGE1-CACHE-DISCIPLINE` 就是釘這件事。

（先前這段寫成「快取 embedding 只用於 `fuser_only`」，把三個模態混為一談。）

Loss 為 Eq.5，**單向 query→gallery**。ULIP 現成的 `ULIPWithImageLoss` 是單塔 tri-modal，不能用。

**[未定 U-13] Full model 用哪一種 fusion，論文沒說 —— 而且它給了兩份不同的清單。**

> **§2.2**：integrated via a fusion layer (e.g., **mean pooling, an MLP, or a
> Transformer-based module**)　←　三種
>
> **§2.4**：combines these modality embeddings via one of several strategies,
> such as **mean pooling, MLP, masked MLP, gated fusion, or Transformer-based
> fusion**　←　五種

**兩份清單自己就不一致**（§2.4 多了 masked MLP 與 gated）。
「such as」+「one of several」= 它在描述一個**選項集合**，不是在指定 MetaFind 用哪個。
能從 Table 3 推出來的只有排除：

```
Fusion = Mean   →  9.4        排除
Fusion = MLPs   →  9.9        排除
Padding with 0  → 10.5        排除 zero-padding（§3.4：Masked modality fusion
Full            → 11.4                            outperformed zero-padding）
```

所以 Full **會遮罩缺席模態**，且不是 Mean、不是普通 MLP。剩下三個候選：
**masked MLP / gated / Transformer**，論文無法再縮小。

程式現行預設是 `masked_mlp`（`metafind/models/fusion.py`）—— 這個選擇合理
（名稱與 §3.4 的 "Masked modality fusion" 直接對應），但**它不是論文真值**，
必須以 U-13 的身分出現在報告裡。另外兩個保留為可選並列為對照。

**[未定 U-16] query 塔與 gallery 塔是否共享權重，論文沒說。**

§2.4 說 "training a **dedicated** query encoder"、稱兩者為
"query encoder / gallery encoder"；§2.6 說 "**Both** query and gallery encoders are trained"。
但從未說清楚：

```
A. backbone 共享、fusion 各自一份
B. 兩塔完全各自一份
C. 全部共享
```

這件事對雙塔的意義很關鍵 —— 論文自己在 Table 1 底下特別指出，baseline 的 PC-Only
之所以虛高，是因為 query 與 gallery 用的是**同一個 embedding**，而 MetaFind 不是。
共享政策直接決定這個差別有多大。記錄選擇。

### Step 2.2　Gallery 索引

**[論文 §2.7]** > all gallery asset embeddings are precomputed and cached

Stage 1 完成後凍結 gallery 塔，對全部 admitted 資產編碼。
先寫 staging，驗證後才 promote（late commit）—— gallery 索引是所有 Table 的共同基準，
壞掉的索引一旦 promote，事後分不出哪些數字被污染。

驗證判準：維度正確、數量等於 manifest、無 NaN／零向量、
**抽 1000 筆自我檢索，目標相似度 == 最大相似度且目標在 argmax tie set 內**
（撈不回自己就是索引壞了；`recall@1 = 1.0` 不是 tie-safe —— 兩筆相同的 embedding 會讓 argmax 回傳另一個 id）。

### Step 2.3　Stage 2：佈局感知微調

**[論文 §2.6]**
> `e_query = Fusion(e_text, e_image, e_pc) + λ · e_layout`（Eq.6，λ 可學）
> stochastic scene dropout (30%)：30% 的批次省略 e_layout
> **Only the query-side fuser and the ESSGNN module are updated; the gallery encoder is frozen**
> 雙向對比 Eq.7a/7b，平均為 Eq.8

注意 Stage 1 是**單向**、Stage 2 是**雙向**，這個差異是論文明寫的。

**「Only the query-side fuser and the ESSGNN module are updated」要驗兩邊。**
這句話排除的不只是 gallery 塔，也包括 **query 側的 text / image / point encoder**。

先前只有 `L1-GALLERY-FROZEN` 驗 gallery 側，但真正危險的是 **query 的 PointBERT**：
Stage 1 的主線 `train_scope = point_encoder+fuser` **會訓練它**，所以它進入 Stage 2 時
`requires_grad` 本來就是 `True`，不明確凍結就會繼續訓練 —— 而且不會有任何錯誤。
新增 `L1-STAGE2-QUERY-ENCODERS-FROZEN`：跑完一步之後，query 的 text / image / point
三個 encoder 都必須與 Stage 1 逐 bit 相同，只有 fusion、ESSGNN 與 `λ` 可以動。

**[U-08a／U-08b —— 2026-08-16 判定。U-08（樣本怎麼組）仍為未定]**

> **判定摘要**：Stage 2 使用**自己的 ProcTHOR gallery**，正樣本就是**同一個 assetId**，
> 完全不需要 ProcTHOR→Objaverse 對應表。ProcTHOR 側的模態由 AI2-THOR **隔離渲染**產生
> （與 n04 同協定），點雲為多視角深度外殼，query 側允許缺點雲。
> 依據見 `graph_spec.yaml` 的 `U-08a`／`U-08b` `decision_basis`。

論文從未定義 Stage 2 的訓練樣本怎麼從 ProcTHOR 建構。拆成三個缺口，
**第二個讓這個階段根本建不起來**：

### U-08　樣本怎麼組

- 目標物件怎麼選（隨機？依放置順序？）
- 「current scene」是目標以外的全部，還是某個前綴？
- 一間房產生幾筆樣本、負樣本怎麼取

### U-08a　正樣本是哪一個 gallery 條目 —— **已判定：同一個 ProcTHOR assetId**

Eq.7a/7b 需要一個 positive。目標是 **ProcTHOR 物件**，gallery 是 **Objaverse-LVIS**，
而兩者的識別碼**交集為 0**（實測 995 vs 46,052，完全不相交）。

**但那個「必須對應」的前提是我們自己加的。** 論文 §2.6 寫的是
「Only the query-side fuser and the ESSGNN module are updated; the gallery encoder
is frozen」—— 凍的是**權重**；Eq. 7a/7b 的分母是 batch `B`，**沒有任何一句把 Stage 2
的 gallery catalog 綁在 Objaverse**。retrieval 系統裡 encoder 與 index 本來就是兩件事，
凍結 encoder 不代表 index 不能換。

**所以正樣本的身分就是 `Fridge_19 → Fridge_19`。**
可能的讀法（都未經證實）：

| | 做法 |
|---|---|
| (a) | 把每個 ProcTHOR 資產對應到最接近的 Objaverse 資產（依類別或 embedding） |
| (b) | 場景階段另外用 ProcTHOR 自己的 ~1,467 個資產建一個 gallery |
| (c) | 用目標物件自己的模態當正樣本，讓 Stage 2 變成自我檢索目標 |

**選定 (b)。** (a) 與依 embedding 對應都是假標籤；(c) 最危險 —— 場景與正樣本之間沒有
ground truth 關係，模型可以完全忽略 ESSGNN 仍然把 loss 降下去，算得出數字卻沒有在學
scene-aware compatibility。已寫進 `stage2_pairing` channel，**報告中列為選擇而非論文規定**。

**[實測 F25]** 那個 gallery 是 **1,467** 個資產，不是論文轉述的「3,000+」。
它就是 Eq. 7a/7b 的負樣本池大小，比 Stage 1 的 46,052 小得多，訓練訊號因此較弱 ——
報告必須寫我們實際的數字。

### U-08b　目標物件的三個模態從哪來 —— **已判定：AI2-THOR 隔離渲染**

**[更正 F24]** 先前這裡寫「ProcTHOR 沒有渲染圖也沒有點雲」。那對 JSONL 成立，
對 ProcTHOR 不成立 —— 房子本來就是給 AI2-THOR 載入的，而 AI2-THOR 會渲染。
實測可取得與 n04 同協定的 11 視角正交隔離渲染。

唯一真缺口是點雲：n03 從完整 mesh 取樣（含被遮蔽面），這裡只能從 11 張深度圖
反投影成**可見外殼**，而 gallery encoder 凍結、PointBERT 沒有機會適應這個位移。
論文自己的設計吸收了這一點：§2.4 明說 query encoder 接受任意模態子集，
所以 **query 走 text+image、點雲選配**，gallery 維持 §2.6 要求的 modality-complete。

**分類為 [未指定・高影響復現選擇]，不是 [偏離]。** 偏離是「論文說 X、我們做 Y」；
這裡是論文要求 Stage 2 的模態卻從未指定 ProcTHOR 的模態怎麼產生 ——
我們是在實作一個缺失的協定，不是違反一個已陳述的協定。

### 因此

`n13_train_stage2` 的 `reads` 已補上 `stage2_pairing`、`pointclouds`、
`post_stage1_embeddings`（**不是 `text_image_embeddings`** —— `actual=trainable` 下 `n06` 不執行，那個 channel 不存在；Stage 1 之後由 `n10b` 產出）、`procthor_object_text` —— 先前的清單根本湊不出 Eq.6 的輸入，
也不知道正樣本是誰。

**U-08a 與 U-08b 決定之前，不要實作這個階段。**

這件事現在由 graph 強制，不靠自律：

```
n09b_resolve_stage2_protocol   ← human 節點，做出這兩個決定
        ↓
G6_stage2_ready             ← G-INVALID gate
        ↓
n13_train_stage2
```

`stage2_protocol.status` 未達 `resolved` 之前，G6 回傳 **`BLOCKED_EVIDENCE`(rc=3)
而不是 FAIL** —— 沒有東西壞掉，只是有個決定還沒做。Stage 2 以外的階段照常進行。

先前把 `stage2_pairing` 交給 `n09_build_splits` 寫、而且是 `write_once`，
在答案還不存在時寫入空值就會**把 channel 永久鎖死**。現在拆成
「可改的決定（`stage2_protocol`）」與「決定後才落定的對照表（`stage2_pairing`）」。

---

## Phase 3 — 評估

### Step 3.1　Table 1：物件級檢索

7 種模態組合 × 我們的變體。必須自寫 instance-level 檢索評估器 ——
ULIP 的 `test_zeroshot_3d_core` 做的是 zero-shot **分類**（`pc @ text_prompt`，target 是類別 id），
不是 48K gallery 的實例檢索。

**[未定 U-09 — 前一版這裡錯了] Gallery 到底是什麼**

**[論文 §2.1]** > retrieves the asset $A^*$ from a **pre-encoded asset database** $\mathcal{A}$
**[論文 §3.1]** > 80% training / 20% testing

論文沒說檢索時的 gallery 是全部 46,052 還是只有 20% 測試集。差別是隨機命中率 5 倍。

**前一版打算「用 baseline PC-Only ≈ 98–99% 反推分母」—— 那是錯的。**
PC-Only 是 query embedding 等於它自己的 gallery embedding，
**無論 gallery 是 46K 還是 9.2K，自我檢索都會趨近 100%**，根本無法區分。

**改為兩個協定都跑，都報**：
```yaml
protocol_A:  query = test split,  gallery = test split   (~9,210)
protocol_B:  query = test split,  gallery = full         (46,052)
```
產出 `R@1_A / R@5_A / R@1_B / R@5_B`，不再鎖成單一數字。

**預期要看到的**：MetaFind 的 PC-Only（63–75）**低於** baseline（98–99）。
那是正確的復現結果，不是失敗 —— 論文自己註腳解釋了原因。

### Step 3.2　等變性驗證

**[論文 Eq.4]** > $(R x^{l+1} + T, h^{l+1}) = \text{ESSGNN}(R x^l + T, h^l, E)$
**[論文 §2.5]** > $e_{\text{layout}} = \operatorname{Pooling}(\{h_i^{(L)}\})$

**[前一版錯誤]** 我原本寫成
`‖ESSGNN(Rx+T) − (R·ESSGNN(x)+T)‖ < 1e-4` 並稱之為「座標通道」。
**這對 `e_layout` 沒有幾何意義** —— 它是從 `h` pooling 來的，`h` 是**不變量**，
不能對它做 `R·(...)+T`。

**改為三個分開的測試**：

| 層級 | 斷言 |
|---|---|
| 層內座標 | `x^{l+1}(Rx+T) ≈ R·x^{l+1}(x) + T`　（**等變**） |
| 層內特徵 | `h^{l+1}(Rx+T) ≈ h^{l+1}(x)`　（**不變**） |
| layout 輸出 | `e_layout(Rx+T) ≈ e_layout(x)`　（**不變**） |

（程式碼裡測的本來就是不變性、是對的；錯的是規格文字。）

**另有兩個論文自身的矛盾要各自 audit，不得阻斷**：
- **RA-1**：§2.5 的 `h⁰ = Concat(x, t)` 與 Appendix C 的「h⁰ 對 SE(3) 不變」前提衝突。字面版**預期失敗**。
- **RA-2**：§2.5 的 `f_x → ℝ³` 與 Appendix C 的證明衝突（提出 `Q` 需要 `φ_x` 是純量）。已實作為純量。

**[未定 U-17] 還有第三處不一致：`d_ij` 到底是距離還是距離平方。**

```
§2.5        d_ij^l = ‖x_i^l − x_j^l‖₂          歐氏距離
Appendix C  m_ij = φ_e(h_i, h_j, ‖x_i − x_j‖², e_ij)   (10)，(11)(12) 同
原始 EGNN                     ‖·‖²             平方
```

這一項**與 RA-1／RA-2 性質不同，不設 audit**：兩者都對旋轉與平移不變，
所以**都不會破壞等變性證明**，沒有「預期失敗」的斷言可寫。
但**餵進 MLP 的數值不一樣，訓練出來的模型就不一樣**。

實作依 Appendix C 與原始 EGNN 用**平方**（`metafind/models/essgnn.py` 的 `radial`）。
這是選擇，不是推導，必須記錄；要量它的話，跑一個變體即可，很便宜。

### Step 3.3　Table 2 / 3：場景級

Algorithm 1 逐物件檢索並放置，需要 **I-Design** 與**真實 mesh 幾何**（所以 GLB 不刪）。

**[R-01 —— 已部分實測 2026-08-15]**

| 項目 | 結果 |
|---|---|
| 能不能裝 | **能**。README／Dockerfile 要的 MinkowskiEngine、dgl、torch 1.12 **都不需要**（只有 `retrieve.py` 用，而那正是 MetaFind 取代的）。另：`requirements.txt` 的 `ag2==0.2.0` **PyPI 上不存在**，要用 `pyautogen==0.2.0` |
| 能不能啟動 | **能**。`create_initial_design` 完成並通過 I-Design 自己的 schema 驗證 |
| 能不能產出場景 | **不能**。Qwen2.5-7B 跑 5 次、**0 個完成**，每次失敗在不同的下游路徑 |

**但沒有基準，所以不能斷定那 5 次是缺陷。** I-Design 沒用它原本的規劃器在本機跑過。

**[2026-08-15 補充，讀 I-Design 原論文後]** 論文 §5.2 把這件事列為第一項已知限制：
*"The pipeline **may fail** to find a solution for object placements when handling
**many objects in a relatively small scene**."* 而先前的 smoke 設定（`n=15` 放進
16 m²）正好落在那個區間 —— 比論文 Table 1 任何一個臥室場景都密。
仍然沒有完成率基準，但結論從「有東西壞了」改成「可能就是論文描述的行為」。
prompt 已改用論文 Table 4 的原文（先前那兩條是我編的）。詳見 **F18**。`setup/patches/` 的三個 patch 是**為了讓場景跑得完而做的工程決定，
沒有論文依據**，其中 02、03 會改變場景與完成率（不是格式調整）。

**[U-21 —— 2026-08-16 判定為讀法 B]** Table 2 的資料流先前根本沒有閉合。

> **判定**：ProcTHOR 只用於 Stage 2 訓練；Table 2 的 200 個場景、query list 與 layout
> **全部由 I-Design 產生**。§3.1 與 §3.3 因此不衝突 —— 它們在講不同階段。
> **驗證**：我們自己的 `tools/idesign_generate.py:166` 呼叫的是
> `IDesign(no_of_objects, user_input, room_dimensions)` —— 吃文字 prompt 與房間尺寸，
> **沒有任何介面吃 ProcTHOR house**。
> 「randomly sampled scenes」略偏讀法 A，但它從未說 sampled **from ProcTHOR**，
> 200 個 prompt 或 configuration 在語法上同樣成立，所以不足以定案。
> 讀法 A 保留為診斷（ProcTHOR leave-one-out 場景補全），**不作為 Table 2 結果**。

論文 §3.3：

> We evaluate MetaFind on the **scene generation pipeline of I-Design** on a set of
> **200 randomly sampled scenes**

Algorithm 1 的 `Require:` 需要**初始場景圖 `G_0`** 與 **asset query list `{Q_1..Q_N}`**。
先前 `n16_compose_scenes` 讀的是 `scene_graphs` —— 那是 **ProcTHOR 房屋**。
**ProcTHOR 房屋是已經完成的佈局，不是生成請求**，兩者不是同一種東西。
graph 裡從來沒有 I-Design 輸出的 channel，所以也沒有任何 gate 會發現這件事。

現在補上 `evaluation_scene_inputs` channel（`{g0_uri, query_list, room_type, source_revision}`），
由 `n15c_prepare_eval_scenes` 產生，而它排在 `G7_composition_protocol` 之後。

**[U-18 —— 2026-08-16 判定]** 「放進場景、更新場景圖」到底產生什麼，論文一個字都沒說。

> **判定：擁有 slot 的是佈局來源，檢索器永遠不發明位置。**
> Stage 2 訓練沿用被移除物件在 ProcTHOR 的原始 pose；Table 2 沿用 I-Design 規劃的 slot。
> 自己寫放置規則被否決 —— 那會讓 Table 2 變成「MetaFind 檢索 ＋ 我們自製的放置演算法」，
> 美感與合理性分數就無法歸因給檢索。
> **新節點的 `t_i` 用檢索到的資產的標註文字，不是 query 文字。** §2.1 定義節點是
> **既有**物件；第 7 行之後，既有的是被檢索到的那一個。沿用 query 文字等於讓場景圖
> 描述「想要的世界」而不是「現在的世界」，與論文自陳的
> 「continuously adapting retrieval results to current scene updates」相違。
> 兩個字串都保留，只有 `t_i` 用後者。

Algorithm 1 第 7 行只有：

```
7:  Place A*_i into the scene, update scene graph: G <- G U {A*_i}
```

但下一輪的第 3 行立刻是 `e_layout <- ESSGNN(G)`。要讓它有定義，新放進去的那個節點必須有：

| 需要 | 論文有沒有說 |
|---|---|
| `t_i`（節點文字特徵） | 沒有 |
| 位置 `x_i` | 沒有 |
| 朝向、尺度 | 沒有 |
| 新的物理邊（support / adjacency 接到誰） | 沒有 |
| 新的語意邊（要不要為新物件生關係句） | 沒有 |

而**這個選擇會改變之後每一次檢索** —— 這正是論文說 iterative 比 parallel 好的機制所在。
`sg4_place` 先前只寫「用真實幾何放置、更新 placed_assets」，不足以定義下一輪的輸入。

**這兩項決定之前不要正式跑 Table 2**，由 `G7_composition_protocol` 擋著，
未決回 `BLOCKED_EVIDENCE`(rc=3)。**Table 1 不經過這道 gate，照常進行。**

**Table 3 的 `w/o iterative retrieval` 不需要重訓一個模型。**
Algorithm 1 是 §2.7 的**推論期程序**，不是模型結構。那一列是
**同一個 Full checkpoint** 換成 `composition_mode: parallel` 再評估：

```
訓練型 ablation（n18）        GAT / Mean / MLP / dropout 10 / dropout 50 /
                              fuser only / zero padding / w/o Layout Context
推論型 ablation（n19 直接評）  w/o iterative retrieval
```

先前 `n18_train_ablations` 宣稱要訓練「八個變體」，把這一列也算進去 ——
那會變成拿兩個不同的 checkpoint 比較，然後把差異歸因於合成策略。
`variant_registry` 現在有 `requires_training` 與 `reuses_ckpt` 兩個欄位，
由 `L1-ABLATION-INFERENCE-ONLY` 釘住。

（`w/o Layout Context` 保留為需要訓練：Table 1 把 `MetaFind w/o ESSGNN` 列為獨立模型，
且它在 Table 3 的 R@1 是 13.5、Table 1 的 text-only 是 13.8，兩者不同 ——
理論上它也可能只是把 `λ·e_layout` 設成 0，但重訓是比較保守的讀法。同樣記錄為選擇。）

**`n18` 先前的 reads 根本湊不出一次訓練。** 它只宣告
`{variant_registry, stage1_ckpt, splits, scene_graphs}` 四條，
但 `n13` 做同一件事需要十一條 —— 要訓練 GAT 變體或 fusion 變體，
至少還需要點雲、快取的 text/image embedding、語意邊、ProcTHOR 物件文字、
以及已決議的 Stage 2 協定。那是一份**不可能被滿足的 dependency contract**，已補齊。

**[偏離 D-2]** 場景評分用 Qwen2.5-VL 取代 GPT-4o。
IDesign 自帶的 `gpt_v_as_evaluator.py` 是 5 個面向 1–10 分，論文 Table 2 是 4 個面向 1–5 分，
論文沒有公佈它改過的 prompt → **[未定 U-10]**，其中 Scene Coherence 對應哪個面向不明。

**換掉裁判之後，Table 2 的絕對數字與論文不再可比**，只有方向性（w/ESSGNN 是否優於 w/o）還成立。

人工評分不做，該欄判 `INSUFFICIENT_EVIDENCE`。

---

## 維度一律不寫死

**論文全文沒有出現任何維度數字**（`1280|768|512|128|64` 在論文中零命中）。
論文只寫 `t_i ∈ ℝ^d`、`f_h : ℝ^(2d+1+e) → ℝ^d`。

因此：

| 參數 | 來源 |
|---|---|
| query/gallery embedding 寬度 | **由 ULIP-2 checkpoint 決定**（實測 `pc_projection` 為 `(768, 1280)`） |
| 語意邊寬度 `e` | 由文字編碼器決定（論文只說 "e.g., CLIP or BERT"） |
| ESSGNN hidden、層數、pooling | **超參數**，不是論文真值 |
| 語意邊投影 | 論文**沒有**這一層，預設不投影 |

前一版把 `Linear(1280→64)` 寫成「改掉就必須測試失敗」是錯的 —— 那是我加的，不是論文的。

---

## 修正紀錄（相對於前一版）

| # | 前一版 | 現在 |
|---|---|---|
| 1 | Stage 1 凍結 backbone、只訓 head 當**主線** | 主線改為訓練 point encoder + fusion；凍結版降為 Table 3 的 ablation 列 |
| 2 | `gallery_size_locked` 單一整數，用 PC-Only 反推 | 雙協定並行，兩組數字都報；PC-Only 無法反推分母 |
| 3 | `e_layout` 寫成等變、可加 `R·(...)+T` | 分成層內座標等變 / 層內特徵不變 / `e_layout` 不變三個測試 |
| 4 | 語意邊 cache key = `(category_a, category_b)` | key = 兩個 object description 的 hash |
| 5 | Stage 2 訓練樣本建構**完全未提** | 列為 U-08，並明列我們採用的協定 |
| 6 | 強制 train/test 房屋不得共用 asset | 移除；論文沒這要求，且會改變 ProcTHOR 分布 |
| 7 | 渲染後**刪除 GLB** | 保留；Table 2 需要真實幾何 |
| 8 | GPT-4o 可 fallback 成 ULIP captions（且是預設） | 移除 fallback；真要用則整份標 `DEGRADED` |
| 9 | 多處寫死 `48000` | 一律 `len(manifest)`（實際 46,052） |
| 10 | `1280 / 128 / 64` 當論文真值並設 L1 測試 | 改為 checkpoint 推導值與超參數 |

## 主要未定項摘要

> **這不是完整清單。** 權威登記表是 `01_GRAPH_SPEC.md` §15 與
> `graph_spec.yaml` 的 `risks_unknowns`（目前 38 條，含 U-34／U-35）。
> 本節只摘錄與建置步驟直接相關者。

完整登記表在 `01_GRAPH_SPEC.md` §15，機器可讀版在 `graph_spec.yaml` 的 `risks_unknowns`；
`G5_report_release` **逐項**檢查處置（不得用區間表示）。

| id | 內容 | 影響哪張表 |
|---|---|---|
| U-01 | 資產數：論文「約 48,000」vs manifest 46,052 | 全部分母 |
| U-02 | 自行取樣的點雲與 ULIP 官方點雲是否一致（**降為診斷**） | — |
| U-03 | 11 個視角的相機擺位 | Table 1 影像欄 |
| U-03a | 11 視角用正交還是透視投影 | Table 1 影像欄 |
| U-04 | 渲染解析度 | Table 1 影像欄 |
| U-05 | adjacency 的判準 | ESSGNN 輸入 |
| U-06 | 語意邊要對哪些物件對；`e_ij` 寬度 | ESSGNN 輸入 |
| U-07 | ProcTHOR 官方 split vs 論文 80/20 | Table 2/3 |
| U-08 | Stage 2 樣本怎麼組（目標選擇、partial scene、負樣本） | Table 2/3 |
| **U-08a** | **正樣本是哪一個 gallery 條目** —— ProcTHOR 與 Objaverse 識別碼交集為 0（**阻斷**） | Stage 2 全部 |
| **U-08b** | **目標物件的 text/image/pc 從哪來** —— ProcTHOR 沒有渲染圖也沒有點雲（**阻斷**） | Stage 2 全部 |
| U-09 | Table 1 的 gallery 範圍**與 query 範圍** | Table 1 全部 |
| U-10 | Table 2 的 Scene Coherence 對應 IDesign 哪個面向 | Table 2 |
| U-11 | 缺席模態怎麼表示（論文只排除 zero-padding） | Table 1 部分模態欄 |
| U-12 | ProcTHOR metadata 怎麼變成 `t_i` 的句子 | ESSGNN 輸入 |
| **U-13** | **Full model 用哪一種 fusion**（§2.4 列五種、沒說是哪個） | **Table 1 全部** |
| **U-14** | **11 張渲染圖怎麼變成一個 `e_image`** | **Table 1 四欄** |
| **U-15** | **標註怎麼序列化成 text encoder 的輸入字串** | **Table 1 四欄** |
| **U-16** | **query / gallery 兩塔是否共享權重** | **Table 1 全部** |
| U-17 | ESSGNN 用 `d` 還是 `d²`（§2.5 vs Appendix C） | 所有訓練結果 |
| **U-18** | **Algorithm 1 放置後圖怎麼更新**（**阻斷**） | **Table 2 全部** |
| U-19 | 邊是有向還是無向 | ESSGNN message passing |
| U-20 | `t_i` 由哪個 encoder 產生 | ESSGNN 輸入與 `d` |
| **U-21** | **Algorithm 1 的 `G_0` 與 query list 從哪來**（**阻斷**） | **Table 2 全部** |
| U-22 | **訓練超參數論文一個都沒給** | 所有訓練結果 |
| U-23 | 三個模態同時被遮罩時代表什麼（獨立 30% → 2.7% 的 query 全空） | Stage 1 訓練訊號 |
| U-24 | `sim(·,·)` 從未定義 | 所有 loss 與排序 |
| U-25 | §2.2 的「adaptive freezing strategies」全文沒有定義 | Stage 2 最佳化 |
| U-26 | `f_h`／`f_x` 是否共用一條訊息，**以及 `f_x` 看到的是 `h^{l+1}` 還是 `h^l`** | ESSGNN 參數化 |
| U-27 | I-Design 自己的輸入（prompt／房間尺寸／物件數）論文全沒給 | Table 2 |
| U-28 | Table 1 在 layout-free 資料集評 `w/ ESSGNN` 時 `e_layout` 是什麼 | **Table 1 那一列 7 格** |
| **U-29** | **物理邊怎麼進 ESSGNN**（`f_h`／`f_x` 只吃一個 `e_ij`，而它是**語意**邊） | **ESSGNN 架構** |
| **U-30** | **沒有語意嵌入時，固定寬度的 `e` 格填什麼**（禁止補零，但記旗標≠說明張量） | **ESSGNN 架構** |
| U-31 | ESSGNN 的 L 層是否共用參數（`θ` 沒有層索引） | 參數量、F11 是否成立 |
| **U-33** | **ESSGNN 有沒有保留 EGNN 的輸入／輸出投影**。§2.5 是 `t_i → h⁰ → L 層 → Pooling`，**兩端都沒有投影**；官方 `egnn_clean.py` 有 `embedding_in`／`embedding_out`，本實作沿用了 | **架構層級差異，不是超參數**。`use_io_projections` **無預設值**，必須由 `essgnn_arch_protocol` 指定，`G6` 檢查 |
| **U-32** | **scene dropout 的粒度**。§2.6 寫 "omitted in 30% of **batches**"（整批），實作原本是每 sample 獨立抽。**注意同節的 modality masking 才是明寫 "independently"**，兩句不是一回事 | **Stage 2 訓練分布**。主線已改為 batch-level；`stage2_protocol.scene_dropout_granularity` 記錄，`G6` 檢查 |

**U-29／U-30 由 `essgnn_edge_protocol` 承載，U-33／U-17／U-26／U-31／U-22 由
`essgnn_arch_protocol` 承載，兩者 `G6` 都在 Stage 2 訓練前強制。**

`use_io_projections` 特別值得說：一個 `bool = True` 的預設**不是決定**，
是官方 EGNN 的慣例靠繼承勝出。它現在**沒有預設值**，不指定就 `TypeError`。
選 `False`（字面復現 §2.5）的代價要一起講：那會強制
`node_feat_dim == hidden_dim == out_dim`，hidden 寬度變成 embedding 寬度（ULIP-2 是 1280），
網路大很多，而且與 F8「1280 寬的 `e_ij` 把幾何訊號壓下約 45 倍」的實測互相作用。
**這個取捨是人要做的判斷，所以它是 human 節點的輸出，不是 dataclass 的預設值。**
登記成 UNKNOWN 是不夠的 —— `essgnn.py` 已經替它們做了決定（假定每條邊都有固定寬度的語意嵌入）。
