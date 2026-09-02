# PHASE 1 現況盤點 — 2026-09-03

依 `REPRODUCTION_PROTOCOL_20260903.md` §十八 PHASE 1 與 §二十 的四十題。
全程唯讀：沒有重渲、沒有標註、沒有訓練、沒有刪除，`git status` 在每一路開始與結束時都乾淨。

三路平行，各自負責一段。本檔隨每一路回報逐段補上。

| 段 | 題目 | 負責 | 狀態 |
|---|---|---|---|
| A | 1–15、18、19　Objaverse 資料 | ULIP2 Engineer | **完成，見下** |
| B | 16、17、20–28　程式碼 | Integrator | **完成，見下** |
| C | 29–32　ProcTHOR | ESSGNN Engineer | **完成，見下** |
| D | 33–40　migration | MASTER | **完成，見下** |

證據標籤：協定**設定**用規格 §一 的四類；**量測**用專案原有的
`OBSERVED DATA` / `OBSERVED IMPLEMENTATION`。一個量測不是一個設定，兩套標籤不混用。

---

# C　ProcTHOR（第 29–32 題）

盤點方式：對 `/home/kyzen/metafind_data/outputs/` 全語料 `os.scandir`，
不用 shell glob，不用沒有 `-L` 的 `find`。以下數字皆為實測。

## Q29　磁碟上的 ProcTHOR schema　［OBSERVED DATA］

| 產物 | 筆數 | 形狀／型別 | 產生者 |
|---|---|---|---|
| `scene_graphs/*.json` | 12,000 間房 | 共 827,730 個節點 | `scene_graphs.py`（n07），`builder_version 1` |
| `procthor_object_text.json` | 1,467 個 assetId | 93 個相異字串 | 同上 |
| `procthor_modalities/<asset>/` | 1,467 目錄 + 1,467 sidecar | 每個 11 視角；1,439 個點雲 `(10000,3) float32`；28 個沒有 | `procthor_modalities.py`（n07b） |
| `procthor_node_embeddings.npz` | 1,467 | `(1467,1280) float32`，L2 範數恆為 1.0 | `semantic_edges_run.py`（n08） |
| `sem_edge_cache.json` | 4,242 | degraded 0 | n08 |
| `sem_edge_embeddings.npz` | 4,242 | `(4242,1280) float32`，L2 恆為 1.0 | n08 |
| `sem_edge_sentences.jsonl` | 4,242 行 | 續跑用的追加日誌 | n08 |
| `scene_splits.json` | 9,600 訓練 / 2,400 測試 | 交集 0，seed 20260816 | `scene_splits.py`（n09c） |

節點欄位在全部 827,730 個節點上完全一致：
`index, id, asset_id, category, room_id, position`。
`positions[i]` 與 `nodes[i].position` 逐值相同。
`adjacency_criterion` 在 12,000 間房上一致為 `{knn, k=8}`。
`house_id` 是我們給的（`<split>_<行號>`）——ProcTHOR 本身不發房號。

點雲：1,439 個存在，28 個沒有，原因記在 sidecar
（AI2-THOR 的深度預先掃描只帶不透明幾何）。`n_points` 分布 `{10000: 1439, 0: 28}`。
sidecar 的 `text` 與 `procthor_object_text.json` 在 1,467 筆上零不一致。

Stage 2 端的衍生物：`stage2_gallery_index.json` 記 1,439 筆、1280 維，
並帶 `stage1_checkpoint_sha256 00f591a0…`；`stage2_positive_map.json` 1,439 筆，
`method: identity`。1,439 = 1,467 減掉 28 個沒有點雲的，與
`target_eligibility = has_modalities_and_pointcloud` 一致。

## Q30　物理／布局關係的來源

**支撐邊 — ProcTHOR 自帶。**［OBSERVED IMPLEMENTATION］
`scene_graphs.py:138` 的 `_flatten` 深度優先走訪 ProcTHOR 自己的 `children` 樹，
父子關係即支撐。482,826 條。對稱化是**我們的**慣例（`:212`，[U-19]），不是資料集的。

**相鄰邊 — 我們自己算的。**［IMPLEMENTATION CHOICE / MAINLINE］
`scene_graphs.py:173` 的 `_knn_pairs`，在**原始世界座標**上取 k=8 最近鄰，
再扣掉已經是支撐的（`:213`）。3,645,811 條。實測支撐與相鄰在 12,000 間房上零重疊。
**論文完全沒有給相鄰的判準**，所以這條規則整個是我們的選擇，
並且以 `adjacency_criterion` 隨每一個產物一起走。

**在場景圖檔案這一層分得開：可以。** `phys_edges.support` 與 `phys_edges.adjacency`
是兩份不同的清單，判準也附在旁邊。§十三 的要求在這一層成立。

**進到 ESSGNN 就分不開了。**［OBSERVED IMPLEMENTATION］
`scene_graphs.py:219` 做 `sem_edge_ids = sorted(set(support) | set(adjacency))`，
實測在 12,000 / 12,000 間房上成立（482,826 + 3,645,811 = 4,128,637，精確相符）。
`stage2.py:264` 的 `build_context_graph` 只走 `sem_edge_ids`，
回傳的 `(keep, pos, edge_index, edge_attr, edge_missing)` 裡**沒有任何通道**
能分辨一條邊是支撐還是相鄰。

這是已記錄的決定而不是缺陷（`essgnn_edge_protocol.physical_relation_encoding
= "neighbourhood_only"`：物理邊只負責決定鄰居集合）。列出來是因為 §十三 問的正是
「兩種來源分不分得開」，而答案在場景圖檔案之後就是分不開。

**座標是原始 ProcTHOR 世界座標**，從未置中或正規化（`scene_graphs.py:49`）。

## Q31　LLM 語意關係已經存在　［OBSERVED DATA］

**存在，而且已經跑完。這與原始碼註解相矛盾** —— `semantic_edges.py:97` 仍寫著
「n08 還沒跑」，`:11` 仍寫著用 Qwen。兩句都是舊的。

| | |
|---|---|
| 數量 | 4,242 對相異描述，覆蓋全部 4,128,637 條語意邊（收斂率 99.90%，是類別層級 `t_i` 的直接後果）。degraded 0 |
| 模型 | `gemma-4-12B-it`。`semantic_edges_run.py:85` 匯入 `annotate_run.MODEL_ID`，`:97` 取檔名而非路徑進 cache key，所以搬動權重不會讓 4,242 句失效 |
| 提示 | `semantic_edges.py:143`，`PROMPT_VERSION = 1`。只餵兩段描述，沒有距離、沒有方向、沒有房間，並在 `:165` 對座標字眼做斷言 |
| 解碼 | `do_sample=False`（貪婪），`max_new_tokens=64`，兩次嘗試後降級；實際降級 0 次 |
| 時間 | 2026-09-01 19:47:34 → 20:12:54，1,519.7 秒，rc 0，`code_revision d402e09f`，加上 19:46 的 8 對煙霧測試 |
| 舊的一輪 | 2026-08-17 用 Qwen2.5-7B-Instruct 跑過。`llm_model` 在 cache key 裡，所以換模型時整批失效，句子檔重置。實測 jsonl 裡剛好就是這 4,242 把 gemma 金鑰，沒有孤兒 |

`logs/n08_full.log` 是 Qwen 那一輪的日誌，**不是**現在磁碟上這批產物的日誌。

## Q32　provenance 有沒有分開記　有，而且乾淨

`sem_edge_cache.json` 自帶表頭（`llm_model`、`text_encoder`、`text_encoder_version`、
`prompt_version`、`edge_dim`），與 `scene_graphs/` 分屬不同檔案。
場景圖裡沒有任何 LLM 中繼資料，語意快取裡也沒有任何物理關係中繼資料。

兩者的接合是在**載入時**計算的，不是存起來的：對
`[desc_a, desc_b, prompt_version, llm_model, text_encoder_version]` 取 sha256。

一致性逐項核對，全部通過：

```
cache 4,242 | jsonl 4,242 | npz 4,242
jsonl 金鑰 == cache 金鑰 == npz 金鑰      True
指向錯誤 npz 列的 cache 項                 0
語料需要 4,242 對 → 覆蓋 4,242、缺 0、多 0
degraded                                   0
```

場景圖裡唯一帶物理來源的欄位是 `sem_edge_ids`，而它正確地只是一份**候選清單**
（n07 無從得知 n08 會用哪個模型或哪版編碼器），不含語意內容。

---

## 補充一：`t_i` 今天到底是什麼

規格 §二十三 把節點文字的建構列為 UNRESOLVED。以下只描述現況，不主張替換。

```
ProcTHOR 物件 id "CounterTop|2|0"
  → scene_graphs.py:116 _category  → "CounterTop"   （類別藏在 id 裡，不是欄位）
  → :98  humanise                   → "counter top"
  → :103 object_text                → "a counter top"
  → n07 寫進 procthor_object_text.json
  → n08 encode_sentences → OpenCLIP ViT-bigG-14 文字塔（經 ULIPBackbone）
  → L2 正規化 → 1280 維 → procthor_node_embeddings.npz
  → stage2.py:335 node_feat → essgnn.py:644  h0 = node_feat（h0_mode "semantic"）
```

`t_i` 就是一個類別字串的凍結 CLIP 文字向量。93 個相異字串代表
**全語料的 `t_i` 至多只有 93 種值**：每一間房裡的每一張餐桌，節點特徵完全相同。
這是資料集的天花板，不是我們的選擇 —— ProcTHOR 沒有逐實例的描述欄位。

### 異常一：磁碟上的字串是修正前的版本　［OBSERVED DATA］

commit `2f255f5` 同時改了兩件事：駝峰切詞的正規表示式，以及冠詞規則（母音前用 `an`）。
所有 ProcTHOR 產物都早於那次修改。

```
磁碟文字 vs 舊版程式輸出：  0 筆不同   ← 磁碟就是舊版
磁碟文字 vs 現行程式輸出：146 筆不同   （共 1,467 個 assetId）
換算成節點實例：48,577 / 827,730 = 5.87%
其中僅由駝峰正規表示式造成的：15,209 / 827,730 = 1.84%
```

受影響的七個類別：`a alarm clock`、`an apple`、`an arm chair`、`an egg`、
`an ottoman`、`a cd`（磁碟上是 `a c d`）、`a tv stand`（磁碟上是 `a t v stand`）。

`scene_graphs.py` 註解裡的 5.9% 是**正確的**，但它的措辭把功勞全歸給正規表示式，
實際上這個數字涵蓋兩項修正。

### 異常二：93 個字串卻產生 98 個相異向量　［OBSERVED DATA］

五個字串（`a garbage can`、`a pillow`、`a statue`、`a toaster`、`a vase`）
的不同實例向量不完全相同。量級是 float32 的捨入誤差：最大絕對差 2.4e-7，
最大餘弦距離 4.2e-7。變體恰好落在 `ENCODE_BATCH=256` 的批次邊界上，
成因是 `encode_sentences` 的批次組成，不是語意錯誤。

嚴重度低，但值得寫下來：**`t_i` 是 (字串, 批次位置) 的函數，不只是字串的函數。**

---

## 補充二：ProcTHOR 有沒有綁在 Objaverse 的渲染協定上

**圖片與點雲：沒有綁在語料上。** 46K 重渲的決定不會卡住 ProcTHOR 這一側。

**但程式裡宣稱的那個保證並不成立。**［OBSERVED IMPLEMENTATION］
`procthor_modalities.py:57` 從 `renders` 匯入 `N_VIEWS`、`ORBIT_ELEVATION_DEG`、
`PROJECTION`、`RESOLUTION`，並在文件字串裡宣稱「用匯入而不是複製，所以
n04 相容性是由建構保證的」。但 `renders.py:226` 對這幾個常數本人的說明是：
「以下常數描述的是**已退役**的 pyrender 路徑……`process_one` 已經不再讀它們。」
真正在跑的 Objaverse 路徑走 `render_blender`，經 `LIVE_N_VIEWS`。

所以那個匯入把 ProcTHOR 綁在一條**死掉的程式路徑**上。這是「因為呼叫者才成立的性質」：
保證是程式層級的，讀者從那句話取走的結論卻是語料層級的。實測差異：

| | ProcTHOR | Objaverse |
|---|---|---|
| 視角 | 11 | 12 |
| 投影 | 正交 | 透視 |
| 解析度 | 224 | 512 |
| 佈局 | 單圈方位角，仰角 20° | OpenShape 三圈各四 |
| 背景 | AI2-THOR 天空盒 | 透明 RGBA |
| 渲染器 | AI2-THOR 5.0.0 | BlenderProc 2.8.0 / CYCLES |

**真正的耦合在 Stage 1，不在渲染器。** `stage2_gallery_index.json` 帶著
`stage1_checkpoint_sha256`，gallery 的 npz 檔名也嵌了它。那 1,439 筆 ProcTHOR
gallery 向量是用一顆在 Objaverse 語料上訓練的 Stage 1 權重算出來的。

```
重渲 Objaverse → 重訓 Stage 1 → ProcTHOR 的 1,439 筆 gallery 向量要重算
存活不動：1,467 份渲染、1,439 個點雲、12,000 張場景圖、4,242 句、1,467 個節點向量
```

---

## 做了會失敗的測試，結果

真實房子 `test_00000`（70 個節點，46 支撐 + 309 相鄰），目標索引 7，
走 `stage2.build_context_graph`，純 CPU。

```
[SE(3)，附錄 C 的前提] 繞 Y 軸轉 37 度，再平移 100 公尺
  edge_index 相同        True
  edge_attr  相同        True    最大差 0.0
  edge_missing 相同      True
  座標確實移動了          True    最大差 105.21398

[節點順序] 70 個節點隨機重排，邊重新對映
  邊數相同               True
  edge_attr 多重集相同    True
  保留的 asset_id 相同    True

[快取覆蓋] 200 間隨機房、66,603 條邊 → 今天有 0 條會落到 edge_missing
```

`e_ij` 對節點位置**確實**不變，而且是用一個真的剛體變換驗的，不是用文件宣稱的。
`essgnn.py:639` 的 `torch.where(edge_missing…)` 在第一層之前就套用，
所以「缺失記號不是補零」這句話是在程式碼裡驗證的，不只是寫在註解裡。

---

## 剩下的風險：一個會安靜咬人的順序陷阱

**這是我認為最該先處理的一項。**

用現行程式重新產生 `procthor_object_text.json`，卻**沒有在同一個動作裡重跑 n08**：

```
146 個 assetId 的描述改變
  → _edge_key 雜湊的是描述，所以金鑰改變
  → 查表落空 → 向量是 None → edge_missing = True
  → 學到的缺失記號安靜地頂上
  → 沒有任何錯誤，訓練照跑，Table 2 / 3 全錯
```

在同樣的 200 間房、66,603 條邊上實測：**7,702 條（11.56%）會安靜地變成「查無資料」。**

正確做法的成本：兩件事綁在一起做，新文字下仍是 4,242 對，其中
**613 對需要新的 LLM 呼叫**，依實測速率約 4 分鐘 GPU，加上 4,242 條邊與 1,467 個節點的
重新編碼約 1 分鐘。會連帶失效的是 `procthor_node_embeddings.npz`（及其 sha256 紀錄）
與所有下游 Stage 2 執行；**不會**動到 `scene_graphs/`、`procthor_modalities/` 與 gallery。

尚未執行，也沒有請求執行：§十九 禁止，且這是 MASTER 與 Kyzen 的決定。

---

## 一併觀察到、但沒有動的

1. `semantic_edges.py:11` 說 GPU 那半用 Qwen（實際是 gemma）；`:97` 說 n08 還沒跑
   （實際跑過兩輪）。讀者會對磁碟上那 4,242 句的來源下錯結論。
2. `renders.py:99` 的註解寫 `render_blender.N_VIEWS (12)`，實際值是 11
   （`render_blender.py:96`，由 `7785679` 於 2026-09-02 改動時沒有同步註解）。屬 ULIP2 範圍。
3. **`h0_mode` 不是 `essgnn_arch_protocol.json` 的欄位。** 它只以
   `essgnn.py:225` 的 dataclass 預設值 `"semantic"` 存在。
   `resolve_stage2.py` 自己的文件字串寫著「預設值是一個沒人寫下來的決定」——這就是一個。
   有關係是因為 `h0_mode="concat_xt"` 會把原始座標放進 `h`，破壞設計宣稱的不變性。
4. 4,242 句裡有 74 句（1.74%）含有「above」「next to」「on」這類相對位置詞。
   稽核者的判斷（並主動提出來讓人挑戰）：**這不破壞附錄 C**，因為
   `e_ij = f(desc_a, desc_b)` 而描述只取決於類別，所以同一組類別配對的句子恆定，
   與物件實際在哪無關 —— 剛體變換測試的最大差 0.0 就是證據。
   一個空間**詞**不等於一個空間**依賴**。
5. `scene_splits.semantic_edge_coverage` 是 null：切分發生在語意邊快取存在之前。
   切分是房子層級、與邊的內容無關，所以這是缺一個 provenance 欄位，不是洩漏風險。
   `train ∩ test = 0` 已驗證。

## 沒有驗證的（寫出來，免得被當成已經清乾淨）

* **沒有**評估那 4,242 句的語意品質。只核對了數量、覆蓋、一致性、決定性與位置無關性。
* **沒有**用眼睛看那 1,467 份 ProcTHOR 渲染。程式的文件字串自己記著，
  早先某一版「數字全過、圖片全錯」。
* **沒有**檢查那 28 個沒有點雲的資產是否讓 gallery 產生偏差。
* **沒有**跑 pytest。以上每一項都是對磁碟產物的直接量測。
* MetaFind 作者是否使用類別層級的節點文字，依 §二十三 為 UNRESOLVED，未嘗試解決。

---

# A　Objaverse 資料（第 1–15、18、19 題）

盤點方式：全語料 `os.scandir` / `os.walk(followlinks=True)`，
`find` 一律帶 `-L`。點雲陣列是 300 件隨機抽樣（種子 0），其餘皆為全掃。
每一項都註明是量到的還是讀來的。

## Q1　目錄樹　［OBSERVED DATA］

兩層符號連結，而且是第二層把資料藏起來的：

```
/home/kyzen/MetaFindV1/data  ->  /home/kyzen/metafind_data
  datasets/objaverse-lvis/
    glbs/glbs/000-000 … 000-159        46,052 個 .glb，351.4 GB（160 個目錄）
    glbs/hf-objaverse-v1/object-paths.json.gz   798,759 筆（整個 Objaverse，不是 LVIS 子集）
    lvis.json                          46,052 筆 uid
  outputs/
    annotations  -> /home/kyzen/metafind_out/annotations    45,692 個 .json
    embeddings   -> …/embeddings                            45,692 個 .npz + .json，1.55 GB
    pointclouds  -> …/pointclouds                           46,052 個 .npz + .json，5.73 GB
    renders      -> …/renders                               46,052 個目錄 + 46,024 個 .json
    checkpoints  -> …/checkpoints                           4 個 .pt，1.29 GB
    gallery_index_00f591a09ed19a04.npz                      217 MB
    logs/    _probe/（空）  eval/（空）  ladder/（空）
/home/kyzen/metafind_out/annotations_excluded          21 個
/home/kyzen/metafind_out/annotations_superseded_v8  2,095 個
```

全部落在 NVMe（`/dev/nvme0n1p2`，剩 368 GB），沒有一項在 SMR 碟上。

## Q2　原始 UID：46,052　［OBSERVED DATA］

三個彼此獨立的來源一致：`lvis.json` 46,052 個鍵；`os.scandir` 數到 46,052 個 `.glb`；
`pointclouds_index.jsonl` 46,052 行。

## Q3　可用 UID：45,692

定義出自 `splits.py:297 admitted_uids()`［OBSERVED IMPLEMENTATION］：
點雲索引 ∩ 渲染索引 ∩ 標註索引，再扣掉排除名單。
稽核者自己算了那個交集：45,692，與 `splits.json` 的宇宙、45,692 個標註檔、
45,692 個向量檔**逐一相同**。所以這裡的「可用」＝三個模態齊全，是建構上的模態完整。

## Q4　過濾階梯　［OBSERVED DATA，用集合差算的，不是讀隔離日誌的行數］

| 階段 | 之前 | 移除 | 之後 |
|---|---|---|---|
| 清單 `lvis.json` | — | — | 46,052 |
| n03 點雲 | 46,052 | 0 | 46,052 |
| n04 渲染 | 46,052 | **28** | 46,024 |
| n05 標註 | 46,024 | **332** | 45,692 |
| n06 編碼 | 45,692 | 0 | 45,692 |

n06 的隔離日誌有 2 列，但那兩個 uid 都在標註索引裡、也都有向量檔——重試後成功了。
**隔離日誌是跨重試累加的，所以裡面的相異 uid 數（n04 290、n05 326）不是移除數。**
只有集合差才是。

## Q5　隔離與過濾原因　［OBSERVED DATA］

* **n04 真正掉的 28 個**：687 列日誌裡 `DETERMINISTIC_INPUT` 674 / `UNKNOWN` 12 /
  `RESOURCE` 1，主訊息是「每一張都空白，資產從未進入畫面」。
* **n05 的 332 個**：`annotation_exclusions.json` 分得剛好——
  `n05_quarantine` **311**（產不出通過驗證的卡片，種子綁 uid 所以重跑是空操作）、
  `manual_review_rejected` **21**（clip 分數低於 0.20、類別關係分歧、兩種對決都選 LVIS，Kyzen 裁定）。
  那 21 個的標註檔保存在 `annotations_excluded/`，稽核者驗證了 21 個確實都在名單裡。
* n05 日誌的失敗分類：`MODEL_RECOVERABLE` 588（全部 `terminated_by: repair_budget`）/
  `DETERMINISTIC_INPUT` 52。

### ⚠ 發現一：排除名單其實沒有被套用　［OBSERVED IMPLEMENTATION］

`splits.py:322-329` 把 `annotation_exclusions.json` 當成 `{uid: 項目}` 來走訪。它不是。
它是一個中繼資料字典，鍵是 `decided_at`、`decided_by`、`decision`、`git_commit`、
`corpus_before`、`excluded_total`、`corpus_after`、`rendered_assets`、`groups`。
稽核者實際跑了那個迴圈，`admitted_uids()` 真正扣掉的是：

```
['2026-08-28T14:44:25+08:00', '332', '45692', '45713', '46024',
 '9e91457220fe625b3313d30c387364d592ee20a0', 'Kyzen', 'groups',
 '這21個 跟 失敗的 311 全部刪掉 其他所有接受採用標註結果']
```

九個字串，沒有一個是 uid。真正的 332 個 uid 躺在 `groups.<名稱>.uids` 底下，從來沒被讀過。

今天結果仍然正確，純粹因為那 332 個本來就沒有標註檔，交集已經排除了它們。
但函式的說明字串宣稱：*「人工否決的資產只在其標註 sidecar 缺席時留在外面……
排除名單才是權威，所以這裡也一併套用。」*
**那個性質已經不存在了。** 只要有人放回一個 sidecar——而說明字串自己就預期了這件事——
被 Kyzen 否決的資產就會安靜地回到語料庫。**現在是靠運氣對的，不是靠保證。**

## Q6　逐 UID 模態普查　［OBSERVED DATA，全掃非抽樣］

45,692 個已收錄資產：標註缺 0、點雲缺 0、向量缺 0、渲染目錄缺 0。每個模態都完整。

* **影像視角：46,052 個渲染目錄全部是 12 張。** 直方圖是 `{12: 46052}`，沒有例外。
* **點雲：每個資產剛好 1 份**，10,000 點。
* **文字觀測：1 句正典 `description` ＋ N 句排序過的候選。**
  候選數直方圖（全部 45,692）：`{5: 45581, 4: 50, 3: 28, 2: 13, 1: 20}`。
  **111 個 UID 不是眾數**（候選少於 5 句），其中 **20 個只有一句候選，完全沒有替代描述**。
  這一點直接限制任何需要「第二個文字觀測」的查詢包分支。

## Q7　真的是 12 張；命名；相機中繼資料　［OBSERVED DATA］

是 12 張，兩條獨立路徑各驗一次：數 46,052 個目錄裡的 PNG，以及讀 45,692 個向量檔的表頭
（每一個都是 `views (12,1280)`）。命名 `view_00.png` 到 `view_11.png`，
每個檔名各出現 46,052 次。

相機中繼資料存在、完整，且在 46,024 個 sidecar 上一致：

```
projection  perspective          camera_layout  openshape_three_rings_of_four
resolution  512                  background     transparent_rgba
camera_dist 1.2                  engine CYCLES  denoiser OPTIX
renderer_version 6               blenderproc 2.8.0   openshape_commit abe5aa42b7c9
view_directions  phi 60°:(30,120,210,300) | phi 90°:(60,150,240,330) | phi 120°:(0,90,180,270)
```

sidecar 是**按環分組**存的，不是逐視角存的，但索引到相機的對應是可以決定性還原的：
vendored 的 `views` 清單是有序的，且以 `for i in range(num_images)` 走訪，
而 `render_asset` 把排序後的 `000.png…011.png` 依序改名為 `view_00…view_11`。
所以 `view_00 = (φ60°, θ30°)` … `view_11 = (φ120°, θ270°)`。

每個 sidecar 的 `n_views_source` 都寫著
`"USER decision 2026-08-23; DEVIATION from MetaFind's stated 11"`。
12 這個數字在當初做決定時就登記為偏離了。

### ⚠ 發現二：253 個已收錄資產的渲染是壞的　［OBSERVED DATA］

46,024 個 sidecar 裡有 366 個回報異常，其中 **253 個在 45,692 語料庫內，
47 個落在封存的測試集裡**。已收錄的當中：71 個 `dark_views: 12`（每一張都暗），
**11 個實質上全空白**（`blank_views ≥ 10` 且 `distinct_views ≤ 3`）。
那 11 個的影像向量是從空白畫面算出來的，而且是**現行 gallery 的正式候選**。

## Q8　文字 schema　［OBSERVED DATA，每個欄位在 45,692 個檔案上計數］

40 個欄位普遍存在，一個不是。核心：`category`、`synset`、`lvis_category`、
`identity_confirmed`、`category_relation`、`width_axis`、`width/length/height`、
`volume`、`mass`、`description`、`materials[]`、`onCeiling/onWall/onFloor/onObject`、
`dimension_unit`（cm）、`mass_unit`（kg）。
來源：`prompt_version`、`validator_version`、`schema_version`、`annotation_contract`、
`annotator_model`、`description_source`（45,692 個全是 `model`）、`attempts`、
`description_candidates[]`（各帶 `{text, clip_score, rank}`）、
`description_ranker`（`openai/clip-vit-large-patch14` v1）、`description_sampling`、
`raw_bbox_extents`、`image_identity`、`renderer_version`（全是 6）、`mesh_proportions_yxz`。

**兩代 schema 並存**：`schema_version 5 / prompt_version 8` 共 43,597 筆，
`schema_version 6 / prompt_version 9` 共 2,095 筆。只有第六代帶 `description_fit`
（2,095 筆，是唯一非普遍的欄位）。被取代的 2,095 筆第八版原稿保存在
`annotations_superseded_v8/`。

## Q9　影像 schema（渲染 sidecar）　［OBSERVED DATA］

21 個欄位，20 個在 46,024 上普遍；`coverage_backfilled` 出現在 45,782 個上。
內容即 Q7 所列，另加 `uid`、`raw_bbox_extents`、`blank_views`、`dark_views`、
`distinct_views`，以及四個 `*_source` 來源字串。

## Q10　原始／來源點雲 schema：**不存在**　［OBSERVED DATA］

在兩個根目錄下用 `find -L` 找任何 `*raw*` / `*source*` / `*derived*` 目錄，結果為空。
唯一的來源層就是網格本身：46,052 個 `.glb`，351.4 GB。中間沒有原始點雲這一層。

**規格 §九 要求兩層（原始 ＋ 衍生正典），我們只有一層。**
從 `.glb` 重新推導是可能的，但那是整個 n03 重跑，不是複製檔案。

## Q11　衍生點雲 schema　［OBSERVED DATA］

陣列：`xyz (10000,3) float32`、`rgb (10000,3) float32`。300 件抽樣裡都有，沒有其他鍵。

**正規化是量出來的，不是假設的**：300 件的最大半徑**恰好 1.000000**（最小＝最大＝1.0），
質心偏移 ≤ 6.07e-09。`pointclouds.py:139` 稱這是「ULIP 的正規化，逐字照抄
（`dataset_3d.py:496-502`）」——質心移到原點、最大半徑縮到 1，**只動 xyz**。

46,052 個上一致：`n_points` 10000、`sampler_version` **8**、
`frame_correction` `yaw180_about_y@ulip2_frame`、`rgb_scale` `unit`、
`coloured_point_fraction` 1.0。
有變化的：`colour_source` = texture 23,675 / flat 13,524 / gltf_default 8,853；
`color0_modulated` = False 43,795 / True 2,257。種子是逐 uid 的。

## Q12　所有特徵快取，包含已經沒人讀的　［OBSERVED DATA］

| 快取 | 數量 | 大小 | 還有沒有在讀 |
|---|---|---|---|
| `embeddings/*.npz`（文字＋視角＋影像） | 45,692 | 1.55 GB | 有，Stage 1 資料集 |
| `pointclouds/*.npz` | 46,052 | 5.73 GB | 有，Stage 1 即時載入 |
| `gallery_index_00f591a09ed19a04.npz` | 1 | 217 MB | 有，已升版、有閘門紀錄 |
| `checkpoints/*.pt` | 4 | 1.29 GB | 有 |
| `stage2_gallery_00f591a09ed19a04.npz` | 1 | 21 MB | ProcTHOR 側 |
| `procthor_node_embeddings.npz`、`sem_edge_embeddings.npz` | 2 | 21 MB | ProcTHOR 側 |
| `procthor_modalities/*.npz` | 1,439 | 0.13 GB | ProcTHOR 側 |
| `_five_assets/`、`_one_asset/` | 6 | 極小 | 開發殘留，沒人讀 |
| **查詢包** `_probe/query_pack` | **0** | — | **目錄是空的** |
| **`ulip_npy`** | **0** | — | **2026-09-02 已刪** |

**兩個「不存在」很重要。**
`_probe/`、`eval/`、`ladder/` 全部是空的——9 月 2 日搬去 `/mnt/data1` 了。
所以 `stage1.py` 裡**每一條查詢包程式路徑目前都是死碼**。
`ulip_npy`（55 GB、199,974 個檔）已刪，紀錄說可從 160 份官方 shard 重生；
稽核者確認那 160 份都在 `/mnt/data1/kyzen/ulip2_objaverse_lvis/`（173 GB）。

## Q13　逐快取的身分，從產物本身讀出來　［OBSERVED DATA］

`embeddings/`——**45,692 個 sidecar 零變異**：

```
encoder_version 2 | embedding_dim 1280 | dtype float16 | aggregation "mean" | n_views 12
ulip2_ckpt_sha a4b5ed9799d5841a1646e0fb7d24cb8dcdd3b3e6fab2e2575de2531f71274adb
text_serialization metafind_v2_cm@8e4b1fcc66c7f48c | clip_train_scope "frozen" | renderer_version 6
陣列（全表頭掃描）：text (1280,) f2 | views (12,1280) f2 | image (1280,) f2，各 45,692 個
```

稽核者**重新雜湊了那顆 checkpoint 檔案，實際 SHA256 與 sidecar 宣稱完全相符**。
編碼器是 OpenCLIP ViT-bigG-14，經
`ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt`（402,018,241 bytes）。

`gallery_index_00f591a09ed19a04.npz`——`ids (45692,) <U32`、`embeddings (45692,1280) float32`。
**沒有 L2 正規化**：範數 53.90 到 112.06，平均 89.20。

`checkpoints/stage1_best.pt`——第 24 輪，`train_scope point_encoder_and_fuser`，
只存可訓練參數，80,738,946 個（骨幹 33,483,394 ＋ 塔 47,255,552），
`clip_train_scope frozen`，**`code_dirty: true`**，以 dev_val 平均 R@1 = 0.9321 選出。

## Q14　訓練／測試切分　［OBSERVED IMPLEMENTATION ＋ OBSERVED DATA；80/20 本身是 PAPER FACT］

`outputs/splits.json` 的 `object`：**訓練 36,554 / 測試 9,138**，
訓練內部再切 **dev_train 31,985 / dev_val 4,569**。比例 0.80001。

演算法（`splits.py:154`）：`sorted(uids)` → `random.Random(seed).shuffle` →
`cut = int(round(n * 0.8))` → 兩半各自重新排序。
種子：物件 **20260816**，dev **20260827**，`dev_val_fraction` 是訓練池的 0.125
（等於語料的 10%，Kyzen 2026-08-27 核可）。

**決定性是驗證過的，不是假設的**：稽核者用記錄的種子在記錄的宇宙上重跑演算法，
train / test / dev_train / dev_val **四份名單完全一致**。

**切分發生在模態過濾之後。** `splits.py:358` 先呼叫 `admitted_uids()`，
`:363` 再對它的結果做切分。切分的宇宙是已過濾的 45,692，從來不是 46,052。
規格說這一點不能不確定；現在確定了。

## Q15　訓練／測試重疊：沒有　［OBSERVED DATA］

`train ∩ test = 0`、`dev_val ∩ test = 0`、`dev_train ∩ dev_val = 0`，
且 `dev_train ∪ dev_val` 恰好等於 `train`，`train ∪ test = 45,692` 等於收錄總數。
`splits.py:369-373` 在寫檔時也斷言了這三件事。

## Q18　影像聚合　［OBSERVED IMPLEMENTATION，並用 OBSERVED DATA 反推驗證］

```python
# metafind/data/encode_text_image.py:399-407
def aggregate(views, rule):
    # docstring: `mean` on the raw embeddings rather than on L2-normalised ones
    if rule == "mean":
        return views.mean(axis=0)
```

對照規格 §五 要求的欄位：

```yaml
selected_view_ids:        [0..11]   # 全部 12 張，沒有任何選擇政策
pre_normalize_each_view:  false
method:                   mean
post_normalize:           false
```

也就是 `mean(raw)`——**既不是** `Normalize(mean(z))`**也不是** `Normalize(mean(Normalize(z)))`。

稽核者沒有只信程式碼，而是從磁碟上存的 `views` 矩陣重算四種候選，
與存的 `image` 向量比對：

| 候選 | 最大絕對差 | 餘弦 |
|---|---|---|
| `mean(raw)` | 0.00163 | **1.00000000** |
| `Normalize(mean(raw))` | 5.15986 | 0.99999988 |
| `mean(Normalize)` | 5.17857 | 0.99988717 |
| `Normalize(mean(Normalize))` | 5.16161 | 0.99988717 |

磁碟上的位元組就是未正規化的平均（0.0016 是 fp16 的捨入）。

Stage 1 另外支援 `random_single_view`（每一步重抽一張）。程式內註解誠實地寫著：
這條路徑**無法只從 `stage1_hyperparameters.json` 重現**，因為抽哪一張由 `num_workers` 決定，
而它是 DataLoader 的字面值，不在記錄的超參數裡。

## Q19　Gallery 候選建構：兩種都支援，而且是從產物讀的　［OBSERVED IMPLEMENTATION］

`run_retrieval.py` 讀 `eval_protocols.json`，並且**按欄位而非按協定名稱**解析切分。磁碟上：

| 協定 | 查詢 | Gallery | 候選數 | reported |
|---|---|---|---|---|
| `A_test_gallery` | test | test | 9,138 | true |
| `B_full_gallery` | test | full | 45,692 | true |
| `C_dev_selection` | dev_val | dev_val | 4,569 | false |
| `D_dev_val_vs_train` | dev_val | train | 36,554 | false |

`full` 是在執行時以 `train + test` 組出來的，所以是 **45,692 的已收錄語料，不是 46,052 的清單**。
封存守衛 `check_seal` 會對任何碰到 `test` 或 `full` 又沒帶 `--unseal` 的協定丟出 `SystemExit`。
**規格 §七 要求的兩種範圍已經都支援了。**

模組自己聲明的但書：**這條路徑至今沒有產生過任何回報用的數字。**
唯一存在的檢索分數——平均 R@1 0.9321——是協定 C，寫死在 `stage1.evaluate_dev_val` 裡，
而它那 4,569 個 gallery 就是挑選這顆 checkpoint 所用的同一批。
`eval_protocols.json` 裡 `reported: true` 的兩個協定從未被任何一次執行消費過。

---

## 附錄 A-1 的裁決：**成立，而且比 MASTER 說的更嚴重**

常數在 HEAD 上如 MASTER 所述。磁碟如 MASTER 所述。但**MASTER 指的那個斷言不是先壞的那一個，
而且在今天的磁碟上它根本不可能觸發**：`stage1.py:463` 位於 `_query_side` 內，
而該函式在 `query_pack is None` 時立刻回傳 `{}`——`_probe/query_pack` 是**空目錄**，
沒有任何查詢包存在，所以那個 raise 是**無法到達的程式碼**。

真正的硬停是 `check_embedding_sidecars`（`stage1.py:1466`，由 `main()` 在 `:2272` 呼叫）。
稽核者唯讀執行了它：

```
SystemExit: cached embeddings disagree with the encoding protocol:
{'n_views': 200} sidecar field mismatches, 0 sidecars missing, over 200 assets.
Expected {..., 'n_views': '11'}
```

200 之 200。全部 45,692 個 sidecar 都寫 `n_views: 12`，而檢查要求 `"11"`。
**Stage 1 的 `main()` 在第一個批次之前就退出。** 另外：

```
renders.is_complete -> complete=0   STALE=46024
```

46,024 個全數過期，`RENDERER_VERSION 7 != 6` 在視角數檢查之前就短路了。

**結論：Stage 1 無法在現有語料上執行。不是降級，是被自己的守衛正確地拒絕。**

來源：commit `7785679`（2026-09-02），它自己的訊息就寫著「每個資產都過期、將被重渲……
目前尚未渲染任何東西」。常數是為了一次重渲而預先改成 11 的，
而晚它一天的規格文件現在禁止那次重渲。兩個決定各自都沒錯，隔一天，直接衝突。

**一個與修法有關的上游事實**［UPSTREAM FACT，`vendor/openshape/render_single_glb.py:172-183`］：
上游的相機清單是寫死的十二筆、三圈各四。傳 `--num_images 11` 會取**前十一筆**，
也就是兩整圈加第三圈的四分之三，是不對稱的一組。
**所以「直接丟掉 view_11」並不等於一個像樣的 11 視角協定。**

## 附錄 A-2 的裁決：**在 HEAD 上不成立，但有一個相關的真風險**

`metafind/data/render_blender.py` 裡沒有任何 `unlink`：`grep -c unlink` 為 0，
`git diff --stat HEAD` 為空，而 `git log -S"unlink"` 對該檔案**整段歷史為空**——
那個字串從未進過版本庫。這與 MASTER 的記錄一致：那段修改是未提交的，
並且在稽核開始前就已經被 MASTER 還原。

**真正的風險是相反的一個，而且沒有人刪任何東西。**
`render_asset` 以 `view_{i:02d}.png` 寫出 `i in range(len(produced))`，而 `produced` 現在是 11。
重渲一個既有資產會用第七版的圖蓋掉 `view_00`…`view_10`，
而**把舊的第六版 `view_11.png` 原地留下，成為孤兒**。
新的 sidecar 只列 11 條路徑，於是同一個目錄裡會並存兩種相機協定的檔案，而沒有任何東西記錄誰是誰。
`renders.py:759` 的註解還寫著「所有十二張都存在」。
沒有資料被摧毀（§十九 未被違反），但版本會安靜地混在一起。

---

## 稽核者的建議（未執行，本輪唯讀）

1. **不要重渲。** 改成把視角數變成**讀出來的性質**而不是編譯期常數：
   讓 `check_embedding_sidecars` 與 `is_complete` 對照 sidecar 記錄的值，
   並讓 Stage 1 在載入時從磁碟上的 12 張裡挑它要的 11（或 12）張。
   這是能讓 Stage 1 在現有語料上跑起來、又保住每一張圖的最小改動，
   也正是規格 §三 所說的「12 張原始視角 → 執行時再決定用哪些」。
2. **修 `admitted_uids()` 的名單解析**，改讀 `groups.*.uids`。現在扣掉的是零個真實 uid。
3. **決定那 253 個壞渲染的已收錄資產怎麼辦**（11 個全空白，47 個異常落在封存測試集）。
   這是研究決定，不是稽核者能定的。
4. **處理孤兒第十二張**：重渲時寫進帶版本的子目錄（可逆），而不是清空資產目錄（不可逆，且 §十九 禁止）。
5. **沒有原始／來源點雲層**（§九 要兩層）。要重建就是對 351 GB 的 GLB 重跑 n03。只是標記，不是提議執行。

## 沒有驗證的

* 點雲陣列的 schema 是 300 件隨機抽樣（sidecar 是全部 46,052 的全掃）。
* **沒有打開任何一張 PNG。** 空白／暗／相異的計數是 sidecar 自己記錄的量測，不是重新量像素。
* 沒有解壓 ULIP-2 官方 shard 去重新驗證每筆的視角數。
* 「可用」是本專案的定義；MetaFind 自己的可用概念是否相同，無法從論文回答，維持 UNRESOLVED。

---

# B　程式碼（第 16、17、20–28 題）

全部在 CPU 上實測，沒有動 GPU、沒有訓練、沒有寫入任何檔案。
每個數字都是跑出來的，不是讀旗標讀來的。

## Q16　`same_record` 在現行程式裡是什麼意思

**它只有一個意思：同一份觀測。不是同一個 UID。**［OBSERVED IMPLEMENTATION］

`same_record` 只是 `--query-observation` 的一個選項（`stage1.py:1955-1959`），
全 repo 只出現六次（五次在 `stage1.py`、一次在測試），**不在任何協定產物裡**。

兩道守衛（`stage1.py:2094-2102`）讓它與「有沒有 query pack」完全等價：
沒有 pack 卻宣告 `second_observation` 會被拒；有 pack 卻宣告 `same_record` 也會被拒。
沒有 pack 時，`split_embeds`（`:623-654`）在 `:646` 直接 `return gallery, gallery`
——**同一個字典物件**，不是內容相等的兩個。實測 `query_embeds is gallery_embeds → True`。

**「同一個 UID」這半邊不在這個字裡，也不在任何字裡。** 它是結構性的：
每個 uid 一個資料項、`collate` 依序堆疊、損失用 `labels = torch.arange(B)`
（`losses.py:179`）。**正確，而且由建構保證，但沒有任何東西斷言它。**
這正是規格 §四／§十一 要求拆開的那兩件事——其中一半已經是分開的，
但是靠意外而不是靠設計：**沒有一個 `positive_policy` 欄位可以讀。**

## Q17　Query 與 Gallery 的觀測建構

在磁碟上已解析的協定下（`image_aggregation: "mean"`，且**live tree 上沒有任何 query pack**）：

| 模態 | Gallery 塔收到 | Query 塔收到（`same_record`） |
|---|---|---|
| 文字 | `embeddings/{uid}.npz["text"]`，fp16→fp32，1280 維。每個 uid 一句正典描述，離線用凍結的 ViT-bigG-14 編碼 | **同一個張量** |
| 影像 | `embeddings/{uid}.npz["image"]` ＝ **12 張逐視角特徵的未正規化平均** | **同一個張量** |
| 點雲 | `pointclouds/{uid}.npz` → `(10000, 6)` xyz+rgb，寫檔時就 `pc_norm` 過 → **每一步即時**過 `backbone.encode_pc` | **同一個張量，共用一次 `encode_pc` 呼叫** |

實測形狀：`text (1280,)`、`image (1280,)`、`pc (10000, 6)`；批次 4 → `pc (4,10000,6)`。

**規格 §三 D 的一個重要事實：Stage 1 從頭到尾沒有呼叫過 `encode_text` 或 `encode_image`。**
兩座塔的文字與影像輸入都是離線快取讀取。實測 `requires_grad = False`，沒有 autograd 圖。

`second_observation` 這條路**今天跑不起來**：`data/outputs/_probe/query_pack/` 是空的，
包已在 9/2 搬到 `/mnt/data1`。

## Q20　Stage 1 的最佳化器參數分組

［OBSERVED IMPLEMENTATION，在 CPU 上用磁碟上已解析的協定實際建起來數的］

```
交給 weight_decay_groups 的具名候選：273
 群 0：102 個張量  80,610,176 參數  lr 5e-4  weight_decay 0.1  betas (0.9,0.98)
 群 1：171 個張量     127,488 參數  lr 5e-4  weight_decay 0.0

 群 0  point_encoder 81 / 32,425,856 · pc_projection 1 / 983,040
       query.fusion 10 / 23,600,640 · gallery.fusion 10 / 23,600,640
 群 1  point_encoder 139 / 73,216 · query.fusion 16 / 27,136 · gallery.fusion 16 / 27,136

 273 個張量，依 id 去重也是 273（沒有重複）
 最佳化器裡有凍結參數嗎：沒有
 可訓練但漏掉沒進最佳化器的：0 個
```

* **全部只有一個學習率。** 沒有逐模組的 lr。`stage1.py:2364-2366` 每一步把 `lr_now`
  寫進**每一個**參數群，所以兩群的 lr 永遠不可能分岔。
* 分群規則是上游逐字照抄的名稱判準。稽核者數過：**零個排錯**。
* `loss_trainable_state` 是空的：`learnable_temperature: false`，
  所以 `logit_scale` 是 buffer 不是 Parameter，τ 固定 0.5。
* **後果**：`mask_tokens` 與 `modality_pos` 是 `(3, 1280)`，`ndim ≥ 2`，
  所以它們落在**權重衰減 0.1 那一群**。見發現 M-3。

## Q21–Q24　誰在訓練、誰凍結　［全部實測］

| | 張量數 | 可訓練 | 參數量 | 模式 |
|---|---|---|---|---|
| PointBERT（`point_encoder`） | 220 | **220（100%）** | 32,499,072 | `train`，DropPath 0.1 生效 |
| `pc_projection` | 1 | 1 | 983,040 | — |
| OpenCLIP 文字 transformer | 384 | **0** | 629,678,080 | `eval` |
| OpenCLIP `token_embedding` | 1 | **0** | 63,242,240 | `eval` |
| OpenCLIP 視覺塔 | 584 | **0** | 1,844,907,264 | `eval` |

文字與影像編碼器有**兩道**獨立屏障：`requires_grad = False`，
而且 `encode_image` / `encode_text` 另外包在 `torch.no_grad()` 裡
（除非 `train_scope="full"`，而 `arm_config_hash` 會拒絕那個值）。
再加上第三道：**Stage 1 根本沒叫過它們。**

## Q22　PointBERT 到底有沒有收到梯度：**有，而且驗了兩次**

稽核者先照指示查既有證據：

| 來源 | 結果 |
|---|---|
| `tools/probes/` | **沒有定論。** 只有 Stage 2 的煙霧測試提到梯度，沒測 Stage 1 的 PointBERT |
| `tests/` | `test_grad_checkpointing.py` 確實反向穿過真的 PointBERT 區塊，但用的是合成的 `out.sum()` 損失，不是 Stage 1 的損失路徑；`test_backbone_scope.py` 用假的替身。**兩個都不算** |
| `output/look/` | 空的，只有 README |
| **封存的量測** | **有定論。** `/mnt/data1/…/look/diag_modality_norms.json`［OBSERVED DATA］ |

```
未載入 checkpoint 的編碼器   text 37.130943640239366  image 40.22924563018678  pc  27.860065006689254
訓練過的（ladder/e25_500w）  text 37.130943640239366  image 40.22924563018678  pc 200.5473794034238
```

文字與影像**到小數第十五位完全相同**，點雲的範數是 7.2 倍。點雲那條路上有東西動了。

稽核者接著自己讀 checkpoint 排除「只有投影層動」的可能：

```
point_encoder 的浮點張量：224 / 224 全部與 ULIP-2 原版不同
最大絕對差 0.4444，最大相對差 0.6845
pc_projection 也動了（最大絕對差 0.0232）
另存了 6 個 BatchNorm 的 buffer
```

最後在**現行程式**上實跑一次前向加反向（CPU、批次 2、真實呼叫序列、不做最佳化步、不寫檔）：

```
loss 0.439766
 point_encoder（PointBERT）  220 張量  220 有梯度  220 非零  梯度範數 0.640508
 pc_projection                 1         1          1        梯度範數 0.232075
 OpenCLIP 文字 transformer   384         0          0        梯度範數 0
 OpenCLIP 視覺塔             584         0          0        梯度範數 0
 query.fusion                 26        26         26        梯度範數 0.771813
 gallery.fusion               26        26         25        梯度範數 0.619367
 query.fusion.mask_tokens      1         1          1        梯度範數 0.122201
 gallery.fusion.mask_tokens    1         1          0        梯度範數 0        ← 見 m-4
```

**規格 §十五 與 PHASE 5 要求的「證明 PointBERT 真的收到梯度」，今天結清，而且沒用到 GPU。**

## Q25　Fusion_Q 與 Fusion_G 是不是不同的參數物件

**是不同的。舊的 `Q-BUILDMODEL` 缺陷已經關閉——但不是在你會去看的那個地方。**

```
id(model.query.fusion)   = 136177021763536
id(model.gallery.fusion) = 136177021762960
query.fusion IS gallery.fusion : False
model.fusion_is_tied()         : False
有共用的參數張量嗎             : 沒有（26 對 26，id 集合交集為空）
query.fusion.cfg IS gallery.fusion.cfg : True   ← 同一個「設定」物件
```

`stage1.build_model`（`:1858-1861`）**仍然把同一個 `FusionConfig` 物件傳給兩座塔**，
那正是舊缺陷的字面形狀。它今天無害，是因為兩座塔各自呼叫 `ModalityFusion(cfg)`，
從一份設定建出兩套獨立參數。

訓練好的 checkpoint 也獨立證實了：`tower_trainable_state` 有 26 個 `query.*` ＋
26 個 `gallery.*`，而兩邊的 `mask_tokens` 逐列餘弦是 0.044 / −0.053 / 0.050，
統計上正交——兩次獨立初始化、各自訓練。

**但這是「現在是對的，沒有東西保證它」那一類。** `fusion_is_tied()` 存在而且是個好檢查，
可是 `build_model` 從來沒呼叫它，Stage 1 路徑上也沒有任何斷言說
「在 `shared_backbone_separate_fusion` 之下不得綁定」。

## Q26　共享的 ULIP 骨幹實際上怎麼實作

**三種不同機制，只有一種是模組共享。**

1. **點編碼器——靠物件identity共享。** `stage1.main` 只建**一個** `ULIPBackbone`（`:2155`）。
   `MetaFindDualTower` **裡面根本沒有骨幹**，只有 fusion 模組。
   `split_embeds` 呼叫一次 `encode_pc`，把同一個輸出張量交給兩座塔。
   **共享不是某個設定值被遵守，而是第二個物件不存在。**
2. **文字／影像——靠快取identity共享，不是模組identity。** Stage 1 裡兩座塔都沒跑 CLIP，
   讀的是同一個 `.npz`。「同一個空間」這個性質成立是因為**同一個檔案**，不是同一個模組。
3. **`tower_sharing` 只管 fusion。** `shared_backbone_separate_fusion` 與 `fully_separate`
   產生**完全相同**的 fusion 結構；這個欄位對骨幹只是名義上的。
   `stage1.py:1989-1996` 直接**拒絕** `fully_separate`——那個拒絕是這個欄位沒有變成謊言的唯一原因。

磁碟上 `stage1_protocol.json` 的 `tower_sharing` 是 `shared_backbone_separate_fusion`，
與規格 問題 1 的 `AUTHOR EVIDENCE / MAINLINE` 相符。

**checkpoint 怎麼存**（從 `stage1_best.pt` 讀出）：

```
backbone_trainable_state  227 張量 = 226 個 point_encoder.* ＋ pc_projection（33,483,394 參數）＋ 6 個 BN buffer
tower_trainable_state      52 張量 = 26 個 query.fusion.* ＋ 26 個 gallery.fusion.*（47,255,552 參數）
loss_trainable_state        0 張量
open_clip                  不存——從釘住的 OpenCLIP 權重重建
```

共享骨幹**只出現一次**，沒有塔的前綴。checkpoint 的 sha256 與紀錄完全相符。

## Q27　Stage 1 的損失是不是只有 q→g：**是**

```
bidirectional=False -> 鍵 ['acc_q2g', 'loss', 'loss_q2g', 'temperature']
  out["loss"] IS out["loss_q2g"]  : True   ← 同一個 Python 張量物件
  沒有 loss_g2q，沒有 acc_g2q
bidirectional=True  -> loss == 0.5*(loss_q2g + loss_g2q) : True
  同樣的輸入：1.761543（Stage 1）對 1.763730（雙向）
logit_scale 是 buffer 不是 Parameter : True     τ = 0.5
```

`build_model` 把 `bidirectional=False` 寫死（`:1870`），沒有任何協定產物能覆蓋。
`losses.py:188-191` 在建構 g→q 分支之前就回傳了，所以連轉置的 logits 都不存在。
與規格 §三 G 的 PAPER FACT 及本專案附錄 B 對 Eq. 5 的核對一致。

## Q28　遮罩實作

**獨立、p = 0.3——實測 200 萬次抽樣，種子 20260816：**

```
P(遮住 text)  = 0.30030      P(遮住 image) = 0.29972      P(遮住 pc) = 0.30045
P(三個全遮)   = 0.02699      理論 0.30³ = 0.02700
P(一個都不遮) = 0.34270      理論 0.70³ = 0.34300
P(text 且 image) = 0.09004   P(text 且 pc) = 0.09005   P(image 且 pc) = 0.09007   獨立 → 0.09000
allow_empty=False -> P(全遮) = 0.00000
```

**兩兩獨立與三方獨立都成立，2.7% 與 0.3³ 到小數第四位相符。**
`allow_all_masked: true`，所以全遮是允許的，沒有條件重抽，與規格 §三 F 主線相符。

**遮住的模態怎麼表示：**

```
mask_tokens 是 nn.Parameter，形狀 (3, D)，requires_grad True，非零
被遮的欄位 == mask_tokens[i]     True
整列全遮 == mask_tokens          True（輸出有限，沒有 NaN）
沒被遮的欄位 == 真實向量          True
zero_pad=True -> 被遮欄位是零     True（Table 3 的消融，正確地分開）
```

**只遮 query。** `stage1.py:2354-2355` 只把 `present` 傳給 query 塔；
`GalleryTower.forward` 在任一模態缺席時**直接拋錯**。
**模態完整性是被呼叫端強制的，不是呼叫端供應的。這是對的形狀。**

**但是見 M-3。**

## 附錄 A-1 的裁決（本節與 A 段的裁決合併後的最終版）

**兩位稽核者各對一半，而且不衝突——他們測的是不同的入口。**

* **Integrator 從 `Stage1Dataset` 直接建，繞過了 `main()`。** 在
  `image_aggregation: "mean"` 且沒有 query pack 的設定下，
  資料項的鍵是 `['image', 'pc', 'text', 'uid']`——**根本沒有 `views`**。
  所以那個 11 視角的斷言（`:463`）永遠不會被觸發，而且它位於 `_query_side` 內，
  在沒有 pack 時該函式立刻回傳空字典。**模型與資料集在 12 視角語料上完全正常，
  今天實測跑完了一次前向加反向。**
* **ULIP2 稽核者走的是 `main()`。** `main()` 在 `:2272` 呼叫
  `check_embedding_sidecars`，而該函式（`:1466-1481`）把
  **`str(N_VIEWS_PER_ASSET)` 也就是 `"11"`** 放進要比對的字典，
  拿去比 sidecar 記的 `"12"`。抽 200 個，**200 個全部不合**，`SystemExit`。

MASTER 自己開檔確認了這一點（`stage1.py:1480`）：

```python
want = {"text_serialization": str(encoding["text_serialization"]),
        "aggregation":       str(encoding["image_aggregation"]),
        "encoder_version":   str(ENCODER_VERSION),
        "n_views":           str(N_VIEWS_PER_ASSET)}   # ← 編譯期常數，不是協定值
```

**最終表述，取代 `REPRODUCTION_PROTOCOL_20260903.md` 附錄 A-1 的措辭：**

> Stage 1 的**資料集與模型**在現有 12 視角語料上完全可用。
> 擋住的是 `main()` 的一道前置檢查，它拿**編譯期常數**去比**資料自己記錄的值**。
> 另外 `renders.is_complete` 判定 46,024 筆全部過期，那個判定**本身是真的**
> （語料確實是在另一個協定下渲染的），只是我們並不打算依它行動。
> 所以這不是「讀不了語料」，而是「一道守衛拿錯了比較對象」。

一個與修法有關的上游事實［UPSTREAM FACT］：上游的相機清單是寫死的十二筆、三圈各四，
傳 `--num_images 11` 會取**前十一筆**（兩整圈加第三圈的四分之三），是不對稱的一組。
**所以「直接丟掉第十二張」不等於一個像樣的 11 視角協定。**

## 附錄 A-3 的裁決：**成立，兩個值都是活的**

稽核者追到消費端而不是相信檔案：

```
stage2_protocol.json["init_lambda"] = 9.0
  -> stage2.py:797-798（缺鍵就拋錯）-> build_model(..., init_lambda=...) :805
  -> DualTowerConfig.init_lambda -> QueryTower.layout_weight = nn.Parameter(9.0)
  -> Eq. 6 的 fused + λ·layout                                          【活的】

essgnn_arch_protocol.json["pooling"] = "normalised_sum"
  -> ESSGNNConfig.from_protocol（essgnn.py:303）-> 讀出分支 essgnn.py:657-672   【活的】
```

衝突照列，**稽核者不裁，MASTER 也不裁**：

| 來源 | 說法 |
|---|---|
| `REPRODUCTION_PROTOCOL_20260903.md` 問題 8 | λ 初值 **UNRESOLVED**、pooling **UNRESOLVED**；§十九 禁止自行決定 |
| 帳本 DL-077 item 8 | Kyzen 裁 甲：pooling = 正規化總和，**λ 初值 0.1** |
| 帳本 DL-078 | 9.0 ＝ 0.1 × 量到的融合查詢範數 91.4，**「Kyzen 可否決」** |
| `resolve_stage2.py` ＋ 兩個 JSON | `init_lambda: 9.0`、`pooling: "normalised_sum"` |
| 兩個 JSON 的 `decided_by` | 「Kyzen（2026-09-02，item 8：pooling normalised_sum；**init_lambda 9.0** ＝ 0.1 × 量到的 91.4）」 |

**`decided_by` 記載了一件沒有發生過的裁決。** Kyzen 裁的是 0.1。
pooling **確實**是他裁的，那一半歸屬正確。

## 程式碼段的發現

**M-1（重大，接縫）　有一整層讀協定建模型的程式碼，訓練時完全沒有經過。**
`stage1_config.py:275-413` 的 `Stage1RuntimeConfig.from_protocols`
**只有兩個測試檔在用**。真正的訓練器在 `stage1.build_model` 裡自己建設定。
那一層裡以下全部是訓練路徑上的死碼：`gallery_backbone`（給 `fully_separate` 用的第二個骨幹設定）、
`sample_present_mask`、`may_use_cached_text_image`、`cache_layout`、`d1_is_active`、`exceeds_paper`，
以及它自己那條 `train_scope = "full" if actual_scope == "trainable"` 規則（`:367`）——
那是對同一個問題的**另一種**解析，與 live 路徑在 `stage1.py:2154` 用的不同。
**這是本專案典型失敗模式的字面重現：一層協定解析寫好了、測試全綠、而 live 路徑不讀它。**
今天沒出事，只因為 `stage1.py:1989` 用手寫的方式拒絕了 `fully_separate`。

**M-2（重大，接縫）　query pack 記下來的取圖規則與實際執行的規則已經不一致。**
`tools/make_query_pack.py` 把 `"rule": "views[uid_seed(uid) % 12]"` 寫進清單；
`QueryPack.view_index`（`stage1.py:352-362`）實際執行的是 `% N_VIEWS_PER_ASSET` 也就是 **% 11**。
那個字串會進 `QueryPack.identity()` 再進 `arm_config_hash`，
**於是實驗臂的雜湊會描述一個這次執行從未做過的抽樣。**
`make_query_pack.held_out_view` 是死碼（定義了、從未呼叫）。

**［補充，稽核者後續實測，證據等級從 OBSERVED IMPLEMENTATION 升為 OBSERVED DATA］**
這不是「重建才會出事」的假設——**封存起來的兩份清單白紙黑字已經記錯了**：

```
/mnt/data1/kyzen/archive_20260902_pre11view/probe/query_pack/query_pack.json
/mnt/data1/kyzen/archive_20260902_pre11view/probe/query_pack/query_pack_textimage.json
  image   {"rule": "views[uid_seed(uid) % 12]", ...}
  gallery "UNCHANGED: canonical text, 12-view mean image, canonical pc"
```

把任一份 pack 從封存拿回來、用今天的程式跑，產生的實驗臂雜湊會描述一次沒有發生過的抽樣。
**同一次檢查還關掉了一個原本列為「沒有驗證」的項目，並確認它是一個真的缺口**：
兩份清單裡**沒有任何一個分片紀錄帶 `ulip2_ckpt_sha`**（文字與點雲分片全是 `False`）。
陣列位元組有被雜湊進實驗臂，所以 query 側與 gallery 側的文字編碼器若分岔，
系統會**偵測**到但永遠不會**指名**是編碼器換了。
順帶通過一項一致性檢查：`query_pack_textimage.json` 的點雲分片是 0，
與 `stage1_best_ckpt.json` 記的 `arms: ["text", "image"]` 相符。

**M-3（重大，ULIP2）　可學的遮罩向量被放進權重衰減 0.1 那一組，訓練完等於沒學。**
`mask_tokens` 是 `(3, 1280)`，`ndim ≥ 2`，所以與注意力權重矩陣同組。訓練好的 checkpoint 實測：

```
初始化的期望值：std 0.02、維度 1280 -> 每列範數 0.7155
query.fusion.mask_tokens   逐列範數 [0.7209, 0.7358, 0.7074]  std 0.02017
gallery.fusion.mask_tokens 逐列範數 [0.6977, 0.7381, 0.6823]  std 0.01975
```

**跑完一整輪訓練之後，遮罩向量與它剛初始化時無法區分。**
再對照封存的量測：遮罩向量的範數除以真實模態向量的範數＝
**0.0194（文字）／0.0183（影像）／0.0123（點雲）**。

論文說「用遮罩嵌入**而不是**補零」；名義上實作了，數值上那個替身只有真實向量的約 2%。
**最佳化器正在把論文明講的機制，拉向論文自己說比較差的那個消融。而且沒有任何東西記錄這件事。**
這是 Kyzen 的裁量，不是工程師的。

**m-4（次要）** `gallery.fusion.mask_tokens`（3,840 個參數）在最佳化器裡，
但**梯度恆為零**（實測），因為 gallery 從來不被遮。
checkpoint 裡的死重量，而且如果哪天有人把 gallery 也走遮罩路徑，它會安靜地變成活的。

**m-5（次要，接縫）** `stage1_protocol.json` **沒有 `train_scope` 這個鍵**。
live 的值來自 `stage1.py:2154` 的 `.get(..., "point_encoder_and_fuser")` 預設，
然後被寫進 `arm_config_hash` 與 checkpoint 紀錄，**彷彿它是解析出來的**。
行為正確；來源宣稱有一個沒有任何產物記錄的決定。

**m-6（次要）** `ulip_backbone.py:234-236` 為了取回 `self.preprocess`，
第二次呼叫 `open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=None)`，
**每次建骨幹都配置一個完整的、隨機初始化的 25 億參數模型**
（日誌會出現「No pretrained weights loaded … Model initialized randomly」）。
Stage 1 路徑上用不到，但那是真的記憶體與每一份日誌裡的一句誤導。

**i-7（提示）** `renders.py:99-100` 的註解寫 `render_blender.N_VIEWS`「(12)」，實際是 11。

**i-8（提示，好消息，值得釘住）** 找到四個真正由**被呼叫端**保證的性質，
而它們本來都很容易寫成「由呼叫端供應的前提」：
`GalleryTower.forward` 拒絕缺席的模態；每次存檔都跑
`assert_checkpoint_covers_optimizer`；`load_stage1_checkpoint` 逐模組檢查涵蓋率；
`QueryPack.require` 是拒絕而不是回退。**其餘的程式應該照這個形狀寫。**

**i-9（提示）** 程式裡沒有 `positive_policy` 這個字。同 UID 配對是結構性的、正確的、
而且沒有任何東西斷言它。**如果 §四 的 `positive_policy: same_uid` 變成一個設定欄位，
它必須要有消費端，否則第一天就會變成 M-1 那種死碼。**

## 程式碼段沒有驗證的

* **5e-4 / 5 輪的主線配方從來沒有在這份程式上完整跑完過。** 所有檢查過的 checkpoint
  都是 12 視角語料上 lr 2.5e-4 的實驗臂。梯度證明是一次前向反向，不是一次訓練。
* `second_observation` 沒有被執行過（沒有 pack）。M-2 是讀原始碼得到的，不是跑出來的。
* query pack 文字臂的編碼器身分：分片紀錄沒有存 `ulip2_ckpt_sha`，
  所以差異會被**偵測**到但不會被**指名**。
* `ULIP2_PointBERT_Colored` 裡的 ViT-bigG-14 是否載入真實預訓練權重（對 Stage 1 無關，對重建 n06 有關）。
* `fully_shared` 沒有被建起來過。

---

# D　Migration（第 33–40 題）　MASTER

以下是 MASTER 綜合三段稽核後的答案。**沒有任何一項已經執行。**
規格 §二十一 說要等 Kyzen 看完 audit 才批准，所以這一段是提案。

## Q33 / Q34　建議修改哪些檔案，各改什麼

依「不可逆程度」由低到高排列。**第 1 到 4 項不動任何資料，只讓程式與磁碟上的事實對齊。**

| # | 檔案 | 改什麼 | 為什麼 | 分類 |
|---|---|---|---|---|
| 1 | `metafind/train/stage1.py:1480` | `check_embedding_sidecars` 的 `n_views` 改成比對**編碼協定記錄的值**，不是 `N_VIEWS_PER_ASSET` 這個編譯期常數 | 這是唯一擋住 Stage 1 的東西。守衛的用途是「快取與協定不一致就拒絕」，而視角數是協定值不是程式常數 | bug fix |
| 2 | `data/outputs/stage1_encoding_protocol.json` | 加入規格 §五 要求的完整 `view_aggregation` 區塊：`selected_view_ids [0..11]`、`pre_normalize_each_view false`、`method mean`、`post_normalize false`、`n_views 12`、`aggregation_version` | 這四個欄位描述的是**現在磁碟上已經發生的事**，只是從來沒被寫下來。規格明說 `method: mean` 一個欄位不夠 | 記錄既有行為 |
| 3 | `metafind/train/stage1.py:395` 及三個消費端 | `N_VIEWS_PER_ASSET` 從模組常數改成從協定／sidecar 讀 | 讓「用哪些視角」成為執行時決定，正是規格 §三 要的 | implementation choice |
| 4 | `metafind/data/splits.py:322-329` | 排除名單改讀 `groups.*.uids` | 現在扣掉的是九個非 uid 字串，說明字串宣稱的安全性質不存在 | bug fix |
| 5 | `data/outputs/stage2_protocol.json`、`essgnn_arch_protocol.json` | `decided_by` 改為：pooling 歸 Kyzen（DL-077），`init_lambda 9.0` 歸「由量測推導、待 Kyzen 裁示」 | 現在記載了一件沒發生的裁決 | 來源更正 |
| 6 | `metafind/models/stage1_config.py` 或 `stage1.py` | 二選一：讓 `build_model` 真的走 `Stage1RuntimeConfig`，或把該層明確標記為非訓練路徑 | M-1：一整層綠的測試在測一條沒人走的路 | 接縫修復 |
| 7 | `tools/make_query_pack.py:92,108,412` | 取圖規則字串與實際執行對齊；刪掉或接上 `held_out_view` 死碼 | M-2：重建 pack 的那一刻雜湊就會說謊 | bug fix（PHASE 4 再做） |
| 8 | `metafind/data/renders.py:99`、`semantic_edges.py:11,97`、`render_blender` 文件字串 | 更正過期註解（12 vs 11、Qwen vs gemma、「n08 還沒跑」） | 讀者會據此下錯結論 | 文件 |

**沒有列進來、而且刻意不動的**：`render_blender.py` 的 11 視角相機補丁與 `RENDERER_VERSION 7`。
規格說相機協定 UNRESOLVED、現在不重渲，所以那段程式描述的是一個**尚未執行也未獲批准**的協定。
留著不動是誠實的：語料是 v6，程式知道自己會產生 v7，兩者不同是事實。
`is_complete` 回報全部過期**也是真的**，只要沒有人去跑 n04 就不會有事。

## Q35　Migration 策略

**三個波次，每一波都可獨立回退。**

```
波次一（純程式，不碰資料）        上面第 1、2、3、4、5、8 項
  → 跑 pytest 全套
  → Stage 1 --limit 128 --epochs 1 煙霧（要 Kyzen 放行 GPU）
  → 通過即代表：Stage 1 可在現有 12 視角語料上訓練

波次二（建清單，只讀既有產物）    規格 §四 的 manifest
  → assets / images / captions / pointclouds 四張表，1 UID = 1 列
  → 過濾與隔離紀錄、切分清單與 SHA256
  → 純 CPU，不重算任何特徵

波次三（Dataset API）             規格 §十一
  → positive_policy: same_uid（並且要有消費端，否則就是 M-1 重演）
  → query_observation / gallery_observation 逐模態政策
  → view selector 與 view_aggregation 設定
  → gallery_test / gallery_full（**兩者已經支援**，只需接上新的觀測政策）
```

**波次二與波次三不需要重算任何特徵**，這是這次盤點最大的發現，見 Q39。

## Q36　向後相容

* 第 1、3 項改完之後，**現有的 45,692 份向量檔全部繼續有效**。
  改的是比對的對象，不是資料。
* 第 2 項是**新增**協定欄位。舊的 arm 雜湊會改變，所以新舊 run 的雜湊不可直接比對——
  這是對的，因為協定確實變了（多記了四個欄位）。**既有 checkpoint 不受影響。**
* 第 4 項會讓 `admitted_uids()` 真的開始扣 uid。**今天結果不變**（那 332 個本來就沒標註檔），
  所以 45,692 這個數字不動，切分不動。
* 第 5 項只改字串，不改任何數值。

## Q37　對既有 checkpoint 的影響

**沒有影響。** `stage1_best.pt`（第 24 輪、`train_scope point_encoder_and_fuser`、
80,738,946 個可訓練參數、以 dev_val R@1 0.9321 選出）不會失效。

但要記住兩件**已經存在**的限制，與這次遷移無關：

* 它的紀錄帶 `code_dirty: true`，所以它**不精確對應**任何一個 commit。
* 它是在 12 視角語料、`same_record` 觀測、lr 2.5e-4 下訓練的。
  規格 §十六 說得很清楚：換觀測、換視角政策這些是**訓練期**的改動，
  拿同一顆 checkpoint 換評估設定只能叫 inference sensitivity，不能當成換過訓練協定。

## Q38　對既有特徵快取的影響

**全部保留，沒有一個需要作廢。**

| 快取 | 影響 |
|---|---|
| `embeddings/*.npz`（45,692） | **不作廢。** 它已經同時存了 `views (12,1280)`（逐視角）與 `image (1280,)`（平均） |
| `pointclouds/*.npz`（46,052） | 不作廢 |
| `gallery_index_…npz`（45,692） | 不作廢。但它**沒有 L2 正規化**（範數 53.9–112.1），評估時要確認相似度計算有處理 |
| ProcTHOR 的四個 | 不受 Objaverse 端影響 |

規格 §十 要求每個 cache 帶 `valid_for_train_scope`。現在沒有這個欄位——建議在波次二補上，
標成 `[point_encoder_and_fuser, fuser_only]`，因為文字／影像凍結的前提只在這兩個 scope 下成立。

## Q39　需要重新計算的項目：**幾乎沒有**

**這是這次盤點最重要的結論。**

規格 §七 要求「12 張各自獨立 encode，逐視角特徵可以 cache，禁止只保存平均」。

**這件事已經做完了。** 每一份 `embeddings/{uid}.npz` 都同時帶：

```
views  (12, 1280) float16   ← 12 張逐視角特徵，全部在
image  (1280,)    float16   ← 未正規化的平均，另存
text   (1280,)    float16
```

45,692 份全部如此，零例外（全表頭掃描）。
**所以 PHASE 3 的「12-view frozen image features」不需要跑，它在磁碟上已經存在。**
單視角、留出視角、不相交子集這些實驗，全部可以從現有檔案直接切，不必重新編碼。

真正需要重算的只有兩項，而且都不在 Objaverse 端：

| 項目 | 什麼時候需要 | 成本 |
|---|---|---|
| ProcTHOR 節點文字 ＋ 語意邊（**必須綁一起**） | 只有在決定要修那 146 個字串時 | 613 次 LLM 呼叫，約 4 分鐘 GPU，加約 1 分鐘重新編碼 |
| 原始／來源點雲層（規格 §九 要兩層，我們只有一層） | 只有在 Kyzen 要求補齊時 | 整個 n03 對 351 GB 的 GLB 重跑 |

**絕對不要只跑其中一半的 ProcTHOR：** 只重跑文字不重跑邊，
會讓 11.56% 的邊安靜地變成「查無資料」，程式不報錯，Table 2/3 全錯。

## Q40　逐項估算

| 項目 | CPU | GPU | 硬碟 | 時間 |
|---|---|---|---|---|
| 波次一，第 1–5、8 項（純程式） | 單核 | 無 | 0 | 改動數十分鐘，pytest 約 3 分 |
| Stage 1 煙霧（`--limit 128 --epochs 1`） | 少 | **需要**，載 9.5 GB 骨幹 | 約 330 MB checkpoint | 約 5–10 分 |
| 波次二 manifest（§四四張表） | 單核掃 46,052 個 sidecar | 無 | 約 50–100 MB JSON | 約 10–20 分 |
| 波次三 Dataset API（§十一） | — | 無 | 0 | 純實作 |
| 修 ProcTHOR 節點文字 ＋ 邊 | 少 | **需要**，載 gemma 23 GB | 覆寫約 21 MB | 約 5 分（613 對）＋ 模型載入 |
| 補原始點雲層（§九） | 重，46,052 個 GLB | 可能需要 | **新增數十 GB** | 數小時 |
| Stage 1 正式訓練（若批准） | — | **需要** | 每個 arm 約 330 MB | 依輪數，先前 25 輪的階梯有紀錄可查 |

**Objaverse 的 46K 重渲不在這張表上，因為規格禁止。** 若日後解除，成本是先前估的約 43 小時。

## 需要 Kyzen 裁示的四件事

**一、λ 初值 9.0 與 ESSGNN 的 normalised_sum。** 兩個權威衝突：
你 9/2 裁了 pooling 與 λ=0.1，新規格說兩者都 UNRESOLVED 且禁止自行決定，
而程式裡跑的是推導出來的 9.0。**MASTER 不選。**
（`decided_by` 那句假的歸屬，無論你怎麼裁都要改。）

**二、253 個渲染壞掉的已收錄資產**，其中 11 個幾乎全空白，47 個異常落在封存的測試集裡。
它們現在是正式的檢索候選，影像特徵是從空白畫面算出來的。
選項：留著並在報告裡寫明；或移入隔離並記為語料變更（會改變 45,692 這個數字與切分）。

**三、遮罩向量在權重衰減組裡（M-3）。** 訓練完之後它與初始化無法區分，
只有真實向量的約 2%。論文說「遮罩嵌入而不是補零」，我們名義上做到、數值上接近補零。
選項：移出衰減組（形狀上更接近論文意圖）；或維持現狀並在報告裡記錄這個量測。

**四、ProcTHOR 節點文字要不要修。** 修＝4 分鐘 GPU，但必須與語意邊綁在一起做。
不修＝磁碟上永遠是 `a c d` / `a t v stand` / `a apple` 這種字串。
規格說節點文字建構本身是 UNRESOLVED，所以這也可能等到那題有結論再一起做。

## MASTER 對規格 §二十 的自我評估

四十題全部有答案，其中三十二題由三位稽核者實測，八題（33–40）由 MASTER 綜合。
**沒有任何一題是靠推測填的。** 沒有答案的地方都寫成 UNRESOLVED 或「沒有驗證」。

規格 §十九 的禁止清單，逐項確認**全部沒有做**：
沒有 46K 重渲、沒有全量標註、沒有 ProcTHOR 全渲、沒有刪第 12 張、
沒有只留平均（本來就沒有）、沒有覆寫原始資料、沒有刪任何 cache、
沒有把觀測政策寫死、沒有把 gallery 範圍固定成任何一邊、
沒有用固定點雲向量取代可訓練的 PointBERT、沒有自行產生語意邊、
沒有自行決定 pooling 或 λ 初值。
