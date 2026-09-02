# PHASE 1 現況盤點 — 2026-09-03

依 `REPRODUCTION_PROTOCOL_20260903.md` §十八 PHASE 1 與 §二十 的四十題。
全程唯讀：沒有重渲、沒有標註、沒有訓練、沒有刪除，`git status` 在每一路開始與結束時都乾淨。

三路平行，各自負責一段。本檔隨每一路回報逐段補上。

| 段 | 題目 | 負責 | 狀態 |
|---|---|---|---|
| A | 1–15、18、19　Objaverse 資料 | ULIP2 Engineer | 進行中 |
| B | 16、17、20–28　程式碼 | Integrator | 進行中 |
| C | 29–32　ProcTHOR | ESSGNN Engineer | **完成，見下** |
| D | 33–40　migration | MASTER | 等 A、B |

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
