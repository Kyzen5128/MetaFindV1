# MetaFind 復現 — 逐步建置流程（對照用）

> 這份是 [`01_GRAPH_SPEC.md`](01_GRAPH_SPEC.md) 的**可執行版本**：把 graph 的節點翻成
> 實際要寫的檔案、要跑的指令、以及**通過條件**。
> 每一步的「驗證」欄都是對**內容**的斷言 —— 「跑完了」「檔案存在」「exit 0」一律不算通過（V3）。
>
> 使用方式：實作時逐步對照，每步的 exit condition 沒過就不要進下一步。

---

## 專案結構（目標）

```
MetaFindV1/
├── docs/
│   ├── metafind_paper.md
│   └── graph/                     ← 本設計文件
├── metafind/
│   ├── state/
│   │   ├── channels.py            # 40 個 channel 的 schema 與 merge policy
│   │   └── checkpoint.py          # B1 durable progress（原子寫入）
│   ├── data/
│   │   ├── objaverse_stream.py    # SG1：shard 串流 + disk backpressure
│   │   ├── render.py              # 11 視角渲染
│   │   ├── annotate.py            # GPT-4o 標註 + C1 修復迴圈
│   │   ├── procthor.py            # SG2：房屋 → scene graph
│   │   └── sem_edges.py           # 語意邊 + 跨 run 快取
│   ├── models/
│   │   ├── ulip_backbone.py       # 凍結的 ULIP-2（D1）
│   │   ├── fusion.py              # mean / mlp / masked_mlp / gated / transformer
│   │   ├── essgnn.py              # EGNN + 語意邊投影（F8）
│   │   ├── dual_tower.py          # query / gallery 雙塔
│   │   └── losses.py              # MetaFindDualTowerLoss（F3，不可複用 ULIP）
│   ├── train/
│   │   ├── stage1_align.py
│   │   └── stage2_layout.py
│   ├── eval/
│   │   ├── retrieval.py           # instance-level R@1/R@5（F3，自寫）
│   │   ├── equivariance.py        # SC-5
│   │   ├── compose.py             # SG4：Algorithm 1
│   │   └── judge.py               # GPT-4o 四維度評分
│   ├── gates/
│   │   └── runner.py              # gate record + exit code contract
│   └── baselines/                 # 6 個 baseline 的統一介面
├── configs/
│   ├── variants.yaml              # 10 個變體的靜態註冊表
│   └── budget.yaml                # G1 的預算上限
├── tests/
│   ├── l1/                        # 47 條
│   └── l2/                        # 15 條
└── runs/                          # 所有產物（內容定址）
```

---

## Phase 0 — 可行性（graph layers 1–4）

### Step 0.1 — `n01_env_bootstrap`

**做什麼**

1. 建 conda/venv，鎖定與 torch 2.9.1 相容的版本組合（**不要照抄 ULIP `requirements.txt`**，見 F4）。
2. Patch `dataset_3d.py:544` 的 `from torch._six import string_classes` → `string_classes = str`。
   建議做成 `metafind/compat/ulip_patch.py`，**不直接改 clone 的 repo**（保持可重新 clone）。
3. 驗證 open_clip `ViT-bigG-14` 可載入。

**驗證（通過條件）**

| 檢查 | 斷言 |
|---|---|
| L1-ENV-ULIP | `ULIP2_PointBERT_Colored` 能被實例化，且 `model.pc_projection.shape[1] == 1280` |
| L1-ENV-EGNN | `EGNN(in_node_nf=1280, hidden_nf=128, out_node_nf=1280, in_edge_nf=64).forward()` 回傳 `h.shape==(n,1280)`, `x.shape==(n,3)` |
| L1-ENV-DET | 記錄 `torch.use_deterministic_algorithms(True)` 是否可行；不可行則記入 `nondeterminism_sources` |

**Exit**：三條 L1 全綠。**失敗不重試**（`DETERMINISTIC_INPUT`），印出版本衝突矩陣。

---

### Step 0.2 — `n02_acquire_sources`

**做什麼**

| 來源 | 取得 | 落點 |
|---|---|---|
| ULIP-2 checkpoint | HF `SFXX/ulip` → `initialize_models` + ULIP-2 PointBERT 10k-xyzrgb | `runs/sources/ulip2/` |
| Objaverse-LVIS 清單 | `lvis.json` + `objaverse_lvis_metadata.json` | `runs/sources/objaverse-lvis/` |
| ProcTHOR-10K | AI2 官方 | `runs/sources/procthor/` |

**驗證**

| 檢查 | 斷言 |
|---|---|
| L1-SRC-HASH | 每個檔案的 sha256 寫入 `source_manifest`（`write_once`）；**不符即 `CONTRACT_VIOLATION` fail closed** |
| L1-SRC-LOAD | ULIP-2 ckpt 能 `load_state_dict`，且 zero-shot ModelNet40 top-1 ≈ **50.6**（README 公佈值，±1.0） |

> L1-SRC-LOAD 很重要：它是「我們拿到的是**對的**權重」的唯一證明。
> 只驗 sha256 只能證明「檔案沒壞」，不能證明「是我們要的那個」（V2：驗不壞 ≠ 驗成立）。

**Exit**：兩條全綠，`source_manifest` 已寫入。

---

### Step 0.3 — `n03_pilot_and_budget`（= SG1 mode=pilot，500 資產）

**做什麼**：用完整的 SG1 流程跑 **500 個資產**，實測三件事：

```
cost_per_asset      = GPT-4o 實際花費 / 500
schema_pass_rate    = 首次通過 schema 的比例
bytes_per_asset     = (渲染 + 點雲 + 標註) 峰值磁碟 / 500
seconds_per_asset   = wallclock / 500
```

外推到 48K，寫入 `cost_projection`。

**A1 決策點**：`cost_projection.money > budget_cap` → 走 `reuse_ulip2_captions` 分支（F7）。
**預設分支是便宜的那個**，不是貴的那個。

**驗證**

| 檢查 | 斷言 |
|---|---|
| L1-PILOT-SIDECAR | 500 筆 sidecar 全部落檔，且 `admitted + quarantined == 500` |
| L1-PILOT-REASON | 每筆 quarantine 都有真實 `exception_type` + `message`（注入一個「只寫失敗」的版本必須被擋） |
| L2-PILOT-EXTRAP | 外推公式對 500 筆自身回代誤差 <5% |

**Exit**：`cost_projection` 四個欄位皆已寫入且非 UNKNOWN。

---

### Step 0.4 — **G1_feasibility**（G-COST）

```
PASS ⟺ disk_free ≥ peak_shard_bytes × 3
     ∧ cost_projection.money ≤ budget.money_cap
     ∧ cost_projection.gpu_hours ≤ budget.gpu_cap
     ∧ pilot_schema_pass_rate ≥ 0.95
```

| verdict | rc | 動作 |
|---|---|---|
| PASS | 0 | → Phase 1 |
| FAIL | 2 | **停**。縮小規模 / 改走 ULIP-2 captions / 清磁碟後重跑 Step 0.3。**不得調高 `money_cap` 讓它變綠**（GE） |
| BLOCKED_EVIDENCE | 3 | 外推缺欄位 → 補跑 pilot；或預算需人核准 → `n22_budget_approval` |

**Gate record** 寫入 `runs/gates/G1_*.yaml`，含 `is_terminal: true`。

> ⚠️ **R-01 提醒**：目前 `disk_free = 108GB`。以 shard=2000 資產估算
> `peak_shard_bytes ≈ 3GB`，`3×3=9GB` 是過得了的 ——
> **但這是因為 D3 串流架構才過得了**。若沒有 D3、要一次落地 70GB+ 原始檔，這個 gate 會直接 FAIL。

---

## Phase 1 — 資料（graph layers 5–7）

### Step 1.1 — `n04_object_prep`（SG1，24 shards，**可與 1.2 / 1.3 平行**）

**每個 shard 的內部流程**

```
s1a validate_mesh      → 非流形/空幾何 → quarantine（不重試）
     ├→ s1b render_11_views   (GPU)   ┐
     └→ s1c sample_pointcloud (10000 pts, xyz+rgb) ┤ join=all
s1d annotate_gpt4o     ← s1b          ┘
s1e validate_annotation → schema 失敗 → C1 修復迴圈（bound 2）→ 仍失敗則 quarantine
s1f encode_1280d       → text/image/pc 各一個 1280-d 向量（D2 的核心）
s1g write_vectors_and_sidecar  → fsync
s1h DELETE_RAW         ← 只有在 s1g 完成且 sha256 已驗證後才執行（不可逆，late commit）
```

**disk backpressure（A1）**：每個 shard 開始前檢查 `disk_free`；
不足 → `shrink_shard` → 仍不足 → **`BLOCKED`**（等人清空間），**不是 FAILED**。

**驗證**

| 檢查 | 斷言 |
|---|---|
| **L2-COMPLETE** | `len(admitted) + len(quarantine) == len(asset_catalog)`，無重複、無非預期成員。**刪掉一筆結果，測試必須失敗** |
| **L2-RESUME** | 在第 13 個 shard 中途 `kill -9`，重跑後 `asset_embeddings` 與不中斷版本**逐位元組相同**，且 GPT-4o 呼叫次數**不增加**（跳過判定依 sha256，不依檔案存在） |
| L1-DELETE-ORDER | sidecar 未 fsync 時呼叫 `s1h` 必須 raise |
| L1-MERGE-COMM | 打亂 shard 完成順序，`asset_embeddings` 結果相同 |
| L1-EMB-DIM | 每個向量 `shape == (1280,)`，無 NaN，`norm > 0` |

**Exit**：`asset_admitted` 已寫入，`quarantine_rate ≤ 2%`。

---

### Step 1.2 — `n05_scene_prep`（SG2，**與 1.1 平行**）

```
s2a parse_house        → ProcTHOR house → rooms → objects{position, metadata}
s2b physical_edges     → 幾何規則（adjacency / support），確定性
s2c enumerate_pairs    → 同房間共現物件對，每房上限
s2d semantic_edge_llm  → 先查 sem_edge_cache[(cat_a,cat_b)]，miss 才呼叫 LLM
                          → C2 修復迴圈（bound 2）→ 仍失敗則標 semantic_edge_missing
s2e encode_edge_text   → frozen text encoder → e_ij (1280-d)
s2f project_edge       → Linear(1280 → 64)  ← F8，必要
s2g assemble_graph     → edge_index（稀疏，自寫，不用 egnn 的 get_edges）
```

**驗證**

| 檢查 | 斷言 |
|---|---|
| L1-CACHE-KEY | 同一 `(cat_a, cat_b)` 第二次查詢 **不觸發 LLM 呼叫**（用呼叫計數斷言，不是看時間） |
| L1-EDGE-SPARSE | `edge_index` 是稀疏的；若退化成全連接（`n×(n-1)` 條邊）必須被擋 |
| L1-EDGE-PROJ | `edge_mlp` 輸入維度 == `2×128 + 1 + 64 == 321`（F8） |
| L2-CACHE-SAVING | 全量跑完後 `llm_calls / total_pairs < 0.1`（快取確實省下 ~40×） |

**Exit**：`scene_admitted` 已寫入，`scene_quarantine_rate ≤ 5%`。

---

### Step 1.3 — `n13_run_baselines`（**與 1.1 / 1.2 平行**）

6 個 baseline × 7 種模態條件。**不依賴我方任何訓練**，所以放在這裡跑完，不佔關鍵路徑。

**必須刻意重現的行為**：baseline 的 PC-Only 用**同一組 embedding** 當 query 與 gallery（F3）。

**驗證**：`L2-PCONLY` —— 斷言 baseline PC-Only 的 query embedding 與 gallery embedding
**逐位元組相同**。這不是 bug，是論文註腳描述的協定，必須忠實重現。

**Required Audit RA-2** 在此產生紀錄（與各 baseline 原論文公佈值比對）。
**它失敗不阻斷任何事**，只縮小「baseline 對照是忠實復現」這個 claim。

---

### Step 1.4 — `n06_build_splits`

```
物件級 80/20（seed 寫入 split_seed，write_once）
房屋級 80/20（依 house_id 切，不是依 room 或 object）
鎖定 gallery_size_locked ← len(asset_admitted)   ⟵ U-04 的收斂點
```

**驗證**

| 檢查 | 斷言 |
|---|---|
| **L2-LEAK** | `train_ids ∩ test_ids == ∅`（物件）∧ `train_houses ∩ test_houses == ∅`（房屋）∧ **同一資產不同時出現在 train 房屋與 test 房屋的節點裡** |
| L1-WRITEONCE | `gallery_size_locked` 二次寫入不同值必須 raise |

---

### Step 1.5 — **G2_corpus_validity**（G-INVALID）

```
PASS ⟺ leakage_count == 0
     ∧ len(admitted) + len(quarantine) == len(catalog)
     ∧ gallery_size_locked 已寫入且 > 0
     ∧ quarantine_rate ≤ 0.02
     ∧ schema_pass_rate ≥ 0.95
```

**on_fail**：停。修資料管線後重跑 Step 1.1–1.4。
**帶著洩漏往下訓練，得到的 R@1 不算數** —— 這正是 G-INVALID 的檢驗問題。

---

## Phase 2 — 訓練（graph layers 8–11）

### Step 2.1 — `n07_train_stage1_align`

在**快取的 1280-d 向量**上訓練雙塔（D2），不碰 ViT-bigG-14。

- Query tower：各模態 30% 獨立遮罩（**masked embedding，不是 zero-padding**）→ Fusion → 1280-d
- Gallery tower：modality-complete → 1280-d
- Loss：**自寫** `MetaFindDualTowerLoss`（Eq.5），**不可複用** `ULIPWithImageLoss`（F3）

**驗證**

| 檢查 | 斷言 |
|---|---|
| L1-LOSS-SYMM | Eq.(7a)+(7b) 的雙向 loss 在 query/gallery 互換時數值相同 |
| L1-MASK-NOTZERO | 遮罩後的 embedding **不等於零向量**（masked ≠ zero-pad，這是 Table 3 最後一列的對照組差異） |
| L1-IDEMPOTENT | 同 config + 同 seed 跑兩次，ckpt sha256 相同（若不同 → 記錄非確定性來源） |

**ckpt 路徑內容定址**：`runs/ckpt/{config_hash}_{data_hash}_{seed}_{code_rev}.pt` → 天然不覆蓋。

---

### Step 2.2 — `n08_build_gallery_staging` → **G3** → `n09_promote_gallery_index`

**這三步的順序是刻意的**（late commit）：

```
n08  寫入 runs/index/staging/{stage1_ckpt_sha}.faiss     ← staging，不是正式位置
G3   驗證 staging 索引                                     ← G-CONTAM
n09  驗證通過才 promote 到 runs/index/promoted/           ← write_once
```

**G3 判準**

```
PASS ⟺ dim == 1280
     ∧ count == gallery_size_locked
     ∧ 無 NaN、無零向量
     ∧ 抽 1000 筆自我檢索 recall@1 == 1.0
```

最後一條是關鍵：**用 gallery 自己的 embedding 當 query，必須 100% 撈回自己**。
撈不回自己 = 索引建構有問題。這也順便解釋了 F3 說的 baseline PC-Only 灌水現象。

**on_fail**：**不 promote**。staging 索引作廢重建。
壞索引一旦 promote，Table 1/2/3 全部被污染且事後分不出來 —— 這就是 G-CONTAM 的檢驗問題。

**索引選擇**：48K × 1280 × 4B = 246MB，**用精確內積，不用 ANN**（消除一個不確定性來源）。

---

### Step 2.3 — `n10_train_headline`（SG3 ×2）+ `n11_equivariance_probe`

兩個 headline 變體：`w/ ESSGNN`、`w/o ESSGNN`。

**ESSGNN 實作要點（F1/F2/F8/F9 全部落在這裡）**

```python
# ✅ 正確：h 只含語意，座標走 coord 通道
h0 = t_i                                    # (n, 1280)，不是 Concat(x, t)
e_ij_proj = Linear(1280, 64)(e_ij)          # F8：先降維
egnn = EGNN(in_node_nf=1280, hidden_nf=128,
            out_node_nf=1280, in_edge_nf=64,
            coords_agg='sum')               # F9：論文 Eq.3 是 sum，不是預設的 mean
h_out, x_out = egnn(h0, x, edge_index, e_ij_proj)
e_layout = pool(h_out)                      # (1280,)，對齊 Eq.6 的殘差加法
```

**`n11_equivariance_probe`（SC-5）**

```
對 100 個隨機場景 × 100 組隨機 (R, T)：
  座標通道：‖ESSGNN(Rx+T)_x − (R·ESSGNN(x)_x + T)‖∞ < 1e-4
  特徵通道：ESSGNN(Rx+T)_h == ESSGNN(x)_h   （完全不變）
```

同時跑 **Required Audit RA-1**：用 `h0 = Concat(x, t)` 的版本跑同一個測試。
**它預期會失敗**（F1）。失敗**不阻斷任何事**，只把 claim 縮小為
「§2.5 的字面寫法與 Appendix C 的證明前提矛盾」。

> 絕對不要因為 RA-1 過不了就放寬 `1e-4` 的容差 —— 那是 anti-pattern #13。

**其他驗證**

| 檢查 | 斷言 |
|---|---|
| L1-SEMEDGE-ZERO | 語意邊全置零時，兩個**幾何不同**的 layout 仍產生不同 `e_layout`（F8 退化偵測器） |
| L1-LAMBDA | `λ` 是可學習純量且有梯度；固定成常數的版本必須被測出來 |
| L1-DROPOUT-30 | scene dropout 在 30% 批次省略 `e_layout`，統計上可驗證 |

---

## Phase 3 — 評估（graph layers 12–16）

### Step 3.1 — `n12_eval_object_retrieval` → Table 1

7 種模態條件 × 2 個變體 = 14 格。**自寫 instance-level 檢索評估器**（F3）。

**驗證**：`L2-ROUTING-COV` —— 7 個條件每一個都被實際走到（不是只跑 full 然後推算）。

**預期要看到 SC-3**：MetaFind 的 PC-Only（63–75）**低於** baseline（98–99）。
**這是正確的復現結果，不是失敗。**

---

### Step 3.2 — `n14_compose_scenes`（SG4）→ `n15_judge_gpt4o` → Table 2

**SG4 = Algorithm 1**，四件套：

| 項目 | 值 |
|---|---|
| progress measure | `placed_count`（merge=`max`，重試不重複計數） |
| semantic exit | `placed_count == N` |
| hard bound | `N ≤ 25` ∧ `wallclock ≤ 600s/場景` ∧ `retrieval_calls ≤ 30` |
| **exhaustion** | 場景標 `incomplete: true`，**排除在 Table 2 平均之外並另行報告** |

**每輪必須清空**：`layout_embedding`、`query_modality_embeds`（`reset_on: loop_entry`）。
忘了清 = 上一個物件的 layout 污染這一輪。

**O4 decision log**：每一步記 top-5 候選 + 分數 + 被選中者 + `λ·‖e_layout‖`。
沒有這個，事後無法解釋「為什麼這個場景不協調」。

**驗證**

| 檢查 | 斷言 |
|---|---|
| L1-ITER-RESET | 第 2 輪進入時 `layout_embedding` 為空；注入殘留值必須被偵測 |
| L1-EXHAUST-MARK | 觸發 bound 後狀態為 `EXHAUSTED` 且帶 `terminated_by`；被標成成功則測試失敗 |
| **L2-DEADLOCK** | 故意把回邊併回 `init` join_group，**必須偵測到 deadlock**（E4 的負向證明） |

---

### Step 3.3 — **G4_human_study_commit**（G-COST）

```
PASS ⟺ equivariance_max_err < 1e-4
     ∧ all(table1_R@1_delta ≤ 3.0pp)
     ∧ scene_complete_rate ≥ 0.90
     ∧ annotator_staffing == 5
```

**為什麼這是 gate**：下一步是 **5 人 × 200 場景 × 4 方法 ≈ 67 人時**。
人的時間花掉就退不回來。錯了要整段重來 —— 這正是 G-COST 的檢驗問題。

`rc=3`（例如標註者還沒排定）→ `n22_budget_approval`，回程走 `reapproval` join_group。

---

### Step 3.4 — `n16_human_study` ∥ `n17_train_ablations`

**`n16`（human，會 `BLOCKED` 數天）**
- 5 位專家 × 200 場景 × 4 維度（1–5 分）
- 逾時 → 提醒 → `BLOCKED`（**不是 FAILED**，不重跑已完成階段）
- **≥4 人完成** → 用完成者計算並記錄 n；**<4 人** → Table 2 人工欄 `BLOCKED_EVIDENCE`（rc=3）
- annotator 身分以 `annotator_hash` 取代（B5）

**`n17`（SG3 ×8）** —— 因為 D2，這 8 個變體從「各需一次完整預訓練」變成「各幾分鐘」：

| # | 變體 | 需重訓 Stage-1？ |
|---|---|---|
| 1 | Full w/ iterative & ESSGNN | 複用 headline |
| 2 | w/o iterative retrieval | **否**（只換 SG4 的 `composition_mode`） |
| 3 | w/o Layout Context | 複用 headline |
| 4 | w/ Layout Context (GAT) | 否（換 layout encoder） |
| 5 | Fusion = Mean | 否 |
| 6 | Fusion = MLPs | 否 |
| 7 | Modality Dropout = 10% | **是**（但在快取向量上，很便宜） |
| 8 | Modality Dropout = 50% | **是** |
| 9 | Train fuser only | 否 |
| 10 | Padding missing modalities with 0 | **是**（U-03：兩版都跑） |

`configs/variants.yaml` 是**靜態註冊表**（cardinality 封閉 → 這是 A1 靜態 fan-out，不是 A3）。

**Required Audit RA-3** 在第 9 列產生紀錄：因 D2 凍結 backbone，
「Fine-tuning entire encoder > train fuser only」**預期無法完整驗證** → 縮小 claim，不阻斷。

---

## Phase 4 — 報告（graph layers 17–20）

### Step 4.1 — `n19_aggregate_tables`

**雙 join_group**（本圖最值得注意的設計）：

```
core     = {n12, n13, n15}  policy=all           ← Table 1 與 Table 2 GPT-4o 欄必須齊
extended = {n16, n18}       policy=all_settled   ← 人工欄與 Table 3 可以缺
trigger  = all_groups_satisfied
```

缺 `extended` 的成員 → 正常產出報告，但**標記 claim 縮小**。
一個標註者請假不該擋掉整份報告；但 Table 1 缺格必須擋。

### Step 4.2 — `n20_compare_to_paper`

逐格輸出三選一：**復現 / 復現失敗 / 證據不足**。對照 SC-1…SC-8。

### Step 4.3 — **G5_report_release**（G-IRREVERSIBLE）

```
PASS ⟺ Table 1/2/3 每格都有明確判定（無留白）
     ∧ 所有 gate record 齊全且 is_terminal == true
     ∧ RA-1 / RA-2 / RA-3 三份紀錄都在
     ∧ D1 / D2 / D3 三項偏離已在報告中明列
     ∧ 所有 UNKNOWN（U-01…U-05）的處置已寫明
```

### Step 4.4 — `n21_publish_report`

發布前建 git tag。**發布是不可逆的** —— 撤稿只能發勘誤 + 標記 tag `INVALIDATED`，
已被讀取的內容撤不回（§11.2 的誠實界線）。

---

## 貫穿全程的三條紀律

### 1. 進度真相是 checkpoint，不是 stdout（B1）

```
原子寫入：tmp → fsync → rename
stdout 失效不得讓工作失敗，但要記 stdout_broken: true
```

48K 資產標註要跑好幾天，SSH 一定會斷。若進度只在 stdout，
斷線後你會看到「好像停了」但 worker 其實還在寫檔 —— 或者反過來，更糟。

### 2. 每筆處理單位都有 sidecar（B2）

```json
{"asset_id":"...", "source_uri":"...", "source_sha256":"...",
 "embed_sha256":"...", "seed":..., "attempt":2,
 "status":"admitted|quarantined", "failure_class":null,
 "exception_type":null, "exception_msg":null,
 "code_revision":"...", "timestamp":"..."}
```

`source_uri + source_sha256` 是 D3 刪掉原始檔之後**唯一的補償路徑**。
沒有它，刪錯了就真的沒了。

### 3. 每個檢查都要有「它真的會擋」的證明（V1/V4）

`validation_plan.yaml` 裡 62 條檢查的 `verified_blocks` **目前全是 `false`**。
實作時每寫完一條檢查，就注入對應的違規，**親眼看到它失敗**，才可以改成 `true`。

> 當一個檢查突然全綠時，第一個懷疑對象是檢查本身，不是被檢查的東西。

---

## 進度追蹤表（實作時勾選）

| Phase | Step | 節點 | 狀態 |
|---|---|---|---|
| 0 | 0.1 | `n01_env_bootstrap` | ☐ |
| 0 | 0.2 | `n02_acquire_sources` | ☐ |
| 0 | 0.3 | `n03_pilot_and_budget` | ☐ |
| 0 | 0.4 | **G1_feasibility** | ☐ |
| 1 | 1.1 | `n04_object_prep` | ☐ |
| 1 | 1.2 | `n05_scene_prep` | ☐ |
| 1 | 1.3 | `n13_run_baselines` | ☐ |
| 1 | 1.4 | `n06_build_splits` | ☐ |
| 1 | 1.5 | **G2_corpus_validity** | ☐ |
| 2 | 2.1 | `n07_train_stage1_align` | ☐ |
| 2 | 2.2 | `n08` → **G3** → `n09` | ☐ |
| 2 | 2.3 | `n10_train_headline` + `n11_equivariance_probe` | ☐ |
| 3 | 3.1 | `n12_eval_object_retrieval` | ☐ |
| 3 | 3.2 | `n14_compose_scenes` + `n15_judge_gpt4o` | ☐ |
| 3 | 3.3 | **G4_human_study_commit** | ☐ |
| 3 | 3.4 | `n16_human_study` ∥ `n17_train_ablations` + `n18` | ☐ |
| 4 | 4.1 | `n19_aggregate_tables` | ☐ |
| 4 | 4.2 | `n20_compare_to_paper` | ☐ |
| 4 | 4.3 | **G5_report_release** | ☐ |
| 4 | 4.4 | `n21_publish_report` | ☐ |
