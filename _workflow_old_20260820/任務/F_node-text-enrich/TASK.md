# TASK — F_node-text-enrich

> **STATUS: BLOCKED。尚不可 fork。**
> 這份 TASK.md 是 execution contract，支線不得自行改寫 Objective / Scope / Non-Scope / DoD。

## Branch Name

`F_node-text-enrich`

## Task ID

S3（`支線任務.md`）／ T2 + T3（`TASKS.md`）／ node `n07_scene_graphs` + `n08_semantic_edges`

## Objective

把 ProcTHOR / AI2-THOR 手邊已有、卻沒用上的真值（尺寸、材質、affordance、支撐方向）
納入 ESSGNN 的節點文字 `t_i` 與語意邊 `e_ij`，並保留支撐關係的方向資訊。

## Why Now

`metafind/data/scene_graphs.py:96-104`：

```python
def object_text(category: str) -> str:
    return f"a {humanise(category)}"
```

`Chair_1` … `Chair_50` 全部塌縮成 `"a chair"`。程式自己承認是 ceiling（登記 U-12）。

實測：1,467 個 assetId → **只有 93 種不重複 node text**；
實際不重複配對 4,242，佔 `C(93,2)+93 = 4,371` 理論上限的 **97.0%**。
若以 assetId 去重（前 1500 房）可達 114,417。

## Blocking Conditions

| 阻擋 | 內容 |
|---|---|
| A | n08 需 GPU（實測 22 分鐘），須等 A 讓出 |
| **D-ζ** | **node text 要納入哪些欄位＝研究決定，需 Kyzen 裁決** |
| **D-η** | 反向邊是否該用不同 `e_ij`（`stage2.py:185-187`）＝研究決定 |

### D-ζ 的候選欄位（Kyzen 裁決用）

| 欄位 | 內容 | 覆蓋 |
|---|---|---|
| `bbox_reported` | **真實公尺尺寸**，與深度殼自量差 5.6 mm | 1,467 / 1,467 |
| `salientMaterials` | 受控詞彙 15 個（Plastic/Metal/Wood/Glass/Fabric/Food/…） | AI2-THOR 可查 |
| `pickupable` / `receptacle` | 真實 affordance | 同上 |
| `parentReceptacles` | 真實擺放對象 | 同上 |

**論文 §2.5 只說 `t_i` 是「a text-derived feature」，未規定內容（U-12/U-20）。
所以納入哪些欄位是 `IMPLEMENTATION CHOICE`，不是 `PAPER-FIDELITY FIX`。**

## Authoritative Inputs

1. `docs/paper/metafind_source/2methdology.tex` §2.3（兩種邊）、§2.5（`t_i`）
2. `docs/graph/graph_spec.yaml` 的 U-12、U-19、U-20、U-06
3. `metafind/data/scene_graphs.py`、`metafind/data/semantic_edges_run.py`、`metafind/train/stage2.py:176-190`
4. `data/outputs/procthor_modalities/*.json`、AI2-THOR runtime metadata

## Dependencies

A（GPU）、D-ζ、D-η。

## Scope

**待 D-ζ / D-η 裁決後由 Master 補完。** 骨架：

### S3a — 節點文字
1. 新增擷取步驟，把 AI2-THOR metadata 寫進 `procthor_object_text.json`
2. `object_text()` 納入 Kyzen 裁定的欄位
3. 重跑 n07 → n08

### S3b — 支撐方向
4. `scene_graphs.py:203` 保留 `support_directed`，**另存為有向欄位，不覆蓋現有無向欄位**
5. `semantic_edges_run.py:139` 主賓順序改由支撐方向決定，而非字母序
6. `stage2.py:185-187` 反向邊處理依 D-η

### 順帶的文件修正
7. 「列出所有在同一 house 共現的物件 pair」→ 實際是 `support ∪ kNN(k=8)`（`scene_graphs.py:210`）
8. 「25% 的關係是錯的」→ 只能說「25% 含 might/could/may/often 這類詞彙標記，**不能推斷正確性**」

## Explicit Non-Scope

- ❌ **不要加 no-relation / 過濾。** 論文 §2.3 只說 "obtained by prompting an LLM on object pairs"，
  完全沒提過濾。加過濾是發明，且會改 `edge_index` 拓樸（`stage2.py:176` 直接迭代 `sem_edge_ids`）。
- ❌ 不動 n04 的單位球正規化
- 不改 ESSGNN 模型本體（E 支線）
- 不處理 U-16（B 支線）
- 不跑 Stage 2
- 不改 `主線.md` / `支線任務.md` / `TASKS.md`
- 不改其他支線的檔案

## Expected Deliverables

1. 更新的 `data/outputs/procthor_object_text.json`
2. 重跑後的 `sem_edge_cache.json`、`sem_edge_embeddings.npz`、`sem_edge_sentences.jsonl`、
   `procthor_node_embeddings.{json,npz}`
3. `任務/F_node-text-enrich/HANDOFF.md`

## Likely Files

`metafind/data/scene_graphs.py`、`metafind/data/semantic_edges_run.py`、
可能 `metafind/train/stage2.py`、`data/outputs/`。

## Required Verification

- [ ] 不重複 node text 數 **> 93**（回報實際數字）
- [ ] 不重複配對數**顯著大於 4,242**（回報實際數字）
- [ ] `sem_edge_embeddings.npz` 維度與 `sem_edge_cache.json` 宣告一致
- [ ] 零全零向量、零降級
- [ ] 有向資訊存在 artifact 中且**可還原**
- [ ] 現有無向路徑**不受影響**（可回退對照）
- [ ] 若採用有向 `e_ij`：`edge_attr` 列數 == `edge_index` 行數，且正反向不同
- [ ] 抽樣人工檢視關係句是否比 category-only 版具體（附 5 組前後對照）
- [ ] `pytest tests/ -q` 全過
- [ ] `tools/check_graph.py` all pass

## Research Risks

- 納入尺寸／材質會**改變 `t_i` 的語意分布**，直接影響 ESSGNN 的輸入。
  這是 `IMPLEMENTATION CHOICE` 的變更，必須在報告中登記，不得寫成貼近論文。
- 支撐方向從無向改有向會改變**圖的拓樸語意**，牽動訊息傳遞。U-19 目前宣告無向為我們的慣例，
  且從 n07 → n08 → stage2 全程一致。改它是換一個實作選擇，不是修 bug。
- 重跑 n08 會**覆蓋現有 artifact**。跑前必須備份，否則失去無向版對照組。

## Implementation Risks

- n08 需 GPU 22 分鐘；n07 分鐘級。
- 必須與 T3 合併一次重跑，避免 n08 跑兩次。

## Codex Review Requirement

**FULL。** 涉及 dataset / feature construction 變更。

## Definition of Done

全部 Required Verification 有實測輸出，且無向版對照組已備份保留。
**不得推進到 Stage 2。**

## Return-to-Master Requirements

標準 HANDOFF 格式，另附前後 node text 統計對照、5 組關係句前後對照、備份路徑。
