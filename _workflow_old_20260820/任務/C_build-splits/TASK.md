# TASK — C_build-splits

> **STATUS: BLOCKED。尚不可 fork。**
> 這份 TASK.md 是 execution contract，支線不得自行改寫 Objective / Scope / Non-Scope / DoD。

## Branch Name

`C_build-splits`

## Task ID

M2.1（`主線.md`）／ T5.1（`TASKS.md`）／ node `n09_build_splits`

## Objective

執行 `n09_build_splits`，產出 Stage 1 所需的三個協定檔：
`splits.json`、`eval_protocols.json`、`stage1_protocol.json`。

## Why Now

`metafind/train/stage1.py:356` 在 `splits.json` 不存在時直接 `return 2`。
三個檔案 Master 已實測**全部不存在**，n09 從未執行過。這是 Stage 1 的硬門檻。

## Current Context

- 三個檔案皆不存在（Master 實測 `ls` 失敗）。
- `admitted_uids()`（`splits.py:157-172`）取
  `pointclouds_index ∩ renders_index ∩ annotations_index` 的交集。
  實測行數：`46052 / 45955 / 45955` → 交集預期 45,955。
- **`splits.py:142` 把 hyperparameters 的 sha256 烤進 `stage1_protocol.json`。**
  τ 事後一改，`stage1.py:83-87` 直接 raise。
- **`splits.py:130` 的 `tower_sharing` 來自常數 `DEFAULT_TOWER_SHARING`（`splits.py:80`），
  `main()` 沒有 CLI 參數。** 同理 `DEFAULT_FUSION`（`splits.py:75`）。
- `main()` 只有 `--seed` 與 `--decided-by` 兩個旗標。

## Blocking Conditions（全部解除才可 fork）

| 阻擋 | 內容 | 來源 |
|---|---|---|
| A | n06 全量完成 | 支線 A |
| B | U-16 證據卷宗完成並經 Master 驗收 | 支線 B |
| **D-α** | τ 裁決（0.5 或 0.07）→ 決定要不要先改 `DEFAULT_HYPERPARAMETERS` 並重跑 n05b | Kyzen |
| **D-β** | U-16 裁決 → 決定要不要先改 `DEFAULT_TOWER_SHARING` | Kyzen |
| **D-γ** | 3 筆 v1 殘留如何處理 | Kyzen |

### D-γ 為何擋 C

`admitted_uids()` 只讀標註 index（45,955 行），**會把 3 筆無 embedding 的 uid 放進 split**；
`stage1.py:110` 的 `np.load(paths.EMBEDDINGS / f"{uid}.npz")` **無防護** → 訓練途中 `FileNotFoundError`。
（Master 實測 `serialize_annotation()` 對這 3 筆丟 `KeyError: 'width'`。）

屬 **dataset construction**，依 `.claude/rules/research-rigor.md` §2 必須由 Kyzen 裁決。

## Authoritative Inputs

1. `docs/graph/node_registry.yaml` 的 `n09_build_splits`（L435）與 `G3_object_corpus`（L474）
2. `docs/graph/graph_spec.yaml` 的 U-09（gallery scope）、U-01（asset count）、U-24（similarity）
3. `metafind/data/splits.py`
4. `metafind/models/stage1_config.py`（`canonical_hyperparameter_hash`）
5. `data/outputs/logs/{pointclouds,renders,annotations}_index.jsonl`

## Dependencies

A、B（Master 驗收後）、D-α、D-β、D-γ。

## Scope

**待 D-α/β/γ 裁決後由 Master 補完。** 骨架：

1. 依 D-α 結果決定是否先改 `DEFAULT_HYPERPARAMETERS` 並重跑 n05b
2. 依 D-β 結果決定是否先改 `DEFAULT_TOWER_SHARING`
3. 依 D-γ 結果處理 3 筆殘留（可能是改 `admitted_uids()`、可能是移動檔案、可能是補標註）
4. 跑 `python -m metafind.data.splits`
5. 驗收

## Explicit Non-Scope

- 不跑 Stage 1（那是 D 支線）
- 不動 n06 產出
- 不改 ESSGNN 相關任何檔案
- 不改 `主線.md` / `支線任務.md` / `TASKS.md`
- 不改其他支線的 `TASK.md` / `HANDOFF.md` / `CODEX_REVIEW.md`

## Expected Deliverables

1. `data/outputs/splits.json`
2. `data/outputs/eval_protocols.json`
3. `data/outputs/stage1_protocol.json`
4. `任務/C_build-splits/HANDOFF.md`

## Likely Files

`data/outputs/*.json`；視 D-α/β/γ 可能觸及 `metafind/data/splits.py`、`metafind/models/resolve_stage1.py`。

## Required Verification

- [ ] 三個 json 皆產出
- [ ] 80/20 切分，`train_fraction == 0.8`
- [ ] 洩漏 0（`set(train) & set(test)` 為空）
- [ ] `gallery_size` **由切分推導**，非寫死
- [ ] `stage1_protocol.hyperparameter_config_hash` == `stage1_hyperparameters.sha256`
- [ ] `python -m metafind.train.stage1 --limit 8 --epochs 1` 至少能通過 `load_protocols()`（不必跑完）
- [ ] `tools/check_graph.py` 的 G3 判準通過
- [ ] `pytest tests/test_splits.py -q` 通過

## Research Risks

- τ 與 tower_sharing 一旦寫進 `stage1_protocol.json` 就綁定了 Stage 1 的科學條件；改動需重跑 n09。
- `admitted_total` 是 Table 1 的分母（U-01/U-09），數字錯會讓所有 recall 都錯。

## Implementation Risks

- 秒級執行，風險低。主要風險在前置的常數修改被漏掉。

## Codex Review Requirement

**MEDIUM。** 若 D-γ 導致修改 `admitted_uids()`，升級為 **FULL** —— 那是 dataset construction 變更。

## Definition of Done

三個 json 產出且通過全部 Required Verification。**不得推進到 Stage 1 訓練。**

## Return-to-Master Requirements

標準 HANDOFF 格式（見 `任務/INDEX.md` 規則），另附三個 json 的關鍵欄位摘要與
`admitted_total` / `len(train)` / `len(test)` 實際數字。
