# TASK — D_stage1-train

> **STATUS: BLOCKED。尚不可 fork。**
> 這份 TASK.md 是 execution contract，支線不得自行改寫 Objective / Scope / Non-Scope / DoD。

## Branch Name

`D_stage1-train`

## Task ID

M3（`主線.md`）／ T5（`TASKS.md`）／ node `n10_train_stage1`

## Objective

先跑 Stage 1 smoke（200 資產、1 epoch），驗收通過後再跑全量訓練，產出 `stage1.pt` 與 `stage1_ckpt.json`。

## Why Now

**Stage 1 從未成功跑過一次。`data/outputs/checkpoints/` 是空的**（Master 實測）。
M4 / M5 / M6 全部等它。

## Current Context

- `data/outputs/checkpoints/` 空。
- `tools/chain_after_n05.sh` 已含一段可複用的 smoke + checkpoint 驗收邏輯（含
  `backbone_trainable_state > 1_000_000` 的 PointBERT 存在性斷言）。
- **已知實作缺口（Master 實讀確認，不擋跑但需裁決）：**
  `stage1.py:309-338` 的 `build_model()` **不使用** `Stage1RuntimeConfig`，
  自己從原始 dict 組 config，且只建**一個** backbone。後果：
  - `tower_sharing="fully_separate"`（需兩個 backbone）**n10 做不到**
  - `stage1.py:322-325` 把**同一個 `FusionConfig` 物件**同時給 query 和 gallery
  - 34 條 `test_dual_tower.py` 測的是訓練器繞過的那個 class

## Blocking Conditions

| 阻擋 | 內容 |
|---|---|
| A | n06 全量 embeddings |
| C | 三個協定檔 |
| **D-ε** | `build_model()` 要走 `Stage1RuntimeConfig`，還是明確記錄為何不走 —— **新增決定，需 Kyzen 裁決**。若 D-β 選 `fully_separate`，此項變成硬阻擋 |

## Authoritative Inputs

1. `docs/paper/metafind_source/2methdology.tex` §2.6（訓練範圍）、`3experiments.tex:15`（τ）
2. `docs/graph/node_registry.yaml` 的 `n10_train_stage1`（L595）
3. `metafind/train/stage1.py`、`metafind/models/{dual_tower,fusion,losses,stage1_config}.py`
4. `data/outputs/stage1_protocol.json`、`stage1_encoding_protocol.json`、`stage1_hyperparameters.json`

## Dependencies

A、C、D-β、D-ε。

## Scope

**待 C 完成、D-ε 裁決後由 Master 補完。** 骨架：smoke → 驗收 → 全量 → 驗收 → 記錄可重現性資訊。

## Explicit Non-Scope

- 不建 gallery 索引（G 支線）
- 不跑 Stage 2
- 不改 ESSGNN
- 不改 `主線.md` / `支線任務.md` / `TASKS.md`
- 不改其他支線的檔案

## Expected Deliverables

1. `data/outputs/checkpoints/stage1.pt`
2. `data/outputs/checkpoints/stage1_ckpt.json`
3. `data/outputs/logs/train_stage1.jsonl`
4. `任務/D_stage1-train/HANDOFF.md`

## Likely Files

`data/outputs/checkpoints/`、`data/outputs/logs/`。**不應有 `metafind/` 變更**（除非 D-ε 裁決要求）。

## Required Verification

- [ ] smoke 通過後才跑全量
- [ ] checkpoint 三段（`backbone_trainable_state` / `tower_trainable_state` / `loss_trainable_state`）都在
- [ ] `backbone_trainable_state` 參數量 > 1,000,000（PointBERT 確實在 checkpoint 裡）
- [ ] `assert_checkpoint_covers_optimizer` 通過
- [ ] `data/outputs/logs/train_stage1.jsonl` 有訓練曲線
- [ ] 記錄：git commit SHA、seed、環境、硬體（**RTX 5090 32GB**，非 repo 多處所寫的 4090 24GB）
- [ ] loss 有限、非 NaN；記錄起訖值

## Research Risks

- **若 D-α 選 τ=0.5：0.5 對 InfoNCE 偏大，可能收斂慢、對比訊號弱。
  訓不起來本身就是值得報告的發現，不是換掉它的理由。** 不得為了讓 loss 好看而改 τ。
- `build_model()` 繞過 `Stage1RuntimeConfig`（D-ε）意味著跑出來的模型與 34 條測試測的不是同一個 class。
  這必須在 HANDOFF 中明載，不得隱藏。

## Implementation Risks

- 長跑；需 `nohup` 與續跑能力。
- 32GB VRAM 的可行性尚未實測（repo 多處以 24GB 為前提，須重新量測 batch_size=64 是否放得下）。

## Codex Review Requirement

**FULL。** 這是本專案第一次真正的訓練，科學條件必須被獨立審查。

## Definition of Done

全量 checkpoint 產出且通過全部 Required Verification。**不得自行推進到 M4。**

## Return-to-Master Requirements

標準 HANDOFF 格式，另附 `.claude/rules/experiments.md` §20 所列的完整實驗報告欄位。
