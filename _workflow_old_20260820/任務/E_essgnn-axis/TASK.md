# TASK — E_essgnn-axis

> **STATUS: BLOCKED。尚不可 fork。**
> ⚠️ **本任務的原始定義（`支線任務.md` S2 ／ `TASKS.md` T1）已由 Master 於 2026-08-20 部分推翻。
> 以本 TASK.md 為準。**
> 支線不得自行改寫 Objective / Scope / Non-Scope / DoD。

## Branch Name

`E_essgnn-axis`

## Task ID

S2（`支線任務.md`）／ T1（`TASKS.md`）／ U-26 ／ C1

## Master 的前提修正（必讀，否則會做錯）

`支線任務.md` S2 與 `TASKS.md` T1 都主張：

> 「**無法在固定其他條件下單獨測 `coord_feat`**」

**Master 於 2026-08-20 實測，此主張不成立。** 實跑四種組合結果：

```
OK    appendix_shared_msg  coord_feat=current   446,212 params
FAIL  appendix_shared_msg  coord_feat=updated   ValueError
OK    sec25_two_mlp        coord_feat=current   412,932 params
OK    sec25_two_mlp        coord_feat=updated   412,932 params
```

（建構參數：`node_feat_dim=edge_feat_dim=out_dim=64, use_io_projections=True, hidden_dim=128, n_layers=4`）

**結論一：在 `sec25_two_mlp` 底下，`coord_feat` 今天就能單獨消融，且參數量完全相同（412,932）。**

**結論二：那個 FAIL 的理由在架構上是成立的。** `essgnn.py:491-503` 的說法是
「附錄版 φ_x 讀 `m_ij`，而 `m_ij` 由 φ_e 從 `h^l` 建構，所以 `h^{l+1}` 在該架構中根本不存在」。
這不是任意拒絕，是架構事實。

> ⚠️ **因此原驗收條件「四種 `(family, coord_feat)` 組合都能建構，無強制」等於要求
> 發明一個附錄沒有的架構。該條驗收已由 Master 刪除，不得執行。**

### 另一項待查的數字矛盾（`UNVERIFIED`）

`支線任務.md` 記載：`appendix_shared_msg` = 164,737 params（4 層），
`sec25_two_mlp` = 213,761 params（2 層）—— 即**附錄版較小**。

Master 實測在上述維度下**方向相反**（附錄版 446,212 > two_mlp 412,932）。

兩者維度設定不同，但**大小關係翻轉**需要解釋。
分類：`UNVERIFIED`。**本支線必須查清楚並在 HANDOFF 中說明**，
不得逕自採用任何一組數字。

## Objective

把 `coord_feat`（唯一 paper-conflicted 的維度）從 `architecture_family`（無論文證據的分類）
的**附屬地位**解放出來，並修正 12 處把 OBSERVED IMPLEMENTATION 寫成 PAPER FACT 的文件與程式註解。

**不改變任何模型的實際數學行為。**

## Why Now

- `essgnn.md`（2026-08-19）已推翻 `docs/graph/graph_spec.yaml` 中 U-26 的既有判讀，
  但**尚未套用到任何檔案**。registry 目前仍記 `U-26 RESOLVED`，理由是已被推翻的符號論證。
- `docs/audit/E_GRAPH_REVALIDATION.md:173` 標 `VERIFIED`，而其理由正是被禁止的符號差異論證。
- 這是純文件與 config 層的清理，**無重跑成本**（ESSGNN 從未訓練過）。

## Blocking Conditions

| 阻擋 | 內容 |
|---|---|
| **FILESYSTEM CONFLICT** | 與支線 B 共用 `docs/audit/C_PAPER_CONTRADICTIONS.md`。**B 未經 Master 驗收前不得 fork E** |
| **D-δ** | 是否採納 `essgnn.md` 的翻案 —— 需 Kyzen 裁決 |

## Authoritative Inputs

1. `docs/paper/metafind_source/2methdology.tex` §2.5（Eq. 3）
2. `docs/paper/metafind_source/appendix.tex`（Eq. 10-13）
3. `docs/paper/egnn_source/sections/model.tex`（φ_x → R^1「outputs a scalar value」）
4. `metafind/vendor/egnn_clean.py`
5. `essgnn.md`（本 repo，441 行，2026-08-19 的重審，**含外部 adjudication 結論**）
6. `metafind/models/essgnn.py`、`metafind/models/resolve_stage2.py`
7. `data/outputs/essgnn_arch_protocol.json`（現值：`architecture_family=appendix_shared_msg, coord_feat=current`）

## Dependencies

B（Master 驗收後）、D-δ。

## Scope

### 必做

1. **`essgnn.py:191-195`** —— 移除 `__post_init__` 由 `architecture_family` 隱式推導 `coord_feat`。
   改為 `coord_feat` 必須顯式提供（比照 `use_io_projections` / `architecture_family` 的 REQUIRED 慣例）。
2. **`essgnn.py:90-91`** —— 註解「different parameter counts, different gradient paths」
   是**我們實作**的性質，非論文性質。改為正確的 evidence class。
3. **`essgnn.py:154-165`** —— `<- primary` 措辭把 INFERENCE 講得像權威，改寫。
4. **`docs/audit/E_GRAPH_REVALIDATION.md:173`** —— 最嚴重的文件錯誤。
   `VERIFIED` 的理由是符號差異論證，該論證已被 `essgnn.md` 推翻。改為正確判定。
5. **`docs/audit/C_PAPER_CONTRADICTIONS.md`** C1 標題「two different ESSGNNs — STRUCTURAL, blocking」改寫。
6. **`docs/graph/00_FINDINGS.md:1091`**、**`docs/graph/README.md:17`**、
   **`docs/graph/02_BUILD_STEPS.md:854`**（根因，U-26 一號兩義）、
   **`docs/graph/graph_spec.yaml:602`**（U-26 被寫成 coord_feat）—— 措辭與編號修正。
7. **`metafind/models/resolve_stage2.py:99-131`** —— ARCH_DECISIONS 的 C1 決策理由改寫。
8. **`tests/test_resolve_stage2.py:179`** —— 移除硬斷言 `== "appendix_shared_msg"`
   （把 INFERENCE 鎖進 CI）。
9. **`tests/test_essgnn.py:24,28`** —— `FAMILY` / `TWO_MLP` 的 primary / competing hypothesis 措辭。
10. 全 repo 搜 `two different ESSGNNs`，應為零命中。
11. 查清上方「數字矛盾」，在 HANDOFF 中說明。

### 明確不做（Master 已刪除的原驗收條件）

- ❌ **不要**讓 `appendix_shared_msg + coord_feat="updated"` 變成可建構。
  該組合在附錄架構下**沒有意義**，`essgnn.py:491-503` 的拒絕是正確的。
- ❌ **不要**為了「讓兩個 family 參數量相同」而改動任一 family 的層數或寬度 ——
  那會改變模型的實際數學行為，屬 `.claude/rules/code-changes.md` §3 禁止的副作用。
  兩者深度／參數量不同**本身就是要記錄的實作事實**，不是要抹平的缺陷。

## Explicit Non-Scope

- 不訓練任何模型
- 不改 `data/outputs/essgnn_arch_protocol.json` 的值（那是已記錄的決定，改它需 Kyzen 裁決）
- 不碰 `data/outputs/` 其他任何檔案
- 不處理 U-16 / tower sharing（B 支線）
- 不處理 n07 / n08 / node text（F 支線）
- 不改 `主線.md` / `支線任務.md` / `TASKS.md`
- 不改其他支線的 `TASK.md` / `HANDOFF.md` / `CODEX_REVIEW.md`

## Expected Deliverables

1. 上列 11 處修正
2. `任務/E_essgnn-axis/CODEX_REVIEW.md`
3. `任務/E_essgnn-axis/HANDOFF.md`

## Likely Files

`metafind/models/essgnn.py`、`metafind/models/resolve_stage2.py`、
`tests/test_essgnn.py`、`tests/test_resolve_stage2.py`、
`docs/audit/{C_PAPER_CONTRADICTIONS,E_GRAPH_REVALIDATION}.md`、
`docs/graph/{00_FINDINGS.md,README.md,02_BUILD_STEPS.md,graph_spec.yaml}`

## Required Verification

- [ ] `ESSGNNConfig` 現在**要求**顯式 `coord_feat`；缺少時報錯訊息清楚
- [ ] `sec25_two_mlp` × `{current, updated}` 兩種組合建構成功且**參數量相同**（回報實測數字）
- [ ] `appendix_shared_msg + updated` **仍然拒絕**，且拒絕訊息說明架構理由
- [ ] 等變測試對**三種**可建構組合都跑（不是四種）
- [ ] 全 repo `grep -rn "two different ESSGNNs"` 零命中
- [ ] `python tools/check_graph.py` all pass（基線 `2275 checks`）
- [ ] `python -m pytest tests/ -q` 全過（基線 442；移除硬斷言後數字可能變動，回報實際值）
- [ ] `git status --short` 沒有 `data/` 變更
- [ ] 數字矛盾（164,737/213,761 vs 446,212/412,932）已查清並說明

## Research Risks

- **最大風險：把清理文件變成改模型。** 本任務不得改變任何 forward pass 的數學。
  若某項修正需要改數學，**停下回報 Master**。
- `coord_feat` 是 `paper-conflicted`，**不是** `paper-backed`。兩個值都是論文寫的，論文自己打架。
  用詞錯了就是把矛盾偽裝成事實。
- `graph_spec.yaml` 的 U-26 目前是 `RESOLVED`。若翻案成立，是否改回 `UNKNOWN`
  **是研究決定，需 Kyzen 裁決**，本支線只提出建議，不逕自更改 `marked` 欄位。

## Implementation Risks

- `graph_spec.yaml` 是 `check_graph.py` 讀的機器可讀契約，編輯後必須重跑。
- 移除 `test_resolve_stage2.py:179` 硬斷言會改變測試總數，屬預期。

## Codex Review Requirement

**FULL，必須執行。** 要求 Codex 挑戰：
(a) 是否有任何修改改變了模型數學；(b) 用詞是否仍把 INFERENCE 當 PAPER FACT；
(c) Master 的前提修正本身是否正確；(d) 數字矛盾的解釋是否成立。

每項 finding 由 Claude 分類 `CONFIRMED` / `PLAUSIBLE` / `REJECTED` / `UNVERIFIED`，寫入 `CODEX_REVIEW.md`。

## Definition of Done

11 項修正完成、全部 Required Verification 有實測輸出、模型數學未變。
**不得宣告 U-26 已重新解決**（那是 Kyzen 的裁決）。

## Return-to-Master Requirements

標準 HANDOFF 格式，另附：逐處修改前後對照、四組合實測參數量、數字矛盾的解釋、
以及明確一行：**「本次未改變任何模型數學行為」**（或列出例外）。
