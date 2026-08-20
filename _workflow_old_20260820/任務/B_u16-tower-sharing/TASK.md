# TASK — B_u16-tower-sharing

> 這份 TASK.md 是本 branch 的 **execution contract**。
> 支線不得自行改寫 Objective / Scope / Explicit Non-Scope / Definition of Done。
> 需要修改請回報 Master。

## Branch Name

`B_u16-tower-sharing`

## Task ID

U-16 ／ T5.3（`TASKS.md`）／ M2.3（`主線.md`）／ S1.4（`支線任務.md`）

## Objective

把「架構圖畫**單一** Fusion Layer 且標 `ULIP-2 (Shared)`」與
「§2.6 gallery 凍結、query fuser 訓練」之間的矛盾，
依專案 C 系列格式**完整登記**，並產出 D-β 裁決所需的完整證據卷宗。

**只登記與陳列證據。不做選擇。**

## Why Now

- U-16 擋住 n09（`metafind/data/splits.py:130` 的 `tower_sharing` 會寫進 `stage1_protocol.json`，
  且 `splits.py:80` 的 `DEFAULT_TOWER_SHARING` **沒有 CLI 參數**可覆寫）。
- n09 擋住 M3 Stage 1。
- `TASKS.md` T5.3 自己寫「這是新的 C 系列矛盾候選，**尚未登記**」—— 工作已被指定但從未開始。
- A 支線的 4 小時 GPU 長跑期間正好完成這件不需 GPU 的事。

## Current Context

Master 於 2026-08-20 **親自開圖**驗證 `docs/paper/metafind_source/MetaFind.drawio.png`：

1. `ULIP-2 (Shared)` 標籤確實存在（OBSERVED，一手圖檔）
2. 圖上**只畫一個** `Fusion Layer`（OBSERVED）
3. **標籤錯置確認**：`Text Encoder → I1..IK`、`Image Encoder → T1..TK`，字母對調（OBSERVED）
4. 文件未載的新觀察：Query Encoder 與 Item Encoder 的箭頭**都指進同一個 `ULIP-2 (Shared)` 方塊**

> ⚠️ **第 3 點同時是反向證據。** 這張圖**已被證實含有轉錄錯誤**，
> 因此 `Shared` 這個標籤**不能單獨凌駕 §2.6 的正文**。這是本任務必須寫清楚的核心論點。

程式端現況：

- `metafind/models/dual_tower.py:315-321` 在 `fully_shared` 下**拒絕** `freeze_gallery()`，
  理由已寫在 docstring：兩塔若是同一模組，「gallery 凍結」與「query fuser 訓練」不可能同時成立。
- `metafind/data/splits.py:80` `DEFAULT_TOWER_SHARING = "shared_backbone_separate_fusion"`
- `docs/graph/graph_spec.yaml` 的 U-16 目前為 `marked: UNKNOWN, blocking: false`

## Authoritative Inputs

依 `CLAUDE.md` §3 權威順序：

1. `docs/paper/metafind_source/2methdology.tex` §2.4、§2.6 —— **逐句**，不是摘要
2. `docs/paper/metafind_source/neurips_2025.tex:88-91` Figure 1 caption 全文
3. `docs/paper/metafind_source/MetaFind.drawio.png` —— **必須自己開圖看，不得引用 Master 對話的描述**
4. `docs/paper/metafind_source/appendix.tex`
5. `docs/paper/ulip2_source/main.tex`（**Level 1**：只能說明 ULIP-2 怎麼定義，
   依 `docs/graph/README.md` 的規則，**不得自動補上 MetaFind 沒寫的部分**）
6. `metafind/models/dual_tower.py:278-330`、`metafind/data/splits.py:75-146`
7. `docs/graph/graph_spec.yaml` 的 U-16 條目
8. `docs/audit/C_PAPER_CONTRADICTIONS.md` 既有 C1–C8 格式

## Dependencies

**無。** 不依賴 A、C、D、E、F。可與 A 平行。

## Scope

1. 逐句掃 §2.4 與 §2.6，列出**每一句**與兩塔權重共享有關的原文，附 `檔名:行號`。
2. 自行開啟 Figure 1 本體並讀 caption 全文，記錄圖上實際畫了什麼。
3. 記錄圖已知的轉錄錯誤（I/T 字母對調），並**明確論述**：這使該圖不能單獨凌駕正文。
4. 依 `docs/audit/C_PAPER_CONTRADICTIONS.md` 既有格式，新增一條 C 系列條目
   （編號**接續現有最大號**，不得與既有條目衝突）。
5. 對三個候選值各列一組分析：

   | 候選 | 支持證據 | 反對證據 | 對 Stage 2 的後果 | 對 Table 1/3 可比性的後果 |
   |---|---|---|---|---|
   | `shared_backbone_separate_fusion` | | | | |
   | `fully_shared` | | | | |
   | `fully_separate` | | | | |

   其中 `fully_separate` 必須註明：`stage1.py:309-338` 的 `build_model()` 只建**一個** backbone，
   n10 目前**做不到** `fully_separate`（Master 已實讀確認）。
6. 更新 `docs/graph/graph_spec.yaml` 的 U-16 條目，**`marked` 維持 `UNKNOWN`**，
   只補充新證據指標與新 C 條目的交叉引用。

## Explicit Non-Scope

- **不決定 U-16 選哪個值。** 裁決權在 Kyzen（D-β）。
- 不改 `metafind/data/splits.py` 的 `DEFAULT_TOWER_SHARING`
- 不改 `metafind/models/dual_tower.py`
- 不改 `metafind/train/stage1.py`
- 不跑 n09、不跑任何訓練
- **不碰 `data/outputs/`** —— A 支線正在寫那裡
- 不處理 τ（D-α）、不處理 3 筆殘留（D-γ）
- **不處理 ESSGNN / U-26 / C1 / S2** —— 那是 E 支線的範圍，
  且 E 與本支線共用 `C_PAPER_CONTRADICTIONS.md`
- 不改 `主線.md` / `支線任務.md` / `TASKS.md`
- 不改其他支線的 `TASK.md` / `HANDOFF.md` / `CODEX_REVIEW.md`
- 不宣告 U-16 已解決、不宣告 M2.3 完成、不推進到 n09

## Expected Deliverables

1. `docs/audit/C_PAPER_CONTRADICTIONS.md` 新增一條 C 系列矛盾條目
2. `docs/graph/graph_spec.yaml` 的 U-16 條目補充證據指標，`marked` 仍為 `UNKNOWN`
3. 三候選值後果對照表（寫在 C 條目內）
4. `任務/B_u16-tower-sharing/CODEX_REVIEW.md`
5. `任務/B_u16-tower-sharing/HANDOFF.md`

## Likely Files

- `docs/audit/C_PAPER_CONTRADICTIONS.md`（新增）
- `docs/graph/graph_spec.yaml`（U-16 條目補充）
- `任務/B_u16-tower-sharing/`（HANDOFF、CODEX_REVIEW）

**不應有任何 `metafind/`、`tests/`、`data/` 變更。若出現，立刻停下回報 Master。**

## Required Verification

- [ ] 每一條原文引用都附 `檔名:行號`，且引文與檔案**逐字相符**（抽驗至少 3 條）
- [ ] Figure 1 的描述來自**分支自己開的圖**，不是引用 Master 對話
- [ ] `python tools/check_graph.py` 仍 all pass（基線 `2275 checks`）
- [ ] `python -m pytest tests/ -q` 仍 `442 passed`
- [ ] `git status --short` **只有 `docs/` 與 `任務/` 底下的變更**
- [ ] 新 C 條目編號未與現有條目衝突
- [ ] 全文複查一次用詞：沒有把 `INFERENCE` 寫成 `PAPER FACT`
- [ ] `graph_spec.yaml` 的 U-16 `marked` 欄位仍為 `UNKNOWN`

## Research Risks

- **最大風險：把圖當成比正文更高的權威。** 圖已證實含轉錄錯誤（I/T 對調），
  `Shared` 標籤是 OBSERVED，不是自動的 PAPER FACT。
- **第二風險：用 ULIP-2 補 MetaFind 的沉默。** `docs/graph/README.md` 明訂
  Level 1 不得自動補上 MetaFind 沒寫的部分。ULIP-2 的雙塔設計不等於 MetaFind 的雙塔設計。
- **第三風險：偷偷做決定。** 「證據強烈支持 X」這種措辭實質上就是決定。
  本任務只能陳列，不能傾向。

## Implementation Risks

- 低。純文件任務，無執行、無 GPU、無資料變更。
- `graph_spec.yaml` 是 `check_graph.py` 讀取的機器可讀契約，
  編輯後**必須**重跑 `check_graph.py`，格式錯誤會讓 2,275 檢查失敗。

## Codex Review Requirement

**FULL，必須執行。** 這是純研究判讀任務，正是 `codex-reviewer` 的適用場景。

要求 Codex 以獨立審查者身分挑戰：

- (a) 引文是否被斷章取義
- (b) 是否有把圖的權威抬過正文
- (c) 三候選的後果分析是否有遺漏
- (d) **是否偷偷做了決定**

每一項 Codex finding 必須由 Claude 分類為
`CONFIRMED` / `PLAUSIBLE` / `REJECTED` / `UNVERIFIED`，附分類理由，寫入 `CODEX_REVIEW.md`。

**Codex 不是科學權威。其結論不得直接寫進 C 條目**，只能作為促使重新查證原始文獻的觸發。

## Definition of Done

上列 Required Verification 全部逐條有實測輸出，**且 U-16 仍為 `UNKNOWN`**。

**Branch 不得宣告 U-16 已解決、不得宣告 M2.3 完成、不得推進到 n09。**

## Return-to-Master Requirements

寫入 `任務/B_u16-tower-sharing/HANDOFF.md`，至少包含：

1. Task ID / Status
2. Objective Result
3. Files Changed（`git diff --stat`）
4. Evidence Used（§2.4 / §2.6 逐句引文清單，含行號）
5. **Decisions Made —— 應為「無」。** 若非如此必須逐條列出並說明為何不可避免
6. Verification Performed / Verification Result
7. Codex Review Result（摘要，細節在 `CODEX_REVIEW.md`）
8. Confirmed / Plausible / Rejected / Unverified Findings
9. **Master-Impacting Findings** —— 明確一行：「是否發現任何會改變 master assumption 的證據」，
   有則列出並標 evidence class
10. Remaining Risks / Blocked Items
11. Recommended Master Update
12. Recommended Next Action

另需附：新增的 C 條目全文、Figure 1 的自主觀察紀錄、三候選後果對照表、
以及明確一行：**「U-16 仍為 UNKNOWN，未做決定」**。
