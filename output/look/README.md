# output/look — 給 Kyzen 看的圖與數字（2026-09-05 整理）

## 報告
- `MetaFind_report_20260904.pptx` / `.pdf`：09-04 進度簡報（Stage 1、Table 1、Stage 2）。
- `pptx/`：簡報的 PptxGenJS 原始碼（`slides/compile.js`、`slides/imgs/`）。`npm install` 後 `node compile.js` 重建。
- `figures/`：簡報用的六張圖 + 聯絡表（`report_fig1_stage1_arms_D.png` … `report_fig6_ulip2_zero_shot.png`、`report_contact_sheet.png`）。
- `ulip2_pull_explainer.png`：ULIP-2 對齊示意。

## 表
- `ARMS_TABLE.md`：所有 Stage 1 arm 的七格對照（含論文正確列 13.8/11.7/75.1/17.2/44.5/45.8/51.7）。
- `STEP3_TABLE.txt`、`RETRACTED_20260903.md`：09-03 的表與撤回紀錄。

## 探針輸出（ledger／docs 有引用，留在原位）
- 觀測構造：`exp_query_observation.json`、`exp_query_pc_observation.json`、`exp_observation_geometry.json`、`exp_observation_matrix.json`、`exp_tower_agreement.json`。
- 文字：`exp_text_fill_ladder.json`、`exp_text_length.json`、`exp_text_observation.json`、`exp_text_template.json`。
- 類別層級 query：`exp_type_level_query*.json`、`exp_ulip_row_category_query.json`、`exp_ulip_scorer_margin.json`、`exp_ulip_table1_D.json`。
- 釋出 ULIP-2 本身：`exp_ulip2_zero_shot_lvis.json`（LVIS zero-shot 50.9/79.3）、`pairwise_crossmodal_diag_val.json`（九格配對）、`ulip2_geometry_1024.json`。
- Stage 2：`exp_stage2_procthor_retrieval*.json`（S2C／S2D arm）。
- `pilot10.json`：10-epoch pilot 紀錄。

正式 Table 1 的數字不在這裡，在 `data/outputs/eval/table1_*/table1.json` 與 `docs/TABLE1_REPORT_20260904.md`。
歸檔清單見 `ARCHIVED.md`（歸檔目錄本身已於 2026-09-05 依 Kyzen 指示刪除）。
