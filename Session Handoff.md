# Session Handoff

> Current handoff only. Replace this file completely when a new handoff is generated.

**Generated:** 2026-08-20 12:50
**Work state:** Continuing
**Source:** Previous Claude Code session

### Authority

- This document records working state, not source-of-truth evidence.
- Project authority documents and current repository state override this handoff.
- Anything marked `(unverified)` must be verified before being treated as fact.

## Handoff: n05 標註重寫完成 → n06 重編碼 → Stage 1

### Objective

n05 已用論文 Figure 2 的 13 欄位 schema 全量重跑完成。下一步是丟棄舊的 n06 快取全量重編碼，然後解除 Stage 1 的三個前置阻斷。

主線與支線的逐項工作分別在 `主線.md` 與 `支線任務.md`，完整項目清單在 `TASKS.md`。

### Decisions made

- n05 schema 照 MetaFind Figure 2（13 欄位），不照 Holodeck 最新版。理由：現行 objathor 用 `depth`／公升，Figure 2 用 `length`／cm³，版本已漂移；論文是復現對象。
- `volume` 由 `w×l×h` 計算，不問模型。理由：Figure 2 的 36000 = 30×30×40，是前三者的乘積而非獨立觀測。
- cm/kg 標為 `UPSTREAM-SUPPORTED INFERENCE`，不是 PAPER FACT。理由：Figure 2 未標單位；Holodeck A.6 明文 cm/kg，且 36000 只有 cm³ 自洽。
- `synset`／`volume`／`mass` 留在標註檔但不進 CLIP 文字。理由：`synset` 是識別碼非語言、`volume` 與三邊重複、`mass` 無視覺依據。待 T4.4 最終確認。
- 不加 n08 no-relation 過濾。理由：論文未提過濾，且會改 `edge_index` 拓樸。
- 不動 n04 的單位球正規化。理由：Objaverse 單位跨 1.3e5 倍，不正規化無法渲染。

### Active facts

- n05 v3 完成：45,952 筆，隔離 3 筆（0.007%）。Source: `logs/n05_v3_full.log`、實測計數。
- 隔離 3 筆原因：`pom.pom.n.01` synset 格式錯 ×2、`mass=50000kg` 密度檢查 ×1。皆非誤殺。
- v3 速度 39.0/min（v1 為 29.0/min）。
- n06 現有快取僅 5,276 筆且文字模板已變，必須全量重編碼（約 4 小時）。
- `n09_build_splits` 從未執行：`splits.json`、`eval_protocols.json`、`stage1_protocol.json` 皆不存在，`stage1.py:356` 會直接 return 2。
- τ = 0.5 是 PAPER FACT（`3experiments.tex:15`），已入 `metafind.models.losses.PAPER_TAU`，偏離時 warn。但 `stage1_hyperparameters.json` 仍為 `0.07/learnable`。
- 論文圖檔已全數解壓（4 篇共 39 張）。`MetaFind.drawio.png` 標示 `ULIP-2 (Shared)` 且只畫一個 Fusion Layer，影響 U-16。
- GPU 為 RTX 5090 32GB，非 repo 多處所寫的 4090 24GB。所有以 24GB 為前提的可行性判斷須重測。
- 測試 442 條全過、`tools/check_graph.py` 2,275 項全過。
- Python: `/home/kyzen/miniconda3/envs/MetaFind/bin/python`
- 進度查詢：`bash tools/n05_progress.sh`、`bash tools/status.sh`

### Open problems

- `onFloor`／`onObject` 準確率天花板：v3 對 AI2-THOR 真值 72.5%／71.0%（多數類基準 58.0%／60.5%），僅比 v1 好 2.5／1.5 個百分點。已排除：兩輪 prompt 調整均只有邊際改善。根因為 n04 正規化使圖上無絕對尺度，非可修 bug，應寫入限制章節。
- 架構圖的單一 Fusion Layer 與 §2.6「gallery 凍結、query fuser 訓練」可能互斥，尚未登記為 C 系列矛盾。
- `Stage1RuntimeConfig` 宣稱是唯一建構路徑，但 `stage1.py:309-338` 不使用它；34 條 `test_dual_tower.py` 測的是訓練器繞過的 class。

### Files changed

- `metafind/data/annotate.py`: schema 改 13 欄位、prompt v3、密度交叉檢查、`PLACEMENT_FLAGS`／`MATERIAL_SYNONYMS`。
- `metafind/data/annotate_run.py`: 完成度檢查加 `prompt_version`；新增 `--uids-file`。
- `metafind/models/resolve_stage1.py`: `TEXT_TEMPLATE` 改公分、placement 四布林轉散文、新增 `placement_phrase()`。
- `metafind/models/losses.py`: `PAPER_TAU = 0.5`、偏離時 warn。
- `metafind/paths.py`: 新增 `_shell_exports()` 與 `__main__`，供 shell `eval "$(python -m metafind.paths)"`。
- `metafind/train/stage2.py`: `load_stage2_protocols()` 實際試建 ESSGNNConfig，`resolved` 不再是無檢查的宣稱。
- `setup/01_storage.sh`: 全重寫。原本建的目錄結構與程式實際用的不同，且會 sudo chown 他人磁碟。
- `setup/02_conda_env.sh`、`setup/04_idesign_env.sh`、`tools/status.sh`、`tools/chain_to_stage1.sh`、`tools/chain_after_n05.sh`: 改由 `paths.py` 供應路徑。
- `tools/build_source_manifest.py`: 新增 `figures()`，manifest 含 `referenced_figures`／`figure_sha256`／`images_in_archive`／`missing_figures`。
- `tools/idesign_generate.py`、`tools/probes/thor_isolated_render_probe.py`: 改用 `paths`。
- `tools/n05_progress.sh`: 新增（未追蹤）。
- `TASKS.md`、`essgnn.md`、`主線.md`、`支線任務.md`: 新增（未追蹤）。
- `docs/paper/*_source/`: 39 張圖已解壓（未追蹤）。
- `docs/audit/C_PAPER_CONTRADICTIONS.md`: S4 加 `[CORRECTED]`，τ 不再列為論文沉默。

### Boundaries

- 不重跑 n05。已完成且驗收過。`annotations_v1_prompt1/`（45,953 筆）與 `annotations_v2_sample/`（200 筆）為備份，不得刪除。
- 不加 n08 relation 過濾。論文未提，且會改圖拓樸。
- 不動 n04 正規化。
- `f_x` 維持純量。§2.5 字面的 `R³` 會破壞論文自身的等變證明（實測誤差 0.43 vs 2.2e-16）。
- 不得將 cm/kg 寫成 MetaFind 明文規定。它是 `UPSTREAM-SUPPORTED INFERENCE`。
- 不得將 ESSGNN 的 `architecture_family` 分類寫成論文定義。外部 adjudication 判定 `UNSUPPORTED`。

### Next step

清空 `data/outputs/embeddings/` 後全量重跑 n06：
`/home/kyzen/miniconda3/envs/MetaFind/bin/python -m metafind.data.encode_text_image`
