# 文件導覽與使用界線

2026-09-07 整理。優先從目前入口讀起；有日期的量測、審查與 handoff 記錄的是當時狀態，不應當成最新程式與執行狀態。

## 目前入口

| 文件 | 用途 |
| --- | --- |
| [專案 README](../README.md) | 安裝、程式與文件入口、研究限制。 |
| [DATA_FLOW.md](DATA_FLOW.md) | producer → artifact → consumer、兩階段訓練與評估、快取更新邊界。 |
| [CUSTOM_TABLE1_EVALUATION.md](CUSTOM_TABLE1_EVALUATION.md) | 已實作的自訂七模態評估：固定名單、觀測、權重與計分。 |
| [REPRODUCTION_REVIEW_20260907.md](REPRODUCTION_REVIEW_20260907.md) | 端到端 paper→code→validation 審查、已修缺陷、未閉合機制及正式實驗前置決策。 |
| [REPRODUCTION_CONTINUATION_20260907.md](REPRODUCTION_CONTINUATION_20260907.md) | 前輪審查：tau／semantic source／no-layout 修補、raw scene 到 Blender 交接。 |
| [REPRODUCTION_RUNTIME_REVIEW_20260907.md](REPRODUCTION_RUNTIME_REVIEW_20260907.md) | 前輪：n11b canonical text、gated 修補、程序觀測、full-layout 整合與真實 ULIP-2／Stage 1 CPU 執行證據。 |
| [REPRODUCTION_TRAINING_REVIEW_20260908.md](REPRODUCTION_TRAINING_REVIEW_20260908.md) | no-layout λ 修正、ESSGNN 獨立數值／梯度、Table 3 可達性、隔離真實 CPU 訓練與 checkpoint 還原。 |
| [REPRODUCTION_SCENE_REVIEW_20260908.md](REPRODUCTION_SCENE_REVIEW_20260908.md) | 最新：slot 數值修補、1,599-case CPU suite、真 gallery／Gemma／三 slot 檢索／GLB 渲染、獨立頂點核對及 Git 交付證據。 |
| [公式／梯度審查](audit/formula_review_20260907.md) | Eq.1–8／附錄對照、獨立 loss oracle、原文矛盾與既有選擇。 |
| [SCENE_COMPOSITION.md](SCENE_COMPOSITION.md) | Algorithm 1 tensor／manifest 入口、query model 匯出及真實場景產物的交接限制。 |
| [IDesign_INPUTS.md](IDesign_INPUTS.md)、[RAW_SCENE_INPUTS.md](RAW_SCENE_INPUTS.md) | 原始 planner JSON、明確模態查詢、checkpoint／gallery／語義身分到可重播 bundle。 |
| [SCENE_PLACEMENT.md](SCENE_PLACEMENT.md) | raw GLB 的真實頂點尺寸、instance 放置、`.blend` 儲存與明確 CPU render config。 |
| [SCENE_SEMANTICS.md](SCENE_SEMANTICS.md) | 只為缺少的 UID pairs 重用 n08 SG2 生成／修復／編碼，輸出可追溯的新 cache。 |
| [tests/README.md](../tests/README.md) | 測試分組、CPU／GPU／hook 指令與覆蓋界限。 |
| [tools/probes/README.md](../tools/probes/README.md) | 主評估入口、保留的診斷、已刪除且撤回的舊 probe。 |
| [CLEANUP_REPORT_20260907.md](CLEANUP_REPORT_20260907.md) | 本次整理、刪除依據、驗證與保留事項。 |

## 原文、規格與歷史資料

- **論文內容權威**：[paper/metafind_source/](paper/metafind_source/)。其餘 `paper/*_source/` 屬各自論文；上游僅在 MetaFind 明確繼承的範圍適用。不得改動原文來使實作通過。
- **衍生規格／稽核**：[graph/](graph/README.md)、[audit/](audit/A_FORMULA_INVENTORY.md)、[DECISION_LEDGER](../workflow/DECISION_LEDGER.md)。需追溯原文與決策日期，不將後續量測當成作者設定。
- **歷史工作狀態**：[PROGRESS](PROGRESS.md)、[MASTER](../workflow/MASTER.md)、[METAFIND_NOTEBOOK](METAFIND_NOTEBOOK.md)、`Session Handoff.md`。已加時效說明；目前程式、產物與使用者新指令優先。
- **歷史實驗**：`TABLE1_REPORT_20260904.md`、`TABLE1_REPORT_20260905_v2.md`、`TABLE1_REPORT_20260906_v3.md`。保留當時分数，不以新程式「校正」舊 JSON 或假稱重跑。
- **較早修正的驗證**：[CODE_REPAIR_REPORT_20260907.md](CODE_REPAIR_REPORT_20260907.md) 與 `audit/code_repair_20260907_checks.json`；自訂評估最初驗證為 `audit/custom_table1_20260907_checks.json`。這些 JSON 的舊測試路徑與 SHA 不改寫；搬移對照見本次整理紀錄。
- **已更正的評估推論**：[其他論文協定](NOTE_20260905_OTHER_PAPERS_RETRIEVAL_PROTOCOLS.md)、[Table 1 的量測意義](NOTE_20260905_TABLE1_WHAT_IT_MEASURES.md)。已撤回全領域全稱、從分數倒推作者 query、融合必定加分等說法。
- **計畫與閱讀筆記**：`PAPER_PIPELINE_FULL_20260905.md`、`DATASET_REBUILD_PLAN_20260906.md`、`NOTE_*`、`*_REVIEW_BRIEF_*` 記錄當時的計畫或審查。計畫寫了某一步，不代表目前已完成；舊 brief 中的程式行號／測試路徑須依當時 revision 理解。
- **規則快照**：`RULES_SNAPSHOT.md`、`_rules_preamble.md` 為衍生文件；目前適用规则見 [AGENTS.md](../AGENTS.md) 與使用者明確指示。本轮不修改 `CLAUDE.md`／`.claude/`。

## 引用舊測試的方式

測試於 2026-09-07 由 `tests/test_*.py` 搬入七個子目錄，檔名不變。現行文件連結與工具已更新；歷史 audit JSON、原始結果與決策紀錄保留舊位置，以免改掉當時驗證的身分。需要現在的路徑時執行：

```bash
rg --files tests | rg '/test_checkpoint_roundtrip\.py$'
```

本次沒有對所有歷史筆記重新做全文論文稽核。已確認的錯誤有明確更正；其餘歷史數字與論文主張，引用前仍需回到對應原文或原始 artifact。
