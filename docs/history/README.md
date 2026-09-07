# 歷史文件與搬移對照

這些文件保留研究判讀、量測、修補與決策追溯，**不代表目前 pipeline 或 corpus 狀態**。操作從 [docs 入口](../README.md) 開始；最新結果見 [本輪審查](../audit/REPRODUCTION_CORPUS_REVIEW_20260908.md)。

2026-09-08，以下 26 份文件由 `docs/<檔名>` 移到 `docs/history/<檔名>`。日期、分數與原始論述保留；一般文件只調整 Markdown 相對連結。`RULES_SNAPSHOT.md` 與 `_rules_preamble.md` 完整 bytes 不變，舊內文路徑依生成時位置解讀。後者仍是 `tools/dump_rules.py` 的來源；工具已改從此目錄讀寫，未重新生成或改動 Claude 規則。

| 現在位置（舊位置皆為 docs/ 根層同檔名） | 原文件標題 |
|---|---|
| [TABLE1_REPORT_20260904.md](TABLE1_REPORT_20260904.md) | Table 1 報告（2026-09-04）—— 主線 P1s，唯一一次 `--unseal` |
| [NOTE_20260905_METAFIND_PAPER_FULL_READ.md](NOTE_20260905_METAFIND_PAPER_FULL_READ.md) | MetaFind 論文逐字讀完 —— 筆記（2026-09-05 05:5x） |
| [REPRODUCTION_SCENE_REVIEW_20260908.md](REPRODUCTION_SCENE_REVIEW_20260908.md) | 真實場景執行與整理交付（2026-09-08） |
| [REPRODUCTION_CONTINUATION_20260907.md](REPRODUCTION_CONTINUATION_20260907.md) | 接續完整檢查與復現修補（2026-09-07） |
| [NOTE_20260904_ULIP2_CHECK_AND_STAGE_SUMMARY.md](NOTE_20260904_ULIP2_CHECK_AND_STAGE_SUMMARY.md) | 筆記 2026-09-04：ULIP-2 有沒有被動、Stage 1 一輪總結、Stage 2 初步結果 |
| [ESSGNN_DIM_REVIEW.md](ESSGNN_DIM_REVIEW.md) | ESSGNN 維度重建 — 審查回覆 v3 |
| [STAGE1_ARCHITECTURE_EQUATIONS_20260905.md](STAGE1_ARCHITECTURE_EQUATIONS_20260905.md) | P1s 系列 Stage 1 —— 架構與訓練配方寫成公式，每塊解釋（v2，2026-09-05） |
| [_rules_preamble.md](_rules_preamble.md) | MetaFind Reproduction — Research Rigor & Decision Authority Rules |
| [METAFIND_NOTEBOOK.md](METAFIND_NOTEBOOK.md) | MetaFind 復現筆記（Master 說明書） |
| [CODEX_REVIEW_BRIEF_20260828.md](CODEX_REVIEW_BRIEF_20260828.md) | Codex review brief -- 2026-08-28, n08 model/encoder change + two chain fixes |
| [PROGRESS.md](PROGRESS.md) | MetaFind 復現 · 進度與排程 |
| [REVIEW_BRIEF_20260902.md](REVIEW_BRIEF_20260902.md) | Review brief -- 2026-09-02, the eleven-view renderer, the ESSGNN recipe, the Stage 2 lambda |
| [TABLE1_REPORT_20260905_v2.md](TABLE1_REPORT_20260905_v2.md) | Table 1 報告 v2（2026-09-05 07:1x）—— 兩顆 backbone、兩種 query 構造、掃描、Stage 2 |
| [NOTE_20260904_TEXT2SHAPE_READ.md](NOTE_20260904_TEXT2SHAPE_READ.md) | Text2Shape 讀後筆記 —— 論文全文＋官方程式碼（2026-09-04） |
| [CLEANUP_REPORT_20260907.md](CLEANUP_REPORT_20260907.md) | 2026-09-07 測試、評估程式與文件整理紀錄 |
| [NOTE_20260905_OTHER_PAPERS_RETRIEVAL_PROTOCOLS.md](NOTE_20260905_OTHER_PAPERS_RETRIEVAL_PROTOCOLS.md) | 其他論文的檢索協定：2026-09-07 更正 |
| [REPRODUCTION_TRAINING_REVIEW_20260908.md](REPRODUCTION_TRAINING_REVIEW_20260908.md) | 真實訓練與公式接續審查（2026-09-08） |
| [NOTE_20260904_ULIP2_FULL_READ.md](NOTE_20260904_ULIP2_FULL_READ.md) | ULIP-2 逐字讀完：論文（arXiv 2305.08275 v4）＋ 官方程式碼 —— 筆記（2026-09-04） |
| [TABLE1_REPORT_20260906_v3.md](TABLE1_REPORT_20260906_v3.md) | Table 1 報告 v3（2026-09-06）—— 兩列都做：Stage 1 頭（w/o ESSGNN）與 Stage 2 共用頭（w/ ESSGNN） |
| [DATASET_REBUILD_PLAN_20260906.md](DATASET_REBUILD_PLAN_20260906.md) | 資料集重建計畫（2026-09-06）—— 照論文重做，只有 LLM 例外 |
| [REPRODUCTION_REVIEW_20260907.md](REPRODUCTION_REVIEW_20260907.md) | 2026-09-07 端到端復現審查與修正 |
| [REPRODUCTION_RUNTIME_REVIEW_20260907.md](REPRODUCTION_RUNTIME_REVIEW_20260907.md) | 正式執行路徑接續審查（2026-09-07） |
| [PAPER_PIPELINE_FULL_20260905.md](PAPER_PIPELINE_FULL_20260905.md) | MetaFind 論文的完整流程（訓練 + 評估，含 Stage 2）—— 逐步列表 |
| [RULES_SNAPSHOT.md](RULES_SNAPSHOT.md) | MetaFindV1 規則快照 |
| [NOTE_20260905_TABLE1_WHAT_IT_MEASURES.md](NOTE_20260905_TABLE1_WHAT_IT_MEASURES.md) | Table 1 到底在測什麼；誰用 ULIP-2 做過檢索、怎麼測（2026-09-05 晚） |
| [CODE_REPAIR_REPORT_20260907.md](CODE_REPAIR_REPORT_20260907.md) | 2026-09-07 程式修正與 Claude 交接 |

移除的重複文件：`docs/NOTE_20260904_ULIP2_TRAINING_READ.md`。它和 [ULIP-2 完整讀本](NOTE_20260904_ULIP2_FULL_READ.md) 都明確指定後者已涵蓋、取代摘要；舊 bytes 仍可由 Git `de6635b` 查回。沒有刪除 ULIP-2 paper source。

本輪暫存的 `docs/G3_MANUAL_EXCLUSION_PROPOSAL.md` 已由使用者採用並整併至 [DL-106](../../workflow/DECISION_LEDGER.md)，不再另留一份「待決定」操作文件。原提案與搬移前後 SHA 保存在本輪整理證據中。

歷史 audit JSON、frozen artifacts、`workflow/DECISION_LEDGER.md`、`Session Handoff.md` 內的舊路徑保留原文。遇到 `docs/某檔.md` 時先用本表定位，不能因路徑搬移就把舊 SHA 或當時執行狀態改成今天的值。
