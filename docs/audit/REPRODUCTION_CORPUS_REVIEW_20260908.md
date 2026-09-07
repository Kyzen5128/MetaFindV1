# 2026-09-08 資料邊界、G3 計帳、評分交接與整理紀錄

本輪補上 n09 的 manifest 邊界、修復等待鏈的人工排除，依已批准 DL-106 實作 G3 部分 preflight，並增加外部場景評分匯入／彙總。`docs/` 根層由 37 份 Markdown 收斂為 9 份操作指南，歷史資料移到 `docs/history/`；`data/outputs` 刪除 2 個空目錄。

這些是程式與驗證進展，**完整 Table 1–3 尚未復現**。原標註程序未重啟；正式 paper corpus 尚未完成，G3 未接入 live chain，也沒有執行場景裁判或人評。

## 1. 範圍與證據層級

- **OBSERVED IMPLEMENTATION／DATA**：本文的程式行為、集合數、程序與測試，對應下列原始 receipts／logs；驗證時間點與 source SHA 均有保留。
- **IMPLEMENTATION CHOICE**：DL-106 獨立 `manual_excluded` 計帳由 Kyzen 明確批准；自訂 Table 1 的 query 定義是本專案可重播比較，不宣稱作者的未公開協定。
- **UNKNOWN**：作者 Table 1 具體 query 觀測與候選 UID 清單、完整 baseline 融合細節、未指定的正式 judge prompt／camera policy 仍未補定。
- 論文內容權威仍是 [MetaFind source](../paper/metafind_source/)。先前公式、真模型訓練與場景結果見 [公式審查](formula_review_20260907.md)、[訓練驗證](../history/REPRODUCTION_TRAINING_REVIEW_20260908.md)、[場景驗證](../history/REPRODUCTION_SCENE_REVIEW_20260908.md)。本輪沒有把 CPU 通過解讀成論文忠實度或正式成績認證。

## 2. 找到並修正的既有問題

| 問題／實際路徑 | 修改與原因 | 驗證／界線 |
|---|---|---|
| n09 的 index 交集可能包含 manifest 外 UID，與 filter ladder 的 manifest 分母不一致。 | [splits.py](../../metafind/data/splits.py) 在套 ledger 後明確拒絕殘留的 manifest 外 UID，不悄悄刪掉。 | 新增不同來源 index／ledger 狀態的回歸。真實舊 corpus 45,692 筆及 paper 當時 index 快照均未發現 rogue UID；證明邊界缺陷，不證明它造成原來的高分。 |
| 等待中的 Stage 1 腳本把 manual entries 的整個 dict 放入檔名，找不到對應 annotation。 | [chain_paper_stage1.sh](../../tools/chain_paper_stage1.sh) 重用 `ledger_excluded_uids`，只解析 manual group 的真正 UID；不帶入歷史 311 筆失敗。 | 原 ledger 確實有 21 筆 objects。舊行為的失效與修正後真 bytes 副本移動分開留存。 |
| 只搬開 annotation，沒有把排除決策留給 n09；恢復 sidecar／index 時可能失去第二層排除。 | 在任何移動前發布僅含 E 的 ledger；保留原 SHA、決策來源與 DL-106，拒絕衝突／alias／殘留 .part；同值重跑不改 bytes、inode、mtime。 | 真實來源副本驗證：21 筆 ledger、當時 9 份可用 annotation 副本移動成功，重跑 0 筆，真實輸入未改。另有中斷後 E 存在、重新納入 sidecar 仍被 n09 排除的回歸。 |

原 source ledger SHA：`39eff095ea6283601d57db7618d39ba2355db694e16b711ca82d5ddc59ef3058`。兩次副本驗證可用 annotation 由 8 增至 9，反映標註仍在進行，並非要求固定有 8／9 筆。paper snapshot 中 `admitted=8` 來自當時未重建的 index，**不能當作標註完成數或最終 corpus 大小**。

證據：[n09 快照](corpus_20260908_artifacts/n09_real_snapshot.json)、[原錯誤診斷](corpus_20260908_artifacts/manual_exclusion_before.json)、[第一次副本修正](corpus_20260908_artifacts/manual_exclusion_after.json)、[最終 E ledger 副本驗證](corpus_20260908_artifacts/manual_e_ledger_real_copy.json)。

## 3. G3：批准的三類計帳與真實 corpus 檢查

[DL-106](../../workflow/DECISION_LEDGER.md) 保留原始 LVIS manifest M，把成功 admitted A、真實處理失敗 Q、人工排除 E 分開：

```text
A、Q、E 兩兩不重疊
A ∪ Q ∪ E = M
Q = 真實 object quarantine UID − A − E
2% 門檻只檢查 |Q| / |M|
另報 |E| / |M| 與 (|Q| + |E|) / |M|
```

2% 是既定專案門檻。既有成功重試、已批准的 DL-026 歷史 runtime guard、n07b/render 場景事件及 systemic 事件分別記錄，不當作新增 object failures，也不以缺 artifact 推造 failure UID。

[新 preflight](../../metafind/gates/g3_object_corpus.py) 讀 split／manifest、quarantine、人工 ledger 及已解析 encoding／training／eval protocols；核對 exact sets、leakage、計數、超參數身分與現有門檻。缺必要證據回 `BLOCKED`／3，確定不合契約回 `FAIL`／2；通過其已實作範圍回 `PASS`／0。規格的 `implemented: false` 保持原狀，未接入 live chain。

| 真實資料根目錄 | preflight 結果 | 本次證據 |
|---|---|---|
| `metafind_data` 舊穩定 corpus | **PASS／0**：46,052 = 45,692 + 339 + 21；Q/M ≈ 0.7361%，E/M ≈ 0.0456%，合計 ≈ 0.7817%。無 missing／unexpected UID，三集合分離。 | [gate record](corpus_20260908_artifacts/g3_final/historical/G3_object_corpus.yaml)、[execution](corpus_20260908_artifacts/g3_final/historical/execution.json) |
| `metafind_data_paper` 在製 corpus | **BLOCKED／3**：尚缺 splits、明示 E ledger、Stage 1 protocols。未把仍在進行的資料當作最終失敗率。 | [gate record](corpus_20260908_artifacts/g3_final/paper_in_progress/G3_object_corpus.yaml)、[execution](corpus_20260908_artifacts/g3_final/paper_in_progress/execution.json) |

兩次 CLI 讀取的真實輸入均在前後比對 bytes；輸出寫到獨立診斷目錄。舊 corpus 的 PASS 只支持上述 preflight，**不證明 embedding bytes、實際 optimizer 執行或作者 Table 1 協定已驗證**。在 paper 主線應等 n09 與相關 protocols 齊全後另跑，不能在 n09 前宣稱完整對帳。

## 4. 本輪新 G3 在審查中另外修掉的問題

以下是本輪新增程式的缺陷及修正，與原 repo 的問題分開計帳：

1. 自訂 `--record run_progress.jsonl` 曾與進度輸出撞名：先回 PASS，後續 progress append 卻破壞 YAML。現在拒絕 record／history／.part／progress 與輸入間的 path、symlink、hardlink alias，新增 CLI 反例。
2. 原 parser 假定 quarantine timestamp 是字串；實際 runlog writer 使用有限浮點值。以真 writer 產出紀錄驗證支援。
3. n07b 的檔案 owner 是 n07b、事件 `stage` 可為 render；依 owner 與 row 共同核對 scope，避免錯算物件 corpus。
4. n04 的 `__run/SystemicFailure/RESOURCE` 與 `__batch_<digits>/BrokenProcessPool` 是系統事件，現在另列，避免當作 manifest 外 asset UID。

保留 [較早錯誤診斷](corpus_20260908_artifacts/g3_initial/g3_diagnostics.json) 及 [CLI 撞名修正前後測試](corpus_20260908_artifacts/g3_regression/)；較早 BLOCKED／測試 pass 並沒有被改寫成最終結果。新版真實 corpus 結果以 `g3_final/` 為準。

## 5. 場景評分：完成匯入與彙總接縫

[scene_scores.py](../../metafind/eval/scene_scores.py) 與 [操作規格](../SCENE_SCORES.md) 接受外部凍結的 scene×method manifest、Gemma judge protocol、原始 response 及四維分數。核對 composition、placement、render、camera、blend、mesh／annotation 與 response hashes；分數限 1–5 的有限數值。

完整場景、未完成場景、評分失敗與缺分各有分母。不將缺分填零；只有完成場景全數有合法評分才給 `mean_over_complete`，否則 `INSUFFICIENT_EVIDENCE`。已評部分的條件平均另列，不能代替完整平均；Human 固定為證據不足。

使用先前真實 3-slot 場景產物執行新 CLI 的 import→aggregate：**1 場景、完成 1、評分 0、缺分 1**，所有完整均分與 Human 均為證據不足。這驗證真 composition／GLB／Blender／PNG 交接與缺分處理，**沒有執行裁判、沒有產生 Table 2 分數**。程式也不驗證 response 到人工填入分數的語義抄錄是否正確。

證據：[真產物交接驗證](corpus_20260908_artifacts/scene_score_handoff/verification.json)、[summary](corpus_20260908_artifacts/scene_score_handoff/summary.json)。正式 scene manifest、judge prompt／generation／view policy 仍需明示研究協定，沒有用 diagnostic fixture 代替正式設定。

## 6. 文件整理

[新 docs 入口](../README.md) 保留 README、DATA_FLOW、CUSTOM_TABLE1_EVALUATION、IDesign_INPUTS、RAW_SCENE_INPUTS、SCENE_COMPOSITION、SCENE_PLACEMENT、SCENE_SEMANTICS、SCENE_SCORES 共 9 份。

- 26 份舊報告、閱讀筆記、計畫與規則快照集中在 [history](../history/README.md)，提供完整搬移對照；調整現行與一般歷史 Markdown 的相對連結。
- 刪除已由 `NOTE_20260904_ULIP2_FULL_READ.md` 完整取代的短版 `NOTE_20260904_ULIP2_TRAINING_READ.md`。兩份原文均明示覆蓋關係，短版沒有 runtime consumer；原文仍可從上一個 commit 取得。
- 刪除尚未入 Git 的 G3 待決提案，批准結果留在 DL-106，原提案 bytes 留作本輪證據。
- graph README 移除與現行執行證據矛盾的舊逐節點狀態表，改指向資料流及有日期的原始紀錄；Gate source-marker 計數區分 G3 部分實作與完整狀態。修正建置步驟仍寫 Qwen judge 的舊說法，沿用既有 DL-077 §13 Gemma 決策。
- `tools/dump_rules.py` 的輸出與 preamble 路徑同步搬到 history。只做 AST／路徑驗證，未執行 generator 或改寫規則快照。
- 歷史 audit JSON、Decision Ledger 與兩個既有 frozen archives 的原始路徑／SHA 不改寫；搬移對照供追查。paper source、vendor、CLAUDE.md、.claude 保持原樣。

[原始整理 receipt](corpus_20260908_artifacts/docs_cleanup.json) 記錄每項搬移、刪除及連結更正；本次 report 放在 audit，避免再增加 docs 根層的歷史報告。

## 7. data/outputs 清理

`data/outputs` 實際指向 `/home/kyzen/metafind/metafind_data/outputs`。只刪除兩個確認為空的 `_render_probe/`、`ladder/`；**沒有宣稱釋放 GB**。

其餘約 15 GB 直接檔案中，`_probe` 約 13 GB 包含仍被 manifest 綁定的 query arrays、重播／resume 與歸因證據；embeddings、pointclouds、renders、annotations、checkpoints 的 symlink 指向共用資料，不能當作重複檔刪掉。`query_pack.json.bak_20260906_oldpaths` 仍是既有 relocation 決策保留的來源證據。

四個 0-byte run.lock 雖通過非阻塞 flock 檢查，但 `/proc` FD 可見性不足，無法完整排除他人持有舊 inode，且刪除不回收檔案內容空間，因此保留。沒有擴張到不可確定安全的 populated artifacts。

[清理 receipt](corpus_20260908_artifacts/data_cleanup.json) 保存前後程序身分、symlink 及 source ledger SHA；已核對保持一致。

## 8. 等待鏈部署與標註保護

R2 尚未 DONE，Stage 1 shell 當時只有 `sleep 60` 子程序。每次部署先驗 PID/start ticks、cmdline、cwd、fd255 inode、原 script SHA、R2 marker 與 owned child，備份後僅停止並替換這個等待 shell。新 shell 使用原 env／cwd／append log，再核對仍等待。

最終 Stage 1 PID **2685619**，script SHA `d18f7accff486a68cdc537ca0d35fa01545b2c8af30b0ab54387843fbda36fb6`。annotation PID **2043427** 與 Stage 2 等待 PID **2494552** 的 start ticks／argv 均保持原樣。真實 E ledger 與 annotation 移動仍排在 R2 完成之後，本輪沒有提前執行。

先 schema 修正、再加入已批准的 durable E ledger，兩次部署各保留獨立 receipt／原 script，不覆蓋第一次證據：[第一次部署](corpus_20260908_artifacts/stage1_deployment.json)、[最終 E ledger 部署](corpus_20260908_artifacts/stage1_e_ledger_deployment.json)。程序狀態是部署時點事實，未來應重跑 `tools/status.sh`。

## 9. 驗證與後續

最後完整 CPU suite 為 **1,805 passed、71 warnings，exit 0**（57.57 秒）；265 個 source 前後 SHA 相同，沒有失敗或跳過。此命令排除 GPU 與 hooks，不能說兩組也已驗過。Graph checker 為 **2,342 checks 全通過，exit 0**，其輸入快照穩定；66 個測試檔／1,182 個頂層測試函式與參數化後的 1,805 cases 是不同計數。見 [graph record](reproduction_corpus_20260908_delivery_push_graph.json) 與 [graph log](reproduction_corpus_20260908_delivery_push_graph.log)。最終 CPU 結果見 [delivery CPU log](reproduction_corpus_20260908_delivery_cpu.log)、[execution](reproduction_corpus_20260908_delivery_cpu_execution.json)。較早的 `cpu`／`final_cpu` logs 保留為當時版本的測試，不能替代本輪最後程式的驗證。修完搬移與相對行號連結後，另跑 `delivery_push_graph`；較早 `delivery_graph`／`delivery_links_graph` 仍保留原快照。最終 [交付核對](reproduction_corpus_20260908_delivery_checks.json) 檢查來源與 log SHA、封存完整性、可點文件的本機目標及程序身分。

後續依序完成 R2 → E ledger／annotation index → n05b → n06 → n09，再核 G3 preflight 的資料證據；現有 Stage 1 chain 會自行接訓練、gallery、G4、既有 retrieval。Stage 2 等待 Stage 1 與 room graphs 準備完成後，建立自身 gallery、訓練 query fusion／ESSGNN，再做 layout-off 物件檢索。**G3 和新的 custom evaluator 目前都不在 live chain 的自動命令裡**，執行順序及驗收必須另外記錄。

新的自訂 Table 1 要先固定 query/gallery UID 與外部描述，同一批資料跑 3 方法 × 2 觀測 × 7 模態的 42 組比較；保留逐 query ranks、有效 token、來源 hash、選模重疊。不能為了接近論文分數調低結果，也不能把 Table 1 layout-off 成績當作 ESSGNN 場景品質。

詳細程式／命令與 Stage 2 路徑見 [CUSTOM_TABLE1_EVALUATION](../CUSTOM_TABLE1_EVALUATION.md)、[DATA_FLOW](../DATA_FLOW.md)，以及使用者指定的 [Notion 執行指南](https://app.notion.com/p/3d4fb0e74e5b80c8b141e686994fef70)。完整 Table 1 協定仍有作者未公開部分；正式 Table 2 scene／judge 配置、GAT 的未知架構等不能自行補成論文事實。

指定 Notion 頁面已寫入並回讀核對：30 個小節、10 張原生表格、8 段 shell 命令完整保存，title／日期／狀態亦核對成功。見 [發布與回讀 receipt](corpus_20260908_artifacts/notion_publication.json) 及 [發布本文](corpus_20260908_artifacts/notion_table1_stage2_published.md)。Notion 正規化了相鄰 bold／code spans；表格內容與命令 bytes 相同，全文忽略空白與 bold 分段後相同。這些命令是可讀模板，本輪沒有用它們啟動正式 GPU 工作。

本次證據封存見 [artifact manifest](corpus_20260908_artifacts/MANIFEST.json)。commit／push 狀態以 Git 與實際 remote 驗證為準，不把尚未發生的 push 寫成已完成。
