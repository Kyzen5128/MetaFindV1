# 2026-09-07 測試、評估程式與文件整理紀錄

本次依使用者授權整理完整 `tests/`，並清理確定失效的評估程式、修正相關 Markdown、補上實際資料流。沒有重訓、重算 Table 1 或把程式測試當成論文復現證據。

## 1. 完成的整理

| 項目 | 處理與理由 |
| --- | --- |
| 52 個測試檔 | 全數保留原 basename，按責任搬到 `tests/data`（13）、`models`（11）、`train`（11）、`eval`（9）、`pipeline`（6）、`gpu`（1）、`hooks`（1）。`git diff` 在未加入新檔前可能顯示舊路徑 deleted，實際內容在新目錄。 |
| 重複測試 | 刪除 `test_annotate.py` 中被後面同名定義遮蔽、AST 完全相同的 3 個 test 與 2 個 helper；保留有效定義。這些重複本來就沒有被 pytest 多收集。 |
| 測試隔離 | `rank_descriptions` 的直接全域賦值改成可還原的 `monkeypatch`；單獨執行的修復案例自行提供 ranker，消除依赖前一測試的情況。僅修改測試，沒有修改標註執行碼。 |
| 測試定位與斷言 | 修正搬移後的 repo 路徑；runlog AST 掃描改從真實 repo 起算並要求非空，避免換工作目錄後空掃也通過。query partner 測試依 view 數取索引；未知 policy 負例不再要求真 corpus。 |
| 排除名單的真實驗證 | 重寫 `test_eval_runner.py` 的 degraded-render exclusion case；移除吞掉任意例外的寫法，在 encoding 入口明確驗證 query／gallery 都是 `[a, c]`。原測試啟動真 DataLoader 後失敗，延遲清理造成跨測試 thread 警告。 |
| 測試說明 | RNG checkpoint 負例實際驗的是 `preserve_rng_state=False`，原名字卻寫成 reentrant 比較；已更正名稱／說明，保留原負向斷言。 |
| 規格檢查 | `tools/check_graph.py` 遞迴掃描新測試目錄，UNKNOWN 編號檢查也涵蓋子目錄；graph README 的頂層 test function 數更新為 968。函式數、pytest case 數與規格 check 數是三種不同計數。 |
| 歷史字串稽核 | `tools/audit_claims.py` 修正搬移路徑，明示為 2026-08-28 的固定文字稽核，不是目前行為測試或新 promotion gate。 |

測試範圍、資料依賴與執行命令见 [tests/README.md](../../tests/README.md)。三個上一輪新增的 custom evaluation 測試全部保留在 `tests/eval/`。

## 2. 確定刪除的程式

以下 4 支 probe 的評估已在 2026-09-03 撤回。本次讀取全部檔案並核對 imports、測試、工具與等待鏈後，才刪除；沒有刪除研究結果 JSON：

- `tools/probes/exp_query_observation.py`
- `tools/probes/exp_text_observation.py`
- `tools/probes/exp_observation_matrix.py`
- `tools/probes/exp_text_template_ablation.py`

**OBSERVED IMPLEMENTATION：** 原共用 gallery 使用零點雲當作存在的 PC 模態，私有 scorer 又採用有利同分與 float32。撤回後已改成拒跑 stub，三個 consumer 因而必定無法完成評估。

仍被兩個 active probes 使用的 `load_tower` 已抽至 `metafind.eval.custom_models.load_fusion_tower`，保留只還原 fusion heads 的語義；兩處 imports 與 checkpoint identity 測試一併更新。刪掉一個只檢查已不存在 stub 的測試，保留零點雲 gallery 與共用 scorer 防護。

撤回理由、保留 helper 與可用入口見 [tools/probes/README.md](../../tools/probes/README.md)。未把「找不到 import」單獨當成刪除理由；獨立執行的歷史實驗與簡報工具仍可能有重播用途，予以保留。

## 3. 文件更正與資料流

- 重寫 [專案 README](../../README.md)，新增 [文件導覽](../README.md)、[DATA_FLOW.md](../DATA_FLOW.md)。資料流明列 producer、輸入、持久化產物、consumer、身份檢查與更新後需重驗的下游。
- 更正兩份 9 月 5 日評估筆記及 v2／v3 報告的因果推論：撤回「全領域只有 shape gallery」「沒有 P→P／組合輸入」「同 UID 融合必定加分」「分數低足以證明作者 query 更弱」等超出證據的結論。Uni3DL 的 Text2Shape 5.8/19.7 不再錯貼為 Cap3D 成績。原始本地量測 JSON 與歷史 Table 1 分數未改。
- `STAGE1_RESOLUTION_PLAN_20260903.md` 標出上游示例與圖中 JSON 不能證明 MetaFind 的 query／CLIP serializer；未改歷史實驗數字。
- `METAFIND_NOTEBOOK.md` 更正權威順序、Stage 2 evaluator 尚未實作的舊說法、從引用 ULIP-2 推定具體 checkpoint 的錯誤、撤回 probe 的現況；其餘歷史段落保留並標記時效。
- `workflow/MASTER.md`、`PROGRESS.md`、`Session Handoff.md` 明確標成歷史快照，導向現在的程式／文件入口。handoff 中「shell 不可提保護路徑、用拼字串繞過」的錯誤指示已改正；Codex 不能宣稱受 Claude hook 保護。
- 維護中的 graph 文件與自訂評估命令改成新測試位置。歷史 audit JSON 的舊路徑、SHA 與測試命令保留，避免把舊驗證改寫成新驗證；本輪 JSON 另存測試搬移對照。

**OBSERVED IMPLEMENTATION：** ProcTHOR metadata 工具改 `procthor_object_text.json`，但 n11b 的文字讀自 `procthor_modalities/<assetId>.json`。node embedding、modality sidecar 與 gallery index 是不同產物；修改文字不會自動更新整條資料流。本輪記清楚這個邊界，沒有擅自重建資料或替換研究設定。

本次修正已確認的錯誤與現在入口，沒有對所有歷史 Markdown 再做一次全文論文稽核。歷史主張仍須按原文、原始 artifact 與後續決策檢查。

## 4. 新鮮驗證與界線

**OBSERVED DATA：**

- 搬移後、刪除 stub test 前：1,236 passed，與整理前 CPU collection 相同。
- 刪除失去測試對象的一項後，完整 CPU suite：**1,235 passed，68 warnings，26.83 秒**。
- graph checker：**2,343 checks，all pass**。這只驗證其明列的文件／程式一致性，不認證論文忠實性。
- helper 抽取相關測試：30 passed；最後路徑修正從 `/tmp` 獨立執行：13 passed。
- 已刪模組的 runtime imports／literal call targets 掃描：222 個 Python 檔，未發現懸空引用；三個保留 consumer 實際 import 通過。等待鏈未引用四個已刪模組。
- `git diff --check`、新入口本機 Markdown 連結檢查通過。程式碼與保護路徑核對細節見 [驗證紀錄](../audit/cleanup_20260907_checks.json)。

本次完整 CPU 命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 /home/kyzen/miniconda3/envs/MetaFind/bin/python -m pytest tests -q -ra   -p no:cacheprovider --ignore=tests/gpu --ignore=tests/hooks
```

warnings 剩既有 Transformer／timm 與刻意測試偏離溫度的警告。曾出現的 `QueueFeederThread` teardown 已定位到吞掉 DataLoader 失敗的 exclusion 測試；同一組 54 cases 修前捕獲 2 個 thread 異常、修後為 0，最新完整 suite 也未再出現。沒有過濾或隱藏警告，沒有改 production multiprocessing 策略。

**未執行：** GPU smoke、Claude hook 測試、真實 ULIP-2 全量推論、重訓、重新評估 Table 1 或完整 Table 2 場景品質。通過 CPU suite 只支持測到的實作行為。

## 5. 標註、排程與檔案安全

標註 PID **2043427** 的啟動時間維持 **2026-09-06 15:24:04**，檢查時仍在執行 `python -m metafind.data.annotate_run --prompt-mode figure2_v10`。等待中的 Stage 1 PID **2465870** 與 Stage 2 PID **2494552** 保留，沒有停止、重啟或改寫 live shell。

比對 580 個保護／凍結檔案的 SHA-256：`metafind/data/**`、`metafind/compat/**`、`metafind/vendor/**`、`metafind/paths.py`、`metafind/runlog.py`、paper source 與 `.claude/**`／`CLAUDE.md`，**沒有內容改變**（不包含 bytecode）。沒有操作真實 annotation 輸入／輸出；annotation 測試只在獨立 pytest process 使用 fake 模型與暫存資料。

整理快照保存在 `/tmp/metafind_cleanup_20260907_205025/workspace_snapshot.tar.gz`；tracked 舊版本亦可從 Git 找回。此快照在測試搬移期間建立，原 tracked 測試以 Git 的整理前 revision 為準。新 custom evaluation 的未追蹤檔也已包含於快照，沒有清除。`Session Handoff.md` 原本被 gitignore 排除，本次仍只作本機工作記憶更正；交付內容另寫於本報告與文件導覽。

本次未 commit；所有變更仍在共同 workspace 可供 review。使用者後續要求的端到端復現審查接著進行，相關修正另留證據，不把本輪整理驗證冒充完整復現驗證。本報告不是新的訓練啟動指令，也不代表 Claude 已讀取文件。
