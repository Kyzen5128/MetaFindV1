# 2026-09-08 接續審查：批准集合、等待鏈與公式驗證

本輪修正 G3 將同名群組誤當批准集合的缺陷，以及等待鏈忽略上游失敗／誤讀歷史完成標記的缺陷。只替換經身分核對、仍在等待的 Stage 1／2 shell；標註與 R2 wrapper 保持原程序。完整 paper corpus、正式訓練與 Table 1–3 仍未完成。

## 1. Audit Scope

以當前 checkout 和實際程序重查：DL-106 原 21 筆人工排除 → G3；R2 → Stage 1 → Stage 2 的等待交接；附錄 Eq.10／13／14 已有獨立 oracle 的有效覆蓋。先唯讀審查與反例，再依既有修正授權進入實作。沒有變更科學選擇或啟動 GPU 工作。

## 2. Authority and Evidence Ledger

| 分類 | 來源與主張 | 界線 |
|---|---|---|
| PAPER FACT | [appendix.tex](../paper/metafind_source/appendix.tex) 第 32／50／65 行：同一舊狀態訊息、座標 residual、特徵 residual。 | 正文與附錄的矛盾仍分開；MLP 形狀、稀疏鄰域和 scalar 選擇並非這三式唯一指定。 |
| IMPLEMENTATION CHOICE | [DL-106](../../workflow/DECISION_LEDGER.md) 保留原 21 UID，只改 A／Q／E 計帳。 | 未授權新增、替換或減少人工排除。2% 是專案處理失敗門檻。 |
| OBSERVED DATA | [原始批准檔](../../workflow/annotation_exclusions_20260828.json) 原樣封存，SHA `39eff095ea6283601d57db7618d39ba2355db694e16b711ca82d5ddc59ef3058`。 | 17,391 bytes；其中歷史 311 筆 n05 失敗不是人工批准集合。 |
| OBSERVED IMPLEMENTATION | [G3](../../metafind/gates/g3_object_corpus.py)、兩份 [Stage 1](../../tools/chain_paper_stage1.sh)／[Stage 2](../../tools/chain_paper_stage2.sh) 腳本與 regression tests。 | 程式／測試不是 paper authority；G3 仍為 standalone input preflight。 |
| OBSERVED DATA | [部署後盤點](gate_chain_20260908_artifacts/status_after_deployment.json)（03:36 台北時間）：24,642 份可解析、契約標記相符的 v10 標註。 | 非原子盤點；沒有解碼核對所有輸入影像，不代表 corpus 完成。 |

## 3. Paper → Specification → Implementation → Validation Matrix

| 要求 | 規格／原文 | 實作與驗證 | 狀態／影響 |
|---|---|---|---|
| E 完整等於已批准人工集合 | DL-106；G3／L2-COMPLETE | 固定批准 source 的 path／SHA，先驗 bytes 再解析；核 E 的增、漏、替換、空集合及 manifest 漏 UID。 | MATCHED；MATERIAL，避免錯誤資料集合被 preflight 放行。 |
| 真失敗與人工排除分開、原分母不變 | G3 既定契約 | 舊穩定 corpus 再跑真 CLI：46,052 = 45,692 + 339 + 21，PASS。 | MATCHED，限 input accounting。 |
| 上游明確失敗不能永遠等待 | shell 工程契約，非論文公式 | R2 非零 EXIT／Stage 1 PIPELINE FAILED 使下游失敗；51 項 shell 測試與獨立審查。 | MATCHED；MATERIAL，停止誤導性等待。 |
| 新執行不能沿用舊 DONE | 每次執行的 START 邊界 | 只讀最後 START 後狀態；同次 failure 保持失敗；沒有明確終態則繼續等待。 | MATCHED；MATERIAL，避免未完成工作提前放行。 |
| 附錄共同訊息的一層值與梯度 | Eq.10／13／14；既有 D-14／U-26 選擇 | 既有獨立 NumPy／逐邊 oracle，兩種 MLP 的全 h／x／e／參數中央差分；fresh 2 passed。 | MATCHED，限指定單層／精度。 |
| G3 作為正式訓練前的完整 barrier | graph n09 → G3 → n10 | standalone record 尚未由 live chain／trainer 強制消費；不更改 `implemented: false`。 | PARTIAL，仍需整合驗證。 |

## 4. Runtime Trace

G3 的 corpus ledger 曾只按 `manual_review_rejected` 名稱納入 E。[修前反例](gate_chain_20260908_artifacts/g3_approval_before/summary.json) 使用真 `gate.run`：任意 `u000`，甚至搭配正確批准來源 SHA，均可在集合總數守恆時得到 PASS。現在由程式固定批准來源；corpus 提供的 `source_ledger` 只作 provenance，不可指定另一份可信來源。若宣告該欄位，其 SHA 與四項決策 metadata 必須與已驗 bytes 一致。

R2 wrapper 已實際寫 `=== R2 START`、`=== R2 EXIT <rc>`，成功才寫 DONE。Stage 1 每次執行／重啟均以 `=== 0 wait for R2` 開始。新版兩條等待迴圈按最後一次開始標記清除歷史狀態；缺檔、缺 START、尚無終態、讀取暫時失敗皆保持等待，不把 timeout 推論成程序失敗。

[部署 receipt](gate_chain_20260908_artifacts/waiting_chain_deployment.json) 記錄：先核對 PID／start ticks／cwd／stdout／fd255／原始與候選 SHA，暫停後確認唯一子程序為 `sleep 60`，只終止指定等待 shell 及其 sleep，確認舊 shell 結束才替換。原 script bytes 留存。新 Stage 1 PID **2721952**、Stage 2 PID **2721958**，都再次確認仍等待；annotation **2043427** 與 R2 wrapper **2043422** 的身分保持相同。這些 PID 是部署時點資料，後續仍應實際查 `/proc`。

## 5. Validation Evidence

- [G3 修後三次 CLI](gate_chain_20260908_artifacts/g3_approval_after/summary.json)：相同兩個反例都 FAIL；舊穩定 corpus PASS，讀入檔案前後 SHA 相同。Q/M ≈ 0.7361%，E/M ≈ 0.0456%，合計 ≈ 0.7817%。
- [G3 與相關接縫測試](gate_chain_20260908_artifacts/g3_approval_after/targeted_tests_execution.json)：240 passed；包含 G3／G4／runlog／splits 五檔，不能當作真實 corpus 執行結果。
- [Shell regression](gate_chain_20260908_artifacts/chain_wait_after.json)：51 passed；[修前](gate_chain_20260908_artifacts/chain_wait_before.json) 為 8 failed／13 passed／30 deselected，保留錯誤原始輸出。兩條 shell 各自通過 `bash -n`。
- [附錄 oracle](gate_chain_20260908_artifacts/appendix_oracle/execution.json)：2 passed／35 deselected／0.61 秒，四份源檔前後 SHA 相同。保存的 log 明示為事後 stdout 摘錄，不冒稱完整原始 log。數值容差 1e-12；梯度 rtol 3e-6、atol 2e-8，中央差分步長 1e-6。
- [最終整合 CPU](gate_chain_20260908_artifacts/integration_cpu_final.json)：**1,853 passed／71 warnings／59.26 秒**，exit 0；286 份來源前後 SHA 一致，排除 GPU／hook，沒有 skip。第一次整合雖 1,853 passed，但執行中 G3 測試補了 global LOGS teardown，來源快照因此不一致；[該次 receipt](gate_chain_20260908_artifacts/integration_cpu.json) 保留 `source_unchanged: false`，不當作最終版本認證。
- [Graph checker](gate_chain_20260908_artifacts/graph.json)：2,342 checks 全通過；此為規格與實作標記的結構檢查，不證明論文忠實度。

## 6. Implementation Choices

沿用既定 21 UID、2% 門檻、D-14／U-26／scalar 解讀。工程上用版本管理中的原始批准 bytes 與固定 SHA 綁定集合，沒有新增使用者研究選擇。未提供可用 CLI 任意替換批准來源的旗標。Test fixtures 可自行隔離 synthetic approval，但真 CLI 使用正式批准來源。

## 7. Deviations

本輪沒有新增或移除科學偏離。既有 CLIP frozen 範圍、LLM 替換、自訂 Table 1 query 協定等仍須按實驗 provenance 揭露。本次兩個工程 bug 沒有證據證明是歷史融合分數偏高的原因。

## 8. Missing or Unreachable Implementation

G3 尚未接入 live chain，也尚未提供正式訓練端驗證 record 的完整路徑。G3 的配置欄位檢查不能保證 optimizer 實際行為；部分可解析的配置仍會被 trainer 拒絕。這些仍是後續工作，不能因本輪批准集合通過就標為全 gate 完成。

ProcTHOR render 等待仍使用其既有成功標記；尚無本輪可驗證的失敗／attempt 契約，沒有自行推定。該輪 log 的 1,465 rendered 是新增 invocation 數，不應和總 1,467 sidecars 混用。

## 9. Unknowns and Conflicts

未發現需要重訂公式的新矛盾。9/7 公式報告說「當輪未持久化」是歷史時點；[9/8 訓練審查](../history/REPRODUCTION_TRAINING_REVIEW_20260908.md) 與現有測試已補足 appendix 單層 oracle，沒有再新增重複測試。它不能認證完整七層 IO projection／pooling，也不能裁決正文與附錄差異。

作者未公開的 Table 1 query／gallery 細節、正式 Table 2 scene／judge 協定等 UNKNOWN 持續存在；本輪沒有以工程預設填補。

## 10. Reproducibility Provenance

起點 commit `f0d0306c060aebc66ac576cd1d680048829e4840`。命令、source hashes、CPU／離線環境、返回碼及部署前後身分均在 [artifact manifest](gate_chain_20260908_artifacts/MANIFEST.json) 對應檔案。封存保留當時原始路徑與 bytes；根目錄 `docs/` 仍只放現行操作指南。舊封存與論文／vendor／Claude 檔案未改。

修前 `chain_wait_before.log` 有 16 行 pytest 原始 traceback 行尾空白，為保留原始 SHA 而不清洗。`git diff --check` 的這 16 項只在該 raw log；排除這一檔後，其餘 staged 內容檢查通過。

## 11. Reproduction Impact

批准集合綁定使 G3 不再以數量正確掩蓋 UID 錯誤；等待修補使後續程序不再吞掉明確上游失敗，且不能利用歷史成功提前執行。這提高復現的資料與執行可追溯性，沒有產生新的模型分數。

## 12. Verdict

**PARTIALLY VERIFIED**。本輪兩項工程修補已有反例、回歸與實際部署／corpus 證據；完整復現仍需完成 paper 標註、編碼、splits、正式 G3／訓練整合、Stage 1／2、固定自訂 query 的評估，以及場景品質評估。標註正在運行是經實際 PID 核對的等待，不能用現有檔案數宣告完成。

## 13. Required User Decision

本輪無新增使用者決策需求。既有批准足以修正兩項工程缺陷；未對尚未指定的科學協定作新決定。
