# 本輪資料邊界與文件交付證據

對應 [審查報告](../REPRODUCTION_CORPUS_REVIEW_20260908.md)。`MANIFEST.json` 綁定本目錄每個檔案的 SHA／大小；其來源路徑僅用於追溯，可能是本機 output 或當時暫存目錄。

| 證據 | 範圍 |
|---|---|
| `g3_final/` | DL-106 後新 G3 對 stable／paper 真實 corpus 的兩次 CLI；子目錄保留自己獨立的原始來源 manifest。 |
| `g3_initial/`、`g3_regression/` | 更早的錯誤診斷與回歸反例；不能替代 final 結果。 |
| `manual_*`、`verify_manual_e_ledger.py` | 原 ledger schema 問題及只在暫存副本內操作的實驗。 |
| `stage1*_deployment.json`、`deploy_waiting_stage1*.py`、`chain_paper_stage1_before*.sh` | 兩次等待 shell 部署及各自原版備份；不要直接重新執行部署 driver。 |
| `scene_score_handoff/` | 真場景產物的 import→aggregate；没有實際 judge，缺分均保持證據不足。 |
| `docs_cleanup.json`、`data_cleanup.json` | 已授權的文件搬移與兩個空目錄刪除；其餘項目保留原因。 |
| `notion_*` | 發布本文、預期結構與回讀 receipt；只保存指定頁面的交付內容，未封存 workspace 祖先頁面。 |
| `run_*` | 當時執行驗證的 drivers；部分 driver 使用本機固定路徑及排他輸出，不是可無條件重跑的公共 CLI。 |

這些小型 receipts 不包含大型模型、NPZ、GLB、blend 或完整資料集。內部 JSON 的原始 paths／hashes 保留不改，移動這些文件並不會令所有 CLI 可在其他機器直接重播。正式使用時仍須取得並核對來源 bytes；不可用修改 manifest 的方式繞過身分驗證。

CPU／graph 原始 logs 與 execution records 位於上層 audit，最終應使用 `reproduction_corpus_20260908_delivery_*`；較早名為 `final_cpu` 的檔案只代表當時 source snapshot。
