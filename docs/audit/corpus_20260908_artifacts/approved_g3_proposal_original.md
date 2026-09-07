# G3 人工排除計帳提案：等待使用者決定

**狀態：PROPOSED，尚未採用。** 這是正式資料驗證規則的研究／工程選擇，不是論文原文，也不是重新詢問是否排除那 21 筆。

**OBSERVED DATA：** 既有 `annotation_exclusions.json` 記錄 Kyzen 2026-08-28 的決定，包含 21 筆 `manual_review_rejected` 與 311 筆當時的 `n05_quarantine`。新 v10 corpus 已排定重試舊失敗，並在 R2 完成後延續 21 筆人工排除。本次等待鏈修正僅讓這個既有決定正確執行。

**UNKNOWN／契約衝突：** [G3／L2-COMPLETE](graph/validation_plan.yaml:1058) 要求 `admitted + quarantined == len(manifest)`，而 quarantine 要有真實例外來源。人工判定拒收不等於程式 exception；目前沒有批准的規則說這 21 筆應算在哪一項。不能為了過關捏造 exception、默默縮小原始 manifest，或直接讓缺少 worker 結果的資產也變成 quarantine。

建議採用第三項 `manual_excluded`，具體定義如下：

| 符號 | 提案定義 |
|---|---|
| M | 原始 LVIS manifest，保持原檔與原始分母。 |
| E | 有批准決策與 ledger 身分的人工排除 UID；要求全部屬於 M，獨立列出數量與 UID。 |
| A | 最終 admitted UID，要求與 E 不重疊。 |
| Q | 有真實 exception 證據的唯一 UID，扣掉 A 與 E。成功重試不算 quarantine；人工拒收不重複算處理失敗。 |

提案的正式檢查為：三集合兩兩不重疊，`A ∪ Q ∪ E == M`，沒有未解釋遺失或 manifest 外 UID；既有處理失敗門檻仍為 `|Q| / |M| ≤ 2%`。另外揭露人工排除率 `|E| / |M|` 與合計排除率 `(|Q| + |E|) / |M|`，但不另新增門檻。

如果採用，需同步更新 G3／L2-COMPLETE 契約、gate 實作與反向案例，並將選擇記入決策紀錄；不能只改程式讓舊條件「看起來通過」。2% 是既有專案規則，論文沒有指定這個門檻。

在決定前，[目前 G3 preflight](../metafind/gates/g3_object_corpus.py) 對非空人工清單回傳 `BLOCKED_EVIDENCE`，保留所有原始資料與例外記錄。它尚未接入既有 live chain；本提案也沒有暫停標註或變更已排定的訓練。
