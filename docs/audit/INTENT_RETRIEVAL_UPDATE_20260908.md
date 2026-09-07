# 2026-09-08 家具需求檢索更新

依使用者「對啊，那你應該這麼設計」與接續提供 ULIP-2 資料位置的指示，新增獨立需求、多個可接受答案的評估。這是 **IMPLEMENTATION CHOICE**；作者完整 Table 1 query／relevance 協定仍是 **UNKNOWN**。目前完成程式與輸入草稿驗證，**真實需求題庫、逐條件人工 qrels 與正式模型評估尚未完成**。

## 改了什麼

- [intent_protocol.py](../../metafind/eval/intent_protocol.py)：query ID 不要求等於候選 UID；只接收明示原始文字／照片／點雲。每題每個可見模態條件各自完整審核固定候選池，未審項不能補零。凍結 spec、qrels、來源與 hashes。
- [intent_metrics.py](../../metafind/eval/intent_metrics.py)：主報 Hit@1／Hit@5，另報真正 Recall@1／Recall@5；候選可以有多個正例。同分按 UID 字典序，不用答案決定排名。
- [intent_retrieval.py](../../metafind/eval/intent_retrieval.py)：預訓練 ULIP-2＋本地 mean、Stage 1、可選 Stage 2-off；query 不從 qrels 補入資料，gallery 始終完整 T/I/P 並預先編碼，Stage 2 保留父 Stage 1 gallery。記錄逐題排名、條件分母、有效文字 tokens、checkpoint 與執行身分。
- [prepare_ulip2_intent_queries.py](../../tools/prepare_ulip2_intent_queries.py)：依 metadata 找指定 shard，抽取指定原始 caption 與可選 xyzrgb；交叉核對 metadata／tar member／內部 dataset、group、id。缺值不改用其他欄位；草稿不自動產生正解。
- 新增[操作指南](../INTENT_RETRIEVAL_EVALUATION.md)，同步入口與[資料流](../DATA_FLOW.md)。舊 42 組保留為同資產辨識診斷，其既有計分與 runner 未修改。
- 更新[十頁 PPT 與講稿](../../notion_paper/README.md)的評估部分：需求、多個答案、Hit／Recall 與標註後的流程。標註頁 2–5、原始椅子圖與進度快照保留。

## ULIP-2 真實輸入驗證

**OBSERVED DATA**：使用者目錄包含 160 個 tar.gz，共 185,079,655,215 bytes；沒有宣稱完整讀過全庫。已抽查實際 NPY 的原始欄位，並以完整指定 shard 掃描，成功輸出下列一件真椅子的草稿：

| 項目 | 實際內容 |
|---|---|
| UID | `8505977020be4fd194854ad3d9808222` |
| shard／member | `000-145.tar.gz`／`000-145/8505977020be4fd194854ad3d9808222.npy` |
| 原始 BLIP | `a chair with a pink cushion and a wooden frame` |
| query ID | `ulip2-8505977020be4fd194854ad3d9808222` |
| 原始 xyzrgb 匯出 | `float32 (10000, 6)`，有限值與 RGB 範圍通過；不重新採樣 |
| 草稿狀態 | `needs_relevance_review`；1 筆成功、0 失敗；没有 qrels／judgments |

草稿留在 `output/validation/intent_retrieval_20260908/ulip2_chair_draft/`，來源紀錄副本見 [ulip2_chair_draft.json](intent_retrieval_20260908_artifacts/ulip2_chair_draft.json)。點雲原檔仍在驗證輸出，不把原始資料集複製進 Git。

BLIP／MSFT caption 可作另一份描述或需求草稿，不能當成人工正確答案。已查看樣本的 `image_feat`／`thumbnail_feat` 是向量而非原圖；不能只看維度就混用到本次 encoder。`retrieval_text` 也不自動視為同一資產的準確描述。上述是對本機實際資料的觀察，不認證 caption generator checkpoint 或全库品質。

## 新鮮驗證

| 檢查 | 結果與界線 |
|---|---|
| 完整 CPU suite | **2040 passed、71 warnings，61.38 秒**；排除 GPU／hooks，無 skip／failure。新增 187 案例，158 份本次綁定 Python 來源在測試前後一致。 |
| 新 protocol／scorer | 缺標、模態條件、來源漂移與非法輸入的負向測試；多正例指標另以獨立數值例及 brute-force oracle 比對。 |
| 三模型 runner | 真小型 fusion／計分，替代 pretrained loader；驗證原始 query、條件專屬 qrels、Stage 2 父 gallery 不變與失敗不發布完整結果。未執行完整預訓練模型的新協定實驗。 |
| 原始資料 adapter | 45 案例涵蓋 tar／NumPy 解碼與 UID 交接；另執行上述真實單 UID 抽取。 |
| PPT | 10 頁／10 份講稿；真 LibreOffice PDF、逐頁文字與文字框讀回通過，總覽人工視覺檢查完成。未使用 PowerPoint 桌面程式開啟。 |

原始 [CPU log](intent_retrieval_20260908_artifacts/cpu_tests.log)、[驗證紀錄](intent_retrieval_20260908_artifacts/verification.json)與 PPT [validation.json](../../notion_paper/source/validation.json)保留。測試只支持上述程式行為，不證明人工標籤有效、獨立測試成立或論文數字已復現。

## 接著怎麼做

1. 先固定評估候選池與來源資產清單，抽出 ULIP-2 原文；保留缺失，核對與本輪 corpus 的交集。
2. 確認需求文字，另取得原始參考照片；T、P 可先準備，沒有照片就不安排 I 條件。
3. 對每題每個可見條件審完全部候選，凍結多正解及判準；完整審核可以先從事先固定的小池開始，揭露規模，不對照論文大池分數。
4. 標註、編碼與指定 Stage 1／Stage 2 checkpoint 就緒後，核對來源並另行跑新協定，報 Hit、Recall、逐題結果與分母。來源 caption 找回同 UID 的實驗仍另列為辨識診斷。

本次沒有修改標註程式、執行中資料、生產鏈、protected paper source、vendor 或 Claude 設定，也沒有將新主評估接入 live chain。先前暫停的等待鏈工作與既有研究缺口不因此宣告完成。
