# MetaFind：標註更新與評估計畫

**10 頁、繁體中文、16:9，約 12–13 分鐘**。每頁備忘稿都有完整講稿。最新版本加入 ULIP-2 的 `Sillita Roja` 單筆資料，主指標改回 **R@1／R@5**；正式完整資料評估尚未執行。

| 檔案 | 用途 |
|---|---|
| [PowerPoint：含逐頁講稿](MetaFind_Table1_Stage2_含講稿.pptx) | 可編輯文字、表格、流程；講稿在每頁備忘稿。沿用原檔名。 |
| [PDF 預覽](MetaFind_Table1_Stage2_含講稿.pdf) | 由實際 PPT 轉出的 10 頁，不含備忘稿。 |
| [逐頁講稿](逐頁講稿.md) | 與 PPT 同步，含每頁來源。 |
| [十頁總覽](總覽.png) | 快速查看全部版面。 |
| [Sillita Roja 原始欄位摘要](source/ulip2_sillita_example.json) | 同一筆資料的 15 欄、原文、形狀、dtype 與來源 SHA；大量向量以 shape 表示。 |

## 十頁內容

1. 標註更新與評估計畫。
2. 11 張獨立圖片如何交給 Gemma 產生資料卡。
3. 真實椅子的 Gemma 資料卡；尺寸與重量是估計。
4. 格式與數值檢查、最多三次回答，以及與論文的差異。
5. 標註進度快照與預估完成時間。
6. **Sillita Roja**：ULIP-2 原始名稱、BLIP／MSFT caption、點資料及預計算特徵。
7. **換一份線索，找回對應資產**：query、固定候選庫與 UID 配對；三模型與七模態。
8. **R@1／R@5**：100 道示意題中，30／65 道找對的計算例。
9. 指標、query／正解、PC baseline、候選規模與訓練設定的差異表。
10. 標註完成後的資料、模型、查詢、配對與正式評估流程。

## 本次 R 指標修正

**PAPER FACT**：論文 [3experiments.tex:18](../docs/paper/metafind_source/3experiments.tex#L18) 使用 top-k retrieval accuracy，名稱是 R@1／R@5。
**IMPLEMENTATION CHOICE**：這次報告以每個 query 的對應資產 UID 為唯一正解，採 `R@k = 100 × 目標排進前 k 名的 query 數 / 全部 query 數`。這也是既有 [same-UID 計分](../docs/CUSTOM_TABLE1_EVALUATION.md#固定清單與計分)的規則，不宣稱作者已明訂完整 UID 配對方式。

使用另一份描述，不需要同時改成「多個合適答案」，也不需要把主指標改名為 Hit。先前簡報額外引入的多正解人工 qrels 與按全部正例比例計算的 Recall，已從本次主計畫移除。每題只有一個正解時，兩種計法的數值相同；第 8 頁的 30%／65% 僅是教學示意，沒有模型實驗分數。

本次更新範圍為簡報、講稿及來源紀錄，沒有重寫評測程式。repo 已有的 `intent_*` 多正解原型仍保留，其 Hit／Recall 輸出不能直接冒充本次單 UID 方案的執行結果。正式準備會以既有 same-UID protocol 的來源核對與固定清單為基礎；原始 ULIP-2 caption 的全量配對，以及直接改用 ULIP-2 點資料的交接，尚未完成正式驗證。

## Sillita Roja 的來源

**OBSERVED DATA**：UID `8505977020be4fd194854ad3d9808222`，原始來源為：

```text
/mnt/data1/kyzen/datasets/ulip2_objaverse_lvis/ULIP-2/objaverse_lvis/000-145.tar.gz
成員：000-145/8505977020be4fd194854ad3d9808222.npy
```

BLIP 原文是 `a chair with a pink cushion and a wooden frame`；MSFT 原文是 `a chair with a red cushion`。xyz／rgb 各為 float16 `(10000,3)`，image_feat 為 float32 `(12,1280)`。兩份 caption 的顏色描述不同，來源存在不等於內容完全正確。原始 NPY 解碼後核對 SHA，再產生本資料夾的欄位摘要。

此筆沒有原始照片、GLB、尺寸或重量。投影片第 6 頁椅子圖沿用既有**同 UID 的本地渲染圖**，不是假稱 shard 內有原圖；第 3 頁的尺寸、重量則來自另外產生的 Gemma 標註。

原始匯出檔留在 `output/inspection/ulip2_one_object_85059770/record.npy` 與 `record.json`。Git 只收錄帶來源 SHA 的小型摘要，不重複提交全部向量。

## 標註快照與評估界線

標註頁 2–5 保留既有資料；快照時間為 **2026-09-08 04:12:30（台北）**，不是重新查詢的即時狀態。本輪排程 46,004 件，已處理 25,089 件（54.54%），成功 25,063 件、失敗隔離 26 件，剩餘 20,915 件。起跑前另有 8 件成功，不計入本輪排程。

若持續運作且速度相近，當時預估 **9/9 上午**完成本輪標註，規劃 **09:00–12:00**；這是 **INFERENCE**，不含編碼、訓練、評估與排除整理。證據見 [進度快照](source/annotation_progress_snapshot.json)。

標註採 Gemma 4 12B／BF16、11 張獨立圖片與本地 prompt／重試。論文採 GPT-4o；相機、CLIP 訓練範圍、候選數與選模資料仍有差異或未知項目。Stage 2-off 沿用父 Stage 1 候選向量，不輸入場景資訊，也不代表場景品質或 Table 2 已驗證。相同 UID 的替代描述檢索，測量換觀測後的辨識能力，不直接證明任意家具需求或替代品推薦的品質。

## 修改、製作與檢查

本次保留 10 頁，將原第 6–10 頁重新安排為單筆資料、查詢流程、R 指標、差異表與後續步驟，修正第 1 頁講稿中的評估目標；其餘標註內容與原始圖片保留。未修改執行中的標註、模型、資料集或線上 Notion。原 [Notion](https://app.notion.com/p/3d4fb0e74e5b80c8b141e686994fef70) 的 snapshot 僅作歷史背景。

使用既有 Python 環境的 `python-pptx 1.0.2` 與 Noto Sans CJK TC，重建 PPT／講稿：

```bash
/home/kyzen/miniconda3/envs/MetaFind/bin/python notion_paper/source/build_deck.py
```

重建會覆寫 PPT／講稿；PDF 和總覽需另外從實際 PPT 渲染。交付檢查包括 10 頁／10 份備忘稿、文字一致、來源與欄位、LibreOffice PDF 渲染、文字框裁切與視覺檢查。結果見 [validation.json](source/validation.json)，來源與交付 hashes 見 [source_manifest.json](source/source_manifest.json) 和 [delivery_manifest.json](source/delivery_manifest.json)。未以 Microsoft PowerPoint 桌面程式開啟。
