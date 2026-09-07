# MetaFind：10 頁簡報與講稿

根據使用者指定的 [Notion 指南](https://app.notion.com/p/3d4fb0e74e5b80c8b141e686994fef70) 製作，內容更新至 2026-09-08。**精確 10 頁、16:9、繁體中文；每頁講稿寫入 PowerPoint 備忘稿。** 建議講述約 17.5 分鐘，可依報告時間調整。

| 檔案 | 用途 |
|---|---|
| [MetaFind_Table1_Stage2_含講稿.pptx](MetaFind_Table1_Stage2_含講稿.pptx) | 可編輯文字、表格與流程；每頁有完整備忘稿。數值圖由 matplotlib 產生，來源與獨立圖檔另存。 |
| [逐頁講稿.md](逐頁講稿.md) | 相同的 10 頁講稿，每頁 363–539 個中文字；文末保留 Notion 的 8 段完整命令模板，不增加投影片頁數。 |
| [PDF 預覽](MetaFind_Table1_Stage2_含講稿.pdf) | 由實際 PPT 轉出的 10 頁投影片，供直接閱讀；此 PDF 不含備忘稿。 |
| [全部頁面總覽](總覽.png) | 一張圖檢視十頁的順序與版面。 |

## 十頁順序

1. Table 1 評估與 Stage 2：問題、證據與下一步。
2. 同資產檢索、房間診斷、完整場景品質的差別。
3. 2 種觀測 × 3 個模型 × 7 種模態＝42 組。
4. UID 正解、cosine、保守同分排名與 R@k 教學例。
5. T＋I／T＋PC 偏高的歷史證據與尚未確定的原因。
6. Prepare → check-only → run → 完整逐筆驗收。
7. G3 人工排除計帳、Stage 1 選模重疊與 A20／B 差異。
8. Stage 2 的場景資料依賴與訓練設定。
9. S2-off／S2-on 分流，以及已完成小型驗證的界線。
10. 後續執行順序與交付依據。

## 講稿與來源

在 PowerPoint 編輯畫面打開下方「備忘稿」，或在簡報者檢視中閱讀。獨立講稿與 PPT 使用同一份逐頁文字；長命令放在講稿附錄，尚未準備的 `/path/to/` 路徑仍須依實際資料指定。

數值與科學界線已另由獨立代理回查原始 JSON、G3 YAML、論文 Table 1 與執行紀錄。圖表並列本地／論文數字時，明記觀測與候選協定未證相同；不將小型測試、G3 部分 PASS 或高分寫成正式論文結果。原始資料與頁面內容見 [source/](source/)。

## 製作與驗證

使用已安裝的 `python-pptx 1.0.2`、matplotlib 和 Noto Sans CJK TC；未啟動任何標註、訓練或評估 producer。來源固定於本次頁面版本，製作程式只服務這份簡報的版面與內容。

```bash
/home/kyzen/miniconda3/envs/MetaFind/bin/python notion_paper/source/build_deck.py
```

上述命令會重新產生此目錄的 PPT／講稿／圖表；編輯成品前應保留自己的版本。它不會自行重跑正式實驗或更新 Notion。

已實際讀回 PPTX：10 張 slides、10 份非空備忘稿、所有逐頁講稿相同；獨立講稿保留 8 段來源命令。以隔離 LibreOffice profile 轉 PDF，確認 10 頁並逐頁檢視總覽，另外檢查數值圖和密集版面。未以 Microsoft PowerPoint 桌面程式開啟；這是實際 LibreOffice 渲染與 PPTX 結構驗證。

驗證摘要與檔案 SHA 見 [source/validation.json](source/validation.json)；封面 PDF 文字抽取的交錯閱讀順序已透過逐行與視覺確認，沒有遺失標題。檔案完整性見 [source/delivery_manifest.json](source/delivery_manifest.json)。
