# 文件入口

2026-09-08 整理：根層保留現行操作指南。主評估改為真實需求與多個可接受答案；原 same-UID 七模態流程保留為診斷。舊報告、閱讀筆記與計畫集中在 [history/](history/README.md)，不再與現行操作文件混列。

| 要做的事 | 文件 |
|---|---|
| 理解資料來源、交接與 cache 更新 | [DATA_FLOW](DATA_FLOW.md) |
| 主評估：需求、逐條件多正解、Hit／Recall；ULIP-2 草稿到人工審核 | [INTENT_RETRIEVAL_EVALUATION](INTENT_RETRIEVAL_EVALUATION.md) |
| 診斷：固定 query/gallery UID 的兩觀測、三方法、七模態 42 組 | [CUSTOM_TABLE1_EVALUATION](CUSTOM_TABLE1_EVALUATION.md) |
| I-Design 輸出轉成明示場景查詢 | [IDesign_INPUTS](IDesign_INPUTS.md) |
| 原始 T/I/PC 與 checkpoint 編成可重播 bundle | [RAW_SCENE_INPUTS](RAW_SCENE_INPUTS.md) |
| 逐步檢索、加入資產、更新圖 | [SCENE_COMPOSITION](SCENE_COMPOSITION.md) |
| 補足 Objaverse 場景所需語義關係 | [SCENE_SEMANTICS](SCENE_SEMANTICS.md) |
| 真 GLB 放置、保存 blend 與明示渲染 | [SCENE_PLACEMENT](SCENE_PLACEMENT.md) |
| 匯入外部四維評分、保留缺分與失敗分母 | [SCENE_SCORES](SCENE_SCORES.md) |

目前完成項目與未驗證範圍見 [最新批准集合與等待鏈審查](audit/REPRODUCTION_GATE_CHAIN_REVIEW_20260908.md) 與 [資料／文件整理紀錄](audit/REPRODUCTION_CORPUS_REVIEW_20260908.md)。[專案 README](../README.md) 提供安裝／狀態入口；[測試導覽](../tests/README.md) 提供 CPU／GPU 分組命令。

| 證據類型 | 位置與界線 |
|---|---|
| 論文內容權威 | [MetaFind source](paper/metafind_source/)；其他 paper source 僅屬各自論文。 |
| 衍生規格與研究決策 | [graph](graph/README.md)、[DECISION_LEDGER](../workflow/DECISION_LEDGER.md)；不得覆蓋原文。 |
| 數學與執行驗證 | [公式／梯度審查](audit/formula_review_20260907.md)、audit 內原始 logs／JSON／小型產物封存。測試只支持其明示範圍。 |
| 歷史筆記、報告、舊路徑 | [history 導覽](history/README.md)。日期內容記錄當時狀態，引用前回查來源與後續更正。 |

歷史 audit JSON、決策紀錄及 frozen archives 中的原始路徑／SHA 保留原貌；搬移對照供查找，不把歷史驗證改寫成當前結果。CLAUDE 規則與論文 source 未因這次整理而修改。
