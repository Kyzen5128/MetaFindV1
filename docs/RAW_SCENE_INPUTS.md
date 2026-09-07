# 原始查詢到可重播場景

[scene/prepare.py](../metafind/scene/prepare.py) 現可還原 Stage 1 的真實 query backbone，讀取已核驗 gallery，以原始文字／圖片／已準備點雲建立 [composition bundle](SCENE_COMPOSITION.md)。Stage 2 的 query fusion、ESSGNN 與 lambda 由完整 exporter 還原。它是工程交接，不是作者未公開的 200-scene 抽樣協定。

## 明確輸入

`request.json` 必須有以下欄位；路徑相對於這份 JSON：

```json
{
  "schema": "metafind.scene_request.v1",
  "provenance": {"source": "這個房間與查詢的實際來源"},
  "stage1_record": "/path/to/stage1_record.json",
  "stage2_record": "/path/to/stage2_record.json",
  "variant": "full",
  "gallery_registry": "/path/to/promoted-gallery_index.json",
  "gallery_ids": "all",
  "asset_texts": "/path/to/asset_texts.json",
  "semantic_cache": "/path/to/sem_edge_cache.json",
  "initial_graph": {"room_id": "explicit-room", "nodes": []},
  "queries": [],
  "mode": "iterative",
  "use_layout": true
}
```

上例的空 queries 是欄位示意，執行時必須換成非空、有明確順序的查詢。每項 query 是 `{"slot": 完整IDesign物件, "text": 原始描述, "images": [明確影像路徑], "pointcloud": 已準備NPZ路徑}`；只保留實際存在的模態，至少一種。`slot` 格式見 [SCENE_COMPOSITION](SCENE_COMPOSITION.md)。不提供隱含 query 模板、隨機挑圖或零向量補模態。

`gallery_ids` 可以是明確、唯一的 UID 清單；`"all"` 表示保持 promoted index 的完整順序。會驗 index bytes、父 checkpoint 和實際載入 gallery encoder 的 v2 forward 身分。尚不接受 legacy gallery 或直接把 ProcTHOR index 當 Objaverse promoted gallery。

`asset_texts.json` 是 `UID -> {"text": 精確節點文字, "relation_text": 可選關係描述}`，須涵蓋本次 gallery 與 G0 內的已放置資產。這些文字描述取回資產，不是 query 希望得到的物件；不以同維度向量代替來源相容性。G0 只能含已有真實 asset ID 的物件；有位置的未檢索 planned slots 仍屬 queries。

## 實際編碼與來源

- **OBSERVED IMPLEMENTATION：** text 由還原後 frozen CLIP 編碼，保留 raw 向量。image 用相同 backbone 的 preprocess，逐個明確 view 編碼後對 raw 向量平均。只支援 checkpoint 已使用的 `image_tokens=1`／`image_aggregation=mean`。
- PC 必須由呼叫者提供已準備的 NPZ：finite float32 `xyz` 和 `rgb` 均為 `(10000,3)`；不在此處偷偷重採樣或再標準化。`fully_separate` checkpoint 使用已還原的 query point path，沒有誤用 gallery point encoder。這個欄位的 hash 證明 bytes 一致，不證明外部 producer 遵守座標或採樣契約。
- full/layout checkpoint 的 node／edge 空間來自它自己綁定的 n08 訓練產物。驗證精確文字／句子、NPZ 內嵌 source digest、UID／key row、實際 text weights／tokenizer／forward implementation 身分。新的 asset node text 由同一個文字編碼器 L2 normalize，與 n08 相同；query raw 向量的處理不同，不能互換。
- 訓練 cache、當前 cache、實際載入文字編碼器必須一致；prompt version／LLM 名稱／text encoder version 也核對。**舊 n08 產物缺 producer binding 時拒絕**，不補寫今天的 sidecar 來宣稱歷史來源已驗證。此處尚無 legacy 降級旗標。
- 真 `no_layout` checkpoint 只允許 `use_layout=false`；不要求或讀取未使用的語義 cache／node vectors，記錄 `not_used`。full checkpoint 的 layout-off 對照仍保留原語義驗證。

## 執行與產物

```bash
python -m metafind.scene.prepare --request /path/to/request.json \
  --out-dir /path/to/new-bundle --device cpu
python -m metafind.scene.compose --manifest /path/to/new-bundle/manifest.json \
  --out /path/to/new-composition.json
```

`--device cpu` 是預設，使用真 ULIP／CLIP 時仍需相應記憶體與本機權重。GPU 空閒時可明確選 `--device cuda`。2026-09-08 已另行執行真 ULIP／CLIP 的三 query CPU 診斷，见 [真場景紀錄](REPRODUCTION_SCENE_REVIEW_20260908.md)；不是一般 pytest 的 tiny seam。

只有實際 CPU composition replay 通過、所有輸入再驗 hash 後，才發布 `manifest.json`。另存 `inputs.pt`、完整 query model、`validation_composition.json`。輸出必須為新目錄；失敗可能留下中間檔，沒有 `manifest.json` 就不是已完成 bundle。全部 bytes、原 request、encoder／gallery 身分及工具 source hash 進 provenance；不更改原 checkpoint 或 canonical corpus。

若當前 trajectory 遇到缺少的關係，拒絕並寫 `required_pairs.json`（UID pairs）及 `missing_semantics.json`。它們只描述這次已遇到的缺口，不列全部 gallery 的 N² pairs，也不把未知 key 改成 `None`。擴充 cache 後，以新的輸出目錄重播；新關係可能改變後續取回資產，因此可能遇到下一批缺口。只有有證據的 SG2 生成／修復耗盡才是 degraded relation。

現可將 `required_pairs.json` 交给 [scene.semantics](SCENE_SEMANTICS.md)，用 `--base-cache` 延伸既有相容證據，再把新 cache 路徑填回 request。它重用 n08 的同一套 prompt／parse／repair／encoder；嚴格比對文字編碼器，但歷史 LLM 權重 bytes 不存在，仍記 `UNKNOWN`。此 prepare→補關係→新目錄重播流程由明確命令串接，尚未包成一個自動循環命令。

完成 composition 後，以 [placement](SCENE_PLACEMENT.md) 的明確 asset manifest、room surfaces 和可選 render config 建立 `.blend`／圖片。模型的真實 front、相機、燈光與正式評分協定仍不能由 query 或分數倒推。

## 已驗證範圍

[CPU 測試](../tests/eval/test_scene_prepare.py) 覆蓋原始文字、兩 view 平均、separate point path、真 n08 schema／source guards、語義與 query 不混用、真小型 checkpoint exporter→manifest→composition、來源替換、缺關係輸出與排他發布。大型 backbone constructor 使用 tiny seam；這些結果不是已執行正式 ULIP-2、生成正式 200 scenes 或重現 Table 2 分數的證據。

另行的 [真場景驗證](REPRODUCTION_SCENE_REVIEW_20260908.md) 使用兩個真單步訓練 checkpoint、六筆 promoted gallery，經兩次真 Gemma 補關係後完成三 slot。prepare 的 validation composition 與另一程序的公開 CLI replay JSON 完全相同；完整正式場景評分仍未完成。
