# I-Design planner 的模型與輸入交接

此頁記錄 planner interface、模型 compatibility patch 與 raw-scene adapter；不是 MetaFind 論文對 200 個場景的定義。原始 I-Design 與 canonical annotation 不由此工具或測試修改。

## 模型設定

[第 04 個 compatibility patch](../setup/patches/idesign-04-configurable-model.patch) 在 01–03 後套用，把 `agents.py` 兩個 filter、`corrector_agents.py` 和 `refiner_agents.py` 各一個 filter，改為讀取必填環境變數：

```bash
export METAFIND_IDESIGN_MODEL=gemma-4-12B-it
```

值必須和 `OAI_CONFIG_LIST.json` 的實際 `model` 及 endpoint served model ID 一致。patch 本身沒有預設模型，缺少或空字串會明確拒絕；不能用 Qwen／GPT 假名表示 Gemma。專案 wrapper 負責依既有「all LLM roles to Gemma」[決議紀錄](../workflow/DECISION_LEDGER.md)（2087–2088 行，2026-08-30 的工作紀錄，並非論文或原對話逐字）設定此環境變數及相同的 config record。原始 GPT 名稱的 Python 變數只是 upstream symbol，實際送出的 config model 必須是真實名稱。

04 保留 temperature、top_p、cache_seed、timeout、JSON response_format 及 02／03 的 graph normalization／bounded correction 行為。02／03 是先前已記錄、會影響 scenes 的 compatibility changes，不是 Gemma 效果已驗證的證據。03 的 correction bounds 也不等於整個 planner 有時間上限：upstream backtracking 另有迴圈，外層執行仍需 timeout 與失敗紀錄。

[CPU 測試](../tests/pipeline/test_idesign_model_config.py) 從本機 git object 讀 pinned I-Design commit `7bc891c72e45f36f6461f5848c3c052723faede4` 的原 bytes 到臨時目錄，實際依序套 01–04，載入三份真實 config modules。AutoGen 的 config file filter／type definitions 使用 stub；未建立 agent、未接觸 endpoint，也未證明 Gemma 的 JSON 或 scene quality。現有 upstream 模組確實設定 `response_format={"type":"json_object"}`；endpoint 是否支援及實際強制 JSON，必須另有運行證據。

## Raw scene graph 到 scene request

[idesign_generate.py](../tools/idesign_generate.py) 目前 planner 工作鏈為 `create_initial_design → correct_design → refine_design → create_object_clusters → backtrack → to_json`。原始 upstream `IDesign.py:311,398` 的最終 JSON 是平坦 list，含物件 slots 和六個 room priors；不是 [compose](SCENE_COMPOSITION.md) 的完整 tensor inputs。

[idesign.py](../metafind/scene/idesign.py) 已實作以下交接，不啟動 planner、backbone 或 renderer：

1. 讀取／hash `scene_graph.json`、planner sidecar、explicit query mapping 與 base request。精確分離 `south_wall、north_wall、east_wall、west_wall、middle of the room、ceiling` 六個 priors，保留為 room metadata，不把它們當檢索 query 或 ESSGNN 已放置資產。`utils.get_room_priors()` 的 priors 含零 thickness 且沒有物件 placement；adapter 驗證其幾何符合 sidecar 房間尺寸。若 sidecar 有 `scene_graph_sha256`，必須吻合。
2. 每個真正物件必須有 unique `new_object_id`、有限 `position`／`rotation.z_angle`、正 `size_in_meters` 及合法 references。缺 position 拒絕整個 scene，不以 filter drop 成較小場景。實際物件數與 room-prior 數分別寫入 provenance；保留完整 sidecar，不能單靠 requested 與 final count 差值推論 pruning 原因。
3. 此入口只處理新房間生成，明確產生 `initial_graph={"room_id": sidecar.scene_id, "nodes": []}`。所有 planned slots 即使有座標仍是待檢索的 queries。base request 禁止自帶 `initial_graph` 或 `queries`。若要輸入真實已存在 assets，須另行建立完整 raw scene request；此 adapter 不由 slot name 推定 UID。
4. query 順序沿用原 JSON 物件順序並凍結，與 upstream `retrieve.py:102` 的遍歷一致。不要未經記錄改成座標排序或 backtracking 的 topological order。每個 query 保留完整原 slot，另帶顯式文字與其實際 query-backbone 編碼。
5. 呼叫者必須提供精確覆蓋所有 slots 的 `slot_id → {text?, images?, pointcloud?}`，至少一個模態且不能包含 prior／extra ID。沒有文字 template 或模態預設；不能先用尚未取回的資產 annotation 製造 query，也不能把 planned style/material 作為取回節點的真實語義。image 路徑保持明確順序，pointcloud 須由呼叫者預先準備；實際內容由下游 raw encoder 驗證／編碼。
6. room dimensions 取原 scene spec／sidecar，和 scene_id 綁定。六個 priors 不提供渲染材質，故 [placement](SCENE_PLACEMENT.md) 的 `surfaces`／camera／light 設定仍需呼叫者明確提供；`surfaces=[]` 可以只保存 assets，不推定 floor/wall 材質。沒有 render config 只做 `.blend`。

adapter 保存四份來源 JSON 的 path/hash/content、slot／query 順序、原始 sidecar 中的 planner 模型／revision／patch 記錄、房間尺寸與 priors，寫入 `provenance.idesign_adapter`。它凍結來源宣告，不獨立证明 sidecar 的 planner 宣告真實。base 中模型、gallery、asset texts、semantic cache 的相對路徑以 base JSON 所在目錄解析；images／pointcloud 的相對路徑以 query mapping 所在目錄解析。JSON 重複 key、缺檔、不完整 slots、錯 room 身分或衝突 base 欄位在發布前拒絕。輸出檔不得已存在，完成後才原子發布。

```bash
python -m metafind.scene.idesign \
  --scene /path/to/scene_graph.json --sidecar /path/to/sidecar.json \
  --query-modalities /path/to/query_modalities.json \
  --base-request /path/to/base_request.json --out /path/to/new-request.json

python -m metafind.scene.prepare --request /path/to/new-request.json \
  --out-dir /path/to/new-composition-bundle --device cpu
```

Python API 為 `build_request(scene_path, sidecar_path, query_modalities_path, base_request_path, out_path)`。`query_modalities.json` 例如：

```json
{
  "chair_1": {"text": "呼叫者提供的完整 query"},
  "table_1": {"images": ["photos/a.png", "photos/b.png"], "pointcloud": "prepared/table.npz"}
}
```

`base_request.json` 提供 [prepare.py](../metafind/scene/prepare.py) 要求的共同設定，沒有隱含 model／gallery：

```json
{
  "schema": "metafind.scene_request.v1",
  "provenance": {"source": "這批模型與資料的來源"},
  "stage1_record": "s1_record.json", "stage2_record": "s2_record.json",
  "variant": "full", "gallery_registry": "gallery_registry.json",
  "gallery_ids": ["明確 gallery UID"], "asset_texts": "asset_texts.json",
  "semantic_cache": "semantic_cache.json", "mode": "iterative", "use_layout": true
}
```

`gallery_ids` 也可明確為 `"all"`；`semantic_cache` 是否必要由實際 S2 layout branch 的下游契約驗證。本 adapter 不產生 room render config：placement 的 `room_id` 取 sidecar.scene_id、`dimensions` 取 sidecar.room_dimensions，`provenance` 綁定 sidecar；`surfaces` 必須另行明確提供，`[]` 僅表示不建立房間表面。沒有 camera／light 設定時保持無 render，不補成正式評估視角。

[CPU adapter 測試](../tests/eval/test_idesign_scene.py) 包含原始 slots／六 priors、不同 JSON 目錄的路徑解析、真實下游 request schema／raw query loader，以及 fail-before-publication；使用 tiny encoder stub，不載模型，不產生真實場景評分。
