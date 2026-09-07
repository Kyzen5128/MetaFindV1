# 逐步場景檢索核心

[compose.py](../metafind/scene/compose.py) 實作 [MetaFind §2.7／Algorithm 1](paper/metafind_source/2methdology.tex)：啟用 layout 時，每個 query 都先編碼目前已放置物件的圖，再檢索並把取回資產放入指定 slot。它讀取已編碼的輸入，不下載模型、不呼叫 planner／LLM、不修改 canonical corpus。

**PAPER FACT：** gallery 預先編碼；逐次檢索、放置、更新 graph，後續 query 使用更新後的 layout。
**IMPLEMENTATION CHOICE：** 依已記錄 U-18／U-21 使用 I-Design 的 slot；node semantics 來自取回資產；n07 的 symmetric kNN（k=8）與 support 規則、n08 cache key。缺失關係沿用目前 resolved edge protocol 的 `learned_missing_token`（U-30 議題），對應 [resolve_stage2.EDGE_DECISIONS](../metafind/models/resolve_stage2.py) 與 ESSGNN consumer；graph_spec 的 U-30 registry 舊標記仍為 UNKNOWN，不能據此稱 registry 已 resolved。這不是作者未公開的完整 200-scene 實驗協定。

## Tensor API

```python
from metafind.scene.compose import compose

result = compose(
    tower,                         # 已完整載入權重的 QueryTower，所有子模組 eval()
    gallery_ids=gallery_ids,        # 排序固定且 unique 的 asset IDs
    gallery=gallery,                # floating Tensor (G, D)，唯讀
    asset_node_embeddings=node_vectors,  # asset ID -> Tensor (node_dim,)
    asset_texts=asset_texts,         # asset ID -> {text, relation_text?}
    initial_graph={"room_id": "room-1", "nodes": initial_nodes},
    queries=queries,                # 順序明確，不能把未放置 slot 預載進 initial_nodes
    semantic_cache=semantic_cache,  # n08 cache key -> Tensor (edge_dim,) 或明確 None
    semantic_meta={"prompt_version": 1, "llm_model": recorded_model,
                   "text_encoder_version": recorded_encoder},
    mode="iterative",              # parallel：每個 query 都只讀 G0
    use_layout=True,               # False：同權重、同圖更新，但不加 layout residual
)
```

`initial_nodes` 每項為 `{"asset_id": ..., "slot": ...}`；每個 query 為 `{"slot": ..., "embeds": {"text": tensor, ...}, "query_text": ...}`。`query_text` 可省略，只記錄 query 來源文字，不用於建立 node feature。可用模態的 tensor 是 `(D,)`，缺模態省略或填 `None`。目前支援 pooled image，即 `image_tokens=1`。

已訓練的 `no_layout` ablation 也可使用此 API，但必須明確傳 `use_layout=False`。它的 `tower.cfg.essgnn=None`，不建立 ESSGNN、不要求 `asset_node_embeddings`／`semantic_cache`／`semantic_meta`；這三項可省略。實體 support／adjacency 圖、slot、取回 asset 的 canonical `text` 仍驗證及記錄。輸出與各圖記錄標記 `semantic_status="not_used"`，trace 的 lambda／layout norms 為 `0.0`；這是未使用 layout 的紀錄值，不是模型含有已學得的零 lambda。對沒有分支的模型傳 `use_layout=True` 會拒絕。

這與 full checkpoint 的 `use_layout=False` 對照不同：full 模型保有 ESSGNN 設定，因此仍按原規則驗證 node vectors、semantic metadata 與 trajectory 的 cache keys，只省略 layout residual。

slot 採 I-Design 已放置物件的原欄位，至少包含：

```json
{
  "new_object_id": "chair_1",
  "position": {"x": 1.2, "y": 2.0, "z": 0.4},
  "rotation": {"z_angle": 90},
  "size_in_meters": {"length": 0.5, "width": 0.5, "height": 0.8},
  "placement": {"objects_in_room": []}
}
```

這是 schema 範例，不是預設位置或論文場景。原 slot 的其他欄位也保留。物件關係使用 `object_id`／`preposition`；`on`／`under` 形成無向 support pair，僅在兩端都已放置時生效。其餘 adjacency 按 n07 重新計算，support pair 不重複列入 adjacency。每個呼叫是一個明確房間。

語義 key 共用 `relation_text_for()` 和 `cache_key()`，不能拿 asset ID 自創另一種 key。缺 key 會拒絕，讓呼叫者先準備相應 n08 證據；只有明確 `None` 才代表缺失關係，並傳 `edge_missing=True` 給 ESSGNN，在內部以可學 missing token 替代。沒有在此工具中自動生成新關係句。

關係生成有独立 [scene.semantics](SCENE_SEMANTICS.md) 入口，可只補缺少的 UID pairs 並保留 SG2 修復耗盡證據；compose 本身仍是固定 inputs 的可重播 consumer。

目前 frozen bundle 必須事先包含此次 trajectory 所需的 cache entries。每步在編碼當前 context 時查 key；最後一次放置後也建立完整 `final_graph`，因此最後新增的 pair 仍需 entry，雖然它不再影響本次排名。不能把「未知／尚未準備」自動轉成 `None`。`MissingSemanticPairsError` 回傳實際缺少的 UID pairs／keys；[raw prepare](RAW_SCENE_INPUTS.md) 將其寫成 `required_pairs.json`，供關係準備後重播。取回哪個 asset 是執行中才知道的，不能假設 ProcTHOR 舊 cache 已覆蓋所有新 Objaverse pair。

空 G0 使用 `layout=None`；一個 node 即使沒有 edge 仍執行 ESSGNN。檢索不排除先前取回過的 asset，因 Algorithm 1 沒有指定禁止重複資產；node instance ID 仍必須唯一。cosine 以 CPU float64 計算，同分保持輸入 gallery 順序，此 tie 規則是實作選擇。

## 凍結輸入與 CLI

CLI 使用完整 QueryTower config/state，加上已編碼的 tensor bundle；全部都在 CPU 載入。manifest 形式如下：

```json
{
  "schema": "metafind.scene_composition.v1",
  "status": "frozen",
  "provenance": {"source": "明確記錄此批輸入與模型的來源"},
  "model": {
    "dtype": "float32",
    "config": {},
    "weights": {"path": "query_tower.pt", "sha256": "實際檔案 hash"}
  },
  "inputs": {"path": "composition_inputs.pt", "sha256": "實際檔案 hash"}
}
```

`model.config` 必須是 `dataclasses.asdict(tower.cfg)` 完整內容，不能使用上例空物件。最外層 `DualTowerConfig`、兩個 `FusionConfig` 與存在的 `ESSGNNConfig` 都拒絕缺漏或多餘欄位，避免漏掉 `prefusion_norm` 等設定後靜默使用預設值。只有模型的 `use_layout=false` 才允許 `essgnn=null`，不填入虛構的 node／edge dimensions。`query_tower.pt` 存完整 `tower.state_dict()`；`composition_inputs.pt` 是上方 `compose` 關鍵字參數的 dict，包含 tensor 與 JSON-compatible 值。透過 `torch.load(weights_only=True)` 載入，state dict 必須完整匹配；相對路徑相對於 manifest。來源須記錄 query encoder、gallery encoder、實際使用的 node／edge encoder、文字及 slot 身分；hash 驗證證明 bytes 一致，不追溯證明上游生產者的科學相容性。

此 CLI 僅接受明確標記 `model.dtype="float32"`、所有浮點 state tensor 也為 float32 的權重，拒絕靜默縮窄 float64 等權重。純 tensor API 沿用呼叫者 model 的 dtype；query／node／edge 輸入轉為該 dtype。兩種入口的 cosine 都共用 retrieval 的 float64 正規化，不以 epsilon 改變合法微小非零向量的長度。

```bash
python -m metafind.scene.compose --manifest /path/to/manifest.json \
  --out /path/to/new-composition.json
```

輸出排他發布，不覆寫既有檔；失敗不產生標記 complete 的結果。每步記錄 graph IDs／positions／edge keys、top-5 cosine、選中 asset、query vector、fusion/layout norms 及 `||λ · e_layout||`，並保存完整 final slots。manifest、來源 hash 與本次 compose source hash 一起進 provenance。它不修改傳入 gallery、graph 或 model state。

## 驗證與尚缺交接

[CPU 整合測試](../tests/eval/test_scene_composition.py) 使用真實的小型 QueryTower、fusion、ESSGNN 與手算期望，覆蓋逐步 context 改變檢索結果、singleton／empty context、U-30、cache 缺失、slot 身分、gallery 不變及 CLI 完整權重還原。它們不是實際 ULIP-2／200 scenes 的執行結果。

下面的 model-only exporter 已接入 [原始查詢 exporter](RAW_SCENE_INPUTS.md)；另有 [I-Design JSON adapter](IDesign_INPUTS.md) 與 [GLB placement／CPU render](SCENE_PLACEMENT.md)。2026-09-08 已另跑真 ULIP／Gemma 的三 slot、六候選完整場景鏈，含 iterative／parallel／layout-off 對照，見 [執行證據](history/REPRODUCTION_SCENE_REVIEW_20260908.md)。正式 U-27 的 200 prompts／尺寸／物件數、judge 與 human 評分仍需各自處理；固定 slots 的檢索、已渲染和已正式評分是不同狀態。

## 從既有 S1／S2 records 匯出模型

```bash
python -m metafind.scene.compose export-model \
  --stage1-record /path/to/stage1_record.json \
  --stage2-record /path/to/stage2_record.json \
  --out-dir /path/to/new-query-model --variant full
```

`export_query_model(stage1_record_path, stage2_record_path, out_dir, variant="full")` 產生完整 `query_tower.pt` 和 `model.json`。把 `model.json` 的 JSON 內容直接放入上述 manifest 的 `model` 欄位即可；它不包含 `inputs`。新目錄不得已存在；只有完整驗證、儲存完成後才原子發布 `model.json`。輸出綁定兩份輸入 record 的 bytes/hash、parent/S2 checkpoint hash、輸出 weights hash 與本工具 source hash。

匯出器全程使用 CPU，不建構 ULIP backbone。它復用現有 S2 architecture／parent metadata 驗證，從同一份已驗 hash 的 S1 checkpoint bytes 還原完整 S1 tower，再完整覆寫 S2 query fusion 及該模型存在的 ESSGNN 和 λ；拒絕 tied fusion、未知／不完整模型設定與非 float32 權重。`--variant no_layout` 會匯出沒有 ESSGNN／lambda weights 的真實模型，後續 inputs 必須設 `use_layout=False`。CPU fixture 使用實際既有 loader 驗證 full 與 no-layout 的 exporter → manifest → 逐步檢索，並檢查輸出每個 tensor 與來源一致；這不是對目前真實 checkpoint 的執行宣稱。

## 完整輸入 bundle 的交接與限制

1. 上述 exporter 使用 [gallery_index](../metafind/train/gallery_index.py) 的 checkpoint record／bytes 驗證及 [run_retrieval](../metafind/eval/run_retrieval.py) 的 `load_stage2_over_stage1`／`overlay_stage2_weights(..., fusion_only=False)`。既有 [custom_models.apply_stage2](../metafind/eval/custom_models.py) 僅覆寫 fusion、停用 layout，不能直接作為此核心的模型。
2. [prepare.py](../metafind/scene/prepare.py) 對 Objaverse gallery 重用 `load_promoted_index_for_checkpoint` 與實際 gallery encoder 身分驗證，按明確 UID 順序或 `all` 取向量。ProcTHOR 診斷重用 `verified_stage2_index`，不在此 raw exporter 裡冒充 Objaverse promoted index；兩者 asset universe 不同。
3. ProcTHOR node vectors／edge cache 可沿用 [Stage2Data](../metafind/train/stage2.py) 的 NPZ ID／row-key／hash 驗證規則；它讀目前 `paths.OUTPUTS`，完整建構还會掃描場景檢查 text/edge 一致性，不能拿它無條件載入其他 root 的歷史 snapshot。匯出時保留 n08 原 `text`／`relation_text` records。已 degraded／null-uri 的 cache entry 可明確轉成 `None`，不存在的 key 仍是待準備。
4. Objaverse 新取回資產的 node semantics 並不存在於 ProcTHOR 的 assetId map。[raw exporter](RAW_SCENE_INPUTS.md) 先以 child 綁定的 n08 node／edge 產物核對實際 text encoder 身分，再編碼呼叫者提供的精確 retrieved-asset `text` 並 L2 normalize。不可將 1280-d retrieval text 向量塞進不同來源的 512-d node encoder，也不靠同維度宣稱相容。n08 現可接入明確提供的同一個 backbone；此 exporter 對沒有 producer source binding 的舊 semantic artifacts 仍拒絕。
5. query 的 raw text／image／PC 由還原後的實際 query backbone 編碼，記 source bytes／影像順序與 PC 契約；fully_separate PC 使用 query point path。[I-Design adapter](IDesign_INPUTS.md) 僅提供原 slot＋顯式 modality mapping，不能以 slot 代替 query encoder，也不能把未取回的資產當作 G0。
