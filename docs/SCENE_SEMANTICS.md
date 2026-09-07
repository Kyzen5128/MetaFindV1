# 為明確指定的場景 pairs 補齊 SG2 cache

[semantics.py](../metafind/scene/semantics.py) 為明確指定的資產 UID pairs 生成缺少的關係句子及向量，再供 [raw scene preparation](../metafind/scene/prepare.py) 使用。它不列舉整個 gallery 的 N² 組合，也不寫 canonical n08、annotation 或原始來源檔案。所有輸出必須放在新目錄。

**OBSERVED IMPLEMENTATION：** 重用 `semantic_edges_run.RelationWriter`、`write_one`、`encode_sentences` 和 `build_cache`。因此 prompt、greedy generation、`MAX_ATTEMPTS=2`、parse／validate／repair 和句子向量 L2 normalization 均來自現有 n08。模型未完成兩次真實嘗試，不會自行建立 degraded／None 代替缺少的關係。

**IMPLEMENTATION CHOICE：** 參考 Stage 2 記錄、其 Stage 1 parent 及 n08 source evidence 必須完整且可驗證。訓練 cache 的 `prompt_version`、`llm_model`、`text_encoder_version` 必須與現有 n08 一致；新文字向量的實際 loaded text encoder fingerprint 也必須相同。舊 n08 沒有 source identity 的產物不接受。

**UNKNOWN：** 歷史 n08／Stage 2 記錄只綁定 LLM 的 cache-key model name，沒有歷史 LLM weight bytes。此入口使用現有 n08 的確切 configured model path，核對 model name 和 writer 實際 `model_id`，並在 `record.json` 記錄 `historical_weight_byte_equivalence: "UNKNOWN"`；不能宣稱它證明了歷史與本次 LLM 權重逐位相同。

## 入口

`required_pairs.json` 的格式是明確 UID pair list：

```json
[["asset-A", "asset-B"], ["asset-A", "asset-A"]]
```

相同 UID 可代表兩個使用相同資產的已放置物件。反向 pairs／相同描述會依 n08 的 symmetric description cache key 合併。UID 必須在 `asset_texts.json`；relation prompt 使用 `relation_text_for()` 選擇的精確字串，而非位置、尺寸或其他幾何欄位。

```bash
python -m metafind.scene.semantics \
  --asset-texts /path/to/asset_texts.json \
  --required-pairs /path/to/required_pairs.json \
  --stage1-record /path/to/stage1_record.json \
  --stage2-record /path/to/stage2_record.json \
  --variant full \
  --base-cache /path/to/existing/sem_edge_cache.json \
  --out-dir /path/to/new-scene-semantic-cache
```

預設 `--device cpu`。需要 GPU 時必須明確指定 `--device cuda`；上述命令需要真實模型，CPU tests 沒有執行它們。

`--base-cache` 可省略；提供時先嚴格驗證其來源、NPZ digest、encoder 和三個 cache-key 版本欄位。輸出合併既有 entries 和新關係，重新建立排序後的 row pointers。所有要求均已命中時，不載入 LLM，也不重新編碼已有向量；既有明確 degraded evidence 原樣保留，不偷偷重試。

若有缺少的 pairs，先生成並保存 `sentences.jsonl`，釋放 LLM 後才載入 Stage 1／CLIP。兩者不同時駐留。僅新增成功句子需要編碼；真正耗盡 SG2 repair budget 的 pairs 會保留原因、`degraded: true` 和 `embedding_uri: null`。這是既有 n08 的可見失敗表示，不等於所有關係成功。

## 產物與驗證

新目錄包含：

- `sem_edge_cache.json`：n08-compatible entries、source identity 和 NPZ uri/hash，可填入 scene request 的 `semantic_cache`。
- `sem_edge_embeddings.npz`：排序後的 keys、float32 向量、內嵌 source identity digest。
- `sentences.jsonl`：本次真正嘗試的 SG2 結果／錯誤；base 全命中時為空。
- `record.json`：完成紀錄、輸入及 checkpoint hashes、要求的 pairs→keys、成功／degraded 計數、模型來源驗證狀態和上述歷史 LLM bytes 限制。

發布 cache 前重新核對所有已讀來源 bytes，並以既有 `read_verified_edge_source()` 檢查新 entries／row pointers／向量。失敗保留新目錄中的部分證據，不覆寫它；重新執行須指定另一個新目錄。

[CPU tests](../tests/eval/test_scene_semantics.py) 使用真正的 tiny S1/S2 checkpoint loader、n08 `write_one`／repair／parse／validate／encoding 與 source-proof reader，只替換巨大模型的載入。它們涵蓋錯模型／錯 encoder 拒絕、來源改動、base cache 合併、去重、真實 degraded、LLM 釋放後才建 encoder，以及新 cache 接入 `prepare.semantic_inputs`。這些測試不證明真實 LLM 生成品質或論文 Table 2 分數。

另行的 [2026-09-08 真場景診斷](history/REPRODUCTION_SCENE_REVIEW_20260908.md) 實際在 CPU 呼叫 Gemma 兩次，分別生成 cake↔pot 與兩個 cake instance 的關係；各成功一組、degraded 0，再用核驗過的真 CLIP 編碼，接回 prepare 完成場景。歷史 LLM bytes 仍標 `UNKNOWN`；生成成功不等於句子品質已通過正式評分。
