# 外部場景評分的驗證與彙總

`metafind.eval.scene_scores` 驗證外部提供的評分與場景產物，保留原始場景分母。它是 **n17／n20 的部分工程元件**，不呼叫 Gemma，不產生提示詞、相機或場景清單，也不實作完整 judge runner、G5 或正式 Table 2。

**PAPER FACT：** [MetaFind §3.3](paper/metafind_source/3experiments.tex#L55) 指定四個維度、1–5 分、200 scenes，評審輸入含 layout 和 rendered views，最後按評審／樣本平均。**已批准 DEVIATION：** [DL-077 第 13 題](../workflow/DECISION_LEDGER.md#L5624) 選 Gemma 作裁判；[D-4](graph/graph_spec.yaml#L174) 不做人工評分，所以 Human 欄固定 `INSUFFICIENT_EVIDENCE`。這些決定沒有提供作者的完整提示詞、視角或 200 場景清單。

以下只說明檔案格式。**所有尖括號內容都是待替換的格式佔位，不是正式評估設定；本文沒有提供可當作實驗結果的分數。** `frozen` 是外部提交者對設定已固定的宣告，驗證器不替代研究決策，也不判定提示詞或模型 revision 字串是否真實。

## 1. 明確提供 judge protocol

`protocol.json` 必須包含以下全部欄位，頂層不得增加其他欄位：

```json
{
  "schema": "metafind.scene_judge_protocol.v1",
  "status": "frozen",
  "dimensions": [
    "overall_aesthetic_and_atmosphere",
    "color_scheme_and_material_choices",
    "scene_coherence",
    "realism_and_3d_geometric_consistency"
  ],
  "score_range": [1, 5],
  "model": {
    "id": "gemma-4-12B-it",
    "revision": "<外部實際記錄的模型版本或身分>"
  },
  "prompt": "<外部明示的完整評審提示詞>",
  "generation": {"<實際生成參數名稱>": "<實際值>"},
  "view_policy": "<外部明示的評審視角選擇規則>",
  "provenance": {"source": "<協定的來源與選定依據>"}
}
```

- 四個維度及順序固定；`score_range` 必須是整數 `[1, 5]`。
- `model` 恰有 `id`、`revision`；兩者為非空字串。`id` 可含路徑／provider 前綴，末段必須為 `gemma-4-12B-it`，遵循 D-8。
- `prompt`、`view_policy` 為非空字串；`generation`、`provenance` 為非空 JSON objects。元件保存其精確內容，不補參數，也不證明 `view_policy` 文字的意圖已被遵守。
- 每個場景實際送評的 camera IDs 必須另外列在 evaluation manifest。不能把 IDesign 的五維 1–10 evaluator 直接除以二或自行映射成這四維。

## 2. 凍結完整的 scene × method 分母

`manifest.json` 頂層恰有以下欄位。`scene_ids`、`method_ids` 必須非空且不重複；`outcomes` 必須恰好覆蓋兩者的笛卡兒積。不能省略失敗場景來縮小分母。

下面的 `scene-A`／`scene-B` 僅為**格式佔位**，不是提供一組評估場景：

```json
{
  "schema": "metafind.scene_evaluation_manifest.v1",
  "status": "frozen",
  "scene_ids": ["scene-A", "scene-B"],
  "method_ids": ["<外部明示的方法ID>"],
  "provenance": {"source": "<完整場景清單、方法及執行的來源>"},
  "outcomes": [
    {
      "method_id": "<外部明示的方法ID>",
      "scene_id": "scene-A",
      "status": "complete",
      "composition": {"path": "<composition.json>", "sha256": "<SHA256>"},
      "placement": {"path": "<placement.json>", "sha256": "<SHA256>"},
      "result": {"path": "<Blender result.json>", "sha256": "<SHA256>"},
      "judged_camera_ids": ["<實際送評的camera_id>"]
    },
    {
      "method_id": "<外部明示的方法ID>",
      "scene_id": "scene-B",
      "status": "incomplete",
      "stage": "<實際失敗階段>",
      "reason": "<實際未完成原因>"
    }
  ]
}
```

各 outcome 恰有所示欄位。`incomplete` 的 `stage`、`reason` 必須非空；它不接受分數。`complete` 的要求是：

1. composition 的 schema 為 `metafind.scene_composition.v1`、status 為 `complete`，且 `room_id == scene_id`。
2. [placement reader](../metafind/scene/placement.py#L278) 通過；其 composition snapshot／SHA 與提供的 composition 完全相同。這也驗證凍結 GLB／annotation 的 bytes、slot 及來源快照。
3. Blender result 為 completed `metafind.scene_placement.v1`，綁同一 placement SHA；instance 順序、asset、slot、room、render config、implementation 及 CPU device 一致。
4. 完整 render camera 清單／順序與 render config 相同；所有 PNG 的 bytes、格式及 resolution 通過。`.blend` 也驗 SHA，但本元件不打開它重算幾何。
5. `judged_camera_ids` 明示非空且無重複，必須都存在於已渲染的 cameras。清單順序也是 score input identity 的一部分。

上述 artifact references 恰有 `path`、`sha256`；SHA 為 64 位小寫十六進位。manifest 中的相對路徑相對於 `manifest.json`；placement/result 內部的相對路徑沿用各自既有契約。

**方法 ID 是外部宣告。** 元件將它與這個 manifest 的精確產物綁在一起，不藉 ID 猜測 checkpoint、重播模型或判斷方法是否符合論文。不同方法的上游 model／gallery／request 身分仍須由其實際執行證據支持。

## 3. 提交真實外部評分或評分失敗

`records.json` 頂層恰有 `schema`、`protocol_sha256`、`manifest_sha256`、`records`。後兩個 SHA 綁前述 JSON 的**原始檔案 bytes**，包括空白；可用 `sha256sum protocol.json manifest.json` 取得。每個 method／scene 最多一筆，且只能指向 complete outcome。

```json
{
  "schema": "metafind.scene_score_submission.v1",
  "protocol_sha256": "<protocol.json的SHA256>",
  "manifest_sha256": "<manifest.json的SHA256>",
  "records": [
    {
      "method_id": "<外部明示的方法ID>",
      "scene_id": "scene-A",
      "status": "scored",
      "inputs": {
        "composition_sha256": "<composition.json的SHA256>",
        "placement_sha256": "<placement.json的SHA256>",
        "result_sha256": "<Blender result.json的SHA256>",
        "renders": {"<實際送評的camera_id>": "<該PNG的SHA256>"},
        "judged_camera_ids": ["<實際送評的camera_id>"]
      },
      "scores": {
        "overall_aesthetic_and_atmosphere": "<替換成實際numeric分數，非字串>",
        "color_scheme_and_material_choices": "<替換成實際numeric分數，非字串>",
        "scene_coherence": "<替換成實際numeric分數，非字串>",
        "realism_and_3d_geometric_consistency": "<替換成實際numeric分數，非字串>"
      },
      "raw_response": {"path": "<原始外部評審response檔案>", "sha256": "<SHA256>"}
    }
  ]
}
```

`scores` 必須恰有四維，每值是 `[1,5]` 的有限 JSON number；接受整數或小數，拒絕 bool、字串、null、NaN／Infinity。**上面的字串佔位會被拒絕，不能直接執行。** `raw_response` 必須指向非空檔案，路徑相對於 `records.json`。它只綁定原始 bytes；本元件不解析回應來推導分數，也不證明提交者抄錄正確或 Gemma 曾經執行。

評分失敗時，同一筆保留 `method_id`、`scene_id`、`inputs`，其餘欄位改成：

```json
{
  "status": "failed",
  "reason": "<實際評分失敗原因>"
}
```

這是欄位替换示意，必須與上述身分／inputs 合併；failed record 不可有 `scores` 或 `raw_response`。未收到評分且也未收到失敗紀錄的 completed scene，可以不出現在 `records` 中，之後明確計為 `missing`，不補零。

## 4. 執行 CLI

先建立新的輸出父目錄，準備好以上三個外部檔案，再執行：

```bash
python -m metafind.eval.scene_scores import \
  --manifest /absolute/path/manifest.json \
  --protocol /absolute/path/protocol.json \
  --records /absolute/path/records.json \
  --out /absolute/path/new-output/imported_scores.json

python -m metafind.eval.scene_scores aggregate \
  --scores /absolute/path/new-output/imported_scores.json \
  --out /absolute/path/new-output/summary.json
```

所有輸出拒絕覆寫；不更新 canonical data、checkpoint 或 annotation。第一次 import 在全部驗證通過後才發布。aggregate 重新驗證來源 JSON、凍結資產、Blender／PNG／response bytes 和元件 implementation SHA；來源移動或內容改變時應重新建立明示的 inputs，不會自動重綁舊分數。此模式保留外部依賴，不是獨立可攜的封存。

Python APIs：[scene_scores.py](../metafind/eval/scene_scores.py)。

- `import_scores(manifest_path, protocol_path, records_path, out_path) -> Path`
- `load_scores(path, *, expected_sha256=None) -> dict`：重新驗證，拒絕輸出被手改或來源漂移。
- `aggregate_scores(path) -> dict`：重新驗證後回傳彙總；寫檔由呼叫者／CLI 負責。

## 5. 分母與輸出界線

每個 method 都沿用完整 `scene_ids`，輸出：

| 欄位 | 意義 |
|---|---|
| `n_total` | 外部宣告的 scene 數，不預設為 200 |
| `n_complete`／`n_incomplete` | 場景與所需 render 產物是否完成 |
| `completion_rate` | `n_complete / n_total` |
| `n_scored` | 已完成且有四維合法分數的場景數 |
| `n_judge_failed` | 場景完成，但外部評分明確失敗 |
| `n_missing_scores` | 場景完成，但未收到分數／失敗紀錄 |
| `mean_over_scored` | 只對有分數的場景逐維算 arithmetic mean；無分數時為 `INSUFFICIENT_EVIDENCE` |
| `mean_over_complete` | 只有所有 completed scenes 都有合法分數時才可用；否則四維皆 `INSUFFICIENT_EVIDENCE` |
| `human` | 四維均為 `INSUFFICIENT_EVIDENCE`，遵循 D-4 |
| `failures` | 保留場景／方法、未完成階段與原因、judge failure 或 missing 狀態 |

只有每個宣告場景都完成且有分數，該 method summary 才標 `complete`；這仍僅是輸入／評分覆蓋狀態，**不是完整論文復現或正式協定通過**。`mean_over_complete` 排除未完成場景，存在已記錄的選擇偏差；不得省略完成率而單獨拿來對照論文。summary 綁 imported score artifact、原始 submission、manifest 與 protocol SHA。

## 6. 驗證範圍

[測試](../tests/eval/test_scene_scores.py) 使用既有 tiny 真 GLB fixture，經真 `prepare_placement`／`load_placement`。外部 Blender result、PNG 與 judge response 是**明確標示的人工 fixture**；沒有 LLM、GPU 或 Blender 執行，也沒有產生可報告的場景品質分數。覆蓋多方法分母、缺分／失敗、非法分数、產物／視角身分、來源漂移、實際模組 CLI 與拒絕覆寫。

正式 judge prompt、200-scene 選取、評審視角和執行證據仍需另行確立。本元件不補齊這些研究設定，也不主張真模型／真場景已經評分。
