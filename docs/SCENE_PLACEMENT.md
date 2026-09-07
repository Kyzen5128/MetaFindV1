# 從 composition 到 Blender 場景

[placement.py](../metafind/scene/placement.py) 將完成的 [composition](SCENE_COMPOSITION.md)、原始 GLB、資產 annotation 與明確房間設定凍結成 bundle；[blender_place.py](../metafind/scene/blender_place.py) 在獨立 CPU Blender process 放置資產、保存 `.blend`，並在提供渲染設定時產生影像。兩者不呼叫 planner／retrieval／LLM，也不改 annotation、原始 GLB 或 canonical corpus。

**PAPER FACT：** [MetaFind §3.3](paper/metafind_source/3experiments.tex) 的場景評估使用 layout 與 rendered views。單獨完成此工具不代表完成 Table 2 的 200 scenes、judge 或 human study。

**UPSTREAM FACT：** [I-Design §3.4–3.5](paper/idesign_source/main.tex) 將取回資產調整到 scene graph 的 bounding box，使用 Blender 視覺化。上游 `place_in_blender.py:77–86,121–147` 提供 bbox center、三軸縮放及 `slot_yaw + 180°`；其腳本沒有完整 camera/light/render 入口，且房間尺寸寫死。此 bridge 只沿用 placement 語義，另行要求明確的 room/render config。

**IMPLEMENTATION CHOICE：** U-18 的 slot 位置、rotation、size 原樣保留；annotation 僅綁定 retrieved asset 身分與語義來源，不覆寫 slot 尺寸。每個 instance 分別 import，以本次新增 objects 集合和專用 parent root 操作，不依賴檔名解析、不跨資產 join。room surface、camera 和 light 都來自輸入設定。

## 入口與產物

```bash
python -m metafind.scene.placement prepare \
  --composition /path/to/composition.json \
  --assets /path/to/assets.json --room /path/to/room.json \
  --render /path/to/render.json --out-dir /path/to/new-placement-bundle

python -m metafind.scene.placement run \
  --manifest /path/to/new-placement-bundle/placement.json \
  --out-dir /path/to/new-blender-output
```

省略 `--render` 只做 placement／`.blend`，不產生影像。`--blender /absolute/path/to/blender` 可明確指定 executable；否則只在現有 `render_blender.BLENDER_INSTALL` 中解析唯一的 Blender，找不到就拒絕，不安裝依賴。既有 n04 的 `render_asset()` 會 normalize 整個物件、產生 asset orbit 並使用 GPU，不能用於此 room placement。

Python API：

```python
prepare_placement(composition_path, asset_manifest_path, room_config_path,
                  out_dir, render_config_path=None)  # -> placement.json Path
load_placement(manifest_path)                        # 驗 hash、身分、設定
run_placement(manifest_path, out_dir, blender=None)   # -> result.json Path
```

bundle 保存原始來源 JSON bytes/hash、完整 planned slots、selected asset IDs、複製的 GLB/annotation bytes/hash 及兩支 implementation source hash。原始輸入日後改動不影響 bundle；更改實作需要重新 prepare。每次 Blender import 都讀經 hash 驗證後的私有暫存副本。

所有輸出目錄必須為新目錄。`placement.json`／`result.json` 僅在工作完成後原子發布；失敗保留證據與 `blender.log`，不產生成功 result。`result.json` 包含每個 instance 的 imported mesh names、縮放、完整 transform、slot frame／world frame 實際 bounds、Blender 版本、`.blend`／render hashes。來源 GLB 的 materials 保留；GLB 自帶 camera/light 被停用，避免覆蓋明確 render config。

## 明確輸入格式

`assets.json` 的 key 是資產 UID；一個 UID 可被多個 slot 使用，每次都是獨立 instance。path 可為絕對路徑，或相對於 `assets.json`。

```json
{
  "provenance": {"source": "此批 raw GLB 與 annotation 的可追溯來源"},
  "assets": {
    "selected-uid": {
      "mesh": {
        "path": "raw.glb", "sha256": "實際 hash",
        "frame": "raw_gltf_y_up"
      },
      "annotation": {"path": "annotation.json", "sha256": "實際 hash"}
    }
  }
}
```

只支援 self-contained static glTF 2 GLB；外部 buffer/image URI、animation 或 skin 會拒絕，不能默默省略依賴或選擇動畫 pose。annotation 若有 `uid`，必須一致。`frame` 必須明確宣告為未經本專案 frame correction 的原始 GLB；hash 固定 bytes，但不能獨立證明提供者對 frame 的宣告正確。

`room.json` 需要 `room_id` 與 composition 一致，`dimensions=[length,width,height]` 為原 planner 房間公尺尺寸；`surfaces` 是明確頂點構成的面，`[]` 表示不建立房間表面。不能從資產外接盒反推房間大小。

```json
{
  "room_id": "實際 room id",
  "dimensions": [10, 8, 4],
  "provenance": {"source": "原始 planner room spec"},
  "surfaces": [{
    "id": "floor",
    "vertices": [[0,0,0], [10,0,0], [10,8,0], [0,8,0]],
    "rgba": [0.6,0.6,0.6,1], "roughness": 0.7
  }]
}
```

以上尺寸和 floor 是格式範例，並非論文預設。各面採標準非金屬 Principled material，顏色／alpha／roughness 明確提供；不新增未列出的牆、天花板或材質。

`render.json` 以下是 **CPU smoke fixture** 的完整例子，不是 Table 2 評估設定：

```json
{
  "provenance": {"source": "test-only CPU render"},
  "engine": "CYCLES", "device": "CPU", "threads": 1,
  "samples": 2, "seed": 17, "resolution": [64,64], "transparent": false,
  "world": {"color": [0.5,0.5,0.5], "strength": 0.2},
  "color_management": {
    "view_transform": "Standard", "look": "None", "exposure": 0, "gamma": 1
  },
  "cameras": [{
    "id": "test-top",
    "matrix_world": [[1,0,0,5], [0,1,0,4], [0,0,1,12], [0,0,0,1]],
    "lens_mm": 32, "sensor_width_mm": 36, "clip_start": 0.01, "clip_end": 100
  }],
  "lights": [{
    "id": "test-area", "type": "AREA", "shape": "DISK",
    "matrix_world": [[1,0,0,5], [0,1,0,4], [0,0,1,12], [0,0,0,1]],
    "energy": 600, "color": [1,1,1], "size": 5
  }]
}
```

使用 Blender Z-up 世界座標與 rigid 4×4 matrix；camera 視線沿 local −Z、local +Y 向上。camera/light 列表、world、color management、解析度、sample count 不補預設，未知欄位拒絕，避免記錄了未消費的設定。現階段只支援 perspective camera、DISK/SQUARE AREA light、CYCLES CPU；固定停用 adaptive sampling 和 denoising，PNG 為 RGBA 8-bit、sRGB 顯示。其他 Cycles 行為沿用所記錄 Blender 版本，並保存於 `.blend`；影像 hash 是實際輸出紀錄，不保證跨硬體 bitwise reproducibility。

## 座標與驗證界線

Blender glTF importer 將 `(x,y,z)` 轉成 `(x,-z,y)`，並對 glTF node transforms 做相同座標轉換。scene `scale_length=1`；接著原始 GLB 使用 `slot_yaw+180°`，將 asset 的完整 bbox center 對齊 slot，按每軸 target/raw extents 縮放。bounds 從 Blender evaluated mesh 的實際頂點計算，包含靜態 morph weights；不用旋轉後可能高估範圍的 local `bound_box` 角點。所有 instance 的原始 hierarchy 與 materials 都保留。

`meshload` 的 `yaw180_about_y@ulip2_frame` 經 Y-up→Z-up 後等於 Z 軸 180°；已校正過的 GLB 再走此入口會重複旋轉，因此明確拒絕該 frame 宣告。normalized pointcloud 不能代替 raw GLB。annotation 的 cm 尺寸和 raw bbox 自身單位都不作 verified physical metres 使用；true front 仍 UNKNOWN，不猜測個別資產 facing。I-Design 本身也記錄 resizing／canonical orientation 的限制，見 [supplementary](paper/idesign_source/supplementary.tex)。

放置後以 slot frame 檢查 bbox center 與三軸尺寸；任意 yaw 的非對稱多 mesh 資產，world AABB 中心不一定等於 slot anchor，因此不以那個錯誤條件移動物件。工具不自動修正碰撞、出界或形變，這些是待評估的場景結果。

[測試](../tests/eval/test_scene_placement.py) 使用真多 mesh GLB，涵蓋原始與凍結來源篡改、錯 UID、frame、room、slot、未知 render config，以及兩個獨立 instance 的 90°／45° 旋轉、實際 bounds、獨立重新開啟 `.blend`、64×64／2-sample CPU render。沒有 canonical 資料、GPU、模型或 LLM 參與。缺 Blender 時 integration tests 明確 skip；不可把 skip 說成已驗證 Blender。

2026-09-08 另執行 [真模型場景鏈](history/REPRODUCTION_SCENE_REVIEW_20260908.md)，以真 Objaverse GLB 完成三個 instance 與 512² CPU render；獨立從原 GLB 與保存的 evaluated vertices 雙向核對 136,925 個頂點，最大差約 4×10⁻⁷ m。這是額外執行證據，不是正式場景品質評分。

獨立 review 發現原有兩個 box-mesh smoke 無法識別旋轉過的非盒形 mesh：raw Y 軸旋轉 45° 的四面體，舊 local-AABB 寫法雖回報寬 2 m、中心置中，保存的實際頂點只有 1.333333 m 寬且 slot-frame 中心偏移 −0.333333 m。修正後新增四面體 regression，從 raw GLB 頂點獨立推導 C 座標轉換、node rotation、actual-vertex scale/center 和 yaw + 180°，再重新開啟 `.blend` 比對全部實際頂點；不以 production helper 或 result 中的自報 bounds 作 oracle。
