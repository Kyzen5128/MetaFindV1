# 場景跨模組 CPU smoke（2026-09-07）

[test_scene_pipeline.py](../../tests/eval/test_scene_pipeline.py) 將實際資料結構依序傳過：

1. I-Design raw scene／sidecar，加上明確的逐 slot text／image／PC mapping。
2. `scene.idesign.build_request`，保留原 slot 順序並凍結來源 hashes。
3. `scene.prepare.prepare`：讀取真實的 fixture protocol／S1／S2 checkpoint records、驗證 embedded metadata 與 bytes、還原模型、讀 promoted gallery、編碼 raw observations、匯出 query model，並以實際 manifest replay 驗證。
4. `scene.compose.run_manifest`，產生逐步檢索與 final graph。
5. `scene.placement.prepare_placement`／`run_placement`，凍結兩個不同的 static GLB 與 annotation，執行真 Blender、儲存 `.blend` 及 PNG。

**替換的接縫只有兩類：** 大型 ULIP backbone constructor／output width 換成 deterministic 2D frozen CPU fixture；實際 gallery encoder identity helper 換成明確的 fixture assertion。`custom_models.load_stage1`、Stage 1／2 metadata／sha 與 partial-state loaders、promoted-index loader、request／prepare／export／composition／placement 都未被 stub。Protocol 路徑由 pytest monkeypatch 指向 `tmp_path`，不讀寫 canonical corpus。Synthetic initializer 與 gallery encoder identity 不能用來證明真 ULIP-2 相容性。

使用真 `no_layout` child；full ESSGNN 語義路徑由其他測試涵蓋。此測試不執行 I-Design planner／LLM、不訓練模型、不評量 retrieval quality，也不是完整 MetaFind／Table 2 實驗。

上述是第一個 no-layout／Blender case 的範圍。2026-09-07 接續新增同檔第二個 **full-layout** case：真 tiny OpenCLIP tokenizer／forward／identity、實際 SG2 repair／degraded cache、缺 pair 拒發布、真 S1／S2 loader 與 manifest、ESSGNN 和三步 iterative retrieval。LLM 邊界仍為受控回應，沒有呼叫真 LLM；此 case 不再執行 Blender。成功 relation vector 與耗盡後 checkpoint 的 missing-edge token 會導致第三步 query embedding 差異，且同權重 layout-off 為另一個明確 counterfactual。最新整合證據見 [正式執行路徑審查](../REPRODUCTION_RUNTIME_REVIEW_20260907.md)。

斷言覆蓋：

- Reverse mapping／gallery storage order 不改變 query slot order 或明確指定的 gallery ID order。
- 原 query text、影像讀取、10000×6 PC 資料、S2 mask tensors 與來源 hashes 穿過真實 loader／bundle。
- 手算融合向量導致第一個 slot 選 A、第二個選 B；canonical retrieved text 與 support pair 保留。
- 兩個不同 GLB 分別產生 2／1 個 mesh objects，防止選中 asset 的交接被忽略。
- Blender 使用 CYCLES／CPU／1 thread／2 samples／64×64；實際 PNG 非均勻，blend／render bytes 符合結果 hash。
- 重新開啟儲存的 `.blend`，直接讀 mesh vertices，在不呼叫 placement bounds helper 的情況下核對兩個 slot 的 asset ID 與 world-space bounds。
- 輸入 gallery／observation／GLB hashes 保持不變。

2026-09-07 第一個 no-layout case 初次執行結果：**1 passed，1.49s**，不是 skip。使用已安裝 Blender 4.2.1；只有 executable 不存在或不可執行時才 skip，不會把渲染或幾何錯誤改成 skip。該次產物留在 pytest 的暫存目錄；後續持久副本另記如下。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' \
/home/kyzen/miniconda3/envs/MetaFind/bin/python -m pytest -q -p no:cacheprovider \
  tests/eval/test_scene_pipeline.py
```

完整 regression 由主流程另行記錄；這個 smoke 只證明上述 fixture 的跨模組交接與實際 CPU placement/render。

前輪完整 suite（1,514 passed）已再次執行此 smoke，並保存 [PNG](scene_pipeline_20260907_artifacts/view_000.png)、[`.blend`](scene_pipeline_20260907_artifacts/scene.blend)、[結果](scene_pipeline_20260907_artifacts/result.json)、[獨立幾何量測](scene_pipeline_20260907_artifacts/blend_geometry.json) 及 [來源／hash 清單](scene_pipeline_20260907_artifacts/evidence.json)。這是同一套 synthetic CPU 證據的持久副本；原始輸入 paths 保留暫存 provenance，不改稱真實論文場景或可攜的完整模型 bundle。
