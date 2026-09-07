# 測試導覽

`tests/` 依資料流分組。測試驗證程式行為與已記錄契約；通過不代表完整復現論文、真實模型分數正確，或某個 artifact 已通過 promotion gate。只有專案明訂的 gate 能決定 promotion。

2026-09-08 補上 ESSGNN 單層的獨立 NumPy forward／有限差分梯度、no-layout／Full × pinned／ratio λ 的真 checkpoint 交接，以及 compose→placement 的 slot 數值契約。最新完整 CPU suite **1,599 passed** 與真場景驗證見 [本輪交付](../docs/REPRODUCTION_SCENE_REVIEW_20260908.md)，真實模型訓練見 [訓練審查](../docs/REPRODUCTION_TRAINING_REVIEW_20260908.md)；大型模型執行不包含在一般 pytest suite 中。

| 目錄 | 覆蓋內容 | 執行需求與界限 |
| --- | --- | --- |
| [data/](data/) | 點雲取樣與顏色、render 幾何／完成性、annotation schema／修復與隔離、文字序列化／編碼快取、observation、Objaverse／ProcTHOR splits、scene graph 與 semantic edge | CPU；主要使用小型陣列、暫存 mesh／檔案及假模型。`test_annotate.py` 的 LVIS vocabulary 檢查會讀本機 metadata。舊 annotation 契約與 v10 分檔保留，不能混稱為同一個 prompt。 |
| [models/](models/) | ULIP checkpoint key／shape、backbone 凍結與分離、fusion／mask token／image token、loss、DualTower layout residual、ESSGNN 幾何性質、Stage 1／2 協定解析、gradient checkpointing | CPU；小型網路與 synthetic checkpoint。PointBERT block 測試需要已安裝 `timm`；不下載或載入完整 pretrained 模型。數學測試只覆蓋其明示的架構與精度。 |
| [train/](train/) | Stage 1／2 資料組批、optimizer／LR、checkpoint 完整還原、有效設定／run identity、Stage 2 room scope／input provenance、preload 與 query partner | CPU；多數使用暫存 artifacts。`test_preload.py`、`test_query_partner.py` 含本機 cache 對照；`test_run_identity.py` 的 recipe key 檢查會讀現有 resolved protocols。 |
| [eval/](eval/) | 七模態檢索、UID 正解／pool 對齊、float64 cosine／悲觀 ties、streaming rank、gallery／checkpoint identity、Stage 2 overlay、custom observation protocol／三列 runner、Text2Shape metric 對照 | CPU；固定小型數值例子、負向注入與暫存 checkpoints。custom CLI 測試執行真正 encoding／fusion／scoring，但替換 pretrained 載入邊界；不產生論文比較分數。 |
| [pipeline/](pipeline/) | 資料工具寫入安全、文字預檢、G4 gallery freeze 與 promotion 消費、shell chain 失敗傳播、runlog 狀態與 provenance | CPU；寫入限定測試暫存目錄。shell chain 使用 stub Python 及隔離資料根目錄。G4 測試是 gate 程式的回歸測試，不是對真實索引執行 G4。 |
| [gpu/](gpu/) | CPU generator 驅動 CUDA modality mask、CUDA ESSGNN 等變性 | 需要 CUDA，沒有可用 CUDA 時會 skip。必須獨立執行；GPU 正在跑 annotation／訓練時不要執行這組。 |
| [hooks/](hooks/) | Claude research-authority hook 對模擬 Write／Bash payload 的判定 | 需要本機 hook 或 `METAFIND_GUARD` 指定的候選檔；缺少時 skip。測試把命令當 JSON 送給 hook，不執行 payload 中的寫入命令；此 hook 不保護 Codex。 |

本輪新增的重點回歸：`train/test_stage2_gallery_sources.py` 驗證 producer／consumer source binding、來源替換與零有效批次；`eval/test_scene_composition.py` 使用真 tiny ESSGNN 驗證逐次 context 改變結果、配置／權重完整性與 CLI；`eval/test_text2shape_eval.py` 驗證分批上游指標及 dense allocation 邊界。這些不需要真 GPU 模型，也不證明 Table 2 品質。

接續新增：`data/test_semantic_provenance.py` 的實際 text encoder／精確文字 binding、`train/test_stage2_temperature.py` 的父子 tau 還原、loss scalar／gradient oracle，以及 `eval/` 的 IDesign adapter、raw prepare、no-layout replay 和 Blender placement。`pipeline/test_idesign_model_config.py` 使用 pinned upstream git bytes 在暫存目錄實際套四個 patches；它不修改原 upstream、不呼叫 LLM。

`pipeline/test_status.py` 用 `/proc` fixture 與真 CLI subprocess 核對實際 invocation、跨 corpus／v10 模式、缺失或損壞檔案、含空格路徑與唯讀性。`eval/test_scene_pipeline.py` 另有 full-layout 路徑：真 tiny OpenCLIP、SG2 修復／耗盡、缺 pair 拒發布、完整 checkpoint export／loader、ESSGNN 與 iterative retrieval。大型模型／LLM 邊界仍替換，不能將此測試稱為正式預訓練模型實驗。

`eval/test_scene_placement.py` 會在找到本機 Blender 4.2.1 時執行 tiny **CPU** subprocess：64×64／2 samples、獨立 instance、重新開啟 `.blend` 與實際 vertex oracle；缺 binary 才 skip。它沒有使用 GPU，也不是完整模型場景的品質驗證。實際頂點測試特別覆蓋旋轉非盒形資產，避免舊局部 bbox 近似自我驗證。

## 執行

從 repository 根目錄執行。以下環境設定避免 pytest plugin 自動載入、bytecode／pytest cache 寫入及模型下載；不會啟用 CUDA。

```bash
conda activate MetaFind
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
export HIP_VISIBLE_DEVICES=''
# 本機 retired pyrender tests 使用 EGL；CUDA 隱藏不等於 OpenGL 已改 CPU。
# 以下固定本機已驗證的 Mesa llvmpipe，其他機器須使用其實際 Mesa vendor JSON。
export PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

# 一般 CPU suite：明確排除 GPU 與 Claude-only hook。
python -m pytest tests -q -ra -p no:cacheprovider \
  --ignore=tests/gpu --ignore=tests/hooks

# 單一領域／單一回歸。
python -m pytest tests/eval -q -ra -p no:cacheprovider
python -m pytest tests/train/test_checkpoint_roundtrip.py -q -p no:cacheprovider

# 查看此次環境實際收集的案例；函式數不等於 parametrize 展開的案例數。
python -m pytest tests --collect-only -q -p no:cacheprovider \
  --ignore=tests/gpu --ignore=tests/hooks
```

本機環境的 Python 亦可直接使用 `/home/kyzen/miniconda3/envs/MetaFind/bin/python`。pytest 暫存檔預設位於系統暫存目錄。若使用 `--basetemp`，請指定本次測試專用目錄；pytest 會清除該目錄既有內容。

GPU 空閒時，獨立執行：

```bash
# 不沿用上方 CPU suite 的 CUDA 隱藏設定；有排程器指定 GPU 時保留它的配置。
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
python -m pytest tests/gpu -q -ra -p no:cacheprovider
```

驗證 Claude hook 或尚未安裝的候選 hook：

```bash
python -m pytest tests/hooks -q -ra -p no:cacheprovider
METAFIND_GUARD=/absolute/path/to/candidate.py \
  python -m pytest tests/hooks -q -ra -p no:cacheprovider
```

裸跑 `python -m pytest tests` 會收集 GPU 與 hook 組；有 GPU 時不會自動為其他工作避讓。CPU、CUDA 與 hook 結果需分別報告，skip 不算 pass。

本輪整合首次明確隱藏 CUDA 後，抓到 `test_renders.py` 兩個 metadata／stale-sidecar 測試原本會呼叫 live n04 的 OPTIX renderer；已改用受控 PNG 接縫，仍執行真 `process_one`／`is_complete`。舊 PyRender 幾何圖像測試保留真 rasterization，最終整合固定 Mesa llvmpipe，並讀取 GL_RENDERER 核對。早期名為「CPU suite」但未固定 CUDA／EGL backend 的 log，不應被當成沒有使用 GPU 的證據。

## 本機資料與副作用

- 資料位置遵循 `metafind.paths`；`METAFIND_DATA` 必須在啟動 Python 前設定。metadata vocabulary 檢查需要 `paths.LVIS_METADATA`；recipe key 檢查需要 `paths.OUTPUTS` 下的三個 Stage 1 resolved protocol JSON。
- preload／partner 對照會讀取現有 embeddings、pointclouds、annotations 與 `splits.json`。缺少其檢查所需的初始 cache 時會 skip；不會自動補建資料。完整 suite 因此不能宣稱完全不依赖本機資料。
- 測試可以在 `tmp_path` 等暫存目錄建立小型 checkpoint、npz、JSON、mesh 和 subprocess 輸出。實際模型載入、網路、整批資料編碼、訓練及正式評估不屬於一般 CPU suite。
- mock／AST 測試保留其具體作用：強迫錯誤輸入、檢查參數傳遞、身份拒絕及不可達的 failure path。它們不是實際模型執行的替代證據。

## 整理與維護規則

2026-09-07 整理時保留全部 52 個測試檔，原 basename 不變，移至上列目錄；三個 custom evaluation 測試位於 `eval/`。歷史 audit JSON 中的舊路徑、SHA 與 command 記錄描述的是當時的檔案，不應改寫成新的驗證結果。

本次只移除 `test_annotate.py` 中被後續同名定義遮蔽、AST 完全相同的三個測試及兩個 helper。它們本來就不會增加 pytest 收集案例。修正 ranker mock 未還原及單測依賴前一測試的問題；負向注入與 meaningful assertions 保留。另重寫 degraded-render exclusion 測試，移除吞掉任意例外的寫法，在 encoding 入口驗證實際排除後的 query／gallery 名單，避免失敗 DataLoader 的延遲清理污染後續測試。

新增測試放在對應領域；來源路徑應從 `metafind.paths.REPO` 或檔案位置解析，不依賴 shell 的工作目錄。替換 module state 使用 `monkeypatch` 或 `try/finally` 還原，讓個別測試與整套測試可獨立執行。資料或模型的真實驗證另行記錄來源身份、執行命令、pass／skip／warning 與未驗證範圍。
