# 接續完整檢查與復現修補（2026-09-07）

**目前仍是 PARTIALLY VERIFIED。** 已把兩階段訓練的重要數值／來源缺陷修正，並補上 I-Design 原始 JSON、raw query、語義準備、逐步檢索、GLB 放置與 CPU render 之間的可執行交接。正式 checkpoint 重訓、全量評估與論文 Table 1–3 數值仍未驗證；沒有用小型測試宣稱完整論文復現。

這份文件接續 [第一輪審查](REPRODUCTION_REVIEW_20260907.md)，不改写其原始測試 log／SHA。使用者目前授權修復、重寫及整理；標註維持不動。先以 paper-audit／reproduction-audit 進行唯讀核對，再在已授權範圍修補確認的缺陷；沒有改動 paper source、vendor、CLAUDE.md 或 Claude hooks。

## 1. 新確認的問題與修補

| OBSERVED IMPLEMENTATION／最小反例 | 修補與驗證範圍 | 對舊結果的限制 |
|---|---|---|
| Stage 2 宣告 fixed tau=.5，載入 parent learned tau=.07 後實際變成 .07。 | 分開父 loss 還原與子 loss 初始化；fixed child 保留自身 recipe，learned child 記父 raw scale／clamp／有效值。 | 舊 checkpoint 的 tau 不會因此自動改正；須按原 record／state 判斷及重訓。 |
| Learned scale 被 clamp 時，訓練 log 記的是未 clamp 的 reciprocal。 | Stage 1／2 log 改用該次 forward 回傳的有效 tau。 | 不改歷史 log，不將記錄修正當成模型分數已重跑。 |
| 雙向 loss 接受重複 labels，inverse permutation 留未初始化元素。 | 驗 labels 型別、shape、device、range；雙向要求 bijection。單向合法重複 label 保留。 | trainer 原本用 arange，不能以此 bug 解釋所有舊 Table 1 分數。 |
| 改 node text、保留 relation_text 與舊 node NPZ，再拍新的 Stage 2 input snapshot，仍能通過。 | n08 node／edge source digest 綁精確文字／句子、UID／key rows 與實際 text encoder；Stage2Data 嚴格重核。 | 舊 n08 缺 producer binding 就是未驗證；不能只補 sidecar。明確 legacy 選项仍標 legacy_unbound。 |
| Stage2Data 驗 NPZ hash 後重新開啟路徑，可能消費檢查後被替換的向量。 | node／modern edge 從同一份已驗 hash 的 bytes 載入；legacy edge 也只讀一次。三種檔案替換反例通過。 | 這是實際消費路徑修正；未聲稱歷史 run 曾發生同時替換。 |
| 合法 no_layout checkpoint 在 retrieval／probe 索引 None lambda，或強制產生 S2-on。 | branch metadata 綁 checkpoint，consumer 按實際分支還原。export→manifest→compose 也支持 no_layout，只允許 layout-off。 | full 模型缺 lambda 仍是壞輸入，不以放寬判定掩蓋。 |
| I-Design 文件命令不能 import metafind；partial batch failure 仍 rc=0，空／非法場景要求可偽成功。 | 修 script import/default upstream 路徑、批次輸入预檢、排他 output、incomplete status／非零退出，物件數剝離六個 room priors。 | 未替既有失敗 batch 改標；未執行真 planner。 |
| Gemma OAI config 遇上寫死的四個 Qwen filter，實際模組載入失敗。 | patch 04 讀同一個真實模型 ID，wrapper 依既有 all-LLM Gemma 決議設定；不使用假別名。 | 沒有改真 upstream 或啟動服務；本機模型／endpoint 品質仍需實跑。 |
| 01–04 全套正確套用，逐張 reverse-check 卻把 01 判成未套用。 | 驗證累積 patch 狀態，setup 補剩餘 chain；未知混合修改拒絕而不覆寫。 | 檢查證明程式差異與指定 patches 一致，不證明生成場景符合論文品質。 |
| 旋轉非盒形 GLB 用 local bbox 角點縮放：宣告寬 2m，實際僅 1.333333m、中心偏移 −.333333m，原自驗仍通過。 | 改成 evaluated mesh 的實際頂點，重新開 `.blend` 與獨立 raw-vertex 座標 oracle 比對。 | 這是新 placement adapter 的審查修正，不能拿來解釋尚未經它產生的舊數據。 |

公式詳細鏈見 [Eq.1–8／附錄審查](audit/formula_review_20260907.md)：獨立 scalar InfoNCE oracle 不用 torch CE／normalize 計算期望，並用中央有限差分核對 query／gallery 梯度。保留正文／附錄在 h0、距離、message 與更新順序的矛盾，以及既有 D-14／U-26 選擇；不宣稱這些歧義已由測試消除。

## 2. 已補的場景功能

1. [scene.idesign](IDesign_INPUTS.md)：完整 raw planner scene／sidecar＋顯式模態 mapping → scene request。精確區分六個 room priors，保留 slot 順序，fresh G0 為空；不憑 slot 名稱猜資產 UID。
2. [scene.prepare](RAW_SCENE_INPUTS.md)：verified S1／S2 records、promoted gallery、原始文字／影像／prepared PC → frozen bundle。PC 使用真正 query point path；new node text 和 trained semantic space 嚴格核對，query 與資產 annotation 分開。
3. [scene.semantics](SCENE_SEMANTICS.md)：只為缺少的 UID pairs 重用 n08 prompt／生成／bounded repair／encoding，延伸到新 cache；不做 gallery N² 擴展，不把未生成關係偽成 degraded。保留歷史 LLM bytes UNKNOWN。
4. [scene.compose](SCENE_COMPOSITION.md)：完整小型模型／weights 還原後，逐步 retrieve→更新已放置 graph；parallel 固定 G0；no-layout 明記不消費語義。缺 pairs 有結構化提示，可補 cache 後重新準備。
5. [scene.placement](SCENE_PLACEMENT.md)：將完整 composition 與 raw GLB／annotation／room／render 配置凍結；每個 instance 保留獨立 mesh，依指定 slot 放置、存 `.blend`，有明確 camera/light 才渲染。

這些步驟可用文件中的 CLI 串接；目前不是一次命令自動規劃、補所有缺關係、生成 200 個正式場景並評分。CPU fixture 的大型 encoder／LLM 邊界仍由 tiny 模型替代，真 Blender subprocess 則實際執行。

[跨模組 smoke](audit/scene_pipeline_20260907.md) 另外將 I-Design JSON → raw query → 真 loader／export／compose → 真 Blender 存檔／渲染連續執行，並在重新開啟 `.blend` 後獨立驗證實際頂點。此測試使用 no-layout child；full ESSGNN、SG2 和語義來源由另外的專項測試覆蓋，不能混稱成一條真 pretrained full-model 場景實验。

## 3. 整合驗證

最終完整測試：**1,514 passed、71 warnings、23.73 秒、exit 0；沒有 skip／failure**，見 [final CPU log](audit/reproduction_continuation_20260907_cpu_final.log)。圖規格 **2,347 checks、all pass**，見 [final graph log](audit/reproduction_continuation_20260907_graph_final.log)。此次完整 suite 排除 GPU／Claude hooks，隱藏 CUDA／HIP，固定 Mesa llvmpipe；實際 GL_VENDOR／GL_RENDERER 也核對為 Mesa／llvmpipe。Warnings 來自既有 Transformer／timm 與明確測試非論文 tau 的提示，沒有把它們改成 skip 或關閉驗證。

完整命令、環境、source hashes、annotation／protected files 核對、七個新／更新 CLI help 與執行狀態記在 [本輪檢查 JSON](audit/reproduction_continuation_20260907_checks.json)。現有 **63 個測試檔、1,087 個頂層 test 函式**，函式數不等於展開後 cases。前一輪 1,315 passed／2,346 graph checks 屬其舊 source snapshot，不能引用成今天所有改動的驗證。

完整跨模組 smoke 的 [CPU 渲染 PNG](audit/scene_pipeline_20260907_artifacts/view_000.png)、[scene.blend](audit/scene_pipeline_20260907_artifacts/scene.blend)、[Blender result](audit/scene_pipeline_20260907_artifacts/result.json)、[獨立頂點量測](audit/scene_pipeline_20260907_artifacts/blend_geometry.json) 與 [hash 清單](audit/scene_pipeline_20260907_artifacts/evidence.json) 已保存。它們是 64×64／2-sample synthetic fixture 證據，不是 Table 2 的真實 scene quality 圖。保存的來源 JSON 保留當時暫存路徑，沒有改寫成可攜的 pretrained bundle。

首次完整回歸保留於 [initial log](audit/reproduction_continuation_20260907_cpu.log)：**2 failed、1,511 passed、71 warnings**。兩個失敗是 n04 metadata／stale-sidecar tests 意外呼叫 OPTIX renderer；GPU 隱藏後 backend 不可用。修的是測試接縫，不是將 renderer 改成另一種科學輸入。舊 PyRender 幾何測試仍保留實際圖像驗證，最終 suite 固定 Mesa llvmpipe；只設 CUDA_VISIBLE_DEVICES 不能證明 EGL 使用 CPU。早期「CPU」標籤的 GPU 隔離範圍宣稱已更正。

## 4. 保護範圍

與第一輪 580 檔快照比較，允許改動其中 `metafind/data/semantic_edges_run.py`，因它是 n08 producer，annotation 的執行路徑不依賴它；其餘 annotation imports／paper／vendor／Claude 檔案須保持原 hashes。本輪沒有對真 corpus 啟動 n08、重建向量、訓練或改寫 live queue。CPU producer／Blender 測試只在暫存目錄產生小型資料。

最後唯讀確認 PID 2043427 仍執行 `annotate_run --prompt-mode figure2_v10`，PID 2465870／2494552 是既有 Stage 1／2 等候鏈。Stage 2 鏈原本包含 n08 重建步驟，會使用更新的 producer；但其末端仍呼叫既有 `exp_type_level_query.py`／`tabulate_table1_final.py`，不會自動改跑新的 `custom_table1`。新評估需在相應 artifacts 完成後按 [自訂協定文件](CUSTOM_TABLE1_EVALUATION.md) 執行。這是目前鏈的觀察，不保證尚未執行的正式步驟已通過。

## 5. 還不能宣稱完成的部分

- **真實模型與資料仍待執行：** v10 annotation 完成後的 cache／S1／gallery／S2／完整自訂評估，要依新 source 與 record 執行並驗證；舊 artifacts 不會因 code 修復而自動成為新結果。
- **Table 1 作者協定仍 UNKNOWN：** query 具體觀測、候選 UID、各模態配對及 baseline 融合細節未完整公開。T+I、T+P 高分不能只看數字判定唯一原因；不透過弱化 query 或調分來貼論文。可執行 [固定相同／不同觀測的七模態比較](CUSTOM_TABLE1_EVALUATION.md)，結果須明確標本地協定。
- **Table 2：** 正式 200 scenes 名單／尺寸／物件數、相機／judge 協定及人工評分仍有前置限制；raw GLB front 本身 UNKNOWN。現有不重跑六個 baselines／不進行人工評分決議未被撤銷。
- **Table 3：** 完整模型＋九個 ablation 的正式訓練／評估未重跑；入口可用不代表各 row 已產生可比較數值。
- **架構與公式歧義：** 延續記錄中的研究解讀、ProcTHOR T+I、frozen CLIP 與 split 選擇，不將它們改標 PAPER FACT。新增重大研究選擇仍需使用者決定。

此文件可供 Claude 接續閱讀；未透過外部訊息工具傳送，也不代表 Claude 已閱讀或批准。
