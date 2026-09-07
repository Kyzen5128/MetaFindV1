# 2026-09-07 端到端復現審查與修正

這是第一輪完成時的審查快照；原始驗證 log／SHA 保留。後續 tau、n08、no-layout 與實際 scene/render 交接修補，見 [接續完整檢查](REPRODUCTION_CONTINUATION_20260907.md)。下文「尚未」與「本輪未執行」描述第一輪，不能拿來否定後續小型 CPU Blender 驗證。

**驗證範圍更正：** 接續檢查發現兩個當時列入 CPU suite 的 metadata tests 會呼叫 n04 OPTIX renderer，舊 PyRender tests 也沒有固定 EGL software backend。因此下文舊「不碰 live GPU」的表述沒有足夠隔離證據，撤回此範圍宣稱；舊 1,315-pass log 本身保留。最終隔離設定與修補見接續報告，未因此改動真實 corpus 或標註程式。

**Verdict：PARTIALLY VERIFIED。** 核心物件檢索、兩階段訓練與 ESSGNN 已有實作，本輪修正實驗身分、凍結狀態、無效訓練和 Stage 2 cache 來源核驗，另補 Algorithm 1 的逐步場景檢索核心。這些成果不等於重現作者的 Table 1–3 數字；本輪沒有啟動真實模型訓練、全量推論、標註、render 或 judge。

## 1. Audit Scope／範圍

使用者將原本的唯讀審查，進一步授權為測試整理、修正／必要重寫，以及從頭到尾盡可能復現論文。本輪先做唯讀 paper/code audit，再依該授權修正已確認的程式問題；沒有把 audit skill 的唯讀階段當作實作授權。

覆蓋鏈：MetaFind 主文與附錄 → Objaverse／ProcTHOR 資料 → Stage 1 → gallery → Stage 2／ESSGNN → Table 1 與自訂七模態評估 → Algorithm 1／Table 2–3 缺口。五個 TeX 檔皆納入閱讀，六張引用圖片也已檢視（架構、資料處理及兩組場景比較）；沒有逐一人工檢視數萬個 mesh、annotation、embedding 或訓練 batch。

標註與 live queue 依使用者要求保留。測試目錄整理、刪除四支已撤回 probe、文件更正的完整歷程另見 [CLEANUP_REPORT_20260907.md](CLEANUP_REPORT_20260907.md)；該報告中的 1,235 cases 是整理階段的歷史結果。

## 2. Authority and Evidence Ledger／權威與證據

| 分類 | 證據與用途 |
|---|---|
| **PAPER FACT** | [neurips_2025.tex](paper/metafind_source/neurips_2025.tex)、[2methdology.tex](paper/metafind_source/2methdology.tex)、[3experiments.tex](paper/metafind_source/3experiments.tex)、[4backgound.tex](paper/metafind_source/4backgound.tex)、[appendix.tex](paper/metafind_source/appendix.tex)。MetaFind 方法／實驗主張只由這些原文支持。 |
| **UPSTREAM FACT** | `/home/kyzen/upstream/` 的 ULIP、EGNN、I-Design、Text2Shape 等實作，只在對應繼承／比較範圍使用。Text2Shape 的兩個計分函式與本地副本 AST 一致；I-Design 舊 judge 的五項 1–10 評分不能代替 MetaFind 四項 1–5。 |
| **IMPLEMENTATION CHOICE** | [graph_spec.yaml](graph/graph_spec.yaml) 的 U-18／U-21、D-14 等已記錄決策，以及每次 run 的 resolved protocols。U-30 的 learned token 依目前 [resolved edge protocol](../data/outputs/essgnn_edge_protocol.json) 與 consumer 實作；registry 仍標 UNKNOWN，不能聲稱該欄位已更新。這些不是新增的論文事實。 |
| **OBSERVED IMPLEMENTATION** | 本報告連結的 production/test 路徑與本輪 diff；[DATA_FLOW.md](DATA_FLOW.md) 記錄實際 producer → artifact → consumer。 |
| **OBSERVED DATA** | 新鮮 CPU 測試、反例、source hashes 與 process snapshot，見 [本輪驗證紀錄](audit/reproduction_20260907_checks.json)。沒有新 Table 1–3 實驗數字。 |

既有 Markdown、tests、handoff 與先前 agent 說法不能填補原文缺失；本輪也未改寫 protected paper／vendor／Claude 設定。

## 3. Paper → Specification → Implementation → Validation Matrix

| 論文要求 | 規格／實作路徑 | 本轮驗證及狀態 |
|---|---|---|
| §2.3：Objaverse 約 48K，11 views，GPT-4o 結構化標註 | n03–n06：`pointclouds`、`renders`、`annotate_run`、`encode_text_image` | **PARTIAL**：資料處理及測試存在；實際 corpus／LLM／文字 serializer 以 run 記錄為準。標註執行中，沒有修改或重算。 |
| §3.1：兩資料集 80/20 | `splits.py`、`scene_splits.py` | **DEVIATION / IMPLEMENTATION CHOICE**：本地 Objaverse 依使用者已批准 D-3b 採 80/10/10 及 dev/holdout aliases。只能按實際 UID／選模紀錄解釋，不能稱為作者原 split。 |
| §2.4：雙塔、完整 gallery、可缺模態 query | `fusion.py`、`dual_tower.py`、Stage1Dataset | **VERIFIED WITHIN TESTS**：缺模態／mask／共享配置與 gallery 完整性測試；凍結狀態缺陷本輪已修。 |
| §2.6 Eq.5：Stage 1 單向 contrastive、各模態獨立 30% mask | `losses.py`、`stage1.py`、resolved protocol | **PARTIAL**：loss/mask 行為有測試；實際優化範圍與 recipe 仍由 checkpoint 決定。本輪拒絕零步訓練。 |
| 附錄 Eq.10/13/14：ESSGNN 共享 message、座標與特徵更新 | `essgnn.py`、既有 D-14／PRIMARY_INTERPRETATION | **VERIFIED WITHIN CHOSEN INTERPRETATION**：逐式碼審與 [ESSGNN tests](../tests/models/test_essgnn.py) 的 SE(3)／translation assertions；不消除正文／附錄差異。 |
| §2.6 Eq.6–8：layout residual、30% batch scene dropout、Stage 2 凍結 gallery | `stage2.py`、`dual_tower.py`、room context／symmetric loss | **PARTIAL**：移除 target、room scope、dropout、loss／freeze 皆有回歸；本輪加來源核驗與零有效 batch 拒絕。實際 ProcTHOR T+I 沿用已批准 DL-104。 |
| §3.2／Table 1：七 query 組合、R@1/R@5，Objaverse Stage 2 關閉 ESSGNN | `run_retrieval.py`、`retrieval.py`、`custom_table1.py` | **PARTIAL**：exact-UID scorer／overlay／fixed pools 可執行，作者 query／gallery 清單及觀測構法仍未完整公開。 |
| §2.7 Algorithm 1：每次取回後更新圖，供下次檢索 | 新 `scene/compose.py`，U-18／U-21／U-30 | **PARTIAL**：真 tiny ESSGNN 的逐步 loop 已驗證；新增從真 S1/S2 record 匯出 query model 的介面。raw input bundle／render／judge 尚未閉合。 |
| §3.3 Table 2：200 scenes，四項 1–5，GPT-4o 與五位專家，layout+render | planner 工具／新 composition core；U-27 尚未解決 | **MISSING**：正式場景清單、完整渲染與評分執行鏈。不能把 tensor output 或 object retrieval 指標當成 Table 2。 |
| Table 3：十列（Full 與九個消融）、Text Only R@1 及場景分數 | fusion variants、ESSGNN ablations；新 parallel/layout-off modes | **PARTIAL / MISSING**：若干機制存在，但完整排程／結果整合與 GAT 替代未完成；不能據此填整張表。 |

## 4. Runtime Trace／確認的執行路徑

1. **Stage 1**：讀取 checkpoint/protocol 有效設定 → query observation 覆蓋 → `--limit` → 檢查完整 batch → seed／backbone → DataLoader → masking／fusion／loss → optimizer → checkpoint。現在資料不足時，在 backbone 建構前即拒絕。
2. **Gallery**：Stage 1 parent／encoder／forward config 還原 → ProcTHOR modality JSON 與實際 image/PC bytes snapshot → 從已核验 bytes 編碼 → 重核來源 → NPZ 內嵌來源摘要 → sidecar。trainer 與 probe 同時核對 index bytes、來源、parent、模態與 encoder。
3. **Stage 2**：核對 room/house inputs → raw modality vectors／fused positives → unique-positive 分批及小 batch 過濾 → 拒絕零可用批次 → backbone／model → 每批移除 target 的 room context → query fusion + λ·layout → symmetric loss → 保存前重核來源。第一批 partition 直接留給 epoch 0，不額外消耗 RNG。
4. **Table 1**：protocol 固定 query/gallery → degraded exclusions／coverage → checkpoint 還原、Stage 2 overlay 且 layout=None → encode 或 promoted index → 明確 UID target map → float64 streaming rank。Text2Shape 附加診斷現在也分批取 top-k，最後只對小型索引表一次匯總。
5. **Composition**：明確 G0、順序固定的 query/slot、gallery → 當前已放置物件的 graph → ESSGNN → fusion + λ·layout → cosine top-k → 將取回資產連同原 slot 加入圖 → 下次重算。node semantics 來自取回資產；future slot 不預先進 G0。此處「放置」是輸出場景 slot／asset 配對，還未執行 GLB 場景渲染。

## 5. 已確認並修正的問題

| 問題／最小反例 | 修正 | 對歷史結果的界線 |
|---|---|---|
| **CONFIRMED**：Stage 1 32 items／batch 64／drop_last=True，loader 為空但仍可存檔並成功返回。 | `stage1.main` 在所有縮池後檢查正整數 batch size 與完整 batch。測到 n=0、63、limit 縮池、query-pack 縮池拒絕，n=64 才到 backbone。 | 只影響無完整 batch 的 run；未證明先前正式 Table 1 run 是零步。 |
| **CONFIRMED**：`random_view` 同 seed 不同 worker 產生不同 query 序列，原 arm hash 相同。 | `arm_config_hash` 記錄有效 worker 數、Python RNG seed 來源與 stream 生命週期；未知 worker 設定拒絕。 | 舊 random_view hash 不可與新 hash 直接當同 treatment；deterministic mean 保留原識別語義。 |
| **CONFIRMED**：gallery 雖無梯度，`model.train()` 仍重新啟用 dropout，重複 forward 不同。 | DualTower 保留明確 freeze 狀態；train 後 gallery 維持 eval。unfreeze 尊重父 mode；修正無參數 fusion 被誤認 tied。 | mean 或 dropout=0 的舊 run 不因此產生新數值差異，不能把它當成所有融合高分的原因。 |
| **CONFIRMED**：舊 raw T/I index 可搭配改過的 modality JSON，原 parent／index／當前 input identity 檢查均通過。 | source identity 綁選用 JSON、ordered views、宣告 PC bytes；NPZ 內嵌摘要，producer 前後及 consumer 重核。 | 舊 index 不能補寫當前 sidecar 就聲稱來源已驗證；新訓練應重建 index。 |
| **CONFIRMED**：Stage 2 同一 asset 的 12 個 instances 分成 B=1，全被小 batch 規則丟棄，仍成功存檔。 | `training_batches` 在 backbone 前拒絕零 usable batches；每個 epoch 另檢查 optimizer 有更新。 | 不修改採樣／loss 或捏造負樣本來讓訓練通過。 |
| **CONFIRMED**：Text2Shape 診斷重建完整 Q×G float64 similarities 及 partition indices，繞過主 scorer 的 streaming 設計。 | adapter 預設每次最多 256 queries，完整 gallery 不變；保留兩個 upstream 函式，拼接 top-k indices 後一次計算指標。 | Q=9,000／G=45,000 的示意陣列主體由約 6.48GB 降至每塊約 184MB，非實測 peak RSS。canonical R@k 路徑不变；upstream ties 仍任意，浮點近同分不保證跨 GEMM shape bitwise 一致。 |

來源核验也拒絕 legacy Stage 2 sidecar 單獨新增 `gallery_source_status=verified`；JSON key 重排不被當成資料漂移。只有既有 `--allow-legacy-gallery-index` 可明確讀取缺來源證據的舊 index，結果標為 `legacy_unbound`。搬移 corpus root 目前仍採嚴格拒絕，未新增路徑重映射政策。

新 composition CLI 在獨立 review 中也找到並修正 manifest 缺 forward 設定時靜默採 default、state dtype 靜默窄化及極小非零向量的 cosine epsilon 問題。這些屬本輪新程式的整合修正，沒有拿尚未使用過的新程式解釋舊 Table 1。

## 6. Validation Evidence／驗證

最終完整 CPU suite：**1,315 passed，71 warnings，26.28 秒，exit 0**；[完整 log](audit/reproduction_20260907_cpu.log)。graph checker：**2,346 checks，all pass，exit 0**；[完整 log](audit/reproduction_20260907_graph.log)。warnings 為既有 Transformer／timm 與刻意偏離溫度的測試提示，沒有 QueueFeederThread exception。`git diff --check` 亦通過。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
/home/kyzen/miniconda3/envs/MetaFind/bin/python -m pytest tests -q -ra \
  -p no:cacheprovider --ignore=tests/gpu --ignore=tests/hooks

PYTHONDONTWRITEBYTECODE=1 /home/kyzen/miniconda3/envs/MetaFind/bin/python tools/check_graph.py
```

先前 scoped evidence 包含 Stage 1 四檔 230 passed 及補充 CLI 5 cases；Stage 2 五檔 89 passed，後續 metadata/key-order 53 passed；Text2Shape 與 retrieval runner 61 passed；新 composition 28 cases。這些子集合相互重疊，不能相加成總測試數。

新 composition 測試使用真 QueryTower／ESSGNN／fusion、小型可手算 tensor 及暫存權重。三步例子為空 G0 取 A、讀 A context 後取 B、讀 A+B 後再取 A；相同模型在 parallel 模式固定讀 G0 而全取 A。這是迭代機制證據，不是場景品質改善證據。

CPU suite 排除 `tests/gpu` 與 `tests/hooks`，設定 offline 及單執行緒 BLAS，不碰 live GPU 工作。沒有針對真 corpus 做 n06 重編碼、G4 promotion、Stage 1／2 訓練或 GPU smoke，也沒有重新量測 paper tables。

## 7. Implementation Choices／沿用的選擇

- ESSGNN 依現有 PRIMARY_INTERPRETATION／D-14 使用附錄所需的 invariant node 表示與更新規則。正文將座標併入 h0 等差異保留為原文歧義，不為通過測試改寫論文。
- U-18／U-21：I-Design slot 提供位置／旋轉／尺寸；G0 只有已放置物件；node text 取 retrieved asset annotation。kNN k=8、support 與 n08 cache key 重用現有程式。
- U-30 議題依目前 resolved edge protocol 的 `learned_missing_token`：明確記錄為失敗的 semantic edge 才使用 learned missing token；未準備的 cache key 直接拒絕。完整 final graph 也需要最後一次放置後的 edge 證據。registry 的 UNKNOWN 舊標記與當前 artifact 不一致，本輪保留此追溯差異，未宣告新的全局研究決策。
- Composition 同分按顯式 gallery 順序選擇；這是無 GT 的選物件規則，不是把 Table 1 悲觀 rank 改成有利同分。
- 自訂 Table 1 採固定 same-UID pools 比較相同／不同觀測；詳見 [CUSTOM_TABLE1_EVALUATION.md](CUSTOM_TABLE1_EVALUATION.md)。它不承諾作者的觀測構法已知。

## 8. Deviations／偏離

保留 [D-1…D-14 registry](../README.md) 的追溯性，啟用狀態需依各次 run record。此次沒有撤销既有「不重跑六個 baseline」（D-3）或「不進行人工場景評分」（D-4）決策，也沒有把已批准的本地 LLM 替代改回 GPT-4o。

目前訓練可使用 cached frozen CLIP、不同文字／view serializer、ProcTHOR T+I、非原作者 split 等選擇；paper 是否逐一確定相反設定，與本次 run 是否不同於選定規格，是兩個分開的判斷。不得以 README 的永久偏離數量代替 checkpoint 驗證。

## 9. Missing or Unreachable Implementation／未閉合部分

1. **真實 scene input bundle**：model-only exporter 能還原 S1/S2 query model，但完整 gallery、planned query embeddings、retrieved-asset node vectors 和新 semantic pairs 尚需由相容 encoder 與來源產物產生。Objaverse UID 不能拿 ProcTHOR assetId node map 直接 join；維度相等也不足以證明 encoder 相同。
2. **I-Design → GLB placement → render → judge**：既有 planner 工具可產生 slots；新 core 回傳 asset／slot／graph／top-k。尚未自動串上實際 mesh render 與四項 judge。
3. **正式 Table 2 清單與人工評分**：作者的 200 scenes 清單與各場景設定未提供；沒有五位專家評分產物。上游 60 prompts 或本地兩個 smoke prompts 不等於該 200 scenes。
4. **Table 3 全表**：GAT 替代仍未實作；完整各 arm 訓練、相同 scene manifest 評分和聚合仍缺。新 `parallel`／layout-off 只覆蓋其中推論機制，不能冒稱十列齊備。
5. **Table 1 原協定**：作者 query 文字／views／PC pairing、精確 gallery/GT 清單與 baseline 融合細節仍未完整指定。自訂評估已可作明確條件下的比較，但不能把缺失細節補成論文事實。

## 10. Unknowns and Conflicts／未知與衝突

**UNKNOWN：** 具體 Table 1 query construction、官方 split/GT、確切 pretrained checkpoint、Transformer fusion 完整配置、U-27 正式場景設定、GAT 詳細架構及 judge 原 prompt。論文有給評估指標與大方向，不是「完全沒有評估方法」；缺的是足以唯一重建所有實驗的細節。

**INFERENCE：** 相同資產的多個模態可能提供互補辨識訊號，因此本地 T+I／T+P 高於單模態是可能的。但分數高本身不證明 leakage，也不能推導融合必定加分、作者一定使用另一資產當 query，或某個 bug 已解釋全部差距。

正文與附錄的 ESSGNN h0／距離／更新描述差異仍依既有決策明列。Table 1 的 Stage 2 row 關閉 ESSGNN 是原文明確描述，不能為壓低分數把 ProcTHOR layout 硬塞進 Objaverse 評估。

## 11. Reproducibility Provenance／可重播證據

本輪基於 Git HEAD `87ccfbbf460cfd2bcc518f654e1b7d0fd3e506d8` 與已有未提交修改，沒有 commit。最終變更 source hashes、paper hashes、命令、exit codes、CPU/graph logs、保護檔案核對與 PID snapshot 均存入 [reproduction_20260907_checks.json](audit/reproduction_20260907_checks.json)。歷史 cleanup/custom audit 不覆寫。

標註 PID 2043427 及等待鏈 PID 2465870／2494552 保持原命令與啟動時間，未停止、重啟或編輯 live shell。修改的 trainer／index 會供等待鏈日後新啟動的 Python 使用；現有 annotation source/imports 與輸入輸出未改動。最终比對 580 個保護／凍結檔案無內容變化；五個 TeX 與六張引用圖的 SHA-256 也全數符合 SOURCE_MANIFEST。

## 12. Reproduction Impact／對復現的影響

本輪把「零步也成功」「同實验 ID 卻不同 query RNG」「frozen 仍 dropout」「舊向量配新來源」「streaming 旁路大型矩陣」這些可測的缺陷排除，並讓 Algorithm 1 的 context 更新可執行、可追查。舊 checkpoint 不會因此自動變成修正後重訓的模型；缺來源的舊 index 也不會自動獲得完整證據。

對使用者最初的高分問題，結論仍是：**現有證據不足以給所有 T+I／T+P 差距一個唯一成因，也不足以認證當前本地表就是作者 Table 1。** 應在同一 parent、UID pools、gallery 定義與 scorer 下執行已寫好的觀測對照，保留 per-query ranks／sources，才可量化哪些觀測選擇造成差異。不能按論文數字調參或刪除容易 query 使分數看起來接近。

## 13. Verdict and Required User Decision／結論與後續決策

**PARTIALLY VERIFIED**：已完成授權範圍內、可由證據支持的修復與部分缺失機制；尚未達成作者 Table 1–3 的完整數值復現。

本輪程式修正不需要另一次權限確認。若繼續進入正式 Table 2／3，以下依 [AGENTS.md](../AGENTS.md)「material paper ambiguity…requires a research decision」要求明確研究決策，不能默默替作者補定：

- U-27：採用使用者提供的正式 200-scene manifest，或另立有名稱、固定 seed／清單／尺寸與物件數的自訂場景協定。後者須標為自訂比較。
- 缺失的 GAT 架構／judge prompt：先固定具體配置及可驗證输入／输出，再跑相應 arm；不能拿現有 ESSGNN 或 I-Design 舊評分器頂替。
- D-3／D-4 若要改為重跑全 baselines 或新增人工研究，需要明確改變既有研究範圍。

這些是正式新實驗的前置條件，不是本輪已授權修正的阻擋理由。本報告供共同 workspace 中後續 Claude／Codex 查閱；未宣稱其他 agent 系統已讀取或收到外部訊息。
