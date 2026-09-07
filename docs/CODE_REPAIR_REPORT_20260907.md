# 2026-09-07 程式修正與 Claude 交接

> 本檔記錄較早一輪修正完成當時的狀態；PID、git 狀態、測試路徑及檔案 SHA 都是當時快照。後續 `tests/` 已按領域搬移，最新執行命令見 [tests/README.md](../tests/README.md)，本輪整理見 [CLEANUP_REPORT_20260907.md](CLEANUP_REPORT_20260907.md)。下列歷史測試數與驗證 JSON 不改寫成新結果。

本次是在 Kyzen 明確授權修改後，修復唯讀審查確認的執行與重現性缺陷。程式修正、交叉審查及 CPU 回歸驗證已完成；**沒有重新訓練模型、重算 Table 1 或改寫既有研究結果，也不能據此宣稱論文已復現。**

基準 revision：`785a2d9`。本次修改尚未 commit。逐檔 SHA、完整測試命令、實際 checkpoint 驗證與排程資訊見 [驗證紀錄](audit/code_repair_20260907_checks.json)。

**1. 已修正的缺陷（OBSERVED IMPLEMENTATION）**

| 問題 | 修正後行為及主要檔案 |
|---|---|
| 載入舊 checkpoint 卻使用現在的 `prefusion_norm`、fusion、image token 數或 training scope；相同權重仍可能產生不同輸出。 | [stage1.py](../metafind/train/stage1.py) 統一還原 checkpoint 的有效配置。新檔保存實際 query/gallery FusionConfig；舊檔從 `arm_config` 還原。缺少核心證據或內容衝突會拒絕。直接呼叫舊 loader 的診斷工具也會偵測 fusion 配置漂移。 |
| 非官方 backbone 的 fuser-only checkpoint 不含 backbone 可訓練權重，下游原本可能默默載回預設 initializer。 | 同一模組以紀錄的 initializer URI 與 SHA 還原；搬移後只接受相同內容。紀錄有 OpenCLIP blob identity 時也核對目前 cache。active indexing、evaluation、Stage 2 與 Table 1 probe 已接入。 |
| `--freeze-gallery` 只凍結 fusion，backbone 仍可能訓練；另外，凍結的隨機 Transformer gallery 原本會漏存。 | 要求此 Stage 1 模式使用 `fuser_only`，不再假稱整個 gallery 已凍結。checkpoint 同時保存凍結的 gallery state；非 mean gallery 缺少必要權重時拒絕載入。 |
| gallery fingerprint 只含參數／buffer，無法發現前向配置改變。 | [gallery_index.py](../metafind/train/gallery_index.py) 新增 hash v2，包含實際 gallery FusionConfig、backbone class/dtype、宣告模態與權重/buffer。query 的合法 Stage 2 更新不改變 gallery identity。 |
| semantic producer 使用 `relation_text`，Stage 2 與 coverage reader 使用 `text`，同一對物件得到不同 cache key。 | [semantic_edges.py](../metafind/data/semantic_edges.py) 的 `relation_text_for()` 成為共用規則：有非空 relation text 時使用它，否則使用 text。node embedding 文字仍獨立處理。 |
| room protocol 遇到舊 house graph 可能直接採整棟房子為 context。 | [stage2.py](../metafind/train/stage2.py) 驗證 protocol 與每份 graph scope；legacy house 僅在明確的 house protocol 下接受。缺失 graph 也不再默默略過。 |
| Stage 2 沒有完整綁定實際讀取的 protocol、graph、node、semantic、split 與 modality 資料。 | 訓練前捕捉輸入內容身份、載入後再核對；checkpoint 嵌入 protocol、graph scope、house 集合、input hashes 與 layout dimensions。replay 能同時驗證內容及目前資料路徑。 |
| Stage 2 overlay 或 ProcTHOR probe 對 parent、模型 bytes、gallery 或部分權重檢查不足。 | [run_retrieval.py](../metafind/eval/run_retrieval.py) 及兩個 active probes 共用驗證入口；檢查 checkpoint SHA、parent、embedded metadata、query coverage，拒絕 gallery 覆寫。封存時可搬移 URI，內容與配置仍須一致。 |
| n06 發生 OOM／部分編碼失敗仍可能回傳成功；替換失敗後舊 embedding 可能留下供下游使用。 | [encode_text_image.py](../metafind/data/encode_text_image.py) 統一 return `progress.rc`；正確分類 RESOURCE、輸入與契約錯誤。替換前把舊檔移到既有 retired 區保留證據；sidecar 發布失敗也退役新 NPZ。 |
| QueryPack `--limit` 在 builder 已覆寫完整檔案後才改 manifest tag。 | [make_query_pack.py](../tools/make_query_pack.py) 在寫任何 array／done-list 前決定限量 tag，完整 shard 保持原樣；拒絕非正數 limit。 |
| multi-view + zero padding 下，image=None 與「image tensor 存在但標示缺失」使用不同填充值。 | [fusion.py](../metafind/models/fusion.py) 統一缺失 image 的填值。兩種表示的實際前向輸出一致。 |
| query observation、image policy、PC perturb 不進入 arm hash；preload 的主程序 Python RNG 未一起設 seed。 | 三項均列入 treatment 身份，新紀錄 `arm_config_version=2`；同時 seed Python、NumPy、Torch。舊 digest 不應拿新版本公式直接重算後宣稱一致。 |
| uint8 baseColorFactor 的 RGB 為 0／1 時被當成 unit scale，可能把極暗色變成白色。 | [pointclouds.py](../metafind/data/pointclouds.py) 保留 dtype 判斷，sampler 8→9。此修正沒有重採樣既有 corpus。 |
| split-folder 第二次執行會被自己產生的 README/uids 擋住，且可能先刪掉部分 symlink。 | [materialize_split_dirs.py](../tools/materialize_split_dirs.py) 完整 preflight 後才重建，僅接受自身 marker 的 metadata；遇到其他實體檔時保留整個 view。 |
| Table 1 probe 無條件依賴另一個 attrs corpus 的 fields cache。 | [exp_type_level_query.py](../tools/probes/exp_type_level_query.py) 改成明確選用。缺少覆蓋時獨立記錄該變體 skipped，主 query pool 不變。保留原有 attribution 與 own-mean 修改。 |
| manifest generator 硬寫 sampler 8，改 source 後也可能錯報資料版本。 | [build_dataset_manifest.py](../tools/build_dataset_manifest.py) 與 [write_archive_manifest.py](../tools/write_archive_manifest.py) 依實際被清點的 index 記錄單版／MIXED／UNKNOWN；不以目前 source 版本冒充舊資料版本。 |
| 排程的 `python | grep | tail` 隱藏 producer 失敗，部分 eval 失敗後仍印 DONE；archive 的 `cp && cp` 也會穿透失敗。 | [chain_paper_stage1.sh](../tools/chain_paper_stage1.sh)、[chain_paper_stage2.sh](../tools/chain_paper_stage2.sh) 使用 pipefail／明確失敗退出，awk 過濾不把空輸出當失敗；archive 分成獨立命令。完成訊號只在所有前置步驟成功後出現。 |

**2. 實際驗證（OBSERVED DATA）**

- 最終 CPU suite：**1,138 passed，34.17 秒**。完整命令、log SHA 與逐檔 source SHA 已保存於驗證紀錄。
- 未執行 `tests/test_cuda_smoke.py`，避免與正在標註的 GPU 工作競爭；未執行與本次修改無關的 Claude-only hook 測試。不能把這兩類算成通過。
- 測試仍有既有 Transformer／刻意偏離論文溫度設定警告，以及偶發 `QueueFeederThread` teardown 警告。相關單檔重跑未重現；原因尚未證實，沒有為消除警告更改 production worker 策略。
- 使用實際 P1s epoch 9 checkpoint，SHA `074f8d98e33a4faccb7147fa78369277be60b2c8da50849b2f7d46219db08f48`，故意給錯的目前配置；新 loader 還原為 `prefusion_norm=True`、Transformer gallery、`image_tokens=1`。七種模態條件對正確參照的最大絕對誤差均為 **0**。這是 **CPU、真實 fusion 權重、合成輸入** 的還原驗證，不是全 corpus retrieval。
- initializer checksum、frozen gallery save/load、Stage 2 archive 搬移、錯 parent／錯 bytes／錯配置、relation cache、輸入替換、限量 shard 隔離、失敗排程與空過濾輸出均有回歸案例。
- `git diff --check` 及兩支 launcher 的 `bash -n` 通過。沒有改 protected paper source、vendor、`CLAUDE.md` 或 `.claude/`。

**3. 正在跑的工作與檔案狀態**

- 標註 PID **2043427** 持續執行 `python -m metafind.data.annotate_run --prompt-mode figure2_v10`；未停止、重啟或修改其程式／輸入／輸出。18:48 後檢查仍為運行狀態；最後已寫出的進度為 13,000/46,004。
- 舊等待排程 2080316、2080317 在修正期間精確暫停，完成驗證後結束舊 shell，再以修正檔啟動，避免 Bash 讀取舊 script buffer。
- 新 Stage 1 等待 PID **2465870**；新 Stage 2 等待 PID **2465873**。两者仍在原本 step 0 等待，尚未開始新訓練。新 PID、launcher SHA、log 路徑亦在驗證紀錄。
- 執行檔位於 `/home/kyzen/metafind/metafind_data_paper/outputs/logs/chain_paper_stage{1,2}_20260906.sh`，與 repo 的兩份 reviewed source 內容一致。原檔保存在同目錄的 `chain_paper_stage{1,2}_20260906.pre_codex_fix_20260907.sh`；原 log 保留並追加重啟紀錄。
- **不要原地修改仍在執行的 shell script。** 日後修改須先處理等待程序，再啟動新 shell。
- 既有 `tools/probes/exp_json_text_query.py` 保持 byte-identical，仍未追蹤；既有 dirty `exp_type_level_query.py` 的 attribution／own-mean 功能保留。partner PC 改成使用正確 query point path，與 fully-separate 修正一致。
- code sampler 現為 9；既有 sampler 8 點雲與其他 corpus artifact **沒有**重生或改標。後續 n03 重跑會要求舊版本重新驗證／採樣；目前排程也沒有額外加入 n03 全量重採樣。

**4. 舊結果與相容性界線**

新 gallery index 預設 v2。舊 v1 僅能透過 `--allow-legacy-gallery-index` 明確使用，仍須驗 parent 配置及舊 hash；此相容模式無法補證當時 producer 未記錄的 forward flags。新排程會建立新 index，不需此旗標。

舊 Stage 2 缺 embedded metadata／input identity 時，相关 evaluator/probe 要求明確 legacy 選項。歷史 ProcTHOR index 缺 modality declaration 的路徑仍會拒絕，**沒有宣稱所有舊 artifact 都能完整重放**。`fully_separate + pc` 的 Stage 2 gallery cache 沒有獨立 query PC 向量，主訓練／probe 明確拒絕；目前宣告 text+image 的路徑不受影響。type-level probe 僅支援 `image_tokens=1`，其他值明確拒絕。

**5. 為什麼這不等於已解釋 Table 1 的高分**

PAPER FACT：[論文 §3.2 與 Table 1](paper/metafind_source/3experiments.tex) 說明 Stage 1 head、Stage 2 shared head 及 Objaverse 評估停用 layout 的關係；[§2 方法](paper/metafind_source/2methdology.tex) 是 modality、gallery 與兩階段訓練的內容權威。程式通過測試不能覆蓋論文沒有交代的設定。

OBSERVED IMPLEMENTATION：你那張表的 own-query 路徑，文字使用同 UID cached text，圖片取同 UID 的 view，canonical PC 也來自同一個物件。模態融合可因此保留很強的物件辨識訊號；query/gallery fusion head 仍是各自的權重。不能把「同物件輸入」直接說成「两個 head 一樣」，也不能僅靠數字較高判定程式灌分。

UNKNOWN：作者 Table 1 query 的文字、圖片、點雲觀測如何與 gallery 對應，仍不足以唯一重建。本次查出的缺陷中，有些影響特定 ablation、切換 corpus 或下一條 Stage 2 執行路徑；**沒有證據顯示它們全部發生在既有 P1s 分數上**，也沒有用修正後的新分數估計影響。

待後續實验完成，應先確認新 artifact 的 parent、配置、輸入與 pool 都吻合，再比較數字。不要調低融合分數來貼近 Table 1；CLIP scope、query construction 等研究決策仍須依論文與既有明確授權處理。Table 2 完整 iterative composition／人工評估、未實作的 GAT ablation 也不應宣稱已由本次 bug fix 完成。

**給 Claude 的具體交接**

請先讀本檔與 `docs/audit/code_repair_20260907_checks.json`，再看目前 git diff。標註與兩支後續等待排程已保留並恢復；不要另開重複工作，不要照舊 handoff 的「tree clean／已復現」敘述宣稱現況。後續以實際新產物驗證及整理 Table 1，比較時完整標示 query construction、gallery pool、Stage 1 parent、Stage 2 lineage 與仍未解決的論文差異。

此檔是已完成修正的交接文件，不是新的研究決策或訓練授權。Codex 目前沒有可對既有 VS Code Claude 對話送訊息的工具；文件已寫入共同 workspace，**尚未宣稱 Claude 已收到或讀完**。
