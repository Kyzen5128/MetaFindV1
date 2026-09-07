# 真實訓練與公式接續審查（2026-09-08）

**Verdict: PARTIALLY VERIFIED。** 接續 [2026-09-07 執行路徑審查](REPRODUCTION_RUNTIME_REVIEW_20260907.md)，本輪完成兩階段真正 CPU 單步訓練、checkpoint 還原及新權重的自訂七模態評估，修補 no-layout 的 λ 交接，核對 Table 3 變體可達性，並補上獨立 ESSGNN 數值／梯度 oracle。標註與現行等候鏈維持原狀。這是程式修補與有限執行證據，不能宣稱 Table 1–3 的正式分數已復現。

## 1. Audit Scope

以 `paper-audit`／`reproduction-audit` 先唯讀查證附錄、Table 3、兩階段訓練與載入。確認實作缺陷後，依使用者已授權的非標註修補範圍實作。實際訓練使用隔離資料根目錄、CPU、離線模型與限制執行緒；不覆寫正式 corpus、舊 checkpoint 或 protocol。

## 2. Authority and Evidence Ledger

| 分類 | 證據 | 所能支持的結論 |
|---|---|---|
| PAPER FACT | [附錄 message](paper/metafind_source/appendix.tex:32)、[座標更新](paper/metafind_source/appendix.tex:50)、[feature 更新](paper/metafind_source/appendix.tex:65) | 附錄使用共享 message、平方距離、舊 h／x 狀態与求和 residual 更新。 |
| PAPER FACT | [Table 3](paper/metafind_source/3experiments.tex:94)、[ablation 說明](paper/metafind_source/3experiments.tex:143) | Full 加九個 ablation；列出 GAT、fuser-only 等比較，但沒有完整 GAT 架構。 |
| PAPER FACT | [實驗設定](paper/metafind_source/3experiments.tex:15)、[Stage 2 說明](paper/metafind_source/3experiments.tex:24) | τ=.5、Stage 2 scene dropout 30%；不足以決定全部 optimizer 與評估細節。 |
| IMPLEMENTATION CHOICE | 既有 D-14／U-26、Stage 1／2 resolved protocols | 本輪維持已選 ESSGNN 附錄分支；不以測試結果消除正文／附錄的差異。 |
| OBSERVED IMPLEMENTATION | `no_layout` 加 pinned λ 的 producer／consumer 矛盾 | trainer 原本記錄不存在參數的 initializer，後續 strict loader 因 `lambda_init` 非空而拒絕。 |
| OBSERVED DATA | 本輪隔離 CPU 執行產物 | 支持實際載入、有限 batch 的更新與還原；不能估計正式檢索品質。 |

## 3. Paper → Specification → Implementation → Validation Matrix

| 需求 | 實作／規格 | 驗證 | 狀態 |
|---|---|---|---|
| 附錄單層 shared-message 更新 | `ESSGCLShared`，保留 D-14／U-26 選擇 | 逐邊 NumPy forward、所有輸入及三個 phi 的逐元素 central differences | MATCHED，限所選單層公式 |
| no-layout 無 Eq.6 λ | 真 trainer initializer 回傳 `None` | Full／no-layout × ratio／pinned，共四種真 payload→strict loader→overlay | MATCHED |
| Stage 1 訓練與儲存 | 官方 ULIP-2、point encoder＋兩個 fuser、固定 τ | 真 CLI、一個 optimizer step、fresh process 真模型 strict restore | PARTIAL，2 train／2 selection |
| Stage 2 訓練與儲存 | 凍結 parent／gallery、ESSGNN＋query fusion | 本輪執行證據見下節 | PARTIAL，診斷資料 |
| Table 3 的完整結果 | Full 加九個 ablation | 變體來源及可達性核對；沒有十列正式訓練與評分 | UNKNOWN／PARTIAL |

## 4. Runtime Trace

Stage 1：複製舊 `attrs_v1` corpus 的 2 筆 dev_train／2 筆 disjoint dev_val，保留來源 SHA → 隔離 hyperparameter artifact 將 batch 64→2、epochs 5→1 → 原始 `metafind.train.stage1` CLI → 官方 ULIP-2／實際 PointBERT forward、反向傳播與 AdamW → 七模態 dev selection → latest／best checkpoint → 另一個程序重新初始化官方模型並 strict restore。

Stage 2：原始 `test_00757` room graph 的 8 個資產、88 張真 PNG、28 條 graph edges → 沿用 16 組歷史關係句文字並用真 CLIP 重編 node／edge → 真 n11b 對 canonical text＋影像編碼 → source v2 gallery＋split／input identity → `Stage2.main` 的 Full 單步更新 → 獨立程序重新載入 parent／child → 比對 layout-present queries → 公開 `custom_table1` CLI 以新 parent／child 跑兩觀測／七模態。

本輪 λ 修正：[initialise_layout_lambda](../metafind/train/stage2.py:555) 在真 query 沒有 layout encoder 時，直接回傳 `None`，不量測 norm、不寫不存在的 scalar。Full 的 literal／ratio 分支保持原本規則。新測試以 mean fusion 的 norm 5 與 12 獨立算出 ratio initializer `.1×8.5=.85`；不靠重寫主程式的 AST 副本作驗證。

## 5. Validation Evidence

完整 CPU suite：**1,558 passed、71 warnings、26.55 秒，exit 0，0 failed／0 skipped**；[原始 log](audit/reproduction_training_20260908_cpu.log)。使用 CUDA／HIP 隱藏、HF offline、Mesa llvmpipe，排除 GPU／Claude hook tests；含本機資料測試、tiny full-layout 與實際 CPU Blender。64 個測試檔、1,107 個頂層 test 函式。測試期間已快照的程式來源沒有變更。

圖規格：**2,347 checks，all pass**；[原始 log](audit/reproduction_training_20260908_graph.log)。各項測試只支持自己的 assertion，沒有新增 promotion gate。

全套測試後，另移除通用 [overlay loader](../metafind/eval/run_retrieval.py:1084) 一句不準確的固定 log「unused at evaluation: layout=None」；該 helper 也被有 layout 的場景呼叫，現在只印實際 λ。沒有變更模型計算。此最後一行修正以相關 overlay／custom model／scene composition 測試重驗：**62 passed、14 warnings、1.00 秒**，[log](audit/reproduction_training_20260908_overlay.log)；後述真 custom evaluator 也使用此版。1,558-case 完整 suite 的 source snapshot 早於這個文案修正，保留原始 SHA，不冒稱相同。

Stage 1 的 [execution.json](audit/validation_artifacts_20260907_08/stage1_cpu_real_20260908/execution.json)／[原始 log](audit/validation_artifacts_20260907_08/stage1_cpu_real_20260908/execution.log)：**exit 0，21.78 秒，1 optimizer step**，執行期間 `metafind/` 非 vendor source 未變。首步實際 LR 是原 scheduler 的 `lr_start=1e-6`，不能將 nominal `learning_rate=.0005` 說成這一步的更新率。

[獨立程序還原紀錄](audit/validation_artifacts_20260907_08/stage1_cpu_real_20260908/verification.json)：**exit 0，25.40 秒**；221 個 point-path 參數與 51 個 fuser 參數相較重新初始化有改變，包括 `pc_projection`、query／gallery 兩個 fuser。所有 checkpoint tensor 精確還原且有限；真實重新 forward 得到有限 loss `0.6863776445`、τ=.5。975 個 frozen tensor 在「載入 checkpoint 前後」相同；這項檢查沒有在另一個訓練程序中逐步記錄 frozen tensor，不能擴張成完整訓練期間 bytes trace。

Stage 1 selection 的兩個候選只有執行意義，mean R@1=.5、R@5=1 不作為模型品質結果。使用既有文字／影像 cache，點雲走真正 trainable encoder；沒有假模型、假 scorer 或替換訓練迴圈。

Stage 2 [完整診斷說明](audit/validation_artifacts_20260907_08/stage2_cpu_real_20260908/README.md)：真語義編碼 **28.99 秒**、真 gallery **345.09 秒**、strict preflight **2.17 秒**、真訓練 **21.93 秒**、另程序還原 **22.97 秒**，各階段成功。8 個 gallery 資產無排除；28 條 graph edges 全覆蓋。新 n08 node 與 normalize 後 n11b text 向量最大絕對差 `3.43e-7`。

[真 optimizer 觀測](audit/validation_artifacts_20260907_08/stage2_cpu_real_20260908/optimizer_observation.json)：單步實際 LR=`1e-6`；109 個 optimizer tensors 中 106 有 finite gradients、105 有改變。query fusion 改變 26 個、ESSGNN 78 個、λ 1 個，三部分均有非零梯度。backbone／gallery 實際 frozen，參數 identity 不在 optimizer；沒有聲稱每個 ESSGNN tensor 都有梯度。Driver 包裝 optimizer／freeze／save 入口作觀測，會呼叫原始 production 函式；它沒有替換數學運算，但不能稱「完全沒有 wrapper」。

[Stage 2 還原](audit/validation_artifacts_20260907_08/stage2_cpu_real_20260908/restore_result.json)：109 個 saved tensors 全部精確相同，8×1280 的 **layout-present** 查詢值與儲存前逐位元一致；parent／gallery／node／edge／input identities 均核驗。舊 restore log 的 layout=None 字樣是上段所述固定訊息，不描述此 run 的 `encode_query(drop_layout=False)`。

最後以兩個新 checkpoint 執行公開評估器：[execution](audit/validation_artifacts_20260907_08/custom_table1_cpu_new_stages_20260908/execution.json)、[結果表](audit/validation_artifacts_20260907_08/custom_table1_cpu_new_stages_20260908/evaluation/table.md)、[逐筆核對](audit/validation_artifacts_20260907_08/custom_table1_cpu_new_stages_20260908/verification.json)。**exit 0，51.32 秒**，執行期间 source 不變。沿用前輪已固定的 2 query／6 gallery protocol，三種模型 × 兩觀測 × 七模態，共 **42 組、84 筆 query records**；重算每組 recall 與輸出相同，非對角 target columns=5、4。Stage 2 overlay 後的 gallery 逐位元不變。

本次 mean baseline 的 R@1 全為 100%；兩個只訓練一步的 fusion heads 大多為 0%，different-observations T+I 為 50%。這些小樣本分數沒有研究比較意義，不能與先前已收斂的 checkpoint 比模型優劣，也不能拿來擬合論文數字。其中一個 query 已用於 Stage 1 checkpoint selection；明記為非獨立測試。此 run 首次補到新自訂評估器的 **真 Stage 2-off 權重** 路徑，沒有放寬舊 checkpoint 的拒絕條件。

ESSGNN 新 [NumPy oracle](../tests/models/test_essgnn.py:609) 使用五個節點、非均勻有向 degree、完全孤立節點及不同 edge attributes。兩種已存在 MLP 結構均比對 forward 與 h／x／e／三個 phi 全部參數的中心有限差分；另外驗證反轉方向、將 edge attributes 歸零、錯用更新後 h 都會改變答案。此為單層公式證據，沒有證明訓練收斂或完整場景品質。

## 6. Implementation Choices

- Stage 1 小型 run 使用舊 `attrs_v1`、12-view cache、`same_record`；batch 2、epoch 1 只寫入隔離 artifact，不是正式 recipe。
- Stage 2 診斷沿用新 Stage 1 小型 checkpoint；ProcTHOR node／gallery 使用 canonical `v3_fit` 與 11-view 真影像。這是明示的跨資料設定，不改寫 parent 身分。
- ProcTHOR 關係句沿用歷史 category 描述配對的既有文字，重新以真 frozen CLIP 編碼；不偽造當前 canonical relation 協定的 cache 身分，也不重用舊 embedding 冒充新產物。
- ESSGNN 原文歧義與現有選擇仍以 [公式審查](audit/formula_review_20260907.md) 所指向的原文／決策為準。

## 7. Deviations

本輪沒有變更正式研究選擇。診斷資料量、batch、舊 Stage 1 caption 與歷史 relation text 均限制於隔離 run，不能列為 paper reproduction。既有 Gemma 代替 GPT-4o、frozen CLIP 範圍等差異仍須在正式報告保留。

## 8. Missing or Unreachable Implementation

Table 3 的 GAT 路徑仍明確 `NotImplemented`；paper 未提供足夠架構設定，本輪沒有發明一個 GAT 再標成作者實驗。`w/o iterative retrieval` 使用 Full checkpoint，將 scene request／frozen inputs 的 `mode` 設為 `parallel`，不另訓練一個同名模型。compose CLI 接收 manifest，沒有 `--mode` 旗標。`Train fuser only` 的 query-only 設定還需要 Stage 1 `--freeze-gallery`；單獨 `--train-scope fuser_only` 會更新兩個 fuser，不能視為該列完整契約。

下表是 **OBSERVED IMPLEMENTATION**，逐列對照 [Table 3](paper/metafind_source/3experiments.tex:94)。可達性不代表已取得該列訓練或評分結果。

| 論文列 | 實際切換／必要 parent 設定 | 現況 |
|---|---|---|
| Full | `variant=full`、Transformer parent、ESSGNN、bidirectional loss | Full 會繼承 parent fusion；名字本身不保證 Transformer。 |
| w/o iterative retrieval | Full checkpoint＋scene request／frozen inputs 的 `mode=parallel` | 真實推論使用固定初始圖；trainer 明確拒絕重訓此列。 |
| w/o layout | `variant=no_layout` | 不建 layout encoder／λ；本輪補 pinned λ 的儲存交接。 |
| GAT | `variant=layout_gat` | 明確拒絕，架構 UNKNOWN。 |
| Mean | mean fusion 的 Stage 1 parent | 真 mean forward；Stage 2 核對父設定。 |
| MLPs | MLP fusion 的 Stage 1 parent | 真 MLP forward；精確層結構是既有 implementation choice。 |
| Dropout 10% | Stage 1 parent `p_mask=.1` | 真模態抽樣；不把 Stage 2 scene dropout 改成 .1。 |
| Dropout 50% | Stage 1 parent `p_mask=.5` | 同上，scene dropout 獨立。 |
| Train fuser only | Stage 1 `--train-scope fuser_only --freeze-gallery` | 單用 scope 不代表 query-only；沿用已記錄限制。 |
| Zero padding | Stage 1 parent `missing_modality_representation=zero_pad` | 真 zero 分支，父設定不符會拒絕。 |

入口：[variant registry](../metafind/models/resolve_stage1.py:625)、[Stage 2 compatibility](../metafind/train/stage2.py:1001)、[checkpoint overlay](../metafind/eval/run_retrieval.py:1050)、[parallel composition](../metafind/scene/compose.py:209)。

## 9. Unknowns and Conflicts

作者 Table 1 的 query/gallery 配對、候選與觀測細節仍不完整；本輪不能由單步 run 或分數反推出作者協定，也不能證明過高的 T+I／T+PC 只有一個原因。Table 2 的正式場景清單、鏡頭與評分細節，以及 Table 3 GAT 架構仍未取得。

## 10. Reproducibility Provenance

每個隔離目錄保存 preparation／來源 SHA、實際命令、環境、log 與 checkpoint，位於 `output/validation/stage1_cpu_real_20260908/` 與 `output/validation/stage2_cpu_real_20260908/`。準備腳本不覆寫既有診斷輸出；重播應使用新的輸出目錄。巨型預訓練模型讀取本機既有模型 cache，無網路下載或 GPU 訓練。

Stage 2 的 train／restore phase 有實際 driver／production hash 與環境記錄；較早的 bind／semantic／gallery phase 尚未加入相同 execution-record instrumentation，保留原始 logs、編碼 source identities 與 gallery `/proc` 環境觀測，不回填當時未記錄的 driver hash。整輪命令、SHA、source 差異與保護核對見 [evidence JSON](audit/reproduction_training_20260908_checks.json)。

保護核對沿用先前 580 檔基線：579 檔一致，唯一差異仍是前輪已授權修正的 `semantic_edges_run.py`；本輪沒有新增保護範圍變更。論文 source／manifest 與 upstream IDesign 均維持原狀。最後 [程序快照](audit/reproduction_training_20260908_status.json) 仍顯示 annotation PID `2043427` 與兩條原等候鏈；本輪診斷程序均已結束，未重啟這些現行程序。

## 11. Reproduction Impact

修正使合法 no-layout recipe 產出的 checkpoint 可以被現有 strict 評估／匯出端接收；不改 Full 主線數學。新增真實執行證據補足「tiny／mock 測試過了，但真模型是否能更新與還原」的缺口。仍需正式資料完成後的訓練與評分，才能評估復現程度。

## 12. Verdict

**PARTIALLY VERIFIED。** 本輪有實際修補與訓練驗證進展；完整論文復現尚未完成。

## 13. Required User Decision

本輪已授權的程式修補與診斷不需要新增決策。若要補 Table 3 GAT 為自訂比較，需另行明定其架構並標明與作者設定的未知差異；目前保留不可執行狀態，不默默代填。
