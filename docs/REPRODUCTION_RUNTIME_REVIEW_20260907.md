# 正式執行路徑接續審查（2026-09-07）

**Verdict: PARTIALLY VERIFIED。** 本輪修正會延續到下一次 Stage 2 訓練的 canonical text 交接錯誤，補 full-layout 整合證據，並第一次用這套新評估入口執行真實預訓練 ULIP-2／既有 Stage 1 checkpoint 的小型 CPU run。未宣稱正式 Table 1–3 復現，也未改 annotation 或目前等候鏈。

接續 [前輪修補報告](REPRODUCTION_CONTINUATION_20260907.md)。前輪是實際修補與驗證進展；本輪重新核對 live handles／來源，沒有把前輪的未完成狀態當成已完成。

## 1. Audit Scope

範圍：annotation 完成後的兩條實際等候鏈、n08／n11b／Stage 2 的文字資料流、gated fusion 排除契約、full ESSGNN 場景交接，以及新七模態 evaluator 的真模型 CPU 執行。以 paper-audit／reproduction-audit 先唯讀核對；確認缺陷後，在使用者既有非標註程式修補授權內實作。

## 2. Authority and Evidence Ledger

| 分類 | 主張與證據 | 界線 |
|---|---|---|
| PAPER FACT | [MetaFind §3.1–3.2](paper/metafind_source/3experiments.tex:15) 指定 tau=.5、七模態 R@1／R@5、Stage 2 scene dropout 30%。 | 不提供完整作者 Table 1 query 配對協定。 |
| IMPLEMENTATION CHOICE | [DL-103 後續主線選擇](../workflow/DECISION_LEDGER.md:6876) 指定 ProcTHOR node／gallery 使用 canonical `v3_fit`。 | 是已記錄選擇，不能改稱 paper fact。 |
| OBSERVED DATA | paper corpus 1,467/1,467 個 modality sidecar text 與 canonical object text 不同；目前 1,467 筆 canonical text 都符合 metadata annotation 的 `serialize_fitted`。 | 只是當時來源快照；caption 步驟仍待執行。 |
| OBSERVED IMPLEMENTATION | 舊 n11b 編碼 renderer sidecar text；等候鏈只重建 canonical map，所以 source hash 有效仍會餵錯已選文字。 | 根因是來源選擇，並非同 encoder 就必須使用同文字的推論。 |
| OBSERVED DATA | 真實 CPU evaluator 2 query／6 gallery，mean＋Stage 1 共 28 個觀測／模型／模態組合完成。 | 舊 attrs_v1 corpus、12 views、非獨立測試集；不測正式新 corpus 品質。 |

## 3. Paper → Specification → Implementation → Validation Matrix

| 要求 | 規格／實作 | 驗證 | 狀態 |
|---|---|---|---|
| 固定 canonical query/gallery 文字 | n11b v2 從 object map 取實際 `v3_fit`，以 metadata annotation 重算比對。 | producer→raw vectors→真 Stage2 `encode_query`、替換來源反例。 | MATCHED，限已選主線／測試範圍 |
| 被排除模態不得影響 fusion | gated 全 inactive row 原本又平均 mask tokens；現在 active weights 會使該列為零。 | 修改被排除 tokens 不得改 output／梯度；含 mixed batch 與 include override。 | MATCHED |
| Algorithm 1 的 full layout 消費 | 真 SG2 cache→checkpoint export／loader→ESSGNN→逐步取回／圖更新。 | 成功／耗盡關係使第三步 query embedding 不同；同權重 layout-off 作對照。 | PARTIAL，tiny 模型／受控 LLM |
| 七模態計分與真權重執行 | 現有 `custom_protocol`／`custom_table1`，固定同資產 UID、兩觀測與 mean／trained fusion。 | 真 ULIP-2 點雲／文字 encoder＋Stage 1；28 份逐 query evidence。 | PARTIAL，2×6 CPU smoke |
| 正式 Table 1–3 | 需要新資料、正式訓練與可比協定。 | 尚未取得正式結果。 | UNKNOWN／PARTIAL |

## 4. Runtime Trace

新 n11b：`procthor_object_text.json`＋`procthor_asset_annotations.json` → 核驗 bytes／逐 asset fitted sentence → `encoded_inputs[asset_id].text` → `backbone.encode_text` → NPZ raw text vectors／fused gallery → `Stage2Data` → `encode_query`。

- 新寫 `source_identity.version=2`，綁 canonical JSON、實際句子、modality records、影像順序／bytes 及宣告 PC。
- renderer sidecar 的舊 text 保留，不再決定文字輸入或文字是否可用；缺 image／declared PC 的排除邏輯保留。
- 實際 encoder 直接消費核驗 snapshot；發布前、Stage 2 與 probe consumer 均重核。
- v1 代表舊 sidecar 文字來源，要求重新建 n11b，不能補 hash 或加 legacy flag 改標 v2。完全沒有 binding 的歷史 index 仍須顯式接受 `legacy_unbound`。
- 目前新 n11b text 入口限定 metadata producer／`v3_fit` 主線；generic／舊文字 map 明確拒絕，沒有靜默 fallback。

現有等候鏈含 metadata→n08→n11b，所以之後新啟動的 Python 會使用修正後 producer；沒有改 Bash 等候流程、重啟程序或預先重建真資料。鏈尾仍是既有 own／weak／partner 診斷，不會自動改用新 custom evaluator。

## 5. Validation Evidence

**完整 CPU suite：1,552 passed、71 warnings、26.18s，exit 0，無 failure／skip。** [原始 log](audit/reproduction_runtime_20260907_cpu.log)。明確排除 GPU／Claude hooks、隱藏 CUDA／HIP、固定 Mesa llvmpipe；包含真 CPU Blender 與新 full-layout tiny case。64 個測試檔、1,105 個頂層 test 函式；不以函式數推算 cases。

**圖規格：2,347 checks，all pass。** [原始 log](audit/reproduction_runtime_20260907_graph.log)。source snapshot 在整套測試期間未變；完整指令、環境、SHA 與保護核對記在 [本輪 evidence JSON](audit/reproduction_runtime_20260907_checks.json)。

真 paper corpus 另做 [唯讀 n11b v2 source capture](audit/reproduction_runtime_20260907_gallery_sources.json)：1,467 個 asset、16,137 個 image references 核對完成，所有 canonical text 符合 metadata 的 fitted sentence；舊 renderer text 則 1,467 筆皆不同。沒有編碼向量或發布 index，這份快照也不代表之後新增 captions 的資料已驗證。

真模型執行記錄：

- [execution.json](audit/validation_artifacts_20260907_08/custom_table1_cpu_real_20260907/execution.json)：CPU、2 threads、`nice -n 10`、CUDA／HIP 隱藏、HF offline；51.74s，exit 0；`metafind/` source 執行期間未變。
- [protocol](audit/validation_artifacts_20260907_08/custom_table1_cpu_real_20260907/protocol/protocol.json)：舊 attrs corpus holdout 排序前 6 UID，前 2 為 query、gallery 反序；不依分數挑樣本。query text 是既有 annotation.description，來源明記，**不宣稱独立標註**。
- [results](audit/validation_artifacts_20260907_08/custom_table1_cpu_real_20260907/evaluation/results.json)、[log](audit/validation_artifacts_20260907_08/custom_table1_cpu_real_20260907/execution.log)：真 ULIP-2 initializer／PointBERT、frozen OpenCLIP text 與已訓練 Stage 1 fusion；影像使用核驗的既有 per-view cache。沒有 monkeypatch 模型／loader／scorer。實際 token 差异與 query/gallery 分離視角均通過。
- 全部 28 組 R@1／R@5 都是 100%。6 個候選的 smoke 沒有足夠難度估计模型品質或融合提升，**這些數字不能替代正式表格**。
- log 的「No pretrained weights loaded」來自 [wrapper 重建 preprocess](../metafind/models/ulip_backbone.py:234) 的另一個 `pretrained=None` factory；被實際使用的模型由 [upstream factory](../metafind/vendor/ulip/models/ULIP_models.py:352) 以 `laion2b_s39b_b160k` 建立，再還原 ULIP／Stage 1 權重。保留原 log，不把此訊息改標為真實模型沒載權重。
- 提供舊 Stage 2 record 的預檢 **exit 1**：缺 embedded metadata；保留原 checkpoint，不放寬 strict loader。成功 run 僅包含 mean／Stage 1。

## 6. Implementation Choices

維持既有 `v3_fit`、frozen CLIP、ProcTHOR T/I、room context 及 D-14／U-26 選擇。gated 全排除輸出零沿用其他 fusion 分支的既有契約，不新增論文假設。新 status 只盤點 cwd 為本 checkout 的程序與指定 corpus，明記不能由 cwd 推出 Python import origin。

## 7. Deviations

真 CPU smoke 使用舊 attrs corpus／12 views、少量既有資產、曾用於選模的 holdout，不能稱作者設定或獨立泛化結果。既有模型替换、CLIP 訓練範圍、未重跑六個 baselines／人工評估等偏離保持原分類，未藉測試通過撤銷。

## 8. Missing or Unreachable Implementation

目前新 evaluator 可實際跑 mean／Stage 1，舊 Stage 2 缺必要 metadata 被拒。新 v10 checkpoint 尚待 annotation 與正式鏈；沒有把舊權重包成新 checkpoint。真 planner、全量三表實驗、正式 scene judge／人工評分尚未完成；full-layout CPU fixture 不填補這些結果。

## 9. Unknowns and Conflicts

作者 Table 1 query 觀測／候選清單／baseline 融合細節、正式 Table 2 scene／camera／judge 協定仍 UNKNOWN；論文正文與附錄的公式歧義保持既有已選解讀。新增 gated 缺陷不涉及目前 Transformer 主線，不能解釋其歷史 T+I／T+P 高分。

## 10. Reproducibility Provenance

[status.py](../tools/status.py) 取代舊 status 的 log 名称、v9 契約與 queue regex 假設，保留 `bash tools/status.sh` 入口；可指定 `--data`／`--json`。只輸出允許的程序資訊，重複 prompt-mode 跟 argparse 取最後值；不使用任意 model 參數或初始 `METAFIND_REPO` 推定 Python 來源。`test_status.py` 覆蓋跨 checkout、真 invocation、v9／v10、損壞 JSON、重複 UID、含空格路徑與唯讀性。

本輪沒有修改 paper source、vendor、CLAUDE／hooks、annotation 程式或真 corpus／等候脚本。初始 580 檔保護快照除前輪已授權 n08 producer 外，其餘 579 保持 SHA。live annotation 與兩個 waiting shell 的當時 PID／來源檔 hashes 記於 evidence JSON；狀態會隨時間改變，不能拿快照認證後續完成。

## 11. Reproduction Impact

n11b 修補使之後 Stage 2 真正消費已選文字協定，避免 source hashes 各自有效卻混用不同版本的文字。gated 修補只影響其全排除交集。新的 full-layout／真模型 CPU evidence 分別縮小語義交接與 pretrained 執行的驗證缺口，沒有證明完整科學結果。

## 12. Verdict

**PARTIALLY VERIFIED。** 程式修补與上述驗證完成；正式訓練、全量評估、論文數值與未公開協定仍未完成。

## 13. Required User Decision

本輪沒有新增需要決定的研究選擇，沿用已明確授權的修補及既有協定。新重大研究歧義仍須另行決定，不能靜默填寫。這份文件可供 Claude 接續；沒有對外傳送訊息，也不代表 Claude 已閱讀或批准。
