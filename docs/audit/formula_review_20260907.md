# 公式與消費端審查（2026-09-07）

範圍：MetaFind 正文 Eq. 1–8、附錄等變性公式 → fusion／loss／dual tower／ESSGNN → Stage 1／2 使用處，以及此次確認的 no-layout 消費端。先依 paper source 核對，再檢查目前程式與 CPU 反例。本文不是「所有公式已無問題」或完整論文復現的聲明；既有正文／附錄矛盾、研究選擇與實際訓練結果須分開判讀。

下列行號以本次工作樹為準；paper source 是內容權威，decision／contract 與 tests 都不是 paper authority。

## Eq. 1–8 crosswalk

| 公式與 PAPER FACT | OBSERVED IMPLEMENTATION | 驗證與限制 |
|---|---|---|
| Eq. 1：對 gallery 的 similarity 取 argmax；[2methdology.tex:7](../paper/metafind_source/2methdology.tex) | [compose.py:202](../../metafind/scene/compose.py) 計算 cosine、按分數排序取第一個 asset，逐步更新已放置 graph。 | [test_scene_composition.py:51、247](../../tests/eval/test_scene_composition.py)：手算逐步結果、top-5 cosine。Cosine 與同分保持 gallery 順序是 IMPLEMENTATION CHOICE；正文 :10 只稱 similarity function。 |
| Eq. 2：`h_next = h + sum f_h(old h_i, old h_j, d, e)`；[正文 :51](../paper/metafind_source/2methdology.tex) | [ESSGCL :424–425](../../metafind/models/essgnn.py) 的 `sec25_two_mlp` 分支照此更新。主線 `appendix_shared_msg` 使用附錄另一套 message 參數化。 | 不能將正文與附錄兩個模型稱為同一公式；見下方 U-26。 |
| Eq. 3：以舊座標差、更新後的 `h_next` 計算座標更新；[正文 :52–54](../paper/metafind_source/2methdology.tex) | [ESSGCL :428–435](../../metafind/models/essgnn.py) 在 `coord_feat="updated"` 下使用新 h；`f_x` 輸出 scalar，座標依鄰居求和。 | 新 h／舊 h 分支確實不同。Scalar codomain、距離平方與 h0 解讀須各自分類，不能以等變性測試消除原文矛盾。 |
| Eq. 4：旋轉／平移下，x 等變而 h 不變；[正文 :61–65](../paper/metafind_source/2methdology.tex) | [ESSGNN :644–653](../../metafind/models/essgnn.py) 主線以 invariant semantic h0、distance 與 scalar coordinate update 維持此性質；輸出 pooling(h)，不輸出 x。 | [test_essgnn.py:102](../../tests/models/test_essgnn.py) 包含兩個 family 的 SE(3) 測試。這不證明 scaling invariance，也不能讓 literal Concat(x,t) 成為 invariant。 |
| Eq. 5：query→gallery InfoNCE，分母為 gallery batch；[正文 :77–79](../paper/metafind_source/2methdology.tex) | [losses.py:182–204](../../metafind/models/losses.py)：normalize q/g，`exp(logit_scale) * q @ g.T`，單向 cross entropy；[stage1.py:2703](../../metafind/train/stage1.py) 設 `bidirectional=False`。 | [獨立 scalar oracle :364–417](../../tests/models/test_fusion_losses.py)：不用 torch CE／normalize 計算期望值，並對 q、g 做中央有限差分；涵蓋非對稱向量與非自逆 label permutation。 |
| Eq. 6：`Fusion + lambda * layout`，lambda 是 learnable scalar；[正文 :85–89](../paper/metafind_source/2methdology.tex) | [dual_tower.py:240、269–276](../../metafind/models/dual_tower.py) 使用原值 Parameter、直接相加；沒有 exp(lambda) 或正值約束。Layout=None 時返回融合向量。 | [test_dual_tower.py:52、66、78、90](../../tests/models/test_dual_tower.py) 檢查精確 residual、lambda gradient、負值／零值與 layout-off。Lambda 初值與 pooling 並非此式提供的值。 |
| Eq. 7：q→g 與 g→q 兩個方向；[正文 :94–99](../paper/metafind_source/2methdology.tex) | [losses.py:211–215](../../metafind/models/losses.py) 用轉置 logits 與 labels 的 inverse permutation；Stage 2 loss 為雙向。 | Scalar oracle 同時算兩套不同分母；雙向只支援每個 query／gallery 各一個 positive 的 bijection。正文 :99 將 B 稱為 negatives，但 Eq. 5 :79 稱 gallery batch；現有實作分母包含 positive，不能把 prose 的歧義宣稱已由 tests 裁決。 |
| Eq. 8：兩方向 loss 的平均；[正文 :100–102](../paper/metafind_source/2methdology.tex) | [losses.py:216](../../metafind/models/losses.py) 是 `0.5 * (loss_q2g + loss_g2q)`；[stage2.py:1516](../../metafind/train/stage2.py) 呼叫此 loss。 | 同一獨立 oracle 檢查數值與 q／g gradients；不是僅比對同一函式輸出的欄位。 |

## 附錄與保留的 interpretation／UNKNOWN

- **PAPER FACT：** [appendix.tex:25–29](../paper/metafind_source/appendix.tex) 的 Eq. 9 假設 h0 invariant、e 與座標無關；Eq. 10 (:32) 以舊 h 和平方距離產生共同 `m_ij`；Eq. 11／12 (:38、44) 是距離與 message invariance；Eq. 13 (:50) 用 `phi_x(m)` 更新座標；Eq. 14 (:65) 用 `h + sum phi_h(m)` 更新特徵；Eq. 15 (:73) 重述等變性。程式 [ESSGCLShared :501–520](../../metafind/models/essgnn.py) 對應此共同 message 路徑，沒有偷偷改為使用更新後 h。
- **DEVIATION／既有 decision：** 正文 [2methdology.tex:44、54](../paper/metafind_source/2methdology.tex) 的 Concat(x,t)、dimension 與 `f_x -> R^3`，不能與附錄 invariant h0／scalar 實作合稱 literal faithful。保留 [D-14，graph_spec.yaml:417](../graph/graph_spec.yaml) 的 semantic h0；保留 [U-26，graph_spec.yaml:1889](../graph/graph_spec.yaml) 已記錄的 appendix primary／正文對照；scalar `f_x` 是 [USER-ratified IMPLEMENTATION CHOICE，DECISION_LEDGER.md:886](../../workflow/DECISION_LEDGER.md)，不是聲稱 upstream 替 MetaFind 裁決。
- **IMPLEMENTATION CHOICE／UNKNOWN：** 正文 Euclidean distance (:54) 與附錄 squared distance (:32) 不同；正文用 N(i)，附錄 :50／65 寫 j≠i。程式保留 resolved protocol／目前 graph neighborhood，不新改完整圖、距離或更新順序。MLP depth／activation、IO projections、layer sharing、pooling、Transformer depth／readout、prefusion normalization 等未由 MetaFind 公式唯一決定。`normalised_sum` 是 [DL-077 item 8](../../workflow/DECISION_LEDGER.md) 與後續 [lambda ratio 說明 :6162](../../workflow/DECISION_LEDGER.md) 的決定；不是正文 :56 的 `Pooling` 已指名該操作。
- **OBSERVED IMPLEMENTATION：** pooling 僅讀 hL，故 independent layers 最後一層 coordinate head 無 loss path；前層 coordinate heads 可透過下一層距離取得梯度。這是現有公式／readout 結構的結果，並非本輪修掉的 bug。[test_essgnn.py:327、403、497](../../tests/models/test_essgnn.py) 的梯度測試主要覆蓋 sec25 分支；不能以此宣稱所有 family 都有獨立數值 oracle。本輪另做兩 family 的逐邊 Python loop／非 norm-only 梯度 inline 檢查，未建立持久 ESSGNN oracle artifact，故不引用精確浮點誤差數字。
- **PAPER FACT 與 choice 分開：** [正文 :75](../paper/metafind_source/2methdology.tex) 是每模態獨立 30% masking；:89 是每 batch 30% scene dropout。[fusion.py:118–160](../../metafind/models/fusion.py) 與 [stage2.py:1499](../../metafind/train/stage2.py) 分別實作兩者。是否允許全 masked query 由明確 `allow_all_masked` 設定決定，不可把強制保留一個模態稱為仍完全獨立。[DL-104 :6902](../../workflow/DECISION_LEDGER.md) 的 Stage 2 text+image 與 gallery 排除 undeclared slots 是已批准 IMPLEMENTATION CHOICE；本輪未改此科學設定或補造 ProcTHOR PC。

## 本輪 confirmed bugs 與修補

1. **固定 Stage 2 tau 被父 state 覆寫。** 實際 partial-loader CPU 反例：父 learned tau=.07、child fixed recipe=.5，修補前 child 實際仍為 .07，record 卻寫 .5。[3experiments.tex:15](../paper/metafind_source/3experiments.tex) 明寫 all experiments tau=.5。[restore_stage1_for_stage2 :1094](../../metafind/train/stage2.py) 現先按已綁定父 recipe 建 temporary parent loss 並用原 Stage 1 loader 還原；fixed child 保持自身 recipe，learnable child 保留原有 parent raw-scale 初始化選擇。`temperature_init` 記 requested／raw initial／effective initial／log scale／source／clamp；`effective_values.init_temperature` 改記實際初始有效值。Stage 1 resume loader 未改。[新測試](../../tests/train/test_stage2_temperature.py) 涵蓋四種 fixed／learned scope 組合、缺 state、缺 metadata、父固定 recipe 衝突及不同 clamp。
2. **Clamp 後的實際 tau 與 log 不同。** 原 `.temperature` 回傳 raw reciprocal；learned scale=1000、max=100 時，forward tau=.01、舊 log=.001。Stage 2 的 train log 現使用該次 forward 回傳的 `out["temperature"]`；Stage 1 log 由 root 同步修補。此處不更改 fixed tau 的 clamp 語意，也不把 clamp 本身稱為論文設定。
3. **雙向 labels 可留下未初始化 inverse。** `[0,0,2]` 使 `empty_like` 的一格未賦值；CPU deterministic fill 反例得到 out-of-bounds target。現在 [losses.py:168–180](../../metafind/models/losses.py) 檢查 long／shape／device／range，雙向另需完整 permutation。單向保留合法重複 positive indices，不發明多 positive loss。現有 trainers 用預設 arange，原本未觸發此公開 API 缺陷。[labels regressions](../../tests/models/test_fusion_losses.py) 與 scalar oracle 同時覆蓋。
4. **合法 no_layout checkpoint 被 full-only consumers 拒絕。** 原 loader 索引 `lambda_init=None`；overlay 對不存在的 layout weight 呼叫 `.item()`；probe 固定產生 S2-on。[stage2_layout_settings :104](../../metafind/train/gallery_index.py)／[retrieval loader :1001](../../metafind/eval/run_retrieval.py) 現按 branch 還原，new `use_layout` 綁 embedded metadata；舊明確 `variant_id=no_layout`＋None 可辨認，full 缺 lambda 仍拒絕。Probe 不偽列 S2-on，記 `disabled_by_checkpoint`。[consumer regressions](../../tests/eval/test_eval_stage2_row.py) 驗證真 metadata／sha、query overlay、gallery 不變與 CPU CLI control flow。新增 temperature／layout／semantic provenance 欄位也不可只在舊 sidecar 單獨加上；[identity tests](../../tests/eval/test_retrieval_checkpoint_identity.py) 覆蓋有／無 embedded metadata 的 legacy 情況。
5. **No-layout export 無法進 composition。** 原 manifest 要求非空 ESSGNNConfig，compose 強制分支，且 QueryTower.lam 在無分支時會拋錯。現在 [compose.py:62、137、264](../../metafind/scene/compose.py) 共用 physical graph，允許真 `essgnn=None` 模型在 `use_layout=False` 下 compose；不讀／不要求未使用 node／edge tensors 或 cache，不造 dimensions，記 `semantic_status=not_used`、trace lambda=0。Slot、canonical asset text、support／kNN trace 保留。Full 模型 layout-off 仍做原語義驗證。[scene tests :103、154、397、421](../../tests/eval/test_scene_composition.py) 覆蓋真 export→manifest→compose、禁止 unused mapping 讀取、full 對照 guard 與錯誤啟用 layout 的拒絕。

## Scoped validation

下列最新針對性 CPU 執行：**223 passed，35 warnings，1.41s**。包含本輪修補與同時整合的相關測試；不是全套 repository 測試的替代。沒有 GPU、真 ULIP-2 訓練、canonical corpus 重建，或 Table 1／2 實驗結果。完整整合測試由主流程另行記錄。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
/home/kyzen/miniconda3/envs/MetaFind/bin/python -m pytest -q -p no:cacheprovider \
  tests/train/test_stage2_temperature.py tests/models/test_fusion_losses.py \
  tests/train/test_train_stage2.py tests/train/test_stage2_guards.py \
  tests/eval/test_eval_stage2_row.py tests/eval/test_retrieval_checkpoint_identity.py \
  tests/eval/test_custom_models.py tests/train/test_stage2_input_identity.py \
  tests/eval/test_scene_composition.py
```

可只重播獨立 Eq. 5／7／8 數值與 gradient oracle：在相同 CPU 環境下執行 `python -m pytest -q -p no:cacheprovider tests/models/test_fusion_losses.py -k values_and_gradients_match_independent_scalar_oracle`。該 oracle 證明選定 cosine／batch convention 的運算與梯度；不能證明作者的未公開 hyperparameters、checkpoint、完整 query/gallery protocol 或正文／附錄矛盾已獲外部解答。
