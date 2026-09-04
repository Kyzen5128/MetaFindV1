# Table 1 報告（2026-09-04）—— 主線 P1s，唯一一次 `--unseal`

Kyzen ✅（2026-09-04 ~17:40）：主線選定後，A／A20／B 只跑一次。本檔是那一次的結果。之後任何數字都不再是 held-out。

## 1. 結果（cosine，float64，R@k，%；Text2Shape 的 RR@k／NDCG@5 在 unit 向量上與 R@k 同值，見 §4）

**A：test → test（query 4,569，gallery 4,569）**

| | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| R@1 | 33.9 | 65.0 | 98.0 | 86.5 | 99.7 | 98.4 | 99.9 |
| R@5 | 68.6 | 86.0 | 99.8 | 97.3 | 100.0 | 99.8 | 100.0 |
| NDCG@5 | 52.3 | 76.6 | 99.0 | 92.6 | 99.9 | 99.3 | 100.0 |

**A20：test → holdout（query 4,569，gallery 9,138 = 論文的 20% 尺寸）**

| | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| R@1 | 25.0 | 57.2 | 96.7 | 80.4 | 99.5 | 97.6 | 99.6 |
| R@5 | 55.7 | 81.3 | 99.6 | 95.3 | 100.0 | 99.7 | 100.0 |
| NDCG@5 | 41.0 | 70.4 | 98.4 | 88.9 | 99.8 | 98.8 | 99.9 |

**B：test → full（query 4,569，gallery 45,692，含 36,554 個訓練資產當干擾）**

| | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| R@1 | 9.6 | 38.4 | 91.9 | 63.0 | 98.1 | 93.2 | 98.8 |
| R@5 | 27.8 | 63.7 | 98.6 | 86.2 | 99.9 | 99.0 | 100.0 |
| NDCG@5 | 18.9 | 51.9 | 95.7 | 75.7 | 99.2 | 96.6 | 99.5 |

**論文 Table 1（MetaFind 列，R@1）**：text 15.2、image 29.7、pc 75.1、text+image 31.1、text+pc 44.5、image+pc 73.5、full 81.5。

## 2. 一句話結論

沒有重現論文的形狀。論文 text+pc（44.5）**低於** pc（75.1）；我們三個協定 text+pc 都**高於** pc，任何含點雲的格都 ≥ 92。A20 的 gallery 尺寸與論文相同（9,138 ≈ 20%），數字仍是這個形狀。只有 text 格（A20 25.0；B 9.6）與 image 格落在論文附近。

原因（INFERENCE，證據在 `NOTE_20260904_ULIP2_CHECK_AND_STAGE_SUMMARY.md` §一.4 與 `docs/audit/RETRIEVAL_EVALUATION_DEFINITION_20260904.md`）：query 的點雲與 gallery 的點雲是同一份資產的同一朵雲，pc 一個人就決定名次；論文的 query 觀測是怎麼來的，論文沒寫（U-09、query construction UNKNOWN）。

## 3. 可追溯資訊

| 項 | 值 |
|---|---|
| checkpoint | `/home/kyzen/metafind_data_attrs/outputs/checkpoints/pilotP1s_split801010_lr1e-4_20260904/stage1_best.pt`，sha256 `074f8d98e33a4fac…`，epoch 9／10，val mean R@1 0.8306 選出 |
| 訓練 code | `1c3c654`，clean；`--query-observation second_observation --query-image-policy single_view --lr 1e-4 --epochs 10`；seed 20260816；tower_sharing `shared_backbone_separate_fusion`；文字模板 attrs_v1；影像 12 視角平均（gallery）／單視角（query）；fp32；RTX 5090，torch 2.12.1+cu132 |
| 評估 code | `6eef8c7`；`code_dirty=True` 只因 `workflow/DECISION_LEDGER.md`（DL-092）尚未 commit，無程式碼改動 |
| split | `data/outputs/splits.json` sha256 `c03e898f…`，80/10/10 D-3b：train 36,554／val 4,569／test 4,569；paper seed 20260816、val seed 20260904 |
| gallery | 發布索引 `gallery_index_074f8d98e33a4fac.npz` sha256 `330a3770…`，encoder sha `60c86751…`，G4 PASS 記錄 `logs/gates/G4_gallery_freeze.yaml` sha256 `cc446b63…` |
| 指令 | `logs/table1_P1s_unseal.sh`（n11 → G4 → n12 → run_retrieval A/A20/B `--unseal`），log `logs/table1_P1s_unseal_20260904.log` |
| 輸出 | `/home/kyzen/metafind_data_attrs/outputs/eval/table1_P1s_unseal_20260904/{table1.json, diagnostics.json, per_query_*.jsonl}` |
| 第一次嘗試 | 21:1x 失敗：正式協定需要發布索引，當時沒有；log 留在 `table1_P1s_unseal_20260904.attempt1_no_index.log`。沒有數字產生 |

## 4. 指標說明

- R@k：正解（同 UID）名次 ≤ k；ties 對模型不利。
- Text2Shape 的 RR@k（recall rate，不是 reciprocal rank）與 NDCG@5：照 kchen92/text2shape 逐字複製（`metafind/eval/text2shape_eval.py`）；在單位向量上 RR@k ≡ R@k。raw-dot 版本在發布索引的 gallery 上不可用（索引只存向量，未存範數），故此表無 raw-dot 欄；val 上的 raw-dot 見 `eval_pilotP1s_split801010_lr1e-4_20260904/table1.json`。
- 協定定義與論文對應：`docs/audit/RETRIEVAL_EVALUATION_DEFINITION_20260904.md`。A 與 B 的 gallery 是本專案的假設（U-09）；A20 是論文 20% 的尺寸，但論文沒寫 gallery 是哪一組。

## 5. 旁列的 arm（只在 val，未動 test）

見 DL-092 的表：P13b（分塔＋bf16）val 平均 83.4；P12b（文字改寫）80.9；P1s 83.1。三者形狀相同。
