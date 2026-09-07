# 評估入口與歷史實驗

本目錄保存診斷與歷史實驗。程式存在、能執行或曾產生數字，不代表其資料、觀測配對與計分設定就是論文的評估協定。

## 目前入口

| 用途 | 入口 | 範圍 |
|---|---|---|
| 固定清單、同觀測／不同觀測的自訂七模態評估 | [`metafind.eval.custom_protocol`](../../metafind/eval/custom_protocol.py)、[`metafind.eval.custom_table1`](../../metafind/eval/custom_table1.py) | **IMPLEMENTATION CHOICE**；完整規則與限制見[自訂評估文件](../../docs/CUSTOM_TABLE1_EVALUATION.md)。 |
| 既有物件檢索流程 | [`metafind.eval.run_retrieval`](../../metafind/eval/run_retrieval.py) | 消費專案的 eval protocols 與 checkpoint／gallery 記錄；專案選定的協定不補足論文未公開的細節。 |
| ProcTHOR 的 Stage 1／Stage 2 診斷 | [`stage2_procthor_retrieval.py`](stage2_procthor_retrieval.py) | 場景檢索診斷，與物件層級七模態評估分開解讀。 |
| 既有 Table 1 觀測與歸因實驗 | [`exp_type_level_query.py`](exp_type_level_query.py)、[`tabulate_table1_final.py`](tabulate_table1_final.py) | 保留既有實驗及其製表入口；2026-09-07 的等待鏈仍引用它們。 |

**OBSERVED IMPLEMENTATION**：`exp_query_pc_observation.py` 的點雲 helper 仍被其他 probe 匯入；`exp_text_length.py` 也提供共用的實驗構造。這些依賴保留，不表示歷史實驗已通過目前自訂評估的資料身分檢查。

## 已撤回並移除的程式

下列四支程式於 2026-09-07 清理移除；其評估結果已在 2026-09-03 撤回：

- `exp_query_observation.py`
- `exp_text_observation.py`
- `exp_observation_matrix.py`
- `exp_text_template_ablation.py`

**歷史缺陷記錄**：原共用 gallery 路徑將零點雲向量當成存在的 PC 模態送入 fusion，並非使用真正的 gallery 點雲特徵。其私有 scorer 又把相似度同分算作有利於模型，使用 float32，與專案共用 scorer 的 float64／不利同分規則不同。因此不能引用這組結果來判定觀測、gallery 大小或文字內容已解釋論文差距。

撤回後，`gallery_vectors`、`recall`、`main` 已改成拒跑 stub；另外三支仍呼叫這些 stub，無法完成評估。此次刪除不恢復其結果的有效性，也不刪除歷史資料產物。既有說明見[研究筆記的撤回記錄](../../docs/history/METAFIND_NOTEBOOK.md)。

原 `load_tower` 不屬於上述評估缺陷，已搬至 [`metafind.eval.custom_models.load_fusion_tower`](../../metafind/eval/custom_models.py)。它只還原 Stage 1 的兩個 fusion heads；呼叫端仍須供應匹配的 backbone 特徵。它不是完整 checkpoint 載入器。

其餘本目錄的歷史 probe 應依各自輸入、checkpoint、輸出及限制解讀；不要將不同池子或不同觀測的數字併作同一條件。論文內容仍以 `docs/paper/metafind_source/` 為準。
