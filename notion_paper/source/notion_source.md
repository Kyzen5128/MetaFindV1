更新：2026-09-08。這頁整理目前已實作的評估、接下來的資料與訓練順序，以及結果可以支持的結論。**已跑通真實模型的小型 CPU 診斷；新的正式 corpus、完整訓練與正式評估結果尚未因此完成。**
本頁的「自訂 Table 1」是明確固定規則的同資產檢索實驗。它沿用論文七種模態與 R@1／R@5，但不宣稱已恢復作者未公開的全部 query／gallery 設定。舊 P1s 的高分原因尚未完整查明，不能把修過的程式缺陷直接等同於歷史分數差距的根因。
<table_of_contents/>
## 1. 先分清楚三件事
<table header-row="true">
<tr>
<td>工作</td>
<td>問的是什麼</td>
<td>完成時的主要證據</td>
</tr>
<tr>
<td>Objaverse 自訂七模態檢索</td>
<td>在固定候選中，能否找回 query 所屬的同一 UID？</td>
<td>protocol、三列模型的 R@1／R@5、逐 query rank、checkpoint／資料身分</td>
</tr>
<tr>
<td>ProcTHOR layout 檢索診斷</td>
<td>同一場景條件下，S1、S2-off、S2-on 的檢索有何差異？</td>
<td>room context、test houses、assetId 正解、三種模型狀態與來源紀錄</td>
</tr>
<tr>
<td>完整場景生成與評分</td>
<td>逐步取回、放置物件後，場景是否更一致、合理？</td>
<td>composition trace、真 GLB placement、render、固定評審協定與四維分數</td>
</tr>
</table>
**PAPER FACT：** [MetaFind §3.1–3.3](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/paper/metafind_source/3experiments.tex) 指定七種 query 條件、R@1／R@5，並將 Objaverse 物件檢索和 ProcTHOR 場景品質分開評估。論文的 Stage 2 權重在 Objaverse 上使用時關閉 ESSGNN；這與在真場景中啟用 layout 是兩個不同實驗。
**IMPLEMENTATION CHOICE：** exact-UID 正解、下面兩種觀測版本、mean baseline 的正規化順序、實際候選清單與同分處理，都是本地明確規則。作者完整的候選清單、觀測配對及 baseline 融合細節仍屬 **UNKNOWN**。
## 2. 新自訂 Table 1：兩種觀測 × 三列模型 × 七種模態
### 2.1 七欄只改 query 可見模態
<table header-row="true">
<tr>
<td>顯示欄位</td>
<td>程式 condition</td>
<td>Query 提供</td>
<td>Gallery 提供</td>
</tr>
<tr>
<td>T</td>
<td>`text`</td>
<td>文字</td>
<td>T＋I＋PC</td>
</tr>
<tr>
<td>I</td>
<td>`image`</td>
<td>圖像</td>
<td>T＋I＋PC</td>
</tr>
<tr>
<td>PC</td>
<td>`pc`</td>
<td>點雲</td>
<td>T＋I＋PC</td>
</tr>
<tr>
<td>T＋I</td>
<td>`text+image`</td>
<td>文字、圖像</td>
<td>T＋I＋PC</td>
</tr>
<tr>
<td>T＋PC</td>
<td>`text+pc`</td>
<td>文字、點雲</td>
<td>T＋I＋PC</td>
</tr>
<tr>
<td>I＋PC</td>
<td>`image+pc`</td>
<td>圖像、點雲</td>
<td>T＋I＋PC</td>
</tr>
<tr>
<td>T＋I＋PC</td>
<td>`full`</td>
<td>三種模態</td>
<td>T＋I＋PC</td>
</tr>
</table>
Gallery 始終是完整觀測。缺少的 query 模態不可以拿另一個 UID 的資料補入；也不能因為某一欄缺資料而悄悄減少 query 數量。
### 2.2 三列模型
<table header-row="true">
<tr>
<td>模型 ID</td>
<td>實際載入及融合方式</td>
<td>能比較什麼</td>
</tr>
<tr>
<td>`ulip2_available_mean`</td>
<td>只載官方 ULIP-2 預訓練權重；每個可用模態先 L2 normalize，取 mean，再 L2 normalize。不存在的模態不加入平均，不放 learned mask token。Gallery 用它自己的預訓練 point encoder。</td>
<td>在這個明確觀測／候選設定下，沒有本地訓練 fusion 的參考表現</td>
</tr>
<tr>
<td>`stage1`</td>
<td>按 checkpoint 綁定的 constructor、初始化、point encoder、query／gallery fusion 與缺模態策略還原，之後凍結並 eval。</td>
<td>完整 Stage 1 訓練系統相對 mean 的差異</td>
</tr>
<tr>
<td>`stage2_layout_off`</td>
<td>驗證 S2 的 parent S1 與權重 bytes，只覆蓋 query fusion，令 `layout=None`；使用與 S1 逐位元相同的 gallery 向量。</td>
<td>S2 更新後的 query fusion 在無 layout 物件檢索上的變化</td>
</tr>
</table>
Stage 1 相對 mean 的提升，可能同時來自 point encoder 微調與 fusion 訓練，不能全歸給 fusion。Stage 2-off 不測 ESSGNN 對場景品質的效果，也不能因為它下降，就直接判定 layout 機制無效。
這個 mean 是本地明確的 baseline 改造，**不是聲稱 MetaFind 作者 Table 1 的 ULIP 那一列已完整重現**。論文確實說額外加入 mean pooling，但沒有完整交代這裡採用的全部正規化與 gallery 規則。
目前 v1 支援 frozen CLIP、mean 圖像快取、每個 tower 一個 image token。Trainable CLIP、多 image tokens、不相容的文字／視角設定、不正確的父 checkpoint，以及缺必要身分的 legacy S2 會拒絕。S2 不接受會連帶修改 gallery 的 tied fusion。
文字由當次實際 backbone 重新編碼，點雲走各模型自己的 point path；圖像使用已核對 bytes 的既有 per-view cache。若歷史 cache 沒有記錄 OpenCLIP 權重 bytes，結果會保留這個來源限制；今天驗過官方 initializer，不能追溯證明歷史 image cache 當時使用了相同權重。
實作：[custom_models.py](https://github.com/Kyzen5128/MetaFindV1/blob/main/metafind/eval/custom_models.py)、[custom_table1.py](https://github.com/Kyzen5128/MetaFindV1/blob/main/metafind/eval/custom_table1.py)。
### 2.3 兩個觀測版本
<table header-row="true">
<tr>
<td>因素</td>
<td>`same_observation`</td>
<td>`different_observations`</td>
</tr>
<tr>
<td>Query 文字</td>
<td>Canonical 序列化全文</td>
<td>外部提供、描述同一 UID 的不同文字</td>
</tr>
<tr>
<td>Gallery 文字</td>
<td>Canonical 全文</td>
<td>同一份 canonical 全文</td>
</tr>
<tr>
<td>Query 圖像</td>
<td>全部 cached views 的特徵平均</td>
<td>該 UID 固定保留的一個 view</td>
</tr>
<tr>
<td>Gallery 圖像</td>
<td>全部 cached views 的特徵平均</td>
<td>排除每個候選 UID 自己保留 view 後，其餘 views 的平均</td>
</tr>
<tr>
<td>Query 點雲</td>
<td>Canonical 點雲</td>
<td>同一 mesh 的另一份 10,000 點 xyzrgb 採樣</td>
</tr>
<tr>
<td>Gallery 點雲</td>
<td>Canonical 點雲</td>
<td>同一份 canonical 點雲</td>
</tr>
<tr>
<td>UID 清單與順序</td>
<td>固定</td>
<td>與 same 版本完全相同</td>
</tr>
</table>
固定視角為 `k = uid_seed(uid) % n_views`。11 views 是 query 1／gallery 10；12 views 是 1／11。每個候選資產都先按自己的 UID 決定保留視角，不隨當前 query 改變，因此 gallery 仍可預先編碼。
兩版本都從相同快取的逐 view 矩陣轉 float32 後平均，不先逐 view normalize。它不直接沿用舊快取中另外量化的 `image` 平均欄位，所以不能說與歷史 evaluator 的 image 向量逐位元相同。
不同 query 文字須經**實際載入的 CLIP tokenizer** 證明有效 token IDs 不同。只在被截斷的尾段改字不合格。不同文字也不等於獨立人工標註；文字由誰產生、是否來自原 annotation，須如實記錄。
不同點雲使用 `uid_seed(uid) + 1000003` 重採樣，凍結 mesh、sampler 與程式身分。它仍來自同一 mesh，不能稱為獨立掃描；若舊 canonical sidecar 缺完整 producer 身分，也不能聲稱兩份點雲嚴格「只有 seed 不同」。
**解讀界線：** different 版本同時改變文字、圖像與點雲。對某個融合欄位，分數差是該組可見觀測的共同變化，不是自動分離出的單因素因果效果。要獨立量測文字或視角貢獻，需另固定其他因素做明示對照，不能把新表本身當成完成全部歸因。
## 3. 固定 UID、GT、R@k 與同分規則
1. `query_uids.json`、`gallery_uids.json` 必須非空且沒有重複；每個 query UID 都必須存在 gallery。
2. 一份 protocol 下，兩種觀測、所有模型、全部七欄共用完全相同的清單與順序。缺資料直接失敗，不在每個模型內各自過濾。
3. GT 是 `query_uid → gallery column`。例如 query 是 `[A,B]`、gallery 是 `[B,C,A]`，target columns 是 `[2,0]`，不是矩陣對角線。
4. 排名使用 float64 cosine；`rank = 1 + 嚴格高於正解的非正解數 + 與正解同分的非正解數`。同分採對模型不利的排名，不偷偷把 target 排在同分候選最前面。
5. `R@k = 100 × rank ≤ k 的 query 數 / 全部 query 數`。JSON 中的 R@1／R@5 為百分比。
6. 分數衡量找回同一個資產，不是分類正確率，也不是「取回不同 UID 但同樣適合這個房間」的品質評分。
結果會另報相對 mean 的 R@1 百分點差，以及融合相對其可見單一模態最佳結果的差。高分本身不是錯誤；若 mean 已接近滿分，這個設定辨別融合優勢的能力可能有限，應保留結果與限制，不以調難度追逐論文數字。
預設結果明記 `evaluation_role=development_comparison`、`independent_test_status=not_certified`。即使檔名叫 holdout、CLI 用 `--unseal`，曾用來挑 checkpoint 或調規則的 UID 也不能重新宣稱未見測試資料。
實作：[custom_protocol.py](https://github.com/Kyzen5128/MetaFindV1/blob/main/metafind/eval/custom_protocol.py)、[score_streaming](https://github.com/Kyzen5128/MetaFindV1/blob/main/metafind/eval/run_retrieval.py)。
## 4. T＋I、T＋PC 為何偏高：目前能證明與不能證明的事
### 4.1 先保留真正的舊分數與條件
以下是**歷史 P1s 診斷**，不是新 custom evaluator 的輸出。兩份原始 JSON 均是 9,138 query 對 9,138 holdout gallery，parent 為 `pilotP1s_split801010_lr1e-4_20260904/stage1_best.pt`。S2 使用其舊 `S2_P1s_none_ft5e-5_allhouses_20260906/stage2_full.pt`。
<table header-row="true">
<tr>
<td>舊 own(attrs)／own view，R@1 %</td>
<td>T</td>
<td>I</td>
<td>PC</td>
<td>T＋I</td>
<td>T＋PC</td>
<td>I＋PC</td>
<td>Full</td>
</tr>
<tr>
<td>P1s S1</td>
<td>24.95</td>
<td>57.33</td>
<td>96.74</td>
<td>80.17</td>
<td>99.54</td>
<td>97.55</td>
<td>99.65</td>
</tr>
<tr>
<td>P1s S2-off</td>
<td>16.38</td>
<td>31.52</td>
<td>66.78</td>
<td>45.96</td>
<td>69.40</td>
<td>69.58</td>
<td>71.01</td>
</tr>
<tr>
<td>論文 w/o ESSGNN</td>
<td>13.8</td>
<td>11.7</td>
<td>75.1</td>
<td>17.2</td>
<td>44.5</td>
<td>45.8</td>
<td>51.7</td>
</tr>
<tr>
<td>論文 w/ ESSGNN，在 Objaverse 關閉 layout</td>
<td>11.3</td>
<td>10.5</td>
<td>63.2</td>
<td>15.9</td>
<td>41.2</td>
<td>42.0</td>
<td>48.2</td>
</tr>
</table>
舊資料來源是本機 `output/look/table1_final_P1s_S1head_holdout.json` 與 `table1_final_P1s_S2head_holdout.json`。P1s record 記 36,554 train、4,569 selection；所以這份 9,138 holdout 不能整體標為不曾參與選模的獨立集合。舊 corpus／checkpoint 的設定也不等於新的 v10／11-view 主線。
### 4.2 已有證據支持的機制
- **T＋PC：PC 自己就很強。** S1 的 PC 已是 96.74%，T＋PC 99.54%。把 query 改成同 mesh 第二次採樣後，PC 仍為 96.25%，T＋PC 仍為 99.15%。因此，「只是重用完全相同的一組點」不足以解釋全部高分；同資產幾何仍能提供很強的辨識訊號。
- **T＋I：本地 image 觀測比論文那一列容易辨識。** S1 的 I 為 57.33%，論文是 11.7%；自己的詳細文字和自己的 view 融合後得到 80.17%。這支持先檢查觀測與資料條件，而不是只檢查 fusion 模組名稱。
- **文字弱化確實影響 T＋I，但不能單獨解釋 T＋PC。** 同一份 holdout 中，換成自己的 category＋size 文字、自己的 thumbnail、canonical PC，T＋I 降至 53.70%；T＋PC 仍有 97.09%，Full 97.13%。這是反證：不能簡化成「文字太豐富，所以所有融合欄都高」。
- **S2 更新與無 layout 評估存在明顯差異。** 相同歷史資料設定下 S2-off 的多欄下降，符合需要區分 S1 與 S2 head 的事實；但這仍不等於完整證明論文所述 feature-attribution drift 是本地差距的唯一原因。
### 4.3 另一份單因素替換診斷支持到哪裡
`output/look/exp_attribution_P1s_val.json` 確實使用同一個 P1s checkpoint，但 query／gallery 是 **4,569 dev_val 對 4,569 dev_val**，image 為 mean。它與上面的 9,138 holdout／single view 不是同一個條件。
<table header-row="true">
<tr>
<td>固定其他因素，只替換下列觀測；R@1 %</td>
<td>I</td>
<td>PC</td>
<td>T＋I</td>
<td>T＋PC</td>
</tr>
<tr>
<td>A：自己的 T、mean I、PC</td>
<td>86.32</td>
<td>97.90</td>
<td>93.76</td>
<td>99.74</td>
</tr>
<tr>
<td>B：只把 T 換成 partner</td>
<td>86.32</td>
<td>97.90</td>
<td>58.96</td>
<td>94.40</td>
</tr>
<tr>
<td>C：只把 I 換成 partner</td>
<td>0.61</td>
<td>97.90</td>
<td>6.04</td>
<td>99.74</td>
</tr>
<tr>
<td>D：只把 PC 換成 partner</td>
<td>86.32</td>
<td>0.07</td>
<td>93.76</td>
<td>0.24</td>
</tr>
</table>
這組固定池、固定 checkpoint 的替換，支持「這個模型的 T＋I 很依賴 image 身分、T＋PC 很依賴 PC 身分」。但 partner 是另一個資產，GT 仍是原 UID，這是故意破壞觀測的歸因診斷，**不是合理的同 UID 不同觀測評估，更不是作者 query 的證據**。93.76% 也不能拿來替換截圖的 80.17%。
### 4.4 仍然 UNKNOWN
目前未找到能完整解釋 S1 Full 99.65% 對論文 51.7%、T＋I 80.17% 對 17.2%、T＋PC 99.54% 對 44.5% 的唯一根因。作者的實際觀測、candidate UID、baseline 正規化與精確 checkpoint 選擇未公開完整。資料版本、pretraining、選模、觀測難度與 fusion 訓練都可能有影響，現有證據尚未把它們全部分離。
先前修補過 checkpoint constructor／來源身分等真實缺陷；修補是必要工程工作，但不能據此宣稱舊截圖的高分已被定位並消除。新的自訂評估目的，是讓後續比較有固定、可重播的規則，而不是藉改 query 拼出論文分數。
## 5. 可以照做的 custom prepare → check → run
以下均為**現有 CLI 支援的命令模板**。資料、清單與不同文字必須先準備；尖括號式的概念佔位不會由工具自動補成正式實驗。不要直接複製文件中的假 UID 產正式結果。
### 5.1 決定輸入檔，固定後再看分數
需要四個檔案：
- `query_uids.json`：JSON array，明確 query UID 順序。
- `gallery_uids.json`：JSON array，包含全部 query，明確候選 UID 順序。
- `query_texts.json`：JSON object，恰好是 `query UID → 不同的真實描述`；不接受額外或缺少 UID。
- `provenance.json`：記資料 root／版本、split 來源、文字的人工或模型來源、prompt／revision，以及是否用於選模。CLI 雖可省略此檔，正式比較應提供。
若從現有 split 匯出，應先明示選哪個集合。custom CLI 沒有 `--query-split`／`--gallery-split`，它只讀已固定的 JSON 清單；也沒有自動生出不同描述的旗標。
### 5.2 設定明確路徑
```bash
cd /home/kyzen/MetaFindV1
PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python
export METAFIND_DATA=/home/kyzen/metafind/metafind_data_paper

# 指向已準備的真實清單／文字／來源紀錄；此目錄不是工具自動建立。
INPUTS=/path/to/frozen_custom_inputs

# 必須是全新的輸出目錄；失敗後重跑也換新名稱。
PROTOCOL_DIR=/path/to/custom_protocol_run01
S1_EVAL_DIR=/path/to/custom_s1_run01
S2_EVAL_DIR=/path/to/custom_s1_s2_run01

# 下面是目前等待鏈預定輸出的路徑；檔案不存在時尚不能執行評估。
S1_REC="$METAFIND_DATA/outputs/checkpoints/paper_v10_same_record_lr1e-4/stage1_best_ckpt.json"
S2_REC="$METAFIND_DATA/outputs/checkpoints/stage2_arms/S2_paper_v10_none_ft5e-5_allhouses_room/variant_ckpts.json"
```
### 5.3 Prepare：CPU 準備觀測並凍結來源
```bash
"$PY" -m metafind.eval.custom_protocol \
  --query-uids "$INPUTS/query_uids.json" \
  --gallery-uids "$INPUTS/gallery_uids.json" \
  --query-texts "$INPUTS/query_texts.json" \
  --provenance "$INPUTS/provenance.json" \
  --out-dir "$PROTOCOL_DIR"
```
它核對 canonical annotation／serializer／embedding sidecar、view count 與檔案 bytes，準備 query PC，寫出 `protocol.json`。不修改 canonical PC、annotation、訓練 checkpoint 或 promoted index。`seed-offset` 的現行預設為 1000003，protocol 會保留實際值。
### 5.4 Stage 1 完成後：先 check-only，再跑 mean＋S1
```bash
"$PY" -m metafind.eval.custom_table1 \
  --protocol "$PROTOCOL_DIR/protocol.json" \
  --stage1-record "$S1_REC" \
  --out-dir "$S1_EVAL_DIR" \
  --check-only

"$PY" -m metafind.eval.custom_table1 \
  --protocol "$PROTOCOL_DIR/protocol.json" \
  --stage1-record "$S1_REC" \
  --out-dir "$S1_EVAL_DIR" \
  --device cpu --batch-size 2 --block 4096 --seed 20260907
```
`--check-only` 不建立大型模型、不產分數，也不建立評估輸出目錄，所以可沿用同一個全新 `--out-dir` 執行下一條命令。它驗的是輸入、record／權重身分與設定相容性；**實際 token 差異和完整 forward 仍要在 run 才驗證**。
CPU 命令可以執行，但全量會比小型診斷耗時得多。GPU 已安排且可用時，可將最後一條改成 `--device cuda --batch-size 16`；工具不會自行從 CPU 切到 GPU，也不應另啟一份工作與正在跑的 annotation／queue 爭用資源。
### 5.5 Stage 2 完成後：同一 protocol 跑三列
```bash
"$PY" -m metafind.eval.custom_table1 \
  --protocol "$PROTOCOL_DIR/protocol.json" \
  --stage1-record "$S1_REC" \
  --stage2-record "$S2_REC" --stage2-variant full \
  --out-dir "$S2_EVAL_DIR" \
  --check-only

"$PY" -m metafind.eval.custom_table1 \
  --protocol "$PROTOCOL_DIR/protocol.json" \
  --stage1-record "$S1_REC" \
  --stage2-record "$S2_REC" --stage2-variant full \
  --out-dir "$S2_EVAL_DIR" \
  --device cpu --batch-size 2 --block 4096 --seed 20260907
```
三列會在這次 run 中一起計算，S2 使用的 S1 parent 不能換成另一個 checkpoint。比較前後兩次執行時，也要確認 protocol／source hashes 沒變；固定 UID 但改了輸入 bytes，仍不是同一個實驗。
**成功產物：** `protocol.json` 副本、`provenance.json`、`results.json`、`table.md`，以及每個模型／觀測／模態的逐 query `.jsonl`。完整三列是 3×2×7＝42 組，總 query records 應為 `42 × n_query`。S1-only 則為 28 組。失敗可能留下 `failure.json` 與中間證據；不得將部分資料拼成完整表。
驗收時核對 `results.json` 的 query／gallery 數、三個 method IDs、兩個觀測版本、七個 condition、有效 token 檢查、逐筆 UID／target column／rank，以及 `validation.stage2_parent_gallery=unchanged`。
完整規格：[CUSTOM_TABLE1_EVALUATION.md](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/CUSTOM_TABLE1_EVALUATION.md)。
## 6. Stage 1 完成後，不只是一個 `.pt` 檔
正確交接順序是：
`最終 admitted corpus → n06 cache → n09 splits／protocols → Stage 1 訓練與選模 → checkpoint＋record → n11 gallery staging → G4 → n12 promotion → S1 retrieval`
G3 preflight 需要讀 n09 的 `splits.json`、Stage 1 training／eval protocols，以及 encoding／hyperparameter records，所以**必須在 n09 產物齊備後另行執行**，不能排在 n09 前面。目前 live 自動鏈尚未插入這個 preflight；上面的箭頭不代表自動等待 G3 PASS。
**已批准的 IMPLEMENTATION CHOICE：** 2026-09-08 使用者採用 [DL-106](https://github.com/Kyzen5128/MetaFindV1/blob/main/workflow/DECISION_LEDGER.md)，將人工排除獨立記為 E，保留原始 manifest M。A 為 admitted；Q 為有真實物件處理失敗證據、扣除成功恢復與人工排除後的 UID。要求 A、Q、E 兩兩不重疊且 `A ∪ Q ∪ E = M`。既有 2% 門檻只計 Q/M，另外揭露 E/M 與 (Q＋E)/M；不替人工排除捏造 exception。
**OBSERVED DATA：** 新版部分 preflight 已在歷史穩定真實 corpus 得到 **PASS：46,052 M＝45,692 A＋339 Q＋21 E**。這不是正在重建的 paper/v10 corpus。該 paper run 缺少 splits、明示 E ledger 與必要 protocols，實際結果仍為 **BLOCKED_EVIDENCE**，不能挪用歷史資料的 PASS。原始結果見 [stable G3 record](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/audit/corpus_20260908_artifacts/g3_final/historical/G3_object_corpus.yaml) 與 [paper-in-progress record](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/audit/corpus_20260908_artifacts/g3_final/paper_in_progress/G3_object_corpus.yaml)。
這是輸入集合計帳／protocol 的**部分實作驗證**；完整 G3 的 `implemented` flag 仍為 `false`，尚未接入 live chain。它不證明 cache tensors、實際 optimizer、完整 Gate 整合或正式論文結果。人工 E 決策已完成；2026-09-08 已部署並驗證新的 Stage 1 等待 shell（PID 2685619），仍停在 R2 wait。annotation PID 2043427 與 Stage 2 等待 PID 2494552 的程序身分保持原樣；這不代表 R2 後的 ledger 發布與 annotation 移動已執行。見 [本輪部署紀錄](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/audit/corpus_20260908_artifacts/stage1_e_ledger_deployment.json)。
Stage 1 完成至少要看：best record 指向的實際 `.pt` 與 SHA、模型／初始化設定、train／selection UID 身分、實際 optimizer steps、loss／selection metrics。record 中的選模集合優先於檔名中的「test」或「holdout」。
目前鏈的 gallery 命令如下。**它們是已排定流程的說明，不是請在等待鏈旁邊再啟一份重建工作。**
```bash
"$PY" -m metafind.train.gallery_index stage1 --stage1-ckpt-record "$S1_REC"
"$PY" -m metafind.gates.g4_gallery_freeze
"$PY" -m metafind.train.gallery_index promote
```
n11 寫 `gallery_index_staging.json` 與 checkpoint-bound NPZ；G4 產 `outputs/logs/gates/G4_gallery_freeze.yaml`；promote 核驗後發布 `outputs/gallery_index.json`。G4 核對身分、集合／數量與 self-match，不能認證作者的 query 定義、獨立測試性或全部模型品質。
新 custom 評估會依自己的固定觀測重新編 gallery，**不直接使用 promoted index 當它的兩套 gallery**；n11／G4／n12 則仍是既有 reported retrieval 與 raw scene exporter 的交接要求。
## 7. 兩條自動等待鏈實際做了什麼
來源：[chain_paper_stage1.sh](https://github.com/Kyzen5128/MetaFindV1/blob/main/tools/chain_paper_stage1.sh)、[chain_paper_stage2.sh](https://github.com/Kyzen5128/MetaFindV1/blob/main/tools/chain_paper_stage2.sh)。下面是腳本可達的工作，不是每一步已完成的宣告。程序是否仍在等待、使用哪個 checkout／corpus，須看部署與狀態紀錄。
### 7.1 Stage 1 chain
1. 等待 `r2_annotate_v10.log` 出現 `=== R2 DONE`。
2. **先發布只包含已批准 manual 21 筆 E 的 ****`outputs/annotation_exclusions.json`****，再搬開對應 annotations，最後重建 annotation index。** 新 ledger 綁 DL-106 與原始決策來源 bytes；歷史 311 筆 n05 失敗不直接整包搬成新 corpus 的失敗。先保存 ledger，才能讓後續 n09 持續排除被恢復或重新標註的 rejected UID。
3. resolve Stage 1 encoding，執行 n06，再產 n09 splits／protocols。
4. 執行 S1：`same_record`、`same_mean`、`--selection-split holdout --lr 1e-4 --epochs 10 --amp off`。
5. 建 n11 gallery、G4、promote，執行既有 `run_retrieval` 的 A20 和 B。
6. 寫 `=== DONE (row 1)`，第二條鏈才接續。
上述新版腳本已完成獨立 review／測試，部署紀錄確認新 shell 使用 SHA `d18f7accff486a68cdc537ca0d35fa01545b2c8af30b0ab54387843fbda36fb6` 並仍在等待。這不等於真實 exclusion 移動已執行：它仍是 R2 完成後才寫入 ledger 與搬移檔案，也仍未自動呼叫 G3 preflight。
腳本的文字標籤會寫「80/20」，但 `splits.py` 實際保留 train 80%、val 10%、test 10%，且 `holdout = val + test`。本條 S1 命令明確在整份 holdout 選模，因此隨後在它上面報分是已選擇的 development/report protocol，**不是未見測試集泛化證據**。
### 7.2 A20 與 B 並非「同一批 query，只改 gallery 大小」
<table header-row="true">
<tr>
<td>現有 protocol ID</td>
<td>Query</td>
<td>Gallery</td>
<td>解讀</td>
</tr>
<tr>
<td>`A20_holdout_vs_holdout`</td>
<td>holdout＝val＋test</td>
<td>holdout</td>
<td>與 live S1 selection pool 重疊</td>
</tr>
<tr>
<td>`B_full_gallery`</td>
<td>test</td>
<td>full＝train＋val＋test</td>
<td>Query 只有 test 子集；本條 live S1 的 selection 已包含它</td>
</tr>
<tr>
<td>`A20_test_vs_holdout`</td>
<td>test</td>
<td>holdout</td>
<td>現有定義，可用於固定 test query 的候選池比較；不是目前 chain step 7 自動呼叫的 A20</td>
</tr>
</table>
Stage 1 chain 實際命令為：
```bash
"$PY" -m metafind.eval.run_retrieval \
  --ckpt-record "$S1_REC" \
  --protocol A20_holdout_vs_holdout --protocol B_full_gallery \
  --out-dir "$METAFIND_DATA/outputs/eval/table1_paper_v10_S1head" \
  --unseal
```
它的輸出是既有 evaluator 的 `table1.json` 與 diagnostics／provenance，不是新 custom 的兩張 same／different tables。來源：[splits.py 的 build_eval_protocols](https://github.com/Kyzen5128/MetaFindV1/blob/main/metafind/data/splits.py)。
### 7.3 Stage 2 chain
1. 等 S1 row 1 DONE，並等 ProcTHOR isolated-asset renders 完成訊息。
2. 用 Gemma 產 ProcTHOR captions，建立 metadata／canonical object text。
3. resolve Stage 2、edge、architecture protocols，graph scope 為 room。
4. n08 產 semantic edge sentences／vectors 與 node embeddings。
5. n09c 產 house train/test split 與 semantic coverage。
6. n11b 按 S1 parent 建 ProcTHOR gallery 與 raw modality vectors。
7. 真正訓練 S2 Full，封存 checkpoint、variant record、log 及 gallery record，更新封存檔案 URI。
8. 以 300 test houses 執行 S1／S2-off／S2-on ProcTHOR probe。
9. 執行歷史 `exp_type_level_query.py` 的 own／weak own／partner 診斷，並製作兩個 head 的表。
**目前兩條鏈都沒有呼叫 G3 preflight、****`metafind.eval.custom_protocol`**** 或 ****`metafind.eval.custom_table1`****。** G3 需在 n09 產物齊備後另行執行；第 5 節的自訂清單／文字準備和評估也需另行執行。所以不能等到 chain 寫 DONE 就說 G3 整合或新評估已完成。
另一個驗收細節：Stage 2 chain 的 ProcTHOR probe 失敗會印出失敗訊息，但繼續跑後面的歷史表格。因此 `DONE (row 2)` 也不保證 ProcTHOR probe 成功；須獨立檢查 probe 的實際輸出。
## 8. Stage 2 資料依賴與訓練契約
### 8.1 不把 Objaverse UID 與 ProcTHOR assetId 混在一起
Objaverse 以 UID 辨識資產；ProcTHOR 以 assetId 辨識可重用物件、以 house／node index 辨識場景 instance。同一 assetId 可在不同 house 出現，house split 分離不代表 assetId split 分離。
<table header-row="true">
<tr>
<td>步驟</td>
<td>讀取／處理</td>
<td>必須交付的產物或檢查</td>
</tr>
<tr>
<td>n07 scene graphs</td>
<td>House layout、位置、room、support／adjacency</td>
<td>`scene_graphs/<house_id>.json`、index、初始 object text</td>
</tr>
<tr>
<td>n07b modalities</td>
<td>已隔離資產的真 renders；可有 depth-shell PC</td>
<td>每 asset sidecar、view PNG 與實際模態清單</td>
</tr>
<tr>
<td>captions／metadata</td>
<td>captions 與 metadata 形成 canonical `v3_fit` 文字</td>
<td>`procthor_asset_annotations.json`、`procthor_object_text.json`</td>
</tr>
<tr>
<td>n09b protocols</td>
<td>固定 room scope、available modalities、positive identity、edge 與 ESSGNN 架構</td>
<td>`stage2_protocol.json`、`stage2_positive_map.json`、`essgnn_edge_protocol.json`、`essgnn_arch_protocol.json`</td>
</tr>
<tr>
<td>n08 semantics</td>
<td>Node 用 `text`；relation 用非空 `relation_text`，否則 fallback `text`</td>
<td>node／edge NPZ、source-binding sidecar、cache 與 sentences</td>
</tr>
<tr>
<td>n09c splits</td>
<td>以 house 分 train／test，檢查 semantic coverage</td>
<td>`scene_splits.json`</td>
</tr>
<tr>
<td>n11b gallery</td>
<td>還原 S1 gallery，編碼已宣告的 ProcTHOR 模態</td>
<td>`stage2_gallery_index.json`、NPZ fused gallery／raw vectors、source identity v2</td>
</tr>
<tr>
<td>n13 Stage 2</td>
<td>查表取 raw query／positive，以移除 target 的 room graph 建 context</td>
<td>checkpoint、`variant_ckpts.json`、實際 training log／source status</td>
</tr>
</table>
當前主線的 ProcTHOR 模態是 **text＋image**。磁碟上有 depth-shell PC，不代表這次宣告使用 PC；也不能把 isolated asset 的 depth shell 稱為完整 mesh surface sample。
新 n11b 的文字來自 canonical `procthor_object_text.json`，並核對 metadata 的 fitted sentence；不再讀可能過期的 renderer sidecar `text`。metadata 改了，n08 node 與 n11b 必須重新產生，舊向量不會自動更新。
Source identity 綁精確 canonical JSON、實際句子、模態 sidecar、使用的 views／順序與宣告 PC bytes。node／edge 也綁實際文字、key／row 與文字 encoder 身分。Producer 發布前與 consumer 讀入時重驗，來源漂移應拒絕；不能替舊 cache 補今天的 hash 就當成當時使用正確來源的證據。
舊 index／semantic artifact 的明示 legacy 入口，只代表接受缺證據的歷史資料並標記 `legacy_unbound`，不是正式來源已驗證。新 custom S2 評估預設不放寬必要的 checkpoint metadata 檢查。
### 8.2 實際 S2 訓練設定
當前 chain 使用 `full`，parent 是上面的 S1 record，recipe 來自 [stage2_hyperparameters_ft_lr5e-5.json](https://github.com/Kyzen5128/MetaFindV1/blob/main/workflow/stage2_hyperparameters_ft_lr5e-5.json)：nominal LR 5e-5、1 epoch、batch 64、固定 tau 0.5、scene dropout 0.3、AdamW／warmup＋cosine。`--query-modality-masking none` 是另行明示的 query 模態策略，不能把它與 scene dropout 混為一談。實際步數與 LR 應讀 training log，不能用名目 LR 當成每一步的 LR。
S2 凍結 backbone／gallery，更新 query fusion、ESSGNN 與 layout 的可學參數。batch 內去除重複 positive assetId；target 從 room context 移除。宣告 Full 卻沒有實際可用 batch 或 optimizer step，應拒絕，不能寫一份零步 checkpoint 冒充完成訓練。
Checkpoint 應包含 parent 身分、stage2／edge／architecture snapshots、graph scope、split／node／edge／gallery inputs 身分、layout dimensions，以及實際 temperature／lambda 設定。消費者按記錄還原，不能用當前 protocol 預設替換歷史架構。
來源：[stage2.py](https://github.com/Kyzen5128/MetaFindV1/blob/main/metafind/train/stage2.py)、[resolve_stage2.py](https://github.com/Kyzen5128/MetaFindV1/blob/main/metafind/models/resolve_stage2.py)、[資料流指南](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/DATA_FLOW.md)。
## 9. S2 完成後的兩條評估支線
### 9.1 Objaverse：S2-off 物件檢索
使用第 5 節的 custom CLI。它不建 ProcTHOR graph，也不把無 layout 的 Objaverse query 人為塞入假場景；只測 child query fusion。結果與 S1 共用同一套 UID、觀測與 parent gallery。
### 9.2 ProcTHOR：layout 的檢索診斷
與現有 chain 相同的入口，可明確選 CPU：
```bash
"$PY" tools/probes/stage2_procthor_retrieval.py \
  --stage1-ckpt-record "$S1_REC" \
  --stage2-record "$S2_REC" --variant full \
  --houses 300 --query-mode none --device cpu \
  --out /path/to/new_procthor_s1_s2_comparison.json
```
`--houses 300` 指 test-house 診斷上限，實際 query 數仍由可用 graphs／samples 決定。`none` 是 query masking mode，不能解讀成「沒有 query 模態」。這支 probe 使用 ProcTHOR asset gallery 與 room context，與 Objaverse 自訂七模態表的資料宇宙不同。
它不是 Table 2 場景美學評分。完整場景另走：明確 I-Design raw scene／sidecar 和 modality mapping → `scene.idesign` → `scene.prepare` → frozen manifest → `scene.compose` → `scene.placement` → Blender。
Layout-on 的 Algorithm 1 每次編碼當前已放置物件的圖，檢索、放入 slot，再更新 graph。`parallel` 對照每次只讀 G0；`use_layout=false` 是同權重關閉 layout residual 的對照。這些是 request／frozen inputs 的欄位，**compose CLI 沒有 ****`--mode`**** 旗標**。
缺 semantic pairs 時，prepare 明確輸出 `required_pairs.json`；需用已批准的 SG2／Gemma 路徑補關係並在新目錄重播，不把未知關係自動當零向量。真 `no_layout` checkpoint 不含 ESSGNN，只能以 `use_layout=false` 使用。
可以照做的完整 schema／命令分別在：[IDesign 輸入](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/IDesign_INPUTS.md)、[raw prepare](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/RAW_SCENE_INPUTS.md)、[composition](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/SCENE_COMPOSITION.md)、[placement／render](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/SCENE_PLACEMENT.md)。
Table 2 已批准改用 Gemma、不做人工評分；Human 欄為 `INSUFFICIENT_EVIDENCE`。目前 `scene_scores` 只驗外部固定 protocol、場景／render bytes、四維 1–5 分及完整分母，**不呼叫裁判、不自動提供正式 200 scenes／prompt／cameras**。尚未拿到分數時不能補零，也不能把一次 render 完成算成 judge 完成。詳見 [SCENE_SCORES.md](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/SCENE_SCORES.md)。
## 10. 已完成的真實小型驗證，支持到哪裡
<table header-row="true">
<tr>
<td>已執行項目</td>
<td>實際範圍／結果</td>
<td>明確限制</td>
</tr>
<tr>
<td>真 ULIP-2＋舊 P1s 的 custom CPU 評估</td>
<td>2 query／6 gallery，兩觀測×兩模型×七欄＝28 組；51.74 秒、exit 0；模型／loader／scorer 未替換</td>
<td>舊 attrs／12-view corpus，小樣本且非獨立測試；舊 S2 缺 embedded metadata，被拒絕而未納入</td>
</tr>
<tr>
<td>真 Stage 1 CLI 與另一程序 strict restore</td>
<td>2 train／2 selection、1 optimizer step；訓練 21.78 秒、還原 25.40 秒</td>
<td>用隔離資料／batch 2／epoch 1，只證明訓練與權重交接可執行，不是正式收斂</td>
</tr>
<tr>
<td>真 Stage 2 Full 訓練與還原</td>
<td>8 個 ProcTHOR 資產、88 PNG、28 edges；真 node／edge 編碼、n11b、1 optimizer step；layout-present 8×1280 query 還原逐位元一致</td>
<td>小型 room，歷史 relation 句子重編、沒有重跑全部正式資料與訓練；真模型診斷不等於科學結果</td>
</tr>
<tr>
<td>新 S1＋S2 單步 checkpoints 的 custom 三列 CPU 評估</td>
<td>2 query／6 gallery，42 組、84 query records；51.32 秒、exit 0；S2 gallery 不變</td>
<td>其中一個 query 曾參與新 S1 selection；mean 滿分不代表正式 baseline 品質</td>
</tr>
<tr>
<td>真模型場景鏈</td>
<td>六筆 promoted gallery、三 slots、真 ULIP／CLIP／Gemma 補關係、iterative／parallel／layout-off、真 GLB／CPU Blender render</td>
<td>使用單步診斷 checkpoints；三個控制條件取回資產相同，不證明 layout 改善品質；不是正式 200 scenes</td>
</tr>
</table>
三列 custom 的獨立 bookkeeping 核對確認：42 組 recall 與輸出的 ranks 一致、84 筆 query records、target columns `[5,4]`，不是假設對角 GT。這份核對沒有重新編碼所有向量或獨立重算 cosine，不能擴大其證據範圍。
可查證的小型紀錄已封存：
- [真 CPU custom 2×6 執行紀錄](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/audit/validation_artifacts_20260907_08/custom_table1_cpu_real_20260907/execution.json)
- [新兩階段 checkpoint 的三列 custom 執行紀錄](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/audit/validation_artifacts_20260907_08/custom_table1_cpu_new_stages_20260908/execution.json)
- [三列 custom 的逐筆核對](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/audit/validation_artifacts_20260907_08/custom_table1_cpu_new_stages_20260908/verification.json)
- [訓練／還原詳細報告](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/history/REPRODUCTION_TRAINING_REVIEW_20260908.md)
- [真場景詳細報告](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/history/REPRODUCTION_SCENE_REVIEW_20260908.md)
上述時間與數字是固定歷史 run 的紀錄，不是對現在程序即時進度的猜測。本輪最後 CPU suite 為 **1,805 passed**，graph checker 為 **2,342 checks 通過**；測試中 265 個程式來源 SHA 前後一致。這支持各測試涵蓋的 assertion，不能取代正式模型評估；見 [交付審查紀錄](https://github.com/Kyzen5128/MetaFindV1/blob/main/docs/audit/REPRODUCTION_CORPUS_REVIEW_20260908.md)。
## 11. 下一步與驗收清單
<table header-row="true">
<tr>
<td>順序</td>
<td>工作</td>
<td>驗收輸出／不能省略的限制</td>
</tr>
<tr>
<td>1</td>
<td>核對新等待腳本部署；等 R2 完成後先保存 manual 21 筆 E ledger，再移動排除 annotations／重建 index</td>
<td>綁原始決策的 DL-106 ledger 與實際部署／執行紀錄；不將歷史 311 筆失敗直接搬入新版 ledger</td>
</tr>
<tr>
<td>2</td>
<td>n05b → n06 → n09</td>
<td>Encoding／hyperparameter／training／eval protocols、cache 與 splits 齊備；此時才具備 G3 preflight 的輸入</td>
</tr>
<tr>
<td>3</td>
<td>在新 corpus 上另行執行 G3 部分 preflight</td>
<td>新資料的 M＝A＋Q＋E、兩兩不重疊、protocol 與真失敗證據；不挪用 stable PASS，不稱完整 G3 implemented；目前 live chain 不會自動跑或等待這一步</td>
</tr>
<tr>
<td>4</td>
<td>S1 正式訓練與選模</td>
<td>真 optimizer steps；best checkpoint＋record＋selection UID 身分；另保留實際 G3 檢查時點，不能回填成已自動攔截</td>
</tr>
<tr>
<td>5</td>
<td>n11／G4／promotion、既有 S1 retrieval</td>
<td>staging／G4／registry／`table1.json`；分別記 A20 與 B 的 query／gallery 分母與 selection overlap</td>
</tr>
<tr>
<td>6</td>
<td>另行固定 custom query／gallery／不同文字，prepare、check、S1-only run</td>
<td>固定 protocol，28 組 mean／S1 結果；有效 token 差異與來源核對；不按分數刪樣本</td>
</tr>
<tr>
<td>7</td>
<td>ProcTHOR captions／metadata → protocols → n08 → scene_splits → n11b</td>
<td>canonical text 一致、room scope、text＋image、semantic source binding、v2 gallery identity</td>
</tr>
<tr>
<td>8</td>
<td>S2 正式訓練與還原</td>
<td>非零 optimizer steps、query／ESSGNN 更新、gallery frozen、child parent／input identity、封存可還原</td>
</tr>
<tr>
<td>9</td>
<td>另行使用同一 custom protocol 跑 mean／S1／S2-off</td>
<td>42 組與 `42×n_query` records，S2 gallery unchanged；不要改觀測或候選去貼近論文</td>
</tr>
<tr>
<td>10</td>
<td>ProcTHOR S1／S2-off／S2-on 診斷</td>
<td>Probe 實際成功 artifact；不能只看 chain DONE</td>
</tr>
<tr>
<td>11</td>
<td>固定正式場景／評審輸入後執行完整場景評估</td>
<td>完整 scene×method 分母、失敗原因、composition／render／judge 身分；缺分數保持不足證據，Human 不偽造</td>
</tr>
</table>
這張表是資料依賴與驗收順序，不是宣稱目前 shell 已把所有步驟串成自動 gate。以上 CLI 片段供操作時閱讀及核對，本次文件更新沒有執行這些命令模板。
唯讀查看目前 checkout／corpus 與程序摘要可用：
```bash
bash tools/status.sh --data /home/kyzen/metafind/metafind_data_paper
```
Status 是程序與 inventory 的觀測入口，不是訓練／評估完成認證。正式交付應同時保留：可重播命令、資料／protocol／checkpoint hashes、完整分母、原始 logs、逐 query 或逐 scene 證據，以及仍屬 UNKNOWN／DEVIATION 的項目。
