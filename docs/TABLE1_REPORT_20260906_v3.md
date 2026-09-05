# Table 1 報告 v3（2026-09-06）—— 兩列都做：Stage 1 頭（w/o ESSGNN）與 Stage 2 共用頭（w/ ESSGNN）

接續 `docs/TABLE1_REPORT_20260905_v2.md`。本檔是 Kyzen 2026-09-06 04:3x 三道命令之後的成果：
「完成 table 1 之前不准停下」「可以先完成 stage 2 再回來測」「我全程不參與，你自己想辦法」。
所以這裡每一個選擇都是我下的，全部標成 IMPLEMENTATION CHOICE / DEVIATION，不會寫成論文規定。

## 1. 論文的兩列各來自哪個模型（PAPER FACT，`3experiments.tex` §3.2）

> "A practical mitigation is to maintain two fusion heads ... **Using the Stage-1 head reproduces the 'w/o ESSGNN' numbers** (omitted for brevity). In our reported results, we instead explore **a single shared head by freezing both encoders in Stage-2, updating only ESSGNN and the fusion**, and applying stochastic scene dropout (30%)."

- **w/o ESSGNN 列 = Stage 1 的 query 頭。** Stage 2 不是這一列的來源。
- **w/ ESSGNN 列 = Stage 2 之後的共用頭。** Stage 2 在 ProcTHOR 上微調 query 融合層＋ESSGNN，gallery 凍結；所以這一列比第一列低（作者稱 feature-attribution mismatch）。
- Objaverse 的資產沒有場景，所以第二列評分時 layout 那一項不存在（INFERENCE：論文沒寫單一資產進 ESSGNN 收到什麼；30% scene dropout 訓的就是這個情況）。

回答 Kyzen 的問題「是不是要完成 Stage 2 才會符合 Table 1」：**第一列不需要，第二列需要。** 兩列都做才是完整的 Table 1。

論文兩列（R@1 / R@5）：

| 列 | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| MetaFind w/o ESSGNN | 13.8 / 23.1 | 11.7 / 19.2 | 75.1 / 78.0 | 17.2 / 21.8 | 44.5 / 71.3 | 45.8 / 73.1 | 51.7 / 76.5 |
| MetaFind w/ ESSGNN | 11.3 / 21.5 | 10.5 / 15.9 | 63.2 / 66.5 | 15.9 / 20.3 | 41.2 / 68.8 | 42.0 / 70.4 | 48.2 / 74.9 |

## 2. 我們的協定（全部有紀錄）

| 項 | 我們 | 論文 | 分類 |
|---|---|---|---|
| 語料 | Objaverse-LVIS 45,692（46,024 − 311 − 21） | 48K | OBSERVED DATA；DEVIATION（數量） |
| 切分 | 80 / 10 / 10（train 36,554 / val 4,569 / test 4,569；D-3b）；選模與報告都在 20%（9,138，Kyzen 9/4 的圖） | 80 / 20 | IMPLEMENTATION CHOICE（論文只寫 80/20） |
| gallery | 20% holdout 全部 9,138 件，gallery 塔三模態齊全（attrs 句、12 視角平均、canonical 點雲） | "pre-encoded asset database"，大小未寫 | IMPLEMENTATION CHOICE |
| query 構造 | 三種，見 §3 | **沒寫** | IMPLEMENTATION CHOICE |
| backbone | (a) 釋出 ULIP-2 權重（P1s，主線）；(b) 在我們 80% 上用 ULIP 官方 `main.py` 從頭訓的 Point-BERT（s1_scratchbb） | ULIP-2；有沒有重訓沒寫 | (a) UPSTREAM 權重；(b) 論文字面 |
| Stage 1 | Point-BERT＋兩座融合塔可訓，CLIP 凍結；Eq.5 單向 InfoNCE τ 0.5；30% 獨立遮罩＋mask token；AdamW lr 1e-4、10 epoch、batch 64 | 見 `STAGE1_ARCHITECTURE_EQUATIONS_20260905.md` | 逐項標註在該檔 |
| Stage 2 | ESSGNN＋query 融合層訓，gallery 凍結；scene dropout 0.3；雙向損失；lr 5e-5、1 epoch；P1s 用全部 9,600 間訓練屋（我們對 ProcTHOR-10K 12,000 間的 80% 切分；論文寫 >10,000 屋）、scratchbb 用 1,500 屋 | §2.6 | lr／epoch／屋數：IMPLEMENTATION CHOICE |
| 指標 | R@1、R@5，正解＝同一資產 | R@1、R@5 | 一致 |

## 3. 三種 query 構造（論文沒寫，所以三種都報）

| 名稱 | text | image | pc | 這是什麼 |
|---|---|---|---|---|
| **own** | 資產自己的 attrs 句（跟 gallery 同一份） | 自己 12 張裡的一張 | 自己那朵（跟 gallery 同一朵） | 之前所有正式列的做法；Ex-MCR 的單模態做法也是這樣 |
| **weak own** | 類別＋尺寸（Figure 1 的 `Platform Bed {size:...}`）或 ULIP-2 附的 BLIP 短句 | 自己的 Sketchfab 縮圖（另一套渲染） | 自己那朵，或**同一個網格再取樣一次**（cos 0.996） | 「同一件、但每個模態都是較弱的另一份觀測」——GPT 建議、之前沒做完的那格 |
| **partner** | 同類別另一件的 attrs 句 | 那一件的一張圖 | 自己那朵 | 唯一能做出論文形狀（pc 最高、加模態變低）的構造；但 text／image 單獨掉到 1～5 |

## 4. 結果

### 4.1 val → val 診斷（4,569 → 4,569；checkpoint 不動；R@1 %）

同一件、但每個模態都換成較弱的另一份觀測（DL-101）。「second sample」＝同一個網格再取樣一次（P1s cos 0.996、scratchbb 0.993）。

### exp_type_level_query_P1s_val  (dev_val 4,569 -> ? 4,569; ckpt pilotP1s_split801010_lr1e-4_20260904; Stage 1 head)

**R@1 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| paper w/ ESSGNN | 11.3 | 10.5 | 63.2 | 15.9 | 41.2 | 42.0 | 48.2 |
| own observations (attrs text, own view, own cloud) | 33.8 | 65.7 | 97.9 | 85.9 | 99.7 | 98.5 | 99.8 |
| own text + own view, cloud = second surface sample | 33.8 | 65.7 | 97.8 | 85.9 | 99.5 | 98.4 | 99.6 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 17.2 | 57.0 | 97.9 | 63.0 | 98.2 | 98.3 | 98.4 |
| weak own trio: category+size, thumbnail, second sample | 17.2 | 57.0 | 97.8 | 63.0 | 97.9 | 98.1 | 98.2 |
| weak own: BLIP caption, own thumbnail, own cloud | 13.1 | 57.0 | 97.9 | 47.6 | 95.1 | 98.3 | 95.2 |
| weak own trio: BLIP caption, thumbnail, second sample | 13.1 | 57.0 | 97.8 | 47.6 | 94.7 | 98.1 | 94.7 |
| partner: same-category other asset's text + view, own cloud | 5.5 | 1.2 | 97.9 | 1.2 | 94.4 | 96.1 | 86.1 |
| partner text + view, cloud = second sample | 5.5 | 1.2 | 97.8 | 1.2 | 94.2 | 95.8 | 85.7 |

### exp_type_level_query_scratchbb_val  (dev_val 4,569 -> dev_val 4,569; ckpt s1_scratchbb_sel20_lr1e-4_20260905; Stage 1 head)

**R@1 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| paper w/ ESSGNN | 11.3 | 10.5 | 63.2 | 15.9 | 41.2 | 42.0 | 48.2 |
| own observations (attrs text, own view, own cloud) | 32.6 | 60.9 | 88.4 | 86.4 | 99.7 | 97.4 | 99.9 |
| own text + own view, cloud = second surface sample | 32.6 | 60.9 | 87.4 | 86.4 | 99.5 | 97.1 | 99.6 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 14.0 | 42.4 | 88.4 | 62.1 | 97.6 | 96.7 | 98.2 |
| weak own trio: category+size, thumbnail, second sample | 14.0 | 42.4 | 87.4 | 62.1 | 97.1 | 96.1 | 97.9 |
| weak own: BLIP caption, own thumbnail, own cloud | 6.8 | 42.4 | 88.4 | 35.1 | 87.7 | 96.7 | 91.8 |
| weak own trio: BLIP caption, thumbnail, second sample | 6.8 | 42.4 | 87.4 | 35.1 | 86.7 | 96.1 | 91.3 |
| partner: same-category other asset's text + view, own cloud | 4.5 | 0.8 | 88.4 | 0.8 | 85.0 | 78.9 | 73.3 |
| partner text + view, cloud = second sample | 4.5 | 0.8 | 87.4 | 0.8 | 84.6 | 78.0 | 72.7 |

### exp_table1_stage2head_scratchbb_val  (dev_val 4,569 -> ? 4,569; ckpt s1_scratchbb_sel20_lr1e-4_20260905; Stage 2 head S2_scratchbb_none_ft5e-5_20260905)

**R@1 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| paper w/ ESSGNN | 11.3 | 10.5 | 63.2 | 15.9 | 41.2 | 42.0 | 48.2 |
| own observations (attrs text, own view, own cloud) | 21.5 | 43.3 | 83.5 | 63.2 | 94.7 | 90.7 | 94.9 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 10.7 | 30.6 | 83.5 | 42.7 | 89.4 | 87.6 | 89.9 |
| weak own: BLIP caption, own thumbnail, own cloud | 6.5 | 30.6 | 83.5 | 24.9 | 79.1 | 87.6 | 81.5 |
| partner: same-category other asset's text + view, own cloud | 3.0 | 1.1 | 83.5 | 1.4 | 76.5 | 62.3 | 57.4 |

把自己的雲弄壞（去顏色、半掃描）會把 pc 格壓到 8～35，但合併格反而升到 pc 之上（P1s 去色：pc 35.1、full 55.7；scratchbb 半掃：pc 21.4、full 80.9）——跟論文方向相反。


**不訓融合、只平均（rung 0，釋出 ULIP-2，val→val）**：gallery＝三個單位向量平均；query＝在場模態的單位向量平均。弱觀測三重奏（類別＋尺寸、縮圖、重取樣雲）：49.8 / 86.0 / 99.5 / 89.1 / 99.6 / 99.5 / 99.6；換件：5.4 / 0.2 / 99.5 / 0.1 / 98.2 / 97.7 / 67.3（raw 平均：4.8 / 0.2 / 99.5 / 0.1 / 86.8 / 52.0 / 20.3，就是基線的形狀）。不訓也一樣：同一件的觀測，合併 ≥ pc。

### 4.2 20% holdout（9,138 → 9,138）最終列 —— 我們的 Table 1

主線 P1s（釋出 ULIP-2 backbone）。格式跟論文一樣 **R@1 / R@5**。gallery＝holdout 全部 9,138 件；query＝同一批 9,138 件。
每一列都寫清楚 query 是什麼；**沒有一列宣稱是論文的復現**（query 來源論文沒寫）。

| 列 | query 構造 | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|---|
| 論文 w/o ESSGNN | ? | 13.8 / 23.1 | 11.7 / 19.2 | 75.1 / 78.0 | 17.2 / 21.8 | 44.5 / 71.3 | 45.8 / 73.1 | 51.7 / 76.5 |
| **我們 w/o ESSGNN**（Stage 1 頭） | own | 25.0 / 54.9 | 57.3 / 80.8 | 96.7 / 99.6 | 80.2 / 95.2 | 99.5 / 100.0 | 97.5 / 99.8 | 99.6 / 100.0 |
| 　同上 | weak own | 12.1 / 32.9 | 47.1 / 76.5 | 96.7 / 99.6 | 53.7 / 81.9 | 97.1 / 99.9 | 97.1 / 99.8 | 97.1 / 99.9 |
| 　同上 | partner | 3.6 / 16.3 | 0.8 / 9.2 | 96.7 / 99.6 | 1.0 / 13.7 | 92.3 / 99.1 | 94.3 / 99.4 | 81.7 / 96.0 |
| 論文 w/ ESSGNN | ? | 11.3 / 21.5 | 10.5 / 15.9 | 63.2 / 66.5 | 15.9 / 20.3 | 41.2 / 68.8 | 42.0 / 70.4 | 48.2 / 74.9 |
| **我們 w/ ESSGNN**（Stage 2 共用頭，layout 不在） | own | 16.4 / 41.5 | 31.5 / 59.5 | 66.8 / 90.4 | 46.0 / 76.0 | 69.4 / 91.4 | 69.6 / 91.7 | 71.0 / 92.3 |
| 　同上 | weak own | 7.0 / 22.3 | 24.5 / 50.2 | 66.8 / 90.4 | 25.4 / 52.8 | 59.5 / 85.6 | 63.4 / 88.3 | 59.6 / 85.7 |
| 　同上 | partner | 3.3 / 13.5 | 1.1 / 7.2 | 66.8 / 90.4 | 1.6 / 11.4 | 58.4 / 85.4 | 51.9 / 81.4 | 38.6 / 70.3 |

weak own ＝ 類別＋尺寸文字、自己的 Sketchfab 縮圖、自己那朵雲。partner ＝ 同類別另一件的文字與圖、自己那朵雲。
（P1s 的選模池是 val 4,569，它是這 9,138 的一半；scratchbb 的選模池就是這 9,138。兩者都照 Kyzen 9/4 的圖「選模與報告都在 20%」報。）

第二顆 backbone（s1_scratchbb：Point-BERT 在我們 80% 上從頭訓；Stage 2 只用 1,500 屋）：

| 列 | query 構造 | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|---|
| w/o ESSGNN（Stage 1 頭） | own | 24.0 / 51.8 | 52.3 / 78.3 | 84.4 / 95.9 | 81.4 / 95.8 | 99.5 / 100.0 | 96.0 / 99.4 | 99.7 / 100.0 |
| 　同上 | weak own | 9.1 / 27.2 | 34.2 / 61.3 | 84.4 / 95.9 | 52.4 / 80.8 | 95.9 / 99.8 | 94.8 / 99.5 | 97.2 / 99.9 |
| 　同上 | partner | 2.5 / 11.1 | 0.8 / 6.1 | 84.4 / 95.9 | 0.7 / 9.1 | 81.9 / 94.6 | 75.2 / 92.5 | 69.1 / 87.7 |
| w/ ESSGNN（Stage 2 頭，1,500 屋） | own | 14.8 / 37.1 | 33.8 / 63.2 | 76.9 / 93.7 | 52.8 / 82.4 | 91.5 / 99.1 | 85.9 / 97.4 | 91.5 / 99.0 |
| 　同上 | weak own | 6.7 / 20.9 | 22.1 / 47.1 | 76.9 / 93.7 | 32.9 / 62.1 | 84.0 / 97.7 | 82.3 / 96.5 | 84.8 / 97.5 |
| 　同上 | partner | 2.0 / 9.2 | 0.9 / 5.4 | 76.9 / 93.7 | 1.0 / 7.5 | 71.4 / 91.0 | 56.0 / 82.0 | 50.3 / 76.9 |

#### 4.2.1 四份完整輸出（R@1 與 R@5，含「重取樣雲」與 BLIP 文字列）

### table1_final_P1s_S1head_holdout  (holdout 9,138 -> holdout 9,138; ckpt pilotP1s_split801010_lr1e-4_20260904; Stage 1 head)

**R@1 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| paper w/ ESSGNN | 11.3 | 10.5 | 63.2 | 15.9 | 41.2 | 42.0 | 48.2 |
| own observations (attrs text, own view, own cloud) | 25.0 | 57.3 | 96.7 | 80.2 | 99.5 | 97.5 | 99.6 |
| own text + own view, cloud = second surface sample | 25.0 | 57.3 | 96.2 | 80.2 | 99.1 | 97.1 | 99.3 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 12.1 | 47.1 | 96.7 | 53.7 | 97.1 | 97.1 | 97.1 |
| weak own trio: category+size, thumbnail, second sample | 12.1 | 47.1 | 96.2 | 53.7 | 96.5 | 96.6 | 96.7 |
| weak own: BLIP caption, own thumbnail, own cloud | 8.0 | 47.1 | 96.7 | 37.8 | 91.8 | 97.1 | 91.9 |
| weak own trio: BLIP caption, thumbnail, second sample | 8.0 | 47.1 | 96.2 | 37.8 | 90.9 | 96.6 | 91.2 |
| partner: same-category other asset's text + view, own cloud | 3.6 | 0.8 | 96.7 | 1.0 | 92.3 | 94.3 | 81.7 |
| partner text + view, cloud = second sample | 3.6 | 0.8 | 96.2 | 1.0 | 91.6 | 93.6 | 80.8 |

**R@5 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 23.1 | 19.2 | 78.0 | 21.8 | 71.3 | 73.1 | 76.5 |
| paper w/ ESSGNN | 21.5 | 15.9 | 66.5 | 20.3 | 68.8 | 70.4 | 74.9 |
| own observations (attrs text, own view, own cloud) | 54.9 | 80.8 | 99.6 | 95.2 | 100.0 | 99.8 | 100.0 |
| own text + own view, cloud = second surface sample | 54.9 | 80.8 | 99.6 | 95.2 | 100.0 | 99.8 | 100.0 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 32.9 | 76.5 | 99.6 | 81.9 | 99.9 | 99.8 | 99.9 |
| weak own trio: category+size, thumbnail, second sample | 32.9 | 76.5 | 99.6 | 81.9 | 99.9 | 99.7 | 99.9 |
| weak own: BLIP caption, own thumbnail, own cloud | 24.9 | 76.5 | 99.6 | 67.4 | 99.2 | 99.8 | 99.3 |
| weak own trio: BLIP caption, thumbnail, second sample | 24.9 | 76.5 | 99.6 | 67.4 | 99.1 | 99.7 | 99.2 |
| partner: same-category other asset's text + view, own cloud | 16.3 | 9.2 | 99.6 | 13.7 | 99.1 | 99.4 | 96.0 |
| partner text + view, cloud = second sample | 16.3 | 9.2 | 99.6 | 13.7 | 99.0 | 99.4 | 95.7 |

### table1_final_P1s_S2head_holdout  (holdout 9,138 -> holdout 9,138; ckpt pilotP1s_split801010_lr1e-4_20260904; Stage 2 head S2_P1s_none_ft5e-5_allhouses_20260906)

**R@1 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| paper w/ ESSGNN | 11.3 | 10.5 | 63.2 | 15.9 | 41.2 | 42.0 | 48.2 |
| own observations (attrs text, own view, own cloud) | 16.4 | 31.5 | 66.8 | 46.0 | 69.4 | 69.6 | 71.0 |
| own text + own view, cloud = second surface sample | 16.4 | 31.5 | 65.8 | 46.0 | 69.0 | 68.7 | 70.4 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 7.0 | 24.5 | 66.8 | 25.4 | 59.5 | 63.4 | 59.6 |
| weak own trio: category+size, thumbnail, second sample | 7.0 | 24.5 | 65.8 | 25.4 | 58.9 | 63.0 | 59.3 |
| weak own: BLIP caption, own thumbnail, own cloud | 6.1 | 24.5 | 66.8 | 19.9 | 56.8 | 63.4 | 56.4 |
| weak own trio: BLIP caption, thumbnail, second sample | 6.1 | 24.5 | 65.8 | 19.9 | 56.4 | 63.0 | 56.1 |
| partner: same-category other asset's text + view, own cloud | 3.3 | 1.1 | 66.8 | 1.6 | 58.4 | 51.9 | 38.6 |
| partner text + view, cloud = second sample | 3.3 | 1.1 | 65.8 | 1.6 | 57.7 | 51.7 | 38.3 |

**R@5 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 23.1 | 19.2 | 78.0 | 21.8 | 71.3 | 73.1 | 76.5 |
| paper w/ ESSGNN | 21.5 | 15.9 | 66.5 | 20.3 | 68.8 | 70.4 | 74.9 |
| own observations (attrs text, own view, own cloud) | 41.5 | 59.5 | 90.4 | 76.0 | 91.4 | 91.7 | 92.3 |
| own text + own view, cloud = second surface sample | 41.5 | 59.5 | 90.1 | 76.0 | 91.2 | 91.4 | 92.2 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 22.3 | 50.2 | 90.4 | 52.8 | 85.6 | 88.3 | 85.7 |
| weak own trio: category+size, thumbnail, second sample | 22.3 | 50.2 | 90.1 | 52.8 | 85.2 | 88.2 | 85.2 |
| weak own: BLIP caption, own thumbnail, own cloud | 19.2 | 50.2 | 90.4 | 45.6 | 85.0 | 88.3 | 84.1 |
| weak own trio: BLIP caption, thumbnail, second sample | 19.2 | 50.2 | 90.1 | 45.6 | 84.6 | 88.2 | 83.8 |
| partner: same-category other asset's text + view, own cloud | 13.5 | 7.2 | 90.4 | 11.4 | 85.4 | 81.4 | 70.3 |
| partner text + view, cloud = second sample | 13.5 | 7.2 | 90.1 | 11.4 | 85.3 | 81.0 | 69.9 |

### table1_final_scratchbb_S1head_holdout  (holdout 9,138 -> holdout 9,138; ckpt s1_scratchbb_sel20_lr1e-4_20260905; Stage 1 head)

**R@1 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| paper w/ ESSGNN | 11.3 | 10.5 | 63.2 | 15.9 | 41.2 | 42.0 | 48.2 |
| own observations (attrs text, own view, own cloud) | 24.0 | 52.3 | 84.4 | 81.4 | 99.5 | 96.0 | 99.7 |
| own text + own view, cloud = second surface sample | 24.0 | 52.3 | 83.3 | 81.4 | 99.1 | 95.6 | 99.4 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 9.1 | 34.2 | 84.4 | 52.4 | 95.9 | 94.8 | 97.2 |
| weak own trio: category+size, thumbnail, second sample | 9.1 | 34.2 | 83.3 | 52.4 | 95.2 | 94.2 | 96.8 |
| weak own: BLIP caption, own thumbnail, own cloud | 4.1 | 34.2 | 84.4 | 26.9 | 82.1 | 94.8 | 88.1 |
| weak own trio: BLIP caption, thumbnail, second sample | 4.1 | 34.2 | 83.3 | 26.9 | 80.9 | 94.2 | 87.3 |
| partner: same-category other asset's text + view, own cloud | 2.5 | 0.8 | 84.4 | 0.7 | 81.9 | 75.2 | 69.1 |
| partner text + view, cloud = second sample | 2.5 | 0.8 | 83.3 | 0.7 | 81.3 | 74.3 | 68.4 |

**R@5 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 23.1 | 19.2 | 78.0 | 21.8 | 71.3 | 73.1 | 76.5 |
| paper w/ ESSGNN | 21.5 | 15.9 | 66.5 | 20.3 | 68.8 | 70.4 | 74.9 |
| own observations (attrs text, own view, own cloud) | 51.8 | 78.3 | 95.9 | 95.8 | 100.0 | 99.4 | 100.0 |
| own text + own view, cloud = second surface sample | 51.8 | 78.3 | 95.6 | 95.8 | 100.0 | 99.4 | 100.0 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 27.2 | 61.3 | 95.9 | 80.8 | 99.8 | 99.5 | 99.9 |
| weak own trio: category+size, thumbnail, second sample | 27.2 | 61.3 | 95.6 | 80.8 | 99.7 | 99.4 | 99.9 |
| weak own: BLIP caption, own thumbnail, own cloud | 14.9 | 61.3 | 95.9 | 54.6 | 96.6 | 99.5 | 98.0 |
| weak own trio: BLIP caption, thumbnail, second sample | 14.9 | 61.3 | 95.6 | 54.6 | 96.3 | 99.4 | 98.0 |
| partner: same-category other asset's text + view, own cloud | 11.1 | 6.1 | 95.9 | 9.1 | 94.6 | 92.5 | 87.7 |
| partner text + view, cloud = second sample | 11.1 | 6.1 | 95.6 | 9.1 | 94.5 | 92.1 | 87.3 |

### table1_final_scratchbb_S2head_holdout  (holdout 9,138 -> holdout 9,138; ckpt s1_scratchbb_sel20_lr1e-4_20260905; Stage 2 head S2_scratchbb_none_ft5e-5_20260905)

**R@1 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |
| paper w/ ESSGNN | 11.3 | 10.5 | 63.2 | 15.9 | 41.2 | 42.0 | 48.2 |
| own observations (attrs text, own view, own cloud) | 14.8 | 33.8 | 76.9 | 52.8 | 91.5 | 85.9 | 91.5 |
| own text + own view, cloud = second surface sample | 14.8 | 33.8 | 75.9 | 52.8 | 91.2 | 85.1 | 91.0 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 6.7 | 22.1 | 76.9 | 32.9 | 84.0 | 82.3 | 84.8 |
| weak own trio: category+size, thumbnail, second sample | 6.7 | 22.1 | 75.9 | 32.9 | 83.4 | 81.2 | 84.1 |
| weak own: BLIP caption, own thumbnail, own cloud | 3.7 | 22.1 | 76.9 | 17.9 | 71.0 | 82.3 | 74.2 |
| weak own trio: BLIP caption, thumbnail, second sample | 3.7 | 22.1 | 75.9 | 17.9 | 70.0 | 81.2 | 73.5 |
| partner: same-category other asset's text + view, own cloud | 2.0 | 0.9 | 76.9 | 1.0 | 71.4 | 56.0 | 50.3 |
| partner text + view, cloud = second sample | 2.0 | 0.9 | 75.9 | 1.0 | 70.5 | 54.8 | 49.6 |

**R@5 (%)**

| query construction | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|
| paper w/o ESSGNN | 23.1 | 19.2 | 78.0 | 21.8 | 71.3 | 73.1 | 76.5 |
| paper w/ ESSGNN | 21.5 | 15.9 | 66.5 | 20.3 | 68.8 | 70.4 | 74.9 |
| own observations (attrs text, own view, own cloud) | 37.1 | 63.2 | 93.7 | 82.4 | 99.1 | 97.4 | 99.0 |
| own text + own view, cloud = second surface sample | 37.1 | 63.2 | 93.4 | 82.4 | 99.1 | 97.2 | 99.0 |
| weak own: category+size text, own Sketchfab thumbnail, own cloud | 20.9 | 47.1 | 93.7 | 62.1 | 97.7 | 96.5 | 97.5 |
| weak own trio: category+size, thumbnail, second sample | 20.9 | 47.1 | 93.4 | 62.1 | 97.7 | 96.4 | 97.3 |
| weak own: BLIP caption, own thumbnail, own cloud | 13.1 | 47.1 | 93.7 | 41.8 | 92.4 | 96.5 | 93.3 |
| weak own trio: BLIP caption, thumbnail, second sample | 13.1 | 47.1 | 93.4 | 41.8 | 91.9 | 96.4 | 93.2 |
| partner: same-category other asset's text + view, own cloud | 9.2 | 5.4 | 93.7 | 7.5 | 91.0 | 82.0 | 76.9 |
| partner text + view, cloud = second sample | 9.2 | 5.4 | 93.4 | 7.5 | 90.7 | 81.6 | 76.5 |


## 5. 判讀

1. **第一列（w/o ESSGNN，Stage 1 頭）對不上，而且 val 上已經把「同一件的觀測」全部試完。** 自己那句／類別＋尺寸／BLIP 短句／Sketchfab 名字；自己的一張圖／自己的縮圖；自己那朵雲／再取樣的雲——不論融合是訓過 10 epoch 的 Transformer 還是完全不訓的平均，Stage 1 頭的合併格都 ≥ pc（pc 96.7）。弱文字最多讓 T+PC 掉 3～5 點（論文掉 30）。論文的排序只在 query 文字與圖來自另一件時出現，而那時 text／image 單獨只剩 1～4（論文 13.8／11.7）。這一列的差距是 query 觀測來源（論文沒寫）加上 pc 對應強度（論文 75、我們 97）；兩者都不是再多跑實驗能補的。
2. **第二列（w/ ESSGNN，Stage 2 共用頭）是今天的新結果。** Stage 2 在 9,600 間屋上微調之後，query 頭被推離 gallery 頭：pc 96.7 → 66.8（論文第二列 63.2）。pc 的對應一鬆，弱文字就真的拖累：weak own 列 pc 66.8 > I+PC 63.4 > full 59.6 ≈ T+PC 59.5 > T+I 25.4 ≈ image 24.5 > text 7.0——**第一次用同一件資產自己的觀測做出論文的排序**。用自己的文字和圖時，合併格在 pc 之上 2～4 點（69～71）。還沒對上的：image 31.5（論文 10.5）、T+I 46.0（15.9）、合併格比論文高 10～30 點；R@5 的 pc 是 90.4（論文 66.5，論文的 pc 從 R@1 到 R@5 幾乎不漲，我們漲 24 點——論文那 1/3 找不到的 query 是「完全找不到」，我們是「差一點」）。
3. **漂移量跟 Stage 2 的長度有關。** scratchbb 的 Stage 2 只有 1,500 屋（約 1,600 步），pc 只掉到 76.9、合併格仍 ≥ pc；P1s 的 9,600 屋（約 10,200 步）掉到 66.8、合併格 ≤ pc。這跟論文自己的話一致：「the fusion layer becomes partially adapted to layout-conditioned features ... residual attribution drift」。但論文的**第一列**（還沒 Stage 2）就已經是 pc 75、合併格在 pc 下面；他們的 Stage 1 頭為什麼沒把 pc 對到 97，論文沒寫，我們沒有材料能重現（DL-099 的融合梯已排除「訓太少」）。
4. **Kyzen 的問題「是不是要做完 Stage 2 才符合 Table 1」**：論文的第一列不是 Stage 2 的產物（§1），但**論文的形狀**在我們這裡確實要到 Stage 2 之後才出現；他的直覺對了一半，而且是有用的那一半。
5. **怎麼讀我們的 Table 1**：own 是正式定義（跟領域裡單模態檢索一致：Ex-MCR 用自己的一張視角圖）；weak own 是唯一同時符合「同一件」與「論文排序」的構造，但只在 Stage 2 頭上成立；partner 排序對、單模態格對不上。狀態：協定 IMPLEMENTED／EXECUTABLE／BEHAVIOR-VERIFIED（parity 列跟正式評估器完全一致）；數字對論文 **UNVERIFIED**，query 構造 **IMPLEMENTATION CHOICE**，Stage 2 屋數與 lr **IMPLEMENTATION CHOICE**，語料 45,692 對 48K **DEVIATION**。不用 PAPER-ALIGNED。
6. **只剩作者能回答的**：(a) Table 1 的 query 文字與影像各是哪一份觀測；(b) gallery 是 20% 還是 48K；(c) Stage 1 頭為什麼 pc 只有 75（CLIP 有沒有微調、訓多久、gallery 塔是不是凍在 ULIP-2）；(d) 第二列的 ESSGNN 對單一資產收到什麼。

## 6. 可追溯

- 程式：`tools/probes/exp_type_level_query.py`（`--pc-policies`、`--stage2-state`、`--query-split`；parity 列＝正式評估器的數字）、`tools/probes/exp_mean_pool_weak_trio.py`、`tools/probes/tabulate_table1_final.py`。
- 鏈子（`metafind_data_attrs/outputs/logs/`）：`chain_thumbnail_query_val.sh`（val 診斷）、`run_mean_pool_weak_trio_20260906.sh`、`chain_stage2_mainline_20260906.sh`（scratchbb 頭評分 → 紀錄保全 → P1s n11b 索引 → P1s Stage 2 → ProcTHOR 探針 → val 評分）、`chain_table1_final_20260906.sh`（holdout 四條）、`make_query_pack_holdout_20260906.sh`（holdout 的重取樣雲與第二句文字）。
- Stage 1 checkpoint：`checkpoints/pilotP1s_split801010_lr1e-4_20260904/stage1_best.pt`（sha 074f8d98…）、`checkpoints/s1_scratchbb_sel20_lr1e-4_20260905/stage1_best.pt`（sha 66de6976…）。
- Stage 2：`checkpoints/stage2_arms/S2_P1s_none_ft5e-5_allhouses_20260906/`（9,600 屋、約 10,200 步、66 分；`workflow/stage2_hyperparameters_ft_lr5e-5.json`；code d58f2a5）、`stage2_arms/S2_scratchbb_none_ft5e-5_20260905/`（1,500 屋；紀錄的 uri 已改指 arm 內複本）。
- ProcTHOR 探針：`output/look/exp_stage2_procthor_retrieval_P1s_allhouses_20260906.json`（S1 86.7／S2-off 34.3／S2-on 35.1）。
- 輸出：`output/look/table1_final_{P1s,scratchbb}_{S1head,S2head}_holdout.json`；val 診斷 `output/look/exp_type_level_query_{P1s,scratchbb}_val.json`、`exp_table1_stage2head_{scratchbb,P1s}_val.json`、`exp_mean_pool_weak_trio_val.json`。
- Ledger：DL-101（弱觀測三重奏）、DL-102（最終 Table 1）；環境／路徑修正在同日 ops 條目。
- 環境：一張 32 GB GPU；conda `MetaFind`；torch 2.12.1+cu132；open_clip 3.3.0。
