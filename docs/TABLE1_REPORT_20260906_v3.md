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
| Stage 2 | ESSGNN＋query 融合層訓，gallery 凍結；scene dropout 0.3；雙向損失；lr 5e-5、1 epoch；P1s 用全部 ProcTHOR 訓練屋（論文 >10,000 屋）、scratchbb 用 1,500 屋 | §2.6 | lr／epoch／屋數：IMPLEMENTATION CHOICE |
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

### 4.2 20% holdout（9,138 → 9,138）最終列

（`chain_table1_final_20260906.sh` 跑完後填。）

## 5. 判讀

（跑完後填。）

## 6. 可追溯

- 鏈子：`logs/chain_thumbnail_query_val.sh`（val 診斷）、`logs/chain_stage2_mainline_20260906.sh`（Stage 2 主線）、`logs/chain_table1_final_20260906.sh`（holdout 最終列）。
- 探針：`tools/probes/exp_type_level_query.py`（`--pc-policies`、`--stage2-state`、`--query-split`）。
- Ledger：DL-101 及其後。
