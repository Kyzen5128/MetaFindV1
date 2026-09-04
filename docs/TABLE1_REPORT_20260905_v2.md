# Table 1 報告 v2（2026-09-05 07:1x）—— 兩顆 backbone、兩種 query 構造、掃描、Stage 2

接續 `docs/TABLE1_REPORT_20260904.md`（P1s，正確論文列已更正）。本檔把 9/5 凌晨整條鏈子與掃描的數字放在一起。
論文目標列（PAPER FACT，`3experiments.tex` Table 1，MetaFind w/o ESSGNN，R@1）：**13.8 / 11.7 / 75.1 / 17.2 / 44.5 / 45.8 / 51.7**。

## 1. 正式協定上的 Table 1（R@1 %）

| 列 | 協定 | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|---|
| P1s：釋出 ULIP-2 backbone，query = 自己觀測 | A20：test 4,569 → holdout 9,138 | 25.0 | 57.2 | 96.7 | 80.4 | 99.5 | 97.6 | 99.6 |
| s1_scratchbb：從頭訓 backbone（DL-095），query = 自己觀測 | holdout 9,138 → holdout 9,138（Kyzen 9/4 的圖） | 24.0 | 52.3 | 84.4 | 81.4 | 99.5 | 96.0 | 99.7 |
| s1_scratchbb，query 文字＋圖 = 同類別別件、點雲自己 | 同上 | 2.5 | 0.8 | 84.4 | 0.7 | 81.9 | 75.3 | 69.1 |
| **論文** | 20% test，gallery 未寫 | **13.8** | **11.7** | **75.1** | **17.2** | **44.5** | **45.8** | **51.7** |

- s1_scratchbb 的 backbone：ULIP-2 官方 `main.py` 在我們 80% 上從頭訓 25 epoch，20% 上 zero-shot top-1 16.2（釋出權重 50.9）。Stage 1 之後 pc 格 84.4（P1s 96.7）——backbone 對齊越弱，兩塔越不容易把「只點雲」對成同一條。
- 「換件」列形狀對（pc 最高、加東西變低），量差 20～37 點。這一列是**我們補的構造**，論文沒寫。
- 選模池：兩顆都照 Kyzen 9/4 的令在 9,138 上選（DL-093），所以 test 半邊不再 held-out（DL-097 已提）。

## 2. Query 觀測掃描（val 4,569 → val 4,569；checkpoint 不動；R@1 %）

| checkpoint | query 文字 | query 圖 | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|---|---|
| P1s | 自己（attrs 句） | 自己單張 | 33.8 | 65.7 | 97.9 | 85.9 | 99.7 | 98.5 | 99.8 |
| P1s | 另一份描述句（desc 根） | 自己單張 | 27.4 | 65.7 | 97.9 | 66.4 | 96.9 | 98.5 | 97.4 |
| P1s | 類別＋尺寸 | 自己單張 | 17.2 | 65.7 | 97.9 | 74.7 | 98.2 | 98.5 | 98.8 |
| P1s | Sketchfab 名稱 | 自己單張 | 6.8 | 65.7 | 97.9 | 52.6 | 94.6 | 98.5 | 96.4 |
| P1s | 同類別別件 | 同類別別件 | 5.5 | 1.2 | 97.9 | 1.2 | 94.4 | 96.1 | 86.1 |
| s1_scratchbb | 自己 | 自己單張 | 32.6 | 60.9 | 88.4 | 86.4 | 99.7 | 97.4 | 99.9 |
| s1_scratchbb | 另一份描述句 | 自己單張 | 21.4 | 60.9 | 88.4 | 58.7 | 94.4 | 97.4 | 97.1 |
| s1_scratchbb | 類別＋尺寸 | 自己單張 | 14.0 | 60.9 | 88.4 | 76.8 | 97.6 | 97.4 | 98.8 |
| s1_scratchbb | Sketchfab 名稱 | 自己單張 | 3.8 | 60.9 | 88.4 | 47.5 | 86.6 | 97.4 | 94.5 |
| s1_scratchbb | 同類別別件 | 同類別別件 | 4.5 | 0.8 | 88.4 | 0.8 | 85.0 | 78.9 | 73.4 |
| 論文 | ? | ? | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |

讀法：
- 只換文字（圖和點雲留自己）：T+PC、full 幾乎不動（≥ 94）。文字不是拖得動點雲的那個模態。
- 圖換成別件才把 I+PC、full 拉下來。**圖是關鍵**。但「同類別別件的圖」太弱（image 0.8），論文 image 是 11.7。

補：9/4 的縮圖探針（P1 舊 checkpoint，protocol D，gallery 41,123；已歸檔 `output_look_20260905/probes/exp_type_level_query_thumbnail.json`）裡有「同一件、但**Sketchfab 縮圖**」當 query 圖的列：

| query 文字 | query 圖 | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|---|
| 自己 attrs 句 | 自己的縮圖 | 12.0 | 16.8 | 65.0 | 42.5 | 95.4 | 72.8 | 94.8 |
| Sketchfab 名稱 | 自己的縮圖 | 0.9 | 16.8 | 65.0 | 14.5 | 56.6 | 72.8 | 65.3 |
| 類別＋尺寸 | 自己的縮圖 | 2.7 | 16.8 | 65.0 | 26.0 | 74.8 | 72.8 | 77.9 |
| 論文 | | 13.8 | 11.7 | 75.1 | 17.2 | 44.5 | 45.8 | 51.7 |

「自己的縮圖」讓 image 格落到 16.8（論文 11.7）、pc 65.0（論文 75.1）、I+PC 72.8——三格靠近了，但 T+PC 95.4、full 94.8 仍遠高於 44.5／51.7，因為文字是自己那句。換成名稱，T+PC 降到 56.6 但 text 掉到 0.9。**沒有一列七格同時對上。**

## 3. 這代表什麼（讀完論文後的判讀）

論文 Table 1 有一個我們任何構造都做不出的組合：**text 單獨 13.8（有用），可是加到 pc 上把 75.1 拉到 44.5（大傷）**。
- 在我們的模型裡，一份單獨能找到 12% 的文字，加到 pc 上一定是幫忙（95.4），不會傷。因為 Fusion 學到「點雲可靠就靠點雲」。
- 論文所有基線（mean pooling）也是「加文字就傷」：pc 98 → T+PC 34，而它們的 text 只有 0.1。MetaFind 的 44.5 落在「mean pooling 的 34」和「我們的 94」之間。
- → **推論（INFERENCE）**：論文的 Fusion 行為接近平均池化，沒有學到偏重點雲。可能是訓練很短、或 Fusion 很接近初始。這跟論文自己那句「dual-tower 引入 cross-modality retrieval 所以 pc 較低」一致：兩座塔沒對齊到我們這種程度。
- 另一條：論文 pc 格 R@5 78.0 只比 R@1 高 2.9，約 22% 的 query 連前五都進不去——像資料不對應，非對得不準。

所以差距是兩層疊起來：(1) query 文字／圖來源比我們弱（縮圖、名稱那種量級）；(2) Fusion 收斂程度比我們低很多。第 (1) 層我們有材料能逼近；第 (2) 層要「故意訓少」才能重現，那是**反向配數字**，不做，除非你點頭當成診斷。

## 4. Stage 2（ProcTHOR，300 間測試房，19,305 query，gallery 1,439；R@1 / R@5）

| 頭 | s1_scratchbb | 之前 S2D（釋出 backbone） |
|---|---|---|
| S1（Stage 1 fusion，無 layout） | 81.8 / 99.0 | 82.4 / 98.1 |
| S2-off（Stage 2 fusion，無 layout） | 44.3 / 87.2 | 36.8 / 78.0 |
| S2-on（＋ λ·ESSGNN） | 43.9 / 86.2 | 32.8 / 76.8 |

Stage 2 把無 layout 的頭從 82 打到 44；加 layout 沒有幫助。論文 w/ ESSGNN 只掉 3～12 點。Stage 2 的正例／query 定義論文沒寫，這裡是我們的選擇（`stage2_protocol.json`）。

## 5. 可追溯

- 從頭訓 backbone：`ulip2_pretrain_run/outputs/ulip2_scratch_metafind80_lr3e-3_ep25_20260904/checkpoint_best.pt`（epoch 23，acc1 16.2），瘦身檔 `models/ulip2_scratch/ulip2_scratch_metafind80_lr3e-3_ep25_20260904_best.pt`。
- Stage 1：`checkpoints/s1_scratchbb_sel20_lr1e-4_20260905/stage1_best.pt`（epoch 9，holdout mean R@1 0.7675），`--backbone-ckpt … --non-official-initialiser --selection-split holdout --lr 1e-4 --epochs 10`。
- 索引／閘：`gallery_index.json`（新 sha 已發布），G4 PASS。
- Table 1 輸出：`eval/table1_s1_scratchbb_sel20_lr1e-4_20260905_own/`、`…_partner/`；掃描：`eval/sweep_qobs_*/`；Stage 2：`checkpoints/stage2_arms/S2_scratchbb_none_ft5e-5_20260905/`、`output/look/exp_stage2_procthor_retrieval_scratchbb_20260905.json`。
- 鏈子：`logs/chain_full_pipeline_scratchbb.sh`、`logs/sweep_query_observation.sh`；決策：DL-093～DL-097。

## 6. 待 Kyzen 決定

1. 最終報告的選模池：val 4,569（D-3b 規定，test 保持 held-out）或 9,138（你的圖，選模＝報告）。
2. 要不要把某一種 query 構造（例如「自己的縮圖 ＋ 較弱文字」）當成我們**補的** Table 1 定義寫進報告——論文沒寫，只能標 IMPLEMENTATION CHOICE。
3. 要不要做「Fusion 收斂程度」的診斷梯（mean pooling → 訓 1 epoch → 訓 10 epoch，同一組 query 構造），驗證第 3 節的推論。這會逼近論文數字，但性質是診斷，不能當復現。
