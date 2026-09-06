# 資料集重建計畫（2026-09-06）—— 照論文重做，只有 LLM 例外

Kyzen 07:0x：「重新處理資料集，把之前的錯誤都修正，這次資料集處理方法直接採用論文所描述的；擬定完流程後二次確認；除了 LLM 其他一律按照論文要求；不清楚就找資料」「全線都給你 直到完成」。

規則：每一步先寫論文原句（PAPER），沒寫的去找上游／作者材料（UPSTREAM／POSTER），還是沒有就標 IMPLEMENTATION CHOICE 並寫理由；Kyzen 9/2 已裁的（DL-077/078/079）直接沿用。舊資料一律不動，全部寫進新根目錄。

## 0. 論文對資料的全部陳述（逐字，`2methdology.tex` §2.3、`3experiments.tex` §3.1、Figure 1/2、海報）

| # | 原句 | 出處 |
|---|---|---|
| P1 | "we utilize the Objaverse-LVIS dataset, which comprises approximately 48,000 distinct 3D assets" | §2.3 |
| P2 | "Each asset is rendered from **11 orthogonal viewpoints** and annotated using GPT-4o" | §2.3 |
| P3 | "These annotations provide rich textual descriptions detailing attributes such as object category, size dimensions, materials, and placement constraints" | §2.3 |
| P4 | Figure 2 的 JSON：`{"annotations": {"category", "synset", "width", "length", "height", "volume", "mass", "description", "materials", "onCeiling", "onWall", "onFloor", "onObject"}}`，例：robot／robot.n.01／30／30／40／36000／2.5／一句描述／["metal","glass","plastic"]／false/false/true/true | Figure 2（`data-preprocess.png`，海報同圖） |
| P5 | Figure 2 左上：同一物件 8 張不同方位與**不同仰角**的渲染圖（含近俯視） | Figure 2 |
| P6 | Figure 1 的 query 文字：`Platform Bed {size:..........}`；query 圖＝一張床的渲染；query 點雲＝床 | Figure 1（`MetaFind.drawio.png`，海報同圖） |
| P7 | "ProcTHOR, which includes over 10,000 generated houses constructed from a curated collection of more than 3,000 unique assets. Each room configuration provides precise spatial coordinates and comprehensive semantic metadata for each asset" | §2.3 |
| P8 | 場景圖兩種邊：(i) physical-relation "spatial dependencies (e.g., 'cup on table')"；(ii) semantic-relation "functional or contextual associations (e.g., 'microscope–lab bench'), obtained by prompting an LLM on object pairs" | §2.3 |
| P9 | 兩個資料集都 80% 訓練／20% 測試 | §3.1 |
| P10 | "48K high-quality 3D assets with structured textual descriptions and multi-view image renders" | §3.2 |
| P11 | ESSGNN 節點 h_i^(0) = Concat(x_i, t_i)，t_i 文字衍生特徵；邊 e_ij = LLM 關係句經凍結文字編碼器 | §2.5 |

論文**沒寫**的：相機參數／解析度／背景／投影；VLM 的 prompt、餵幾張圖、輸出格式以外的細節；點雲來源與點數；文字怎麼進 CLIP（JSON 原字串？哪些欄位？）；影像 11 張怎麼變一條向量；切分 seed；ProcTHOR 節點文字怎麼來；語意邊的 LLM 與 prompt；adjacency 門檻。程式碼與資料沒有公開（海報 QR 已失效、OpenReview 擋機器人、搜尋無 repo；見 ledger 07:5x）。

## 1. 現況與要改的地方（「之前的錯誤」）

| 步 | 現在（v6 語料） | 論文 | 判定 → 這次 |
|---|---|---|---|
| 資產數 | 46,052 glb；45,692 進語料（332 排除：311 標註失敗、21 人工剔除） | ~48K（LVIS 實際 46,207） | 到不了 48K（DEVIATION，不可避免）。**311 個失敗重試**（舊 bug：重試種子綁 uid，重跑等於沒重跑）；21 個維持 Kyzen 人工剔除 |
| 視角 | **12** 張（OpenShape 三圈×4） | **11**（P2） | **改 11**：Kyzen 9/2 甲（DL-077 Q2、DL-079 Q3 嚴格重渲）：單圈 11 個等距方位角、仰角 20°、透視、512 px、黑底合成；渲染器 v7 程式已備妥但**從未跑過** |
| 渲染引擎 | OpenShape 的 Blender 腳本（CYCLES、camera_dist 1.2、35 mm 鏡頭、bbox 立方正規化 ×0.8、一盞 30 kW 面光、透明 RGBA 存檔；OptiX 是我們加的） | 沒寫 | 維持。單圈是 UPSTREAM FACT（ULIP-2 §4.1：Blender、一圈 12 張每 30°）；20°／1.2／512 px 上游沒有（UNKNOWN）；單圈對上論文的「orthogonal」與 Figure 2 的多仰角是 Kyzen 裁的 DEVIATION（DL-077 Q2），不是論文沉默 |
| VLM | gemma-4-12B-it | GPT-4o | 維持（Kyzen 裁定的唯一例外） |
| VLM 輸入 | 12 張 | 11 張（P2 的「rendered from 11 ... and annotated」） | **11 張全餵** |
| 類別 | 提供 LVIS 標籤給模型，模型只能細化不能改（v8/v9） | VLM 產生類別（P3、P4）；有沒有給標籤沒寫 | **維持提供標籤**：論文沒說不給；v7 實測不給標籤時 gemma 錯認 28%（GPT-4o 不會這麼差）。重分類為 IMPLEMENTATION CHOICE（DL-007 當年標 DEVIATION 是把「VLM 產生」讀成「不給提示」） |
| 尺寸 | 模型只估高度，寬／長由網格比例算出 | VLM 給 width/length/height（P4 為整數公分） | **改成 VLM 直接估三個尺寸**（P4）；網格比例只存側檔供診斷，不進標註 |
| volume／mass | 有 | P4：volume = w×l×h（36000 = 30·30·40）、mass 2.5 | volume 由 w×l×h 算；mass 由 VLM 估 |
| synset | LVIS 表查詢；模型類別非 LVIS 詞則 WordNet | P4：VLM 給 | 問 VLM，格式檢查；無效則 WordNet 查（IMPLEMENTATION CHOICE） |
| description | 5 個候選由 CLIP-L 排名選能塞進 77 token 的 | 一句（P4 例句約 25 詞） | **一句，不排名**；prompt 要求像 P4 例句那樣一句話 |
| 文字進 CLIP | 句型模板（v2_cm／attrs_v1） | P4 JSON、P6 `Platform Bed {size}` | **主線：gallery 文字＝Figure 2 JSON 原字串**（欄位順序、整數格式照 P4）。CLIP 77 token 截斷（論文自己的例子就是 135 token；CLIP 看到的是 category／synset／三個尺寸／volume／mass／description 前十幾個字，**materials 與四個擺放旗標永遠進不了編碼器**）。編碼器原本會拒收超長文字（審查 BLOCKER），現在只在這個模板下改成照 CLIP 的方式截斷並記 `text_truncated`。INFERENCE＋IMPLEMENTATION CHOICE。**對照臂：v2_cm 句型**（全部欄位、塞得進 77），同一語料再訓一次 Stage 1，量這個字面讀法的代價 |
| 影像→一條向量 | 12 張各過 ViT-bigG 取平均 | 沒寫；上游 ULIP-2 訓練時每步隨機一張、零樣本評估不用圖、只對文字 prompt 做 normalise→mean→normalise | 11 張取平均（DL-079 Q5 主線，Kyzen 裁的 IMPLEMENTATION CHOICE） |
| 點雲 | ULIP-2 官方 10,000 點 xyz+rgb | 沒寫 | 維持（UPSTREAM） |
| 切分 | 80/10/10（20% 再對半） | 80/20（P9） | **80/20**，seed 20260816；選模與報告都在 20%（Kyzen 9/4 圖） |
| ProcTHOR 渲染 | 11 張、224 px、正交、白底、AI2-THOR 天空盒（退役協定） | 沒寫 | **統一成 Objaverse 同協定**（DL-077 Q11 甲）：AI2-THOR 重渲 1,467 件 |
| ProcTHOR 節點文字 | 類別名 "a counter top"（93 句） | "comprehensive semantic metadata"（P7） | 9/2 的前提「metadata 不存在」是錯的：ProcTHOR 官方 release 有 `asset-database.json`、`ai2thor-object-metadata.json`、`placement-annotations.json`、`receptacles.json`（沒下載而已）。**先抓下來讀，再決定 t_i 的文字來源**（上游 metadata 優先於 gemma 描述；DL-077 Q10 是在錯前提下裁的） |
| 場景圖邊 | support ∪ kNN(k=8) adjacency；語意邊 = gemma 一句話→凍結 CLIP 文字塔；**圖的單位＝整棟房子** | P8；論文四處寫 **room-level／each room／single-room scenes** | 邊：維持（support 是 ProcTHOR 的 children 樹；kNN k=8 是 IMPLEMENTATION CHOICE）。**單位要改成每個房間一張圖**（審查 MAJOR：現在 kNN 會穿牆連到隔壁房間）。h⁰ = t_i（不是 Concat(x_i, t_i)）是論文與其附錄前提互相矛盾後已記錄的解法，維持 |

## 2. 流程（依賴順序）與成本

新根目錄 `/home/kyzen/metafind/metafind_data_paper/`（`METAFIND_DATA`），實體資料在 `metafind_out/{renders_v7,annotations_v10,embeddings_v10}`；`pointclouds` 共用；舊語料完全不動。

```
R1  渲染 v7        46,052 glb → 11 張/件；8 個 Blender 行程 → 實測 120 件/分，6 小時（08:15–14:14）；46,012 成功、40 隔離  GPU  ✔
R2  標註 v10       gemma，11 張一次 prefill，Figure 2 十三欄，一次呼叫（`annotate_run --prompt-mode figure2_v10`，`metafind/data/annotate_v10.py`）；先 5 件煙霧測；全量 46,012 件    GPU（R1 之後）
R3  點雲           不動
R4  編碼 v10       文字 = figure2_json；圖 = 11 張平均；約 2～3 小時                      GPU
R5  切分           80/20 seed 20260816，資產層級
R6  ProcTHOR 統一  AI2-THOR 重渲 1,467（小時級）→ gemma 描述（2.5 h）→ 物件文字 → 語意邊（35 分）
                   → 節點向量 → Stage 2 索引（10 分）                                     GPU（可與 R2 交錯）
R7  Stage 1        論文字面線（DL-077 Q4／DL-079 Q5）：兩塔同一筆紀錄、query 30% 遮罩、
                   影像＝11 張平均（兩側同一向量）；Point-BERT 微調、CLIP 凍結；lr 1e-4、10 epoch、
                   batch 64、τ 0.5、單向 InfoNCE；在 20% 上選模                          GPU 數小時
R8  Stage 2        9,600 屋、lr 5e-5、1 epoch（預先登錄，Table 1 之後不改）、scene dropout 0.3、雙向；兩座 encoder 凍結，只訓 ESSGNN＋query 融合   GPU 約 1 小時
R9  Table 1        兩列（Stage 1 頭／Stage 2 頭 layout 不在）、20% → 20%；own 為正式列，
                   weak own／partner 為對照列                                             GPU 半小時
```

## 3. 二次確認（執行前）

1. 本文件每一列對照 §0 的原句與 ledger 的裁決，我自己再核一次（✔ 2026-09-06 08:0x）。
2. 渲染器 v7 先煙霧測 3 件：確認 11 張、仰角 20°、透視、512 px、側檔記錄 `renderer_version: 7`；肉眼看圖。（✔ 08:1x；全量 ✔ 14:14）
3. 唸給兩位唯讀審查者（`ulip2-reviewer`：R1–R5、R7；`essgnn-reviewer`：R6、R8）對照論文與上游，發現寫回本文件再動 R2 以後的步驟。R1 的參數是 Kyzen 9/2 的裁決，煙霧測通過即開跑（它是最長的一段）。
4. 每個長工作啟動前寫 ledger：目的、命令、輸出路徑、預估時間；跑完寫結果。

## 4. 不做、寫清楚

- 48K 到不了；GPT-4o → gemma；ProcTHOR 節點語意來源不存在（用 gemma 描述補）；ViT-bigG 不微調（32 GB 裝不下 AdamW 狀態）。
- Table 2（GPT-4o／5 位專家評 200 個場景）不在本次範圍。
- 不寄信給作者（對外動作，需 Kyzen ✅）。

## 5. 可追溯

- 本文件；ledger DL-103（開跑）與後續；`workflow/DATA_PLAN_PAPER_FIRST.md`（9/2 原計畫）、DL-077/078/079（裁決）。
- 程式：`metafind/data/render_blender.py`（v7 常數）、`metafind/data/annotate.py`（v10 prompt，待改）、`metafind/models/resolve_stage1.py`（`figure2_json`）。

## 6. 執行紀錄

- 08:15–14:14 R1 完成：46,012／46,052，40 隔離（33 全空白圖、7 未知；34 個在 v6 也失敗）。`renders_index.jsonl` 46,012。
- 15:0x R2 程式完成（v10）＋單元測試 13 個通過；5 件煙霧測啟動。
- 後續鏈子 `logs/chain_paper_pipeline_20260906.sh`（等 R2 的 `=== R2 DONE`）：21 件人工剔除 → n05b → n06 → n09 → Stage 1（same_record、same_mean、lr 1e-4、10 epoch、20% 選模）→ n11/G4/n12 → Table 1 列 1 → n11b → Stage 2（9,600 屋）→ ProcTHOR 探針 → Table 1 兩列（holdout）→ 表格。

### 6.1 二次確認結果（15:05–15:25，兩位唯讀審查者）

ULIP-2 側：BLOCKER 1（編碼器拒收超長文字 → 改成 figure2_json 模板下截斷並記錄）、MAJOR 2（JSON 數字格式 `30.0`→`30`；v10 沒檢查 11 張／renderer v7 → 加守門）、MINOR 6（synset 存在性、重試 salt、類別字數、描述下限、materials 超過 6 拒收、traceback）、INFO 4（上游渲染細節、影像聚合的上游做法、CLIP 凍結的標註、盲標註）。
ESSGNN 側：MAJOR 1（場景圖單位應為房間）、MINOR 5（兩座 encoder 凍結的措辭、h⁰、節點文字格式未定、ProcTHOR metadata 其實存在、統一渲染路徑未實作）、INFO 4。
全部處置寫在 ledger DL-103「Second check」條。
