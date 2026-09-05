# Table 1 到底在測什麼；誰用 ULIP-2 做過檢索、怎麼測（2026-09-05 晚）

Kyzen：「釐清 table 1 到底在測什麼；找一下有沒有人也用 ulip2 架構做檢索，怎麼測的，尤其多模那邊」。
另附 Kyzen 轉來的 GPT 分析之逐點對照（§4）。

## 1. Table 1 在測什麼（PAPER，`3experiments.tex` §3.1–3.2 + Table 1）

| 項 | 論文寫的 | 沒寫的 |
|---|---|---|
| 任務 | **instance 檢索**：給某個資產的部分模態，從 gallery 找回**同一個資產** | — |
| 資料 | Objaverse-LVIS 48K，80% 訓／20% 測 | gallery 是 20% 還是 48K |
| query | 七種模態組合：T、I、PC、T+I、T+PC、I+PC、T+I+PC；缺的模態在 query 塔用 masked embedding | **query 的 T／I／PC 各自是哪一份觀測**（自己的？別的視角？別的描述？） |
| gallery | 「pre-encoded asset database」；gallery 塔三模態齊全 | — |
| 指標 | R@1、R@5 | tie 規則、是否多 seed |
| 基線 | 預訓練單塔編碼器 ＋ mean pooling 合併可用模態，對「pre-encoded gallery」檢索 | 基線 gallery 是哪個向量 |
| 作者對數字的說明 | 基線 PC-only 98 是「query 與 gallery 用**同一條向量**」造成的膨脹；MetaFind 雙塔「introduces more cross-modality retrieval」所以 PC-only 較低（75.1） | — |

**論文列的形狀（GPT 的修正，我採納）**：不是「模態越多越低」，而是

$$P\ (75.1) > \text{Full}\ (51.7) > T{+}P\ (44.5) \approx I{+}P\ (45.8) \gg T\ (13.8) \approx I\ (11.7) \approx T{+}I\ (17.2)$$

PC 單獨最好；加弱的 T 或 I 會傷 PC；三個一起時稍微救回一點。

## 2. 誰用 ULIP-2 做過檢索、怎麼測

| 論文 | 3D 空間 | 檢索任務 | query 是什麼 | gallery | 數字 |
|---|---|---|---|---|---|
| **Ex-MCR**（arXiv 2310.08884） | **ULIP v2 (PointBERT)** 凍結 | 3D–image 檢索，Objaverse-LVIS | 「ULIP v2 為每個物件提供 12 張渲染圖，**隨機選一張當配對影像**」→ 物件**自己的一張視角圖** | 3D 物件 | ULIP v2：mAP 11.41、R@1 6.00、R@5 15.63；Ex-MCR 6.23／2.54／8.25（gallery 尺寸未在摘取內註明） |
| **OmniBind**（arXiv 2407.11895） | 綁定多個空間（含 ULIP-2／Uni3D 系） | 3D–image 檢索，Objaverse-LVIS **46,205** 件 | 資料集渲染圖 | 3D 物件 | R@1 46.55／R@5 69.92 |
| Uni3D、OpenShape | 自家編碼器 | 圖／文字→形狀 | 圖或文字 | 形狀向量 | **只有定性圖**；OpenShape 原文：「these input texts are typically not present in the raw texts of the retrieved shapes」 |
| ULIP-2 本身 | — | — | — | — | 沒有檢索實驗 |
| Text2Shape 系（TriCoLo、Parts2Words、COM3D、SCA3D、Uni3DL） | 自家 | 文字→形狀，ShapeNet 椅桌 | **一句人寫描述**（每形狀約 5 句） | **只有形狀** | T→S R@1 13～17（1,434 候選） |

**結論**：
- 有人用 ULIP-2 做檢索，但**全部是單模態 query → 3D**（圖→3D、文字→3D）。
- **沒有任何一篇做「多模態組合 query」（T+I、T+PC、T+I+PC）的 3D 檢索。** MetaFind 的七格表在領域裡沒有先例。
- 單模態 query 的觀測來源：Ex-MCR 用物件**自己的一張視角圖**（跟我們 P1 的 query 圖一樣做法）；Text2Shape 系用**人寫的一句話**（不在 gallery 裡）。
- 2D 世界的類比是 **Composed Image Retrieval**（reference image ＋ modification text → target image）：query 的圖**永遠不是**目標自己的圖，這是任務定義。MetaFind 沒有寫它的 query 是否照這個精神。

## 3. 我們每一格為什麼高——用現有數據回答

| 論文的現象 | 我們的對應數據 | 判讀 |
|---|---|---|
| 基線 PC-only 98 = 同一條向量 | 釋出 ULIP-2 不訓，query 自己那朵雲：PC 100.0 | 一致，這格我們也是「同一條向量」 |
| MetaFind PC-only 75.1（雙塔） | P1s 97.9；從頭訓 backbone 84.4；只訓 Fusion 1 epoch 92.7；gallery 凍結 99.4 | 雙塔在我們這裡沒有把 PC 拉到 75；query 雲＝gallery 雲，Point-BERT 是指紋 |
| 加 T 傷 PC（75→44.5） | 自己的文字：99.7；**別件的文字＋圖**：74.7（P1）／81.9（scratch，9,138） | 只有 query 文字**不認得這件**時才會傷 |
| 加 I 傷 PC（75→45.8） | 自己的一張圖：98.5；別件的圖：75.1（P1） | 同上 |
| Full 稍高於 T+P | 別件構造下 full 50.4 **低於** T+PC 74.7 | 方向相反：我們的 Fusion 被兩個錯誤模態拖得更慘；論文的 Fusion 三個一起時反而救回 |

## 4. 對 GPT 分析的逐點對照

GPT 的診斷（query 三模態全是同一 UID 的證據 → 多一模態＝多一份證據 → 合併格只會升）與 DL-098～DL-100 的結論相同，**同意**。它提的「最乾淨的實驗」是：同一資產、三個模態都換成另一份觀測（T'_A、I'_A、P'_A）對 canonical gallery。**這三軸我們都已量過**，結果如下（val，R@1 %）：

| 軸 | 換成什麼 | 出處 | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|---|---|
| 文字 | 另一份描述句（desc_v1） | 掃描 DL-098 | 27.4 | 65.7 | 97.9 | 66.4 | 96.9 | 98.5 | 97.4 |
| 圖 | 12 張裡的一張（不含在平均裡的做法同值） | P1s | 33.8 | 65.7 | 97.9 | 85.9 | 99.7 | 98.5 | 99.8 |
| 三個一起 | P5 arm：另一描述＋重取樣雲＋單張圖 | ARMS_TABLE C | 37.9 | 73.0 | 96.6 | 83.5 | 98.1 | 98.8 | 99.1 |
| 點雲 | 重取樣 10k | `exp_query_pc_observation` B（gallery 36,554） | 11.6 | 29.7 | 63.8 | 67.5 | 94.6 | 76.2 | 97.3 |
| 點雲 | 去顏色 | 同上 | 11.6 | 29.7 | 8.9 | 67.5 | 20.4 | 14.1 | 31.4 |
| 點雲 | 半掃描 | 同上 | 11.6 | 29.7 | 5.4 | 67.5 | 28.0 | 17.2 | 46.6 |
| 點雲 | 稀疏 1k | 同上 | 11.6 | 29.7 | 0.1 | 67.5 | 1.5 | 1.1 | 5.4 |

讀法：
- 同一資產的另一份觀測（另一句描述、另一張圖、重取樣的雲）**仍然認得這件**：cos 0.80／0.93／0.99。所以合併格照樣 ≥ pc。GPT 預測的「pc 78、T+PC 55、full 60」不會出現；重取樣列就是答案（63.8 / 94.6 / 97.3）。
- 把點雲弄壞（去色、半掃、稀疏）能把 pc 格壓到 5～9，但**文字和圖會把它救回來**（full 31～47 > pc）。方向跟論文相反：論文是 PC 最強、T／I 拖後腿；我們弄壞 PC 之後變成 T／I 最強、PC 拖後腿。
- 論文的形狀需要**同時**：PC 認得這件（75）、T／I 不太認得（13.8／11.7）。在我們的資料裡，「T／I 不太認得」只有換成別件才做得到。所以 GPT 排序的第一位（observation overlap）我同意，但它建議的實驗已經做過、答案是否定的；剩下的事實是：**論文的 query 文字／影像來源比「同一資產的另一份觀測」弱得多**，論文沒寫它是什麼。
- GPT 的第 ③ 點（Point-BERT 微調加強指紋）：凍結 backbone 只訓 Fusion（rung 1）pc 92.7、gallery 整個凍結 99.4——沒有解掉。已測。

## 5. 現在能下的結論

1. Table 1 是「用資產的部分模態找回同一資產」；論文沒寫 query 各模態的來源。
2. 領域裡沒有多模態組合 query 的 3D 檢索先例；單模態先例（Ex-MCR、Text2Shape 系）的 query 都不是 gallery 裡那份向量。
3. 我們所有「同一資產、另一份觀測」的構造都認得這件 → 合併格 ≥ pc；只有 query 文字／圖來自別件才翻。論文的文字／圖介於兩者之間（text 13.8），我們手上沒有這種觀測。
4. 這不再是實驗能解的問題，是資料定義問題：要 (a) 問作者，或 (b) Kyzen 決定一種「query 文字／圖」的來源當我們補的定義。

來源：Ex-MCR https://arxiv.org/abs/2310.08884 ；OmniBind https://arxiv.org/abs/2407.11895 ；其餘見 `NOTE_20260905_OTHER_PAPERS_RETRIEVAL_PROTOCOLS.md`。
