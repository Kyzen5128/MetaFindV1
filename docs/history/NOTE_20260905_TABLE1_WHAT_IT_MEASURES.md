# Table 1 到底在測什麼；誰用 ULIP-2 做過檢索、怎麼測（2026-09-05 晚）

> **2026-09-07 更正：** 本檔數字是 9 月 5 日的歷史量測，本輪未重跑。先前「融合必定加分」「只有別件觀測才會降分」「論文 query 必定較弱」等結論超出證據，已在下文修正。現行自訂評估見 [CUSTOM_TABLE1_EVALUATION.md](../CUSTOM_TABLE1_EVALUATION.md)，上游任務更正見 [另一份筆記](NOTE_20260905_OTHER_PAPERS_RETRIEVAL_PROTOCOLS.md)。

Kyzen：「釐清 table 1 到底在測什麼；找一下有沒有人也用 ulip2 架構做檢索，怎麼測的，尤其多模那邊」。
另附 Kyzen 轉來的 GPT 分析之逐點對照（§4）。

## 1. Table 1 在測什麼（PAPER，`3experiments.tex` §3.1–3.2 + Table 1）

| 項 | 論文寫的 | 沒寫的 |
|---|---|---|
| 任務 | 資產檢索；Eq. 1 從資產庫選擇 A*，§3.2 說明基線 PC 使用相同 embedding | exact-UID ground truth 是與這段說明相容的實作解讀；作者未提供逐 query 的正解檔與完整判定規則 |
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
| **OmniBind**（arXiv 2407.11895） | 綁定多個空間 | 3D–image 檢索，Objaverse-LVIS **46,205** items | 原文表頭未完整說明 query 方向／視角 | 3D–image 配對評估 | R@1 46.55／R@5 69.92；不能直接與不同 query/gallery 配置比較 |
| Uni3D、OpenShape | 自家編碼器 | 圖／文字→形狀 | 圖或文字 | 形狀向量 | **只有定性圖**；OpenShape 原文：「these input texts are typically not present in the raw texts of the retrieved shapes」 |
| ULIP-2 官方分類入口 | 對齊空間 | 點雲→類別文字 prototype | 點雲 | 類別候選 | 分類 top-k accuracy；不是 MetaFind 七格 retrieval runner |
| Text2Shape 系（TriCoLo、Parts2Words、COM3D、SCA3D、Uni3DL） | 自家 | 文字→形狀，ShapeNet 椅桌 | **一句人寫描述**（每形狀約 5 句） | **只有形狀** | T→S R@1 13～17（1,434 候選） |

**結論**：
- 文獻提供多種單模態與組合輸入示例，但不能推成「全部只有單模態」。
- OpenShape 明寫 P→P 與雙 shape input，OmniBind §4.3 明寫 embedding arithmetic。它們可提供設計參考，仍不等於 MetaFind 完整七格評估協定。
- 單模態 query 的觀測來源：Ex-MCR 用物件**自己的一張視角圖**（跟我們 P1 的 query 圖一樣做法）；Text2Shape 系用**人寫的一句話**（不在 gallery 裡）。
- 2D 世界的類比是 **Composed Image Retrieval**（reference image ＋ modification text → target image）：query 的圖**永遠不是**目標自己的圖，這是任務定義。MetaFind 沒有寫它的 query 是否照這個精神。

## 3. 我們每一格為什麼高——用現有數據回答

| 論文的現象 | 我們的對應數據 | 判讀 |
|---|---|---|
| 基線 PC-only 98 = 同一條向量 | 釋出 ULIP-2 不訓，query 自己那朵雲：PC 100.0 | 一致，這格我們也是「同一條向量」 |
| MetaFind PC-only 75.1（雙塔） | P1s 97.9；從頭訓 backbone 84.4；只訓 Fusion 1 epoch 92.7；gallery 凍結 99.4 | 雙塔在我們這裡沒有把 PC 拉到 75；query 雲＝gallery 雲，Point-BERT 是指紋 |
| 加 T 傷 PC（75→44.5） | 自己的文字：99.7；**別件的文字＋圖**：74.7（P1）／81.9（scratch，9,138） | 在這批實驗中換成別件觀測會降分；不足以證明它是唯一原因 |
| 加 I 傷 PC（75→45.8） | 自己的一張圖：98.5；別件的圖：75.1（P1） | 同樣只支持已測設定的方向 |
| Full 稍高於 T+P | 別件構造下 full 50.4 **低於** T+PC 74.7 | 方向相反：我們的 Fusion 被兩個錯誤模態拖得更慘；論文的 Fusion 三個一起時反而救回 |

## 4. 對 GPT 分析的逐點對照

**更正（INFERENCE）：** 同 UID 的觀測重用可能讓識別變容易，但不保證加一個模態後排名只會上升；融合權重、正規化、模型與候選集都可能改變排名。下面保留當時對同一資產另一份觀測（T'_A、I'_A、P'_A）的量測，僅代表這些已測配置（val，R@1 %）：

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
- 已測的替代觀測仍保留強識別訊號；重取樣列為 PC/T+PC/full = 63.8/94.6/97.3。這沒有重現論文排序，但不能推出其他觀測或模型也不會出現融合降分。
- 把點雲弄壞（去色、半掃、稀疏）能把 pc 格壓到 5～9，但**文字和圖會把它救回來**（full 31～47 > pc）。方向跟論文相反：論文是 PC 最強、T／I 拖後腿；我們弄壞 PC 之後變成 T／I 最強、PC 拖後腿。
- 論文的 PC/T/I 分数是觀察到的結果，不能反推出 query 資料一定更弱。當時實驗没有找到同時吻合所有格子的設定；剩餘原因仍是 UNKNOWN，需要凍結資料、模型與候選集後逐項比較。
- GPT 的第 ③ 點（Point-BERT 微調加強指紋）：凍結 backbone 只訓 Fusion（rung 1）pc 92.7、gallery 整個凍結 99.4——沒有解掉。已測。

## 5. 現在能下的結論

1. PAPER FACT：Table 1 比較七種 query 模態組合與 R@1/R@5；query 觀測、候選名單與正解檔不足以完整重建。
2. UPSTREAM FACT：已有點雲 query 與組合輸入示例；各自的資料、任務、scoring 不能直接當成 MetaFind 的設定。
3. OBSERVED DATA：本檔已測配置大多保留很高的融合分數。這限制了當時的若干假說，沒有證明所有同 UID 融合必定加分，也沒有定位作者較低分的單一原因。
4. IMPLEMENTATION CHOICE：可以用固定 manifest 的自訂方法繼續評估與診斷。現行實作見 [CUSTOM_TABLE1_EVALUATION.md](../CUSTOM_TABLE1_EVALUATION.md)；報告必須說明其自訂性與已使用過的測試集界限，不能用調低分數作為成功標準。

來源：Ex-MCR https://arxiv.org/abs/2310.08884 ；OmniBind https://arxiv.org/abs/2407.11895 ；其餘見 `NOTE_20260905_OTHER_PAPERS_RETRIEVAL_PROTOCOLS.md`。
