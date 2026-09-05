# 其他論文的 3D 檢索是怎麼定 query／gallery 的（2026-09-05）

Kyzen：「你去找其他篇論文怎麼做的」。看的是 MetaFind Table 1 的基線與同領域的 3D 檢索工作，只問一件事：
**query 拿的是什麼、gallery 裝的是什麼、正解怎麼定、指標多少。**
來源：本機的論文原檔／官方 repo（OpenShape、Uni3D、TriCoLo、Parts2Words、text2shape），以及 arXiv HTML（SCA3D 2502.19128、COM3D 2405.04103、Uni3DL 2312.03026、OmniBind 2407.11895）。arXiv 部分是網頁摘取，引句以其為準，未逐字讀全文。

## 1. 一覽

| 論文 | 資料集 | query | gallery | 正解 | 指標與數字 |
|---|---|---|---|---|---|
| Text2Shape (2018) | ShapeNet 椅桌，Text2Shape 描述 | **一句人寫的描述**（每形狀約 5 句） | 形狀（體素／形狀編碼器），**gallery 裡沒有文字** | 描述所屬的那個形狀 | text→shape RR@1 0.40／RR@5 2.37（我們 9/4 逐字讀過） |
| TriCoLo (2022) | 同上 | 同上 | 同上（三模態對比訓練，檢索時形狀端） | 同上 | 官方 README 表列 RR@1／RR@5／NDCG@5；評估碼改自 Text2Shape |
| Parts2Words (2023) | 同上 | 同上 | 同上 | 同上 | text→shape RR@1 12.7／RR@5 33.0（Uni3DL 表引） |
| COM3D (2024) | 同上，測試 1,434 形狀 ×~5 句 | 同上 | 同上 | 同上 | T→S RR@1 13.12／RR@5 33.48／NDCG@5 23.89；S→T 20.03／48.32／15.62；**沒用 Objaverse** |
| SCA3D (2025) | 同上（train 11,498／test 1,434） | 同上 | 同上 | 同上 | T→S RR@1 16.67／RR@5 38.90／NDCG@5 28.17；S→T 27.22／55.56／19.04 |
| Uni3DL (2023) | Text2Shape；另有 **Cap3D-Objaverse**（660K 生成描述，**80/20 隨機切**） | Cap3D 生成的描述 | 形狀 | 描述所屬形狀 | Text2Shape T→S R@1 5.8／R@5 19.7；Cap3D 部分表內數字同為 5.8／19.7（網頁摘取，無基線） |
| OmniBind (2024) | Objaverse-LVIS **46,205** 件 | **影像**（資料集的渲染圖） | 3D 物件 | 同一件 | 3D-image retrieval R@1 **46.55**／R@5 69.92；zero-shot 分類 top-1 64.67 |
| OpenShape (2023) | Objaverse 全集 | 一張圖／一段文字／一朵點雲 | 形狀向量，cosine kNN | — | **只有定性圖**；原文：「these input texts are typically not present in the raw texts of the retrieved shapes」 |
| Uni3D (2024) | — | 圖／文字 | 形狀 | — | README 只有定性圖 |
| ULIP-2 (2024) | — | — | — | — | 沒有檢索實驗，只有 zero-shot 分類 |

## 2. 共通規則（整個領域）

1. **gallery 只有形狀。** 沒有任何一篇把「形狀自己的文字」或「形狀自己的圖」放進 gallery 向量。檢索一律是跨模態：文字→形狀、圖→形狀。
2. **query 文字是形狀「之外」的東西**：人寫的描述（Text2Shape 系）或另外生成的描述（Cap3D）。同一形狀有多句，任何一句都可當 query；沒有一篇拿「建 gallery 用的那份文字」回頭當 query。
3. **正解 = 同一個形狀（instance）**，不是同類別。
4. **數字量級**：人寫描述→形狀，1,434 個候選，R@1 落在 **13～17**（SCA3D 最高 16.7）；生成描述→形狀（Cap3D，大候選池）R@1 **5.8**；渲染圖→形狀，46,205 個候選，R@1 **46.6**（OmniBind）。

## 3. 對 MetaFind Table 1 的意義

- MetaFind 的 **text-only 13.8** 正好落在「一句描述→形狀」的領域水準（13～17）。這不像「gallery 裡含同一份文字」的分數（我們自己那份文字當 query 是 34～52）。**推論**：論文的 query 文字是一份 gallery 沒見過的描述——跟整個領域一樣。
- MetaFind 的 **image-only 11.7** 遠低於 OmniBind 用渲染圖查 46,205 件的 46.6（我們釋出 ULIP-2 在 4,569 件是 70.4）。**推論**：論文的 query 圖不是標準的物件渲染圖，或它的影像塔被改弱了。
- MetaFind 的 **pc-only 75.1**：全領域沒有「點雲查點雲」的檢索基準（大家都當它是平凡的自己找自己，論文基線列的 98 也這樣說）；論文自己的解釋是雙塔混合。
- MetaFind 的 gallery 含三模態、query 是子集，這個設計**全領域沒有先例**；所以「query 文字要跟 gallery 文字不同」這條領域規則，在 MetaFind 裡沒有現成寫法可抄。
- 11 個視角 × GPT-4o：論文說「rendered from 11 views and processed with GPT-4o」。若是**每個視角各一份描述**，就天然有 11 份文字可供「gallery 用一份、query 用另一份」，跟領域做法一致。我們只有一份標註（Gemma），所以做不到這一層；這是資料層的缺口，不是程式。

## 4. 我們試過的對應

- P5：另一份「同一份標註的不同寫法」＋另一張視角 → full 99.1（文字 cos 0.80，太像；點雲仍是自己的）。
- 掃描（DL-098）：換掉文字（描述句／類別尺寸／名稱）動不了合併格；只有換圖才動。
- 領域做法沒有 query 帶點雲的例子，所以「query 點雲＝gallery 點雲」造成的 98 沒有前例可對。

## 5. 來源

- SCA3D: https://arxiv.org/abs/2502.19128 ；COM3D: https://arxiv.org/abs/2405.04103 ；Uni3DL: https://arxiv.org/abs/2312.03026 ；OmniBind: https://arxiv.org/abs/2407.11895
- 本機：`docs/paper/openshape_source/sections/experiments.tex`（Multi-modal 3D Shape Retrieval 段）、`/home/kyzen/upstream/Uni3D/README.md`、`/home/kyzen/upstream/tricolo/README.md`、`/home/kyzen/upstream/Parts2Words/README.md`、`docs/NOTE_20260904_TEXT2SHAPE_READ.md`。
