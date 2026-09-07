# 其他論文的檢索協定：2026-09-07 更正

這份筆記更正 9 月 5 日把有限文獻觀察推成「整個領域規則」的錯誤。上游可以提供可借用的評估方法，不能補成 MetaFind 作者未公開的 Table 1 設定。本文不是新的研究決策。

## 可確認的差別

| 來源與分類 | 實際任務／方法 | 對 MetaFind Table 1 的界限 |
| --- | --- | --- |
| ULIP-2、OpenShape、Uni3D 官方分類入口（UPSTREAM FACT） | 點雲與類別文字 prototype 比相似度，以類別 label 算 top-k accuracy。OpenShape 的 LVIS reader 使用 1,156 類。 | 類別分類的 top-1/top-5 不等於找回指定 UID 的 R@1/R@5，不能直接搬來填七格檢索表。 |
| OpenShape 檢索段（UPSTREAM FACT） | 文字、圖片或點雲 query 對 shape embedding 做 cosine kNN；還展示兩個 shape query，以兩個相似度的最小值排名。 | 原文有 P→P 與組合輸入示例，但不是 MetaFind 的七種模態、mean pooling、固定正解 UID 評估協定。 |
| Uni3DL（UPSTREAM FACT） | Text2Shape 的 text→shape R@1/R@5 為 5.8/19.7；另用 Cap3D 做預訓練與任務消融。 | 舊表把 5.8/19.7 再貼給 Cap3D，是錯誤歸屬；Cap3D Table 5 報的是 T2S R@1 5.5、S2T R@1 8.0，不能混成 R@1/R@5。 |
| OmniBind（UPSTREAM FACT） | Table 1 的 3D–Image retrieval 有 46,205 items，Full 的 R@1/R@5 為 46.55/69.92；另展示 embedding 加減的組合理解。 | 分類表的 46,832 與檢索表的 46,205 是不同任務分母。檢索方向、配對 UID 與 query 視角不能只憑表頭自行補出。 |
| MetaFind（PAPER FACT） | 作者另加 mean pooling 適配基線；自家 gallery 使用 T/I/P，query 評估七種模態組合，報 R@1/R@5。 | 不能假設基線原始論文的 gallery 配置就是 MetaFind 適配後的配置；作者未完整指定的部分仍為 UNKNOWN。 |

來源：

- ULIP-2 官方分類入口：[main.py](https://github.com/salesforce/ULIP/blob/main/main.py)；本機 `/home/kyzen/upstream/ULIP/main.py`。
- OpenShape：[檢索原文](paper/openshape_source/sections/experiments.tex)，Multi-modal 3D Shape Retrieval 段；[官方分類程式](https://github.com/colin97/OpenShape_code/blob/master/src/train.py)，`test_objaverse_lvis`。本機 `/home/kyzen/upstream/OpenShape_code/src/train.py`。
- Uni3D：[官方評估入口](https://github.com/baaivision/Uni3D/blob/main/main.py)。
- Uni3DL：[原文 §4.1、Table 3、Table 5](https://arxiv.org/html/2312.03026v1)。
- OmniBind：[原文 §4、Table 1–2、§4.3](https://arxiv.org/html/2407.11895v1)。
- MetaFind：[§3.1–3.2](paper/metafind_source/3experiments.tex)、[§2.2–2.4](paper/metafind_source/2methdology.tex)。

Text2Shape 系的描述→形狀 retrieval 也可作為自訂評估參考，但需逐篇固定 query、候選集、配對正解與 scoring。不能把不同資料集上的 13～17% 當作 MetaFind text-only 應落入的範圍。本輪沒有重跑這些上游模型。

## 撤回的舊推論

1. **「整個領域 gallery 只有形狀、沒有 P 查 P」：撤回。** OpenShape 原文已有 point-cloud query，MetaFind 自家 gallery 明確含三模態。不同研究的任務需分別陳述。
2. **「沒有任何組合 query 先例」：撤回。** OpenShape 有雙 shape input，OmniBind 有 embedding arithmetic；這些示例仍不等於 MetaFind 的七欄 benchmark。
3. **「因為 text-only 數字接近 Text2Shape，作者一定用了另一份描述」：撤回。** 候選數、模型、資料分布和正解定義不同；數字接近不能辨識 query 來源。
4. **「image-only 很低，證明作者 query 不是標準渲染圖或影像塔較弱」：撤回。** 這些只能是待檢驗假說；尚未有控制其他條件的證據。
5. **「11 個視角代表 11 份可輪流當 query 的 caption」：撤回。** MetaFind 說明 11 views 與結構化標註，沒有定義逐視角 caption 數量與 query/gallery caption 分配。
6. **「同物件模態融合必定增加準確率」：撤回。** 同 UID 不保證向量融合後排名單調；需對固定模型、觀測與候選集實測。

## 本專案採用的可評估範圍

[自訂七模態評估](CUSTOM_TABLE1_EVALUATION.md) 明確凍結 UID pool、觀測、checkpoint 與 scoring，分別比較 canonical 觀測和同物件的另一份觀測，並包含 mean baseline、Stage 1 和可選 Stage 2 head。這是 **IMPLEMENTATION CHOICE**，能比較本專案的方法，不能宣稱已找回作者的原始 Table 1 協定。

舊 P5、DL-098 等量測仍留在原始 JSON 與歷史報告；本次沒有更改那些分數。其效力限於當時測過的配置，不能拿來排除所有其他觀測或架構。
