# 資料前處理：以論文為主的重做計畫（2026-09-02）

Kyzen：「一切以論文為主」。這份文件只認 MetaFind 論文原文；ULIP-2 與 CAMERA 的做法只在論文沒寫時當候選，不當標準。
掃描結果在 `output/look/data_scan_against_paper.json`（`tools/probes/data_scan_against_paper.py`）。

## 一、論文說了什麼、我們現在是什麼

| 項目 | 論文（出處） | 我們現在 | 判定 |
|---|---|---|---|
| 資產數 | 「approximately 48,000」（`2methdology.tex:28`）；「48K unique」（`3experiments.tex:8`） | 官方 LVIS 名單 46,052；語料 45,692 | 到不了 48K，維持 45,692 並明寫 |
| 視角 | 「rendered from **11 orthogonal viewpoints**」 | **12**（Blender、三圈×4、512px、透視、黑底） | **要改成 11**。哪 11 台相機論文沒寫 |
| 標註模型 | GPT-4o | gemma-4-12B-it | 偏離，Kyzen 已裁 |
| 標註內容 | category、size dimensions、materials、placement constraints | 四欄 100% 齊 | 一致 |
| 文字進編碼器 | 「rich textual descriptions」 | 一句結構化句（≤75 token） | 一致（格式論文沒寫） |
| 點雲 | 沒寫 | 10,000 點 xyz+rgb（ULIP-2） | 維持 |
| 切分 | 兩資料集 80/20 | 36,554/9,138 互斥；房子 9,600/2,400 | 一致 |
| ProcTHOR 房子 | >10,000 | 12,000（train 10,000 + val 1,000 + test 1,000） | 一致 |
| ProcTHOR 資產 | 「more than 3,000 unique assets」 | 原始 12,000 間房裡（含 children）只有 **1,467** 個 assetId；我們的圖 1,467，沒漏 | 3,000 是 AI2-THOR 目錄大小，不是房子裡出現的數量；維持 1,467 並明寫 |
| ProcTHOR 節點語意 | 「comprehensive semantic metadata」 | 原始資料只有類別名（93 個相異句） | 論文說的來源不存在；用渲染圖讓 gemma 描述（已核可，未跑） |
| 邊 | 物理（adjacency、support）＋語意（LLM 句子→凍結文字編碼器） | 同；語意句 4,242 句（gemma）、1280 維 | 一致 |
| ProcTHOR 渲染 | 論文沒寫 | 11 張、224px、正交、白底（退役常數） | 論文無目標；要與 Objaverse 用同一協定（同一個圖片編碼器） |
| Stage 1 輸入 | 每件資產三模態齊全；query 30% 遮罩 | 同 | 一致；**重現線用同一筆紀錄（same_record）** |

## 二、要改的（依賴順序）

```
1  Objaverse 重渲 11 視角        Blender；46,024 件；上次全語料重渲花了數天
   相機組（論文沒寫，等 Kyzen 定）：
     甲  單圈 11 個等距方位角、仰角 20°、透視相機、512px、黑底
         （ULIP-2 的單圈慣例改 11 台；退役渲染器已定義過這個佈局 `ulip2_azimuth_orbit_11`）
     乙  同上但正交相機（把「orthogonal」讀成正交投影）
   我建議甲：編碼器在 ULIP-2 訓練時看的是透視渲染；「orthogonal viewpoints」更像在講視角組而非投影
2  重新編碼圖片向量（n06）        11 張 × 45,692 過 ViT-bigG；約 2～3 小時
3  重新標註（gemma，11 張）        80 小時；先 100 件 → 1,000 件 → test 子集試點；同一趟產 canonical / paraphrase / single-view / short
4  ProcTHOR 統一到同一協定        AI2-THOR 重渲 1,467 件（小時級）→ gemma 描述資產（2.5 h，已核可）→ 物件文字（修字後）
                                 → 語意邊（35 分）→ 節點向量 → Stage 2 索引（10 分）
5  query pack 重建（延伸軌用）     分鐘級
6  Stage 1                       先 10 輪確認 loss 會降、dev-val 贏過未訓練骨幹 → 目標 250 輪（ULIP-2）
```

**快速版（若不想花數天重渲）**：圖片維持 12 張（記為偏離），標註只餵 11 張（`--arm eleven_view`，80 h），其餘同上。差別只在「圖片模態有 12 張而非 11 張」。

## 三、不改、但要在報告裡寫清楚
- 45,692 ≠ 48K；ProcTHOR 1,467 ≠ 3,000（目錄 vs 出現）；GPT-4o → gemma；ProcTHOR 節點語意來源不存在。
- 重現線（R 軌）= 論文字面：兩座塔讀同一筆紀錄、query 遮罩 30%；分數會接近 100，與論文的 13.8/51.7 不一致，這個差異本身就是結果（原因無法從外部歸因）。
- 第二觀測（E1 軌）、場景替換（E2）是我們的延伸，不是論文。

## 四、CAMERA / ULIP-2 官方評估的定位（修正先前的用法）
只用來確認**儀器**：骨幹能重現官方零樣本數字（50.56/78.93）、評估器排名正確、點雲與渲染與官方等價。
它們**不是** MetaFind 的協定：CAMERA 是單模態 text→3D、單塔；MetaFind 是雙塔、融合多模態 query 對融合 gallery。
從今天起，數字的標尺只有論文的七條件 R@1/R@5（80/20，全池主報）。

## 五、Stage 2 一個要先知道的數字
ESSGNN 改成 7 層＋sum pooling 後，**未訓練**的 λ·e_layout 比 Fusion 輸出大 27 倍（平均，最大 40 倍；`output/look/stage2_smoke_seven_checks.json` 的 `layout_term_scale`），
step 0 的 loss 4.08 ≈ 亂猜 4.16。sum pooling 讓 e_layout 的大小跟房間物件數（10～100+）成正比。
論文只說 λ 是「learnable scalar」、沒給初值。選項：λ 初值 1/30 左右；或把 e_layout 正規化再加；或 pooling 改回 mean。**等 Kyzen／Codex 決定**，不要先偷偷改。
