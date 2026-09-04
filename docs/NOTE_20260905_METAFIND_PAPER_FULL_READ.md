# MetaFind 論文逐字讀完 —— 筆記（2026-09-05 05:5x）

Kyzen：「你給我去看論文 不准使用腳本 做成筆記」。這份是**用 Read 工具逐檔逐行讀** `docs/paper/metafind_source/` 的
`neurips_2025.tex`（主檔：摘要、導論、總結、致謝）、`2methdology.tex`、`3experiments.tex`、`appendix.tex`、`4backgound.tex`（相關工作）
之後寫的。沒有 grep、沒有正規表達式、沒有腳本。引文為原文逐字；我的話另起標「→」。

---

## 0. 讀完最重要的一句（之前所有分析都沒有正面對待它）

§3.2 Retrieval Performance on Objaverse-LVIS，原文：

> "Notably, since other models do not adopt a dual-tower design, their "PC only" performance reflects retrieval using identical embeddings for both query and gallery, leading to inflated accuracy. **In contrast, our dual-tower framework introduces more cross-modality retrieval, which results in lower accuracy under the "PC only".** Nevertheless, MetaFind demonstrates stronger performance under partial modality conditions, highlighting its capability in multimodal fusion."

→ 作者**自己解釋**了 pc 75.1 為什麼比基線的 98 低：基線 query 和 gallery 是**同一條向量**（同一朵雲、同一個編碼器）；MetaFind 是雙塔，query 塔（只給點雲、其他兩格是 mask）出來的向量，對 gallery 塔（三模態齊全）出來的向量，作者稱之為「cross-modality retrieval」，所以低。
→ 這句話的含意：**論文的 query 點雲就是資產自己那朵**（跟基線同一份），pc 低不是因為 query 觀測換了，而是因為他們的兩座塔**沒有**把「只有點雲的 query」和「三模態的 gallery」對到同一條向量。
→ 我們的兩座塔**對到了**（pc 98），所以我們的 Fusion 學到的解跟他們的不一樣。這跟我昨晚「query 觀測不是自己那份」的推論**相反**。昨晚的推論來自基線列 text 0.1／T+I 0；作者這句話是針對 pc 格的直接說明。兩者要一起看（見 §6）。

---

## 1. 摘要與導論（`neurips_2025.tex`）

- 兩個核心問題：(i) 檢索忽略空間、語意、風格限制；(ii) 沒有專為 3D 資產檢索設計的標準範式，現有方法靠通用 3D 形狀表示模型。
- 「supports arbitrary combinations of text, image, and 3D modalities as queries」。
- ESSGNN：plug-and-play、等變、「captures spatial relationships and object appearance features」。
- 迭代建構場景。
- 導論第 3 段（逐字要點）：
  - 「MetaFind builds upon ULIP2 [xue2024ulip]」。
  - 「We adopt a dual-encoder architecture [DPR]」。
  - 「**we annotate 48K 3D assets from the Objaverse-LVIS subset, each rendered from 11 views and processed with GPT-4o to generate structured text descriptions**」。
  - ESSGNN 訓在 ProcTHOR（>10,000 houses）。
  - 「two-stage training: (1) pretraining on object-level data for cross-modal grounding and (2) fine-tuning on room-level scenes」。
- Figure 1 圖說（逐字要點）：「both the user query and candidate assets are encoded using the ULIP-2 backbone」；「On the asset side, **each 3D asset in the repository is pre-encoded independently by ULIP-2 into a fixed vector**」；「the similarity between the layout-aware query embedding and the precomputed asset embeddings is computed, and the top-matching asset is selected」。
- 總結段：「asset annotations rely on GPT-4o, which can introduce language bias, hallucinations, and occasional mislabeling」。
- 致謝：NVIDIA Academic Grant、雲端算力。

## 2. 方法（`2methdology.tex`）

**2.1 Task Definition**
- Q = {q_text, q_img, q_pc, q_layout}；A* = argmax_{A∈𝒜} sim(f_query(Q), f_gallery(A))；𝒜 = 「pre-encoded asset database」。
- 「The task is challenging due to the multimodal nature of user queries, partial modality absence, and the necessity for accurate layout awareness」。

**2.2 Method Overview**
- 「both leveraging the ULIP-2 embedding backbone」。
- 「The gallery encoder precomputes embeddings for assets using three available modalities, which are then stored for efficient retrieval.」
- 「each available modality is independently encoded using the ULIP-2 backbone, and these modality embeddings are subsequently integrated via a fusion layer, such as mean pooling, an MLP, or a Transformer-based module」。
- 訓練兩階段：「First, we train the query and gallery encoders to learn fundamental multimodal embedding alignment without spatial context. Subsequently, we fine-tune the query encoder—particularly the fusion module and the layout encoder—using layout-aware room-level datasets. This fine-tuning stage employs **adaptive freezing strategies, selectively freezing components like the gallery encoder**」。

**2.3 Data Preparation**
- Objaverse-LVIS「approximately 48,000 distinct 3D assets」；「Each asset is rendered from **11 orthogonal viewpoints** and annotated using **GPT-4o**」；屬性：「object category, size dimensions, materials, and placement constraints」。
- Figure 2 圖說：「rendered from multiple orthogonal views and passed through a VLM」。
- ProcTHOR：>10,000 houses、>3,000 unique assets；每間房「precise spatial coordinates and comprehensive semantic metadata for each asset」。
- 場景圖兩種邊：「(i) physical-relation edges ... (e.g., "cup on table"); (ii) semantic-relation edges ... (e.g., "microscope–lab bench"), obtained by prompting an LLM on object pairs」。

**2.4 Dual-Tower Architecture and Fusion Design**
- 第一句：「**While prior works typically align 3D encoders to a fixed CLIP embedding space by freezing pretrained text and image encoders, our MetaFind framework adopts a more flexible dual-tower design.** It enables context-aware, multi-modal queries by **training a dedicated query encoder** that fuses arbitrary modality subsets」。
  → 這句是「對比凍結 CLIP 的做法」。它沒說 MetaFind 解凍 CLIP，但語氣是把「凍結 CLIP」放在對立面。CLIP 凍不凍：**論文沒明說**。
- 「separate encoders for the query and gallery. Each tower leverages ULIP-2 to independently encode available modalities」。
- Fusion 候選：「mean pooling, MLP, masked MLP, gated fusion, or Transformer-based fusion」。
- 「**The gallery encoder is modality-complete and frozen after pretraining**, while the query encoder remains flexible: It accepts any subset of modalities and can be augmented with a layout-aware vector.」

**2.5 ESSGNN**
- 動機：GAT 對全域平移、縮放敏感；改用 EGNN（藥物設計）。
- h_i^(0) = Concat(x_i, t_i)；x_i ∈ ℝ³，t_i ∈ ℝ^d「text-derived feature」。
- 邊：空間邊（adjacency、support）；語意邊 = LLM 關係句 → 「frozen text encoder (e.g., CLIP or BERT)」→ e_ij。
- EGCL：h_i^{l+1} = h_i^l + Σ_j f_h(d_ij, h_i, h_j, e_ij)；x_i^{l+1} = x_i^l + Σ_j (x_i − x_j)·f_x(d_ij, h_i^{l+1}, h_j^{l+1}, e_ij)；f_h: ℝ^{2d+1+e}→ℝ^d，f_x: ℝ^{2d+1+e}→ℝ³（**f_x 值域寫 ℝ³**；附錄證明裡 φ_x 是純量權重——兩處不一致，D0-009 早已記）；d_ij = ‖x_i − x_j‖₂（正文）vs ‖·‖²（附錄）。
- e_layout = Pooling({h_i^(L)})；L、Pooling 種類未給。
- SE(3) 等變，附錄證明。

**2.6 Training Strategy**
- Stage 1（逐字）：「both query and gallery encoders are trained on large-scale object-level data from Objaverse-LVIS, where each asset has full modality inputs (text, images, and point clouds). We introduce stochastic modality masking to simulate partial-modality queries: each modality in the query has a 30% probability of being independently masked. Rather than zero-padding, we apply masked embeddings to ensure flexibility and prevent model degradation. The goal is to align all available modality combinations into a shared embedding space. The gallery encoder is trained to be modality-complete, and both towers share the contrastive retrieval objective」＋ L_pre（單向 query→gallery，負例 A' ∈ 𝓑 = gallery batch，τ）。
- Stage 2（逐字要點）：e_query = Fusion(e_text, e_img, e_pc) + λ·e_layout，λ 可學純量，「residual design」；「stochastic scene dropout ... omitted in 30% of batches」；「Only the query-side fusion layer and the ESSGNN module are updated during this stage; the gallery encoder is frozen」；雙向損失 L = ½(L^{q2g} + L^{g2q})，𝓑「batch of negatives」。

**2.7 Inference and Iterative Composition**
- 「all gallery asset embeddings are precomputed and cached」。
- Algorithm 1：每放一件更新場景圖，重算 e_layout（寫成 EGNN(G)）。
- 效率段落最後一句：「we have clarified it in the revision」→ 這是審稿回覆殘留，說明 arXiv v1 已是修訂版。

## 3. 實驗（`3experiments.tex`）

**3.1 Setup**
- Datasets：Objaverse-LVIS 48K；ProcTHOR-10K >10,000 houses、>3,000 assets；「In both datasets, we allocate 80% of the data for training and reserve the remaining 20% for testing.」
- Baselines：ULIP、OpenShape、SCA3D、Uni3DL、Uni3D、OmniBind（Base／Large／Full）。「we limit our baselines to pre-trained single-tower encoders that support at least one of the three modalities ... we extend each baseline by adding a simple mean pooling layer to aggregate available modalities, and use these fused embeddings to retrieve from a pre-encoded gallery. For completeness, we also include our own dual-tower model with a mean fusion layer but without layout context as a direct ablation baseline. **The temperature is 0.5 for all experiments.**」
- Metrics：「top-k retrieval accuracy (R@1, R@5)」；場景：GPT-4o 評分 + 人類偏好。

**3.2 Retrieval Performance on Objaverse-LVIS**
- 「48K high-quality 3D assets with structured textual descriptions and multi-view image renders」。
- 七種條件。
- 「MetaFind without ESSGNN outperforms all baseline models across different settings.」
- **§0 那句**（pc only 為何低）。
- ESSGNN 後掉分的解釋：Stage 2 在 ProcTHOR（layout-rich、不同資產分布）微調 fusion → 「feature-attribution mismatch」；「A practical mitigation is to maintain two fusion heads ... Using the Stage-1 head reproduces the "w/o ESSGNN" numbers (omitted for brevity). In our reported results, we instead explore a single shared head by **freezing both encoders in Stage-2, updating only ESSGNN and the fusion**, and applying stochastic scene dropout (30%)」。

**Table 1（R@1 / R@5）**，逐字抄：

| Method | Text | Image | PC | T+I | T+PC | I+PC | T+I+PC |
|---|---|---|---|---|---|---|---|
| ULIP | 0.1 / 0.9 | 0.1 / 1.3 | 97.9 / 99.4 | 0 / 0.3 | 33.9 / 58 | 22.6 / 41.6 | 6.4 / 15.9 |
| OpenShape | 0.6 / 1.7 | 0.3 / 1.1 | 98.4 / 99.7 | 0 / 0.5 | 35.1 / 61.4 | 25.0 / 44.3 | 7.0 / 17.2 |
| SCA3D | 6.9 / 10.4 | – | 98.1 / 99.3 | – | 39.7 / 65.2 | – | – |
| Uni3DL | 4.5 / 9.2 | – | 98.5 / 99.8 | – | 37.4 / 63.9 | – | – |
| Uni3D | 1.7 / 3.9 | 1.2 / 2.5 | 98.3 / 99.4 | 0.5 / 1.1 | 36.3 / 63.6 | 26.1 / 44.8 | 8.2 / 19.1 |
| OmniBind (Base) | 1.2 / 2.8 | 0.6 / 1.4 | 98.3 / 99.6 | 0 / 0.4 | 34.0 / 55.9 | 21.5 / 38.7 | 5.5 / 13.8 |
| OmniBind (Large) | 2.7 / 4.0 | 0.9 / 1.8 | 98.2 / 99.3 | 0.1 / 0.4 | 35.2 / 56.7 | 23.4 / 40.9 | 6.0 / 16.7 |
| OmniBind (Full) | 5.3 / 11.7 | 2.3 / 3.5 | 99.0 / 99.7 | 0.5 / 1.2 | 37.5 / 60.8 | 27.5 / 46.4 | 11.9 / 23.4 |
| **MetaFind w/o ESSGNN** | **13.8 / 23.1** | **11.7 / 19.2** | **75.1 / 78.0** | **17.2 / 21.8** | **44.5 / 71.3** | **45.8 / 73.1** | **51.7 / 76.5** |
| MetaFind w/ ESSGNN | 11.3 / 21.5 | 10.5 / 15.9 | 63.2 / 66.5 | 15.9 / 20.3 | 41.2 / 68.8 | 42.0 / 70.4 | 48.2 / 74.9 |

→ 觀察：MetaFind pc 的 R@5 只有 78.0（R@1 75.1）——R@1 到 R@5 幾乎不漲。基線 pc 是 98→99.5。其他 MetaFind 格 R@5 漲很多（T+PC 44.5→71.3）。**pc 格「R@5 ≈ R@1」是很怪的形狀**：代表有約 22% 的 query，它自己那件連前五名都排不進——不是差一點點，是完全找不到。這比較像那 22% 的 query 點雲跟 gallery 不對應（例如另一份雲、另一種取樣、或資料對錯），或者 gallery 那格向量被別的模態帶走了。純「兩塔沒對齊」通常會讓 R@5 明顯高於 R@1。**這是一條新線索，之前沒注意過。**

**3.3 Scene-Level**
- I-Design 管線，原本用 OpenShape 檢索；「No retrieval accuracy」；四維度 1–5 分；GPT-4o + 五位專家；200 個隨機場景；GPT-4o 看「scene layouts and rendered views」。
- Table 2：ULIP 2.70–3.02；OpenShape 2.95–3.28；w/o ESSGNN 3.22–3.55；w/ ESSGNN 4.04–4.25。圖說寫「MetaFind (with GSSNN)」（筆誤）。

**3.4 Ablation（Table 3，Text Only R@1 / GPT-4o 美感 / 場景一致性）**
- Full (bidirectional, iterative, ESSGNN) 11.4 / 4.1 / 4.2；w/o iterative 11.3；w/o Layout Context **13.5**；GAT 11.0；Fusion=Mean 9.4；MLPs 9.9；Dropout 10% 7.3；50% 13.2；Train fuser only 8.7；zero padding 10.5。
- 文字：「six dimensions: layout encoding, modality fusion strategies, modality dropout robustness, fusion granularity, gallery encoder flexibility, and missing modality handling」（**第五項 gallery encoder flexibility 在表裡沒有對應列**）。「MLP and the final selected Transformer outperform others」；「a 30% rate strikes the best balance」；「training only the fusion module in the query encoder improves efficiency, full encoder fine-tuning yields better performance by allowing earlier layers to adapt to modality-aware supervision」。
- → Table 3 的 Full 是**w/ ESSGNN** 的模型（11.4 ≈ Table 1 w/ ESSGNN text 11.3）；「w/o Layout Context 13.5」≈ Table 1 w/o ESSGNN text 13.8。所以消融各列的基準是 11.4，不是 13.8（昨天我配錯，DL-097 已改）。
- → Dropout 50% = 13.2 **高於** Full 11.4，作者卻說 30% 最好（「higher rates introduce instability」）。表和文字方向不一致。

## 4. 附錄（`appendix.tex`、`4backgound.tex`）

- 廣泛影響段。
- 相關工作（4backgound）：場景生成兩派；MetaSpatial（同作者）；3D 檢索：PointCLIP、CLIP-Forge、ULIP、OpenShape、SCA3D、COM3D、OmniBind；「Our model supports free-form modality combinations and is robust to missing inputs through stochastic masking」。
- 等變證明：假設 h⁰ 對 SE(3) 不變、e_ij 只來自文字；m_ij = φ_e(h_i, h_j, ‖x_i − x_j‖², e_ij)；x 更新 x_i + Σ_{j≠i}(x_i − x_j)·φ_x(m_ij)；h 更新 h_i + Σ φ_h(m_ij)。→ 這裡 φ_x 是對 m_ij 的函數（純量），正文卻寫 f_x → ℝ³；正文距離無平方，附錄有平方。
- 實驗分析：兩間房的定性比較。

## 5. 論文沒寫的（逐字讀完後的完整清單）

資料：切分 seed／方式；11 視角的相機參數與背景；GPT-4o prompt 與輸出格式；ProcTHOR 場景圖細節；語意邊用哪個 LLM 與文字編碼器；t_i 用哪個編碼器。
模型：Fusion Transformer 的層數／頭數／讀出方式；mask embedding 是否可學；兩座塔的參數共用；CLIP 塔凍不凍；Point-BERT 起點（釋出權重或重訓）；ESSGNN 層數、Pooling、隱藏維、λ 初值。
訓練：優化器、lr、batch、epoch、排程、seed、選 checkpoint 規則（Stage 1 與 Stage 2 都沒有）；Stage 2 的正例定義與 query 構造；Stage 2 gallery 是哪一池。
評估：gallery 是 20% 還是 48K；query 的文字／影像具體是哪一份（見 §6）；tie 怎麼算；是否多 seed。

## 6. 對「Table 1 為什麼對不上」的重新整理（讀完之後）

兩條線索，方向不同：

**線索 A（§0 那句，作者直說）**：pc 75.1 低是因為雙塔「cross-modality」——query 塔（只點雲）對 gallery 塔（三模態）。這暗示 query 點雲**就是**自己那朵，而作者的兩座塔沒有把兩邊對成同一條向量。我們的對成了（98）。差別在**訓練結果**，不在資料。
- 可能原因（INFERENCE）：他們 τ=0.5 下訓得少／Fusion 沒學到「有點雲就抄點雲」；gallery 塔「frozen after pretraining」；或 mask embedding 的處理讓 pc-only query 被稀釋。都無法從論文證實。
- 新線索：MetaFind pc 的 R@5 78.0 只比 R@1 75.1 高 2.9——約 22% 的 query 連前五都進不去。這種「找不到就完全找不到」的形狀，比「對得不夠準」更像**資料層的不對應**（那 22% 的 query 點雲跟 gallery 的不是同一份，或標籤對錯）。

**線索 B（基線列）**：ULIP 基線 text 0.1、image 0.1、T+I 0。釋出的 ULIP-2 在我們的資料上 text→pc 是 17.5、image→pc 70.4。0.1 接近隨機。這說明論文基線的 query 文字／影像**跟 gallery 幾乎無關**——若是同一資產的描述與渲染圖，不可能只有 0.1。所以基線那邊的 query 文字／影像來源跟我們不同（或有 bug）。

→ A 與 B 合起來：論文的 query 點雲＝自己那朵（A），但 query 文字／影像很可能不是我們這種「同資產、同 CLIP、同一條向量」的東西（B）。這跟昨晚的「換件」實驗（文字＋圖換同類別別件、點雲留自己：full 50.4 vs 51.7）一致——**點雲留自己、文字圖換弱的**。差別在論文的 text 13.8 高於我們換件的 5.3：論文的文字還帶一點個體資訊。
→ 但 A 也說 pc-only 本身應該落到 75，而換件實驗的 pc 是 86.1（P1，val）。所以就算 query 文字／圖換了，我們的 pc 格還是比論文高 11 點，這 11 點是**塔對齊程度**的差異（或那 22% 的資料不對應），資料層解釋不了。

**結論**：Table 1 的差距不是一個原因。至少兩層：(1) query 文字／影像的來源（論文沒寫，B 線索指向「弱於自己那份」）；(2) pc-only 的塔對齊程度／或資料對應（A 線索與 R@5 形狀）。第 (1) 層我們有實驗能逼近；第 (2) 層目前沒有任何可驗證的抓手，論文也沒給。

## 7. 這次讀到、之前寫錯或漏掉的

- Table 3 消融基準是 11.4（w/ ESSGNN），不是 13.8 —— 昨晚配錯，DL-097 已改；本次逐字確認。
- 導論明寫「rendered from 11 views」——不只 §2.3；我們 12 視角是 DEVIATION（已記）。
- §2.4 首句把「凍結 CLIP」放在對立面；CLIP 是否凍結我一直當成理所當然的凍結，論文其實**留白且語氣偏向不凍**。要列為 P0 未決（GPT 也提）。
- §3.2 作者對 pc 75.1 的解釋（§0）—— 兩週來沒有一份文件正面引用過這句。
- pc 格 R@5 ≈ R@1 的形狀 —— 今天第一次注意到。
- Table 3 的 Dropout 50% (13.2) > Full (11.4)，文字卻說 30% 最好 —— 表文不一致。
- 「gallery encoder flexibility」列在六個消融維度裡，表中沒有對應列。
- 「we have clarified it in the revision」—— v1 已是修訂稿。
