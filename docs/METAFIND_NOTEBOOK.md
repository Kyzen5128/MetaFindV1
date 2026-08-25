# MetaFind 復現筆記（Master 自用說明書）

撰寫：Master（Claude），2026-08-25。
目的：把 MetaFind（arXiv 2510.04057）的架構、流程、訓練方法、參數、上游技術、參考文獻做一次全面整理，
作為我自己與所有角色的工作依據，並交 GPT 外部審查。

**閱讀基礎（全部逐字讀完的一手來源）：**

| 來源 | 本地路徑 | 行數 | 狀態 |
|---|---|---|---|
| MetaFind 論文源碼 | `docs/paper/metafind_source/` | 541 tex 行 | 逐字讀完（含全部圖說、附錄證明） |
| ULIP-1 論文 | `docs/paper/ulip1_source/main.tex` | 1,125 | 逐字讀完 |
| Point-BERT 論文 | `docs/paper/pointbert_source/Pointbert_arxiv.tex` | 624 | 逐字讀完 |
| OpenShape 論文 | `docs/paper/openshape_source/` | 672 | 逐字讀完（含 supplementary） |
| ProcTHOR 論文 | `docs/paper/procthor_source/` | 2,417 | 逐字讀完（supplement 略過純圖片排版行） |
| Flamingo 論文 | `docs/paper/flamingo_source/` | 3,593 | 方法/實驗/訓練全部讀完；datasheet／model card 行政段落僅掃過 |
| ULIP-2 論文 | `docs/paper/ulip2_source/` | — | 先前已逐字讀完（見 evidence 檔） |
| EGNN 論文 | `docs/paper/egnn_source/`（如在庫） | — | ESSGNN 的數學母體，關鍵式已核對 |
| 上游程式碼 | `/home/kyzen/upstream/{ULIP, OpenShape, egnn, IDesign}` | — | 關鍵路徑逐行核對過（引用附行號） |

每份 `*_source/` 都有 `SOURCE_MANIFEST.json`（sha256 指紋）。

**證據分級**（本文件每一條重要論述都掛標籤）：
`PAPER FACT`＝MetaFind 明文寫的｜`UPSTREAM FACT`＝上游論文/官方程式碼明文｜
`OBSERVED IMPL`＝我們 repo 現況｜`OBSERVED DATA`＝實際資料量測｜
`INFERENCE`＝推論｜`IMPL CHOICE`＝來源沒講、我們選的｜`DEVIATION`＝故意偏離｜`UNKNOWN`＝沒證據。

---

## 1. MetaFind 是什麼（一段話）

MetaFind 是一個**場景感知的 3D 資產檢索系統**：使用者給出任意組合的查詢
（文字 / 圖片 / 點雲，可再加上「目前房間已擺了什麼」的佈局資訊），
系統從一個預先編碼好的 3D 資產庫（gallery）裡找出最合適的資產，
一件一件擺進場景，逐步蓋出風格一致、空間合理的 3D 房間。
（PAPER FACT — abstract, `neurips_2025.tex:77`）

兩個核心零件：
1. **雙塔檢索（dual-tower）**：query encoder＋gallery encoder，骨幹都是 ULIP-2。（PAPER FACT `neurips_2025.tex:90`）
2. **ESSGNN**：SE(3) 等變的場景圖編碼器（EGNN 的語意邊擴充版），把「房間現況」壓成一個 layout 向量。（PAPER FACT `2methdology.tex:36-65`）

任務形式化（PAPER FACT `2methdology.tex:6-9`，Eq. 1）：

```
Q = {q_text, q_img, q_pc, q_layout}（每一項都可缺）
A* = argmax_{A∈𝒜} sim( f_query(Q), f_gallery(A) )
```

---

## 2. 整體架構

### 2.1 資料流（正向一遍）

```
【Gallery 塔（離線、一次算完）】
  每件資產：text 標註 + 多視角渲染圖 + 點雲
      → ULIP-2 各自編碼 → ModalityFusion（modality-complete）
      → 一支固定向量，存起來                       (PAPER FACT 2methdology.tex:14,34)

【Query 塔（線上）】
  使用者給的模態子集 → ULIP-2 各自編碼
      → ModalityFusion（缺的模態用 learned mask embedding 補）
      → Fusion(e_text, e_img, e_pc)
  房間場景圖 G →ESSGNN→ e_layout
  e_query = Fusion(...) + λ · e_layout              (PAPER FACT 2methdology.tex:85, Eq.)
  λ 是 learnable scalar                             (PAPER FACT 2methdology.tex:87)

【檢索】 cos 相似度 → top-1 資產 → 擺進場景 → 更新場景圖 → 下一件
                                                    (PAPER FACT Algorithm 1, 2methdology.tex:117-135)
```

留意：**被選出的資產也會進到下一輪的 query 側**——場景圖多了一個節點，
ESSGNN 重算 e_layout。圖上（MetaFind.drawio.png）綠色箭頭 Selected Assets → Query Encoder → ESSGNN 就是這個意思。
（PAPER FACT，圖＋Algorithm 1 第 6 行；這是先前吵過並確認的點）

已知圖的小錯：MetaFind.drawio.png 把 Text→I1..IK / Image→T1..TK 的標籤畫反了（OBSERVED，對照內文可判定為 typo）。

### 2.2 ULIP-2 骨幹（兩塔共用的底座）

- ULIP-2 = ULIP 框架 + 大規模自動標註。三模態編碼器：
  - **文字/圖片**：OpenCLIP **ViT-bigG-14**（凍結）。（UPSTREAM FACT ulip2 main.tex:609，verbatim 有 "freeze it during the pre-training"）
  - **點雲**：**Point-BERT**（唯一在訓練的編碼器）。（UPSTREAM FACT）
- Point-BERT 架構（UPSTREAM FACT Pointbert_arxiv.tex:121,141,216,594,597）：
  FPS 選 group 中心 → kNN 取鄰點 → 減中心座標 → mini-PointNet（2 層 MLP）作 patch embedding
  → 位置編碼 = MLP(中心座標) → 加 [CLS] → Transformer（depth 12, dim 384, heads 6, stochastic depth 0.1）
  → 全域特徵 = Concat(CLS, max-pool over tokens)。
- 我們的輸入規模引用 **ULIP-2 官方 yaml**（不是 Point-BERT 論文）：
  `npoints 10000 / num_group 512 / group_size 32`，帶顏色點雲。
  （UPSTREAM FACT `upstream/ULIP/models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml`；
  我們 `metafind/models/ulip_backbone.py:92` N_POINTS=10000 一致，OBSERVED IMPL）
- 為什麼凍 CLIP：ULIP-1 作者明文「更新 CLIP 編碼器會因資料量小而 catastrophic forgetting」
  （UPSTREAM FACT ulip1 main.tex:286），且其源碼藏有未發表 ablation：解凍後 zero-shot top-1 由 37.1 → **0.0**
  （UPSTREAM FACT ulip1 main.tex:1078-1092，註解掉的表格）。
  OpenShape 也獨立做過：解凍 finetune CLIP text encoder 一個 epoch，無提升且傷泛化，故全程凍結
  （UPSTREAM FACT openshape supplementary.tex:192-194）。Flamingo 同樣：finetune 凍結 LM 掉 8.0%（UPSTREAM FACT flamingo content.tex:374-378）。
  → 我們 Stage 1 凍 CLIP、只訓 PointBERT＋fusion，是 **DL-032 USER 決策**，證據鏈完整。

### 2.3 雙塔怎麼共用權重（我們的實作）

- MetaFind 說「separate encoders for the query and gallery」（PAPER FACT `2methdology.tex:34`）
  但沒說兩塔是否共享 ULIP-2 權重（UNKNOWN — U-16 的殘留部分）。
- 我們目前鎖定：**`shared_backbone_separate_fusion`** — 一份 ULIP-2 骨幹、兩份獨立 ModalityFusion。
  （IMPL CHOICE，`metafind/data/splits.py` DEFAULT_TOWER_SHARING；`dual_tower.py` 的 fully_shared 模式會 rebind fusion）
  U-16「PointBERT 一份還是兩份」仍列在待 Kyzen 的清單上。

### 2.4 ModalityFusion 與缺模態處理

- 論文列了五種 fusion 候選：mean pooling / MLP / masked MLP / gated / **Transformer**（PAPER FACT `2methdology.tex:34`），
  消融文字說最後選 **Transformer**（PAPER FACT `3experiments.tex:143`「the final selected Transformer」）。
  但 Table 3 的 Full 用哪一種沒有明說行號對應（U-13；由 n09 解析，`resolve_stage1.py:288` 附註）。
- 缺模態：**不是补零**，用 **masked embeddings**（PAPER FACT `2methdology.tex:75`；Table 3 顯示补零 R@1 10.5 vs full 11.4）。
  我們的實作：per-modality learnable mask token，`nn.Parameter(zeros(m,d))`、std 0.02 初始化
  （OBSERVED IMPL `metafind/models/fusion.py`；初始化細節為 IMPL CHOICE）。

### 2.5 ESSGNN（數學核心）

場景圖 G=(V,E)：節點＝已擺物件，帶 3D 座標 x_i∈R³ 與文字特徵 t_i∈R^d；
節點初始化 h_i⁰ = Concat(x_i, t_i)。（PAPER FACT `2methdology.tex:42-45`）

邊有兩種（PAPER FACT `2methdology.tex:28,47`）：
- **物理邊**：空間依賴（"cup on table"、adjacency、support）。
- **語意邊**：用 LLM 對物件對生成關係句，再用**凍結的文字編碼器**（CLIP 或 BERT）編成 e_ij。

訊息傳遞（改版 EGCL，PAPER FACT `2methdology.tex:50-53`）：

```
h_i^{l+1} = h_i^l + Σ_{j∈N(i)} f_h(d_ij, h_i, h_j, e_ij; θ_h)
x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i^l − x_j^l) · f_x(d_ij, h_i^{l+1}, h_j^{l+1}, e_ij; θ_x)
d_ij = ‖x_i − x_j‖₂；f_h, f_x 都是 MLP
```

L 層後 e_layout = Pooling({h_i^L})。（PAPER FACT `2methdology.tex:55-57`；Pooling 型式沒指定 → UNKNOWN/IMPL CHOICE）

SE(3) 等變性：appendix.tex:16-76 給了完整證明；關鍵前提是 **e_ij 不依賴座標**（只來自文字），
所以距離項旋轉平移不變 → 訊息不變 → 座標更新等變。
與原 EGNN 的差別＝把離散 edge feature 換成語意向量（PAPER FACT appendix.tex:20）。
注意論文自稱「separating spatial and semantic channels」（`neurips_2025.tex:100`）在方法章的落地就是 e_ij 進 f_h/f_x 的 message 內。

為什麼不用 GAT：對全域平移/縮放太敏感、在未正規化座標系不穩（PAPER FACT `2methdology.tex:40`）。

EGNN 出處：Satorras et al. 2021（arXiv 2102.09844），MetaFind 引用其藥物設計脈絡（PAPER FACT `2methdology.tex:42`）。
**EGNN 是參考架構**（Kyzen 定調），MetaFind 沒講的 ESSGNN 細節依 standing rule 回 EGNN 官方實作（`upstream/egnn`）。

---

## 3. 資料準備

### 3.1 物件層（Objaverse-LVIS，Stage 1 用）

- ~48,000 件 3D 資產（PAPER FACT `2methdology.tex:28`；Objaverse-LVIS 官方全集是 46,832 件/1,156 類 —
  UPSTREAM FACT openshape experiments.tex:7。MetaFind 的「48K」與 46,832 的差 → 未解釋，記 UNKNOWN；
  我們實際語料 46,052 件在跑 n05，OBSERVED DATA）。
- 每件資產 **11 個正交視角渲染** + **GPT-4o 結構化標註**
  （類別、尺寸、材質、擺放限制）（PAPER FACT `2methdology.tex:28`, `neurips_2025.tex:100`）。
- 我們的 DEVIATION（USER 決策）：**12 視角**（三圈仰角 × 每圈 4 個），Blender 渲染，512px、透視 35mm、RGBA
  （OBSERVED IMPL `metafind/data/render_blender.py:88`，`renders.py:27` 記載 11→12 的決策理由）。
  對照上游：OpenShape 每件 render **12 張彩圖，preset 均勻相機位**（UPSTREAM FACT openshape method.tex:77）；
  ULIP-1 是 30 RGB+30 depth、每步隨機取 1 張（UPSTREAM FACT ulip1 main.tex:236）。
- 切分：80% train / 20% test（PAPER FACT `3experiments.tex:8`）。

### 3.2 場景層（ProcTHOR-10K，Stage 2 用）

- ProcTHOR-10K：10,000 間程序生成的房子（另有 1,000 val + 1,000 test；UPSTREAM FACT procthor 04_analysis）。
  資產庫 1,633 件、108 類；房子以 JSON 描述，物件帶精確座標與語意 metadata
  （UPSTREAM FACT procthor 01_intro.tex:10, supp §Assets/Datasheet）。
- MetaFind 說用「curated collection of **more than 3,000** unique assets」（PAPER FACT `2methdology.tex:28`, `3experiments.tex:8`）
  — 與 ProcTHOR 官方 1,633 對不上。這是**已登記的衝突**（U-21 相關）：
  可能含 ProcTHOR 官方之外的擴充，論文沒解釋 → UNKNOWN，Table 2 的 gallery 範圍問題也懸在這裡。
- 從每個房間抽 **關係場景圖**：物理邊＋LLM 生成的語意邊（PAPER FACT `2methdology.tex:28`）。
- ProcTHOR 房子生成機制（讀完 supplement 的重點，供 ESSGNN 鏈開工用）：
  room spec 樹 → 邊界切割 → Lopes et al. 遞迴分房 → 門/窗/牆材質抽樣 →
  地板物件按 room weight r_w∈{0..3}、edge/corner/middle 註記擺放；
  **SAG（Semantic Asset Group）**：桌+椅這類共現群組用相對錨點放置（18 組，2,000 萬組合）；
  牆面物件（窗/畫/電視）、檯面小物件依經驗機率 spawn；材質/顏色/燈光隨機化；
  BFS 驗證器保證每房 ≥5 個可達點。（UPSTREAM FACT procthor 07_supp §House Generation 全節）
  → 這些先驗未來讀場景圖、判斷「什麼算合理擺放」時都用得上。

### 3.3 我們的資料管線（repo 節點對應，OBSERVED IMPL）

```
物件鏈： n02 取資產 → n03 點雲 → n04 渲染(Blender 12 views) → n05 GPT 標註(進行中)
        → n06 CLIP/文字/圖嵌入快取 → n09 Stage-1 訓練 → n10 gallery 48K 向量 → Table 1/3
場景鏈： n07 ProcTHOR 房子 → n07b 場景圖抽取 → n08 語意邊 → n09b/c ESSGNN → n11b Stage-2
        → n13 檢索 → n14 I-Design 管線 → n15/16 GPT-4o+人評 → n17 → Table 2
```
（架構圖 Artifact：MetaFind 復現全地圖，favicon 🔥）

---

## 4. 訓練方法

### 4.1 為什麼先 Stage 1 再 Stage 2

Stage 2 的 λ·e_layout 是**殘差**，掛在一個「已經對齊好的嵌入空間」上；
先把跨模態空間練穩（Stage 1），再讓 layout 資訊當加成而不破壞原空間
（PAPER FACT `2methdology.tex:87` "This residual design allows layout reasoning to enhance retrieval
without disrupting the original embedding space"）。

### 4.2 Stage 1 — Cross-Modal Alignment Pretraining

| 項目 | 內容 | 證據 |
|---|---|---|
| 資料 | Objaverse-LVIS 48K，全模態 | PAPER FACT 2methdology.tex:75 |
| 誰在訓練 | query＋gallery 兩塔都訓（"both query and gallery encoders are trained"） | PAPER FACT 2methdology.tex:75 |
| 訓練深度 | 論文沒說訓到多深；DL-032 定案：CLIP text/image 凍結，訓 PointBERT + pc_projection + 兩塔 Fusion | USER 決策 DL-032（依 UPSTREAM FACT ulip2:609） |
| 模態遮罩 | 每個 query 模態 **獨立 30% 機率被 mask**；用 masked embedding 不是补零 | PAPER FACT 2methdology.tex:75 |
| 損失 | **單向** InfoNCE：−log softmax(sim(f_q(Q), f_g(A))/τ)，負例=batch 內 gallery | PAPER FACT 2methdology.tex:76-79 (Eq.) |
| τ | **0.5、全實驗固定** | PAPER FACT 3experiments.tex:15 |
| 訓完 | gallery encoder 凍結，48K 資產向量預先算好存庫 | PAPER FACT 2methdology.tex:34,111 |

**與上游損失的關係**（重要對照）：
- ULIP-1：L = α·L(I,S)+β·L(I,P)+θ·L(P,S)，α=0、β=θ=1，每組**雙向**、τ **可學**
  （UPSTREAM FACT ulip1 main.tex:267-285）。
- OpenShape：4 項雙向（P↔T、P↔I）、τ 可學、g^T/g^I 可學線性投影、**凍結 CLIP 並預先快取文字/圖嵌入**
  （UPSTREAM FACT openshape method.tex:60-68）。
- MetaFind Stage 1 明文改成**單向 + 固定 τ=0.5** → 論文蓋過上游，照 MetaFind 做。
  Stage 2 的 ½(q2g+g2q) 才回到上游式的雙向形。
- OpenShape 的「凍結即快取」直接對應我們 n06 的 CLIP 嵌入快取（訓練時只有 PointBERT 是活的），
  也讓「梯度累積≠更多負例」的 InfoNCE 陷阱基本失效（負例數由實 batch 決定）。

### 4.3 Stage 2 — Layout-Aware Fine-Tuning

| 項目 | 內容 | 證據 |
|---|---|---|
| 資料 | ProcTHOR 房間 + 場景圖 | PAPER FACT 2methdology.tex:83 |
| 查詢構造 | e_query = Fusion(e_text, e_img, e_pc) + λ·e_layout | PAPER FACT 2methdology.tex:85 |
| λ | learnable scalar；**初值論文沒講 → 開放決策（見 §7）** | PAPER FACT :87 / UNKNOWN |
| 誰在訓練 | **兩個 encoder 都凍**，只訓 ESSGNN + fusion（+λ）；單一共用 fusion head | PAPER FACT 3experiments.tex:24 |
| （方法章表述） | 「只更新 query-side fusion 與 ESSGNN；gallery encoder 凍結」 | PAPER FACT 2methdology.tex:89（與上行一致） |
| scene dropout | 30% 的 batch 拿掉 e_layout，逼模型也能吃無 layout 查詢 | PAPER FACT 2methdology.tex:89, 3experiments.tex:24 |
| 損失 | **雙向**：L = ½(L^q2g + L^g2q)，τ=0.5 | PAPER FACT 2methdology.tex:91-102 |
| 正例配對 | 論文沒明說怎麼配 query↔asset；我們用 leave-one-out（從房間移除一件資產當目標，其餘做場景圖） | UNKNOWN → IMPL CHOICE（`stage2.py` encode_query，OBSERVED IMPL；GPT 指出與迭代推論有分布落差，已登記） |

Stage-1 頭 vs Stage-2 頭：論文自己說有「兩個 fusion head」的緩解方案（layout-free head / scene-aware head），
但**報告的數字用單一共用 head**（凍雙塔＋30% scene dropout），並承認留有 attribution drift 損失
（PAPER FACT `3experiments.tex:24` 全段——Table 1 w/ ESSGNN 掉分的官方解釋）。

### 4.4 超參數（分級列表——這節 GPT 最該審）

**MetaFind 論文明文（PAPER FACT）：**
- τ = 0.5，全實驗固定（3experiments.tex:15）
- 模態遮罩率 30%（2methdology.tex:75）；Table 3 掃過 10%/50%，30% 最佳
- scene dropout 30%（2methdology.tex:89）
- split 80/20（3experiments.tex:8）
- 評估指標 R@1/R@5（3experiments.tex:18）

**MetaFind 沒寫、由上游論文補（UPSTREAM FACT，依 standing rule 採用）：**
- lr 1e-3、batch 64、AdamW、（ULIP-1 還寫了 250 epochs）— ulip1 main.tex:367-370
- ULIP 官方 repo 補充：wd 0.1、betas (0.9,0.98)、eps 1e-8、warmup 1 epoch（lr_start 1e-6）、cosine
  — `upstream/ULIP/main.py`（注意：repo 的 default lr 是 3e-3，與論文 1e-3 不同；我們取論文值，衝突已登記）
- OpenShape 側寫（參考，不直接採用）：batch 200 單卡 A100、PointBERT-32.3M 用 lr 5e-4、指數 LR、~300 A100-hr
  — openshape supplementary.tex:190

**我們 repo 目前值（OBSERVED IMPL `resolve_stage1.py:237` DEFAULT_HYPERPARAMETERS；候選、尚未全數 USER 簽核）：**

```json
{ "optimizer":"adamw", "learning_rate":1e-3, "weight_decay":0.1, "scheduler":"cosine",
  "batch_size":64, "epochs":50, "p_mask":0.3,
  "init_temperature":0.5, "learnable_temperature":false, "seed":20260816 }
```

- `learnable_temperature:false` 是 **USER 批准的 IMPL CHOICE**（推論依據：論文把 f_h/f_x/λ 都叫 learnable、
  卻兩處把 τ 叫 hyperparameter，且「0.5 for all experiments」；resolve_stage1.py C-001 有完整論證）。
- `epochs:50` 在任何論文都找不到（**UNKNOWN**）。ULIP-1 寫 250（ShapeNet 規模），Objaverse 規模沒人寫。待定。
- seed 20260816 = IMPL CHOICE。

**Stage 2 超參數：論文全部沒寫（UNKNOWN）。** EGNN 官方實作與 ProcTHOR 都不提供對應（不同任務）。待議。

---

## 5. 推論與迭代構圖

- Gallery 嵌入全部預先算好、快取（PAPER FACT 2methdology.tex:111）。
- **Algorithm 1**（2methdology.tex:117-135）：逐件檢索
  `for i=1..N: e_layout←ESSGNN(G); 編碼 Q_i 各模態; e_query=Fusion+λ·e_layout; 取 argmax; 擺入; G←G∪{A*}`。
- 迭代 vs 一次抓完：迭代較慢但空間/風格一致性顯著較好；效率優先時可用區域分解——
  房間切語意區域、區域內序列檢索、區域間平行（PAPER FACT 2methdology.tex:113-115）。
- Table 3：w/o iterative 只掉一點（R@1 11.4→11.3、Aesthetic 4.1→4.0）——同一組權重換組合方式即可測
  （OBSERVED IMPL：我們的 no_iterative variant 正是 reuses_ckpt="full"）。

---

## 6. 評估協定

### 6.1 Table 1 — 物件層檢索（Objaverse-LVIS 20% 測試集）

- 七種查詢條件：T / I / PC / T+I / T+PC / I+PC / T+I+PC；指標 R@1/R@5（PAPER FACT 3experiments.tex:24 表）。
- Baselines（ULIP、OpenShape、SCA3D、Uni3DL、Uni3D、OmniBind）都是單塔模型，
  外加 mean pooling 湊成雙塔式比較；另附我們自己的 mean-fusion 無 layout 版當 ablation
  （PAPER FACT 3experiments.tex:15）。
- 重要判讀：單塔 baseline 的 "PC only" 是 query=gallery 同一支嵌入，98%+ 是灌水；
  MetaFind 雙塔 PC-only 75.1 反而是誠實跨模態數字（PAPER FACT 3experiments.tex:24 文字段）。
- **對照上游評估法**：ULIP/OpenShape 官方評估是 zero-shot 分類（1,156 類名做文字檢索，
  UPSTREAM FACT `upstream/ULIP/main.py:399-406`、openshape experiments.tex:7），
  **不是** MetaFind 的 asset-level R@k。MetaFind 的檢索協定是它自己的，不能拿上游數字直接對表。
- 上游可繼承的評估零件：**文字模板 ensemble**（64 templates 平均，ulip1 main.tex:244-250；OpenShape 沿用）。

### 6.2 Table 2 — 場景層品質（I-Design 管線）

- 把 MetaFind 插進 I-Design 場景生成管線當檢索器（原版用 OpenShape）；
  比較 w/ 與 w/o ESSGNN（PAPER FACT 3experiments.tex:55）。
- 四維度各 1–5 分：Aesthetic / Color&Material / Scene Coherence / Realism&Geometry；
  GPT-4o ＋ 5 位人類專家、200 個隨機場景、給 layout+渲染圖、分數取平均（PAPER FACT 3experiments.tex:55）。
- 我們的落地：`upstream/IDesign`（retrieve.py 是被替換的 OpenShape 檢索步；gpt_v_as_evaluator.py 是 n17 評測模板）。
- 未解：Table 2 的 gallery 是 1,439 件 ProcTHOR 資產還是 46K？（U-21 gap，UNKNOWN，待 Kyzen）

### 6.3 Table 3 — 消融（Text-only）

軸：layout 編碼（ESSGNN/GAT/無）、fusion（Mean/MLP/最終 Transformer）、
遮罩率（10/30/50%）、訓練粒度（train fuser only vs full）、补零 vs mask。
「Train fuser only 8.7」是**編碼器粒度消融**（fuser-only vs 連 PointBERT 一起訓），不是拿掉 ESSGNN——
這個判讀已與 GPT 核對一致。（PAPER FACT 3experiments.tex:94-113＋143）

---

## 7. 論文沒講、要 Kyzen 拍板或已拍板的清單

| # | 議題 | 狀態 |
|---|---|---|
| DL-032 | Stage 1 CLIP 凍不凍 → **凍**（照 ULIP-2） | ✅ USER 決策 2026-08-25 |
| τ 可學性 | 固定 0.5 | ✅ USER 批准 IMPL CHOICE |
| 11→12 視角 | Blender 12 views（三圈×4） | ✅ USER 決策（DEVIATION，已記錄） |
| tower sharing | shared_backbone_separate_fusion | ✅ 已鎖（U-16 殘留：PointBERT 一份/兩份） |
| U-14 | 訓練時圖片用 12 視角平均 vs 每步隨機 1 張（上游法） | ⏳ 待 Kyzen A/B（我推薦 B=隨機 1 張訓練＋gallery/eval 用平均，依 standing rule；OpenShape method.tex:77 verbatim "randomly sample one rendered image or thumbnail for each shape"） |
| λ 初值 | 論文沒講。GPT 主張 0.1（LayerScale 系）；我對案：Flamingo 最近似情境（新分支掛進凍結模型）用 **tanh(α), α=0** —「初始化時輸出與原模型完全一致」，且拿掉 0-init gating 掉 4.2% 還不穩（UPSTREAM FACT flamingo content.tex:187-189, 350-352）。ReZero 也是 0-init。 | ⏳ 與 GPT 收斂中（R1-R4 未回），收斂後給 Kyzen 選 |
| epochs | 任何論文都沒有 Objaverse 規模的 epoch 數 | ⏳ UNKNOWN，配 recipe 一起上呈 |
| Stage 2 正例配對 | leave-one-out 是 IMPL CHOICE，與迭代推論有分布落差 | ⏳ 已登記，待議 |
| Table 2 gallery 範圍 | 1,439 vs 46K；「3,000+ curated assets」與 ProcTHOR 官方 1,633 衝突 | ⏳ UNKNOWN（U-21） |
| 48K vs 46,832 | LVIS 官方數 vs 論文 48K | UNKNOWN（不擋工，語料以實際 46,052 為準） |
| Pooling 型式 | e_layout 的 Pooling 論文沒指定 | ⏳ 依 standing rule 先看 EGNN 官方（sum/mean），開 ESSGNN 鏈時定 |

**Standing rule（Kyzen 2026-08-25 訂）**：MetaFind 沒講的，預設照上游官方（ULIP-2/EGNN/OpenShape 的論文＋程式碼），
記 UPSTREAM FACT；上游也沒有才上呈。Blender 事件是反面教材：動手前必先讀上游怎麼做。

---

## 8. 參考技術總表（誰貢獻了什麼）

| 技術 | 在 MetaFind 的角色 | 我們的引用源 |
|---|---|---|
| ULIP-2 (arXiv 2305.08275) | 兩塔骨幹；凍 CLIP 訓 3D 的框架 | ulip2_source + `upstream/ULIP` |
| ULIP-1 (arXiv 2212.05171) | 框架母體：損失結構、凍結理由、lr/batch/optimizer 出處 | ulip1_source |
| Point-BERT (arXiv 2111.14819) | 3D 編碼器架構 | pointbert_source + vendored code |
| OpenCLIP ViT-bigG-14 | 文字/圖片編碼器（凍結） | ulip2 main.tex:609 |
| OpenShape (arXiv 2305.10764) | 12 視角渲染、嵌入快取、batch 構造參照；Table 1/2 baseline | openshape_source + `upstream/OpenShape` (src/train.py:106-109) |
| EGNN (arXiv 2102.09844) | ESSGNN 的數學母體（**參考架構**） | `upstream/egnn` |
| ProcTHOR (arXiv 2206.06994) | Stage 2 場景資料來源 | procthor_source |
| Objaverse(-LVIS) (arXiv 2212.08051) | Stage 1 資產庫 | 資料在庫 |
| GPT-4o | 資產標註（n05）＋場景評分（n15/n17） | PAPER FACT |
| I-Design (arXiv 2404.02838) | Table 2 場景生成管線 | `upstream/IDesign` |
| DPR (arXiv 2004.04906) | 雙塔檢索範式的概念出處 | PAPER FACT neurips_2025.tex:84 |
| Flamingo (arXiv 2204.14198) | λ 初值辯論的最近似先例（tanh(α), α=0） | flamingo_source |
| GAT / PointCLIP / SCA3D / Uni3D(L) / OmniBind | 對照組與 baseline | PAPER FACT |

---

## 9. Repo 落地對照（給角色查表用）

| 論文概念 | 檔案 |
|---|---|
| 雙塔 + λ | `metafind/models/dual_tower.py`（init_lambda 目前 1.0 — **佔位，λ 決策未定**，OBSERVED IMPL） |
| Fusion + mask token | `metafind/models/fusion.py` |
| 損失（單向/雙向、τ=0.5） | `metafind/models/losses.py`（PAPER_TAU） |
| ESSGNN | `metafind/models/essgnn.py`（鏈停工中，待 Kyzen 開） |
| Stage 1 解析器 | `metafind/models/resolve_stage1.py`（DEFAULT_HYPERPARAMETERS + Table 3 variants） |
| Stage 2 查詢構造 | `metafind/models/resolve_stage2.py`（leave-one-out） |
| 渲染 | `metafind/data/render_blender.py`（12 views） |
| 標註 | `metafind/data/annotate_run.py`（n05 進行中） |
| 嵌入快取 | `metafind/data/encode_text_image.py`（n06） |
| 上游 verbatim 證據 | `workflow/blocks/ULIP2/evidence/UPSTREAM_ULIP1_POINTBERT_VERBATIM.md` |
| 決策帳 | `workflow/DECISION_LEDGER.md`（DL-010, DL-028~032） |

---

## 10. 給審查者（GPT）的話

請重點檢查：
1. §4 的每一條證據標籤有沒有升級錯誤（尤其 PAPER FACT vs UPSTREAM FACT vs IMPL CHOICE）；
2. §7 清單有沒有漏掉論文的其他沉默點；
3. §4.2 損失方向性（Stage 1 單向）與 §4.3 凍結範圍（3experiments.tex:24）的解讀；
4. §6.1 「上游評估是分類不是檢索」的判定；
5. 任何你認為與一手來源矛盾的句子——請附你自己的出處（章節/行號），口說無憑。
