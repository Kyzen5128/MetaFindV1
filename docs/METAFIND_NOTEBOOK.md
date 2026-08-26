# MetaFind 復現筆記（Master 說明書・完整版）

撰寫：Master（Claude）。v2 2026-08-25 全面擴充；**v3 2026-08-26 依外部審查修正**。
**狀態：條件通過，尚未 final。** 外部審查提出 3 項 P0，我逐條核對一手來源後**全部確認成立**，已在本版修正（MF-14 錯標、20% test 早停污染、U-16 推論過度），另補 5 個新登記沉默點（§9.5）、修正 1 個事實錯誤（n05 標註模型不是 GPT-4o，是 gemma-4-12B-it，見 §2.3）。**v3.1 同日追補**：Kyzen 指出我沒照 standing rule 走完（MetaFind 找不到 → 查 ULIP-2 官方**程式碼** → 再問人），我上一輪只 grep 論文就宣告「無源」。回查 `upstream/ULIP` 後，epochs（250）、不早停、取 best checkpoint、驗證資料不得取自最終回報 split，**四條上游都有明確答案，已全部採用**；同時查出 lr 是上游自己打架（論文 1e-3 vs 官方腳本明傳 3e-3，§10 #11 新登記）。**v3.2 再追補**：Kyzen 質疑「真的找不到嗎」，回頭把上游全部翻完，結論是**我上一輪標的「上游答不了」也是錯的**。OpenShape 訓練迴圈（`src/train.py:190-201`）提供了不需要任何 held-out 資料的 checkpoint 選擇依據（in-batch contrastive accuracy）＋每 epoch 存 latest／定期快照，**§10 #3 因此撤回上呈**。同時查出兩個事實錯誤：**PointBERT 深度是 18 不是 12**（yaml 的 live 區塊＋checkpoint 實測 18 blocks），以及 lr 有四個候選、其中 5e-4 是唯一針對 PointBERT 這個量級（實測 32.5M）給的值。**只剩 §10 #11（lr）一題需要拍板**，因為那是上游彼此分歧、不是沉默。
**v3.3**：Kyzen 再指出「EGNN 也一樣」，確實如此。ESSGNN 的層數與 pooling 我同樣只讀論文公式就標成「我們自己選」，回查 EGNN 官方 repo 後發現：**層數 4 是 N-body 的值、QM9 是 7**；**pooling 官方 readout 是 sum 不是 mean**；而 MetaFind 引 EGNN 的脈絡明寫 drug design＝QM9。兩處建議改，見 §3.4 對照表。
流程教訓已寫成常駐規則 `.claude/rules/upstream-lookup.md`（四步查完才准上呈，含五次失敗紀錄）。
目的：MetaFind（arXiv 2510.04057, NeurIPS 2025 格式, 作者 Pan/Lu/Liu）的**每一個技術細節**——
架構、張量維度、公式、資料格式、訓練配方、評估協定、已知矛盾、每個開放決策與其預設方案——
一次寫清楚，讓實作規劃不再返工。這份文件交 GPT 外部審查。

---

## 0. 證據紀律（讀這份文件的規則）

每條重要論述掛標籤：

| 標籤 | 意思 |
|---|---|
| `PAPER` | MetaFind 原文明文（附 tex 檔行號） |
| `UPSTREAM` | 上游論文/官方程式碼明文（附出處） |
| `IMPL` | 我們 repo 已實作的現況（附檔案行號） |
| `DATA` | 實際資料/量測結果 |
| `INFER` | 推論（有論證、無明文） |
| `CHOICE` | 來源沒講、我們選定的實作選擇（附決策編號） |
| `DEV` | 故意偏離原文（附理由） |
| `UNKNOWN` | 沒證據，禁止假裝有 |

矛盾登記簿：`docs/audit/C_PAPER_CONTRADICTIONS.md`（C1–C8 矛盾、S1–S6 沉默）。
未知登記簿：U-xx 編號散見各 protocol 與本文件 §10。
決策帳：`workflow/DECISION_LEDGER.md`（DL-xxx）。

**一手來源（全部逐字讀完）**：metafind_source 541 行；ulip1 1,125；pointbert 624；
openshape 672（含 supp）；procthor 2,417（含 supp）；flamingo 方法/實驗/訓練全部；
ulip2 先前讀完。每份都有 `SOURCE_MANIFEST.json` sha256。
上游程式碼：`/home/kyzen/upstream/{ULIP, OpenShape, egnn, IDesign}`，關鍵行號逐一核對。

---

## 1. 任務定義與符號

（PAPER `2methdology.tex:6-9`, Eq. 1）

```
查詢  Q = {q_text, q_img, q_pc, q_layout}    每一項都可缺席
資產庫 𝒜（gallery），每件資產 A 有完整三模態
檢索  A* = argmax_{A∈𝒜} sim( f_query(Q), f_gallery(A) )
```

- `sim(·,·)`：論文只說 "the similarity function"（**S1 沉默**）。
  → CHOICE（U-24）：cosine similarity——ULIP/OpenShape/CLIP 全都用它；
  實作為 L2-normalize 後點積（IMPL `losses.py:166-170`）。protocol 記錄任何別的值都會被拒收。
  **上游分歧（2026-08-25 補查）**：MetaFind 引用的另一個範式源 DPR 用的是**未正規化點積**
  （UPSTREAM DPR p.3 Eq.1 `sim(q,p)=E_Q(q)ᵀE_P(p)`，本地 `docs/paper/dpr_2004.04906.pdf`）。
  兩個上游答案不同；取 cosine 因為嵌入空間是 CLIP 的（ULIP-2 是更近的上游）。分歧已登記，不是沒查。
- 挑戰點（PAPER :10）：查詢是多模態的、模態會缺、要 layout-aware 才能空間合理。

---

## 2. 架構總覽與資料流

### 2.1 完整正向流（含每一步的張量形狀）

```
════════ Gallery 塔（離線，一次算完）════════
資產 A：
  text 標註字串 ──CLIP text encoder(凍)──► e_text  (1280,)
  11/12 視角渲染圖 ──CLIP image encoder(凍)──► 每視角 (1280,)，聚合成 e_img (1280,)
  點雲 (10000,6) ──PointBERT(訓)──► (768,) ──pc_projection (768×1280)──► e_pc (1280,)
  {e_text, e_img, e_pc} ──Gallery ModalityFusion──► e_gallery (1280,)
  → 存進 gallery index（Stage 1 訓完後全部重算一次、凍結）

════════ Query 塔（線上）════════
使用者給的模態子集 → 同一 ULIP-2 骨幹編碼 →
  缺席模態插 learned mask token（不是补零）→
  Query ModalityFusion → Fusion(e_text, e_img, e_pc)  (1280,)
場景圖 G ──ESSGNN──► e_layout (1280,)
e_query = Fusion(...) + λ · e_layout          λ 為 learnable scalar
════════ 檢索 ════════
cosine(e_query, 每個 e_gallery) → top-1 → 擺入場景 → G 增加一個節點 → 下一件
```

證據：雙塔（PAPER `2methdology.tex:34`）；gallery modality-complete 且訓後凍結（PAPER :34）；
Eq. 6 殘差（PAPER :83-87）；Algorithm 1 迭代（PAPER :117-135）；
1280 = ViT-bigG-14 嵌入寬度（**量測事實，論文全文沒有任何維度數字**——IMPL `ulip_backbone.py:90`，
所以我們的 config 一律把 dim 設成 REQUIRED 參數、不給「看起來像論文值」的預設）。

**被選中的資產回流 query 側**：MetaFind.drawio.png 綠色箭頭 Selected Assets → Query Encoder → ESSGNN
＋ Algorithm 1 第 6 行 `G ← G ∪ {A*}`。下一輪的 e_layout 已含它。（PAPER）
已知圖錯：drawio 圖把 Text→I1..IK / Image→T1..TK 標籤畫反（OBSERVED typo，內文可判定）。

### 2.2 ULIP-2 骨幹（精確規格）

| 元件 | 規格 | 證據 |
|---|---|---|
| 文字/圖片編碼器 | OpenCLIP **ViT-bigG-14**，輸出 1280-d，**凍結** | UPSTREAM ulip2 main.tex:609 "freeze it during the pre-training"；DL-032 |
| 3D 編碼器 | **Point-BERT**：FPS 選 512 group 中心 → kNN 32 鄰點/組 → 減中心座標 → mini-PointNet(2 層 MLP) patch embed → 位置編碼=MLP(中心) → prepend [CLS] → Transformer **depth 18** / dim 384 / heads 6 / encoder_dims 256 / **drop_path 0.1** → 全域特徵 = Concat(CLS, max-pool)。**參數量實測 32.5M** | 流程 UPSTREAM Pointbert_arxiv.tex:121,141,594,597。**⚠ 深度是 18 不是 12（2026-08-26 更正）**：12 是 Point-BERT **論文**的原始架構；ULIP-2 加深了它，`ULIP_2_PointBERT_10k_colored_pointclouds.yaml` 的 `model.depth: 18` 正是 `ULIP_models.py:364` 讀進去的區塊，且**我們載入的 checkpoint 實際數出 18 個 transformer block**（OBSERVED DATA）。原寫 12 是引論文而沒驗 checkpoint |
| 點雲輸入 | **10,000 點 × (xyz+rgb)**；num_group 512 / group_size 32 | UPSTREAM `ULIP_2_PointBERT_10k_colored_pointclouds.yaml`；ULIP-2 App. A.1：10k xyzrgb 50.6 vs 8k xyz 48.9 |
| pc_projection | (768 → 1280) 線性投影，**可訓** | IMPL `ulip_backbone.py:91,286`（checkpoint 形狀驗證） |
| checkpoint | 官方釋出 ULIP-2 權重為起點（不重新預訓練——官方腳本假設 8 GPU，本機 1 張 RTX 5090 32,607 MiB） | CHOICE（checkpoint_initialization，`ulip_backbone.py:7-12`） |

**凍結 CLIP 的證據鏈（三個獨立來源）**：
1. ULIP-1 作者：更新 CLIP → catastrophic forgetting（UPSTREAM ulip1:286）＋源碼藏著沒發表的表：
   解凍 → zero-shot top-1 從 37.1 掉到 **0.0**（UPSTREAM ulip1:1078-1092）。
2. OpenShape：解凍 finetune text encoder 一個 epoch → 無提升、傷泛化 → 全程凍
   （UPSTREAM openshape supp:192-194）。
3. Flamingo：finetune 凍結 LM → −8.0%（UPSTREAM flamingo content.tex:374-378）。
MetaFind 自己（`2methdology.tex:32`）語氣偏向「不凍、更彈性」→ 與 ULIP-2 衝突 → **U-34**，
Kyzen 選 A（凍）→ **DL-032**。`full`（連 ViT-bigG 一起訓）保留為 RA-3 審計路徑，不是主線。

**載入安全（IMPL `ulip_backbone.py:276-322`，照抄即可）**：
checkpoint 只含 point_encoder（226 tensors）+ pc_projection + logit_scale；open_clip 那 974 個 key
由 create_model 另行供給，所以必須 `strict=False` ＋三道斷言：
(a) missing 全部屬於 open_clip 前綴；(b) point_encoder 權重存在；(c) 載入前後參數**真的變了**
——否則隨機初始化的 PointBERT 會輸出形狀正確、範數正常、內容全錯的嵌入，檢索只是「爛」，看不出原因。

**訓練/凍結模式切換（IMPL `ulip_backbone.py:218-262`）**：
requires_grad 和 train()/eval() 是兩件事——PointBERT 有 drop_path 0.1，留在 train() 建 gallery index
會得到**非確定性的索引**。`set_train_scope()` 存在就是為了堵這個 bug。

### 2.3 前處理（每模態的精確規格）

**點雲（n03，IMPL `pointclouds.py` docstring 全文）**
- mesh 表面積加權取樣 10,000 點 + 從貼圖插值 RGB。
- `pc_norm` **只作用於 xyz**：減質心、除最大半徑（單位球）——照抄 ULIP `dataset_3d.py:496`；
  六欄一起 normalize 是另一個張量（錯誤）。（UPSTREAM/IMPL `ulip_backbone.py:115-132`）
- RGB 尺度 [0,1]：**強烈指示、未證明**（ULIP 無色 fallback 是 0.4；但那是 ModelNet 路徑）——
  U-02 追蹤，每個 sidecar 寫 `rgb_scale:"unit"` 以便日後查驗。
- **不做 FPS**：ULIP 的 Objaverse 路徑讀現成 10k 點後隨機排列；FPS 只在 ShapeNet 路徑。
- 我們的雲 ≠ ULIP 官方釋出的雲（U-02 記錄；不是 gate，因為 Stage 1 本來就訓 point encoder）。
- ProcTHOR 資產無 mesh 管線 → **multiview depth shell**（n07b），存世界座標，
  **編碼時**才 pc_norm ＋ 灰色 0.5 填 RGB（幾何無色，不捏造顏色）
  （IMPL `ulip_backbone.py:138-165` prepare_depth_shell；24 件透明資產 depth 拍不到 → 排除出 gallery，F26）。

**渲染（n04，IMPL `render_blender.py:88-107`）**
- 論文說 **11 orthogonal viewpoints**（PAPER `2methdology.tex:28`）→ 我們 **12 視角**（USER 決策，DEV）：
  三圈極角 φ=60°/90°/120°（上/平/下），每圈 4 個方位角、圈間錯開 30°——
  這是 **OpenShape 官方腳本的佈局**，不是 ULIP-2 句子裡的單圈 360/12。
- Blender Cycles、512×512、透視 35mm、CAMERA_DIST 1.2、RGBA 透明背景、OptiX 降噪（USER 決策 2026-08-24）。
- 上游對照：OpenShape 12 彩圖 preset 均勻相機（UPSTREAM method.tex:77）；ULIP-1 是 30 RGB+30 depth（UPSTREAM ulip1:236）。
- 教訓（Blender 事件）：n04 第一版用 pyrender 沒查上游 → 全語料重渲。**動手前先讀上游程式碼**。

**文字（n05 標註 → n06 編碼）**
- **⚠ 標註模型：論文是 GPT-4o，我們實際跑的不是（DEVIATION D-2）。**
  PAPER FACT：`2methdology.tex:28` 寫 GPT-4o。
  OBSERVED IMPL／DATA：`annotate_run.py:84` `MODEL_ID = ".../gemma-4-12B-it"`，
  實際產出的 sidecar 每一筆都記 `annotator_model = ".../gemma-4-12B-it"`（親自開檔核對）。
  D-2 於 2026-08-24 由 Kyzen 決定重新指向 gemma（原文：「D-2 改成 gemma」），
  歷經 Qwen2.5-VL-7B → Qwen3.8-27B → gemma-4-12B-it，**沒有一個是 GPT-4o；偏離是被重新指向，不是被解除**。
  本筆記 v2 此處原寫「GPT-4o 標註」，是把 PAPER 當成 IMPL，**2026-08-26 修正**。
- 標註內容：對 12 視角圖生成結構化欄位：category、尺寸(cm)、materials、placement 四布林、
  synset、volume、mass 等（schema 由 n05 contract 鎖定；PAPER Fig.2 只給 schema 不給字串化規則）。
- **序列化模板（U-15，CHOICE，D0-008 §11.3 批准）**（IMPL `resolve_stage1.py:119`）：
  ```
  "{description} {category} made of {materials}, roughly {width} by {length} by {height} centimetres, {placement}."
  ```
  synset（識別碼不是語言）、volume（凍結塔不會做乘法）、mass（無視覺根據）不進字串、留在磁碟。
- CLIP 77 token 上限：超長**拒收**而不是靜默截斷（截掉的是尾部的 placement）；
  token 計數用不 padding 的 BPE 真數（padded tokenizer 飽和在 77 會把 89-token 偽裝成邊界案例）
  （IMPL `encode_text_image.py:66-90`）。
- **不用 ULIP 的 64 模板 ensemble**：那是類別名稱的 prompt 工程（UPSTREAM ulip1:244-250）；
  MetaFind 的輸入是完整描述句，模板 ensemble 無對象。若日後做 zero-shot 分類對照才需要。

**圖片編碼（n06）**
- open_clip ViT-bigG-14 標準 preprocess（224、CLIP normalize）。
- **11/12 視角全部各存一支 1280-d float16**（28 KB/資產、全語料 ~1.3 GB），
  聚合方式（U-14：mean）另存於 record——聚合是可換的，重編碼不必（IMPL `encode_text_image.py:23-33`）。
- 快取合法性規則：**只有凍結的編碼器輸出可以快取**。點雲嵌入不可快取（Stage 1 在訓 PointBERT）；
  早期版本快取全三支＝把主線偷換成 Table 3 的 "train fuser only"（8.7 分那行）。已修正。

### 2.4 ModalityFusion（精確規格，IMPL `fusion.py` 全文）

- 五種都已實作（論文列的五種，PAPER `2methdology.tex:34`）：mean / mlp / masked_mlp / gated / **transformer**。
- **預設 = transformer**：PAPER `3experiments.tex:143`「the final selected Transformer」（U-13 已解，
  曾被誤設 masked_mlp，已修）。Table 3 的 Mean 9.4 / MLPs 9.9 < Full 11.4 一致。
- Transformer 變體規格（**維度是 CHOICE，論文無任何數字**）：
  `d_model=1280, nhead=8, num_layers=2, dim_feedforward=2048, norm_first=True, batch_first=True`
  ＋ per-modality learnable position embedding（std 0.02）。輸出 = active slots 的加權平均。
- mask tokens：`nn.Parameter(zeros(3,1280))`, init normal std 0.02（CHOICE）。
- `zero_pad=True` 只為重現 Table 3 「Padding missing modalities with 0」那行（10.5 vs 11.4）。
- **U-11（CHOICE）**：缺席 slot 仍參與聚合、帶著 mask token——否則「rather than zero-padding」
  沒有對照物（掉掉 slot 就沒有東西可 pad）。`include_absent_slots=True`。
- **U-23（已量化的邊角）**：三模態獨立 30% 遮罩 → 0.3³=2.7% 的 query **三個全被遮**，
  變成純 mask-token 查詢仍參與對比學習。字面讀法已實作（`allow_empty=True` 預設），
  加 `allow_empty=False` 旗標可強制留一個——效果可量測，不用猜。
- NaN 防護（IMPL）：gated 全 -inf softmax、transformer 全 padded row 都有 fallback。

### 2.5 雙塔共享（U-16，三種讀法全部可跑）

（IMPL `dual_tower.py:54-75`）
```
shared_backbone_separate_fusion   一份 ULIP 骨幹、兩份獨立 Fusion   ← 目前主線（CHOICE）
fully_shared                      一份骨幹、同一個 Fusion 物件（rebind 模組不是共用 config）
fully_separate                    兩份骨幹、兩份 Fusion（backbone 層級尚未實作，僅 fusion 層級）
```
- 歷史 bug：把同一個 FusionConfig 給兩塔 ≠ 共享權重（兩次建構＝兩套參數）。已修為 rebind 模組。
- **重要推論（IMPL `dual_tower.py:300-321`）**：`fully_shared` 與**論文 Stage 2 的凍結契約不相容**——
  2.6 要求「凍 gallery、訓 query fusion」，同一個 Parameter 物件無法同時滿足這兩件事，程式碼直接 raise。
  這是**契約不相容**（可寫進報告的結論），不是數值會爆炸。
- **U-16 狀態：PAPER-AMBIGUOUS，維持現行 CHOICE（2026-08-26 覆核）**。
  這不是論文沉默、而是論文自打架：正文 `2meth:34` 寫 "separate encoders"，
  **Figure 1 卻標 `ULIP-2 (Shared)` 且只畫一個骨幹方塊**（已核對原圖）。
  現行 `shared_backbone_separate_fusion` 是唯一同時滿足兩處的讀法：一份骨幹對上圖，兩份 Fusion 對上正文。
  fully_separate 保留為競爭假設（PointBERT×2 約 +32M 參數，顯存可行，但 backbone 層級尚未實作）。
  DPR 的 "two independent BERT networks" 只是雙塔**範式**的出處，不是本篇 backbone 的實作 authority。

---

## 3. ESSGNN（逐條數學＋論文自我矛盾的完整處理）

### 3.1 場景圖定義

- 節點 = 已擺物件：座標 x_i∈R³（**場景座標不 normalize**——論文的賣點就是能吃未正規化座標系，
  PAPER `2methdology.tex:40`；IMPL `essgnn.py:544-550` 註明「pre-centring 會把要展示的能力刪掉」）
  ＋文字特徵 t_i∈R^d（編碼器沒指定，**S6/U-20 開放**）。
- 邊兩種（PAPER :28,47）：物理邊（adjacency/support）＋語意邊（LLM 生成關係句 → 凍結文字編碼器 → e_ij）。
- **U-29（CHOICE）**：2.5 的 f_h/f_x 只吃**一個** e_ij 參數，物理邊沒有數學入口 →
  物理邊只決定鄰域 N(i)、邊特徵純語意（唯一讓公式簽名字面成立的讀法）
  （IMPL `resolve_stage2.py:20-25`, EDGE_DECISIONS topology="support_union_adjacency"）。
- **U-19（CHOICE）**：邊無向（對稱），配 n07 存的邊。
- **U-30（CHOICE）**：語意邊快取缺失 → **learned missing-edge token**（`nn.Parameter(1280 或 e 維)`, std 0.02）。
  禁零填——零是邊空間的合法點，會被讀成真關係（L1-SEMEDGE-NO-ZEROFILL）。
  token 必須是 nn.Parameter 走 optimizer；早期版本用 seeded numpy 向量＝永不更新、不進 checkpoint。
- 完全圖不可行的量測：12,000 間房的語意邊 ~1.3e5 條 vs 完全圖需 ~1.07M 條 LLM 關係（DATA，`essgnn.py:412-414`）。

### 3.2 論文的兩個 ESSGNN（C1，最大的矛盾）

**§2.5 版（sec25_two_mlp）**——兩個獨立 MLP，各吃原始 tuple（MF-2 / MF-3，逐字）：
```
h_i^{l+1} = h_i^l + Σ_{j∈N(i)} f_h(d_ij^l, h_i^l, h_j^l, e_ij; θ_h)          f_h: R^(2d+1+e)→R^d
x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i^l−x_j^l)·f_x(d_ij^l, h_i^{l+1}, h_j^{l+1}, e_ij; θ_x)   f_x: R^(2d+1+e)→R³
d_ij^l = ‖x_i^l−x_j^l‖₂（一次方；上標 l＝每層用當層座標重算）
```
（「literal 2.5」完整組合 = sec25_two_mlp ＋ euclidean ＋ h⁰=Concat(x,t) ＋ coord_feat=updated——
這一整包就是 RA-1 審計要跑的競爭假設；主線只在各子項有獨立裁決處偏離，見 §3.3。）
**Appendix 版（appendix_shared_msg）**——一個共享 message，φ_x/φ_h 只看 message：
```
m_ij      = φ_e(h_i^l, h_j^l, ‖x_i−x_j‖², e_ij)
x_i^{l+1} = x_i^l + Σ_{j≠i} (x_i^l−x_j^l)·φ_x(m_ij)
h_i^{l+1} = h_i^l + Σ_{j≠i} φ_h(m_ij)
```
不同參數量、不同梯度路徑、不同函數。**兩個都實作了**（IMPL ESSGCL / ESSGCLShared），
**主線 = appendix 版**（DECIDED 2026-08-17，INFER 非 paper fact）：
2.5 同一段帶三個抄寫錯（C2/C3/C4），appendix 內部自洽、等變性證明就是為它寫的。
sec25 版留作競爭假設要量測，不是 fallback。

### 3.3 五個子矛盾的處理（每個都有裁決）

| # | 矛盾 | 裁決 | 依據 |
|---|---|---|---|
| C2 | 2.5 寫 h⁰=Concat(x_i,t_i)；appendix 證明假設 h⁰ 對 SE(3) 不變 | **h⁰ = t_i**（`h0_mode="semantic"`）；Concat 版留作 RA-1 審計（預期等變測試失敗） | Concat 讓 e_layout 繼承座標敏感性——正是 ESSGNN 要消滅的病 |
| C3 | 2.5 型別 f_x→R³；證明要 Q 能提出括號、只有純量成立 | **φ_x 輸出純量，無旗標、無條件**（DL-004 USER 核准：判 PAPER-AMBIGUOUS，不寫「論文錯了」） | EGNN model.tex 明文 "φ_x: R^nf → R^1 outputs a scalar value"（一致但非證據） |
| C4 | h 寬度不閉合（h⁰ 是 d+3，f_h 讀 2d 寫 d） | 隨 C2 的 semantic 讀法自動閉合 | — |
| C5 | 2.5 用 j∈N(i)；appendix 用 j≠i | **N(i)**（scene graph 的邊），**座標式 MF-13 與特徵式 MF-14 同時套用** | EGNN 自己說兩者都是合法選項（model.tex verbatim），且說明鄰域限制要在兩式一起套；e_ij 只在相連 pair 存在。**補登記的證明前提（2026-08-26）**：appendix 的等變性代數是對固定 `j≠i` 寫的；改成 sparse N(i) 後，證明額外需要「**N(i) 在剛體變換下不變**」。我們的邊來自 support／adjacency，平移旋轉後不變，前提成立，但這是**我們補的假設，論文沒寫**，必須列進等變性測試的前置條件。 |
| C6/U-17 | d_ij 一次方（2.5）vs 平方（appendix、EGNN） | **squared** 為主線；`distance` 旗標保留 euclidean | 兩者都 SE(3) 不變、只是餵 MLP 的數字不同；appendix 帶著證明 |

其他登記：C7 = Stage 1 單向 / Stage 2 雙向、論文未解釋（照做即可）；
C8 = 論文把 SE(3) 說成「縮放敏感」的解法但 SE(3) 不含縮放（RA-4 審計：等變性對 scale 不成立，照實報告）。

### 3.4 ESSGNN 已鎖定的完整配置（IMPL `resolve_stage2.py:86-140` ARCH_DECISIONS）

```json
{ "architecture_family": "appendix_shared_msg",
  "use_io_projections": true,        // U-33：embed_in(t_i→hidden) + embed_out(hidden→1280)，EGNN 慣例
  "distance": "squared",             // U-17
  "coord_feat": "current",           // appendix 家族唯一合法值（φ_x 讀 m_ij，由 h^l 建）
  "layer_sharing": "independent",    // U-31：L 層各自參數（shared 也實作，另一個模型）
  "pooling": "mean",                 // ⚠ 建議改 sum，見下方 EGNN 官方對照
  "hidden_dim": 128,                 // 與 EGNN QM9 的 nf=128 一致
  "n_layers": 4,                     // ⚠ 建議改 7，這個 4 是 N-body 的值不是 QM9 的
  "mlp_structure": "linear_silu_linear" }  // U-35：Linear→SiLU→Linear（描述程式碼，防漂移斷言）
```
＋ PRIMARY_INTERPRETATION（`essgnn.py:276-281`）：
`h0_mode=semantic / coords_agg=sum（Eq.3 是 Σ；EGNN 預設 mean，F9）/ edge_proj_dim=None / normalize_coord_diff=False`。

尚缺三個 REQUIRED 維度（建 config 時必填、來源不是論文）：
`node_feat_dim`（t_i 編碼器寬度，U-20 開放）、`edge_feat_dim`（e_ij 編碼器寬度，U-06 開放：CLIP 1280 / BERT 768 / CLIP-B 512）、
`out_dim=1280`（必等於 fusion 輸出，Eq. 6 相加才成立——建構子強制檢查）。
**規劃預設（待批）**：t_i 與 e_ij 都用「與骨幹同一顆凍結 CLIP text encoder」→ 1280/1280，
理由：不引入第二顆文字模型、與 query 側同空間；appendix 說 "e.g., CLIP or BERT" 允許。

**⚠ 三處偏離 EGNN 官方預設，我原本標成「我們自己選的」是沒查 repo（2026-08-26 補查）**

MetaFind `2meth:42` 引 EGNN 的脈絡明寫是 **drug design**，那正是 EGNN 官方的 **QM9** 任務，
所以 QM9 的設定才是對應的上游基準（N-body 是另一個任務、另一組數字）。

| 項目 | EGNN QM9 官方 | 我們 | 判定 |
|---|---|---|---|
| `nf` / hidden | **128**（`main_qm9.py:30`） | 128 | ✅ 一致 |
| `n_layers` | **7**（`main_qm9.py:34`） | **4** | ❌ **4 是 N-body 的值**（`main_nbody.py:35`）；我把兩個任務的設定混搭了 |
| readout ／ Pooling | **sum**（`qm9/models.py:83` `torch.sum(h, dim=1)`） | **mean** | ❌ MetaFind 只寫 `Pooling(...)` 沒命名＝真沉默，standing rule 應照 sum |
| `attention` | **啟用**（`main_qm9.py:32` default 1，做法是 `Sigmoid` 閘乘在 edge 輸出上） | 未實作 | ⚪ **維持不加**：MetaFind 的 MF-2/3/10/13/14 明文沒有 attention 項，論文有寫就不是沉默 |
| `act_fn` | SiLU | SiLU | ✅ |
| `residual` | True | True（殘差在外） | ✅ |
| `normalize` ／ `tanh` | 皆 False，原碼註解寫「論文沒用」 | 皆 False | ✅ |
| `embedding_in/out` | 有 | `use_io_projections=True` | ✅ |

**（2026-08-27 更正）「不需要裁決」是越權，已撤回。**
兩個值都屬 **Type C**（EGNN 在 QM9 這個**別的任務**上的實驗設定），依 Rule 16 必須經 Kyzen 核可。
現況：`n_layers = 7` 已核可；`pooling` 保留 sum 為起點但要**短跑比 sum vs mean** 後再定。
原文如下（保留供對照）：兩者都是「MetaFind 真沉默、EGNN 官方有明確值」的情形，
但會改動已落檔的 `essgnn_arch_protocol`，屬協定變更，先報備再改。

**Stage 2 訓練超參數也不是空白**（`main_qm9.py`）：
`Adam` · `lr 1e-3` · `weight_decay 1e-16` · `batch 96` · `epochs 1000`。
任務不同（分子性質回歸 vs 對比檢索）不能整包照搬，但**它不是「沒有出處」**，
§10 #8 原寫「S4 全無」是低估。

工程細節（IMPL）：φ_x 最後一層 xavier gain 0.001＋零 bias——第 0 步大位移會讓等變性數值檢查失去意義；
F8 觀察：1280 維 e_ij 旁邊只有 1 個幾何純量，幾何訊號可能被淹沒——這是論文設計的性質，
量測回報、不悄悄修正（edge_proj_dim 旗標保留）。

### 3.5 等變性驗證計畫

- SC-4/5/6 三條分開斷言：座標通道等變、h 不變、e_layout 不變（三件事不是一件）。
- `test_se3_equivariance` 兩個家族都跑；RA-1（concat_xt 預期失敗）、RA-4（scale 預期失敗）是「預期失敗」審計。
- 隨機 Q∈SO(3)、T∈R³，容差內 assert `ESSGNN(Qx+T, h, E).e_layout == ESSGNN(x, h, E).e_layout`。
- **群的記法差（非矛盾）**：方法章 Eq. 4 寫 R∈SO(3)（旋轉，主張 SE(3)）；appendix Eq. 9/15 寫
  Q∈R^{3×3} orthogonal（含鏡射＝O(3)）。證明對整個 O(3)+平移（E(3)）成立，**強於**內文主張的
  SE(3)，所以主張被涵蓋、不衝突。測試以 SO(3) 為準即可，鏡射 case 可加測不必加。

### 3.6 公式逐條對照表（論文 20 條 display 公式 ↔ 本筆記 ↔ 程式碼）

編號取自 `docs/audit/A_FORMULA_INVENTORY.md`（sha256 驗證過的 tex 原文）。
判定分四類：**✅ 逐字等價**｜**🔁 數學等價（形式改寫，附推導）**｜**⚖️ 有裁決的偏離（論文自我矛盾，
選擇已登記）**｜**✍️ 論文未定義、由已登記 CHOICE 補**。

| MF | 論文原式（出處） | 本筆記/程式碼 | 判定 |
|---|---|---|---|
| MF-1 | `A*=argmax_{A∈𝒜} sim(f_query(Q), f_gallery(A))`（2meth:7） | §1 同式；`sim`=cosine 為 U-24 CHOICE（S1 沉默） | ✅＋✍️(sim) |
| MF-U1 | `h_i^(0)=Concat(x_i,t_i)`（2meth:43） | 主線 h⁰=t_i（C2 裁決：印刷式違反 appendix 自己的不變性前提，且會讓 e_layout 繼承座標敏感——論文自己的定理與動機都站 t_i 這邊）；Concat 版留 RA-1 預期失敗審計 | ⚖️ C2 |
| MF-2 | Eq.2 h 更新（2meth:50） | ESSGCL `f_h(cat[h_i,h_j,radial,e_ij])`＋殘差在外。**[降級 2026-08-26]** 論文引數序是 `(d, h_i, h_j, e)`，程式是 `[h_i, h_j, d, e]`：在 MLP 函數族上可由第一層權重欄置換互相表示，**架構等價但同一組參數值不逐字等價**，故不宜標 ✅ | 🔁 引數重排等價 |
| MF-3 | Eq.3 x 更新，f_x→R³（2meth:50,54） | §3.2 逐字；實作 f_x 輸出**純量**（C3/DL-004：R³ 讀法讓論文自己的 Eq.13 證明失效——`Σ(Qx_i−Qx_j)φ_x = QΣ(x_i−x_j)φ_x` 只對純量成立） | ⚖️ C3 |
| MF-U4 | `e_layout=Pooling({h_i^(L)})`（2meth:55） | 同式；Pooling 未命名（S2）→ mean CHOICE | ✅＋✍️ |
| MF-4 | Eq.4 等變條件 R∈SO(3)（2meth:62） | §3.5；測試照此斷言 | ✅ |
| MF-5 | Eq.5 L_pre 單向 InfoNCE（2meth:76） | `CE((q̂ĝᵀ)/τ, arange(B))`。**推導**：CE 第 i 列 = −log[exp(q̂_i·ĝ_i/τ)/Σ_j exp(q̂_i·ĝ_j/τ)]；sim=cosine=正規化點積、分母 Σ_{A'∈B} 含正例（CE 天然包含）、batch 取平均＝逐 query 式的期望。`losses.py:170-175`。**[加註前提 2026-08-26]** 這個等價依賴三個論文沒明講的實作條件：(a) 正例是 batch 對角線；(b) 正例包含在分母 𝓑 內；(c) code 的 batch 平均對應論文的逐樣本 loss。三者都是我們的 reconstruction，不是論文寫的 | 🔁 條件式（對角配對＋正例∈𝓑） |
| MF-6 | Eq.6 `e_query=Fusion(e_text,e_img,e_pc)+λ·e_layout`（2meth:84） | `dual_tower.py:248` `fused + lam*layout` 逐字；layout 缺席時整項省略（U-28 CHOICE；內文 :83 本來就說 e_layout「optional」） | ✅ |
| MF-7 | Eq.7a/7b 雙向（2meth:93） | 7a=CE(logits)；7b=CE(logitsᵀ)。**推導**：logitsᵀ[i,j]=q̂_j·ĝ_i/τ=sim(e_gallery_i, e'_query_j)/τ（cosine 對稱），分母恰為 Σ_{e'_query∈B}。前提＝gallery batch 與 query batch 同組配對（`losses.py:188` 註明；解耦 gallery 需另算 logits）。**[加註前提 2026-08-26]** 轉置成立還額外要求 `sim(a,b)=sim(b,a)`，而 **MetaFind 從未指定 sim**，對稱性來自我們選的 cosine（U-24）。所以這條同時帶一個 ✍️ | 🔁 條件式 ＋ ✍️（對稱 sim） |
| MF-8 | Eq.8 `L=½(q2g+g2q)`（2meth:100） | `losses.py:193` `0.5*(...)` | ✅ |
| MF-9/15 | 等變條件 Q orthogonal（app:25,72） | §3.5 群記法差說明 | ✅ |
| MF-10 | `m_ij=φ_e(h_i,h_j,‖x_i−x_j‖²,e_ij)`（app:31） | ESSGCLShared `phi_e(cat[h_i,h_j,sq,e_ij])`，distance=squared | ✅（註1） |
| MF-11/12, U15-17 | 證明中間步（app:37-55） | 非實作對象；§3 證明摘要與其一致 | ✅ |
| MF-13 | `x_i^{l+1}=x_i^l+Σ_{j≠i}(x_i−x_j)·φ_x(m_ij)`（app:49） | 同式但 Σ over **N(i)**（C5：EGNN model.tex 明文兩者皆合法選項；完全圖需 ~1.07M 條 LLM 邊 vs 實際 ~1.3e5）；φ_x 純量（同 C3） | ⚖️ C5 |
| MF-14 | `h_i^{l+1}=h_i^l+Σ_{j≠i}φ_h(m_ij)`（app:64） | 殘差位置一致（在 φ_h 外；EGNN 的 φ_h 內建殘差，MetaFind 沒有，U-35 註記）**但求和範圍同樣被 C5 改成 N(i)**（IMPL `essgnn.py:458` 用 `row`＝edge_index）。**[修正 2026-08-26]** 本表原標 ✅，只核對了殘差位置、漏核求和範圍：C5 同時作用於 MF-13 與 MF-14，EGNN model.tex 也明說鄰域限制要在座標式與聚合式**兩邊一起**套用。這是筆記自身的疏漏，不是論文歧義。 | ⚖️ C5 |

註1：MLP 輸入的**引數順序**（論文寫 f_h(d, h_i, h_j, e)；程式碼 cat 順序是 [h_i, h_j, d, e]）
對學習到的線性層無語意差——第一層權重矩陣的欄排列而已。列出以免日後被當成發現。

文字性「公式」另兩條也對上：f_h/f_x 型別簽名 R^(2d+1+e)（2meth:54）＝code `2*h+1+edge_dim`；
d_ij 定義（2meth:54 一次方 vs app Eq.10-12 平方）＝C6/U-17 已裁決 squared、旗標保留。

**結論（2026-08-26 修訂措辭）**：20 條全部已建立 mapping。**其中 MF-14 原本標錯**（漏把 C5 套到特徵式），那是筆記自身的疏漏、已修正。其餘無法逐字對上的部分，分別是：
(a) 論文自我矛盾處的已登記裁決（C2/C3/C5，各附「為什麼論文自己的定理站我們這邊」）；
(b) 數學等價改寫（MF-5/7 的 CE 形式，推導已附）；
(c) 論文未定義符號（sim/Pooling/Fusion）由已登記 CHOICE 補位。

---

## 4. 訓練配方

### 4.1 Stage 1 — Cross-Modal Alignment Pretraining

**資料**：Objaverse-LVIS ~48K（論文值；LVIS 官方 46,832/1,156 類——差額 UNKNOWN；
我們實際語料 46,052，DATA）。80/20 split（PAPER `3experiments.tex:8`），
split 固定於 n09、seed 記錄（IMPL `splits.py`）。

**誰在訓**（DL-032 定案）：
```
凍結：ViT-bigG-14 text + image（→ n06 快取合法）
訓練：PointBERT + pc_projection + Query Fusion + Gallery Fusion（兩塔都訓，PAPER 2methdology.tex:75）
τ：固定 0.5 buffer，不是 Parameter
```

**每步流程（目前提案，非定稿——本節仍有 epochs / λ / U-16 / 評估 gallery 四項未決）**：
1. 取 batch 64 個資產。query 側與 gallery 側是**同一批資產**（正例=對角線）。
2. gallery 側：三模態齊全 → cached e_text/e_img ＋ 現算 e_pc → Gallery Fusion。
3. query 側：`sample_modality_mask(B=64, p=0.3)` 每模態獨立遮罩（per-sample per-modality；
   PAPER "independently"）→ 被遮的插 mask token → Query Fusion。
4. 損失 **Eq. 5 單向 q→g**（PAPER :76-79）：
   `L = CE( (q̂ ĝᵀ)/τ , arange(B) )`，q̂ ĝ 皆 L2-normalized，τ=0.5。
   （C7：與上游雙向不同，論文明文，照做。）
5. AdamW(wd 0.1, betas(0.9,0.98), eps 1e-8)、cosine、warmup 1 epoch（lr_start 1e-6）、batch 64。
   出處：batch/AdamW = UPSTREAM ulip1:367-370；wd/betas/warmup/cosine = UPSTREAM `upstream/ULIP/main.py:47-59,203`。
   **⚠ lr 是上游自己打架（2026-08-26 查 repo 後升級的認識）**：
   ULIP-1 **論文** `main.tex:367-370` 寫 **1e-3**；
   但 ULIP **官方預訓練腳本** `scripts/pretrain_pointbert.sh` **明確傳 `--lr 3e-3`**，
   `main.py:52` 的 default 也是 3e-3。
   所以這不是「default 沒改到」的遺留值，而是**作者實際跑的就是 3e-3，跟自己論文寫的不一致**。
   兩邊都是一手來源，standing rule 在這裡失效（上游內部衝突）→ **需要 Kyzen 裁決，見 §10 #11**。
6. epochs：**250，UPSTREAM FACT（2026-08-26 更正，原標 UNKNOWN 是錯的）**。
   我上一輪只 grep 了 ULIP-2 **論文**就宣告無源，**漏查 repo**，違反 standing rule 的「論文＋程式碼」。
   三方一致：ULIP-1 論文 `main.tex:367-370` 寫 250；`main.py:47` default 250；
   官方腳本 `pretrain_pointbert.sh` 沒有覆蓋它。ULIP-2 共用同一支 `main.py`（模型名 `ULIP2_PointBERT_Colored`），
   官方沒有釋出 ULIP-2 專屬的預訓練腳本，所以 ULIP-2 的預訓練輪數就繼承 250。
   **（2026-08-27 更正）** 上一版寫「依 standing rule 採用 250」是越權 —— 依現行治理規則，
   `epochs` 屬 **Type B**，`main.py:47` 的 argparse 預設更屬 **Type D**，兩者都不能單獨落地。
   **250 現在的地位是 Kyzen 核可的 IMPLEMENTATION CHOICE（2026-08-27），上游參考 ULIP-1 論文，
   並且要先跑 5 → 10 → 25 pilot 慢慢加上去**，不是無條件跑滿。
   `resolve_stage1.py` 目前的 50 在 pilot 期間反而是合適的起點。

7. **早停與 checkpoint 選擇：上游有明確做法，照抄即可（2026-08-26 查 repo 後改寫）**。

   **⚠ 先撤回**：本節原寫「以 20% 測試集 R@1 早停」。那會污染 Table 1——
   MetaFind `3exp:8` 那 20% 是 **test**，拿它選 checkpoint 就變成 model-selection set。
   `splits.py:65` 只有 `TRAIN_FRACTION=0.8`、沒有 validation。撤回。

   **上游怎麼做（UPSTREAM FACT，`upstream/ULIP/main.py:212-240`）**：
   - **不早停**。`for epoch in range(start_epoch, args.epochs)` 一路跑滿 250。
   - 每個 epoch 跑一次驗證，`is_best = acc1 > best_acc1`，存 best checkpoint（`main.py:225-231`）。
   - **驗證資料是訓練集以外的獨立 benchmark**：`--validate_dataset_name` default `modelnet40`（`main.py:40`），
     而預訓練跑在 ShapeNet／Objaverse 上。**上游從不從訓練集切 validation，也從不用最終回報的 split 選 checkpoint。**

   **（2026-08-27 更正）上游的紀律有三條，但不是「全部採用」—— 那句是越權，已撤回。**
   三條分別經 Kyzen 逐項裁決（見 §4.3 台帳）：不設自動早停 ✅ · 取 best ✅（但依據改為 D-3 的
   10% validation，因為我們沒有上游那個獨立下游 benchmark） ·
   **選擇依據必須來自不是最終回報的那個 split**。第三條正好證實上一輪的擔心是對的，
   而且這不是我們的潔癖，是上游自己在守的規矩。

   **唯一缺口**：上游有一個現成的獨立 benchmark（ModelNet40）可當驗證，
   MetaFind 訓練與評估**都在 Objaverse-LVIS 上**，缺這個前提。
   這一小塊上游答不了 → 上呈 Kyzen（§10 #3），選項見該處。

**InfoNCE 負例紀律**：負例數 = 實際 batch 內 gallery 數。**梯度累積不增加負例**——
但 CLIP 兩支已凍結快取、只有 PointBERT 活著，batch 64 單卡預估可原生放下；
仍須一次實測 forward 峰值記憶體後才可宣稱（目前 INFER）。

**Stage 1 完成後**：gallery encoder 凍結（`freeze_gallery(True)`：requires_grad=False ＋ eval()），
`set_train_scope("fuser_only")` 讓 PointBERT 進 eval（drop_path 0.1 否則索引非確定性），
全 46K 資產重新編碼 → gallery index（n10）。

**檢查點選擇（2026-08-27 Kyzen 拍板，DEVIATION D-3）**：以 **10% validation** 的
Mean R@1（跨模態條件平均）選 best，平手用 Mean R@5。**10% test 全程不參與選擇，只在最後報一次分。**
上一版寫「以測試集 R@1 選 best」是漏刪的殘句，與同節 §4.1 的撤回自相矛盾，已刪除。
保存 optimizer state、seed、
git SHA、protocol hashes（`stage1_protocol` 帶超參數 artifact 的 hash，IMPL splits.py）。

### 4.2 Stage 2 — Layout-Aware Fine-Tuning

**資料**：ProcTHOR-10K 房間（PAPER）；80/20 by n09c scene_splits。

**訓練集構造（U-08 系列，全部已解、IMPL `resolve_stage2.py:60-77` STAGE2_DECISIONS）**：
```
gallery_scope        = procthor            （Stage 2 的庫 = ProcTHOR 資產，非 46K）
positive_identity    = same_asset_id       （leave-one-out：目標資產自己是正例）
modality_source      = ai2thor_isolated    （資產隔離拍攝出 text/image/pc）
pointcloud_source    = multiview_depth_shell（透明資產拍不到 → 排除，F26）
sampling_unit        = object_instance     （每個合格實例一個樣本）
target_removed_before_essgnn = True        （目標先從場景圖移除再算 e_layout——防資訊洩漏）
samples_per_house    = all_eligible
epoch_definition     = 一輪走完全部 leave-one-out 樣本
batch_positive_uniqueness = True           （同 batch 不得有重複正例資產——InfoNCE 假負例防護）
```
- **已登記風險**：leave-one-out（從完整房間移走一件）與推論時的漸進式構圖（房間從空開始長大）
  分布不一致——GPT 提出、我方確認，緩解方案（curriculum / 隨機保留子集）列開放實驗。

**誰在訓**（PAPER `3experiments.tex:24`＝`2methdology.tex:89`，一致）：
```
凍結：兩塔的 ULIP-2 全部（含 PointBERT）＋ Gallery Fusion
訓練：ESSGNN ＋ Query Fusion ＋ λ
單一共用 fusion head（論文報告值的設定；雙 head 方案論文自己提了但沒用）
```

**每步流程**：
1. batch 取 B 個 leave-one-out 樣本（B 未定，見 §10；規劃預設 64 對齊 Stage 1）。
2. 每樣本：場景圖（少了目標）→ ESSGNN → e_layout；目標資產的可用模態 → Query Fusion。
3. `e_query = Fusion(...) + λ·e_layout`。
4. **scene dropout 30%，batch 粒度**（U-32：論文寫 "omitted in 30% of **batches**"——
   一個 batch 一次抽籤，全 batch 同 regime；per-sample 讀法留作 variant。
   注意與 Stage 1 的 per-sample 模態遮罩**不是同一種粒度**，早期版本搞混過）
  （IMPL `dual_tower.py:333-357`）。
5. 損失 **Eq. 7a/7b/8 雙向**：`L = ½(CE(logits) + CE(logitsᵀ))`，τ=0.5
   （轉置合法的前提：gallery batch 與 query batch 是同一組資產；解耦 gallery 要另算 logits——
   IMPL `losses.py:188-193` 有註記）。
6. λ 初值：**未定**（§10 決策 #2）。optimizer/lr：論文全無（S4）→ 規劃預設沿 Stage 1 的
   AdamW，lr 待小掃描（1e-3 / 1e-4），以 val 曲線定，全程記 CHOICE。

**Stage 2 gallery index**：ProcTHOR 合格資產（有 depth-shell 點雲者），
positive_map 由 n09b 寫死（identity mapping；無點雲者不得為正例——「有名字沒向量的損失」禁止）。

### 4.3 超參數總帳（單一真相表）

**DEVIATION D-3 — 資料切分（2026-08-27，Kyzen 拍板）**

論文 `3experiments.tex:8` 的全部原話：

> "we allocate **80%** of the data for training and reserve the remaining **20%** for testing"

**兩份，沒有 validation，而且對「怎麼挑 checkpoint」一個字都沒寫。**

Kyzen 的裁決：把**保留的 20% 再對半切**，10% 挑 checkpoint、10% 報分。

```
80%  訓練          ← 與論文完全一致，一個資產都沒動
10%  validation    ← 挑 checkpoint（查詢）
10%  test          ← 最後報一次分（查詢）

gallery（候選池）= 整個 20% = 9,211      ← 不對半切
```

**gallery 為什麼不切**：在檢索裡候選池大小就是題目難度。池子砍半，R@1 會**機械性地變高**，
那個分數就不能跟論文並列了。validation 的資產只是**當幹擾項**留在池子裡，從不當 test 的查詢，
所以不構成洩漏。

**這是對論文明文的偏離，記為 D-3。** 偏離的只有「查詢從哪來」，訓練量與候選池大小都與論文一致。

IMPL `metafind/data/splits.py`：`split_heldout()` · `V_val_selection` 協定（`reported: False`）·
三向洩漏斷言 · `splits.json` 帶 `deviation: "D-3"`。
n09c（房屋）**未動** —— Stage 2 在 G6 後面，開了再比照辦理。

---



| 參數 | 值 | 等級 | 出處 |
|---|---|---|---|
| τ | 0.5 固定 | PAPER | 3experiments.tex:15 "The temperature is 0.5 for all experiments" |
| τ 可學性 | 否 | CHOICE（USER 批准） | 論文把 f_h/f_x/λ 叫 learnable、τ 兩處叫 hyperparameter（詞彙對比推論，`resolve_stage1.py` C-001） |
| 模態遮罩 | 30%，模態間獨立 | **PAPER（模態獨立）＋CHOICE（樣本獨立）** | 2meth:75 明寫 "each modality ... independently masked"，模態間獨立是 PAPER；「不同樣本的遮罩也獨立」論文沒寫，是實作選擇（2026-08-26 拆級） |
| scene dropout | 30% per-**batch** | **PAPER，含粒度（2026-08-26 升級）** | 2meth:89 原句就是 "omitted in 30% of **batches**"，30% 與 batch 粒度都有直接文字支持；仍屬 CHOICE 的只有 RNG 實作細節 |
| split | 80/20 兩資料集 | PAPER | 3experiments.tex:8 |
| batch / optimizer | 64 / AdamW | UPSTREAM | ulip1 main.tex:367-370；`main.py:50` default 亦 64 |
| **lr** | **5e-4 起跑** | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27） | 上游四個候選：1e-3（ulip1:367-370）· 3e-3（`pretrain_pointbert.sh`，但那是 ULIP-1 8192 點）· 5e-4（OpenShape supp:190 指定給 32.3M PointBERT；Point-BERT 論文 `:216` 亦同）· 4e-4（supp:190，72.1M 版，同節 `:146` 說該版嚴重過擬合）。**Kyzen 定 5e-4 起跑**，第一輪 sweep 2.5e-4 / 5e-4 / 7.5e-4 / 1e-3，3e-3 不入第一輪。**不得用 10% test 調 lr** |
| wd / betas / eps / warmup | 0.1 / (0.9,0.98) / 1e-8 / 1 ep cosine | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27），上游參考 `upstream/ULIP/main.py` | Kyzen 的理由：MetaFind 未交代訓練配方，Stage 1 骨幹 lineage 來自 ULIP-2，第一版先用 ULIP 成熟配方，避免一次動太多變因。pilot 顯示不穩仍可調 |
| epochs | **250（上限）** | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27），上游參考 ULIP-1 論文 `main.tex:367-370` | Kyzen 原話：「250 啊 因為 ULIP 有說喔 我覺得就參考啊」「要測試所以慢慢加」。**先跑 5 → 10 → 25 pilot** 確認 loss／梯度／記憶體／管線正常，再進完整訓練。250 是上限，不是無條件跑滿。**`main.py:47` 的 argparse 預設是 Type D，不能單獨當根據** |
| 早停 | **不設自動早停** | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27） | Kyzen 的理由：現在就設 patience=20 等於偷偷先決定了 model selection 協定。`early_stopping=False`，**保留依實際 validation 曲線人工中止的權利**。上游參考 `main.py:212`（跑滿，無 early-stop 分支）。**10% test 不得參與任何中止判斷** |
| checkpoint 選擇 | **10% validation 的 Mean R@1 取 best**，平手用 Mean R@5 | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27，DEVIATION D-3） | 上游有 best-checkpoint 的做法（`main.py:225-231`），但上游是拿**獨立下游 benchmark**（`main.py:40` default modelnet40）當依據，我們沒有那個 benchmark。所以改由 D-3 切出的 10% validation 擔任。**IMPL `splits.py` `V_val_selection`，`reported: False`** |
| seed | 20260816 | CHOICE | resolve_stage1.py |
| λ 初值 | 未定（code 佔位 1.0） | **UNKNOWN→待決** | §10 #2 |
| fusion transformer 尺寸 | 2 層/8 頭/ffn 2048 | CHOICE | 論文無維度 |
| ESSGNN hidden | **未決** | **待決，排在四項之後** | 128 出自 EGNN 論文 `appendix.tex:135` 的 **QM9 實作細節節** → **EGNN EXPERIMENT-SPECIFIC SETTING**，不能單靠 lineage 落地。且 QM9 的 15 維原子 one-hot → 128 是**放大**，我們 1280 → 128 是**壓縮**，同一個數字方向相反。順序見 `docs/ESSGNN_DIM_REVIEW.md` §7 |
| ESSGNN 層數 | **7** | **USER-APPROVED IMPLEMENTATION CHOICE**，上游參考 EGNN QM9 | 7 出自 `main_qm9.py:34` 與論文 `appendix.tex:135`，屬 **EGNN EXPERIMENT-SPECIFIC SETTING**。注意 `qm9/models.py:46` 的 class 建構子預設**也是 4**，是 QM9 腳本顯式覆寫成 7 —— 更證明 7 是實驗設定不是 intrinsic 架構。原文寫「4 是 N-body 的值」不完整，已更正 |
| ESSGNN pooling | **sum 為起點，之後短跑比 sum vs mean** | **USER-APPROVED**（2026-08-17 定 sum；2026-08-27 Kyzen 加註「先測測看有沒有效」） | 論文只寫 `Pooling({h_i^(L)})`，**未命名** → PAPER AMBIGUITY。QM9 用 sum（`qm9/models.py:83`、`appendix.tex:135`）屬 EXPERIMENT-SPECIFIC。**原本「因為 QM9 目標是外延量所以 sum 物理正確」的理由已撤回** —— EGNN 從未如此陳述，且 QM9 亦含 HOMO/LUMO/gap/μ 等非外延目標。sum 會讓 ‖e_layout‖ 隨物件數變動而單一 λ 補不回；mean 則弱化物件數資訊。**兩者短跑比較後再定** |
| Stage 2 lr/batch/epochs | 未定 | UNKNOWN（S4） | 規劃：沿 Stage 1＋小掃描 |

---

## 5. 推論與迭代構圖

（PAPER `2methdology.tex:109-135`）
- gallery 嵌入全預先快取。
- Algorithm 1 逐件：`e_layout←ESSGNN(G)` → 編碼 Q_i → Eq. 6 → argmax → 擺入 → `G←G∪{A*}`。
  （演算法第 3 行寫 "EGNN(G)"——即 ESSGNN，原文用語不一致，無實質歧義。）
- 無 layout 查詢（Table 1 情境）：`layout=None` → e_query = Fusion 輸出，λ 項整個省略
  （**U-28 CHOICE**：論文承認 mismatch 但沒說省略/清零/繞路，我們選省略；影響 Table 1 的 14 格中 7 格）。
- 效率模式（PAPER :115）：全序列（最高品質）↔ 區域分解（語意區域內序列、區域間平行）。
- Table 3 佐證：w/o iterative 只掉 0.1 R@1 / 0.1 分——同一組權重換組合模式即可測
  （IMPL variants: `no_iterative` reuses_ckpt="full"）。

---

## 6. 評估協定（三張表逐一拆解）

### 6.1 Table 1 — Objaverse-LVIS 物件檢索

- 七個查詢條件：T / I / PC / T+I / T+PC / I+PC / T+I+PC；R@1、R@5（PAPER）。
- 正解 = 查詢對應的那件資產自己（跨模態自檢索）。
- **U-09（雙協定並跑）**：gallery 是全 46K 還是 20% 測試集（9,211）？論文沒說，
  R@1 差很大 → **兩個協定都跑、都報**（IMPL `splits.py` docstring；
  想從 baseline 98% PC-only 反推庫大小是不可能的——自檢索兩種庫都趨近 100%）。
  query=測試集 20% 也是假設，已記錄。
- MetaFind 預期型態（PAPER Table 1）：單塔 baseline 的 PC-only 98%+ 來自 query 與 gallery 用同一支嵌入，論文自己的用詞是 **"leading to inflated accuracy"**（`3exp:24`）。我們雙塔 PC-only 是 75.1 ／ 63.2（w/ ESSGNN）。照論文用語寫 inflated 即可，不要寫成「別人灌水、我們誠實」，那是價值判斷、比出處強。
- w/ ESSGNN 在 Table 1 全面掉分（如 T-only 13.8→11.3）：官方解釋 = Stage 2 fusion 適應了
  layout 特徵、在無 layout 資料上 attribution mismatch（PAPER `3experiments.tex:24` 全段）。
- **上游數字不可直接對表**：ULIP/OpenShape 官方評估是 1,156 類 zero-shot 分類
  （UPSTREAM `upstream/ULIP/main.py:399-406`、openshape experiments.tex:7），不是 asset-level R@k。

### 6.2 Table 2 — 場景品質（I-Design 管線）

- 管線：I-Design（LLM agent 設計房間 → 檢索 → 擺位）；把它的 OpenShape 檢索步換成 MetaFind
  （PAPER `3experiments.tex:55`；IMPL 模板 `upstream/IDesign/retrieve.py`）。
- 四維度 1–5 分：Aesthetic / Color&Material / Scene Coherence / Realism&Geometry；
  **GPT-4o ＋ 5 位人類專家**、200 個隨機場景、提供 layout＋渲染圖、prompt 對齊各維度、平均（PAPER）。
  評分模板：`upstream/IDesign/gpt_v_as_evaluator.py`（n17）。
- 對照組：ULIP / OpenShape / MetaFind w/o ESSGNN / w/ ESSGNN。
- **開放（U-21）**：此表檢索庫是 ProcTHOR 資產（~1,439 合格）還是 46K？
  ＋論文 "3,000+ curated assets" vs ProcTHOR 官方 1,633（UPSTREAM procthor 01_intro.tex:10）對不上。

### 6.3 Table 3 — 消融（Text-only）

| 行 | 我們的 variant 對應（IMPL resolve_stage1.py VARIANTS） |
|---|---|
| Full (bidirectional, iterative, ESSGNN) 11.4 | `full`（訓練） |
| w/o iterative 11.3 | `no_iterative`——**同權重**換 parallel 組合 |
| w/o Layout Context 13.5 | **INFER，checkpoint 來源未定（2026-08-26 降級）**：論文沒說這行用哪個 checkpoint，可能是 Stage-1 模型、也可能是 Stage-2 訓完但停用 layout。注意 Table 1 的 Text-only `w/o ESSGNN` 是 **13.8**，與此處 **13.5** 不相等，兩者不能直接畫等號 |
| GAT 11.0 | layout_encoder=gat variant（需實作 GAT baseline） |
| Fusion=Mean 9.4 / MLPs 9.9 | fusion 換型重訓 |
| Dropout 10% 7.3 / 50% 13.2 | p_mask 換值重訓 |
| Train fuser only 8.7 | train_scope="fuser_only"。**INFER（2026-08-26 降級）**：最合理讀法是**編碼器粒度消融**（只訓 fuser vs 連編碼器一起訓），不是拿掉 ESSGNN——ESSGNN 另有 `w/o Layout Context` 一行，且 `3exp:143` 明說 "full encoder fine-tuning yields better performance by allowing earlier layers to adapt"。但**論文沒寫這個消融跑在 Stage 1 還是 Stage 2**，而 `3exp:24` 又說回報的 Stage 2 兩塔全凍，階段歸屬仍是 INFER。原註「與 GPT 核對一致」已刪：與另一個模型核對不構成科學證據 |
| Zero-pad 10.5 | fusion zero_pad=True |

### 6.4 比對論文數字的紀律

協定沒對齊（gallery 大小、query 集、聚合方式）之前，**數字相近不等於復現成功**；
每次比對必附：git SHA、protocol hashes、gallery 規模、seed。單 run 不下推翻性結論。

---

## 7. 上游技術棧（誰貢獻什麼＋我們繼承哪部分）

| 技術 | arXiv | 在 MetaFind 的角色 | 我們繼承的具體內容 |
|---|---|---|---|
| ULIP-2 | 2305.08275 | 兩塔骨幹 | 凍 CLIP、10k colored 點雲 yaml、checkpoint |
| ULIP-1 | 2212.05171 | 框架母體 | lr/batch/AdamW/凍結理由＋0.0 崩潰表、pc_norm、損失結構對照 |
| Point-BERT | 2111.14819 | 3D 編碼器 | 架構全規格（vendored code 的紙本依據） |
| OpenCLIP ViT-bigG-14 | — | 文/圖編碼器 | 1280-d、77 token、preprocess |
| OpenShape | 2305.10764 | 渲染/快取/檢索先例＋baseline | 12 視角佈局、隨機單視角訓練先例、凍結即快取、batch 構造（src/train.py:106-109） |
| EGNN | 2102.09844 | ESSGNN 數學母體（**參考架構**） | EGCL 式、φ_x 純量、N(i) 選項、MLP 形狀 variant 清單 |
| ProcTHOR | 2206.06994 | Stage 2 場景源 | 10K+1K+1K 房、JSON 格式、SAG/擺放先驗、1,633 資產事實 |
| Objaverse(-LVIS) | 2212.08051 | Stage 1 資產庫 | 46,832/1,156 官方數 |
| GPT-4o | — | 論文指定的標註器（n05）與場景評審（n15/n17） | **標註側是 DEVIATION D-2**：實跑 `gemma-4-12B-it`，非 GPT-4o（見 §3.1）。評審側 n17 尚未執行，模型未定（`Q-JUDGE-MODEL` 開放）。prompt 一律是我們的 CHOICE（contract 鎖定） |
| I-Design | 2404.02838 | Table 2 管線 | retrieve.py 替換點、評分模板 |
| DPR | 2004.04906 | 雙塔範式概念源 | 概念引用而已 |
| Flamingo | 2204.14198 | λ 初值辯論先例 | tanh(α) α=0、拆掉掉 4.2%＋不穩 |
| GAT | 1710.10903 | 被否定的 baseline | Table 3 GAT 行需要它 |
| SCA3D/Uni3D(L)/OmniBind/PointCLIP | — | Table 1 baselines | 各自 released checkpoint 評估 |

**Standing rule（Kyzen 2026-08-25）**：MetaFind 沉默 → 預設照上游官方（論文＋程式碼），記 UPSTREAM；
上游也沉默才上呈。反例教訓 = Blender 事件（沒查 OpenShape 渲染器，全語料重渲一次）。

---

## 8. Repo 對應表（實作查表用）

| 概念 | 檔案（行號=關鍵處） |
|---|---|
| 骨幹/凍結/載入斷言/pc_norm | `metafind/models/ulip_backbone.py`（:90 dims, :115 pc_norm, :218 scope, :276 載入） |
| Fusion 五型/mask token/遮罩抽樣 | `metafind/models/fusion.py`（:98 sample_modality_mask, :147 ModalityFusion） |
| 雙塔/λ/scene dropout/凍結排程 | `metafind/models/dual_tower.py`（:207 layout_weight, :300 freeze_gallery, :333 dropout） |
| ESSGNN 兩家族/missing token/pool | `metafind/models/essgnn.py`（:313 ESSGCL, :383 ESSGCLShared, :501 missing token） |
| 損失/τ/雙向 | `metafind/models/losses.py`（:70 PAPER_TAU, :182 單向, :190 雙向） |
| Stage 1 超參數/文字模板/variants | `metafind/models/resolve_stage1.py`（:119 TEXT_TEMPLATE, :237 DEFAULTS, :288 VARIANTS） |
| Stage 2 protocol/正例圖/edge/arch | `metafind/models/resolve_stage2.py`（:60 STAGE2, :79 EDGE, :86 ARCH） |
| split/雙評估協定 | `metafind/data/splits.py`、`scene_splits.py` |
| 點雲 | `metafind/data/pointclouds.py`（n03） |
| 渲染 | `metafind/data/render_blender.py`（:88 N_VIEWS 12, :103 view 佈局） |
| 標註 | `metafind/data/annotate_run.py`（n05，跑步中） |
| 嵌入快取 | `metafind/data/encode_text_image.py`（n06，:66 77-token 拒收） |
| 矛盾登記 | `docs/audit/C_PAPER_CONTRADICTIONS.md`（C1-C8, S1-S6） |
| 決策帳 | `workflow/DECISION_LEDGER.md`（DL-004, DL-010, DL-028~032…） |
| 上游 verbatim 證據 | `workflow/blocks/ULIP2/evidence/UPSTREAM_ULIP1_POINTBERT_VERBATIM.md` |

管線節點鏈：
```
物件鏈： n02→n03(點雲)→n04(渲染)→n05(標註,跑步中)→n05b(編碼協定)→n06(快取)
        →n09(split+協定)→n10(Stage1訓練+gallery index)→Table 1/3
場景鏈： n07(房)→n07b(場景圖+depth shell)→n08(語意邊)→n09b/c(Stage2協定+scene split)
        →n11b(Stage2訓練)→n13(檢索)→n14(I-Design)→n15/16(評分)→n17→Table 2
Gates：G6 擋 Stage 2（essgnn_arch_protocol 未 resolved 不得開跑）；DL-028/029/030/031 管流程。
```

---

## 9. 風險登記（實作前先知道的坑）

1. **快取毒藥**：凡在訓的編碼器，其輸出不可快取（快取全三支＝偷換成 8.7 分的 ablation）。
2. **drop_path 非確定性**：建 index 前 PointBERT 必須 eval()。
3. **strict=False 載入**：三道斷言缺一不可（形狀對、範數對、內容全錯的嵌入無症狀）。
4. **InfoNCE**：梯度累積不加負例；同 batch 正例唯一性（Stage 2 protocol 已鎖）。
5. **τ 漂移**：任何 ≠0.5 或 learnable 的 run，constructor 直接 warn DEVIATION（losses.py:123）。
6. **座標**：資產雲 normalize、場景座標**不** normalize——兩件事，方向相反，各有理由。
7. **77 token**：超長標註拒收；計數用未 padding 的 BPE。
8. **fully_shared 到 Stage 2 必炸**：這是紙面矛盾不是 bug，raise 是對的。
9. **硬體**：單卡 RTX 5090 32,607 MiB；`full`（訓 ViT-bigG）大概率放不下（INFER 未量測，禁止當量測講）。
10. **磁碟**：SMR 碟只放冷資料；45,952 個 .npz 這種高頻小檔寫 NVMe（量測 w_await>5,000ms）。
11. **符號連結**：checkpoints/embeddings 是 symlink——任何 `find` 前加 `-L`。
12. **GPU 檢查**：每個吃重步驟量它跑在 GPU 還是 CPU（函式庫預設常偷偷回 CPU）。
13. **上游先讀**：任何新元件動工前先讀上游官方實作（Blender 事件）。

---

## 9.5 本輪外部審查新增登記的沉默點與矛盾（2026-08-26）

以下五條是外部審查指出、我逐條核對原文後**確認成立**的新登記項。它們原本不在 v2 裡。

**S-A｜Algorithm 1 沒有定義「擺放算子」。** （PAPER 沉默，`2meth:117-135`）
演算法只寫 `Place A* into the scene, update scene graph: G ← G ∪ {A*}`。
但下一輪 ESSGNN 需要新物件的 `x_i ∈ R³`，而論文**沒有定義**：誰決定位置、旋轉、支撐面、
碰撞怎麼解、pose 怎麼寫回圖。Table 2 很可能是由 I-Design 管線代勞，
但**泛用的 MetaFind Algorithm 1 本身沒有 placer 介面**。
影響：Stage 2 訓練與迭代推論之間的邊界不完整，必須自己補一個 placer 契約並登記為 CHOICE。

**S-B｜論文對 baseline 是否雙塔自相矛盾。** （PAPER 內部矛盾，`3exp` Baselines 段 vs §3.2）
Baselines 段寫 OpenShape "adopts a **dual-tower** contrastive retrieval design"；
§3.2 卻寫 "since other models **do not adopt a dual-tower design**"。
後句過度概括。影響：作者用來解釋 baseline PC-only 98–99% 的敘事有瑕疵，
我們複述時不可照抄「所有 baseline 都不是雙塔」。

**S-C｜Fusion 之前要不要先正規化各模態，論文沒說。** （PAPER 沉默）
`2meth:30-35, 74-100` 只說各模態編碼後送 Fusion。
是 `各模態先 normalize → Fusion`，還是 `原始向量 → Fusion → 最後才 normalize`？
對 mean、MLP、Transformer 三種融合都會改變結果。
本筆記對「最終 q/g 做 cosine 正規化」寫得很清楚，但**融合輸入端的正規化政策同樣要鎖，目前未鎖**。
開訓前必須定案並登記。

**S-D｜Eq. 7 的分母到底含不含正例。** （PAPER 沉默，`2meth:93-100`）
原文寫 𝓑 "denotes the **batch of negatives**"，字面上不含正例；
但標準 InfoNCE 與我們的 CE 實作**分母都含正例**。
若照字面理解成純負例，CE 實作就不等價。
協定必須明寫「分母 = 正例 ＋ batch 內負例」，並標 INFER／reconstruction choice，
不能讓 CE 實作把這個選擇藏起來。

**S-E｜Table 3 少一個消融維度的對應列。** （PAPER 內部不一致，`3exp:143`）
作者說消融涵蓋六個維度，明列包含 **gallery encoder flexibility**；
但 Table 3 的九列裡（layout×3、fusion×2、dropout×2、granularity×1、missing-modality×1）
**沒有任何一列對應 gallery encoder flexibility**，表中也沒有 gallery 凍結／解凍的行。
登記為 UNKNOWN，不得假設某一列就是它。

---

## 10. 開放決策清單（含建議預設——Kyzen 拍板用）

### 已拍板（列出免得重問）
| 決策 | 結果 |
|---|---|
| DL-032 CLIP 凍結 | ✅ 凍（照 ULIP-2） |
| τ=0.5 固定不可學 | ✅ USER 批准 |
| 12 視角 Blender（DEV 11→12）＋OptiX | ✅ USER 決策 |
| C1 架構家族 | ✅ appendix_shared_msg 主線（sec25 為競爭假設） |
| DL-004 f_x 純量 | ✅ PAPER-AMBIGUOUS 裁決 |
| tower_sharing | ✅ shared_backbone_separate_fusion |
| U-08 全系列（Stage 2 樣本構造） | ✅ 見 §4.2 |
| U-11/U-13/U-15/U-17/U-19/U-28/U-29/U-30/U-31/U-32/U-33 | ✅ 見各節 |

### 待拍板（附我的建議；**2026-08-25 逐條重驗過「論文真的答不了嗎」**——結果見各行）
| # | 議題 | 重驗結果 | 我的建議＋理由 |
|---|---|---|---|
| 1 | **U-14** 訓練時圖片視角 | **證據升級（2026-08-26）**：最近的一手來源其實是 **ULIP-2 自己**，`ulip2 main.tex:612` 原文 "randomly sample its 2D rendered image **I ~ render(O)**"。再加 OpenShape method.tex:77 "randomly sample one rendered image or thumbnail"、ULIP-1 main.tex:236 每步隨機 1/60。**三個上游一致，其中一個就是我們的直接骨幹**，所以訓練側屬強 UPSTREAM FACT，不是 UNKNOWN。但**「gallery／評測用 12 視角平均」上游沒有背書**，那一半仍是我們的 CHOICE | **B**：訓練隨機單張、gallery 與評測用平均。快取已存 per-view 嵌入，換法零成本。仍列表是因為 n05b 協定已寫成 mean，改它算協定變更、要你簽 |
| 2 | **λ 初值** | **雙方已收斂（2026-08-26）**：外部審查撤回原本的 0.1 主張，與我方一致。依據：MetaFind `2meth:87` 自述殘差設計是為了 "without disrupting the original embedding space"；Flamingo `content.tex:187-189` 是最同構的先例（凍結骨幹＋新分支＋殘差＋純量閘＋**初值 0**，目的明寫為初始化時行為等同原模型），拆掉該機制掉 4.2% 且訓練不穩 | **raw λ = 0.0，不加 tanh、不加 clamp、不加 LayerNorm**。MetaFind 寫的是 learnable scalar λ，**不要把 Flamingo 的 tanh(α) 一起搬過來**——tanh 會把有效閘限制在 (-1,1)，那是更大的模型改動。最小偏離就是 `nn.Parameter(0.0)` |
| 3 | **Stage 1 選 checkpoint 的依據** | **撤回上呈，上游全部答完了（2026-08-26 二度回查）**。我上一輪說「上游有獨立 benchmark、我們沒有，這塊答不了」，**那句話同樣是沒查完就下的結論**。OpenShape `src/train.py:190-201` 的訓練迴圈：跑滿 `max_epoch`（1000）**不早停**、**每個 epoch 存一次 `latest`**、每 `save_freq`（20）存一次快照、各 benchmark 各存各的 best。關鍵是它同時追蹤 **`img_contras_acc` / `text_contras_acc`**（`train.py:124,133`），那是**訓練批次自己的對比準確率**，不碰任何 held-out 資料。ULIP 同樣跑滿不早停（`main.py:212`） | **不需要你在互斥選項裡挑，照上游做即可**：跑滿 250、不早停、每 epoch 存 latest、每 N epoch 存快照、全程記錄 in-batch contrastive accuracy。**要用哪個 checkpoint 可以事後再定**，而且不論選「訓練對比準確率最好的」或「最後一個」，**都不碰那 20% 測試集**，污染問題自動消失 |
| 4 | **U-16** 塔權重共享 | **[已撤回並修正 2026-08-26]** 我上一輪寫「standing rule 預設應是 fully_separate」，**這個推論不成立，撤回**。理由：standing rule 只適用於**論文沉默**；這裡論文並不沉默，而是**自我矛盾**——正文 `2meth:34` 寫 "separate encoders"，但 **Figure 1 圖上明確標 `ULIP-2 (Shared)`，而且整張圖只畫一個 ULIP-2 方塊**（已親眼核對 `MetaFind.drawio.png`）。另外 DPR 在 MetaFind 裡是**雙塔檢索範式的概念出處**，不是 backbone 實作 authority（不像 ULIP-2／EGNN 是實作母體），所以 DPR 用兩份 BERT 推不出 MetaFind 的 ULIP-2 必須兩份參數。"separate encoders" 在架構敘述裡也可以只表示兩個邏輯塔／介面，不必然等於參數不相交。 | **維持現行 `shared_backbone_separate_fusion` 不變**，狀態 **PAPER-AMBIGUOUS ＋ USER CHOICE**。現行選擇反而是唯一能同時滿足兩處文字的讀法：一份骨幹（對上圖的 Shared）、兩個獨立 Fusion 介面（對上正文的 separate）。fully_separate 的證據**變強但沒到推翻**，保留為競爭假設。若你之後想改，那是新決策，不是修正錯誤 |
| 5 | **U-20/U-06** t_i 與 e_ij 編碼器 | 論文明文 "e.g., CLIP or BERT"（2meth:47）——作者自己給選單不給答案，確認無解 | 同骨幹 CLIP——不引入第二顆模型、同嵌入空間 |
| 6 | **U-21** Table 2 gallery＋資產數 | **新量測**：12,000 間房實際只用 **1,528 個 unique assetId**（train 10K 房=1,036）；ProcTHOR 官方庫 1,633；MetaFind 說 "3,000+"——**三個數字互相都對不上**，任何在手論文/資料都湊不出 3,000 | 維持 procthor scope、照實報三方差異；等 GPT 有無新讀法 |
| 7 | **U-09** Table 1 gallery | MetaFind 沒說；上游評估是分類不是檢索、無協定可繼承——確認無解 | **雙協定都跑都報**（已實作，維持） |
| 8 | **Stage 2 超參數** | **原寫「S4 全無」是低估（2026-08-26 補查）**。EGNN 官方 QM9 有完整一組：`Adam` · `lr 1e-3` · `wd 1e-16` · `batch 96` · `epochs 1000`（`main_qm9.py:13,15,28,46`）。任務不同（分子回歸 vs 對比檢索）不能整包照搬，但不是沒有出處。另外 ESSGNN 的**架構**超參數更是有明確上游值，見 §3.4 對照表 | 架構值照 EGNN QM9（層數 7、pooling sum）；訓練值沿 Stage 1 的 AdamW＋cosine，lr 以 EGNN 的 1e-3 為起點小掃描，全部記 CHOICE |
| 9 | Stage 2 正例分布落差 | 無上游做 layout-aware 檢索，確認無解 | 先 leave-one-out（論文可辯護的最小讀法），落差列為已知限制 |
| 10 | 48K vs 46,832 vs 我們 46,052 | 無解（論文不解釋 48K 從哪來） | 不擋工；報告用實際語料數 |
| 11 | **Stage 1 學習率** | **四個候選，各有一手出處（2026-08-26 查全）**：`1e-3` = ULIP-1 論文 main.tex:367-370，也是 OpenShape config default；`3e-3` = ULIP 官方腳本 `pretrain_pointbert.sh` 明傳、`main.py:52` default，但那條腳本跑的是 **ULIP-1 PointBERT 8192 點**，不是我們的 10k colored；`5e-4` = **OpenShape supp:190 明說「32.3M 版 PointBERT 用 5e-4，其他模型 1e-3」**，且 ULIP-2 yaml 的 optimizer 區塊也是 5e-4（該區塊 tri-modal 訓練不讀，但數字同源）。**實測：我們載入的 PointBERT 是 32.5M**，正落在 OpenShape 特地為它調小 lr 的那個量級 | **建議 5e-4**：它是唯一「針對 PointBERT 這個 backbone、在這個參數量級」明確給出的值；其餘三個不是別的 backbone 就是別的點數規模。這是上游彼此分歧、不是沉默，standing rule 選不出來，要你拍板 |

---

## 11. 給審查者（GPT）的指令（第二輪範圍已收斂）

**第一輪結果**：3 項 P0 全數採納並修正，P1／P2 大部分採納（見下方對照）。
未採納的只有一項：外部審查建議把 U-14 的 gallery/eval 平均也視為可推導，我維持它是 CHOICE——ULIP-2 那句只講訓練取樣，沒有背書評測聚合方式。

**第二輪只需審**：(a) 本版三處 P0 修正是否真的把問題關掉；(b) §9.5 五條新登記的分類是否正確；(c) 有無新的 stronger-than-evidence 措辭。
不需要再從頭複述全文。

### 原始指令（保留）

逐節審，重點：
1. §3 的 C1-C6 裁決——你若認為 2.5 版才是作者跑的，拿出內文證據；
2. §4 配方的每一格證據標籤有無升級（PAPER↔UPSTREAM↔CHOICE 混淆即是錯）；
3. §6.1 雙協定設計、§6.3 Table 3 對應表有無漏行；
4. §10 建議預設你反對哪個——**必須附一手出處（tex 行號或官方 code 行號）**，
   部落格/論壇/記憶不算；
5. 找出本文件沒登記的論文沉默點（對照你自己讀的原文）。
回覆格式：逐條「同意／反對＋出處／補充」。
