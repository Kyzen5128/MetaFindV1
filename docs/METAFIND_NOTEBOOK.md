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

**一手來源（⚠ **2026-08-27 更正**：上一版寫「全部逐字讀完」，那比本文件自己的紀錄強一階 —— 同一份文件至少五處記載了相反的事：`:4`「只 grep 論文就宣告無源」、`:390`「漏查 repo」、`:868`「沒查完就下的結論」、§7 的 Blender 事件。正確的說法是**部分逐字、部分 grep 覆蓋**，逐字讀完的清單見各節的行號引用）**：metafind_source 541 行；ulip1 1,125；pointbert 624；
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
  （IMPL `ulip_backbone.py:138-165` prepare_depth_shell；**28** 件透明／鏡面資產 depth 拍不到 → 排除出 gallery，F26。⚠ **2026-08-27 更正：上一版寫 24，是舊值。** 實測 `procthor_modalities/*.json` 缺 `pointcloud_uri` 者 **28** 筆，`1,467 − 28 = 1,439`，與本文件 §6 的「~1,439 合格」一致；`1,467 − 24 = 1,443` 對不上。同一個過期的 24 也活在 `gallery_index.py:244/:253`，那兩處未動，屬 Engineer lane）。

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
  ＋文字特徵 t_i∈R^d（論文沒指定編碼器 → S6；~~**U-20 開放**~~ **U-20 已於 2026-08-27 由 Kyzen 拍板：OpenCLIP ViT-bigG-14，d=1280**，見本文件 §10「U-20」列。此處先前仍寫「開放」，2026-08-28 由 ULIP2 Engineer 據此誤判 n08 被未決事項擋住 —— 是本文件漂移，不是他讀錯）。
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
  "use_io_projections": null,        // ⚠ U-33 已於 2026-08-27 撤回為待決，見 §10。null = 未決，resolver 應拒絕產出
  "distance": "squared",             // U-17
  "coord_feat": "current",           // appendix 家族唯一合法值（φ_x 讀 m_ij，由 h^l 建）
  "layer_sharing": null,             // ⚠ U-31 已於 2026-08-27 撤回為待決，見 §10。null = 未決
  "pooling": "mean",                 // ⚠ 現行值，Kyzen 2026-08-19 核可。sum 是未核可的建議，見 §4.3
  "hidden_dim": null,                // ⚠ 待決。128 是 EGNN QM9 的實驗設定（Type C），且 DIM_REVIEW §7 Step 5 要求前四步先定
  "n_layers": 4,                     // ⚠ 現行值，Kyzen 2026-08-19 核可。7 是未核可的建議，見 §4.3
  "mlp_structure": "linear_silu_linear" }  // U-35：Linear→SiLU→Linear（描述程式碼，防漂移斷言）
```
＋ PRIMARY_INTERPRETATION（`essgnn.py:276-281`）：
`h0_mode=semantic / coords_agg=sum（Eq.3 是 Σ；EGNN 預設 mean，F9）/ edge_proj_dim=None / normalize_coord_diff=False`。

尚缺三個 REQUIRED 維度（建 config 時必填、來源不是論文）：
`node_feat_dim`（t_i 編碼器寬度，~~U-20 開放~~ **= 1280，U-20 已拍板 ViT-bigG-14**）、`edge_feat_dim`（e_ij 編碼器寬度，~~U-06 開放：CLIP 1280 / BERT 768 / CLIP-B 512~~ **也是 1280，且不是獨立選項**：[OBSERVED IMPLEMENTATION] `semantic_edges_run.py:247` 的 `encode_sentences()` 同時服務邊（`:378`）與節點（`:394`），共用同一個 `TEXT_ENCODER`，`:414` 的 `edge_dim = embeddings.shape[1]` 直接跟著編碼器走 —— **U-20 一次裁決同時決定兩者，U-06 沒有獨立的選擇空間**。仍然開放的是另一件事：ESSGNN 內部要不要把 1280 投影到更窄，那是模型內的一層，**不需要重跑 n08**）、
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
🔴 **2026-08-27 二度更正：`n_layers = 7` 與 `pooling = sum` 兩者都查無核可，兩條都撤回。**
現況：協定檔 `essgnn_arch_protocol.json` 是 **`n_layers = 4`、`pooling = "mean"`**，
`decided_by = "Kyzen (2026-08-19, C1 決定後補寫)"`。
`DECISION_LEDGER.md` 與 `C_PAPER_CONTRADICTIONS.md` **對這兩項都沒有任何決定條目** ——
只有列舉「協定有哪些 key」的地方提到它們的名字。
**7 與 sum 都是我從 EGNN QM9 提的建議，我後來把自己的建議寫成了他的核可。**
兩者都要 Kyzen 拍板，或先做短跑比較再上呈。
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
   `resolve_stage1.py` 現在是 `epochs: 5` ＋ `max_epochs: 250`（Kyzen 2026-08-27 選的階梯／上限拆分）。⚠ 上一版寫「目前的 50」已過期；結論不變，pilot 從小開始是對的。

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
   三條分別經 Kyzen 逐項裁決（見 §4.3 台帳）：不設自動早停 ✅ · 取 best ✅（但依據改為
   **開發期在 80% 內部的 dev-val**，因為我們沒有上游那個獨立下游 benchmark） ·
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

**檢查點選擇（2026-08-27 Kyzen 拍板，DEVIATION D-3，同日修正）**：
**開發期**在 80% training pool 內部切 dev-val，用它的 Mean R@1（跨模態平均，平手用 Mean R@5）
定下 lr、輪數與 checkpoint 政策；**正式期**設定鎖死、從頭重訓完整 80%、不中途挑 checkpoint、
最後才第一次打開 20% test 考一次。**20% test 全程不參與任何選擇。**
（上一版寫「以測試集 R@1 選 best」是漏刪的殘句；再上一版寫「切 10% val 但 gallery 維持整個 20%」
則有 transductive contamination，兩者都已撤回，見 §4.3 的 D-3 全文。）
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

**DEVIATION D-3 — 資料切分與 checkpoint 選擇（2026-08-27，Kyzen 拍板；同日經外部審查修正）**

論文 `3experiments.tex:8` 的全部原話：

> "we allocate **80%** of the data for training and reserve the remaining **20%** for testing"

**兩份，沒有 validation，而且對「怎麼挑 checkpoint」一字未提。**

**⚠ 本節第一版已作廢。** 它寫「把保留的 20% 對半切成 val/test，gallery 維持整個 20%，
val 的資產只當幹擾項所以不構成洩漏」。**那句「不構成洩漏」不成立**，外部審查指出，我查證後接受：

```
val 的查詢要從 9,211 個候選裡被找出來，其中約一半是未來的 test 資產
換一個 checkpoint → 那些 test 資產的向量跟著變 → val 分數跟著變
而 checkpoint 是用 val 分數挑的
→ test 資產參與了 model selection
```

沒有用到 test 的**答案**，但用了它們當**幹擾項**。這是 transductive contamination，
違反 Rule 10（論文指定的 test set 不得用於 model selection）。

**現行做法（Kyzen 核可）：開發期與正式期分開。**

```
開發期
  從論文的 80% training pool 內部再切 dev-val
  20% test 完全封存，一次都不打開
  用 dev-val 決定：lr · 訓練輪數 · checkpoint 政策

正式期
  設定全部鎖死，從頭重訓一次，吃完整 80%
  正式那一跑不中途挑 checkpoint
  最後第一次打開 20%：query = 20%，gallery = 20%，考一次
```

**三個量都與論文一致**：訓練 80% · gallery 20% · test 全程未參與選擇。

**代價**：正式那一跑沒有 validation 支撐的 checkpoint 選擇，輪數在開發期就定死。
與已核可的「不設自動早停、跑固定輪數」相容。

**正式期仍然可以存 checkpoint**（crash recovery、稽核、除錯），
但正式分數只能用**事前已指定**的 checkpoint 規則。
跑完 test 之後說「epoch 170 比 180 好，改報 170」＝ 重新污染。

**已知風險（外部審查提出，尚未解決）**：開發期在 **70%**（`0.125 × 0.80 = 0.10` 切走，
46,052 → dev_train 32,236 / dev_val 4,605 / test 9,211）上定的最佳輪數，
⚠ 上一版寫 72%，那是 `DEV_VAL_FRACTION = 0.1` 時的數字，Kyzen 已核可 0.125。
搬到完整 80% 上不保證仍是最佳 —— 資料變多，每個 epoch 的 step 數變多，
overfitting 時點也可能後移，而**方向不保證**。
建議在 80% 內部做多個 dev fold，取一個穩定的 training-duration 規則，
而不是相信單一次 split 的最佳輪數。**全程不碰 test。**

**IMPL 狀態（2026-08-27 更新）**：`metafind/data/splits.py:146` 已有 `split_dev()`，
`DEV_VAL_FRACTION = 0.125`（`:95`，Kyzen 2026-08-27 核可）。⚠ **上一版寫「尚未實作」已過期。**

⚠ **本節之後不再複述程式碼的當下狀態。** 這份筆記的用途是記錄**決定**，
決定不會過期，程式碼會。要看現值請直接開 `splits.py`。

---



| 參數 | 值 | 等級 | 出處 |
|---|---|---|---|
| τ | 0.5 固定 | PAPER | 3experiments.tex:15 "The temperature is 0.5 for all experiments" |
| τ 可學性 | 否 | CHOICE（USER 批准） | 論文把 f_h/f_x/λ 叫 learnable、τ 兩處叫 hyperparameter（詞彙對比推論，`resolve_stage1.py` C-001） |
| 模態遮罩 | 30%，模態間獨立 | **PAPER（模態獨立）＋CHOICE（樣本獨立）** | 2meth:75 明寫 "each modality ... independently masked"，模態間獨立是 PAPER；「不同樣本的遮罩也獨立」論文沒寫，是實作選擇（2026-08-26 拆級） |
| scene dropout | 30% per-**batch** | **PAPER，含粒度（2026-08-26 升級）** | 2meth:89 原句就是 "omitted in 30% of **batches**"，30% 與 batch 粒度都有直接文字支持；仍屬 CHOICE 的只有 RNG 實作細節 |
| split | 80/20 兩資料集 | PAPER | 3experiments.tex:8 |
| batch / optimizer | 64 / AdamW | UPSTREAM | ulip1 main.tex:367-370；`main.py:50` default 亦 64 |
| **lr** | **5e-4 起跑** | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27） | 上游四個候選：1e-3（ulip1:367-370）· 3e-3（`pretrain_pointbert.sh`，但那是 ULIP-1 8192 點）· 5e-4（OpenShape supp:190 指定給 32.3M PointBERT；Point-BERT 論文 `:216` 亦同）· 4e-4（supp:190，72.1M 版，同節 `:146` 說該版嚴重過擬合）。**Kyzen 定 5e-4 起跑**，第一輪 sweep 2.5e-4 / 5e-4 / 7.5e-4 / 1e-3，3e-3 不入第一輪。**不得用那 20% test 調 lr**（⚠ 上一版寫「10% test」，那個切法已於同節撤回，見 D-3） |
| wd / betas / eps / warmup | 0.1 / (0.9,0.98) / 1e-8 / 1 ep cosine | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27），上游參考 `upstream/ULIP/main.py` | Kyzen 的理由：MetaFind 未交代訓練配方，Stage 1 骨幹 lineage 來自 ULIP-2，第一版先用 ULIP 成熟配方，避免一次動太多變因。pilot 顯示不穩仍可調 |
| **weight decay 分組** | `p.ndim < 2` 或名稱含 `bias`／`ln`／`bn` → **wd = 0**；其餘 → **wd = 0.1** | **USER-APPROVED**（2026-08-27），上游**機制**照抄 `main.py:129-135` | Kyzen 2026-08-27：「找得到原架構怎麼設定的，就照原架構」。這是**機制**不是數值，故可直接落地。一律 0.1 會把 LayerNorm 的縮放參數往 0 拉，破壞正規化 |
| **cosine 起訖 lr** | `lr_start = 1e-6` · `lr_end = 1e-5` | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27），上游參考 `main.py:53-56` | ⚠ **這兩個是數值不是機制**，不適用「照原架構」自動落地（外部審查指出，已修正）。Kyzen 2026-08-27 明確選甲：照 ULIP，理由是與 base lr 5e-4 同一套配方，不混搭。**warmup + cosine 的形狀**才是機制，那部分本來就可繼承 |
| **D_m（訊息寬度）** | `D_m = D_h` | **USER-APPROVED**（2026-08-27），上游**機制**照抄 `egnn/models/gcl.py:165-168` | EGNN 的 `edge_mlp` 輸出就是 `hidden_nf`。論文沒要求 `D_m = D_h`（`ESSGNN_DIM_REVIEW.md` v3 §3 已拆開五個獨立寬度），所以這是照原架構機制，**不是 PAPER FACT** |
| epochs | **250（上限）** | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27），上游參考 ULIP-1 論文 `main.tex:367-370` | Kyzen 原話：「250 啊 因為 ULIP 有說喔 我覺得就參考啊」「要測試所以慢慢加」。**先跑 5 → 10 → 25 pilot** 確認 loss／梯度／記憶體／管線正常，再進完整訓練。250 是上限，不是無條件跑滿。**`main.py:47` 的 argparse 預設是 Type D，不能單獨當根據** |
| 早停 | **不設自動早停** | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27） | Kyzen 的理由：現在就設 patience=20 等於偷偷先決定了 model selection 協定。`early_stopping=False`，**保留依實際 validation 曲線人工中止的權利**。上游參考 `main.py:212`（跑滿，無 early-stop 分支）。**那 20% test 不得參與任何中止判斷**（⚠ 上一版寫「10% test」，同上） |
| checkpoint 選擇 | **開發期用 80% 內部的 dev-val 定政策；正式期鎖死重訓、不挑** | **USER-APPROVED IMPLEMENTATION CHOICE**（2026-08-27，DEVIATION D-3） | 上游有 best-checkpoint 做法（`main.py:225-231`），但它拿的是**獨立下游 benchmark**（`main.py:40` default modelnet40），我們沒有。改為全部在 80% 內部完成選擇。**20% test 全程封存。IMPL 尚未實作** |
| seed | 20260816 | CHOICE | resolve_stage1.py |
| λ 初值 | 未定（code 佔位 1.0） | **UNKNOWN→待決** | §10 #2 |
| fusion transformer 尺寸 | 2 層/8 頭/ffn 2048 | CHOICE | 論文無維度 |
| ESSGNN hidden | **未決** | **待決，排在四項之後** | 128 出自 EGNN 論文 `appendix.tex:135` 的 **QM9 實作細節節** → **EGNN EXPERIMENT-SPECIFIC SETTING**，不能單靠 lineage 落地。且 QM9 的 15 維原子 one-hot → 128 是**放大**，我們 1280 → 128 是**壓縮**，同一個數字方向相反。順序見 `docs/ESSGNN_DIM_REVIEW.md` §7 |
| ESSGNN 層數 | ⚠ **未決。現行協定是 `4`。** | 🔴 **2026-08-27 更正：上一版標 USER-APPROVED 是錯的，查無此決定。**與 pooling 同一個錯：`DECISION_LEDGER.md`／`C_PAPER_CONTRADICTIONS.md` 均無 `n_layers` 決定條目；協定檔 `n_layers = 4`，`decided_by = Kyzen (2026-08-19)`。**7 是我從 EGNN QM9 `main_qm9.py:34` 提的建議**（Type C，別的任務的實驗設定），需 Kyzen 核可才能落地 | 7 出自 `main_qm9.py:34` 與論文 `appendix.tex:135`，屬 **EGNN EXPERIMENT-SPECIFIC SETTING**。注意 `qm9/models.py:46` 的 class 建構子預設**也是 4**，是 QM9 腳本顯式覆寫成 7 —— 更證明 7 是實驗設定不是 intrinsic 架構。原文寫「4 是 N-body 的值」不完整，已更正 |
| ESSGNN pooling | ⚠ **未決。現行協定是 `mean`。** | 🔴 **2026-08-27 更正：上一版標「USER-APPROVED（2026-08-17 定 sum）」是錯的，查無此決定。** | **查證（ESSGNN REVIEWER 2026-08-27 提出，我複驗）**：`DECISION_LEDGER.md` 與 `C_PAPER_CONTRADICTIONS.md` **沒有任何一條 08-17 的 pooling 決定**；magnetic `essgnn_arch_protocol.json` 的 `decided_by` 是 **`Kyzen (2026-08-19, C1 決定後補寫)`**、`pooling` 是 **`mean`**。**08-17 是 U-26（架構家族）的日期，我把它安到 pooling 頭上了。**<br>**⚠ 而我第一次撤回時寫的「`mean` 是 Kyzen 08-19 核可的現行值」，也是過度解讀同一個戳。** ESSGNN ENGINEER 讀那個檔案時發現：**`decided_by` 是整份檔案的一個頂層欄位，不是逐 key 的**，它同時蓋住五個參數（`n_layers` / `pooling` / `hidden_dim` / `use_io_projections` / `layer_sharing`）。**我對其中兩個當成核可、對另外三個當成待決，而我給不出理由。**<br>**這變成一個必須由 Kyzen 回答的問題，見 §10 新增條目。**<br>`sum` 是我 08-26 從 EGNN QM9 提的**建議**，該建議的理由已於 08-27 撤回（EGNN 從未說過「因為目標外延所以 sum」，且 QM9 亦含 HOMO/LUMO/gap/μ 等非外延目標）。**我把自己的建議標成他的核可 —— 那正是 Rule 16 禁止的無聲晉升，而且是我犯的。**<br>論文只寫 `Pooling({h_i^(L)})`，**未命名** → PAPER AMBIGUITY。工程取捨仍成立：sum 會讓 ‖e_layout‖ 隨物件數變動而單一 λ 補不回；mean 則弱化物件數資訊。**現行值維持 `mean`，改成 sum 需要 Kyzen 核可，或先做短跑比較再上呈。** |
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
- MetaFind 預期型態（PAPER Table 1）：單塔 baseline 的 PC-only 98%+ 來自 query 與 gallery 用同一支嵌入，論文自己的用詞是 **"leading to inflated accuracy"**（`3exp:24`）。**PAPER Table 1（`3exp:45,46`）：MetaFind 自報的雙塔 PC-only 是 75.1（w/o ESSGNN）／ 63.2（w/ ESSGNN）。**
  ⚠ **2026-08-27 更正**：上一版寫「**我們**雙塔 PC-only 是 75.1／63.2」。
  **那是論文自己的數字，不是我們的。** 而且我們**從未訓練過 Stage 1** ——
  實測 `checkpoints` 0 個檔、`embeddings` 0 個檔、`run_progress.jsonl` 的 n10 紀錄 0 筆。
  **「我們」在那句話裡不可能指涉任何測量，因為沒有測量存在。**
  同一句話的結尾在警告不要寫得比出處強，而那句話本身把別人的成績掛了我們的名字。照論文用語寫 inflated 即可，不要寫成「別人灌水、我們誠實」，那是價值判斷、比出處強。
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
| GPT-4o | — | 論文指定的標註器（n05）與場景評審（n15/n17） | **標註側是 DEVIATION D-2**：實跑 `gemma-4-12B-it`，非 GPT-4o（見 §2.3）。評審側 n17 尚未執行，模型未定（`Q-JUDGE-MODEL` 開放）。prompt 一律是我們的 CHOICE（contract 鎖定） |
| I-Design | 2404.02838 | Table 2 管線 | retrieve.py 替換點、評分模板 |
| DPR | 2004.04906 | 雙塔範式概念源 | 概念引用而已 |
| Flamingo | 2204.14198 | λ 初值辯論先例 | tanh(α) α=0、拆掉掉 4.2%＋不穩 |
| GAT | 1710.10903 | 被否定的 baseline | Table 3 GAT 行需要它 |
| SCA3D/Uni3D(L)/OmniBind/PointCLIP | — | Table 1 baselines | 各自 released checkpoint 評估 |

**⚠ 舊 standing rule 已於 2026-08-26 被 Kyzen 整份改寫取代。** 舊文如下，保留供對照，**不再有效**：

> ~~MetaFind 沉默 → 預設照上游官方（論文＋程式碼），記 UPSTREAM；上游也沉默才上呈。~~

**現行規則（`docs/_rules_preamble.md`，commit 9218616）—— 上游是必查的證據來源，不是自動的決策來源：**

| 型 | 內容 | 繼承權限 |
|---|---|---|
| **A** | 架構／數學／模組機制 | 可作首選重建候選，標 UPSTREAM FACT，**絕不標 PAPER FACT** |
| **B** | 數值超參／訓練配方／checkpoint 政策／early stopping | **只能 UPSTREAM CANDIDATE，需 Kyzen 逐項核可才入協定** |
| **C** | 上游「別的任務」的實驗設定（EGNN 在 QM9 用 7 層） | 權威性最弱 |
| **D** | argparse／函式庫預設 | **永遠不能單獨解決 MetaFind 沉默** |

Rule 0：`Evidence discovery is NOT decision authority.`
Rule 16：進入官方協定只有三條路 —— MetaFind PAPER FACT ／ Kyzen 明確核可 ／
既有 ledger 條目載明該**具體參數**。UPSTREAM FACT 本身不夠。

**Kyzen 2026-08-27 補充**：「找得到原架構怎麼設定的，就照原架構」。
適用於**機制**（weight decay 分組、warmup 形狀、`D_m = D_h`），
不適用於**數值**（lr、輪數 —— 上游彼此打架且隨任務而異，仍需核可）。

**「必須去查上游」這一半沒有變，而且是強制的。** 反例教訓 = Blender 事件
（沒查 OpenShape 渲染器就用 pyrender，全語料重渲一次）。

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
場景鏈： n07(ProcTHOR 房→場景圖)→n07b(每個資產的隔離渲染／depth shell／文字)
        →n08(語意邊)→n09b/c(Stage2協定+scene split)→n11b(Stage2 gallery index，
        用凍結的 Stage 1 gallery 塔編碼)→n13(Stage 2 訓練)→n14(等變性檢驗)
        →n15a/b/c(評估場景協定與準備)→n16(Algorithm 1 場景組合)→n17(模型評審)→Table 2

⚠ **2026-08-27 整條更正。** 上一版從 n11b 之後**每一個節點都標錯**，而且是整體位移：
寫成 `n11b(Stage2訓練)→n13(檢索)→n14(I-Design)→n15/16(評分)`。
逐條對照 `docs/graph/node_registry.yaml` 的 `purpose`（我用 yaml.safe_load 讀出來的）：

| 上一版寫 | 實際是 |
|---|---|
| n11b = Stage 2 訓練 | **Stage 2 gallery index** —— 用**凍結的** Stage 1 gallery 塔編碼 ProcTHOR 資產 |
| n13 = 檢索 | **Stage 2 訓練**（fuser 與 ESSGNN 的 layout-aware 微調） |
| n14 = I-Design | **等變性檢驗**（三個層級的 SE(3) 行為） |
| n15/16 = 評分 | n15 = Table 1 檢索評估；n15a/b/c = 評估場景協定與準備；**n16 = Algorithm 1 場景組合** |
| n07b = 場景圖 + depth shell | **場景圖是 n07 寫的**；n07b 是「每個資產的隔離渲染／depth shell／文字」 |

**這一節是「Repo 對應表」，是給人查表用的。照上一版排工作順序，會把訓練排在建索引之前。**
（ESSGNN ENGINEER 2026-08-27 逐條讀 node_registry.yaml 後提出，我複驗確認。）
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

## 9.6 「協定寫了，生效的那條路沒讀」—— 2026-08-27 一天內第四例

**這是本專案目前最會產生沉默錯誤的一個形狀，四個實例的機制完全相同：**

| # | 協定寫在哪 | 生效的路徑讀的是什麼 | 狀態 |
|---|---|---|---|
| 1 | `fusion.py:89` 已改 `transformer` | `stage1.py:322` 與 `stage1_config.py:367` 都讀 `training_protocol["fusion"]`，來源是 `splits.py:75` = **`masked_mlp`** | 🔴 **已發生**。修在不生效的檔上 |
| 2 | `image_aggregation` 可設 `random_single_view` | `stage1.py:103` 存了 `self.aggregation` 卻**從未再讀**；`:110-121` 直接讀 `cached["image"]`（已聚合），`z["views"]` 無人碰 | ⚠ **地雷已埋**，改值的那一刻踩到 |
| 3 | `tower_sharing="fully_separate"` 定義為兩個骨幹（`dual_tower.py:60`） | `stage1.py:368` 只建一個，且不驗證數量 | ⚠ 設下去照跑，跑出來是 `shared_backbone_separate_fusion`（INTEGRATOR 提，UNVERIFIED） |
| 4 | `Stage1RuntimeConfig.from_protocols` 做協定驗證 | 只有 `tests/` 呼叫；訓練器 `stage1.py:309` 直讀原始協定字典 | ⚠ **所有協定驗證在真實訓練路徑上都不執行**（INTEGRATOR 提，UNVERIFIED） |

**共同形狀**：協定層有一個看起來被消費的欄位，執行層有另一條路。
**兩邊都不會報錯**，因為沒有任何東西比對它們。

**諷刺的是 `stage1_config.py:8-20` 的 docstring 教的正好是這個陷阱**：

> A trainer that writes `FusionConfig(dim=d)` gets `masked_mlp` whatever the
> protocol resolved to, and nothing downstream notices.

**寫下這句話的檔案，正是 #1 發生的地方。** 規則不會自己跑到手邊。

**驗證方法**（下次遇到同型時直接用）：不要問「協定寫了什麼」，
要從**執行入口**往回追到那個值第一次被讀取的地方，並確認中間沒有第二條路。
`grep -rn '<常數名>' metafind/ --include=*.py` 若只回到定義處與註解，
那個欄位就是死的。

---

## 9.7 儀器解析度不足，而輸出看起來完全正常（2026-08-27 新增）

**這是 §9.6 的近親，不是同族，而且區分它有實際後果。**

```
§9.6   兩條路徑不一致 —— 協定寫了 X，生效的那條路讀 Y
       抓法：從執行入口往回追到值第一次被讀的地方

§9.7   只有一條路徑，而它的解析度不夠
       輸出完整、沒有被截斷、沒有被別的路徑繞過 —— 它就是分不出來，而且不會說
       **§9.6 的抓法對這一族完全無效**
```

**抓法**：用同一個輸入跑兩次（換種子／換取樣），**把儀器自己的散布報在結果旁邊**。
比自噪更近的差異，一律不准聲稱分得出來。

**實例（同一天，同一個問題，兩個錯的數字）：**

```
1  點數 ＋ bbox 當幾何指紋 → 天花板 0.90
   bbox 取到小數三位，1mm 差異就算不同幾何 → **在噪音上過度分割 → 高估**

2  D2 描述子但只取 6,000 對點 → 「同 mesh 與不同 mesh 重疊」
   自噪 median 0.124 > 不同 mesh 的最小距離 0.115
   → **整個分布差異被自己的噪音蓋住 → 一度誤判描述子不可用**

修正後  D2 @ 100,000 對點，閾值取自噪 max 0.0518，complete-link → 0.709，區間 [0.638, 0.709]
```

### §9.7 第三個實例：取樣點不夠密，而輸出看起來完全正常（2026-08-28，MASTER 自己犯的）

ULIP2 Engineer 固定排除三個測試檔（`test_renders` / `test_annotate` / `test_cuda_smoke`），
理由是「會開 Blender 撞 GPU」。我去驗，得到：

```
200 passed in 7.89s     nvidia-smi 44 MiB     pgrep blender → 無
```

**我據此裁定「它們 mock 掉 subprocess、不碰 GPU」，並要他恢復。結論對，理由整個是錯的。**

他反駁時附了**跑的期間每秒取樣**的資料。我自己重做（0.25 秒一次，跑滿整個視窗）：

```
[OBSERVED DATA 2026-08-28，MASTER，每 0.25 秒取樣]
  跑前         44 MiB
  期間峰值   **3,762 MiB**
  期間 blender 程序數峰值  **2**
  200 passed in 7.85s
```

**它們真的開了兩個 Blender、真的吃了 3.7 GB。7.8 秒跑得完，是因為 Blender 開得快、
渲染的東西小，不是因為它沒被開。**

> **「跑前跑後都是 44 MiB」與「全程 44 MiB」是兩個不同的命題。**（ULIP2 Engineer 措辭）
> 一個 7.8 秒的視窗，前後各採一點，中間的峰值不會出現在任何一次讀數裡。
> **而輸出完全正常**：測試綠、時間短、兩次讀數一致 —— 沒有任何東西提示我漏了東西。

**裁決不變，但支持它的事實換掉了**：規則要綁 GPU 餘裕，不綁檔名 ——
不是因為那三個檔不碰 GPU，**是因為它們需要約 3.8 GB 餘裕**
（卡空 44/32,607 MiB → 無風險；n05 佔 29,751 MiB 時 → 3.8 GB 進不去 → OOM，
那正是他當初撞到的）。**綁檔名的話那三個檔壞了沒人會知道；
而若筆記寫成「它們不碰 GPU」，下一個人會在 n05 那種情境下照著撞上去。**

⚠ 實作上他建議**不要**加 conftest 自動 skip，我採用：
**一個 skip 是綠的**，會讓「因為卡忙沒跑」和「因為壞了沒跑」在輸出裡長得一樣。

---

> **先量儀器，再量對象。** 兩次錯都是因為沒做這一步。
> 而第二次特別危險：它會讓人得出「這個方法不行」，然後退回第一個錯的方法。

---

## 9.11 兩個訓練器都會 NameError，而全套測試是綠的（2026-08-28 新增）

**發現者：ULIP2 Engineer，起因是 Kyzen 要他「把 Stage 1 全部步驟寫好送審才准跑」並與本文件逐項比對。
我獨立驗證兩處。**

### 兩處 CONFIRMED

```
[OBSERVED IMPLEMENTATION，我 grep HEAD 版本]
metafind/train/stage1.py:593   lr=round(sched.get_last_lr()[0], 8)
metafind/train/stage1.py:595   for p in params
  → `sched` 全檔僅此一處出現；`params` 在 main() 內僅此一處
    （:95/:96 是 dict key，:514 是別的函式的區域變數）
  → **第一個 epoch 的第 20 步 NameError。** 成因：8/27 把 torch scheduler
    換成上游的預先算好陣列時，刪了 scheduler 與 params，留著引用它們的 log 行。

metafind/train/stage2.py:570   load_stage1_checkpoint(...)
metafind/train/stage2.py:504   from metafind.train.stage1 import build_model, load_protocols
  → **沒有 import 它。** 對照 gallery_index.py:174 有。
  → Stage 2 一走到載入 checkpoint 那行就 NameError。
```

**兩個訓練器，兩個都跑不動，而 `pytest` 全綠。**

### 為什麼綠

- Stage 1：那 13 個測試測的是 `cosine_schedule` 與 `weight_decay_groups` 兩個**純函式**，從未進過訓練迴圈。迴圈要 9.5 GB 骨幹＋GPU 才進得去。
- Stage 2：`stage2.py:5` 自己寫著「it has never been executed, because it needs `stage1_ckpt`」。**從來沒跑到那一行。**

### 修法值得記下來：補的是規則，不是那個 case

ULIP2 沒有只補一個測試。他加了 `tests/test_train_stage1.py` 的 AST 走訪：
**訓練模組裡任何函式都不得讀取「非參數、非本函式賦值、非模組層級、非 builtin」的名字。**
靜態檢查，因為它保護的迴圈進不去。**把 `sched` 那行種回去 → 紅並指名；拿掉 → 綠。474 passed。**
他用它掃八個檔，只有 stage1（已修）與 stage2（ESSGNN 線）中招。

### 續：那支規則測試本身還沒被完整校準（2026-08-28 稍晚）

ESSGNN Engineer 想把這條規則推廣，**自寫了一支 AST 掃描器，掃出 74 個「未定義名稱」。
他抽驗三個，三個全是掃描器自己的 bug**：`ast.AnnAssign`（帶型別註記的模組常數，
如 `fusion.py:57 MODALITIES: tuple[str, ...] = (...)`）沒收、巢狀函式的參數被算到外層
（`scene_graphs.py:139 def visit(obj, parent)`）、global/nonlocal 沒處理。
**74 個裡真的只有一個 —— 就是 `stage2.py:570`，而那是用 grep 確認的，不是用掃描器。**
他自己的判定：**「一天前才被逐行審過的 `fusion.py` 亮六個，那就是儀器壞了的訊號」**，
並歸為他今天第三次沒先量儀器（前兩次：點數＋bbox 指紋、D2 @ 6,000 對點）。

⚠ **但雙方提出的校準都只測了一半。** 他提的「餵一個已知乾淨的模組，確認是綠的」
只測**假陽性**。ULIP2 確實種了 `sched` 回去驗證會紅 —— **但那是在 `stage1.py`，
也就是那支檢查為它而寫的檔。** 「八個檔只中兩個」目前有兩種讀法無法區分：
判準準確，或**判準太窄**。74 vs 2 只排除了太寬。

**因此套用到 `stage2.py` 之前要跑兩步**：
(1) 假陽性 —— 乾淨模組必綠；
(2) 假陰性 —— **在那個乾淨模組**（不是 `stage1.py`）裡種一個同型 bug，必須紅並指名，拿掉必須綠。
**第 2 步紅不起來就回報、不要套用**，那表示它只在出生地有效。

**校準結果（ESSGNN Engineer 執行，2026-08-28，唯讀）**：拿 ULIP2 那支檢查的邏輯掃他這邊七個檔 ——
`stage2.py` 1 個（`main() line 570: load_stage1_checkpoint`，正是 grep 已獨立確認的那個），
`gallery_index.py` / `semantic_edges_run.py` / `resolve_stage2.py` / `essgnn.py` /
`scene_graphs.py` / `fusion.py` 全部 0。
**兩步都過，而且假陰性這一步比我設計的更強**：不是種一個假 bug，
是它在一個**不是為它而寫的檔**上抓到一個真的、已被獨立確認的 bug。**推廣性成立。**
（對照他自己那支掃描器在同一批檔上：`fusion.py` 6 個、`scene_graphs.py` 5 個。同樣的檔，他的 11、ULIP2 的 0。）

🔴 **但校準過程中查出第三件，而且是我們三個都沒想到的**：

```
tests/test_train_stage1.py:472
    src = (repo / "metafind" / "train" / "stage1.py").read_text()
    ← **路徑硬寫死。committed 的那支測試只掃這一個檔。**
```

**所以「他用那個檢查掃了八個檔，只有 stage1 與 stage2 中招」——
committed 的測試做不到這件事。那個數字來自一次沒有留下的臨時執行。**
**與本節的 F8 三處數字完全同型：一個結論，而產生它的程式碼不會產生它。**

→ **判準沒問題（剛驗過），問題在覆蓋範圍，而覆蓋範圍是一行路徑。**
`stage2.py` 那個 NameError 到今天才被發現，不是因為沒人寫檢查，**是因為檢查只看一個檔**。
→ 修法因此比想像的簡單：**把那行硬寫死改成模組清單**（`metafind/train/*.py` 與 `metafind/models/*.py`），
**這是擴大既有檢查的覆蓋範圍，不是新寫檢查。**
⚠ 擴大後 `metafind/data/*.py` 仍未涵蓋 —— 其中只有兩個檔驗過乾淨，**其餘不得視為已保證。**

> **ESSGNN Engineer 的收尾，採用**：今天最有效的那個守衛之所以有效，
> 不是因為署名，**是因為它被寫成「檢查一條規則」而不是「檢查一個案例」**——
> 它自己的註解逐字：「Adding a fourteenth test of the same shape would not have caught it either.」
> **它唯一的缺陷不在判準，在覆蓋範圍。**

> **兩句要一起記住**（ESSGNN Engineer 前句、我後句）：
> **「自己臨時寫的守衛更要驗，因為沒有人會來驗它。」**
> 今天我給他的兩道守衛（F8 方向、frozen scope 檢查）都是我臨時寫的，兩道都錯，
> 兩道都是他去讀程式碼才發現。差別只在我的有署名所以他會去讀 ——
> **署名不是品質，是可追溯性。**

> **與 §9.8 的關係**：§9.8 是「防線是綠的但它量錯」；這一條是「**沒有防線，而全綠讓人以為有**」。
> 兩者的共同點是綠燈本身不帶資訊。差別在 §9.8 要執行防線去對它自己的宣稱，
> 這一條要問「**這條路徑上到底有沒有任何測試進得去**」。

---

## 9.12 決定了、記進本文件了、但沒有任何程式碼消費它（2026-08-28 新增）

ULIP2 拿本文件逐項比對實作時抓出兩個。**兩個都不是 bug，是「決定與實作之間斷線」，
而且斷線的兩端都在本文件裡看起來是完整的。**

**(a) dev-val 沒有消費者。**
本文件 `:517` 逐字寫著「用 dev-val 決定：lr · 訓練輪數 · checkpoint 政策」。
`splits.py` 8/27 已做出 `C_dev_selection` 協定與 4,602 筆 dev-val（`DEV_VAL_FRACTION = 0.125`）。
[OBSERVED IMPLEMENTATION] **`grep dev_val metafind/train/stage1.py` = 0。訓練迴圈裡沒有任何驗證步驟。**
→ **pilot 跑完，沒有東西可以拿來決定 lr 或輪數，只有 train loss。**

**(b) checkpoint 選擇沒有實作。**
本文件 v3.2 記著 OpenShape `src/train.py:190-201` 的 in-batch contrastive accuracy
可在沒有 held-out 的情況下選 checkpoint，並據此**撤回了 §10 #3 的上呈**。
[OBSERVED IMPLEMENTATION] `stage1.py:621` 每個 epoch 無條件 `save_checkpoint`，
全檔 grep `best` 零命中。**不早停是對的（`for epoch in range(epochs)` 跑滿），
但「取 best checkpoint」不存在。**

**兩件都需要 Kyzen 裁決，不由工程師自行補**：要用哪個指標選模型、要不要每 epoch 跑 dev-val，
是研究行為（`acc_q2g` 已經在 metrics 裡算了，但「拿它當選擇依據」是一個決定）。
**已列入 §10 待拍板。**

> **這一族的形狀**：§9.6 是「協定寫了、生效的路沒讀」；這一條是「**本文件寫了、根本沒有那條路**」。
> 前者 grep 得到一個沒被讀的欄位，後者 grep 得到零 —— **而零看起來就像沒有這個議題。**

---

## 9.10 兩個文字空間，靜靜地不同 —— U-34 若翻案，沒有任何東西會發現（2026-08-28 新增）

**發現者：ESSGNN Engineer。他更正的是我。我逐條驗過，他對，而且他的版本比我的更糟也更準。**

### 我說錯的

我對他說：「U-34 若從 frozen 翻成 trainable，Stage 1 會動到文字塔，
你的 n08 產物（4,242 邊 + 1,467 節點）全部作廢」，並要他在 n08 開跑前
檢查 `stage1_encoding_protocol.json` 的 `actual_clip_train_scope` 是否仍是 `frozen`。

**兩句都錯，而且錯在同一個地方：n08 根本不讀 Stage 1 的任何東西。**

[OBSERVED IMPLEMENTATION]

```
semantic_edges_run.py:322  ULIPBackbone(BackboneConfig(device="cuda",
                                        train_scope="fuser_only"))   ← 沒傳 checkpoint
ulip_backbone.py:100       checkpoint: Path = DEFAULT_CKPT
ulip_backbone.py:87        DEFAULT_CKPT = paths.ULIP2_CKPT           ← 官方釋出的 ULIP-2 權重
stage1.py:64               CKPT_PATH = paths.CHECKPOINTS / "stage1.pt"   ← 另一個檔
stage1.py:262              「Only what the optimizer moves.」          ← 只存 optimizer 動過的
```

→ **n08 與 n06 都讀 `ULIP2_CKPT`。「一個文字空間」目前是由建構保證的，不是由紀律保證的。**
這比我給的理由（「Stage 1 沒去動它」）強一級。我的前置檢查則完全無效：
即使那個欄位翻成 `trainable`，n08 也會若無其事地跑完，因為它不看那個檔。

### 真正的失效方式，形狀完全不同

```
U-34 = frozen（現在）
  文字權重只在 ULIP2_CKPT，stage1.pt 裡沒有
  檢索空間 與 t_i/e_ij = 同一份權重                      ✅

U-34 → trainable
  Stage 1 更新文字塔 → 它進入 stage1.pt
  檢索空間  = 訓練「後」的文字空間
  n08 仍讀 ULIP2_CKPT → t_i/e_ij = 訓練「前」的文字空間
  → 兩個文字空間，維度相同、名稱相同、沒有任何檢查會發現
```

**不是「要重跑」，是「不用重跑，而且錯了也不會有人知道」。**
重跑至少是個看得見的動作；這個沒有動作、沒有錯誤、沒有訊號。

**而 U-20 的核可理由逐字是「同一個專案不要兩套文字理解」——
U-34 一翻就有兩套，而 U-20 的機器化執行點是零（§10 的 U-20 條目：兩個維度都還沒進協定）。**

### 缺口在哪：n08 記名稱，gallery 記指紋

```
sem_edge_cache 的 provenance（semantic_edges_run.py:473-475）
  llm_model · text_encoder · text_encoder_version · prompt_version · edge_dim
  → **五個都是名稱或數字，沒有一個能分辨「訓練前」與「訓練後」的同名權重**

gallery index 那邊已經有機制
  gallery_index.py:70  gallery_encoder_sha256(backbone, model)
  → **n08 沒有這個，而它應該有同一個**
```

### 要做的（併入 §10 U-20 的第 ④ 項，同一個 raise 點）

1. `sem_edge_cache` 的 provenance 記下 **n08 實際編碼所用 checkpoint 的身分**（路徑＋sha256），不是只記名稱。
2. n13 啟動時比對：**n08 產物的 checkpoint 指紋 == gallery index 的 encoder 指紋**，不同就 raise。

**分類**：ESSGNN 節點的缺陷，不是新選擇。三道閘，等 Kyzen 放行。

> **這一族與 §9.6 的差別**：§9.6 是「協定寫了、生效的路沒讀」。
> 這一條是**兩條路都在跑、各自都自洽、而它們讀的是不同的權重**——
> 兩邊都不會拋例外，因為兩邊都沒有錯。錯的是它們之間沒有人比對。

---

## D-5 只重跑超長的 2,095 筆（2026-08-28，Kyzen 自己提出並選定）

**這是第四條路線，不在 MASTER 送出的甲／乙／丙裡。Kyzen 看完那三個之後自己提出，並否決了工程師的建議。**

### 他的原話（ULIP2 Engineer 視窗，時間順序，經該角色轉述並由 MASTER 向 Kyzen 複核）

```
1  「我覺得這2095個物件能單獨重新跑嗎? 並且給規定限制字數」   ← 重跑＋限字數，兩個要素都是他提的
2  （工程師回：可以，3.7 GPU 小時；但那 2,095 筆不是隨機的，只重跑它們會讓
    「擺放最靈活的那群」文字品質特別好，是假訊號。建議 D 全，80 小時，一致。）
3  「我只想跑D 小」                                          ← 明確否決 D 全
4  「我可以不管指紋不指紋的嗎?? 我現在就是小重跑 D 小
     後續我資料也是 原標註結果 + D 小重跑結果 可以嗎?」
5  「啊記得原 D 小要刪掉喔」
```

### 分類：**DEVIATION D-5**

**不是 IMPLEMENTATION CHOICE** —— 工程師提出了科學上的反對意見，USER 聽過之後明確否決。
記錄反對意見本身，因為它會影響結果的解讀：

> **那 2,095 筆不是隨機樣本。** [OBSERVED DATA 2026-08-28，MASTER 獨立重現]
> 依 placement 旗標數分組，被判超長的比例：
> 0 旗標 3.20% · 1 旗標 1.52% · 2 旗標 4.44% · **3 旗標 43.97% · 4 旗標 82.58%**
> （富集 9.59x 與 18.01x）。
> → **重跑的那一批，系統性地就是「擺放方式最多元」的那群物件。**
> → 重跑後它們的描述會受一個**不同的**（且更嚴格的）字數合約約束，
> **而其餘 43,597 筆不會。** 語料因此含有一個與語意屬性相關的文字品質差異。
> **任何「描述品質 vs 檢索表現」的結論都要先排除這個混雜。**

**USER 已被告知此點並選擇 D 小（3.7 GPU 小時）而非 D 全（80 小時）。決定成立，混雜記錄在案。**

### 尚未取得核可的三個參數（MASTER 2026-08-28 登記）

工程師明確區分了「他核可的」與「我推導的」，這一節照他的區分寫，不擴大：

| 項目 | 狀態 | 依據 |
|---|---|---|
| 重跑 2,095 筆 | ✅ USER，他自己提的 | 上面第 1 句 |
| 「限制字數」這件事 | ✅ USER，他自己提的 | 上面第 1 句 |
| 只跑 D 小、不跑 D 全 | ✅ USER，明確否決建議 | 上面第 3 句 |
| 語料 = 43,597 原標註 ＋ 2,095 重跑 | ✅ USER | 上面第 4 句 |
| 舊的 2,095 筆不留 | ✅ USER | 上面第 5 句 |
| **字數上限 = 15** | 🔴 **未核可** | 工程師由量測推出（45,692 筆實測 token/word：中位 1.302 · 平均 1.308 · **p99 1.579** · 最大 2.344；15 字 p99 ≈ 23.7 tok，額度 25；18 字 p99 ≈ 28.4 爆）。**推導可查，核可沒有。** ⚠ 15 × 最大比值 2.344 = 35 tok，**蓋不住最大值** —— 硬保證仍只在 `_fit_description()`，不在字數上限。 |
| **PROMPT_VERSION 8→9** | 🔴 **未核可** | USER 要的是「限制字數」；改版本號是工程師選的手段。**副作用**：43,597 筆 v8 紀錄變成非當前合約 → 未來不帶 `--force` 的 n05 會判 UNACCOUNTED 而停住（補救工具 `declare_annotation_provenance.py --declare` 存在）。 |
| **`rank_used` 事前門檻** | 🔴 **未核可** | 工程師建議 rank1 ≥ 70% 通過、< 50% 回頭調字數。**必須在跑之前訂** —— 跑完再討論等於讓資料定義成功條件。 |

### 🔴 一個不會被任何機制修正的殘留

[OBSERVED DATA] **既有的 19 筆 pilot embedding 與那 2,095 筆的交集 = 0。**
→ v9 落地後它們的 `expected_text` 不變 → `is_complete` 六道全過 → **永久被跳過**。
→ 它們是語料中唯一「未經現行閘門產生、且不會被任何自動機制修正」的部分。
**形狀與 §9.11 同族：不是有人做錯，是沒有任何機制會發現。** 待 USER 處置。

---

## 9.13 5→10→25 那道階梯，三階不是同一條路（2026-08-29 新增）

**發現者：ULIP2 Block Engineer。ESSGNN Block Reviewer 往前推了一步，INTEGRATOR 指出它改變了結果的身分。**

### 機制

[OBSERVED IMPLEMENTATION] `cosine_schedule` 的曲線跨度是 `epochs × niter_per_ep`。
→ **改 `--epochs` 會同時改退火速率。**
→ `e5` / `e10` / `e25` 各自跑在不同的 lr 排程上，**是三個獨立實驗，不是同一條路的三段。**

Kyzen 核可的語意是「同一條路走得更遠，每階看過才准往上」（[[pilot-ladder-before-full-epochs]]）。
**那個語意與實作不符，而且沒有任何東西會報錯。**

### 已跑的兩階，身分要重寫

```
e5   mean R@1 0.9571（epoch 4）   在「跨度 = 5」的退火下
e10  mean R@1 0.9471（epoch 9）   在「跨度 = 10」的退火下
```

- ❌ **不可讀成「訓練久了反而變差」。** 那是兩條不同的退火曲線的終點。
- ❌ **e5 不能當短跑基線，e10 不能當「變差的證據」。** 這一格是 **UNKNOWN，不是負面結果**。
- ⚠ `--lr-horizon`（釘住跨度、提早停）落地之後，**上面兩個數字與未來任何一次跑也不可比**。
  → **階梯第一階要重跑才有基線。** 已花掉的 GPU 時間不可回收。

### 🔴 而且雜訊底線從來沒有人量過（ESSGNN Block Reviewer 提出）

```
每階 n = 1 · 無重複跑 · 無多 seed
e5 → e10 的差距 = 0.0100
同樣設定重跑兩次會差多少 = **沒有人量過**
```

**在雜訊底線未知之前，0.0100 不支持任何方向的結論。**
「七項有六項退步」聽起來一致，**但若單次重跑的標準差就有 0.01，六項同向是常見的。**

→ **建議（與上一節合併，只多花一階的 GPU）**：用固定跨度重跑第一階**兩次**
（同設定、不同 seed），一次拿到基線＋變異數。
→ **先量儀器，再量對象**（§9.7 同一條）。**解析度不夠時，輸出照樣是乾淨漂亮的數字。**

### 對「請求放行 GPU」那份報告的影響

**不是「修好了，接著跑 e25」，是「用固定跨度重跑 e5 → e10 → e25，且第一階跑兩次」。**
若報告只寫前者，那個 `✅` 會涵蓋錯的東西。

### 連帶：e5 的 artifact 已滅失

`stage1_best.pt` 是固定路徑，`chain_overnight.sh:110` 是「跑完才歸檔」，
所以 08:17 起跑的 e10 把它蓋掉了。**每輪總分還在**（`train_stage1_dev_val.pre_runid.jsonl`，
**倖存的唯一原因是同日 MASTER 裁決「改名不刪」**），**七條件與權重沒了**。
→ 證據等級分開標：**總分 OBSERVED DATA，七條件與權重 REPORTED**，記在
`data/outputs/ladder/e5_RECOVERED/e5_recovered.json`。
→ 違反 `.claude/rules/experiments.md` §10（需要比較的輸出不應互相覆蓋）。
**已發生的損失，登記不追究；後續三階各自獨立輸出路徑。**

---

## 9.14 「負載不是必要條件」不等於「跟訓練無關」（2026-08-29 新增）

今日機器硬重置 **5 次**。[OBSERVED DATA] 第 3 次（06:07:19）**機器完全閒置**：
該視窗零檔案寫入、journal 零行 python/訓練痕跡；對照組（有訓練的 boot）同樣搜尋是 15 行。

**這是一個存在性反例，n=1 就足夠，它推翻的是：**
> ❌ 「負載是崩潰的**必要**條件」

**它沒有推翻，而且兩者仍可能為真：**
> 「負載是充分條件」 · 「負載會提高機率」

**這句要寫死**（ESSGNN Block Reviewer 要求，我採納）：
**下一個讀的人若把它讀成「跟訓練無關」，會在沒有理由的情況下恢復長跑。**

### 已被自己的資料推翻的假設（記錄，因為它們曾經被當成結論）

| 假設 | 提出 | 推翻它的量測 |
|---|---|---|
| OOM | Kyzen 懷疑 | 崩前 RAM 剩 52,963 MB；且今日一次**真** OOM 有 kernel 紀錄、只殺該 unit、機器活著 —— 五次崩潰一項都沒做 |
| 電源容量不足 | ULIP2 Engineer | PSU 是 ASUS ROG STRIX **1200W 白金**，NVIDIA 對 5090 的建議是 1000W。**提出者自己撤回** |
| 功率抖動 | ULIP2 Engineer | 第五次崩潰前三分鐘抖動只有 **7.8 W**（先前訓練 13.9 W），比之前更穩，照樣崩。**提出者自己撤回** |

**剩餘嫌疑：記憶體 / XMP（DDR5-6000 on Arrow Lake）。這是 INFERENCE，不是量測** ——
要驗需 Kyzen 親自進 BIOS 關 XMP 或跑 memtest86。

---

## 9.9 `run_progress` 把「部分執行」與「什麼都沒產出」都記成 SUCCESS（2026-08-28 新增）

**起因**：我對 ULIP2 Engineer 說「共用 runlog 裡有 4 列 n08 是你留的」。**錯的，只有 2 列是他的。**
但我去查歸屬時，發現那四列本身有比歸屬更重要的問題。

[OBSERVED DATA `data/outputs/logs/run_progress.jsonl`，`stage == "n08_semantic_edges"` 全部四列]：

```
RUNNING  started 1786939789  rev 73f9f8a5   （2026-08-17）
SUCCESS  ended   1786940916  rc=0           歷時 1,127 秒
RUNNING  started 1787901160  rev 9e914572   （2026-08-28）
SUCCESS  ended   1787901162  rc=0           歷時 1.93 秒
```

**兩個問題，都不是誰的操作失誤，是 runlog 的形狀：**

1. **2026-08-17 那次記 `SUCCESS rc=0`、跑了 1,127 秒，而 n08 的三個產物一個都不存在**
   （`sem_edge_cache.json` / `sem_edge_embeddings.npz` / `sem_edge_sentences.jsonl` 今日實測皆 absent）。
   要嘛它成功了卻沒寫產物，要嘛產物後來被移走而 runlog 不知道。**兩種情況 runlog 都說 SUCCESS。**
2. **2026-08-28 那次是 `--limit-pairs 5 --skip-encode`，1.93 秒，同樣記 `SUCCESS rc=0`，
   而 runlog 裡沒有任何欄位說它是部分執行。**

→ **只讀 `run_progress` 的人會看到「n08 成功兩次」，然後推論 n08 已經跑過。**
兩次都不是。這正是 `tools/status.sh` 那族「假數字」的上游 —— 修了 status.sh 的 `find -L`，
但它讀的這份紀錄本身分不出「跑完」「跑一點點」「跑完但沒東西」。

**要做的（列 n13／n08 前置，尚未實作）**：`runlog` 的 SUCCESS 至少要帶
(a) 是否為受限執行（`--limit-*` / `--skip-*` 任一存在即記錄其值），
(b) 宣告的產出路徑在結束當下是否存在。**沒有 (b)，SUCCESS 只證明 process 沒有拋例外。**

---

## 9.8 防線寫了、具名了、解釋了 —— 但防線自己量錯（2026-08-28 新增）

**發現者：ESSGNN Engineer。我獨立重現，並在過程中發現他自己也被同一件事騙過。**

### 這一族與前兩族的差別，在「怎麼抓」

| | 形狀 | 抓法 |
|---|---|---|
| §9.6 | 協定寫了，生效的那條路沒讀 | **讀**那兩條路，對照 |
| §9.7 | 一條路，但儀器解析度不夠 | **先量儀器自己的雜訊** |
| **§9.8** | **防線存在、名字正確、論證完整 —— 但它量錯** | **執行那道防線，把它的輸出跟它自己的宣稱對照** |

**§9.6 和 §9.7 讀程式碼就能發現。§9.8 讀不出來。**
`test_f8_does_not_generalise_to_the_appendix_layer` 這個測試：
名字精確、docstring 解釋了它為什麼存在、斷言有意義、**而且它是綠的**。
跑整套 pytest 不會有任何訊號。

### 這次的實例

`tests/test_essgnn.py:255`：

```python
for seed in range(6):                              # seed 從未被使用
    n = geometric_sensitivity(16,   family="appendix_shared_msg")
    w = geometric_sensitivity(1280, family="appendix_shared_msg")
    ratios.append(w / n)
```

`geometric_sensitivity()`（`:204`）沒有 seed 參數 → 呼叫 `make_scene(n, cfg)` →
`make_scene`（`:38`）的 `seed: int = 0` 每次 `torch.manual_seed(0)`。
**六次呼叫回傳位元相同的值。宣稱六樣本，實際單樣本。**

[OBSERVED DATA 2026-08-28，MASTER 於 CPU 獨立重跑，未修改任何檔案]：

```
appendix 連續六次 narrow(16) → 1.250882e-01 × 6，all identical = True
appendix   narrow 1.250882e-01  wide 3.740838e-01  ratio 2.990560
two_mlp    narrow 4.463185e+01  wide 1.889776e+00  ratio 0.042341
```

### 🔴 重測之後，上面那幾句話有一半是我自己的錯（2026-08-28 稍晚）

ULIP2 Engineer 指出：我量到的 `2.990560` 是在**測試檔自己的設定**
（`hidden_dim=32, n_layers=3`）與 **單一 seed 0** 下得到的，
而 `essgnn_arch_protocol.json` 的實際配置是 `hidden_dim=128, n_layers=4`。
**「我用一支已知會單樣本的函式，去糾正它產出的單樣本數字」—— 同一個病。他對。**

我寫了一個**真的會傳 seed** 的版本（`make_scene(n, cfg, seed=s)`，s=0..5），
兩種家族 × 兩種配置各量六次
[OBSERVED DATA 2026-08-28，MASTER，CPU，未修改 repo 任何檔案]：

| 家族 | 配置 | ratio 中位數 | 範圍 | ratio > 1 |
|---|---|---|---|---|
| `appendix_shared_msg` | 測試 cfg 32/3 | 0.1104 | [0.0576, **2.9906**] | 2/6 |
| `appendix_shared_msg` | **協定 cfg 128/4** | **0.3066** | **[0.1349, 0.8591]** | **0/6** |
| `sec25_two_mlp` | 測試 cfg 32/3 | 0.0594 | [0.0040, 0.2039] | 0/6 |
| `sec25_two_mlp` | 協定 cfg 128/4 | 0.1037 | [0.0187, 0.5826] | 0/6 |

**三件事因此翻案：**

1. **`2.990560` 是六個 seed 裡的最大值，不是代表值。** 它就是 seed 0。
   我拿一個離群點當結論。**撤回。**
2. **docstring 的數字不是憑空來的。** 它宣稱 appendix median 0.126 / range 0.031–2.278，
   我在同一個測試配置下量到 0.1104 / [0.0576, 2.9906] —— **同一個量級、同一個形狀**；
   two_mlp 宣稱 0.055 / 0.007–0.243，我量到 0.0594 / [0.0040, 0.2039]，**幾乎重合**。
   → **它們來自一個「有把 seed 傳下去」的較早版本。壞掉的是後來的程式碼，不是當初的量測。**
   前一版此處寫「來自某個沒有留下的更早版本或臨時執行」暗示可疑 —— **語氣撤回，事實保留：
   現在這份程式碼產不出它們。**
3. 🔴 **最重要的一項：在協定的真實配置下，加寬 e_ij 在我們的家族上也會壓制幾何，0/6 例外。**
   → **ESSGNN Engineer 最初的警告是對的。** 他用錯了數字，我又用更糟的量測「推翻」他。
   **U-20 的 1280 確實會壓幾何訊號，中位數約 3 倍**（two_mlp 約 10 倍，我們較輕但同向）。

**唯一沒被推翻的**：`00_FINDINGS.md:277` 的 `50.9 / 1.14` 仍對不上我的 `44.63 / 1.89`
（同為 seed 0、測試 cfg）。那一處待重測。

🔴 **是四份。而我在寫這一節的時候，親手示範了這一節在講的事。**

我原本在此處寫「是三份不是四份，`semantic_edges_run.py` 沒有數字（grep `50.9` 零命中）」。
**那句話錯了，ESSGNN Engineer 當場糾正，我驗證後確認他對：**

```
工作樹（含 ULIP2 2026-08-28 未提交的編輯）
    grep 50.9 → 零命中                                  ← 我 grep 到的

git show HEAD:metafind/data/semantic_edges_run.py
    :83  #     e_ij width      geometric sensitivity
    :84  #            16                       50.9
    :85  #          1280                       1.14     ← 在

git show 158a337:...（他 8/22 引用時的版本）
    :84  50.9   :85  1.14                               ← 也在
```

**兩邊都是真的檔案，答案相反。差別是一筆未提交的編輯。**
在他做出宣稱的時間點是四份；在我 grep 的時間點是三份。
**而且因為那筆編輯未提交，若它被 revert 或三道閘沒過，第四份會回到樹上。**
所以記成「已在一筆未提交的編輯中移除」，**不記成「從未存在」**。

> **在共用工作樹上引用檔案內容時，說明引的是工作樹還是已提交狀態。**
> **兩者可以同時為真且互相矛盾，而 `grep` 只會給你其中一個，不會提示另一個存在。**
> （ESSGNN Engineer 措辭，原樣採納。同一天同一個檔案，他早上引舊版說「沒修」、
> 我下午引新版說「從沒有過」—— 同一個機制，兩個方向各一次。
> 他當時給的教訓是「送出前重讀」；這次證明不夠：**重讀也可能讀錯那一個「現在」。**）

⚠ **但那第四份還壞在另一個地方，與計數無關**：HEAD `:86-88` 寫
「`f_h` takes (2*hidden + 1 + e) ... drowns that scalar about 45-fold」，
而 ULIP2 今天新寫的版本也保留了「drown ... **inside `f_h`**」這個說法。
`f_h` 是 two-MLP 層（`essgnn.py:325`），附錄層用的是 `phi_e`（`:422`）。
**它描述的機制屬於我們沒選的那一支，卻被寫成無爭議的通則**，
而實測 ratio 2.99 正是它的反例。
（形狀與 ESSGNN Engineer 引 two_mlp 的**數字**去講 appendix 同源，一個引數字、一個引機制。）

### 結論的範圍（2026-08-28 稍晚重測後的最終版）

- ✅ **F8 在 two_mlp 上成立**：協定 cfg 六 seed 中位數 `0.1037`，0/6 超過 1。
- ✅ **F8 在我們的家族（appendix）上也成立，只是較輕**：協定 cfg 中位數 `0.3066`，
  範圍 `[0.1349, 0.8591]`，**0/6 超過 1**。→ **U-20 的 1280 確實壓幾何，約 3 倍。**
- ❌ ~~「F8 不能推廣到 appendix」~~ —— **這是我的錯，來自單 seed 0 + 測試 cfg 的離群值 2.99。撤回。**
- ⚠ **測試的斷言 `max(ratios) > 0.243` 在協定 cfg 下會失敗**（實測 max 0.8591 > 0.243 → 其實通過；
  但那是在測試自己的 cfg 32/3 下 max 2.9906 才通過得那麼寬）。修測試時要一併決定
  **它到底該在哪一組 cfg 上斷言** —— 現在斷言的是一組我們不會跑的配置。
- ❗ **對 U-20 的實際影響**：不重開（Rule 13，有 USER 核可），但那條「量測回報」的承諾
  現在有了具體數字：**約 3 倍壓制，方向確定，0/6 例外。**
  緩解出口仍是 `edge_proj_dim`（`essgnn.py:202`，目前 `None`），Stage 2 有數字後回頭。<br><br>🔴 **2026-08-28 再查出一筆：U-20 拍板了，但它在機器上沒有任何代表。**（ESSGNN Engineer 提出，我逐條驗過。）<br>[OBSERVED DATA] `essgnn_arch_protocol.json` 只有十二個 key：`architecture_family · coord_feat · decided_at · decided_by · distance · hidden_dim · layer_sharing · mlp_structure · n_layers · pooling · status · use_io_projections` —— **`node_feat_dim` 與 `edge_feat_dim` 都不在裡面。**<br>[OBSERVED IMPLEMENTATION] 全 `metafind/` 每一個用到這兩個維度的位置，不是**探針假值**就是**從產物讀**：`resolve_stage2.py:229`（512/512，round-trip 探針）· `stage2.py:104`（1/1/1，另一個探針）·`stage2.py:552-553`（生效路徑，`data.node_dim` / `data.edge_dim`）·`stage2.py:314` 與 `:323`（從 `procthor_node_embeddings.json` 與 `sem_edge_cache` 讀）。**沒有任何一處從協定讀，因為協定裡沒有那兩個 key。**<br>→ **`stage2.py:314-327` 那兩道 raise 比的是「產物 json 宣告的維度」對「產物 npz 的實際 shape」——產物對產物。產物跟自己永遠一致，所以 n08 若因任何原因寫出 512 寬的東西，n13 會照跑，沒有任何訊號。**<br>⚠ **這筆債早就登記過，而且指名了到期日**：`docs/graph/01_GRAPH_SPEC.md:1123`（第 201 項）逐字：「U-12／U-20 …… 都還沒進 `essgnn_arch_protocol`……**在 `n13` 實作前必須進 `n09b`／`G6`**，否則就是 Stage 1 剛修完的錯換到 Stage 2 重演」。**n13 已經 680 行寫完了。債沒付。**同一節的收尾（`01_GRAPH_SPEC.md:1128-1131`）逐字，**是連續兩輪各寫一次的兩條教訓，不是一條**：<br>　「**這輪要記住的**：第十六輪學到『protocol 要真的被執行的東西讀走』，這輪學到的是**它得在整條路徑上都被讀走**。`actual=trainable` 在 `n10` 的入口是通的、出口是斷的 —— **一個只在半條路徑上成立的合約，看起來與成立的合約一模一樣。**」<br>**我們這次兩條都沒守住，而且是更徹底的版本**：第十六輪那條 —— 不是「沒被執行的東西讀走」，是**協定裡沒有那個 key，沒有東西可讀**；下一輪那條 —— 不是「入口通、出口斷」，是**整條路徑上都不通**。**他們當初抓到的那次至少入口是通的。**<br>（最後那句「只在半條路徑上成立的合約，看起來與成立的合約一模一樣」正是本文件 §9.8 的形狀，而它比 §9.8 早了一輪就被寫下來。**§9.8 也是再犯，不是新發現。**）<br>❌ **把 `resolve_stage2.py:229` 的 512 改成 1280 不算付這筆債**：探針照樣不讀協定，生效路徑照樣不看它。**改完會看起來像付了，其實沒有。** 改仍可做（讓鷹架不像決定），但必須在原地註明它是鷹架、真值從產物讀、U-20 尚未進協定。<br>✅ **真正要做的（列 n13 前置，研究關鍵，三道閘）**：把兩個維度加成協定的真 key，並在 n13 啟動時比對「協定宣告的維度」對「產物實際的維度」，不符就 raise。**那是 U-20 唯一可能的機器化執行點。**<br>（ESSGNN Engineer 的措辭，採用：🔴 **2026-08-28 我又用 `| head` 截斷過的輸出下了「只有一個」的結論，而且把它當 MEASURED 報給兩個工程師。**Codex 抓到。我對兩人說「全 repo 讀 `sem_edge_cache` 的只有 `stage2.py:320`」——**實際 16 處，其中 `scene_splits.py:60/:141`（n09c）是真的讀者。**我的指令是 `grep -rn ... | head`，輸出剛好滿 10 行，`head` 是我自己加的視窗。**裁決本身不變**（Stage 1 那條路確實沒有讀者，閘門仍該放寬），但我給的理由是錯的；ULIP2 已把理由從「只有一個讀者」改成「Stage 1 側沒有讀者」——**後者才是這個閘門要的**。這條與長期記憶 `never-reason-from-my-own-truncation` 是同一條，**寫過了還是再犯。**<br><br>**§9.6 的變體 —— 不只是「改在不生效的檔上」，是「改在不生效的檔上，然後以為債清了」。**）

**修測試時的範圍限制（ESSGNN Engineer 提出，採納）**：要修的是
「單樣本偽裝成六樣本」與「三處對不上的數字」，**不是「F8 這個發現要重來」**。
四處都錯的觀感容易讓修的人順手把結論一起撤掉。

**分級**：`geometric_sensitivity()` 是 F8 範圍主張的唯一守門員 → 研究關鍵測試 →
走 code → 審 → 放行三道閘。`:233` 的 `wide < narrow/5` 同為單樣本，**兩個一起重測**。
**列入 n13 前置，2026-08-28 未動。**

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

**S-C｜Fusion 之前要不要先正規化各模態。** ⚠ **2026-08-27 大幅縮小：現況已與論文記號一致，缺的只是一筆記錄。**

**上游答不了，而且理由很硬**：ULIP 與 OpenShape **都沒有 fusion 模組**，
兩家的 `normalize` 都是「進 logits 前的最後一步」（`ULIP/models/losses.py:34-36`、
`OpenShape/src/train.py:62-68`）。**「fusion 之前」這個位置在上游不存在**，
所以這不是沒查到，是問題在上游無從提出。

**Step 1 給了方向（INFERENCE，不是 PAPER FACT）：**

```
Eq.6   e_query = Fusion(e_text, e_img, e_pc) + λ·e_layout
       Fusion 收的是未加修飾的 e_*，式子裡沒有 hat 也沒有 ‖·‖
Eq.5/7 正規化整個藏在 sim(·,·) 裡（U-24 讀成 cosine）
       → 論文的記號把正規化放在「最後算相似度」那一步，不在融合輸入端

補強   Eq.6 是殘差相加，論文自述目的是 "without disrupting the original
       embedding space"（2methdology.tex:87）。若在 Fusion 輸出後、
       加 λ·e_layout 之前先 normalize，那個殘差就不是加在「原本的嵌入空間」上了。
       不 normalize 才守得住作者自己給的理由。
```

**而現行程式碼已經正是這樣**（`ULIP2 ENGINEER` 查，我複驗）：
`fusion.py` 與 `dual_tower.py` 各零筆 normalize；
全 repo 只有 `losses.py:166-167` 一處，就在算相似度之前。

→ **正確描述不是「政策未鎖、可能往兩邊跑」，而是「現況已對，但從沒被記成一個決定」。**
→ 要 Kyzen 的因此只有一句追認：把「只在 sim 之前 normalize、fusion 輸入端不 normalize」
  記為 IMPLEMENTATION CHOICE（依據：Eq.6 的記號＋它自述的殘差理由），還是要改成別的。
`2meth:30-35, 74-100` 只說各模態編碼後送 Fusion。
是 `各模態先 normalize → Fusion`，還是 `原始向量 → Fusion → 最後才 normalize`？
對 mean、MLP、Transformer 三種融合都會改變結果。
本筆記對「最終 q/g 做 cosine 正規化」寫得很清楚，但**融合輸入端的正規化政策同樣要鎖，目前未鎖**。
開訓前必須定案並登記。

**S-D｜分母含不含正例。** ⚠ **2026-08-27 重新分類：本條原本的框架是錯的，已改寫。**

原本寫「PAPER 沉默，`2meth:93-100`」，並把 Eq.7 那句當成 Stage 1 的問題。
**實際上論文對同一個符號 𝓑 給了兩個不同的定義，而 Stage 1 那個是明確的：**

```
Eq.5（Stage 1）  2methdology.tex:79  逐字
  "where τ is a temperature hyperparameter and 𝓑 denotes the gallery batch."
                                                         ^^^^^^^^^^^^^^^^^
Eq.7（Stage 2）  2methdology.tex:99  逐字
  "where τ is a temperature hyperparameter, and 𝓑 denotes the batch of negatives."
                                                         ^^^^^^^^^^^^^^^^^^^^^^
```

| 階段 | 論文說什麼 | 分類 | 要不要 Kyzen 拍板 |
|---|---|---|---|
| **Stage 1（Eq.5）** | 「the **gallery batch**」—— gallery batch 當然含正例那一件 | **METAFIND PAPER FACT。不是沉默，也不是歧義。** | **不用。** 我們的 CE 實作與它一致 |
| **Stage 2（Eq.7）** | 「the **batch of negatives**」，與 Eq.5 對同一符號的定義**不一致** | **PAPER 內部矛盾（C 類），不是沉默（S 類）** | 要，但**不擋 Stage 1** |

**上游兩家獨立佐證，都把正例放進分母（`ULIP2 ENGINEER` 2026-08-27 查，我複驗）：**

```
upstream/ULIP/models/losses.py:43-51      CE 的分母是整列 logits，正例是其中一欄
upstream/OpenShape/src/train.py:66-73     同一形狀，labels = arange(batch)
```

**還有一個不需要出處的論證**：若分母真的只含負例，分子的正例項不在分母裡，
`-log(exp(s⁺)/Σexp(s⁻))` 對 `s⁺` **沒有上界，損失可以無限往下掉**。
那不是 InfoNCE，是一個發散的目標函數。
→ **Eq.7 的 "negatives" 是鬆散措辭，Eq.5 的 "gallery batch" 才是精確的。**

**Stage 2 的建議**：照 Eq.5 的定義，記為 `DEVIATION-from-Eq.7-wording`。待 Kyzen 簽。

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
| U-11/U-13/U-15/U-17/U-19/U-28/U-29/U-30/U-32 | ✅ 見各節 |
| **U-31 / U-33** | **⚠ 2026-08-27 更正：不再列為已拍板。** `docs/ESSGNN_DIM_REVIEW.md` v3 §7 把 `layer_sharing` 與 `use_io_projections` 列為 OPEN，與本表衝突。依 Rule 16，「曾被某個 block 標 resolved」不構成 promotion path —— 需要指得出**具體參數**的核可。**在指出之前一律視為待決** |
| **n07c ProcTHOR 節點文字重生** | ✅ **USER-APPROVED 2026-08-27，並於同日升格為 DEVIATION D-4。**<br>🔴 **升格的理由（2026-08-27 實測，是我先前沒查的）：論文指定的來源不存在。**<br>`2methdology.tex:28` 逐字：「Each room configuration provides precise spatial coordinates and **comprehensive semantic metadata** for each asset」。<br>**實測 ProcTHOR-10k 的 house JSON，每個物件只有**：`id` · `assetId` · `position` · `rotation` · `kinematic` · `children` · `isDirty` · `layer` · `material`。<br>**掃 300 間房、20,560 個物件（含 children），`material` 欄位 100% 是 `None`。** 沒有材質、沒有顏色、沒有風格、沒有尺寸描述。`ai2thor` 的 `metadata.json` 只有 `server_types`，與資產無關。<br>→ **能從那份 metadata 生出來的最多就是 `assetId` 的類別名，也就是現況的 93 個字串。**<br>→ **所以 n07c 不是「補足論文沒說的」，是「論文說的那個來源名不副實」。分類為 DEVIATION，不是 IMPLEMENTATION CHOICE。**<br>⚠ **連帶影響**：`2methdology.tex:47` 說語意邊是「prompting an LLM **with object descriptions**」——**物件描述是 LLM 的輸入**。輸入湊不出來，所以我們卡在論文的上游，n08 也連帶受影響。<br><br>**原始理由（仍然成立）**： 現況 `procthor_object_text.json` 1,467 個資產只有 **93 個不同字串**，`source` 全部是 `procthor_category`（抓類別名套「a ___」模板，冠詞也沒處理：`"a apple"`）。最極端：74 個不同的 side table 共用 `"a side table"` → **拿到完全相同的節點特徵**。材料早已備妥：`procthor_modalities/` 底下每個資產都有 `view_00~10.png` 與 `pointcloud.npz`（2026-08-16）。**做法**：11 張渲染圖丟 `gemma-4-12B-it`（與 n05 同流程同模型）重生描述，成本 1,467 × 6.2s ≈ 2.5 小時，**須等 n05 讓出 GPU**。**分類**：論文對 ProcTHOR 節點文字**完全沉默** → 這是 IMPLEMENTATION CHOICE，**與 D-2 分開記**（D-2 是「論文寫 GPT-4o、我們跑 gemma」的偏離；n07c 是「論文沒寫、我們自己決定」的選擇），模型與流程引用 D-2。✅ **驗收門檻 — USER-APPROVED 2026-08-27（Kyzen 選甲）**：達成率 ≥ **0.70**，
達成率 = 實際相異率 ÷ 幾何天花板。**天花板 = 0.709**，區間 **[0.638, 0.709]**，
**不是精確值**。現況 93 個相異字串 → 達成率 **0.089**；門檻 0.70 需約 **728 個相異 caption**。
**重算參數**：D2 形狀分布 · 100,000 對點（rng seed 0）· 64 bin · 尺度正規化 + L2 ·
閾值 **0.0518 取自儀器自噪 max**（同資產換種子，12 探針，median 0.0329）· complete-link ·
1,439 個有點雲的資產 → 1,012 群 + 28 = 1,040 → 1,040/1,467。
**single-link 已檢驗並拒絕**：最大群 125 成員、群內直徑 0.2902 = 噪音底線 5.6 倍，
成員含 Alarm_Clock_21 與 Book_10..14 —— 鬧鐘和書被串成同一個「幾何」。
**精度上限**：要更準需比對 mesh 而非點雲，ProcTHOR 沒有 mesh 管線（未驗證 GLB 是否可得）。
🔴 **撤回**：先前寫「點數＋bbox 的 0.90 是下界、會低估」—— **實測相反，是高估**，
bbox 取到小數三位，1mm 差異就算不同幾何，在噪音上過度分割。詳見 §9.7。
🔴 **獨立發現**：`<Mesh>_<N>` 變體不保證同幾何（`TV_Stand_206` 1.20m vs 1.89m）。
**其餘要量的**：**成功標準不是「1,467 筆都有文字」**，要重新量：不同字串數、同類別內的不同比例、顏色／材質／風格屬性的多樣性、空值、幻覺抽樣 |
| **n07c 的輸出位置** | ✅ **USER-APPROVED 2026-08-27：另存新檔，舊檔保留。** 不覆蓋 `procthor_object_text.json`。理由兩條：(1) n07c 是 **DEVIATION D-4**，偏離的前後對照要留得住，否則無法證明「換了之後真的比較好」；(2) 該檔同時是 **n07 的產物**（`node_registry` 的 n07 `writes` 含它），覆蓋 = 兩個節點寫同一個 channel，且舊值不可回溯 —— 正是 §9.6 那族「誰改了什麼看不出來」的形狀（INTEGRATOR 提出）。**成本：下游改一行指到新檔，協定要記讀哪一個。** |
| **U-20（ESSGNN 節點／邊文字編碼器）** | ✅ **USER-APPROVED 2026-08-27：改用 OpenCLIP ViT-bigG-14，1280 維。** 取代 8/17 由 Claude 自選、且理由已被程式碼推翻的 `ViT-B/32 / 512`。<br>**分類：IMPLEMENTATION CHOICE，不是 PAPER FACT 也不是 UPSTREAM FACT。**<br>· 論文只寫 `a frozen text encoder (**e.g.** CLIP or BERT)`（`2methdology.tex:47`），而且那句講的是**邊**；**節點的 `t_i` 論文連例子都沒有**。<br>· **上游答不了**：ULIP-2 沒有場景圖、沒有節點特徵這個位置，問題在上游提不出來。<br>**Kyzen 的理由：一致性。** Stage 1 的 ViT-bigG-14 是被 ULIP-2 checkpoint 逼的（`ulip_backbone.py:90` `EMBED_DIM = 1280`，投影層形狀 `(768,1280)`），沒得選；**同一個專案不要兩套文字理解**。而 n07c 反正要重跑，此時換成本最低。<br>⚠ **時序**：`n07c 換文字 → U-20（本條）→ n08 跑一次`。本條不定，n08 不能跑；定了，n08 只跑一次。<br>⚠ **邊的寬度另計**：我先前用「EGNN 的 `in_edge_nf` 是 0/2 所以邊該窄」論證過，**那條已撤回**（Type C，推不出架構原則）。邊要不要也用 1280、還是用較窄的，**目前無依據，待後續。**<br><br>🟠 **2026-08-28 補交叉指標（ESSGNN Engineer 提出，我查證後修正了它的範圍）。**<br>他提的事本身成立：**U-20 把 e_ij 定在 1280，而 §3.4 的 F8 量到的正是「加寬語意邊會壓掉幾何訊號」。**[OBSERVED DATA `docs/graph/00_FINDINGS.md:270-280`，2026-08-15 固定種子]：`|∂e_layout/∂pos|max`，語意邊全部置零，`edge_feat_dim=16 → 50.9`、`1280 → 1.14`，**約 45 倍**。<br>🔴 **但那組數字不能直接套在我們身上，他引用時沒帶這個限定。** [OBSERVED IMPLEMENTATION `tests/test_essgnn.py:233-247`] 該測試自己的 docstring 逐字：「**F8 MEASURED, and it belongs to the TWO-MLP layer specifically**」，🔴 **但那份 docstring 宣稱的「六個種子」是假的，我自己重跑過（見下），所以它的中位數與範圍都不要引用。**同檔還有一個具名測試 `test_f8_does_not_generalise_to_the_appendix_layer` 專門釘住這件事。<br>**我們的架構家族是 `appendix_shared_msg`**（`data/outputs/essgnn_arch_protocol.json`），**正是 F8 不穩定成立的那一支。** 所以正確的說法不是「U-20 選到最差的寬度」，而是：**在我們選定的架構上，加寬 e_ij 讓幾何訊號變強，不是變弱 —— 方向與 F8 相反。**<br>🔵 **2026-08-28 我親自重跑（CPU，`tests/test_essgnn.py` 的 `geometric_sensitivity()`，未動任何檔）：**<br>　`appendix_shared_msg`：narrow(16) `1.250882e-01` · wide(1280) `3.740838e-01` · **ratio 2.990560**<br>　`two_mlp`：narrow(16) `4.463185e+01` · wide(1280) `1.889776e+00` · **ratio 0.042341**（F8 在這一支確實成立，約 24 倍壓制）<br>　→ **兩支架構方向相反，而我們是 ratio > 1 的那一支。**<br>🔴 **同時查出那個具名測試本身壞了（ESSGNN Engineer 發現，我逐行驗證並重現）：**`tests/test_essgnn.py:255` 的 `for seed in range(6):` 迴圈裡，**`seed` 從頭到尾沒有被使用**；`geometric_sensitivity()`（`:204`）沒有 seed 參數，內部呼叫 `make_scene(n, cfg)`，而 `make_scene`（`:38`）的 `seed: int = 0` 每次都 `torch.manual_seed(0)`。**六次呼叫回傳位元相同的值**（我實測六次全部 `1.250882e-01`）。**它宣稱量了六個種子，實際上把同一件事量了六次。**<br>　連帶：docstring 的 `median 0.126 / range 0.031–2.278` **這份程式碼產不出來**（實測單點 2.99，落在它自己宣稱的範圍之外）；`docs/graph/00_FINDINGS.md:277` 的 `16 → 50.9 / 1280 → 1.14` 也對不上我重跑的 `44.63 / 1.89`。**那些數字來自某個沒有留下的更早版本或臨時執行，目前不可重現。**<br>　修法不是改一行：`geometric_sensitivity()` 要加 seed 並往下傳，兩個 F8 測試的 docstring 數字要重測，§3.4 與本條要跟著更新。**它是 F8 範圍主張的唯一守門員，屬研究關鍵測試，走三道閘，列入 n13 前置。**<br>→ **不重開 U-20**（有 USER 核可、理由清楚、Rule 13）。要記的是那條承諾的出口：§3.4 寫的「量測回報、不悄悄修正（`edge_proj_dim` 旗標保留）」，目前 `edge_proj_dim = None`（`essgnn.py:202`）。**Stage 2 有數字後，這是唯一的緩解位置，而它不需要重跑 n08。**<br>（記這一條的理由：那組數字先前只活在對話與 `00_FINDINGS.md` 裡，U-20 條目本身看不到它。這正是 §9.6 那族「兩份文件之間掉下去」。） |
| **~~U-20 原始登記~~** | **⚠ 標「已解決」但有兩個缺陷。** `DECISION_LEDGER.md:557` 記錄現況為 `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`、512 維（實測 `procthor_node_embeddings.json` 的 `text_encoder_version` 相符）。**缺陷一**：它是十個 RESOLVED 條目中**唯一 `decided_by` 沒有 USER 的**（`DECISION_LEDGER.md:564`）。**缺陷二**：ESSGNN Engineer 2026-08-22 證明它的理由被程式碼推翻（`:585`）—— 原理由稱「共用編碼器才能共用語意空間」，但 `essgnn.py:456` 的 `use_io_projections` 已把 `t_i` 投影成學出來的 128 維，`:471` 的 `e_ij` 則原始 512 維進入，**兩者本來就不在同一空間**。**待 Kyzen 裁決** |

**（2026-08-27）注意**：Stage 1 骨幹是 ViT-bigG-14 / 1280 維（`ulip_backbone.py:90`），
ESSGNN 節點向量是 ViT-B/32 / 512 維。**兩者本來就不同，不是誰算錯**，但 U-20 未經裁決。

### 🔴 最優先：2026-08-19 那個戳，對五個參數各自是不是核可？

**這一題答完，下面五項同時有答案。ESSGNN ENGINEER 2026-08-27 提出，我複驗確認前提成立。**

`data/outputs/essgnn_arch_protocol.json` 全文只有**一個**頂層 `decided_by`：

```
n_layers            4
pooling             "mean"
hidden_dim          128
use_io_projections  true
layer_sharing       "independent"

decided_by  "Kyzen (2026-08-19, C1 決定後補寫)"    ← 五個共用這一個戳
decided_at  2026-08-19T13:17:25
```

**我先前對其中兩個當成核可（`n_layers` `pooling`）、對另外三個當成待決，而我給不出理由。**

| 讀法 | 推論 | 對 resolver 的影響 |
|---|---|---|
| **A** 這個戳是逐參數的真核可 | 五個**全部已核可** | 檔案照現況，一個字不用改 |
| **B** 這個戳只是「C1 決定後把檔案補寫出來」的一次批次蓋章 | 五個**都缺乏足以進協定的逐項 authority** | 見下方結案 |

## ✅ 已結案 — 2026-08-27，Kyzen 核可 **B**（經外部審查修正措辭與範圍）

**措辭要精確**（外部審查改的，我原本講得太滿）：

> **不是「證明 2026-08-19 從未有任何口頭核可」——Rule 2 不允許這樣反推。**
> 是「**以目前可核證的證據，這幾項沒有足夠的 Rule 16 authority，所以不能維持 `resolved`**」。

**而且不能一刀切。同一個戳底下的欄位要逐個看它自己的來源：**

| 欄位 | 判定 | 理由 |
|---|---|---|
| `architecture_family` | **保留 RESOLVED** | U-26 有獨立裁決（2026-08-17，`decided_by` = user + external review，`C_PAPER_CONTRADICTIONS.md:55`）。Rule 13：不重開 |
| `coord_feat = current` | **保留**，但分類是 **DERIVED**，不是獨立核可 | 選了 `appendix_shared_msg` 之後，`m_ij = φ_e(h_i^l, ...)` 用的就是舊的 `h^l`；`updated` 是 §2.5 家族的語義。**它是 U-26 的架構後果** |
| `distance = squared` | 🔴 **改為 UNRESOLVED** | `01_GRAPH_SPEC.md:608` **U-17 明列 UNKNOWN**。§2.5 寫 `‖·‖₂`、Appendix C 用 `‖·‖²`，兩者都 SE(3) 不變、都不破壞證明，**但數值不同**。⚠ **我先前完全沒查這一項** |
| `mlp_structure = linear_silu_linear` | 🔴 **改為 UNRESOLVED** | `01_GRAPH_SPEC.md:622` **U-35 明列 UNKNOWN**，原文自己說「那從來不是一個決定，只是它剛好長這樣」。⚠ **我先前完全沒查這一項** |
| `n_layers` `pooling` `hidden_dim` `use_io_projections` `layer_sharing` | **五個都 UNRESOLVED** | 無逐項 authority |
| `status` | 🔴 **不得為 `resolved`** | 見下 |

**「為什麼出現這個值」與「它有沒有 authority」是兩件事**，五個的來源其實不同：
`hidden_dim=128` 有 QM9 的 experiment-specific 先例 · `use_io_projections=True` 有 EGNN 實作先例 ·
`n_layers=4` / `layer_sharing=independent` / `pooling=mean` 是現行實作值。
**但沒有一個具備 Rule 16 要求的逐項核可。**

### 🔴 連帶查出三個獨立缺陷（外部審查提出，我逐條驗過）

```
缺陷 1  status 的「產生條件」與「讀取條件」不一致，而且它是活的閘門
        resolve_stage2.py:297   arch_status = "resolved" if ARCH_DECISIONS["architecture_family"] else ...
                                ← 只要「架構家族」有值，整份就標 resolved
        essgnn.py:231           if protocol.get("status") != "resolved": raise
                                ← 執行期真的讀它
        → 寫檔的只看一個欄位，讀檔的以為全部都定了。**這是 §9.6 的又一例。**

缺陷 2  G6 只檢查「有沒有值 / status」，不檢查「憑什麼」
        有值 ✓ · status resolved ✓ · 誰決定的？→ 閘門看不到
        → 就算這次修乾淨，default → materialize → resolved → G6 PASS 的路徑仍然開著。
          **Rule 16 沒有被機器化。**

缺陷 3  U-26 的文件漂移
        C_PAPER_CONTRADICTIONS.md:55   U-26 DECIDED，appendix_shared_msg is primary
        01_GRAPH_SPEC.md:619           U-26 UNKNOWN，候選欄還寫「實作依 §2.5」
        → 兩份文件對同一個 id 給相反狀態。不修的話，
          下一個人會把一個 Rule 13 不該重開的問題重開。
```

**這三個都不是本題的答案，是本題查出來的。列在這裡免得又只活在對話裡。**

---


**戳的字面偏向 B**：它自己說是「在 C1 決定**之後**補寫的」，而 **C1 決定的是架構家族，不是這五個**。
一個描述「跟著別的決定順手補寫」的戳，很難同時是對這五個參數各自的核可。
**而 Rule 16 要的正是指得出具體參數的核可 —— 這個戳指的是一次寫檔動作。**

**我查不到 2026-08-19 當天實際同意了什麼**：`DECISION_LEDGER.md`、
`C_PAPER_CONTRADICTIONS.md`、以及全 repo 的 `.md` 都沒有對應條目。
那段脈絡在對話紀錄裡，不在 repo 裡。**所以我無法代答，必須 Kyzen 裁。**

**在 A/B 未定之前，`resolve_stage2.py` 的那五個值不能寫任何一版** ——
兩種讀法給出兩份不同的目標值，寫哪一版都是替他選。

---

### 待拍板（附我的建議；**2026-08-25 逐條重驗過「論文真的答不了嗎」**——結果見各行）
| # | 議題 | 重驗結果 | 我的建議＋理由 |
|---|---|---|---|
| 1 | **U-14** 訓練時圖片視角 | **證據升級（2026-08-26）**：最近的一手來源其實是 **ULIP-2 自己**，`ulip2 main.tex:612` 原文 "randomly sample its 2D rendered image **I ~ render(O)**"。再加 OpenShape method.tex:77 "randomly sample one rendered image or thumbnail"、ULIP-1 main.tex:236 每步隨機 1/60。**三個上游一致，其中一個就是我們的直接骨幹**，所以訓練側屬強 UPSTREAM FACT，不是 UNKNOWN。但**「gallery／評測用 12 視角平均」上游沒有背書**，那一半仍是我們的 CHOICE | **B**：訓練隨機單張、gallery 與評測用平均。⚠ **2026-08-27 更正：「換法零成本」是錯的，已撤回。** per-view 嵌入確實已快取（**重編碼**成本為零，這半句仍成立），但**沒有任何程式碼消費它**：`random_single_view` 在 `metafind/` 底下只出現兩次，都在 `stage1_config.py`（`:125` 常數定義、`:241` 註解），**沒有第三處**。`stage1.py:103` 存了 `self.aggregation` 之後**從未再讀取**；`:110-121` 的 `__getitem__` 直接讀 `cached["image"]`，也就是已聚合的那一支，`z["views"]` 在訓練路徑上無人碰。**所以現在把協定改成 `random_single_view`，訓練會照樣用 12 張平均，協定會記載 random_single_view，而且不會有任何東西報錯。** 真實成本：約 10 行（`__getitem__` 依 aggregation 從 `z["views"]` 隨機取一列）＋ 一個測試。不大，但不是零，而且不是純協定改動 —— 要送審。（ULIP2 ENGINEER 2026-08-27 提出，Master 複驗確認）|
| 2 | **λ 初值** | **雙方已收斂（2026-08-26）**：外部審查撤回原本的 0.1 主張，與我方一致。依據：MetaFind `2meth:87` 自述殘差設計是為了 "without disrupting the original embedding space"；Flamingo `content.tex:187-189` 是最同構的先例（凍結骨幹＋新分支＋殘差＋純量閘＋**初值 0**，目的明寫為初始化時行為等同原模型），拆掉該機制掉 4.2% 且訓練不穩 | **raw λ = 0.0，不加 tanh、不加 clamp、不加 LayerNorm**。MetaFind 寫的是 learnable scalar λ，**不要把 Flamingo 的 tanh(α) 一起搬過來**——tanh 會把有效閘限制在 (-1,1)，那是更大的模型改動。最小偏離就是 `nn.Parameter(0.0)` |
| 3 | **Stage 1 選 checkpoint 的依據** | ⚠ **本條已於 2026-08-27 整條撤回，由 §4.3 的 DEVIATION D-3 取代。撤回理由：本條的整個論證建立在一個已被推翻的前提上。** 它宣稱 OpenShape 用 in-batch contrastive accuracy 選 checkpoint —— **OpenShape 沒有這樣做**。`upstream/OpenShape/src/train.py` 的 `best_img_contras_acc` / `best_text_contras_acc` 只在 `:27-28` 初始化、`:46-47` 從 checkpoint 讀回、`:167-168` 存進每個 checkpoint，**全檔沒有任何一行更新或比較它們**。真正會存 `best_*` 的只有 `:244` `:247` `:297`，全部由 **held-out benchmark** 的準確率驅動。所以本條不是「另一個管別的事的方案」，是**一個錯的方案**。以下原文保留供對照，**不再有效**：~~撤回上呈，上游全部答完了（2026-08-26 二度回查）~~。我上一輪說「上游有獨立 benchmark、我們沒有，這塊答不了」，**那句話同樣是沒查完就下的結論**。OpenShape `src/train.py:190-201` 的訓練迴圈：跑滿 `max_epoch`（1000）**不早停**、**每個 epoch 存一次 `latest`**、每 `save_freq`（20）存一次快照、各 benchmark 各存各的 best。關鍵是它同時追蹤 **`img_contras_acc` / `text_contras_acc`**（`train.py:124,133`），那是**訓練批次自己的對比準確率**，不碰任何 held-out 資料。ULIP 同樣跑滿不早停（`main.py:212`） | ⚠ **已撤回。** 原建議是「不需要你在互斥選項裡挑，照上游做即可」—— 那句話同時是**越權**（替 Kyzen 宣告不需裁決）與**事實錯誤**（上游沒有那個做法）。**現行做法見 §4.3 的 D-3**：開發期在 80% training pool 內部切 dev-val 定政策，正式期鎖死重訓完整 80%、不中途挑、最後才開 20% 考一次 |
| 4 | **U-16** 塔權重共享 | **[已撤回並修正 2026-08-26]** 我上一輪寫「standing rule 預設應是 fully_separate」，**這個推論不成立，撤回**。理由：standing rule 只適用於**論文沉默**；這裡論文並不沉默，而是**自我矛盾**——正文 `2meth:34` 寫 "separate encoders"，但 **Figure 1 圖上明確標 `ULIP-2 (Shared)`，而且整張圖只畫一個 ULIP-2 方塊**（已親眼核對 `MetaFind.drawio.png`）。另外 DPR 在 MetaFind 裡是**雙塔檢索範式的概念出處**，不是 backbone 實作 authority（不像 ULIP-2／EGNN 是實作母體），所以 DPR 用兩份 BERT 推不出 MetaFind 的 ULIP-2 必須兩份參數。"separate encoders" 在架構敘述裡也可以只表示兩個邏輯塔／介面，不必然等於參數不相交。 | **維持現行 `shared_backbone_separate_fusion` 不變**，狀態 **PAPER-AMBIGUOUS ＋ USER CHOICE**。現行選擇反而是唯一能同時滿足兩處文字的讀法：一份骨幹（對上圖的 Shared）、兩個獨立 Fusion 介面（對上正文的 separate）。fully_separate 的證據**變強但沒到推翻**，保留為競爭假設。若你之後想改，那是新決策，不是修正錯誤 |
| 5 | **U-20/U-06** t_i 與 e_ij 編碼器 | 論文明文 "e.g., CLIP or BERT"（2meth:47）——作者自己給選單不給答案，確認無解 | 同骨幹 CLIP——不引入第二顆模型、同嵌入空間 |
| 6 | **U-21** Table 2 gallery＋資產數 | **新量測**：12,000 間房實際只用 **1,528 個 unique assetId**（train 10K 房=1,036）；ProcTHOR 官方庫 1,633；MetaFind 說 "3,000+"——**三個數字互相都對不上**，任何在手論文/資料都湊不出 3,000<br><br>🔵 **2026-08-28 重量，並補上先前沒建立的那一半：1,528 不只是「我們抓到的」，是「全部」。** 遞迴走完 `train/val/test.jsonl` 全 12,000 間房的 `objects` 樹（含 `children`）＋ `doors` ＋ `windows`：`objects` **1,467** · `doors` **40** · `windows` **21**，合計 **1,528**。house JSON 的頂層只有 `doors/metadata/objects/proceduralParameters/rooms/walls/windows` 七個鍵，`rooms` 只帶 `roomType`（4 種），`walls`/`proceduralParameters` 不帶 `assetId`。**沒有第八個地方可以再撈出資產，1,467 沒有漏抓。**<br>· 1,467 ≤ ProcTHOR 官方庫 1,633 **自洽**（166 件庫存資產在這 12,000 間房裡從未被抽中），所以缺口只有一個：MetaFind 的 "3,000+"。**[UPSTREAM FACT `procthor 03_generation.tex:36`] 逐字：「its database of 1633 household assets across 108 categories」** —— 上游一級來源與釋出資料同時否證 3,000。<br>· 附帶量到、與本行無關但影響 §6.2 切分討論：ProcTHOR 官方把資產切 train/val/test（train 996 / val 345 / test 315，兩兩交集僅 94~95）；**我們的 `scene_splits.json` 是對 12,000 間房重抽 80/20（seed 20260816），不是沿用官方資產切分**，因此測試房裡的資產大多在訓練房出現過。論文 `3experiments.tex:8` 只寫「80% / 20% of the data」，沒說切房還是切資產 → **這條是既有 IMPLEMENTATION CHOICE，本次只是量出它的後果，未改動。** | 維持 procthor scope、照實報三方差異；等 GPT 有無新讀法。**「1,467 是否漏抓」已結案：沒有漏。**<br>🔴 **「3,000 從哪來」也結案了：查無此數，四級來源全找過。**<br>① MetaFind `2methdology.tex:28` 逐字「a curated collection of **more than 3,000 unique assets**」——**沒有引用、附錄也沒有再提**。<br>② ProcTHOR 論文八個 `.tex` 全文搜 `3,000` / `3000`：**零命中**。其自述為 [UPSTREAM FACT `03_generation.tex:36`]「its database of **1633** household assets across 108 categories」。<br>③ 官方站點：ProcTHOR 1,633；AI2-THOR 平台本身 **"Over 2000 unique objects"**（ai2thor.allenai.org）——**都不是 3,000**。<br>④ 釋出資料實測，六種數法全部撞不到 3,000：`assetId` **1,467** · ＋doors 40 ＋windows 21 = **1,528**（上限）· `assetId` 去尾號類別 460 · 底名 100 · `object_id` 型別 93 · 物件**實例**總數 827,730（均 69/房）。<br>→ **分類 UNKNOWN，且是「來源不存在」型的 UNKNOWN，不是「我們還沒找到」型。**<br>⑤ **2026-08-28 直接問了引擎本人，天花板確定了。** 起一顆無頭 `CloudRendering` Controller（`ai2thor 5.0.0`，build `f0825767cd50d69f666c7f282e54abfe58f1e917`），`step(action="GetAssetDatabase")` 回 **1,934** 個 assetId（[OBSERVED DATA]，樣本 `Sink_1/2/3`）。對照：ProcTHOR-10K 實際擺出的 1,467 objects ＋ 40 doors ＋ 21 windows **全部是這 1,934 的子集，缺 0**；**1,528 ＋ 從未被擺過的 406 ＝ 1,934，帳完全平。** ProcTHOR 自述的 1,633 落在 1,528 與 1,934 之間，也自洽（產生器可抽的庫 1,633，其中 1,528 真的被抽到過）。<br>→ **這台機器上物理存在的家具上限就是 1,934。3,000 不在任何一層：不是資料、不是產生器的庫、不是引擎的全庫。**（先前用 `strings` 掃 `data.unity3d` 想數 —— **那個方法無效且我知道它無效**：`Fridge_19` 撈到 0 次而它確實存在，bundle 是壓縮的。此處記下來是因為 §9.7 講的正是這件事：解析度不足的儀器會給出乾淨但錯的數字。） 實作不受影響（我們用資料裡真有的 1,467）；**影響的是 §6.2 Table 2 與論文的可比性**：若論文真的檢索過 3,000 件資產，其檢索庫比我們大一倍，Table 2 數字不可直接對比。此限制須隨 Table 2 一併報告。 |
| 7 | **U-09** Table 1 gallery | MetaFind 沒說；上游評估是分類不是檢索、無協定可繼承——確認無解 | **雙協定都跑都報**（已實作，維持） |
| 8 | **Stage 2 超參數** | **原寫「S4 全無」是低估（2026-08-26 補查）**。EGNN 官方 QM9 有完整一組：`Adam` · `lr 1e-3` · `wd 1e-16` · `batch 96` · `epochs 1000`（`main_qm9.py:13,15,28,46`）。任務不同（分子回歸 vs 對比檢索）不能整包照搬，但不是沒有出處。另外 ESSGNN 的**架構**超參數更是有明確上游值，見 §3.4 對照表 | 架構值照 EGNN QM9（層數 7、pooling sum）；訓練值沿 Stage 1 的 AdamW＋cosine，lr 以 EGNN 的 1e-3 為起點小掃描，全部記 CHOICE |
| 9 | Stage 2 正例分布落差 | 無上游做 layout-aware 檢索，確認無解 | 先 leave-one-out（論文可辯護的最小讀法），落差列為已知限制 |
| 10 | 48K vs 46,832 vs 我們 46,052 | 無解（論文不解釋 48K 從哪來） | 不擋工；報告用實際語料數 |
| 11 | **Stage 1 學習率** | ✅ **已於 2026-08-27 由 Kyzen 拍板：5e-4 起跑**，第一輪 sweep `2.5e-4 / 5e-4 / 7.5e-4 / 1e-3`，**3e-3 不入第一輪**。記為 **USER-APPROVED IMPLEMENTATION CHOICE**，上游列為佐證而非依據。⚠ 承重的只有 **OpenShape supp:190**（同一顆 backbone、同一個尺寸、同一個目標，Type B）；**Point-BERT `:216` 的 5e-4 是 MPM 遮罩點建模，與我們的 tri-modal 對比對齊不是同一個任務 → Type C，不是第二張獨立的票**（ULIP2 REVIEWER 2026-08-27 指出）。⚠ **日後不得把這個 sweep 描述為「照 ULIP 的設定」** —— ULIP 論文 `main.tex:370` 是 `1e-3`、官方腳本 `pretrain_pointbert.sh` 明傳 `3e-3`，Kyzen 把前者收進 sweep 上緣、把後者排除。照的是機制，偏離的是數值。以下原始四候選分析保留供對照：**四個候選，各有一手出處（2026-08-26 查全）**：`1e-3` = ULIP-1 論文 main.tex:367-370，也是 OpenShape config default；`3e-3` = ULIP 官方腳本 `pretrain_pointbert.sh` 明傳、`main.py:52` default，但那條腳本跑的是 **ULIP-1 PointBERT 8192 點**，不是我們的 10k colored；`5e-4` = **OpenShape supp:190 明說「32.3M 版 PointBERT 用 5e-4，其他模型 1e-3」**，且 ULIP-2 yaml 的 optimizer 區塊也是 5e-4（該區塊 tri-modal 訓練不讀，但數字同源）。**實測：我們載入的 PointBERT 是 32.5M**，正落在 OpenShape 特地為它調小 lr 的那個量級 | ~~建議 5e-4~~ → **已拍板，見左欄。** 原建議理由仍成立：它是唯一「針對 PointBERT 這個 backbone、在這個參數量級」明確給出的值 |

---

## 11. 給審查者（GPT）的指令（第二輪範圍已收斂）

**第一輪結果**：3 項 P0 全數採納並修正，P1／P2 大部分採納（見下方對照）。
⚠ **本句是 2026-08-26 的狀態，已過期。** 原文：~~未採納的只有一項：外部審查建議把 U-14 的 gallery/eval 平均也視為可推導~~。該判斷本身仍成立（ULIP-2 那句只講訓練取樣，沒有背書評測聚合方式，所以 gallery/eval 平均維持 CHOICE），但「未採納的只有一項」這個計數在 2026-08-27 之後不再正確 —— 見本節各條的撤回與更新。

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
