# Stage 1 從零對照審計（2026-09-03）

Kyzen 的指令：「把它當作一個新專案去處理，什麼都不知道不清楚，論文說什麼就怎麼設計，找出錯誤。」

做法：只讀 A0（`docs/paper/metafind_source/metafind_arxiv_v1.html`，含 Figure 1 / Figure 2），對每一個細節寫下「論文原文 → 只憑原文會怎麼設計 → 我們的程式實際怎麼做（檔案:行）→ 判定」。判定只有四種：

```
✅ 一致       論文有講，我們照做
⚪ 論文沒講   我們填了一個值；不能叫論文設定
🟡 有疑點     論文的圖或文與我們的做法之間有張力，尚未測
🔴 錯誤       與論文原文相反，或邏輯上站不住
```

程式碼引用一律是 2026-09-03 的 HEAD（`6ef3dca` 之後）。

---

## A. 資料

| # | 論文原文（逐字） | 只憑原文的設計 | 我們的實作 | 判定 |
|---|---|---|---|---|
| A1 | §2.3 "the Objaverse-LVIS dataset, which comprises approximately 48,000 distinct 3D assets" | 48K 資產 | 官方 LVIS 清單 46,052 → 點雲 ∩ 渲染 ∩ 標註 = **45,692** | 🔴 **偏離**，且從官方清單做不到 48K；必須一直掛著 |
| A2 | §2.3 "Each asset is rendered from 11 orthogonal viewpoints" | 11 個正交視角 | **12** 視角（OpenShape 三圈 × 4，`render_blender.py`） | 🔴 **偏離**；但今天量了 11-of-12 兩邊都換：57.8/84.7/78.4/96.3/99.6/94.2/100.0 對 12 視角 58.0/84.6/78.8/96.5/99.6/94.1/100.0，**指紋不動**，所以不是差距來源 |
| A3 | §1 / §2.3 "annotated using GPT-4o" | GPT-4o 標註 | gemma-4-12B-it | 🔴 **偏離**（D-2）；未量化影響 |
| A4 | Figure 2 的紀錄有 13 個欄位：`category, synset, width, length, height, volume, mass, description, materials, onCeiling, onWall, onFloor, onObject`；範例 `width: 30, length: 30, height: 40`（整數）；`description` 一句 135 字元 | 標註 schema 照這 13 欄 | 13 欄全部都有（`annotations/*.json`）；尺寸存全精度浮點，字串化時取一位小數 | ✅ schema 一致；尺寸精度 ⚪（量過：粗到 10 公分只動 0.1 分） |
| A5 | **Figure 1 的文字輸入畫的是** `Platform Bed` 換行 `{size:..........}` —— 類別名 + 一個 JSON 片段 | 餵給文字塔的字串 = **類別 + 結構化欄位**，短 | `TEXT_TEMPLATES["v2_cm"]` = **160 字元自由描述放最前面** + 類別 + 材質 + 尺寸 + 擺放，中位 68 token（上限 77） | 🟡 **圖不支持我們的格式**。圖上沒有長描述。`attrs_v1`（純填表）比現行更接近圖。量過：純填表 text-only 8.3、現行 58.0、論文 13.8 |
| A6 | §2.2 "each available modality is independently encoded using the ULIP-2 backbone" | 文字、影像、點雲各自過 ULIP-2 對應的塔 | 文字 → OpenCLIP ViT-bigG-14 text；影像 → 同 model image；點雲 → PointBERT + `pc_projection`（`ulip_backbone.py:438/447/401`） | ✅ |
| A7 | 論文全文**沒有**說 11 張圖怎麼變成一個影像模態 | ——（見 C4） | 12 張各自編碼後 **raw 平均**成一支向量（`encode_text_image.aggregate`, `view_aggregation.method = mean`） | ⚪；而且 **Figure 1 畫的是 I1…IK 多支向量進 Fusion**，見 C4 |
| A8 | 論文全文沒有說點雲怎麼取樣 | —— | 從 GLB 面積加權取 10,000 點 + 貼圖插值 RGB，xyz 減質心除最大半徑（ULIP `pc_norm` 逐字），RGB [0,1]（`pointclouds.py:138,598`） | ⚪；與上游 ULIP-2 的 10k xyzrgb 一致 |
| A9 | 論文沒有說渲染背景、解析度、相機 | —— | Blender Cycles 512², 透視 35mm, 距離 1.2, RGBA 合成到**黑底**（`view_io.py`）, CLIP 標準前處理 224 | ⚪ |
| A10 | §2.3 "annotated using GPT-4o. These annotations provide rich textual descriptions detailing attributes such as object category, size dimensions, materials, and placement constraints." | 標註**內容**要涵蓋四類屬性 | 涵蓋 | ✅（內容）；字串化 ⚪（見 A5） |

## B. 骨幹

| # | 論文原文 | 只憑原文的設計 | 我們 | 判定 |
|---|---|---|---|---|
| B1 | §2.2 "both leveraging the ULIP-2 embedding backbone"；**Figure 1 的方塊標 `ULIP-2 (Shared)`，整張圖只有一個**；§2.4 "separate encoders for the query and gallery" | 一份 ULIP-2；「separate encoders」指的是圖上的兩個外殼 `Query Encoder` / `Item Encoder`，而不是兩份 ULIP-2 | 一份骨幹（`shared_backbone_separate_fusion`） | ✅ 骨幹一份 |
| B2 | 論文**沒有**寫文字／影像塔凍不凍；§3.4 "full encoder fine-tuning yields better performance by allowing earlier layers to adapt to modality-aware supervision" 說**編碼器有在訓** | 至少有編碼器層在訓；哪些不明 | CLIP 文字／影像凍結（梯度實測恆 0）；PointBERT + `pc_projection` 訓（`_apply_train_scope`） | ⚪ 論文沒指名；🟡 **§3.4 那句支持「編碼器早期層有適應」，我們只讓點雲塔適應**。文字／影像塔完全凍結是我們從 ULIP-2 lineage 借的，不是論文 |
| B3 | 論文沒寫從哪個權重起跑 | —— | 官方釋出 ULIP-2 checkpoint（sha 驗證） | ⚪；Kyzen 裁定 |
| B4 | ULIP-2 自己的 `logit_scale`（τ 可學，初值 0.07）存在 checkpoint 裡 | 論文 τ=0.5 固定 → 不用它 | 我們的 loss 自帶 `logit_scale` buffer = log(1/0.5)，ULIP 的那顆不參與 | ✅ |

## C. 塔與融合

| # | 論文原文 | 只憑原文的設計 | 我們 | 判定 |
|---|---|---|---|---|
| C1 | §2.4 "MetaFind employs a dual-tower architecture with separate encoders for the query and gallery." | 兩個塔 | `QueryTower` / `GalleryTower` | ✅ |
| C2 | **Figure 1：`ULIP-2 (Shared)` 方塊裡面只畫了一個 `Fusion Layer`**，Query Encoder（粉線）和 Item Encoder（黑線）都接進同一個方塊；caption "each 3D asset in the repository is pre-encoded independently by ULIP-2 into a fixed vector" | 照圖：**一個 Fusion Layer，兩塔共用**；兩塔的差別只有 (a) query 側有遮罩 (b) query 側加 ESSGNN | **兩份獨立參數的 Fusion**（`MetaFindDualTower.__init__`：`QueryTower(cfg)` + `GalleryTower(cfg)` 各建一份）。`fully_shared` 選項存在但 `freeze_gallery` **拒跑**，理由寫「凍 gallery 與訓 query fusion 不能同時成立」 | 🟡 **重大疑點**。那個拒跑理由是我們的推論，不是論文：§2.6 說凍 gallery 是 "to reduce training costs and preserve asset embedding consistency"，§2.7 說 gallery "precomputed and cached"——**把 gallery 向量快取起來就同時滿足這兩句**，不需要第二份 Fusion。圖的證據偏向一份。**從未測過** |
| C3 | §2.2 "integrated via a fusion layer, such as mean pooling, an MLP, or a Transformer-based module"；§3.4 "the final selected Transformer" | Transformer fusion | Transformer（`fusion.py`） | ✅ 種類 |
| C4 | **Figure 1：每個編碼器輸出 K 支向量（P1…PK, I1…IK, T1…TK）一起進 Fusion Layer**；但 **Eq. 6 寫的是 `Fusion(e_text, e_img, e_pc)`**——每個模態一支 | 兩個證據方向相反：圖畫 K 支，公式寫一支。圖上的 K 也可能只是「一批 K 個資產」的示意（E1…EN 與 K 不同也支持這讀法） | Fusion 吃 **3 支**向量（text 1、image 1、pc 1；`fusion._stack` → `(B, 3, D)`）；另有 `image_tokens=12` 的實驗 arm（P3） | ⚪ **UNRESOLVED**，主線維持每模態一支（Eq. 6 直接支持）。「每視角一個 token」只是一個實驗 arm，不是圖的定論；此列先前寫成「圖偏向 K token」是講過頭，2026-09-03 晚間修正 |
| C5 | Transformer 的層數、頭數、FFN、dropout、位置向量、讀出方式：論文**全部沒寫** | —— | 2 層 / 8 頭 / FFN 2048 / dropout 0 / 每模態一個可學位置向量 / 輸出 = active slot 平均（`fusion.py:176-183, 295`） | ⚪ 六個值全是我們的 |
| C6 | §2.6 "Rather than zero-padding, we apply masked embeddings" | 缺席模態放一個 masked embedding，不是零 | learned mask token（`nn.Parameter`，std 0.02，不做 weight decay）替換缺席 slot（`fusion._stack`） | ✅ 不補零；⚪ token 是可學還是固定，論文沒說 |
| C7 | 論文沒說缺席 slot 的輸出要不要參與最後的向量 | —— | 參與注意力**也參與平均**（`include_absent_slots=True`）：text-only 的輸出 = (text slot + 2 個 mask slot 的輸出) / 3 | ⚪；🟡 讀法二（只平均存在的 slot）同樣合理，未測 |
| C8 | 論文沒說進 Fusion 前要不要正規化 | —— | **不正規化**（`fusion.py:277` 直接 `x + modality_pos` 進 Transformer） | ⚪；🟡 量到訓練後 pc 範數 139 對 text 37 / image 40，點雲在數值上主導。打亂畫廊 pc → 全部掉到個位數 |
| C9 | §2.4 "The gallery encoder is modality-complete" | gallery 三模態必須齊 | `GalleryTower.forward` 缺一個就 raise | ✅ |
| C10 | §2.4 "the query encoder remains flexible: It accepts any subset of modalities" | query 任意子集 | `present` mask 七條件 | ✅ |

## D. 訓練

| # | 論文原文 | 只憑原文的設計 | 我們 | 判定 |
|---|---|---|---|---|
| D1 | §2.6 "both query and gallery encoders are trained on large-scale object-level data from Objaverse-LVIS, where each asset has full modality inputs" | 兩塔都訓；每個資產三模態齊 | 兩份 Fusion 都訓、PointBERT 訓、CLIP 凍（見 B2） | ✅ 兩塔都訓 |
| D2 | §2.6 "each modality in the query has a 30% probability of being independently masked" | 每個 query、每個模態獨立 Bernoulli(0.3) | `sample_modality_mask(p=0.3)` per-sample per-modality；全遮 2.7% 允許（`allow_all_masked=True`） | ✅；全遮是否允許 ⚪ |
| D3 | §2.6 "The gallery encoder is trained to be modality-complete" | gallery 側不遮 | 不遮 | ✅ |
| D4 | Eq. 5：`L_pre = −log exp(sim(f_q(Q), f_g(A))/τ) / Σ_{A'∈B} exp(sim(f_q(Q), f_g(A'))/τ)`，"B denotes the gallery batch" | 單向 q→g InfoNCE，負例 = 同批 gallery | `MetaFindContrastiveLoss(bidirectional=False)`：`CE(scale · q̂ĝᵀ, arange(B))` | ✅ |
| D5 | §3.1 "The temperature is 0.5 for all experiments." | τ = 0.5 | 0.5 固定（buffer） | ✅；「不可學」⚪（論文說是 hyperparameter） |
| D6 | §2.1 "sim(·,·) denotes the similarity function"——**沒有定義** | —— | cosine（`F.normalize` 兩邊） | ⚪；🟡 量過：換點積 → 9.5/11.7/9.4/13.8/14.0/13.5/15.9（image 正好 11.7，pc 崩到 9.4）。sim 是活的軸 |
| D7 | 論文**沒有**寫 query 和 gallery 是不是讀同一份紀錄 | —— | `split_embeds`：沒有 pack 時 `return gallery, gallery`，**同一個 dict、同一批張量**（連 pc 都是同一次 PointBERT forward） | ⚪ 論文沒講；🔴 **但邏輯上站不住**，見 E1 |
| D8 | 論文沒寫 optimizer / lr / batch / epochs / warmup / wd | —— | AdamW, lr 5e-4, wd 0.1（bias/LN/mask token 不 decay）, batch 64, cosine + 1 ep warmup, 5 輪（上限 250） | ⚪ 全部借 ULIP-1 官方；Kyzen 核可 |
| D9 | 論文沒寫資料增強 | —— | 文字／影像是快取向量，無增強；點雲固定一份取樣，無增強 | ⚪ |
| D10 | §3.1 "80% … for training and … 20% for testing"——沒有 validation | —— | 80% 內再切 12.5% 當 dev_val 選 checkpoint；20% 封存（D-3） | ⚪ 必要的補充；已登記偏離 |
| D11 | 論文沒寫 checkpoint 怎麼選 | —— | dev_val 七條件平均 R@1 取最佳 | ⚪ |

## D′. 公式（逐字）與程式對應 —— 推理從這裡起

[KYZEN 2026-09-03]「補充公式很重要，可以看公式去推理。」以下每條都是 HTML 逐字，後面接程式碼對應與從公式本身能推出的結論。

**Eq. 1（§2.1，檢索）**
```
A* = argmax_{A ∈ 𝒜} sim( f_query(Q), f_gallery(A) )
```
- `sim` 沒定義（§2.1 "denotes the similarity function"）。程式：`normalize_for_scoring` 後點積 = cosine。
- 公式只說 `f_query` 與 `f_gallery` 是兩個函數，**沒說它們參數相不相交**（見 C2）。

**Eq. 5（§2.6，Stage 1 損失）**
```
L_pre = − log  exp( sim(f_query(Q), f_gallery(A)) / τ )
               ─────────────────────────────────────────
               Σ_{A' ∈ B}  exp( sim(f_query(Q), f_gallery(A')) / τ )
```
"where τ is a temperature hyperparameter and B denotes the gallery batch."；§3.1 "The temperature is 0.5 for all experiments."

程式（`losses.py`）：`q̂ = normalize(f_q)`, `ĝ = normalize(f_g)`, `logits = (1/τ)·q̂ĝᵀ`, `loss = CE(logits, arange(B))`。第 i 列的 CE = `−log[exp(q̂ᵢ·ĝᵢ/τ) / Σⱼ exp(q̂ᵢ·ĝⱼ/τ)]`，與 Eq. 5 逐項相同（分母含正例本身，CE 天然如此）。**單向**：沒有 `logitsᵀ` 那一項。✅

**從 Eq. 5 能推出什麼（E1 的代數）**：
- 對固定的 `f_gallery`，Eq. 5 對 `f_query(Q)` 的最小值在 `sim(f_query(Q), f_gallery(A)) = 1`，即 `f_query(Q) ∝ f_gallery(A)`。
- 若 Q 與 A 是**同一份紀錄**（`same_record`），存在一個平凡解 `f_query ≡ f_gallery`，梯度會往那裡走；訓練後 `full` 條件（Q = A 的三模態全開）自然 `sim → 1`，R@1 → 100。**這是公式推出來的，實測 0.9998、cos 0.9989 只是確認。**
- 論文 `full` = 51.7。所以論文評估時 `Q ≠ A 的紀錄`——至少一個模態是另一份觀測。**公式 + Table 1 就足以排除 `same_record`。**

**Eq. 5 與 τ = 0.5 的數值意義**：`1/τ = 2`，logits 落在 [−2, 2]。B = 64 時，全部負例正交（sim = 0）的損失下限 = `−log[e² / (e² + 63)] = 2.254`；隨機 = `ln 64 = 4.159`。先導的損失貼在 2.25 上（`pilot10.json`），**代表 q̂ 與所有負例幾乎正交、與正例幾乎重合**——正是平凡解。

**Eq. 6（§2.6，Stage 2 查詢）**
```
e_query = Fusion(e_text, e_img, e_pc) + λ · e_layout
```
"λ is a learnable scalar"。程式：`dual_tower.py` `fused + self.lam * layout`；`layout is None` 時回傳 `fused`（Table 1 的 layout-free 評估）。✅；λ₀ 沒給。

**Eq. 7 / 8（§2.6，Stage 2 損失）**
```
L_q2g = − log exp(sim(e_query, e_gallery)/τ) / Σ_{e'_gallery ∈ B} exp(sim(e_query, e'_gallery)/τ)
L_g2q = − log exp(sim(e_gallery, e_query)/τ) / Σ_{e'_query ∈ B} exp(sim(e_gallery, e'_query)/τ)
L_layout = ½ (L_q2g + L_g2q)
```
程式：`bidirectional=True` → `½(CE(logits) + CE(logitsᵀ))`。`logitsᵀ` 的第 i 列分母是 `Σⱼ exp(q̂ⱼ·ĝᵢ/τ)`，恰為 `Σ_{e'_query∈B}`，前提是 `sim` 對稱（cosine 是）且 query batch 與 gallery batch 是同一組配對。✅

**沒有公式的東西**（論文用文字帶過）：Fusion 的內部（§2.2 只列名字）；masked embedding 是什麼向量（§2.6 "we apply masked embeddings"）；影像多視角怎麼變一支（沒有任何句子）；`Pooling`（§2.5 未定義）。這四個在公式層級就是空白，只能用實驗填。

## E. 邏輯檢查（不靠任何設定，只靠 Table 1 的數字）

**E1. 「query 讀的是 gallery 自己那一筆」與 Table 1 的 full = 51.7 不能同時成立。**

- Table 1 是 exact-instance（§3.2 "identical embeddings … inflated" 只在自己找自己時有意義）。
- `full` 條件 = 三模態全在，沒有遮罩。
- 若 query 與 gallery 讀同一份觀測：
  - **共用一個 Fusion（Figure 1 讀法）**：`f_q(A) ≡ f_g(A)`，同函數同輸入，R@1 = 100 減重複資產。
  - **兩份 Fusion（我們）**：Eq. 5 直接把 `f_q(A)` 拉向 `f_g(A)`；10 輪先導在 36,554 畫廊上已到 **0.9998**，cos(q_A, g_A) = 0.9989。
- 論文的 full 是 **51.7**。
- 所以：**不管 Fusion 是一份還是兩份，論文評估時 query 看到的觀測不可能是 gallery 自己那一筆。** 這不是設定，是 Table 1 自己逼出來的。

**E2. 同一顆釋出 ULIP-2、零訓練，論文 text→PC 0.1 / image→PC 0.1，我們 24.5 / 58.4。** 文獻 [30] 就是 ULIP-2。差 245–580 倍，在任何 MetaFind 元件介入之前。與 E1 一致：論文的 query 觀測比 gallery 紀錄弱很多。

**E3. 論文的 ULIP 列單調下降（pc 97.9 > T+PC 33.9 > I+PC 22.6 > full 6.4）** 只在「畫廊 = 純 PC 向量」時出現（我們 B1 有這形狀，B2 沒有）。所以 baseline 的畫廊是 PC，query 是可用模態平均——這半句與 §3.1 "simple mean pooling layer" 一致。

## F. 錯誤與疑點清單（照重要性排）

| 序 | 項目 | 級別 | 依據 | 下一步 |
|---|---|---|---|---|
| 1 | Query 觀測 = gallery 紀錄（`same_record`） | 🔴 **邏輯錯誤** | E1 + E2 | 主線改為 query 讀第二份觀測（B / C arm）；機制已接線（`--query-image-policy`、query pack text/pc） |
| 2 | 文字字串化把 160 字元自由描述放最前面 | 🟡 圖不支持 | A5；量過 58.0 → 8.3 | 主線候選 `attrs_v1`；重編文字快取後重訓 |
| 3 | Fusion 吃 3 支平均向量而非每視角 token | 🟡 圖畫的是 K 支 | C4 | 用快取 `views` 做「每視角一個 token」的 fusion arm，不用重編碼 |
| 4 | 兩份 Fusion vs 一份共用 | 🟡 圖畫一份 | C2 | 解除 `fully_shared` 的拒跑（凍 gallery = 快取向量），跑一個 arm |
| 5 | 進 Fusion 前不正規化 | 🟡 量到 pc 主導 | C8 | 重訓 arm：per-modality L2 |
| 6 | CLIP 文字／影像塔完全凍結 | 🟡 §3.4 說 "earlier layers adapt" | B2 | 排後；先做 1–5 |
| 7 | 缺席 slot 參與平均 | ⚪ | C7 | 排後 |
| 8 | 12 視角 / Gemma / 45,692 | 🔴 偏離但已知 | A1–A3 | 11-of-12 已證明不是差距來源；Gemma 待 300–500 筆對照 |
| 9 | sim = cosine | ⚪ | D6 | 與 5 一起看（正規化後 cosine / dot 應收斂） |

## F′. 旁證：CAMERA（薛聖群）的 ULIP 筆記，2025-06-26 / 08-07 / 08-12

Kyzen 問「Camera 的論文有沒有可以參考的」。三頁都是 **ULIP-1 在 ShapeNet / Text2Shape** 上的實驗，不是 MetaFind 證據；但三件事對我們的設計有用：

1. **ULIP-2 的 caption 是每張渲染圖各生一句**（他們引 ULIP-2 原文："For each rendered image, we employ BLIP-2 … to generate 10 detailed descriptions independently, which are then ranked using CLIP-VIT-Large … we use the top 1"）。這正是 Figure 1 畫 T1…TK 的來源：**K 個文字向量 = K 張圖各一句**。我們的 `description_candidates`（5 句、CLIP 排名）是同一個機制，所以「替代描述」當第二份觀測有上游依據。
2. **他們的 fusion 對 (1+K) 個 token 用 CLS 讀出**（RAG 版本），不是平均——C7 的另一個具體候選。
3. **ULIP 訓練時點雲有增強**（random dropout / scale / shift / rotate）；我們 Stage 1 沒有（D9）。論文沉默；上游候選。

他們的 T2S R@1 = 13.50 是 14,966 件 ShapeNet 池、ULIP-1 訓 250 輪的數字，與 MetaFind 的 13.8 只是數字相近，**不得引為證據**（見 §7.5）。

## G. 這份審計改變了什麼判斷

之前把 `same_record` 當成「論文最字面的讀法（A arm）」與 B、C 並列。**E1 把它從候選裡拿掉**：它與 Table 1 的 full = 51.7 矛盾，不是候選，是排除項。主線必須是 query 讀第二份觀測；剩下要選的是**哪一種**第二份觀測（B 或 C）。

Figure 1 另外給了兩個之前沒認真對待的證據：Fusion Layer 只畫一個（C2），每模態多支向量進 Fusion（C4）。兩個都能用快取資料便宜測。
