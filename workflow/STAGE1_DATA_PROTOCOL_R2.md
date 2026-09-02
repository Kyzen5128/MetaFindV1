# Stage 1 資料協定 — 修正版（2026-09-02，回應 Codex 第一輪核對）

標籤：**VERIFIED** 可由論文、程式、資料或 log 直接證實；**ASSUMPTION** 有依據但官方細節不足；
**RECOMMENDATION** 我的建議；**MISSING-EVIDENCE** 仍缺程式或資料證據，不得凍結。
與第一版相比改了什麼，見最後一節「修正清單」。凍結前必做的事，見「凍結前清單」。

---

## 一、洩漏的界線

**VERIFIED**
- Gallery 含正確資產不是洩漏；洩漏是 query 的原始觀測直接參與了它要找的那一列 gallery 向量的建構。
- MetaFind `2methdology.tex:75`：Stage 1 只寫 query 端每模態獨立 30% 遮罩、gallery modality-complete；**沒有交代兩端是否用不同觀測**。
- `3experiments.tex:24`：「identical embeddings … inflated」講的是**別的模型的 embedding**，不是原始觀測；MetaFind 被拿來對比。
- Figure 1 caption：gallery「pre-encoded independently … into a fixed vector」；query 是「the user's input」。
- 我們的實作與量測：`same_record` 構造下，零參數 raw-mean 在 dev 池 99.56、在 45,692 全池同句 94.05／換句 60.30；訓練後 dev 池 89.36。
- 程式：`stage1.py` 的 `--query-observation {same_record, second_observation}` 現為必填並與 `--query-pack` 互檢（commit b04e004），並寫進 checkpoint metadata（今日）。

**ASSUMPTION**
- Table 1 的 query 用資料集自己的模態。哪一份觀測，論文沒寫。

**修正後的說法（取代第一版「13.8 與字面構造不相容」）**
> 論文的 13.8 與**我們目前的** same-record／modality-complete 實作及其對照結果**不一致**（empirical discrepancy）。
> 不能據此斷定作者一定用了 pure-PC gallery 或不同觀測；separate fusion heads、不同訓練狀態、未公開的 wrapper 都可能改變結果。
> 這是一個尚未歸因的差異，不是數學上的不可能。

**分離等級（RECOMMENDATION，名稱依 Codex 修正）**
| 等級 | query 與 gallery 的關係 | 量到什麼 | 可否稱 observation-disjoint |
|---|---|---|---|
| 0 same-record | 同向量 | 認得自己 | 否 |
| 1 same-evidence paraphrase | 同一組圖生成的另一句／同 mesh 重抽點 | paraphrase／取樣穩健性 | 否（文字 cos 0.85、點雲 cos 0.944） |
| 2a generator-disjoint external caption | 不同產生器（官方 BLIP／Azure 縮圖句） | 對短句、異模型描述的檢索 | **未確認**：縮圖是否與 gallery 用的渲染同源，要查清才能稱 observation-disjoint |
| 2b held-out-view caption | 只看一張不在 gallery 平均裡的視角寫的句子 | 從部分觀測找回 | 是（證據來源不重疊） |
| 2c human / external observation | 使用者自己的描述、照片、掃描 | 部署情境 | 是 |
| 3 cross-modal only | text → pc-only gallery（CAMERA 式） | 純跨模態對齊 | 無自匹配（但不是 MetaFind 的 gallery） |
| 4 task-level | 場景挖洞、多正例 | 能否填進去 | 無自匹配 |

---

## 二、視角

**VERIFIED**
- MetaFind：11 個正交視角用於資料準備與 GPT-4o 標註（`2methdology.tex:28`）；image pooling 與 Table 1 的 query-view 構造**未公開**；一資產一個固定 gallery 向量（Figure 1）。
- ULIP-2 §4.1：Blender 12 張、每 360/12 度；每張 BLIP-2 10 句、CLIP-ViT-L 排序、取 top-1。
- **ULIP-2 §3.3（升級為 VERIFIED）**：訓練時「randomly sample its 2D rendered image I ~ render(O), with its BLIP-2 generated language description T ~ blip2(I)」——每步隨機抽一張，並用**那張圖自己的** caption。
- 另一件事、分開寫：官方釋出的 `objaverse_lvis/*.npy` **沒有保存逐視角 caption**（只有 `text`、`blip_caption`、`msft_caption`、`retrieval_text` 各一份，皆縮圖層級）。
- 官方 `image_feat` 與我們特徵的相容性（`tools/probes/official_image_feat_compat.py`，1,930 件同資產）：
  norm 43.07±1.94 vs 43.46±1.64（同尺度、都未正規化）；同資產最近視角 cosine 平均 0.8515（p05 0.72、最低 0.31），異資產 0.4968；
  12 視角平均向量兩邊 cosine 0.8937；我們的文字當 query → 官方 12 視角平均 gallery R@1 **76.74**、我們的 **80.88**（池 1,930）。
  **判定：同一個特徵空間、不需額外投影；但相機不同（單圈 vs 三圈），數值不可互換。**
- ProcTHOR 的 1,467 件資產是 AI2-THOR 目錄，與 Objaverse-LVIS 不同資產庫。
- 實測：query 視角在不在 gallery 平均裡值 10.09 分；單張 vs 12 張平均當 query 52.2 vs 52.7；官方像素 vs 我們像素 58.53 vs 54.84（p=0.099）。

**RECOMMENDATION（措辭依 Codex 修正）**
- 主要方案：**在目前同一批 12-view cached features 的限制下，最簡單且可稽核的方案**是「1 張當 query（`uid_seed(uid) % 12`）、其餘 11 張平均當 gallery」。它不是唯一可 defend 的設計；外部圖片、不同 renderer、真實場景裁切都可構成其他協定，只是目前沒有那些資料。
- 一資產一列；聚合維持 mean 並記為選擇；不用 ProcTHOR 重渲 Objaverse。

---

## 三、80/20 與 gallery

**VERIFIED**：兩資料集 80/20（`3experiments.tex:8`）；候選池大小論文**未公開**；先按 uid 切、同 uid 三模態同 split（`splits.py`）；官方 LVIS 名單 46,052、我們 45,692 是其子集。

**RECOMMENDATION**：主報 B（test query → 45,692 全 gallery）、副報 A（test → test-only 9,138）；兩者都保存精確 query 數、候選數、chance rate；報告時明寫「論文未公開候選池大小」。訓練資產當 distractor 不是 test 洩漏，方向未知，所以兩者都報。

---

## 四、ULIP-2 官方資料

**VERIFIED**（開檔）：約 80 萬件 = 整個 Objaverse；LVIS 名單 46,052；`image_feat (12,1280) float32`、`thumbnail_feat (1280,)`、`xyz/rgb (10000,3) float16`；無像素；無逐視角 caption；檔內無 checkpoint id。
**VERIFIED（程式）**：`prompt_avg` 是 OpenShape 的 **prompt-template 平均**——`/home/kyzen/upstream/OpenShape/src/data.py:33`：`self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"`。**它不是多句 caption 的平均。第一版用它支持「gallery 用多句平均」，撤回。**

**RECOMMENDATION（條件式）**：官方 `image_feat` 通過上述 space-compatibility 測試後，可作為凍結的 gallery 圖片表示（同空間、無需投影）；但不與我們的特徵混用（相機不同），且任何需要像素的步驟（單視角句、重標註）官方沒有。點雲可直接用。

---

## 五、文字

**VERIFIED**
- ULIP-2 逐視角、每張 10 句、CLIP-L 排序取 top-1（§4.1）；MetaFind 四個欄位且 "such as" 非窮舉（`2methdology.tex:28`）。
- 我們：JSON 留磁碟、一句進編碼器；tokenizer 77 tokens（含起止符）、真實 BPE 計數、超過拒收；3,000 件抽樣 canonical 句 p50 67 / p90 72 / max 75 / 超過 77 者 0。
- 排序器是 CLIP-ViT-L（`describe_rank.py`），不是檢索用的 ViT-bigG。
- query pack 有 68 列候選句與 canonical 字串相同（bug）：**已修**（`tools/make_query_pack.py` 跳過 byte-equal 候選；`--limit` 改寫獨立 tag；resume 核對 row）。要重建文字 shard 才生效。

**撤回／修正**
- 「官方 caption 低 15 分所以品質差、不適合當 gallery」：**撤回**。15 分可能來自句長、視角、對齊、prompt、checkpoint 或構造，不能等同語意品質。官方 caption 優先作 external-query arm；作 gallery 留作 ablation。
- 「每句 75 tokens、多句平均」：**不凍結**。先在 test 子集做長度消融：short / medium / current-long / multi-sentence-mean，再決定重標註格式。

**RECOMMENDATION**：文字 query 拆四個子手臂分開報（same-evidence paraphrase；generator-disjoint external caption；held-out-view caption；human/external 若有）。只有後兩者在證據來源不重疊時稱 observation-disjoint。
Schema 與 prompt 草案維持第一版（canonical / paraphrase / single_view / short），但 single_view 需要**獨立一次 image call**，成本不能假設等於原 80 小時；先 20–100 件 prompt QA → 500–1,000 件 token/欄位/幻覺稽核 → test 子集 retrieval pilot → 再全量。

---

## 六、點雲

**VERIFIED**：重抽點 cos 0.944、0.9 分（取樣變異）；ULIP 官方 loader 有 point dropout / scale / shift / jitter 增強；MetaFind 未提第二份點雲。
**新增 VERIFIED（`tools/probes/depth_shell_conventions.py`）**
- Unity 左手系：對 400 件 Objaverse 測試雲做 z 鏡射，PointBERT 嵌入 cosine 平均 **0.9963**（最低 0.9812）、自檢索 top-1 **100%** → 編碼器對鏡射不敏感，**深度殼不需要鏡射**（幾何單元測試已做）。
- 灰色常數 0.5 vs 0.4：1,439 件 ProcTHOR 殼，cosine 平均 **0.9562**（p05 0.889、最低 0.78）、自檢索 top-1 只有 **93.47%**（最差排名 10）→ 常數**有影響**；已改為 ULIP 慣例 0.4（`ulip_backbone.py`），Stage 2 索引重建中。

**RECOMMENDATION（三層，依 Codex）**：same full cloud = identity control；full resample = sampling sensitivity（不當 headline、可 on-the-fly 或只存 seed）；partial/depth cloud = different-observation arm（單視角深度雲，與 ProcTHOR 深度殼同程式）。

---

## 七、loss

**VERIFIED**：B=64（63 個負例）、τ=0.5 固定、cosine；解析參考值 chance 4.1589、負例正交 2.2540、實測負例 cosine 0.0007 → 2.2553、simplex 2.2257；實際 same-record 2.3354–2.3427（acc 0.98–1.0）、second-observation 2.41–2.43（acc 0.95–0.97）。
**撤回**：第一版的「完美排序者可達下限 L\* = mean log(1+Σ exp((s_ij−s_ii)/τ))」——用當前相似度矩陣算就是當前 loss 本身，不能證明訓練榨乾了資料。
**改用的診斷**：(1) 實際 s_pos、hardest negative、負例分布；(2) top-1 margin = s_pos − max s_neg；(3) 訓練前後 margin 與檢索指標的變化；(4) 固定 τ、B 的解析參考值；(5) 多 seed held-out retrieval。
**狀態**：loss plateau「已解釋、不需先修改 loss」；新資料協定下是否學得更好**未解決**。

---

## 八、AdamW 稽核（依 Codex 要求實測）

**VERIFIED（`output/look/stage2_optimizer_audit_flat.json` 與 `stage2_smoke_seven_checks.json`，每個張量列 requires_grad / grad is None / grad norm / 一步後的 delta）**：
| 張量 | 舊構造（flat AdamW，mask token 在 optimizer） | 新構造（decay groups，mask token 凍結） |
|---|---|---|
| `query.layout_weight`（λ，0 維） | grad 6.73e-2，**delta 5.50e-4 = lr + lr·wd·λ**（被 decay） | grad 6.76e-2，**delta 5.00e-4 = lr**（不 decay） |
| `query.fusion.mask_tokens` | grad **不是 None，是全零張量**（norm 0），delta 3.56e-6 = **只有 decay 在動** | 凍結：grad None、delta 0 |
| `query.layout_encoder.missing_edge_token` | grad 全零張量，delta 3.63e-6 = 只有 decay | grad 全零，delta 0（1 維 → 不 decay 組） |
| `query.fusion.modality_pos` | 正常：grad 1.42e-2，delta 5.03e-4 | 正常 |
| fusion 權重／bias | 正常 | 正常 |
| `gallery.*` | grad None、delta 0 | 同 |

Codex 說得對的部分：PyTorch AdamW 確實跳過 `grad is None` 的參數。實測的關鍵是**這兩個 token 的 grad 不是 None，是零張量**（`torch.where` 的未選分支仍產生一個全零的 grad），所以 AdamW 照樣對它們施加 weight decay：每步乘 (1 − lr·wd)，delta = 5e-4 × 0.1 × |p| ≈ 3.6e-6，與量到的一致。第一版「沒有梯度仍會 decay」的**結論成立，但理由要改成「梯度是零張量而非 None」**。
新構造下七項檢查全過（`output/look/stage2_smoke_seven_checks.json`），λ 一步移動量正好等於 lr。

**修正**：`stage2.py` 改用 Stage 1 相同的 `weight_decay_groups`（ULIP 規則：bias、norm、0/1 維張量不 decay；λ 與缺邊 token 都在此組）並讀取 artifact 的 betas/eps；`query_modality_masking` 記入 `stage2_protocol`（現值 `none`，是選擇不是 bug；`p_mask` 為未實作的消融）；在 `none` 下 fusion 的 mask token **凍結**（沒有梯度路徑的參數不進 optimizer）。

---

## 九、程式稽核結果（今日完成五份；修掉的與未修的）

已修（測試 581 通過）：
- Stage 2：checkpoint 紀錄**寫入磁碟**（`variant_ckpts.json`，含 stage1 sha、索引 sha、配方檔與 sha、有效值、arch protocol、code revision；同名需 `--overwrite`）；`load_variant` 核對 fusion / p_mask / missing-modality 三欄；`--epochs 0` 拒跑；只把 query 塔切 train 模式；協定欄位與程式硬編碼**互相核對**（不一致就拒跑）；語意邊快取的列標籤與節點向量 sha 都驗證。
- fusion：`masked_mlp` 尊重 `include_absent_slots`；模態 present 但沒給向量 → 拒絕。losses：`labels` 非恆等排列時 g2q 用反排列；固定 τ 不再被 clamp。
- dual tower：刪掉沒有呼叫者的模型端 scene-dropout 抽樣與 `drop_layout`（含它們只測自己的五個測試）。
- 評估端：`rank_of_target` 拒絕非有限相似度（NaN 曾算成 rank 0 = 命中）；reported 協定下 query 被 pack 丟掉就拒跑；provenance 加入 checkpoint phase、`sealed_read_on_nonfinal_checkpoint`、splits 的 sha256/seed/admitted_total、query 視角在 gallery 平均裡的註記；gallery encoder hash **納入 BatchNorm buffers**（新索引記 `gallery_encoder_hash_includes_buffers`，讀端依旗標計算）。
- Stage 1：`train_scope` 不再被 hash 兩次；缺席模態表示法只接受 learned_token/zero_pad；快取 embedding 的 sidecar 與編碼協定**核對**（模板、聚合、編碼器版本、視角數）；`--query-observation` 寫進 checkpoint；dev-val 的 worker 數跟訓練一致；checkpoint 還原時**buffers 也要覆蓋**；`full` 提前拒跑；QueryPack 身分改 hash 所用 shard 的陣列位元組而非會被改寫的 manifest；`splits.admitted_uids` 套用人工排除清單。
- 資料：query pack 三個 bug；點雲索引只收現行 sampler 版本且 npz 存在者；渲染 bbox 與抽點用同一種 mesh 過濾；`scene_graphs.humanise` 修駝峰拆字與冠詞（生效需重跑物件文字→語意邊→節點向量→Stage 2 索引，**與 ProcTHOR 渲染協定決定一起做**）；語意邊 prompt 文字釘到版本號（測試）。
- 骨幹：深度殼灰色 0.4。

未修、待決或待重跑：
- ProcTHOR 視角用退役常數（11 張、224px、正交、白底）vs Objaverse（12 張、512px、透視、黑底）：**研究決定**（重渲或登記為偏離），未動。
- 節點文字 5.9% 節點受影響（48,577/827,730）：程式已修，資料鏈未重跑（見上）。
- 官方 caption 是否與 gallery 渲染同源：**MISSING-EVIDENCE**。
- `gallery_index.promote` 的 gate 紀錄路徑會被下一次 G4 覆寫（提供 sha 但路徑會失效）：未動。
- n06 快取鍵未綁 OpenCLIP 權重 sha：未動（改了會迫使全量重編碼）。
- 評估端沒有 Stage 2 checkpoint 的評估路徑（Table 1 第二列）：範圍缺口。

---

## 凍結前清單（對應 Codex §六）
1. 五份稽核：**完成**（本節）。
2. `prompt_avg`：**完成**，是 prompt-template 平均。
3. 官方 `image_feat` 相容性：**完成**，同空間、不同相機、不可互換。
4. query pack 68 筆與 `--limit`：**已修**（需重建文字 shard）。
5. Unity 左右手系幾何測試：**完成**，不需鏡射。
6. 節點文字品質量化：**完成**（7/93 相異字串、5.9% 節點）。
7. Stage 2 checkpoint 寫入磁碟：**已修**。
8. Stage 2 modality masking：記為選擇 `none`，消融另建。
9. RGB 0.5 vs 0.4：**已量**（有影響）→ 採 0.4、索引重建。

## 修正清單（第一版 → 本版）
1. 「13.8 與字面構造不相容」→ empirical discrepancy。
2. 「唯一可 defend 的設計」→ 目前資料限制下最簡單可稽核的主要方案。
3. ULIP-2 隨機抽視角＋該視角 caption：LIKELY → VERIFIED（§3.3）；釋出檔無逐視角 caption 分開寫。
4. 文字分級改四級；external caption 只稱 generator-disjoint。
5. 官方 caption 15 分差距不等於品質；保留 gallery ablation。
6. `prompt_avg` 語意：撤回多句平均的支持。
7. 75-token／多句平均：不凍結，先做長度消融。
8. 官方 image_feat：條件式可用，已做測試。
9. L\* 公式撤回，改 margin 診斷。
10. AdamW：以逐張量實測取代論證。
11. 分歧點 1（重抽點 pack）：接受保留為 sampling-control、低優先；分歧點 2（11 視角重標註）：接受先做 pilot 階梯。
