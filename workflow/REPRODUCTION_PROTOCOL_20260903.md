# MetaFind 復現協定最終審查稿

> **來源**：Kyzen 2026-09-03 貼入，GPT × Codex 共識整理 + Claude 實作任務。
> **地位**：這份取代 `DATA_PLAN_PAPER_FIRST.md` 與 `AUTOPILOT.md` §四 的執行順序。
> 它不是 MetaFind 論文，也不是上游程式碼；它是**本專案目前的執行規格與證據標籤約定**。
> 論文原文、上游官方實作、磁碟上的產物仍然高於它（`CLAUDE.md` §3 的權威階層不變）。
> 逐字保存，未經 Kyzen 同意不得改寫內容；補充一律另開段落並標明日期與作者。

---

## 目的

目前不是直接宣布某一套 MetaFind reproduction protocol 已經唯一確定。

我們現在要做的是：

1. 把 **論文明確公開的事實**固定下來。
2. 把 **作者 rebuttal 提供的額外證據**與 paper 本文分開。
3. 把 **論文沒有公開的 protocol**明確標成 unresolved。
4. 先使用目前已有的 Objaverse / ProcTHOR 資料，驗證：
   * dataset pipeline
   * Stage 1 training
   * gradient flow
   * retrieval evaluator
   * protocol sensitivity
5. 資料處理必須保持可逆，避免現在猜錯 camera / observation / gallery protocol 後又重新跑數十小時。
6. Claude 負責實作與 audit；GPT/Codex 負責 protocol 判斷與審核。

---

# 一、統一使用的四種證據標籤

後面所有設定只能歸到以下四類。

```text
PAPER FACT
= MetaFind 論文正文直接支持。

AUTHOR EVIDENCE / MAINLINE
= paper 本身沒有完全釐清，但作者 rebuttal / figure 提供很強證據；
  因此目前復現主線採用。

UNRESOLVED
= 論文、Figure、作者回覆目前仍不足以唯一確定。

IMPLEMENTATION CHOICE / MAINLINE
= 為了能實作與重現，我們目前選擇的 protocol；
  不能寫成作者明確公布的設定。
```

---

# 二、目前真正要審的 10 個問題

## 問題 1：Query / Gallery 是否共享同一份 ULIP-2 backbone？

### Paper 能確認

MetaFind 使用 dual-tower architecture：`Query tower` / `Gallery tower`。
正文描述 `separate encoders for query and gallery`，並且兩邊都 `leverage ULIP-2`。

因此 paper 本身可以證明：

```text
Query path   = ULIP-2 based
Gallery path = ULIP-2 based
```

但單靠正文不能證明 `query.point_encoder is gallery.point_encoder`。
也就是正文沒有精確到 Python parameter object / weight-sharing level。

### 作者 rebuttal 額外證據

作者 rebuttal 明確描述 `two encoders share same backbone`，以及
`Both query and gallery encoders share the ULIP-2 backbone`。
Figure 又標示 `ULIP-2 (Shared)`。

### 最終判定

```text
AUTHOR EVIDENCE / MAINLINE
```

目前主線採：

```text
                    Shared ULIP-2 backbone
                  Text / Image / PointBERT
                            │
                ┌───────────┴───────────┐
                │                       │
            Query path              Gallery path
                │                       │
            Fusion_Q                Fusion_G
                │                       │
                q         ↔             g
```

即：ULIP-2 backbone weights **shared**；Query Fusion **separate**；Gallery Fusion **separate**。

### Sensitivity arm

另外允許做 `PointBERT_Q != PointBERT_G`，也就是 untied backbone experiment。
但 untied Q/G PointBERT **不是**目前 paper-faithful mainline。

---

## 問題 2：Objaverse 的 11 viewpoints 到底怎麼拍？

### PAPER FACT

MetaFind 只明確說 `11 orthogonal viewpoints`。

### UNRESOLVED

目前沒有公開：azimuth、elevation、camera distance、FOV、projection type、resolution、
background、lighting、camera target、object scale convention。
也沒有公開：我們現有 12 views 中作者到底會使用哪 11 張。

### 特別注意

`orthogonal viewpoints` 不能直接推出 `orthographic camera projection`。這是兩個完全不同概念。

### 最終判定

```text
11 viewpoints = PAPER FACT
exact camera protocol = UNRESOLVED
```

Claude 不可自行把某組（11 equally spaced azimuth / 20° elevation / 512×512 /
black background / perspective）寫成 paper protocol。

---

## 問題 3：現在是否應該重新渲染全部 Objaverse？

### 最終結論

```text
現在不要做 46K 全量 rerender。
```

這不是 paper fact，而是目前的工程決策。原因：

```text
exact camera protocol unresolved
        ↓
現在重新 render 46K
        ↓
等於把我們猜的 camera protocol 固化
        ↓
若之後猜錯，需要全部重跑
```

### 但現在資料照樣要處理

目前已有 12 views，所以現在應：

```text
12 raw views → 全部保存 → 逐 view preprocessing
→ 逐 view frozen image feature → runtime 再決定使用哪些 view
```

所以「暫時不 rerender」**不等於**「現在不處理 image data」。恰恰相反：

> 現在要先把已有資料整理到未來切任何 observation/view protocol 都不用重新 encode。

---

## 問題 4：Stage 1 Query / Gallery positive 到底是不是 same record？

### 第一層：asset identity

**PAPER FACT**：Stage 1 正樣本 `Query A ↔ Gallery A`，也就是 `positive_policy = same_uid`；
負樣本則是其他 asset `A'`。所以 `same UID` 是 paper fact。

### 第二層：observation

Paper 沒有告訴我們 Query Text_A 是否等於 Gallery Text_A、Image、PC 亦然。
例如完全可能：`UID = chair_001`，Query image `view_03`，Gallery image `views_01~11 aggregate`，
仍然是 same UID positive。

### 最終判定

```text
same UID         = PAPER FACT
same observation = UNRESOLVED
```

所以不能再用一個 `same_record` 同時代表兩件事。應拆成：

```yaml
positive_policy: same_uid

query_observation:
  text: ...
  image: ...
  pc: ...

gallery_observation:
  text: ...
  image: ...
  pc: ...
```

---

## 問題 5：11 張 image 到底如何變成一個 Image modality？

### UNRESOLVED

MetaFind 沒有公開：single view / mean pooling / max pooling / attention pooling /
learned view fusion / 其他 view aggregation。

MetaFind 所說的 `Text, Image, Point Cloud → Fusion` 是 **modality fusion**。
它不等於解釋 `view_01 … view_11 → Image modality feature` 這個 **view aggregation**。
兩者不能混在一起。

### 目前可以測的 implementation candidate

例如 mean pooling，但只能標 `IMPLEMENTATION CHOICE`。

### aggregation config 必須完整記錄

不能只記 `method: mean`。必須至少：

```yaml
view_aggregation:
  selected_view_ids: [...]
  pre_normalize_each_view: true_or_false
  method: mean
  post_normalize: true
```

因為 `Normalize(mean(z_i))` 與 `Normalize(mean(Normalize(z_i)))` 不是同一個 protocol。
還要記：number_of_views、selected_view_ids、view_selection_policy、aggregation_version。

---

## 問題 6：Table 1 的 Image-only Query 到底使用什麼 observation？

### UNRESOLVED

論文只報 `Image Only`，但沒有定義 single image / random image / same multi-view mean /
held-out image / disjoint view subset。

### Dataset / evaluator 至少要支援

```text
A. same multi-view aggregate
B. single view
C. held-out single view
D. disjoint-view subset
```

### 重要：evaluation sensitivity 與 training protocol 必須分開

假設某一顆 checkpoint 是用 multi-view training，evaluation 時換成 held-out view，
這只能叫 `inference sensitivity` 或 `distribution-shift sensitivity`。
不能說「我已經測完 held-out-view training protocol」。

如果要比較 `Training A: multi-view` vs `Training B: single-view`，必須**重新 train**。
同理：same observation vs second observation、different caption policy、
different PC observation、different positive construction，
如果發生在 training pipeline，都必須重新 train 才是公平比較。

---

## 問題 7：Table 1 的 Gallery candidate pool 到底是多少？

**PAPER FACT**：`80% train / 20% test`。

**但 paper 沒公開** Table 1 evaluation 是 `test query → test gallery` 還是
`test query → full Objaverse repository`。

因此 `query split = test` 與 `gallery scope` 必須視為兩個獨立維度。

### 最終判定

```text
80/20 ratio = PAPER FACT
Table 1 gallery scope = UNRESOLVED
```

Claude 必須同時支援 `gallery_test` 以及 `gallery_full`。Evaluator：

```yaml
query_split: test
gallery_scope: test
```

與

```yaml
query_split: test
gallery_scope: full
```

都能直接跑。每次 evaluation 必須打印 `number_of_queries`、`number_of_gallery_candidates`。

### 禁止反推作者 hidden protocol

即使 `test→full` 剛好最接近 paper 七格數字，也只能說
`numerically/behaviorally consistent with paper`，不能說「作者一定就是 test→full」。

---

## 問題 8：ESSGNN pooling 與 λ 到底怎麼設定？

MetaFind 定義：

```text
e_query = Fusion(e_T, e_I, e_P) + λ · e_layout
```

**PAPER FACT**：`λ = learnable scalar`。

**UNRESOLVED**：沒有公開 λ initialization（1、0.1、1/30、0 都不是 paper fact）。
同時，論文描述 graph node features 經 ESSGNN 後 aggregate 成 global layout embedding，
但沒有明確公開 sum / mean / max / attention pooling。

### 最終判定

```text
λ learnable      = PAPER FACT
λ initialization = UNRESOLVED
global pooling   = UNRESOLVED
```

例如 `mean pooling + λ=0.1` 可以測，但只能叫 `IMPLEMENTATION CHOICE`。

### 目前動作

這題暫時不要在 dataset phase 鎖死。等真正進 Stage 2 前再做專門審查。

---

## 問題 9：Stage 2 是否繼續做 Stage 1 的 30% modality masking？

### Stage 1

**PAPER FACT**：Query 的 Text / Image / PC 每個 modality `independently masked with p = 0.3`；
Gallery `modality-complete`。

### Stage 2

Paper 明確公布的是 `30% scene dropout`，也就是某些 batch `omit e_layout`。
Paper 沒有說 Stage 2 繼續對 Text/Image/PC 做 Stage-1 modality masking。

### 最終主線

```text
Stage 1 = modality masking
Stage 2 = scene dropout
```

不要自行變成 `Stage 2 = scene dropout + modality masking`。
如果未來做這個，只能標成 `ablation / extension`。

---

## 問題 10：ProcTHOR node 的 text-derived feature 到底怎麼產生？

MetaFind node：`v_i = (x_i, t_i)`，其中 `x_i` = 3D coordinate、`t_i` = text-derived feature。
Paper 同時說 ProcTHOR 有 `semantic metadata`。

**PAPER FACT**：node 包含 coordinates；node 包含 text-derived semantic feature。

**UNRESOLVED**：沒有說 `t_i` 到底由 category name、full metadata template、description，
還是 LLM-generated object description 產生。

因此 `ProcTHOR node text serialization = UNRESOLVED`。

### 但 Semantic Edge 是另一件事

這部分 paper 明確很多。流程是：

```text
object descriptions → LLM → natural-language relation sentence
→ frozen text encoder → semantic edge feature e_ij
```

所以必須把 `node text feature` 與 `semantic edge text` 完全分開。

---

# 三、除了 10 題之外，以下訓練規則已經固定

這些不要 Claude 再當 unresolved 猜。

## A. Stage 1 Encoder Fine-tuning

Paper 明確說 Stage 1 `both query and gallery encoders are trained`。
Table 3 又有 `Train fuser only = 8.7`，而正文解釋 `full encoder fine-tuning` 效果較好，
因為 earlier layers 能夠適應 modality-aware supervision。

因此 `Stage 1 encoder fine-tuning = PAPER FACT`。

## B. PointBERT trainable

PointBERT 是目前 ULIP-2 implementation 中的 3D encoder。
所以若我們要實現 paper 所描述的 encoder fine-tuning，`PointBERT trainable` 是
`STRONG IMPLEMENTATION MAPPING`。不能再把 PointBERT 固定 embedding 當 Stage-1 train input。

主線：

```text
Point Cloud → PointBERT online forward → PC embedding → Fusion
→ Stage 1 loss → backprop → PointBERT
```

## C. 但目前 Stage 1 optimizer scope 仍不是 paper 完整規格

Paper 沒有公開逐 module 的 optimizer group、per-module learning rate、exact freeze boundary。

因此目前 MetaFindV1 的 `train_scope = point_encoder_and_fuser`，即
PointBERT trainable、Fusion trainable、Text encoder frozen、Image encoder frozen，
必須標 `IMPLEMENTATION MAINLINE` 而不是 `PAPER FACT`。

## D. Text/Image frozen cache 的適用範圍

因為目前主線 Text/Image frozen，所以可以 cache text feature 與 image feature。
但這些 cache **只適用於目前 `point_encoder_and_fuser` train scope**。

如果未來測真正的 full encoder fine-tuning arm，而 Text/Image encoder 也需要 trainable，
就不能再用固定 feature 取代 encoder forward，必須 raw text → online text encoder、
raw image → online image encoder 重新前向。

## E. Point Cloud exact representation

目前我們使用 ULIP-2 10K XYZRGB、ViT-bigG variant，以及 canonical 10K XYZRGB point cloud。
這是 `IMPLEMENTATION MAINLINE`，不是 MetaFind paper fact。

MetaFind 沒有完整公開 exact ULIP-2 checkpoint、8K XYZ or 10K XYZRGB、
point sampling protocol、normalization protocol、RGB usage。

因此資料層必須同時保留 `source/raw point cloud` 與 `derived canonical point cloud`，
不能只剩 10K XYZRGB。

## F. Stage 1 Masking

**PAPER FACT**：每個 Query modality independently masked，`p = 0.3`。
所以字面上 `P(all masked) = 0.3^3 = 0.027`，也就是 2.7%。

**主線**：保留 all-masked，不 conditional redraw。

### Mask representation

Paper 明確表示 masked modality 使用 `masked embedding`，不是 `zero padding`。
但 paper 沒有證明 one learnable mask token per modality。
因此目前 `learnable mask embedding` 是 `IMPLEMENTATION CHOICE`。

### all-masked 如何評估

這類 sample 沒有 asset-specific observation information，
所以不要拿 all-masked R@1 當模型能力指標。
主要監控：number of samples、loss、embedding norm、similarity distribution、
gradient behavior、training stability。

## G. Stage 1 Loss

**PAPER FACT**：Stage 1 Eq.5 `L_Stage1 = L_{q→g}`，即單向 contrastive loss。
不要自行改成 `0.5(q→g + g→q)`。

## H. Stage 2 模組 freeze/train scope

### FROZEN

```text
Query-side ULIP-2 modality encoders
Gallery-side ULIP-2 modality encoders
Gallery Fusion
Gallery tower
```

### TRAINABLE

```text
Query-side Fusion
ESSGNN
λ（如果作為獨立 learnable parameter）
```

也就是：

```text
Query modalities
       │
       ▼
Frozen ULIP-2 encoders
       │
       ▼
Trainable Query Fusion ───────────────┐
                                      │
Scene graph                           │
    │                                 │
    ▼                                 │
Trainable ESSGNN ─────────────────────┤
                                      ▼
                     Fusion + λ·layout representation
                                      │
                                      ▼
                           Frozen Gallery tower
```

這才對應 `only query-side fusion layer and ESSGNN are updated`。

## I. Stage 2 Loss

**PAPER FACT**：`L_layout = ½ (L_{q2g} + L_{g2q})`，也就是 bidirectional contrastive loss。

因此 Stage 1 是 `q → g`；Stage 2 是 `0.5 × (q → g + g → q)`。不能混。

## J. Stage 2 scene dropout

**PAPER FACT**：`30% scene dropout`。
主線：30% batches omit layout component，不自行加入 Stage-1 modality masking。

---

# 四、Dataset Manifest 必須怎麼設計

這是 Claude 現在真正要實作的核心。

## 1. 一個 asset UID 永遠只是一個 asset

主 manifest：`1 UID = 1 row`。

禁止 `1 UID × 12 views → 12 gallery assets`，
也禁止 `1 UID × 10 captions → 10 gallery assets`。
views / captions / observations 應使用 child records：

```text
assets
├── uid
├── split
├── modality availability
├── metadata
└── provenance

images
├── uid
├── view_id
├── image_path
└── camera metadata

captions
├── uid
├── caption_id
├── text
└── provenance
```

## 2. Manifest 必須記錄 provenance

每個 asset / derived artifact 儘可能保存：source dataset、source dataset version、
source path、source file sha256、UID、modality availability、view_id、camera metadata、
dtype、shape、preprocessing name、preprocessing version、encoder name、
encoder checkpoint SHA、feature dimension、feature dtype、split、
filter/quarantine status、filter reason。

## 3. 原始資料不能覆寫

必須 `raw/`、`derived_v1/`、`derived_v2/` … 或等價 versioned layout。
不能 process 完直接覆蓋 raw image / raw PC / raw metadata。

---

# 五、Train/Test Split 完整要求

**PAPER FACT**：只有 `80% train / 20% test`。

Paper 沒公開 official UID lists、seed、randomization algorithm、split implementation、
filter before/after split。

## Claude 實作規則

如果已有可信官方 split，優先完整保存，並記錄來源、dataset version、manifest hash。
如果沒有，才建立 deterministic UID-level 80/20 split，且必須標 `IMPLEMENTATION CHOICE`。

## Split 必須保存

seed、split algorithm、algorithm version、complete input UID universe、
train UID manifest、test UID manifest、manifest SHA256。

必須記錄 split 發生在 modality filtering 前或後。不能不知道。

## 必須 assert

```text
train_uid ∩ test_uid = ∅
```

---

# 六、Filtering / Quarantine 規則

缺模態 asset 不可 silent drop，必須進 quarantine 或有 filter_reason，例如：
missing_image、missing_pc、missing_text、corrupted_image、invalid_pc、annotation_parse_failed。

每一道 filter 都要輸出 before count / removed count / after count。

最後完整報：raw assets、usable assets、train assets、test assets、quarantined assets。

---

# 七、Image 現在怎麼處理

目前已有 12 views。

**現在做**：12 raw images 全保存。每張記 uid、view_id、path、camera metadata if available、sha256。

## Frozen image feature

因目前 Stage-1 implementation mainline 是 Image encoder frozen，所以
12 views 每一張獨立 encode，12 個 per-view features，可以 cache：

```text
UID/
├── view_00.feature
├── view_01.feature
...
└── view_11.feature
```

## 禁止只保存 mean

可以另外 cache mean feature，但不能取代 per-view features。
因為之後還要測 single view、held-out view、disjoint views、different selected 11、
different normalization。

---

# 八、Text 現在怎麼處理

全部保存：raw annotation、structured JSON、individual fields、canonical description、
alternative captions、caption provenance。

禁止現在把所有東西 collapse 成唯一字串後丟掉原資料。

## Frozen text features

目前 Stage-1 mainline Text encoder frozen，因此 text features 可以 cache。
但 metadata 必須記 encoder/checkpoint SHA、tokenizer version、serialization version、
feature dimension、dtype。

如果未來改成 Text encoder trainable，固定 cache 就不能拿來取代 online encoder。

---

# 九、Point Cloud 現在怎麼處理

至少保存兩層：`source/raw PC` 與 `derived canonical PC`。
目前主線可以產生 10K XYZRGB，但一定標 `IMPLEMENTATION MAINLINE`，不是 MetaFind FACT。

如果成本合理，可以另外保存 resampled PC observation，方便之後測
`Query PC = sample A / Gallery PC = sample A` vs `Query PC = sample B / Gallery PC = sample A`。

## Stage 1 PC embedding 禁止固定化

因為目前 PointBERT trainable，所以正式 Stage-1 train path 必須
`PC → PointBERT online → embedding → Fusion → loss`，
不能 `PC → 提前 PointBERT cache → fixed embedding → Fusion`，
否則就會偷偷變成 PointBERT frozen。

固定 PC embedding 只能用於：ULIP baseline、diagnostic、retrieval sanity test、
frozen-backbone ablation。

---

# 十、Feature Cache 必須與 train scope 綁定

每個 cache 要至少有 encoder_name、checkpoint_sha、preprocessing_version、feature_dim、dtype。
最好再有：

```yaml
valid_for_train_scope:
  - point_encoder_and_fuser
  - fuser_only
```

例如 frozen image cache 可以用在 `point_encoder_and_fuser`，
不能自動用在 `full_encoder_finetune`。
防止之後改 train scope，Claude 還繼續偷讀固定 cache。

---

# 十一、Query / Gallery Dataset API

不要再有含糊的 `same_record`。應該至少：

```yaml
positive_policy:
  type: same_uid

query_observation:
  text:
    policy: canonical
  image:
    policy: single_view
  pc:
    policy: canonical

gallery_observation:
  text:
    policy: canonical
  image:
    policy: multi_view
  pc:
    policy: canonical
```

以後可以切：canonical、alternate_caption、single_view、same_mean、held_out_view、
disjoint_views、canonical_pc、resampled_pc，而不用改 raw dataset。

---

# 十二、Gallery Manifest

建立至少 `gallery_test` 與 `gallery_full`。
每個 gallery item 對應 exactly one asset UID，不能因為 views/captions 被重複展開。

Evaluator 每次必須記：query split、query count、gallery scope、gallery candidate count、
checkpoint、observation policy、aggregation policy。

---

# 十三、ProcTHOR 現在怎麼處理

目前先做 metadata / graph-ready preprocessing。每個 object 至少保存：
scene_id、room_id、object_id、asset_id、category、raw semantic metadata、
position xyz、rotation、scale、bbox / size、geometry reference。

## Relations provenance 分開

**Source/physical/layout relations**（support、containment、adjacency、spatial layout relation）
要記 `source-derived / ProcTHOR-derived`。

**Semantic relations**（LLM 後續產生的 semantic edge）必須另外記
`LLM-derived`、model、prompt version、generation version。

兩種 edge 不准混。

## 現在不要自行生成 semantic edge

目前先把 source metadata、physical relations、object descriptions / candidate inputs 整理好。
等 Stage 2 protocol 準備完整後，再執行 LLM semantic relation generation。

---

# 十四、Table 1 `MetaFind w/ ESSGNN` 的正確理解

完成 Stage 2 後：

```text
Stage 2 checkpoint → 回 Objaverse-LVIS → 沒有 scene layout
→ object retrieval evaluation
```

不是「在 Objaverse-LVIS 人工塞 ProcTHOR graph」。

因此 Table 1 `w/ ESSGNN` 主要是在看：

> 經 Stage-2 contextual training 後的模型，重新回到 object-level retrieval 時的表現。

---

# 十五、現在舊資料到底拿來做什麼

## 第一用途：Correctness validation

先驗證：UID pairing 是否正確、train/test 是否 leakage、
Query masking 是否每 modality independent p=.3、Gallery 是否 modality-complete、
all-masked 是否真的約 2.7%、mask representation 是否非 zero padding、
Stage1 q→g loss 是否正確、PointBERT 是否真的收到 gradient、
Text/Image 是否真的 frozen、Fusion 是否收到 gradient、
R@1/R@5 evaluator 是否正確、positive UID indexing 是否正確、
gallery candidate count 是否正確。

目標：**先證明 implementation 沒寫錯**。

---

# 十六、舊資料第二用途：Protocol sensitivity

同一 checkpoint 可以先改 evaluation：
`test gallery` vs `full gallery`；
`same aggregate` vs `single view` vs `held-out view`；
different candidate pool size。

這些用來看 paper Table 1 對 protocol 有多敏感。

## 但是注意

同一 checkpoint 只能做 `inference sensitivity`，不能替代重新 training。

以下如果改動 training protocol，都必須重新 train：
same observation vs second observation during training、
single-view vs multi-view during training、alternate caption during training、
alternate PC observation during training、positive pair construction、
fusion input design、shared vs untied backbone、train scope。

---

# 十七、如何利用 Table 1 判斷 protocol

Paper `MetaFind w/o ESSGNN` 大致提供七種 modality condition：
Text、Image、PC、Text+Image、Text+PC、Image+PC、Full。

我們可以比較不同 protocol 下的整體 7-dimensional performance fingerprint，
而不是只盯 Image R@1。

例如某設定七個數字整體 shape 比較像 paper，只能說
`this protocol is behaviorally/numerically more consistent`，
不能說「作者一定就是用這個 protocol」。

---

# 十八、現在真正的工作流程

## PHASE 1 — Current-state audit

Claude 先**只讀檢查**目前：Objaverse、ProcTHOR、Stage1 code、Dataset code、
Feature cache、Evaluator、Configs。不先大規模跑。

## PHASE 2 — Dataset manifest / provenance

建立：UID-level asset manifest、image child records、caption child records、
PC records、filter/quarantine records、split manifests。

## PHASE 3 — Reversible preprocessing

現在可以做：12-view image indexing、12-view frozen image features、
text structured records、frozen text feature caches、raw/source PC preservation、
canonical PC generation、ProcTHOR metadata normalization、graph-ready source relations。

## PHASE 4 — Dataset API

實作：same_uid positive policy、query_observation、gallery_observation、
view selector、view aggregation config、gallery_test、gallery_full。

## PHASE 5 — Stage 1 correctness test

先小規模測：loss、masking、all-masked、gradient、optimizer scope、retrieval indexing。
尤其要確認 **PointBERT receives non-zero gradients**，而 Text/Image currently do not，
符合現在 implementation mainline。

## PHASE 6 — Evaluation sensitivity

用現有 checkpoint / 小規模 checkpoint 測：test vs full gallery、single vs aggregate、
same vs held-out observation at inference、normalization variants、candidate pool。

## PHASE 7 — 決定哪些 protocol 值得重新 train

只有 training-time protocol 改動才正式建立新的 experiment arm，例如：
same observation training、second observation training、single-view training、
multi-view training、shared backbone、untied backbone。

## PHASE 8 — 再決定昂貴資料重製

等前面都確認後才決定：46K Objaverse rerender 是否有必要、
GPT-4o 全量 reannotation 是否有必要、ProcTHOR render 是否有必要、
semantic-edge LLM generation 何時開始。

---

# 十九、Claude 現在禁止先做的工作

在 audit 回報以前，不要：

```text
46K full rerender
80h full annotation
ProcTHOR full rendering
刪除第 12 view
只留下 multi-view mean
覆寫 raw data
刪除舊 cache
把 observation policy 硬編碼
把 gallery scope 固定成 test
把 gallery scope 固定成 full
用 fixed PC embedding 取代 trainable PointBERT
自行生成 ProcTHOR semantic edges
自行決定 ESSGNN pooling / λ init
```

---

# 二十、Claude 第一個回覆必須完整回答

不要先大量改 code。先回報：

```text
1. 目前 Objaverse directory tree
2. 原始 UID 數
3. usable UID 數
4. 每道 filtering：before / removed / after
5. quarantine / filter reasons
6. 每 UID：text observation count / image view count / PC availability
7. 現在到底是不是 12 views；view naming；有沒有 camera metadata
8. 現有 text schema
9. 現有 image schema
10. 現有 raw/source PC schema
11. 現有 derived PC schema
12. 現有 feature caches
13. 每種 cache：encoder / checkpoint / preprocessing / dtype / dimension
14. 現有 train/test split：UID lists / algorithm / seed / 是否 deterministic /
    filtering 前或後切
15. 是否存在 train/test UID overlap
16. 現有 same_record 到底代表：same UID？same observation？兩者？
17. 現有 Query/Gallery observation construction
18. 現有 image aggregation：哪些 views / pre-normalization / aggregation method /
    post-normalization
19. 現有 gallery candidate construction：test-only / full / other
20. 現有 Stage1 optimizer parameter groups
21. PointBERT requires_grad 狀態
22. 實際證明 PointBERT 是否收到 gradient
23. Text encoder requires_grad 狀態
24. Image encoder requires_grad 狀態
25. Fusion_Q / Fusion_G 是否 separate parameters
26. Shared ULIP backbone 現行程式實際怎麼實作
27. Stage1 loss 現行程式是不是 q→g only
28. masking implementation：是否 independent 30% / masked embedding 怎麼做 /
    是否允許 all-masked
29. ProcTHOR 現有資料 schema
30. ProcTHOR physical/layout relation 來源
31. 是否已有 LLM semantic relation
32. 如果有，provenance 是否分開
33. 建議要修改哪些檔案
34. 每個檔案預計修改內容
35. migration strategy
36. backward compatibility
37. 是否影響已有 checkpoint
38. 是否影響已有 feature cache
39. 需要重新 compute 的項目
40. 每項預估：CPU / GPU / disk / runtime
```

---

# 二十一、Claude audit 完成後才執行的工作

我們看完 audit 後，再批准：哪些 manifest 要建立、哪些 cache 要重建、
哪些資料可直接沿用、哪些 protocol 可以只做 evaluator change、
哪些 protocol 必須重新 train、哪些昂貴 preprocessing 真正必要。

---

# 二十二、給 Codex 的最終審查要求

Codex 不需要再從零重新寫一份。請它針對整份完整規格檢查：

```text
A. 是否還有 paper fact 被錯標成 implementation choice
B. 是否還有 implementation choice 被錯標成 paper fact
C. 是否有論文已明確寫、但我們仍錯列 unresolved
D. 是否有 evidence source 混淆：paper / figure / rebuttal
E. 是否有 Stage1 / Stage2 train scope 矛盾
F. 是否有 cache 會阻斷 trainable encoder gradient
G. 是否有 data leakage 風險
H. 是否有 UID/gallery duplication 風險
I. 是否有不可逆 preprocessing
J. 是否有 evaluator protocol 無法版本化
```

如果沒有 substantive disagreement，直接 `CONFIRMED`。
如果有，只列：項目 / 目前文件說法 / 你不同意的原因 / 直接 evidence / 建議修正。

---

# 二十三、目前最終狀態摘要

```text
Shared ULIP-2 weights          → AUTHOR EVIDENCE / MAINLINE
Untied PointBERT               → sensitivity

11 viewpoints                  → PAPER FACT
exact camera                   → UNRESOLVED

same UID                       → PAPER FACT
same observation               → UNRESOLVED

view aggregation               → UNRESOLVED
image-only observation         → UNRESOLVED

80/20                          → PAPER FACT
exact split                    → IMPLEMENTATION CHOICE
gallery test/full              → UNRESOLVED

Stage1 encoder fine-tuning     → PAPER FACT
PointBERT trainable            → STRONG MAPPING
point_encoder_and_fuser        → IMPLEMENTATION MAINLINE
Text/Image frozen              → IMPLEMENTATION MAINLINE

Stage1 modality masking 30%    → PAPER FACT
masked embedding != zero pad   → PAPER FACT
learnable mask token           → IMPLEMENTATION CHOICE
all-masked 2.7%                → consequence of independent masking

Stage1 q→g loss                → PAPER FACT

Stage2 modality encoders frozen→ PAPER FACT
Gallery tower frozen           → PAPER FACT
Query Fusion trainable         → PAPER FACT
ESSGNN trainable               → PAPER FACT
scene dropout 30%              → PAPER FACT
Stage2 bidirectional loss      → PAPER FACT

λ learnable                    → PAPER FACT
λ init                         → UNRESOLVED
ESSGNN global pooling          → UNRESOLVED

ProcTHOR node coordinates      → PAPER FACT
ProcTHOR node text-derived feat→ PAPER FACT
node text construction         → UNRESOLVED
semantic-edge LLM relation gen → PAPER FACT

10K XYZRGB                     → IMPLEMENTATION MAINLINE
```

---

## 最後一句給 Claude

> **你的任務不是替作者補完沒公開的 protocol，而是先把 MetaFindV1 的資料、training、
> cache、evaluator 做成可驗證、可版本化、可切換 protocol 的系統。現在先 audit 現有實況，
> 再提出 migration plan；不要在 audit 前進行 46K rerender、全量 annotation
> 或其他昂貴且建立在 unresolved assumption 上的工作。**

---

# 附錄 A：MASTER 收到這份文件時，repo 與它相牴觸的三點

由 MASTER 於 2026-09-03 加註，供 audit 對照。這一節是 MASTER 的觀察，不是原文件的一部分。

**A-1　程式碼已經把相機協定寫死成 11 視角，而磁碟上的語料是 12 視角。**
`renders.RENDERER_VERSION = 7`、`render_blender.N_VIEWS = 11`、
`stage1.N_VIEWS_PER_ASSET = 11`（`metafind/train/stage1.py:395`，於 `:463` 對載入的
`views` 矩陣做斷言）。磁碟上實測：render sidecar `renderer_version = 6`、12 views；
embedding npz `views` shape `(12, 1280)`、sidecar `n_views = 12`。
後果：**Stage 1 現在無法讀取現有語料**，`renders.is_complete` 也會判定全部 46,024 筆過期。
本文件問題 2 判定 exact camera protocol 為 UNRESOLVED、問題 3 禁止現在重渲，
因此這組寫死的常數與本文件直接牴觸。屬於問題 40 的 migration 項目。

**A-2　工作區有一段未 commit 的修改會刪掉第 12 張圖。**
`metafind/data/render_blender.py`，`render_asset` 在搬入新圖前 `unlink` 掉
`asset_dir` 內所有既有 `view_*.png`。本文件十九節明文禁止「刪除第 12 view」。
該修改**尚未 commit、尚未測試、尚未審查**，應在 audit 期間還原或轉為只在重渲時生效。

**A-3　ESSGNN pooling 與 λ 初值已經寫進 protocol，而本文件判定兩者 UNRESOLVED。**
`metafind/models/resolve_stage2.py` 目前為 `pooling: "normalised_sum"`、`init_lambda: 9.0`；
`stage2_protocol.json` 的 `decided_by` 欄位寫「Kyzen 決定 init_lambda 9.0」。
帳本 DL-077 記錄 Kyzen 裁的是 0.1；DL-078 記錄 9.0 是由一次量測推導、且註明「Kyzen 可否決」。
本文件問題 8 判定 λ init 與 global pooling 皆為 UNRESOLVED，並在十九節禁止自行決定。
**兩個權威來源相衝突**（Kyzen 先前的裁決 vs 本文件），MASTER 不自行選擇，
按 `CLAUDE.md` §3 留為顯性衝突，等 Kyzen 裁示。
無論如何，`decided_by` 把未發生的裁決寫成 Kyzen 的決定，這一點必須更正。
