# ULIP-2 逐字讀完：論文（arXiv 2305.08275 v4）＋ 官方程式碼 —— 筆記（2026-09-04）

Kyzen 指示：「`docs/paper/ulip2_source/ulip2_arxiv_v4.html` 跟官方程式碼一樣逐字看完整理成 md 筆記」。

讀的東西：

| 來源 | 位置 | 版本 |
|---|---|---|
| 論文 HTML | `docs/paper/ulip2_source/ulip2_arxiv_v4.html` | arXiv 2305.08275 **v4**，2024-04-26，CC BY-SA 4.0（CVPR 2024 版） |
| 官方程式碼 | `/home/kyzen/upstream/ULIP` | commit `95d480f`（2026-06-02，只是補 SECURITY.md） |
| 我們的 vendor 複本 | `metafind/vendor/ulip/` | 17 個檔與上游逐 byte 相同（`ulipdiff.py`，見 `NOTE_20260904_ULIP2_CHECK_AND_STAGE_SUMMARY.md` §一.1） |

標籤：**PAPER** = ULIP-2 論文原文；**CODE** = 官方 repo；**DATA** = 釋出的 checkpoint／我們量到的；**INFER** = 我的推論；**UNKNOWN** = 兩邊都沒有。
之前那份 `NOTE_20260904_ULIP2_TRAINING_READ.md` 是「訓練配方」的摘要；這份是完整讀本，涵蓋它，之後以這份為準。

---

## 第一部分：論文逐節

### 0. 標題頁與摘要（PAPER）

- 作者：Le Xue 等 12 人，Salesforce AI Research 為主，另有 UPenn、UT Austin、Stanford。
- 摘要主張：既有框架的語言描述「不可擴展、不多樣」；ULIP-2 用大型多模態模型**自動**幫 3D 形狀生成整體（holistic）描述，**只需要 3D 資料本身**，不需人工標註；並且把 backbone 放大。
- 摘要數字：zero-shot 分類 Objaverse-LVIS **50.6** top-1、ModelNet40 **84.7** top-1；ScanObjectNN 微調 **91.5** 整體準確率，模型只有 **1.4M** 參數（那是 PointNeXt）。
- 程式碼與資料集：`https://github.com/salesforce/ULIP`。

### 1. Introduction（PAPER）

- Figure 1 說明：ULIP-2 用大型多模態模型為**每一張**從整體視角渲染出的 2D 圖生成描述；利用「已預先對齊且凍結」的視覺語言特徵空間，把三元組（整體文字、影像、3D 點雲）對齊；預訓練後只用 3D encoder 做下游任務；**整個過程只需要 3D 資料**。
- 論點：語言模態是 3D 多模態資料的瓶頸。既有方法（[34] ReCon、[52] ULIP）用人工類別名與 metadata 短描述，不可擴展、細節不足、有雜訊。
- Figure 2：同一個物件（Alfonso X 雕像）不同視角看不到的部分不同（椅子／權杖），**所有視角的描述合起來**才是完整資訊；metadata 的人工標題「Estatua de Alfonso X - José Alcoverro (1892)」沒什麼語意，可能反而傷害預訓練。
- 核心推論（§1 第 4 段）：若能從「任意視角」渲染，所有渲染圖近似包含此形狀的全部資訊；同理所有視角的語言描述近似包含全部可用語言表達的資訊。實務上取**有限固定的一組整體視角**，並且**語言用跟渲染同一組視角**，於是問題化為「描述給定視角的 2D 圖」。
- 三項貢獻：(1) 免人工的可擴展預訓練，任何 3D 資料集都能用，含未標註；(2) Objaverse-LVIS 50.6 top-1，超過 OpenShape 3.8 個百分點、框架更簡單；ModelNet40 84.7 超過部分全監督方法；ScanObjectNN 91.5 只用 1.4M 參數；還展示 3D→語言生成；(3) 釋出 ULIP-Objaverse 與 ULIP-ShapeNet 三元組。

### 2. Related Work（PAPER）

- 多模態表示學習：CLIP／SLIP 這類「各模態獨立編碼再對齊」的架構簡單、能大規模訓練。
- ULIP [52] 是三元組（點雲—影像—語言）對齊的先驅；I2P-MAE [59] 沒有語言對齊；**OpenShape [22] 是同期工作**，仍靠人工標註與複雜資料工程；ULIP-2 更簡單卻在 LVIS top-1 高 3.8。
- 生成式大型多模態模型：用 **BLIP-2 [17/18]** 生成描述；§5.2 有 BLIP vs BLIP-2 消融。
- 3D 點雲理解：PointNet、PointNeXt、Point-BERT；ULIP-2 用 **Point-BERT 與 PointNeXt** 當 3D encoder。

### Table 1：Zero-shot 3D 分類（PAPER，完整抄錄）

| 模型 | 預訓練資料 | 預訓練方法 | 人工描述？ | LVIS top-1 | LVIS top-5 | MN40 top-1 | MN40 top-5 |
|---|---|---|---|---|---|---|---|
| PointCLIP | – | – | – | 1.9 | 5.8 | 19.3 | 34.8 |
| PointCLIPv2 | – | – | – | 4.7 | 12.9 | 63.6 | 85.0 |
| ReCon | ShapeNet | ReCon | ✓ | 1.1 | 3.7 | 61.2 | 78.1 |
| CLIP2Point | ShapeNet | CLIP2Point | ✗ | 2.7 | 7.9 | 49.5 | 81.2 |
| Point-BERT | ShapeNet | OpenShape | ✓ | 10.8 | 25.0 | 70.3 | 91.3 |
| Point-BERT | Objaverse(no LVIS)+ShapeNet | OpenShape | ✓ | 38.8 | 68.8 | 83.9 | 97.6 |
| Point-BERT | Objaverse+ShapeNet | OpenShape | ✓ | 46.5 | 76.3 | 82.6 | 96.9 |
| Point-BERT | Objaverse+ShapeNet+(2 extra) | OpenShape | ✓ | 46.8 | 77.0 | 84.4 | 98.0 |
| Point-BERT | ShapeNet | ULIP | ✓ | 2.6 | 8.1 | 60.4 | 84.0 |
| Point-BERT | ShapeNet | ULIP-2 | ✗ | 16.4 | 34.3 | 75.2 | 95.0 |
| Point-BERT | Objaverse(no LVIS)+ShapeNet | ULIP | ✓ | 21.4 | 41.9 | 68.6 | 86.4 |
| Point-BERT | Objaverse(no LVIS)+ShapeNet | ULIP-2 | ✗ | 46.3 | 75.0 | 84.0 | 97.2 |
| Point-BERT | Objaverse+ShapeNet | ULIP | ✓ | 34.9 | 61.0 | 69.6 | 85.9 |
| Point-BERT | **Objaverse+ShapeNet** | **ULIP-2** | ✗ | **50.6** | **79.1** | **84.7** | **97.1** |

注意最後一列的預訓練資料**包含 LVIS**（Objaverse 全集）。這就是釋出的、我們在用的 checkpoint（README「ensembled objaverse-shapenet」）。

### 3. Method（PAPER）

**3.1 Preliminary: ULIP** —— ULIP 的三元組：(1) 從 3D 形狀抽點雲；(2) 多視角渲染影像；(3) 把 metadata 的描述詞與類別名套模板成句子。ULIP 用 **SLIP 的 ViT-B** 編碼器，把 3D 特徵對齊到語言／影像共享空間。ULIP-2 沿用其預訓練框架，並以 ULIP 為基線。

**3.2 Scalable Triplet Creation** —— 給一個 3D 物件：從表面抽點雲；從多個視角生成影像；用 **BLIP-2** 為**每張**渲染圖生成描述；每張圖生成**一組句子**，用 **CLIP 相似度排名**，**取 top-1 彙整**成三元組裡的語言模態。只需要 3D 資料本身。

**3.3 Tri-modal Pre-training** ——
- 用 OpenCLIP 最大的編碼器 **ViT-G/14**，「**freeze it during the pre-training**」；其已預對齊的特徵空間是目標空間，要把 3D 模態併進去。
- 每一步：給 3D 形狀 **O**，抽點雲 **P**，「**randomly sample** its 2D rendered image I ~ render(O)」，「with its BLIP-2 generated language description **T ~ blip2(I)**」——文字是**那張被抽到的圖**的描述。
- f^I = E_I(I)、f^T = E_T(T) 由**凍結**的影像／文字編碼器算出；要訓練的是 3D encoder E_P，使 f^P = E_P(P) 與其影像、文字特徵對齊。
- **Eq. 1**（3D↔影像，CLIP 式對比）：

  L_P2I = −½ Σ_i [ log exp(f_i^P·f_i^I/τ) / Σ_j exp(f_i^P·f_j^I/τ) + log exp(f_i^P·f_i^I/τ) / Σ_j exp(f_j^P·f_i^I/τ) ]

  i、j 是取樣索引，**τ 是可學的溫度**。第一項：同一樣本的 3D·影像內積要在「影像換成別人」之中突出；第二項：在「3D 換成別人」之中突出。
- **Eq. 2**（3D↔文字）：同型，把 I 換成 T。
- **Eq. 3**：min over E_P of L_P2I + L_P2T。**沒有影像↔文字項**（那兩塔凍結）。

**3.4 Scaling Up** —— 把視覺語言 backbone 從 ViT-B 換到更大，並研究 3D backbone 尺寸；其他設定不變；結果在 Table 9。

### 4. Experiments（PAPER）

**4.1 三元組建構** ——
- **Objaverse**：約 **800K** 真實 3D 形狀，每個 metadata 有 "name" 欄。用 **Blender** 渲染 **12** 張圖，**每 360/12 度一張**。每張圖用 **BLIP-2-opt6.7B** 獨立生成 **10** 段描述，用 **CLIP-ViT-Large** 圖文相似度排名，依 §5.4 消融**取 top-1**。跟 ULIP 與 OpenShape 一樣，每個形狀取 **10k、8k、2k** 三種點數。釋出為 ULIP-Objaverse triplets。
- **ShapeNet**：公開子集約 **52.5K** 形狀、**55** 類。跟 ULIP 一樣取 **30** 個等距視角，每個視角渲染 **RGB 圖與深度圖**。描述生成方法同 Objaverse。釋出為 ULIP-ShapeNet triplets。
- 「更多實作細節與消融在附錄」——但附錄只有 A.1、A.2 兩個消融，**沒有任何訓練超參**（見下方 UNKNOWN 清單）。

**Table 2 統計**：ULIP-Objaverse 點雲 ~800k、影像 ~10 million、語言 ~100 million；ULIP-ShapeNet 點雲 ~52.5k、影像 ~3 million、語言 ~30 million。（800k×12 ≈ 9.6M 張圖；×10 句 ≈ 96M 句，對得上「每張圖 10 句」都釋出、訓練只用 top-1。）

**4.2 下游任務** ——
- ModelNet40：合成 CAD，~9.8k 訓練／~2.5k 測試。
- **Objaverse-LVIS：Objaverse 的子集，有人工驗證的類別標籤，~46k 樣本、~1.2k 類**，用於開放世界 zero-shot 分類。**論文把它當測試集，沒有任何 LVIS 內部切分。**
- ScanObjectNN：真實掃描，~2.9k 樣本、15 類。
- 「follow the same dataset setup and preparation protocols used in ULIP and OpenShape」。
- 三個任務：zero-shot 3D 分類、標準 3D 分類（微調）、3D→語言生成。
- 指標：zero-shot 用 **top-1／top-5 accuracy**；標準分類用 overall／class-average accuracy；生成用 **CIDEr**（依 X-InstructBLIP）。**沒有任何檢索（retrieval）指標。**
- Backbone：Point-BERT（transformer，zero-shot 強）、PointNeXt（輕量，ScanObjectNN 強）。

**4.3 與基線比較** ——
- Zero-shot：「follow the same procedure as in ULIP and OpenShape」。同資料下 ULIP-2 vs ULIP：ShapeNet 預訓練 MN40 top-1 **+14.8**、LVIS top-1 **+13.8**；Objaverse(no LVIS)+ShapeNet 預訓練 MN40 **+15.4**、LVIS **+24.9**。超過 OpenShape 的 46.8。
- Figure 3：X-InstructBLIP 框架做 3D→語言生成的示意。
- **Table 3 ScanObjectNN（hardest set）**：PointNet 3.5M 68.2/63.4；PointNet++ 1.5M 77.9/75.4；DGCNN 1.8M 78.1/73.6；MVTN 11.2M 82.8；RepSurf-U 1.5M 84.6；Point-MAE 22.1M 85.2；PointMLP 12.6M 85.7/84.4；Point-M2AE 15.3M 86.4；PointCMT 12.6M 86.7/84.8；ACT 22.1M 88.2；P2P 89.3；Recon-s 19.0M 89.5；I2P-MAE 12.9M 90.1；**Point-BERT official 83.1 → w/ULIP 88.7 → w/ULIP-2 89.7**（+6.6 對 official）；**PointNeXt scratch 87.5/85.9 → w/ULIP 90.1/89.2 → w/ULIP-2 91.1/90.3 → 加 voting 91.5/90.9**。
- 3D→語言：把凍結的 Point-BERT（ULIP vs ULIP-2 預訓練，同樣 Objaverse+ShapeNet）插進 X-InstructBLIP，其他不變；**Table 4 CIDEr 132.2 → 160.5（+28.3）**。

### 5. Ablation（PAPER，全部在 ShapeNet 預訓練 + SLIP ViT-B、ModelNet40 zero-shot 的設定下，除非另註）

| 表 | 變因 | 結果（top-1 / top-5） |
|---|---|---|
| Table 5 | 語言模態：人工描述 vs top-1 整體 BLIP-2 描述（其他全同 ULIP） | 60.4/84.0 → **69.7/88.1** |
| Table 6 | BLIP vs BLIP-2 | 67.7/88.6 vs **69.7/88.8** |
| Table 7 | 視角數（每視角配其 top-1 描述）1／2／15／30 | 54.8/77.9、58.1/80.5、69.3/88.6、**69.7/88.8** |
| Table 8 | 每視角取 top-k 描述（k=1／3／5／10；k>1 時「ensemble」多句） | **69.7/88.8**、66.7/87.2、66.4/87.7、66.3/85.1 |
| Table 9 | CLIP 大小 × 3D encoder 參數量（Objaverse no-LVIS 預訓練；MN40 與 LVIS） | ViT-B 21.9M：71.4/89.7、28.3/52.6；ViT-G 21.9M：76.3/94.1、35.0/62.5；ViT-G 5.3M：75.0/94.7、34.1/61.1；**ViT-G 32.5M：77.0/94.0、35.7/62.9（選用）**；43.1M：76.8/94.8、35.9/62.6；85.7M：76.5/94.7、35.9/62.7 |

- §5.4 的解讀：top-1 CLIP 排名的描述「更抗雜訊」。
- §5.5：CLIP 越大越好；3D backbone 在 ~32.5M 到頂，之後邊際遞減，故選 32.5M；只用 ShapeNet 時用 5.3M。
- Table 8 圖說補充：「top-5 BLIP-2 captions selected means that in the pre-training, we will ensemble the top-5 CLIP ranked captions as the language modality」——這對應程式碼裡「一個樣本多句→各自正規化→平均→再正規化」的 forward（見第二部分 §2.2）。

### 6. 結論、限制、影響（PAPER）

- 限制：只用**物件級** 3D 資料，跟場景級的分布與複雜度不同；場景級是未來方向。
- 廣泛影響：減少人工標註，可能影響低技術勞動市場。

### 附錄 A（PAPER）

- **A.1 Table 10**（Point-BERT w/ ULIP-2，Objaverse+ShapeNet 預訓練，OpenCLIP ViT-G，LVIS zero-shot）：**8k xyz 48.9/77.1；10k xyzrgb 50.6/79.1**。理由：為了公平對比 OpenShape（用 10k 彩色點雲），「adopt the same 3D input preprocessing as in OpenShape」。
- **A.2 Table 11**（ShapeNet 預訓練 + SLIP ViT-B，MN40）：PointNeXt ULIP 56.2/77.0 → ULIP-2 **72.8/95.7**；Point-BERT ULIP 60.4/84.0 → ULIP-2 **75.2/95.0**。因 Point-BERT 較好放大、zero-shot 較強，主要實驗都用 Point-BERT。

### 論文**沒有**寫的（UNKNOWN，論文層級）

1. 任何優化超參：學習率、優化器、weight decay、batch、epoch、warmup、AMP。
2. 訓練時的點雲增強、影像增強、影像解析度。
3. Point-BERT 是從頭訓還是從 Point-BERT 自監督權重初始化。
4. 12 個視角的仰角、相機距離、背景、光照（只寫「每 30° 一張」）。
5. BLIP-2 生成用的 prompt、取樣參數。
6. 「ensembled」checkpoint 的確切訓練資料組合（README 說 Objaverse+ShapeNet；Table 1 最後一列同）。
7. 任何檢索協定或指標；任何 LVIS 內部 train/val/test 切分。
8. 如何選 checkpoint（哪個 epoch）。

---

## 第二部分：官方程式碼逐檔

### 2.0 repo 全貌與 README（CODE）

- README：ULIP（CVPR 2023）與 ULIP-2（CVPR 2024）共用一個 repo。**在 8 張 A100 上預訓練；CUDA 11.0、PyTorch 1.10.1、Python 3.7.15**。
- 資源 2025-05-23 全部搬到 Hugging Face `SFXX/ulip`；2024-06-17 釋出 CVPR2024 版「ensembled pre-trained model (10k xyzrgb points)」；README 明說 CVPR2024 版模型比初版 arXiv 好，數字要對 CVPR2024 版。
- README 表：`openshape-pointbert-10kxyzrgb-ensembled-objaverse-shapenet-abo-3d_future` 46.8/77.0/84.4/98.0；`ULIP2-PointBERT-10kxyzrgb-ensembled-objaverse-shapenet` **50.6/79.1/84.7/97.1**。
- 支援四個 3D backbone：PointNet2(ssg)、PointBERT、PointMLP、PointNeXt。預訓練腳本「預設 8 GPU」。
- README 的資料準備段落**全是 ULIP-1 的**（ShapeNet-55、ModelNet40、`initialize_models`）；**沒有 ULIP-2 Objaverse 三元組的訓練說明**。
- requirements：`open-clip-torch==2.24.0`、`timm==0.4.12`、`easydict`、`open3d`、`h5py`、`wandb`、`lmdb`、`termcolor`。

**repo 裡沒有的東西（CODE，確認缺席）**：ULIP-2 Objaverse 三元組的訓練 Dataset 類、BLIP-2 生成描述的腳本、Blender 渲染腳本、CLIP 排名腳本、「ensemble」Objaverse+ShapeNet 的資料合併邏輯、ULIP-2 的 pretrain 腳本。釋出的是**資料**與 **checkpoint**，不是完整的 ULIP-2 訓練管線。

### 2.1 `main.py`（576 行，改自 SLIP）

**參數與預設（`get_args_parser`, :34-86）**

| 參數 | 預設 | 備註 |
|---|---|---|
| `--pretrain_dataset_name` | `shapenet` | ULIP-1 的 |
| `--pretrain_dataset_prompt` | `shapenet_64` | 64 句模板 |
| `--validate_dataset_name` | `modelnet40` | |
| `--validate_dataset_prompt` | `modelnet40_64` | |
| `--use_height` | off | PointNeXt 才開 |
| `--npoints` | 8192 | ULIP-2 測試腳本給 10000 |
| `--model` | `ULIP_PN_SSG` | |
| `--epochs` | **250** | |
| `--warmup-epochs` | **1** | |
| `--batch-size` | **64**／GPU | |
| `--lr` | **3e-3** | |
| `--lr-start` | 1e-6 | warmup 起點 |
| `--lr-end` | 1e-5 | cosine 終點 |
| `--update-freq` | 1 | 梯度累積 |
| `--wd` | **0.1** | |
| `--betas` | **(0.9, 0.98)** | |
| `--eps` | 1e-8 | |
| `--eval-freq` | 1 | **實際沒用到**（迴圈寫死 `epoch % 1 == 0`，:219） |
| `--disable-amp` | off → **AMP 預設開** | |
| `--workers` | 10 | |
| `--seed` | 0 | `seed = args.seed + rank`（:100） |
| `--evaluate_3d` / `--evaluate_3d_ulip2` | | 純測試模式 |
| `--test_ckpt_addr` | | |

**主流程（`main`, :90-264）**
- 建模：`getattr(models, args.model)(args=args)`，DDP `find_unused_parameters=False`。
- 損失：`models.get_loss(args)` → `ULIPWithImageLoss`（不管哪個模型都同一個）。
- **優化器分組（:124-138）**：`requires_grad=False` 的參數跳過並印 `in optimizer freeze`；`p.ndim < 2` 或名字含 `bias`／`ln`／`bn` 的 → weight decay 0；其他 → wd 0.1。AdamW(lr, betas, eps)。`GradScaler(enabled=not disable_amp)`。
- Resume：`--resume` 指定檔，`strict=False`；否則自動找 `output_dir/checkpoint.pt`（注意 `save_on_master` 存的檔名是 `checkpoint_{epoch}.pt` 與 `checkpoint_best.pt`，**沒有** `checkpoint.pt`，所以自動 resume 實際上不會觸發——OBSERVED）。
- `cudnn.benchmark = True`。
- **影像 transform（:176-182）**：`RandomResizedCrop(224, scale=(0.5, 1.0))` → `ToTensor` → `Normalize(ImageNet mean/std)`。**這是 SLIP/ULIP-1 的前處理**；`ULIP2_PointBERT_Colored` 從 `open_clip.create_model_and_transforms` 拿回的 `preprocess` 被丟掉（`ULIP_models.py:354` 的 `_, preprocess` 沒被使用），也就是說**若照這份 main.py 訓 ULIP-2，影像會用 ImageNet 正規化而不是 OpenCLIP 自己的**——OBSERVED；實際 ULIP-2 用什麼，UNKNOWN（訓練管線不在 repo）。
- Dataset：`get_dataset(train_transform, tokenizer, args, 'train')` 與 `'val'`；tokenizer 是 repo 自帶的 `SimpleTokenizer`（CLIP BPE）。
- DataLoader：train `drop_last=True`、`customized_collate_fn`；val `drop_last=False`。
- **LR 排程（:203-204）**：`utils.cosine_scheduler(lr, lr_end, epochs, iters_per_epoch, warmup_epochs, lr_start)`，**逐 iteration** 的表。
- **每個 epoch（:212-264）**：`train()` → `test_zeroshot_3d_core(val_loader)` → `acc1 > best_acc1` 才是 best（嚴格大於）→ **is_best 或 epoch % 50 == 0 才存檔**（`checkpoint_{epoch+1}.pt`，best 再複製成 `checkpoint_best.pt`）→ 最後一個 epoch 另存 `checkpoint_last`（`epoch: 'last'`）。log 寫 `log.txt` 一行 JSON。

**`train()`（:267-347）**
- `model.train()`。
- 每個 batch：`lr = lr_schedule[it]` 寫進所有 param_group；**輸入取法 `pc = inputs[3]`、`texts = inputs[2]`、`image = inputs[4]`**——這是 `ShapeNet.__getitem__` 的回傳順序 `(taxonomy_id, model_id, tokenized_captions, data, image)`，所以 `main.py` 的訓練迴圈**綁死在 ULIP-1 ShapeNet 資料類**上。
- `amp.autocast` 下 `outputs = model(pc, texts, image)` → `criterion(outputs)` → `loss /= update_freq`。
- loss 非有限 → `sys.exit(1)`。
- `scaler.scale(loss).backward()`；到累積步才 `scaler.step` / `scaler.update` / `zero_grad(set_to_none=True)`。
- **每步之後 `logit_scale.data.clamp_(0, 4.6052)`**（註解寫「clamp logit scale to [0, 100]」，exp(4.6052) ≈ 100）。
- 記 `loss`、`ulip_loss`、`ulip_pc_image_acc`、`ulip_pc_text_acc`、`lr`、`logit_scale`、記憶體。

**`test_zeroshot_3d_core()`（:350-441）**
- `model.eval()`；讀 `data/templates.json[validate_dataset_prompt]`（**64 句**，例：`a point cloud model of {}.`、`There is a {} in the scene.`…）。
- 標籤：`objaverse` 在資料集名裡 → `dataset.lvis_metadata['all_keys']`；否則 `data/labels.json[name]`（modelnet40 有 40 類）。
- 每個類別：64 句 → tokenize → `encode_text` → **L2 正規化 → 對 64 句取平均 → 再正規化** → 堆成 `text_features`。
- 每個 batch：`encode_pc` → 正規化 → `logits = pc @ text.T` → `accuracy(topk=(1,5))` → `scaled_all_reduce`（只有一個 process 時是 identity；程式碼裡自己留了 TODO 說 correct 的 all-reduce 沒做）→ 順便統計每類 top-1／top-5（算了但只留在變數裡，沒印）。
- 回傳 `{'acc1', 'acc5'}`。

**`test_zeroshot_3d()` vs `test_zeroshot_3d_ulip2()`（:443-496）**
- ULIP-1 版：去掉 `module.` 前綴，優先用 checkpoint 內存的 `args.model` 建模，`strict=True`。
- **ULIP-2 版：用命令列 `args.model` 建模，`strict=False`**——所以釋出的 checkpoint 可以只含 point_encoder／pc_projection／logit_scale，OpenCLIP 的權重從 `open_clip` 線上載入（跟我們量到的 checkpoint 內容一致——DATA，見 `ulip_backbone.py` 檔頭註解）。

**`accuracy()`（:555-569）**：標準 topk。

### 2.2 `models/ULIP_models.py`（445 行）

- `LayerNorm`（fp16 轉 fp32）、`QuickGELU`、`ResidualAttentionBlock`、`Transformer`：這是 **ULIP-1 自帶的 CLIP 文字塔**（從 openai/CLIP 改），ULIP-2 不用。
- **`ULIP_WITH_IMAGE`（:71-174，ULIP-1）**：`visual` = timm `vit_base_patch16_224`；文字 transformer width 512、12 層、8 頭、context 77、vocab 49408；`image_projection` 768→512、`text_projection` 512→512；`logit_scale = log(1/0.07)`；`pc_projection` = `Parameter(pc_feat_dims × 512)`，`normal_(std=512^-0.5)`。`encode_pc = point_encoder(pc) @ pc_projection`。`forward`：每個樣本的多句文字各自 `encode_text` → 正規化 → 平均 → 再正規化。
- **`ULIP2_WITH_OPENCLIP`（:177-232，ULIP-2）**：
  - `self.open_clip_model` = 傳入的 OpenCLIP 模型；`encode_image`／`encode_text` 直接呼叫它。
  - `logit_scale = Parameter(log(1/0.07))`（≈ 2.659，即 τ 初值 0.07）。
  - `self.tokenizer = open_clip.get_tokenizer('ViT-bigG-14')` —— **建立了但 forward 沒用**；訓練時 token 來自 dataset 的 `SimpleTokenizer`。兩者都是 CLIP BPE、77 長度（OBSERVED；等價與否沒有實測——UNKNOWN 的小項）。
  - **`pc_projection = Parameter(pc_feat_dims × 1280)`，`normal_(std=1280^-0.5)`**；1280 = ViT-bigG 的嵌入維。
  - `forward(pc, text, image)` 跟 ULIP-1 同型：多句正規化→平均→再正規化；回傳 `text_embed`、`pc_embed`、`image_embed`、`logit_scale.exp()`。
- `get_loss` → `ULIPWithImageLoss`；`get_metric_names` 四個名字。
- ULIP-1 的 builder（`ULIP_PN_SSG` :243、`ULIP_PN_MLP` :278、`ULIP_PointBERT` :314、`ULIP_PN_NEXT` :372、`ULIP_CUSTOMIZED` :409）：建 timm ViT-B + 3D backbone；非測試模式時載入 `./data/initialize_models/slip_base_100ep.pt`，**名字對得上的參數 `requires_grad=False` 並印 `load … and freeze`**——這是明確凍結 SLIP 的寫法。
- **`ULIP2_PointBERT_Colored`（:352-370）**：
  1. `open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')`；
  2. **`open_clip_model.eval()`**，**沒有任何 `requires_grad=False`**；
  3. 讀 `./models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml`，`PointTransformer_Colored(config.model, args=args)`，`pc_feat_dims = 768`；
  4. 包成 `ULIP2_WITH_OPENCLIP`。

  **論文 vs 程式碼的出入（CODE vs PAPER）**：論文 §3.3 明說凍結；這個 builder 只 `eval()`。若照 `main.py` 直接訓，`named_parameters()` 裡 OpenCLIP 的參數 `requires_grad` 仍是 True，會被放進 AdamW。釋出的 checkpoint 只含 point_encoder（226 個張量）、pc_projection、logit_scale（DATA，`ulip_backbone.py` 檔頭記錄），**與凍結一致**；ULIP-2 實際訓練的程式不在 repo，所以「他們訓練時到底有沒有凍結」從程式碼無法證明，只能依論文與 checkpoint 內容判定為凍結。我們的 `ULIPBackbone` 明確 `requires_grad_(False)` 且 `eval()`（OBSERVED，`ulip_backbone.py:285-292`）。

### 2.3 `models/losses.py`（62 行）

```
labels = local_batch_size * rank + arange(local_batch_size)      # 每張卡自己的正例索引
pc, text, image 各自 F.normalize(dim=-1)
pc_all, text_all, image_all = all_gather_batch([...])              # 無梯度的 all_gather
logits_per_pc_text  = s · pc    @ text_all.T
logits_per_text_pc  = s · text  @ pc_all.T
logits_per_pc_image = s · pc    @ image_all.T
logits_per_image_pc = s · image @ pc_all.T
loss = [CE(pc→text) + CE(text→pc)]/2 + [CE(pc→image) + CE(image→pc)]/2
```
- `s = logit_scale.exp()`，等於論文的 1/τ。`F.cross_entropy` 預設對 batch 取 **mean**；論文 Eq.1-2 寫的是 Σ_i，差一個 batch 常數，優化上等價於 lr 縮放。
- 負例 = **所有 GPU** 的樣本（8 卡 × 64 = 512 個候選）；all_gather 是無梯度版（`all_gather_batch`），`GatherLayer`／`all_gather_batch_with_grad` 存在但**沒被用**。
- 順便算 `pc_text_acc`、`pc_image_acc`（batch 內 argmax 命中率）。
- **沒有 image↔text 項**，與 Eq. 3 一致。

### 2.4 `models/pointbert/point_encoder.py`（354 行）

- `Mlp`（fc→GELU→drop→fc→drop）、`Attention`（`qkv_bias=False`、scale = head_dim^-0.5）、`Block`（**pre-norm**：`x + drop_path(attn(norm1(x)))`，`x + drop_path(mlp(norm2(x)))`，mlp_ratio 4）、`TransformerEncoder`（**每一層都 `block(x + pos)`**：位置嵌入在每層重新加，不是只加一次）。
- **`PointTransformer`（ULIP-1，xyz）**：`Group(num_group, group_size)` → `Encoder(encoder_dims)` → `reduce_dim`（256→384）→ `cls_token`（zeros）、`cls_pos`（randn）→ `pos_embed` = Linear(3,128)→GELU→Linear(128,384) → `blocks`（drop_path 依 `linspace(0, drop_path_rate, depth)` 逐層遞增）→ `LayerNorm` → **readout = concat[cls token, 其餘 token 的 max]** = 2×384 = **768**。非測試模式會 `load_model_from_ckpt('./data/initialize_models/point_bert_pretrained.pt')`（把 `transformer_q.` 前綴去掉，`strict=False`）。
- **`PointTransformer_Colored`（ULIP-2，xyzrgb）**：結構同上，只差 `Encoder(..., input_dim=6)`；**沒有 `load_model_from_ckpt` 的呼叫，直接印 `training from scratch for pointbert.`** 並印參數量。→ 從程式碼看，ULIP-2 的 Point-BERT **是從頭訓**（CODE）；論文沒寫（PAPER 沉默）。`load_model_from_ckpt` 方法保留但沒人呼叫。
- `get_loss_acc`／`build_loss_func`：Point-BERT 分類殘留，ULIP 不用。

### 2.5 `models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml`（31 行）與 `PointTransformer_8192point.yaml`

| 區塊 | 值 | 有沒有被 ULIP 讀 |
|---|---|---|
| `model` | `trans_dim 384`、**`depth 18`**、`drop_path_rate 0.1`、`cls_dim 40`、`num_heads 6`、`group_size 32`、`num_group 512`、`encoder_dims 256` | **有**（`ULIP_models.py:364` 只傳 `config.model`） |
| `optimizer` AdamW lr 5e-4 wd 0.05 | | **沒有**（Point-BERT 重建訓練的殘留） |
| `scheduler` CosLR 200 epochs, initial 10 | | 沒有 |
| `npoints 10000`、`total_bs 32`、`step_per_update 1`、`max_epoch 300`、`grad_norm_clip 10`、`consider_metric CDL1` | | 沒有（`CDL1` = Chamfer L1，是 dVAE 重建指標） |

8192 版差別只有 `depth 12`、`npoints 8192`、scheduler 300。**釋出的 ULIP-2 checkpoint 確實是 18 層**（DATA，先前從 `.pt` 數過，記在 `upstream-lookup.md`）。

### 2.6 `models/pointbert/dvae.py`（354 行）

- **`Group`**：輸入 B×N×C；C>3 時拆 xyz 與 rgb；**`misc.fps(xyz, num_group)`** 取 512 個中心；`knn_point(group_size=32)`（純 torch：`square_distance` + `topk(largest=False, sorted=False)`）；鄰域座標**減去中心**（局部座標）；rgb 用同樣索引取出**原樣**接在後面（不減、不正規化）；回傳 `neighborhood B×512×32×6`、`center B×512×3`。
- **`Encoder`**（mini-PointNet）：`Conv1d(input_dim→128)→BN→ReLU→Conv1d(128→256)` → 對 32 點取 max 得全域 256 → 與每點特徵 concat 成 512 → `Conv1d(512→512)→BN→ReLU→Conv1d(512→encoder_dims)` → 再 max → B×512×256。
- `DGCNN`、`Decoder`、`DiscreteVAE`：Point-BERT 的 dVAE tokenizer，**ULIP 對齊路徑不用**（`point_encoder.py` 只 import `Group` 與 `Encoder`）。但 `from knn_cuda import KNN` 在**模組層級**執行，沒裝就 import 失敗——我們的 `ulip_patch.py` 放了替身（OBSERVED，見 `NOTE_20260904_ULIP2_CHECK_AND_STAGE_SUMMARY.md`）。

### 2.7 `models/pointbert/misc.py`（296 行）

- 活的：`fps(data, number)` = `pointnet2_ops.furthest_point_sample` + `gather_operation`（CUDA 擴充；我們換成純 torch 同演算法，見上）。
- 其餘：`index_points`、`worker_init_fn`、lambda 排程、BN momentum 排程、`seprate_point_cloud`、matplotlib 畫圖、`random_dropping`、`random_scale`——Point-BERT 殘留，ULIP 路徑沒用。兩個被註解掉的 `fps` 純 torch 版本（隨機起點）留在檔案裡。

### 2.8 `data/dataset_3d.py`（667 行）

**通用函式**：`pil_loader`（RGB）；`pc_normalize`（去質心、除最大半徑）；`farthest_point_sample`（numpy 版，隨機起點）；增強：`rotate_point_cloud`（繞 y 軸隨機 0–2π）、`random_point_dropout`（每朵隨機 0–87.5% 的點**被換成第 0 個點**）、`random_scale_point_cloud`（0.8–1.25）、`shift_point_cloud`（±0.1）、`jitter_point_cloud`、`rotate_perturbation_point_cloud`（三軸小角度）。

**`ModelNet`（驗證用）**：讀 `modelnet40_{train,test}.txt`；優先讀 `modelnet40_{split}_{npoints}pts_fps.dat` 快取；`pc_normalize(xyz)`；**若 `use_10k_pc and use_colored_pc`：改讀 `modelnet40_colored_10k_pc.npy`，rgb 一律填 0.4**（灰色常數）；`use_colored_pc` 單獨為真時也填 0.4。test split 不 shuffle 點。回傳 `(points, label, label_name)`。

**`ShapeNet`（ULIP-1 訓練用；也是 `main.py` 訓練迴圈唯一相容的類）**：
- 讀 `taxonomy.json` 的 synset 名；`{subset}.txt` 列表；`whole=True` 時**把 test.txt 也併進訓練列表**（`Dataset_3D` 寫死 `'whole': True`）。
- 點雲：`IO.get` 讀 npy；`uniform=True` 且 npoints < N 時 numpy FPS，否則隨機取；`pc_norm`；`augment=True`：dropout → scale → shift → 小旋轉 → 繞 y 大旋轉。
- 文字：synset 名以逗號拆成同義詞，**隨機挑一個**；`use_caption_templates=False` → **不套模板，一句**；tokenizer → `(1, 77)`。
- 影像：**30 個角度（0,12,…,348）隨機一個**，**RGB 或深度圖（`''` / `'_depth0001'`）隨機一種**，`train_transform`。壞圖直接 raise。
- 回傳 `(taxonomy_id, model_id, tokenized_captions, data, image)`。

**`Objaverse_Lvis_Colored`（ULIP-2 的 LVIS 測試用）**：
- `npoints = 10000` 寫死；讀 `data/objaverse-lvis/lvis.json`（uid → npy 相對路徑）與 `objaverse_lvis_metadata.json`（`value_to_key_mapping`、`key_to_id`、`all_keys`）。
- npy 是 dict：`xyz`、`rgb`；**`pc_norm(xyz)` 後直接 concat `rgb`（rgb 不動）**；`use_color=True` 寫死；`use_height=False` 寫死。
- 回傳 `(data, label, name)`。**它沒有 `__getitem__` 給訓練用的三元組**——再次證明 ULIP-2 Objaverse 訓練 Dataset 不在 repo。
- `dataset_catalog.json` 把它的 `usage` 標成 `train`，但類別本身不看 `subset`，實際用途是 zero-shot 驗證（`test_ulip2_pointbert_objaverse_lvis.sh`）。

**`customized_collate_fn`**：丟掉 `example[4] is None` 的樣本（影像缺失），其餘同 default_collate。用到 `torch._six`（新版 PyTorch 沒有；我們的 patch 補上）。

**`Dataset_3D`**：`'colored' in args.model.lower()` → `use_colored_pc`；`args.npoints == 10000` → `use_10k_pc`；由 `dataset_catalog.json` 決定 config yaml 與 split。

### 2.9 `utils/utils.py`（241 行）

- `cosine_scheduler(base, final, epochs, niter_per_ep, warmup_epochs, start_warmup)`：warmup 段 `linspace(start, base, warmup_iters)`；其餘 `final + ½(base−final)(1+cos(π·t/T))`；長度 = epochs × iters。**沒有 PyTorch 內建排程器**。
- `save_on_master`：`checkpoint_{epoch}.pt`；best 複製為 `checkpoint_best.pt`。
- `all_gather_batch`（無梯度）／`GatherLayer`＋`all_gather_batch_with_grad`（有梯度，未使用）／`scaled_all_reduce`。
- `init_distributed_mode`：讀 `RANK`/`WORLD_SIZE`/`LOCAL_RANK` 或 SLURM；nccl。
- `get_dataset` → `Dataset_3D(...).dataset`。
- `GaussianBlur`（SimCLR 增強）未使用。

### 2.10 `utils/tokenizer.py`（150 行）

CLIP 的 `SimpleTokenizer`（BPE 49408 詞彙）；`__call__(texts, context_length=77)`：`[SOT] + bpe + [EOT]`，**超過 77 直接截斷 `tokens[:77]`**（截斷時 EOT 會被切掉；open_clip 的 tokenizer 會保留 EOT——這是兩個 tokenizer 的小差別，對 ULIP-2 是否有影響 UNKNOWN，因為訓練用的 tokenizer 不在 repo 可證）。

### 2.11 `scripts/`

- `pretrain_pointbert.sh`：`torch.distributed.launch --nproc_per_node=8 main.py --model ULIP_PointBERT --npoints 8192 --lr 3e-3`——**ULIP-1**。其他三個 pretrain 腳本：pointmlp `--lr 1e-3`、pointnet2 `--lr 3e-3`、pointnext `--lr 1e-3 --use_height`。
- **沒有任何 ULIP-2 的 pretrain 腳本。**
- `test_ulip2_pointbert_modelnet40.sh` / `test_ulip2_pointbert_objaverse_lvis.sh`：`--model ULIP2_PointBERT_Colored --npoints 10000 --evaluate_3d_ulip2 --validate_dataset_name={modelnet40|objaverse_lvis_colored} --test_ckpt_addr $1`。
- 四個 ULIP-1 test 腳本：`--evaluate_3d --npoints 8192`。

### 2.12 `data/*.yaml`、`dataset_catalog.json`、`templates.json`、`labels.json`

- `Objaverse_Lvis_Colored.yaml`：`N_POINTS: 10000`；`DATA_PATH`／`PC_PATH` 是作者機器的絕對路徑（`/export/einstein-vision-hs/...`），而且**類別程式碼根本不讀這兩個欄位**（路徑寫死 `data/objaverse-lvis`）。
- `ShapeNet-55.yaml`：`DATA_PATH`、`PC_PATH`、`IMAGE_PATH`。`ModelNet40.yaml`：40 類、不用法向量。
- `templates.json`：`modelnet40_64` 與 `shapenet_64` 各 **64** 句。`labels.json`：只有 `modelnet40`（40 類）。

### 2.13 其他

- `utils/build.py`＋`utils/registry.py`：mmcv 式 Registry，`DATASETS.build(cfg, default_args)`。`utils/config.py`：yaml 合併與 `_base_`。`utils/io.py`：npy／pcd／h5／txt 讀取。`utils/logger.py`、`models/pointbert/logger.py`：logging。`models/pointbert/checkpoint.py`：缺鍵／多鍵訊息格式化。`models/customized_backbone/`：空模板（輸入 B×N×3，輸出 B×feat）。
- `models/pointmlp/`、`models/pointnet2/`、`models/pointnext/`：其他三個 backbone，**不是我們用的**，我沒有逐字讀（見末段）。

---

## 第三部分：論文與程式碼對照（衝突、沉默、補足）

| 項 | PAPER | CODE | 判定 |
|---|---|---|---|
| CLIP 塔凍結 | §3.3 明說 freeze | `ULIP2_PointBERT_Colored` 只 `eval()`，未設 `requires_grad=False`；ULIP-1 builder 有明確凍結 | **衝突**；checkpoint 只含 3D 側（DATA）→ 依論文＋DATA 判凍結；我們明確凍結 |
| 損失 | Eq.1-3，τ 可學 | `ULIPWithImageLoss` 對稱 P2T + P2I，`logit_scale` 可學、每步 clamp ≤ 4.6052 | 一致；clamp 是 CODE 補的 |
| 每步觀測 | 點雲固定、隨機一張圖、該圖的描述 | `main.py` 迴圈只接 ShapeNet 類（隨機角度、隨機 RGB/深度、隨機同義詞）；ULIP-2 Objaverse 類缺席 | ULIP-2 的實際取樣程式 **UNKNOWN**，只能依論文 |
| Point-BERT 初始化 | 沉默 | `PointTransformer_Colored` 從頭訓（印 `training from scratch`） | CODE 補足：**從頭** |
| Point-BERT 結構 | §5.5 說 32.5M 參數 | yaml `depth 18`、384 維、6 頭、512 群×32 點、輸出 768 | CODE 補足；checkpoint 18 層（DATA） |
| 3D 輸入 | A.1：10k xyzrgb，前處理同 OpenShape | `Objaverse_Lvis_Colored`：`pc_norm(xyz)`＋原樣 rgb | 一致 |
| 影像前處理 | 沉默 | `main.py`：RandomResizedCrop 224 (0.5–1) + ImageNet 正規化；OpenCLIP 的 `preprocess` 被丟掉 | CODE 是 ULIP-1 的；ULIP-2 實際 **UNKNOWN** |
| 優化器／lr／epoch／batch | 沉默 | 預設 AdamW (0.9,0.98) eps 1e-8 wd 0.1（bias/ln/bn/1-D 不衰減）、lr 3e-3、warmup 1 epoch、cosine 到 1e-5、250 epoch、64×8、AMP | CODE 預設＝ULIP-1 腳本值；ULIP-2 是否同值 **UNKNOWN**（無 ULIP-2 腳本） |
| 選 checkpoint | 沉默 | 每 epoch ModelNet40 zero-shot top-1，最好者存 best | CODE 補足 |
| Zero-shot 評估 | §4.3「同 ULIP／OpenShape」 | 64 模板→正規化→平均→再正規化；cosine top-1/5 | CODE 補足；我們用同法重現 LVIS 50.9/79.3（DATA） |
| 文字多句聚合 | Table 8 圖說「ensemble top-k」 | forward：各句正規化→平均→再正規化 | 一致（訓練用 top-1 時只有一句） |
| 檢索指標／LVIS 切分 | 無 | 無 | **兩邊皆無**；MetaFind Table 1 的 ULIP 列是 MetaFind 自訂協定 |

---

## 第四部分：對我們 Stage 1 的意義（OBSERVED IMPLEMENTATION 對照）

| 項 | ULIP-2（PAPER/CODE） | 我們 | 標籤 |
|---|---|---|---|
| CLIP 塔 | 凍結 | 凍結；文字／影像向量走 n06 快取 | 同 |
| Point-BERT | 從頭訓 250 epoch 對齊到 CLIP 空間 | **載入釋出權重**；Stage 1 訓練時可解凍（P1s 系列）或 P13 複製第二份給 query 塔 | MetaFind 加的 |
| 要學的 | Point-BERT + pc_projection + logit_scale | 同 + 兩個 Transformer Fusion | MetaFind 加的 |
| 損失 | 對稱 P2I + P2T，τ 可學（初 0.07） | MetaFind Eq. 5：**單向** query→gallery InfoNCE，**τ = 0.5 固定** | MetaFind PAPER FACT |
| 每步觀測 | 隨機視角 + 該視角 top-1 BLIP-2 句 | 文字 = GPT-4o 結構化描述（P10/P12 試不同序列化）；影像 = 單視角或 12 平均 | MetaFind 論文未寫細節 |
| 點雲增強 | ShapeNet 類有 dropout/scale/shift/rotate；Objaverse 類無（UNKNOWN） | 無 | — |
| lr | 3e-3（CODE 預設；ULIP-2 實值 UNKNOWN） | 1e-4（掃描後；3e-3 把預訓練 Point-BERT 打壞） | 我們量的 |
| epoch | 250 | 10（pilot ladder） | 我們量的 |
| wd 分組 | bias/ln/bn/1-D 不衰減 | 照抄 | 同 |
| 排程 | 逐 iter cosine，warmup 1 epoch 1e-6→lr→1e-5 | 同型 | 同 |
| AMP | 開（fp16 GradScaler） | 預設關；`--amp bf16` 可開（P13b 用） | 差異已記 |
| 選 checkpoint | 每 epoch ModelNet40 zero-shot top-1 | 每 epoch val→val 七格平均 R@1 | 機制同、資料不同 |
| 負例池 | 8 卡 all_gather，512 | 單卡 batch 64 | 差異 |

一句話：ULIP-2 是「把一顆從頭來的 Point-BERT 拉進凍結的 CLIP 空間」，用 800K 物件、250 代、lr 3e-3、512 個負例。MetaFind 拿那顆**已對齊好的** Point-BERT 加 Fusion 做子集→資產檢索，資料只有 36K，所以配方只能繼承**機制**（凍結 CLIP、對比損失、wd 分組、每代驗證選模、cosine 排程），**數值**（lr、epoch、batch）必須自己量。

---

## 第五部分：讀了什麼、沒讀什麼（誠實清單）

**逐字讀完**：論文 HTML 全文（含 Table 1–11、附錄、參考文獻列表）；`main.py`、`models/ULIP_models.py`、`models/losses.py`、`models/pointbert/point_encoder.py`、`models/pointbert/dvae.py`、`models/pointbert/misc.py`、兩個 pointbert yaml、`data/dataset_3d.py`、`utils/utils.py`、`utils/tokenizer.py`、`utils/build.py`、`utils/config.py`、`utils/io.py`、`models/pointbert/checkpoint.py`、`models/customized_backbone/customized_backbone.py`、全部 `scripts/*.sh`、`data/*.yaml`、`dataset_catalog.json`、`requirements.txt`、`README.md`；`templates.json`／`labels.json` 用程式數過鍵與長度。

**只看了函式清單、沒逐字**：`utils/registry.py`（mmcv Registry）、`utils/logger.py`、`models/pointbert/logger.py`。

**沒讀**：`models/pointmlp/`、`models/pointnet2/`、`models/pointnext/`（不是我們的 backbone）；`CONTRIBUTING*.md`、`CODE_OF_CONDUCT.md`、`AI_ETHICS.md`、`SECURITY.md`、`assets/`；HF `SFXX/ulip` 上的資料卡與檔案清單（本地沒有）。

**仍然 UNKNOWN、兩邊都答不了**：ULIP-2 Objaverse 訓練的 Dataset／取樣程式、影像前處理、實際 lr／epoch／batch、BLIP-2 prompt、Blender 相機參數、ensemble 的資料混合方式。這些若對 MetaFind 重現有影響，只能標 IMPLEMENTATION CHOICE，不能標 UPSTREAM FACT。
