# ULIP-2 是怎麼訓練的 —— 論文 + 官方程式碼重讀（2026-09-04）

> 2026-09-04 晚：這份是訓練配方摘要；完整逐字讀本（論文全節＋程式碼逐檔＋衝突表）在 `docs/NOTE_20260904_ULIP2_FULL_READ.md`，以那份為準。

Kyzen 指示：「先去讀論文以及 ulip2 官方程式先去了解它怎麼訓練的」。來源：`docs/paper/ulip2_source/ulip2_arxiv_v4.html`
（CVPR 2024 版）與 `/home/kyzen/upstream/ULIP`（官方 repo）。標籤：UPSTREAM PAPER = ULIP-2 論文；UPSTREAM CODE = repo；
兩者衝突處明標。

## 1. 資料：一個 3D 物件 → 三元組（UPSTREAM PAPER §3.2、§4.1）

- 點雲：從表面取樣；釋出三種點數 10k／8k／2k；ULIP-2 Objaverse 主模型用 **10k xyzrgb**（README 149，yaml `N_POINTS: 10000`）。
- 影像：Blender 渲染 **12 張**，每 30° 一張（ShapeNet 用 30 張）。
- 文字：**每張圖**用 BLIP-2-opt6.7B 生 10 句，用 CLIP ViT-L 圖文相似度排名，**取 top-1**。所以每個物件有 12 句描述，一張圖配一句。
- Objaverse 約 800K 物件；三元組已釋出（`SFXX/ulip`），我們下載在 `/mnt/data1/kyzen/ulip2_objaverse_lvis/`。

## 2. 每一步餵什麼（UPSTREAM PAPER §3.3）

> given a 3D shape O, we extract its 3D point cloud P, **randomly sample** its 2D rendered image I ~ render(O), with its BLIP-2 generated
> language description T ~ blip2(I)

每個 batch 裡的每個物件：點雲固定、**隨機挑一張視角圖、文字就是那張圖的 top-1 描述**。文字與圖片是綁在同一視角的。
（repo 裡 `dataset_3d.py` 的 `ShapeNet.__getitem__` 是 ULIP-1 的版本：隨機視角、隨機一句名稱、再套模板；ULIP-2 Objaverse 的
pretrain dataset class **沒有在 repo 裡**，只釋出資料與 checkpoint —— OBSERVED CODE。）

點雲在 dataset 端做 `pc_norm`（去質心、除最大半徑），訓練時 `random_sample` 取點；ShapeNet 版還有 `random_point_dropout`、`random_scale`。

## 3. 模型與凍結（UPSTREAM PAPER §3.3；UPSTREAM CODE `models/ULIP_models.py:180-232, 352-370`）

- 文字／影像編碼器：OpenCLIP **ViT-bigG-14**（laion2b_s39b_b160k）。論文：「freeze it during the pre-training」。
- 點雲編碼器：Point-BERT（`ULIP_2_PointBERT_10k_colored_pointclouds.yaml`，depth 18），後接 `pc_projection`（768→1280，
  `nn.init.normal_(std=1280**-0.5)`），這兩個是**唯一要學的**。
- `logit_scale = log(1/0.07)` 可學，每步 clamp 到 ≤ 4.6052（即溫度 τ ≥ 0.01）。
- 文字 forward：一個樣本可帶多句（模板），各自編碼→正規化→平均→再正規化（`forward`，`ULIP_models.py:210-217`）。

**論文與釋出程式碼的一個出入（OBSERVED CODE）**：`ULIP2_PointBERT_Colored` 只呼叫 `open_clip_model.eval()`，**沒有把 OpenCLIP 參數的
`requires_grad` 設成 False**；`main.py:124-137` 的 optimizer 只排除 `requires_grad=False` 的參數。照釋出的程式直接跑，ViT-bigG 會被
放進 AdamW。ULIP-1 的三個 builder（`:272/:308/:346`）有明確凍結 SLIP。論文明說凍結；我們的 `ULIPBackbone` 也明確凍結（PAPER 勝，
且釋出的 checkpoint 只含 point_encoder／pc_projection／logit_scale，與凍結一致 —— OBSERVED DATA）。

## 4. 目標函數（UPSTREAM PAPER Eq. 1–2；UPSTREAM CODE `models/losses.py:14-50`）

```
L = L_P2I + L_P2T
L_P2I = ½ [ CE(pc→image) + CE(image→pc) ]      對稱 InfoNCE，logits = logit_scale · cos
L_P2T = ½ [ CE(pc→text)  + CE(text→pc)  ]
```
三個向量都先 L2 正規化；負例 = 同一 batch（多卡 all_gather 後）的其他樣本；**沒有 image↔text 項**（那兩座塔是凍結的）。

## 5. 優化與時程（UPSTREAM CODE `main.py` 預設值＋`scripts/pretrain_pointbert.sh`）

| 項 | 值 | 位置 |
|---|---|---|
| optimizer | AdamW，betas (0.9, 0.98)，eps 1e-8 | `main.py:60-61,137` |
| weight decay | 0.1；bias／ln／bn／一維參數不衰減 | `main.py:59,124-135` |
| lr | 預設 3e-3；官方 pretrain 腳本也給 `--lr 3e-3`（8 卡 × batch 64） | `main.py:52`，`scripts/pretrain_pointbert.sh` |
| schedule | cosine，warmup 1 epoch 從 1e-6，尾端 1e-5 | `main.py:48-55,203` |
| epochs | 250 | `main.py:47` |
| batch | 64 / GPU，8 GPU | `main.py:50`，腳本 |
| AMP | 預設開（`--disable-amp` 才關），GradScaler | `main.py:63,139,303` |
| 每步 | 前向→loss→backward→step→`logit_scale.clamp_(0, 4.6052)` | `main.py:303-322` |
| 驗證 | **每個 epoch** 在 `validate_dataset`（預設 ModelNet40）做 zero-shot 分類，top-1 最好者存 `checkpoint_best.pt`；另每 50 epoch 存一份 | `main.py:216-240` |
| seed | 0 + rank | `main.py:99-101` |

Zero-shot 驗證怎麼算（`main.py:355-400`）：每個類別名套 64 個模板句→編碼→正規化→平均→再正規化；點雲編碼→正規化；cosine 取最大。
LVIS 評估同法（`test_ulip2_pointbert_objaverse_lvis.sh`，類別來自 `objaverse_lvis_metadata.json`）。

## 6. 對照我們的 Stage 1（哪些繼承、哪些是 MetaFind 自己的）

| 項 | ULIP-2 | 我們（Stage 1） | 標籤 |
|---|---|---|---|
| CLIP 塔 | 凍結 | 凍結，向量走 n06 快取 | 同 |
| 要學的 | Point-BERT + pc_projection | 同 + 兩個 Transformer Fusion（+ P13 第二份 Point-BERT） | MetaFind 加的是 Fusion |
| 損失 | 對稱 P2I + P2T | MetaFind Eq. 5：**單向** query→gallery，τ = 0.5 固定 | PAPER FACT（MetaFind） |
| 每步觀測 | 隨機視角 + 該視角的 top-1 描述 | 快取：文字 = GPT-4o 整份描述的一種寫法；影像 = 單視角（P1）或 12 平均；P6 試過每步隨機視角 | MetaFind 論文未寫 |
| 文字 | 每視角一句 BLIP-2 | 每資產一份 GPT-4o 結構化描述 | MetaFind PAPER FACT（§2.3） |
| lr | 3e-3（點雲編碼器從頭對齊） | 1e-4（掃描後；3e-3 會把預訓練好的 Point-BERT 打壞，5b） | 我們量的 |
| epochs | 250（從頭對齊 800K 物件） | 10（P1e25 量過 25 不改形狀） | 我們量的 |
| 選 checkpoint | 每 epoch zero-shot ModelNet40 top-1 | 每 epoch val→val 的七格平均 R@1 | 機制同，資料不同 |
| AMP | 開 | 預設關，`--amp bf16` 可開（量過等價） | — |
| wd 分組 | 是 | 照抄 | 同 |

一句話：ULIP-2 是「把一個點雲編碼器從頭拉進凍結的 CLIP 空間」，所以要 800K 物件、250 代、3e-3。MetaFind 是拿那顆已經對齊好的
編碼器再加 Fusion 做「子集找資產」，訓練資料只有 36K，配方不能照抄 ULIP-2 的數值，只能照抄機制（凍結 CLIP、對比損失、wd 分組、
每代驗證選模）。
