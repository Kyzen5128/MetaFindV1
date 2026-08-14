# 前置調查發現（在設計 graph 之前必須先知道的事）

> 這份文件記錄實際檢查 `docs/metafind_paper.md`、`/home/kyzen/ULIP`、`/home/kyzen/egnn`
> 與本機環境後得到的**硬事實**。§15 的 graph spec 是從這些事實長出來的，不是從模板套出來的。
> 每一條都標了「影響哪個設計決策」。

---

## F1. 論文內部矛盾：`h^(0) = Concat(x_i, t_i)` 會破壞 SE(3) 等變性

**證據**

- 論文 §2.5 Eq.(未編號)：`h_i^(0) = Concat(x_i, t_i)` —— 把 3D 座標 `x_i` 直接塞進節點特徵 `h`。
- 論文 Appendix C 的證明前提：*"Assuming that `h^0` is **invariant** to SE(3) transformations on `x`"*。
- 官方 EGNN 實作 [`egnn_clean.py:95-103`](/home/kyzen/egnn/models/egnn_clean/egnn_clean.py) 的 `forward(h, edge_index, coord, ...)`
  **刻意把 `h` 與 `coord` 分開兩個參數**，`h` 全程不含座標；座標只透過
  `coord2radial()`（`egnn_clean.py:84-93`）以 `||x_i - x_j||²` 這個**不變量**進入訊息。

**結論**：若照 §2.5 字面實作 `Concat(x, t)`，`h^0` 隨 `x` 一起被旋轉平移，
Appendix C 的證明**不成立**，Eq.(4)/(9) 的等變性會實測失敗。
這是論文自身的矛盾，不是我們的實作錯誤。

**影響設計**

- 主線實作採 `h^(0) = t_i`（僅語意），座標走 `coord` 通道 → 等變性成立，與 Appendix C 一致。
- `Concat(x, t)` 版本保留為**對照組**，並掛一個 **Required Audit（RA-1）**：
  它**預期會失敗**等變性測試。這正是 skill §13.4 的典型場景 ——
  必跑、必留紀錄、**但不得阻斷**，其失敗只能縮小「我們復現了 §2.5 的字面寫法」這個 claim。
- 若把它設成 blocking gate，會犯 anti-pattern #10（gate 氾濫）與 #13（為了變綠而放寬檢查）。

---

## F2. ULIP-2 的 embedding 維度是 **1280**，不是 512

**證據**：[`ULIP_models.py`](/home/kyzen/ULIP/models/ULIP_models.py) `ULIP2_WITH_OPENCLIP.__init__`

```python
self.tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 1280))
```

（舊的 `ULIP_WITH_IMAGE` 才是 512。MetaFind 說 build upon **ULIP-2** → 走 1280 這條。）

**影響設計**：Eq.(6) `e_query = Fusion(e_text, e_img, e_pc) + λ·e_layout` 要求
**ESSGNN pooling 後的輸出必須是 1280 維**，否則殘差加不起來。
→ `ESSGNN.out_node_nf` 與 pooling 後的投影維度寫死 1280，並列為 L1 測試。

---

## F3. ULIP 現成的 loss 與 eval **都不能直接用**

**loss**：[`models/losses.py:14-62`](/home/kyzen/ULIP/models/losses.py) `ULIPWithImageLoss`
是**單塔 tri-modal**（pc↔text、pc↔image 四向 cross-entropy），
不是 MetaFind Eq.(5)/(7a)/(7b) 的**雙塔 query↔gallery** 對比。

**eval**：[`main.py:350-441`](/home/kyzen/ULIP/main.py) `test_zeroshot_3d_core` 做的是
**zero-shot 分類**：

```python
logits_per_pc = pc_features @ text_features.t()   # text_features = 每個「類別」的 prompt embedding
(acc1, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 5))
```

target 是 LVIS 類別 id，不是 asset id。而論文 Table 1 是
**instance-level 檢索**（48K gallery 裡撈回同一個 asset）。

**佐證**：論文 Table 1 註腳說 baseline 的 "PC Only" 高達 98–99% 是因為
*"retrieval using identical embeddings for both query and gallery"* ——
只有 instance-level 檢索才會有「query 就是 gallery 自己」而趨近 100% 的現象；
zero-shot 分類不會這樣。

**影響設計**

- 必須自寫 `MetaFindDualTowerLoss` 與 `InstanceRetrievalEvaluator` 兩個節點，不能複用。
- baseline 的 PC-Only 灌水現象要**刻意重現**（否則我們的 Table 1 對不上）→ **Required Audit（RA-2）**。

---

## F4. `torch._six` 已在 PyTorch 2.0 移除，本機是 **torch 2.9.1** → repo 開箱即壞

**證據**

```
data/dataset_3d.py:544:from torch._six import string_classes
本機: torch 2.9.1+cu128, cuda True
```

`requirements.txt` 還鎖 `timm==0.4.12`、`open3d==0.16.0`、`open-clip-torch==2.24.0`，
在現代 Python/torch 上不會乾淨安裝。

**影響設計**：獨立的 `n01_env_bootstrap` 節點（§7 N1：失敗模式與其他節點完全不同），
並有 L1 smoke test（ULIP-2 ckpt 真的載入、EGNN forward 真的跑出正確 shape）。
不做這步，後面每個節點都會以「看起來像資料問題」的形式失敗。

---

## F5. 單張 RTX 4090 (24GB) vs 官方腳本假設 **8 張 GPU**

**證據**

```
scripts/pretrain_pointbert.sh:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 ...
本機: 1× NVIDIA GeForce RTX 4090, 24564 MiB
```

ULIP-2 的 frozen backbone 是 open_clip **ViT-bigG-14**（~2.5B 參數）。
Eq.(5) 的分母 `Σ_{A' ∈ B}` 是 **in-batch negatives** —— 對比學習的檢索品質
高度依賴 batch size，單卡 24GB 直接跑會把 batch 壓到遠低於論文設定。

**這是整個復現最大的風險。**

**影響設計 → 架構決策 D2（見下）。**

---

## F6. 磁碟：`/` 只剩 108GB，但 `/mnt/data1` 有 **779GB** — 風險已解除

**證據**

```
/dev/mapper/vgubuntu-root  3.6T  3.5T  108G  98% /            ← home 所在，很緊
/dev/sda1                  3.6T  2.7T  779G  78% /mnt/data1   ← 資料放這裡
README.md:37: Skip downloading the full rendered_images (~1TB) if not needed.
README.md:36: A 420GB subset is available ... under the `only_rgb_depth_images` folder
data/objaverse-lvis  →  不存在，需下載
```

粗估我們自己需要的量：

| 項目 | 估算 |
|---|---|
| ViT-bigG-14 + ULIP-2 checkpoint | ~15 GB |
| 48K 點雲 @10000 pts × (xyz+rgb) × float32 | 240KB × 48K ≈ **11.5 GB** |
| 48K × 11 views 渲染圖 @~100KB | 528K 張 ≈ **53 GB** |
| ProcTHOR-10K | ~10 GB |
| 快取 embedding（D2 的產物） | 48K × 3 × 1280 × 4B ≈ **0.7 GB** |
| **合計** | **~90 GB** |

> **2026-08-14 更新**：所有大型資料改放 `/mnt/data1/kyzen/MetaFind`（779GB 可用），
> repo 內以 `./data` symlink 指向。90GB 對 779GB 綽綽有餘。
>
> **後果：原本的 R-01 風險解除，D3 從「必要」降級為「可選」。**
> 這是設計文件裡少數「因為新事實而放寬」的地方 —— 值得標明，因為
> **放寬約束和放寬檢查是兩回事**：前者是事實變了，後者是為了讓紅燈變綠燈。

**影響設計 → 架構決策 D3（見下，已改寫）。**

---

## F7. GPT-4o 標註是主要金錢成本，且**可能不必付**

48K assets × 11 views。若每個 asset 一次呼叫送 11 張圖：

```
48,000 calls × (~11×1.1K image tokens + prompt) ≈ 12K input tokens/call
                                                 + ~400 output tokens/call
```

以此量級估算約 **US$1.5K–2.5K**（實際單價須以送出當日的官方價目為準，
本估算只用於決定「要不要先跑 pilot」，不作為預算承諾）。

**但是**：ULIP-2 釋出的 Objaverse Triplets **本身就附帶生成好的 captions**
（HuggingFace `SFXX/ulip`）。直接沿用可把這筆成本降到接近 0，
代價是文字分佈與論文的「GPT-4o 結構化描述（category / size / materials /
placement constraints）」不同。

**影響設計**：`n03_preflight_budget` 先用 **500 asset pilot** 實測單價與 schema 通過率，
再決定走 (a) 自行 GPT-4o 標註 或 (b) 沿用 ULIP-2 captions。
這個決策點是 **A1**（依實測成本與預算比較，確定性判定），不是讓模型自己決定。

---

## F8. EGNN 的 `edges_in_d` 就是語意邊的插槽，但**維度會壓垮幾何訊號**

**證據**：[`egnn_clean.py:22-26`](/home/kyzen/egnn/models/egnn_clean/egnn_clean.py)

```python
input_edge = input_nf * 2
edge_coords_nf = 1
self.edge_mlp = nn.Sequential(
    nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf), ...)
```

edge MLP 的輸入維度 = `2×hidden_nf + 1 + in_edge_nf`。
若直接餵 CLIP-1280 的語意邊嵌入、`hidden_nf=128`：

```
256 (語意 h) + 1 (幾何 ||x_i-x_j||²) + 1280 (語意邊) = 1537
```

**幾何訊號只佔 1/1537。** ESSGNN 可能退化成「語意圖神經網路」，
Table 3 想證明的「ESSGNN 優於 GAT 是因為等變性」就無法歸因。

### 2026-08-15 實測（`tests/test_essgnn.py`，固定種子）

用 `|∂e_layout/∂pos|max` 量幾何敏感度，語意邊全部置零：

| `edge_feat_dim` | 幾何敏感度 |
|---|---|
| 16 | **50.9** |
| 1280 | **1.14** |

**加寬語意邊使幾何敏感度下降約 45 倍**，F8 的疑慮確實成立，而且可量化。

一併排除掉一個錯誤的量法：我原本用「零掉語意邊後，兩個幾何不同的 layout 的
`e_layout` 餘弦相似度」當判準，門檻設 0.9999。實測發現**幾何改變 cos=0.9999、
語意改變 cos=0.99999，兩者都接近 1** —— 因為未訓練的殘差網路 `h + Σf(...)`
在 `f` 初始化接近零時由殘差流主導，而兩次前向的 `h⁰` 相同。
那個判準測的是「有沒有訓練過」，不是「幾何有沒有接進去」。

改用梯度之後，兩個問題分開了：
- **幾何有沒有接進去** → `∂e_layout/∂pos ≠ 0`，架構性質，未訓練也成立
- **幾何夠不夠強** → 敏感度比值，且只斷言**方向**（加寬必定壓抑幾何），
  絕對量級要由訓練完的模型在 `n11` 回答

**影響設計（2026-08-15 修正）**：原本我打算加一層 `Linear(1280 → 64)` 投影。
**那是我自作主張，論文沒有這個東西** —— §2.5 的 `f_h : ℝ^(2d+1+e) → ℝ^d` 直接吃
原始維度的 `e_ij`。依「依照論文，不要自己判斷」的指示，**預設不投影**
（`edge_proj_dim=None`）。

投影保留為 config flag 但預設關閉，理由是它讓「有沒有退化」變成可量測的對照，
而不是只能猜。真正的偵測器是 `tests/test_essgnn.py::test_geometry_still_distinguishes_layouts_without_semantic_edges`：
把語意邊全部置零，兩個幾何不同的 layout 仍須產生不同的 `e_layout`。

**若照論文做真的退化了，那是論文設計的性質，要報告出來，不是偷偷修掉。**

**另一個未定**：論文只說文字編碼器是 "e.g., CLIP or BERT"，
所以 `e` 的寬度（1280 / 768 / 512）是輸入而非常數 → 列為 **U-06**。

---

## F10. §2.5 的 `f_x : ℝ^(2d+1+e) → ℝ³` 與 Appendix C 的證明互相矛盾

**證據**

- §2.5 Eq.(3)：`x_i^{l+1} = x_i^l + Σ_j (x_i^l − x_j^l) · f_x(...)`，並定義 `f_x : ℝ^(2d+1+e) → ℝ³`
- Appendix C Eq.(13) 的推導：
  `Σ (Q x_i + g − Q x_j − g) · φ_x(m_ij) = Q Σ (x_i − x_j) · φ_x(m_ij)`
  把 `Q` 提到求和外面，**只有在 `φ_x` 是純量時才成立**。
  若 `f_x` 輸出 ℝ³ 且為逐元素相乘，旋轉不可交換，等變性當場失效。
- 官方 EGNN [`egnn_clean.py:33`](/home/kyzen/egnn/models/egnn_clean/egnn_clean.py)：
  `layer = nn.Linear(hidden_nf, 1, bias=False)` —— **純量**。

**結論**：§2.5 對 `f_x` 值域的宣告是錯的。照字面實作會直接毀掉論文的核心主張。

**影響設計**：`f_x` 一律實作為純量，**不提供 flag**（提供選項等於暗示它是可選的設計，
但它不是，它是論文的筆誤）。此矛盾寫入報告。

---

## F11. `e_layout` 只讀 `h`，導致最後一層的座標 MLP 收不到梯度

**證據**：`e_layout = Pooling({h_i^(L)})` 只用 `h`。第 `l` 層的座標更新之所以有作用，
是因為第 `l+1` 層會用更新後的 `x` 重算 `‖x_i − x_j‖²`。最後一層沒有下一層，
它更新出來的 `x` 沒有任何消費者。

實測（`tests/test_essgnn.py::test_gradients_reach_every_parameter_except_the_final_f_x`）：
`layers.{L-1}.f_x.*` 的梯度全為 `None`，其餘參數都有梯度。

以論文的 `L=4` 計算，**四分之一的座標參數從未被訓練**。

**影響設計**：這是論文架構本身的性質，不是缺陷。**不修**（修了就是偏離論文），
改以測試釘住這個確切模式 —— 若日後 readout 改動，測試會抓到而不是默默吸收。

---

## F9. 其他實作細節

| 項目 | 事實 | 影響 |
|---|---|---|
| `unsorted_segment_sum` 用 `scatter_add_`（`egnn_clean.py:149-154`） | GPU atomics → **非確定性** | 列入 `nondeterminism_sources`；等變性測試容差不能設成 exact |
| `coords_agg` 預設 `'mean'`（`egnn_clean.py:17`） | 論文 Eq.(3) 是 **sum** | 設定必須明寫 `coords_agg='sum'`，列 L1 |
| `E_GCL.forward` 回傳的第三個值是**原封不動的** `edge_attr`（`egnn_clean.py:103`） | 語意邊不會被逐層更新 | 與論文一致（`e_ij` 為常量），寫進 postcondition |
| `get_edges()` 建**全連接**圖（`egnn_clean.py:167-176`） | 我們要的是稀疏的 physical+semantic 邊 | 必須自寫 `edge_index` 建構；不可複用 |
| `Objaverse_Lvis_Colored.__getitem__` 只回傳 `(pc, label, name)` | **沒有圖片、沒有 caption** | tri-modal 資料要自己組，不能複用這個 Dataset |
| `Objaverse_Lvis_Colored` 需要 `data/objaverse-lvis/lvis.json` 等 | 檔案不存在 | 列入 `n02_acquire_sources` 的明確產出 |

---

## 由 F1–F9 推導出的三個架構決策

### D1 — 不重訓 ULIP-2，直接用官方 released checkpoint 當 frozen backbone

**理由**：F5（單卡）。ULIP-2 在 ensembled Objaverse+ShapeNet 上的預訓練需要 8×A100 等級資源。
論文 §2.6 Stage 1 訓練的是 **MetaFind 自己的雙塔**（"both query and gallery encoders"），
ULIP-2 只是 embedding backbone（§2.2 "both leveraging the ULIP-2 embedding backbone"）。
→ 凍結 ULIP-2 的 text/image/point encoder，只訓練 projection + fusion。

**這是與論文的偏離，必須在報告中明說**，並列為 risk R-03。

### D2 — 把 frozen backbone 的輸出**預先算好並快取**，訓練只在 1280-d 向量上進行

**理由**：F2 + F5。既然 backbone 全程 frozen，每個 epoch 重跑 ViT-bigG-14 是純浪費。

```
一次性：48K assets → (text, image, pc) → 3 × 1280-d float32 → 48K × 3 × 1280 × 4B ≈ 737 MB
之後：Stage-1 / Stage-2 / 10 個 ablation 變體全部在這 737MB 上訓練
```

**這是整個設計最重要的一個決策，它同時解決三件事**：

1. **單卡可行**：訓練對象縮成 MLP 級別，batch size 可以開到論文等級甚至更大
   （對比學習的 in-batch negatives 直接受益 → F5 的風險被消掉）。
2. **Ablation 從不可行變成 trivial**：Table 3 的 10 個變體（含需要重跑 Stage-1 的
   `Modality Dropout=10%/50%`、`Padding missing modalities with 0`）
   原本各需一次完整預訓練；快取後每個只是幾分鐘的 head 訓練。**成本下降約兩個數量級。**
3. **磁碟壓力解除**：見 D3。

**代價**：無法微調 point encoder（論文 Table 3 的
「Fine-tuning the entire encoder outperformed training the fuser only」這條結論
我們只能部分驗證）→ 列為 risk R-04 與 Required Audit RA-3。

### D3 — Shard 串流式資料準備（保留串流，**取消強制刪檔**）

**原本的理由**：F6 只有 108GB，必須邊做邊刪。
**現在**：`/mnt/data1` 有 779GB，總需求 ~90GB → **不需要刪原始檔**。

```
for shard in shards(48K assets, size=2000):
    下載 shard 原始 mesh/pc
    渲染 11 views → 標註 → 編碼成 1280-d
    寫入 embedding + per-item sidecar
    # 刪除原始檔：預設關閉（--keep-raw，預設 true）
```

**保留 shard 串流的理由（即使磁碟夠用）**：
- 可中斷續跑的粒度：24 個 shard 各自 checkpoint，比一次 48K 好復原
- backpressure 仍然有用（避免意外把 779GB 塞爆）
- 保留原始渲染圖反而是**好事**：之後想換標註模型、換視角數，不用重渲染

**設計上的變更**：

| 項目 | 原本 | 現在 |
|---|---|---|
| `sg1_delete_raw` 節點 | 必要，`mutate`，不可逆 | **可選**，預設關閉 |
| 對應的 rollback | `compensating_action`（重下載） | 預設不需要；開啟刪檔時才啟用 |
| `RESOURCE` 降級政策 | 縮 shard / `BLOCKED` 等空間 | 保留，但實務上不會觸發 |
| 風險 **R-01** | RISK（最高） | **已解除** |

> per-item sidecar 仍然必須記 `source_uri + sha256`（B3 provenance）——
> 不是為了刪檔後補救，而是為了**產物可追溯**：每個 embedding 都要能指回它從哪來。

---

## 這些發現如何改變 graph 的形狀

| 若沒做這次調查，會畫出的圖 | 實際該畫的圖 |
|---|---|
| `prep → pretrain → finetune → eval` 四段循序 | 前面多一整段 **Stage 0 可行性**（環境修補 + 預算 pilot + 硬體 preflight），且它有自己的 gate |
| 昂貴的是「訓練」，gate 擋在訓練前 | 昂貴的是**標註與編碼**（一次性），訓練反而便宜 → **G-COST gate 前移到資料階段** |
| Ablation 是最貴的尾巴，需要 G-COST gate | Ablation 因 D2 變便宜；**最貴且不可逆的尾巴是 5 位專家 × 200 場景的人工評分** |
| 等變性是「訓練完看一下」的指標 | 等變性是 **G-INVALID 判準**，且 `Concat(x,t)` 版本是預期失敗的 **Required Audit** |
| 資料準備是一次 fan-out | 是**帶 disk backpressure 的 shard 串流**，且含不可逆的刪檔動作 |

---

## 待解 UNKNOWN（未經確認，不得用猜測填空）

| id | 未知項 | 影響 | 如何解除 |
|---|---|---|---|
| U-01 | 場景級檢索的 gallery 到底是 48K Objaverse 還是 3K+ ProcTHOR 資產？論文未明說 | Table 2 的整個評估協定 | 先做小規模雙版本試跑比對；或聯繫作者 |
| U-02 | Table 3 的場景級分數用幾個場景？論文只在 §3.3 說 Table 2 用 200 | Table 3 的可比性與成本 | 先以 50 場景試跑並宣告加寬容差 |
| U-03 | `Padding missing modalities with 0` 是作用在 Stage-1 還是 Stage-2 fusion？ | 該 ablation 要重跑哪一段 | 兩版都跑（D2 之後成本可忍受） |
| U-04 | Table 1 的 gallery 大小（48K 全量？還是測試集 20%？）→ 直接決定 R@1 的分母 | **所有 Table 1 數字的可比性** | 用 baseline PC-Only ≈ 98–99% 反推分母後驗證 |
| U-05 | ProcTHOR 的 `t_i` 用它自己的 metadata 文字，還是要走 GPT-4o 標註？ | SG2 是否相依於 SG1 | 主線假設用 ProcTHOR 自帶 metadata（兩者可平行）；列為假設 A-02 |

**U-04 特別重要**：它是唯一一個「猜錯會讓整張 Table 1 失去意義」的未知。
因此 `G2_corpus_validity` 的判準之一就是「gallery 大小已鎖定並寫入 write_once channel」。
