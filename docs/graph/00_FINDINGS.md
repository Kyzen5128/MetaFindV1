# 前置調查發現（在設計 graph 之前必須先知道的事）

> 這份文件記錄實際檢查論文與程式碼後得到的**實測事實**（F 系列）與由此推導的**架構決策**（D 系列）。
>
> **權威順序**：`docs/paper/metafind_source/`（作者 arXiv TeX）> `02_BUILD_STEPS.md`（最新決策）
> 公式逐條見 `docs/audit/A_FORMULA_INVENTORY.md`；矛盾見 `C_PAPER_CONTRADICTIONS.md`。
> PDF 轉出的 Markdown 副本已刪除；散文與公式一律讀 `docs/paper/*_source/*.tex`。
> > `01_GRAPH_SPEC.md` > 三個 YAML > 本文件。
>
> F 系列是實測，可信；**D 系列是決策，會隨新事實改變** —— 本文件的 D1/D2 已於 2026-08-15
> 大幅修正（先前的版本把論文列為較差的 ablation 當成主線）。
> 若本文件與 `02_BUILD_STEPS.md` 衝突，**以後者為準**。
>
> ⚠️ **本文件不使用 `U-nn` 也不使用 `R-nn` 編號。** 它曾經有自己的一套 `U-01`…`U-05`、`U-06`、`U-15`，
> 與 `01_GRAPH_SPEC.md` §15 的登記表**編號相同但意義完全不同** —— 例如舊的 `U-04`
> 是「Table 1 gallery 大小」，登記表的 `U-04` 是「渲染解析度」。
> 靠「本文件權威最低」不足以避免這個問題：agent 用關鍵字搜尋時照樣會撈到。
> 它也曾用 `R-01` 指「磁碟空間風險」，而登記表的 `R-01` 是「I-Design 尚未驗證能否執行」。
> 舊編號已全部移除，唯一的登記表在 `01_GRAPH_SPEC.md` §15，
> 機器可讀版在 `graph_spec.yaml` 的 `risks_unknowns`，由 `tools/check_graph.py` 對齊。

---

## F1. 論文內部矛盾：`h^(0) = Concat(x_i, t_i)` 會破壞 SE(3) 等變性

**證據**

- 論文 §2.5 Eq.(未編號)：`h_i^(0) = Concat(x_i, t_i)` —— 把 3D 座標 `x_i` 直接塞進節點特徵 `h`。
- 論文 Appendix C 的證明前提：*"Assuming that `h^0` is **invariant** to SE(3) transformations on `x`"*。
- 官方 EGNN 實作 [`egnn_clean.py:95-103`](/home/kyzen/upstream/egnn/models/egnn_clean/egnn_clean.py) 的 `forward(h, edge_index, coord, ...)`
  **刻意把 `h` 與 `coord` 分開兩個參數**，`h` 全程不含座標；座標只透過
  `coord2radial()`（`egnn_clean.py:84-93`）以 `||x_i - x_j||²` 這個**不變量**進入訊息。

**2026-08-15 逐字重讀後補上兩條獨立證據（不需要等變性論證）**

**其一：Eq.(2) 自己的型別就不成立。** §2.5 定義
`f_h : ℝ^(2d+1+e) → ℝ^d`，而 Eq.(2) 是殘差式 `h^{l+1} = h^l + Σ f_h(...)`，
所以 `h^l` 必須是 `ℝ^d`。但 `t_i ∈ ℝ^d`、`x_i ∈ ℝ³`，
`Concat(x_i, t_i) ∈ ℝ^{d+3} ≠ ℝ^d`。

```
h^0 = Concat(x, t)  ∈ ℝ^{d+3}
h^1 = h^0 + f_h(…)  ∈ ℝ^{d+3} + ℝ^d     ← 加不起來
```

**第 0 層的殘差根本無法相加。** 這不必談旋轉、不必看 Appendix C，
純粹是維度就不對。

**其二：Introduction 自己說是分開的。** 論文第 40 行寫
ESSGNN *"maintains equivariance to rotation and translation by **separating
spatial and semantic channels**"* —— **分離**，正好與 `Concat` 相反。

**結論**：若照 §2.5 字面實作 `Concat(x, t)`，除了 `h^0` 隨 `x` 旋轉平移、
Appendix C 的證明不成立之外，**它連 Eq.(2) 自己的型別都對不上**，
而 Introduction 描述的機制也是相反的。
四條證據都指向同一個方向：這是論文的筆誤，不是我們的實作錯誤。

**影響設計**

- 主線實作採 `h^(0) = t_i`（僅語意），座標走 `coord` 通道 → 等變性成立，與 Appendix C 一致。
- `Concat(x, t)` 版本保留為**對照組**，並掛一個 **Required Audit（RA-1）**：
  它**預期會失敗**等變性測試。這正是 skill §13.4 的典型場景 ——
  必跑、必留紀錄、**但不得阻斷**，其失敗只能縮小「我們復現了 §2.5 的字面寫法」這個 claim。
- 若把它設成 blocking gate，會犯 anti-pattern #10（gate 氾濫）與 #13（為了變綠而放寬檢查）。

---

## F2. ULIP-2 的 embedding 維度是 **1280**，不是 512

**證據**：[`ULIP_models.py`](/home/kyzen/upstream/ULIP/models/ULIP_models.py) `ULIP2_WITH_OPENCLIP.__init__`

```python
self.tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 1280))
```

（舊的 `ULIP_WITH_IMAGE` 才是 512。MetaFind 說 build upon **ULIP-2** → 走 1280 這條。）

**影響設計**：Eq.(6) `e_query = Fusion(e_text, e_img, e_pc) + λ·e_layout` 要求
**ESSGNN pooling 後的輸出必須與 embedding 同寬**，否則殘差加不起來。

> **⚠️ 本節原本寫「寫死 1280，並列為 L1 測試」，那已被推翻。**
> **論文全文沒有出現任何維度數字。** 1280 是我們讀 ULIP-2 checkpoint 得到的事實，
> 不是論文真值。現行規格改為**從 checkpoint 推導寬度**，並禁止任何地方寫死。

---

## F3. ULIP 現成的 loss 與 eval **都不能直接用**

**loss**：[`models/losses.py:14-62`](/home/kyzen/upstream/ULIP/models/losses.py) `ULIPWithImageLoss`
是**單塔 tri-modal**（pc↔text、pc↔image 四向 cross-entropy），
不是 MetaFind Eq.(5)/(7a)/(7b) 的**雙塔 query↔gallery** 對比。

**eval**：[`main.py:350-441`](/home/kyzen/upstream/ULIP/main.py) `test_zeroshot_3d_core` 做的是
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

> **⚠️ 本節原本還寫「baseline 的 PC-Only 灌水現象要刻意重現 → Required Audit（RA-2）」，兩處都錯。**
> **RA-2 是 `f_x → ℝ³`**，與 baseline 無關（編號張冠李戴）。
> 而且我們**不重跑任何 baseline**（偏離 D-3），根本無從重現它們的數字。
> 正確的說法是 **SC-2**：我們自己的 PC-Only 應當**低於**論文公佈的 baseline 值 ——
> 那是要重現的**方向**，不是要重現對方的灌水。

---

## F4. `torch._six` 已在 PyTorch 2.0 移除，本機是 **torch 2.9.1** → repo 開箱即壞

**證據**

```
data/dataset_3d.py:544:from torch._six import string_classes
本機: torch 2.9.1+cu128, cuda True
```

`requirements.txt` 還鎖 `timm==0.4.12`、`open3d==0.16.0`、`open-clip-torch==2.24.0`，
在現代 Python/torch 上不會乾淨安裝。

**影響設計**：獨立的 `n01_env_bootstrap` 節點（分解理由 N1：失敗模式與其他節點完全不同），
並有 L1 smoke test（ULIP-2 ckpt 真的載入、EGNN forward 真的跑出正確 shape）。
不做這步，後面每個節點都會以「看起來像資料問題」的形式失敗。

---

## F5. 單張 RTX 4090（此處的 `RTX 4090` 為前一台機器；本機實測為 RTX 5090 32GB，凡以 24GB 為前提的可行性判斷都要重新量測） (24GB) vs 官方腳本假設 **8 張 GPU**

**證據**

```
scripts/pretrain_pointbert.sh:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 ...
本機: 1× NVIDIA GeForce RTX 4090, 24564 MiB     ← 前一台機器，量於 2026-08-15
```

> ⚠ **[2026-08-30]** 上面那一行是**前一台機器**的量測，保留為紀錄。
> **現在的機器實測是 `NVIDIA GeForce RTX 5090, 32,607 MiB`**（`nvidia-smi`，driver 595.84）。
> 「8 卡 vs 1 卡」這個核心落差不受影響 —— 顯存 24→32 GB 換不到七張卡。
> 但本節任何以 24GB 為前提的**可行性**推論，在 5090 上都**未重量，標 UNVERIFIED**。

ULIP-2 用 open_clip **ViT-bigG-14**（~2.5B 參數）當 text/image backbone，
其論文 §3.3 明文 **"freeze it during pre-training"**。
（先前這裡多寫了一個 "the"，已更正 —— 這份文件的紀律不容許改寫當引文。）

實作上有個對不上的地方：`ULIP2_PointBERT_Colored` 只呼叫 `eval()`，**沒有**設
`requires_grad = False`，而 `main.py` 的 optimizer 用 `if not p.requires_grad: continue`
挑參數 —— 所以照公開碼跑，那 2.5B 參數會進 optimizer。
同檔的 **ULIP-1 五個 factory 都有明確凍結**，只有 ULIP-2 的沒有。

> 這是 **ULIP-2 程式與它自己論文的落差**，不是「官方設計不凍結」。
> 先前本文用這個落差論證「凍結是我們的偏離」，那個推論已撤回（見 U-34）。
Eq.(5) 的分母 `Σ_{A' ∈ B}` 是 **in-batch negatives** —— 對比學習的檢索品質
高度依賴 batch size，單卡直接跑會把 batch 壓到遠低於論文設定（**論文根本沒公佈 batch size**，見 U-22；
「遠低於」是相對於上游 8×512 的 all-gather，見 D-10）。⚠ 32GB 上能容納多大的 batch **從未量測，UNVERIFIED**。

**這是整個復現最大的風險。**

**影響設計 → 架構決策 D2（見下）。**

---

## F6. 磁碟：`/` 吃緊，資料改放 `$METAFIND_DATA` —— 但那是**共用碟**，風險未完全解除

> 🔴 **[過期 2026-08-30 標記，內容保留不刪] 這一節分析的磁碟已經不是現在這台機器的磁碟。**
> 當時的前提：`/` 是 3.6 T 的 LVM、用到 98%，資料放在 `/dev/sda1` 的 `$METAFIND_DATA`。
> **現況實測（`df -h`、`readlink -f data`）**：
> `/` 是 **`/dev/nvme0n1p2`，937 G，用到 66%**；`/dev/sda1` 是 **3.6 T、只用 14%** 的 SMR 碟；
> 而 repo 的 `data` symlink 指向 **`/home/kyzen/metafind_data`，也就是那顆 NVMe**，不再是 `sda1`。
> ⚠ 連帶：`CLAUDE.md` §9 寫的 `data -> /home/kyzen/data/MetaFind` **也已失效**（該路徑不存在）；
> 那份檔不在本輪可改範圍，已上呈 MASTER。
> 本節的吞吐量、容量與瓶頸結論**全部未在新硬體上重量，標 UNVERIFIED**。

**證據**

```
/dev/mapper/vgubuntu-root  3.6T  3.5T  108G  98% /            ← home 所在，很緊
/dev/sda1                  3.6T  2.7T  779G  78% $METAFIND_DATA   ← 資料放這裡
README.md:37: Skip downloading the full rendered_images (~1TB) if not needed.
README.md:36: A 420GB subset is available ... under the `only_rgb_depth_images` folder
data/objaverse-lvis  →  不存在，需下載
```

粗估我們自己需要的量：

| 項目 | 估算 |
|---|---|
| 項目 | 當初估計 | **實測** | 差異 |
|---|---|---|---|
| 模型權重（Qwen2.5-VL 16G、Qwen2.5 15G、bigG 9.5G、CLIP-B/32 0.6G） | ~15 GB | **42 GB** | 估計時只算了 bigG |
| 46,052 朵點雲 | 11 GB | **5.6 GB** | npz 壓縮 |
| 46,052 × 11 張渲染圖 | 51 GB | **7.3 GB** | 每張估 100KB，實際 ~14KB |
| ProcTHOR-10K | ~10 GB | **395 MB** | 它是 JSONL，不是資產本身 |
| 場景圖 ＋ ProcTHOR 資產模態 | — | **445 MB** | 當初沒有這兩項 |
| 標註 ＋ 索引 | — | **170 MB** | |
| conda env ＋ AI2-THOR build | — | **9.3 GB** | 當初沒算 |
| **合計（不含 GLB）** | **~88 GB** | **≈ 64 GB** | |
| Objaverse-LVIS GLB 原檔 | ~216 GB | **328 GB** | **少報 52%** |
| **總計** | **~304 GB** | **392 GB** | |

**估計錯得最離譜的是渲染圖**：每張估 100 KB、實際約 14 KB，所以 51 GB 變成 7.3 GB。
反方向錯得最離譜的是 GLB 與模型權重。**兩邊互相抵銷，總量看起來還好，
但沒有一項是準的** —— 這種表只要沒重新量過就會一直錯下去。

> **2026-08-16 更新**：`$METAFIND_DATA` 是**共用磁碟**，不是我們獨佔。
> 3.6 TB 裡我們佔 392 GB，其餘是 `abo_dataset` 1.7 TB、`cheng` 804 GB、
> `yucheng` 163 GB 等他人資料，目前全碟剩 374 GB。
> **先前寫「779GB 可用、90GB 綽綽有餘」已不成立** ——
> 可用空間會隨別人的用量變動，規劃時不能當成固定值。
>
> **後果：磁碟風險解除。**
> 這是設計文件裡少數「因為新事實而放寬」的地方 —— 值得標明，因為
> **放寬約束和放寬檢查是兩回事**：前者是事實變了，後者是為了讓紅燈變綠燈。
>
> （本段原本寫「原本的 R-01 風險解除」。**那個 `R-01` 是本文件自己的編號**，
> 與登記表的 `R-01`（I-Design 未驗證）**同號不同義**，已移除。）

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

> **⚠️ 本節的結論已被推翻，保留是為了記錄推翻的理由。**
>
> 原本寫「pilot 之後決定走 (a) 自行標註 或 (b) 沿用 ULIP-2 captions」。
> **fallback 到 ULIP-2 captions 這個分支已取消。** 那些 caption 沒有
> `placement_constraints`，而它正是 layout-aware 檢索成立的訊號 ——
> 換掉等於做另一個實驗，而不是便宜地做同一個實驗。
> 若真的要用，整份結果必須標 `DEGRADED`，不得當成主線復現。
>
> 另外，金錢成本本身也已不適用：**GPT-4o 已由 Qwen2.5-VL 取代（偏離 D-2）**，
> 本節的美元估算只剩歷史意義。

---

## F8. EGNN 的 `edges_in_d` 就是語意邊的插槽，但**維度會壓垮幾何訊號**

**證據**：[`egnn_clean.py:22-26`](/home/kyzen/upstream/egnn/models/egnn_clean/egnn_clean.py)

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

> 🔴 **[過期 2026-08-30 標記，數字保留不刪]**
> **上面這兩個數字現在不可重現，「約 45 倍」這個量級不得再被引用。**
>
> 1. **重測對不上。** 2026-08-28 依同一條路徑重跑 `two_mlp` 得到 **44.63 / 1.89**，
>    不是 50.9 / 1.14。原數字來自某個沒有留下的更早版本或臨時執行。
>    出處：`docs/METAFIND_NOTEBOOK.md`（見該檔 §9.13 一帶與 `:1298`）。
> 2. **產生它的測試有缺陷。** `tests/test_essgnn.py` 的
>    `for seed in range(6):` 迴圈裡 **`seed` 從頭到尾沒有被使用**；
>    `geometric_sensitivity()` 沒有 seed 參數，內部的 `make_scene` 每次都
>    `torch.manual_seed(0)`。**六次呼叫回傳位元相同的值** ——
>    它宣稱量了六個種子，實際上把同一件事量了六次。
> 3. **範圍不對。** 這組數字量的是 `two_mlp` 這一支；
>    我們定案的架構家族是 **`appendix_shared_msg`**
>    （`data/outputs/essgnn_arch_protocol.json`，hidden 128 / n_layers 4）。
>    該配置下六 seed（**真的有傳 seed**）的比值中位數 **0.3066**、
>    範圍 `[0.1349, 0.8591]`、**ratio > 1 的次數 0/6** ——
>    **方向一致仍是壓制**，但幅度與 `two_mlp` 不同（同配置 `two_mlp` 中位數 0.1037）。
>    ⚠ 不得寫成「壓制 3.3 倍」；離散度尚未歸因。
>
> **結論不變的部分**：「加寬語意邊會壓抑幾何訊號」這個**方向**在兩個架構家族都成立。
> **失效的部分**：`50.9 / 1.14` 這兩個值，與由它們算出的「45 倍」。
> 修法（重測兩個 F8 測試的 docstring 數字、給 `geometric_sensitivity()` 加 seed 並往下傳）
> 屬 ESSGNN 區塊，不在本文件。

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
所以 `e` 的寬度（1280 / 768 / 512）是輸入而非常數。
（登記在 `01_GRAPH_SPEC.md` §15。）

---

## F10. §2.5 的 `f_x : ℝ^(2d+1+e) → ℝ³` 與 Appendix C 的證明互相矛盾

**證據**

- §2.5 Eq.(3)：`x_i^{l+1} = x_i^l + Σ_j (x_i^l − x_j^l) · f_x(...)`，並定義 `f_x : ℝ^(2d+1+e) → ℝ³`
- Appendix C Eq.(13) 的推導：
  `Σ (Q x_i + g − Q x_j − g) · φ_x(m_ij) = Q Σ (x_i − x_j) · φ_x(m_ij)`
  把 `Q` 提到求和外面，**只有在 `φ_x` 是純量時才成立**。
  若 `f_x` 輸出 ℝ³ 且為逐元素相乘，旋轉不可交換，等變性當場失效。
- 官方 EGNN [`egnn_clean.py:33`](/home/kyzen/upstream/egnn/models/egnn_clean/egnn_clean.py)：
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

論文只寫 "After $L$ layers"，**沒有給 L 的值**（`L=4` 是先前草稿自己填的）。
以常見的 `L=4` 為例，四分之一的座標參數從未被訓練；比例是 `1/L`。

> **⚠️ 這個結論只在「每層獨立權重」下成立，而那是我們的實作選擇。**
> 論文寫的是 `f_h(...; θ_h)` 與 `f_x(...; θ_x)`，**`θ` 上沒有層索引**，
> 從未說明 L 層是各自有一組權重、還是共用同一組（登記為 **U-31**）。
> 若是共用，最後一次 `x^L` 更新確實沒有下游消費者，
> **但同一組 `f_x` 參數仍會從前 L−1 次使用收到梯度**，
> 「最後一層的座標頭永遠沒被訓練」就不成立。

**影響設計**：在**目前採用的獨立層實作**下，這是架構的性質，不是缺陷。**不修**，
改以測試釘住這個確切模式 —— 若日後 readout 改動，測試會抓到而不是默默吸收。

---

## F12. ProcTHOR-10K 實際只有 1,467 個 unique asset，論文說「3,000+」

**實測**（2026-08-15；當時的抓取腳本已隨資料重置刪除，數字記錄於此）

| | latest revision | 舊版 `ab3cacd`（pre-AI2THOR-5.0） |
|---|---|---|
| houses | 12,000（10k train + 1k val + 1k test） | 12,000 |
| **unique assetId** | **1,467** | **1,467** |
| unique 物件類型 | 93 | 93 |
| objects/house | 69.3（最多 245） | 69.0 |

把門、窗一起算進去也只有 1,467（`objects` 996 + `doors` 26 + `windows` 14，
以 train 單獨計為 1,036）。**兩個 revision 完全一樣**，所以版本差異不是解釋。

**最可能的解釋**：論文的
*"constructed from a curated collection of more than 3,000 unique assets"*
指的是 ProcTHOR **產生器可取用的資產庫**，而不是這 12,000 間房實際實例化的集合。
無法在不取得完整 AI2-THOR 資產庫的情況下驗證，記為未定。
（登記在 `01_GRAPH_SPEC.md` §15。）

**影響設計**

- 若場景級檢索的 gallery 是 ProcTHOR 資產，實際只有 1,467 個而非 3,000+。
  但 IDesign 的 `retrieve.py` 是對 **Objaverse** 檢索，
  所以主線不受影響；此數字只在報告中如實記錄。
- **69 objects/house（最多 245）遠高於我原本的假設。** SG4 的
  `N ≤ 25` 上限是針對 Algorithm 1 要放置的查詢數，不是房間總物件數，
  但場景圖的節點數會到 245 → ESSGNN 的 batch 記憶體要照這個規劃，
  而不是照「約 20 個節點」。

---

## F13. 渲染前的正規化會摧毀絕對尺度 —— **在本實作中** size 只能是類別先驗

**證據**（2026-08-15 實測；當時的渲染模組已隨資料重置刪除，結論保留）

渲染前必須把 mesh 置中並縮放到單位球，否則以公釐建模和以公尺建模的同一個物件
會產生完全不同的圖，image tower 學到的是**建模單位**而不是形狀。
但這個正規化的代價是：

| 物件 | 原始 extents (m) | 正規化後 |
|---|---|---|
| 桌子 | 0.5 × 1.8 × 0.4 | 0.523 × 1.884 × 0.419 |
| 杯子 | 0.08 × 0.1 × 0.08 | 1.06 × 1.325 × 1.06 |

**長寬比保留，絕對尺度歸零。**

論文 §2.3 的標註包含 "size dimensions"，而**在我們的管線裡**標註模型只看得到
scale-normalised 的渲染圖 —— 它**不可能量到**真實尺寸，只能依「這是什麼東西」給出類別先驗。

> **⚠️ 這是本實作的限制，不是論文的性質。**
> 原本本節寫「GPT-4o 當初產生的那些尺寸數字，性質上也是同一回事」——
> **這句話推不出來**：論文從未說它給 GPT-4o 看的是 scale-normalised 渲染圖，
> 也沒說標註流程裡沒有一併提供尺寸 metadata。
> 尺度歸零是**我們加上 unit-sphere 正規化**的後果，只能描述我們自己。

**影響設計**

- 標註 prompt 要明說渲染圖是 scale-normalised，要模型依物件類別估計，
  而不是假裝在量測。
  **這條尚未實作** —— `n05_annotate` 還沒有程式，所以 `metafind/data/` 裡沒有
  prompt，也沒有 `test_prompt_explains_that_renders_are_scale_normalised`。
  先前這裡直接用反引號寫了那個測試名，讀起來像它已經存在。
  已登記的檢查是 `L1-RENDER-SCALE-INVARIANT`。
- **原始 bounding box 記進 render sidecar**（`extents_m`、`volume_m3`）。
  我們手上本來就有，記下來幾乎零成本，而且讓模型的估計**可稽核**
  —— 之後可以直接比對「模型猜的尺寸」與「mesh 真實尺寸」的相關性。
- 尺寸欄位不得被當成量測值使用。

---

## F14. 其他實作細節（原編號 F9，與 F8→F13 的順序衝突，改編號）

| 項目 | 事實 | 影響 |
|---|---|---|
| `unsorted_segment_sum` 用 `scatter_add_`（`egnn_clean.py:149-154`） | GPU atomics → **非確定性** | 列入 `nondeterminism_sources`；等變性測試容差不能設成 exact |
| `coords_agg` 預設 `'mean'`（`egnn_clean.py:11` 的簽章預設；`:17` 是賦值，`:75-80` 是分支） | MetaFind Eq.(3) 與 Appendix C Eq.(13) 都是**裸 sum**（見 **F15**：EGNN 論文 Eq.(4) 另有 `C = 1/(M−1)`，MetaFind 兩處都拿掉了） | 設定必須明寫 `coords_agg='sum'`，列 L1 |
| `E_GCL.forward` 回傳的第三個值是**原封不動的** `edge_attr`（`egnn_clean.py:103`） | 語意邊不會被逐層更新 | 與論文一致（`e_ij` 為常量），寫進 postcondition |
| `get_edges()` 建**全連接**圖（`egnn_clean.py:167-176`） | 我們要的是稀疏的 physical+semantic 邊 | 必須自寫 `edge_index` 建構；不可複用 |
| `Objaverse_Lvis_Colored.__getitem__` 只回傳 `(pc, label, name)` | **沒有圖片、沒有 caption** | tri-modal 資料要自己組，不能複用這個 Dataset |
| `Objaverse_Lvis_Colored` 需要 `data/objaverse-lvis/lvis.json` 等 | 當時不存在；**2026-08-15 已由 `n02_download` 取得**（`lvis.json` 與 `objaverse_lvis_metadata.json` 皆在 `datasets/objaverse-lvis/`） | 已是 `n02_download` 的明確產出 |

---

## F21. U-02 第一次真的量了 —— 幾何吻合，而顏色缺口是**我們的 bug**

拿官方 shard 裡 6 個我們也有 GLB 的資產，逐一比對。

**幾何**（`pc_norm` 後的 Chamfer，各取 2,000 點）：

| | 值 |
|---|---|
| 同一資產配對距離 中位數 | **0.00318** |
| 不同資產基線 中位數 | **0.05880** |

**低 18 倍。** 取樣程序與 ULIP 產生的雲描述的是同一個物體。

**顏色**：一開始 6 個裡 2 個對不上，而**兩個都是我們標記為 `fallback_grey` 的**：

```
1dc0fe17c77e   我們 0.400   官方 1.000
93b4b53d0985   我們 0.616   官方 0.704
```

追下去發現 `main_color = [102 102 102 255]`，而 **102/255 恰好等於 0.4** ——
那是 **trimesh 自己的預設灰**，不是資產的顏色。
**glTF 2.0 規定 `pbrMetallicRoughness.baseColorFactor` 缺省時是 `[1,1,1,1]`**，
所以「有材質、沒貼圖、沒 factor」的物件是**白的**，不是未知。
我把規格定義的白，塗成了 ULIP 用於「整個資料集沒有顏色通道」的灰。

修正後六個全部吻合：

```
平均絕對差 0.0021    最大 0.007
```

而 smoke corpus 的 `fallback_grey` 從 9/60 **降到 0/60**，
`coloured_point_fraction` 60/60 都是 1.000。

**這一格的教訓**：`fallback_grey` 這個計數先前讀起來像「這些資產本來就沒顏色」，
實際上是「這些資產的顏色我們沒讀出來」。
**兩者在 sidecar 裡長得一模一樣，只有跟上游成品比對才分得出來。**

（附帶：`fallback_grey` 現在**實際上到不了** —— trimesh 一定會補一個
`SimpleMaterial`，所以 `gltf_default` 會先命中。它保留為最終保險，
但不能再被描述成活躍的 fallback。）

---

## F22. 前處理是**磁碟綁死**，不是 CPU 也不是 GPU

> 🔴 **[過期 2026-08-30 標記，內容保留不刪] 這一節分析的磁碟已經不是現在這台機器的磁碟。**
> 當時的前提：`/` 是 3.6 T 的 LVM、用到 98%，資料放在 `/dev/sda1` 的 `$METAFIND_DATA`。
> **現況實測（`df -h`、`readlink -f data`）**：
> `/` 是 **`/dev/nvme0n1p2`，937 G，用到 66%**；`/dev/sda1` 是 **3.6 T、只用 14%** 的 SMR 碟；
> 而 repo 的 `data` symlink 指向 **`/home/kyzen/metafind_data`，也就是那顆 NVMe**，不再是 `sda1`。
> ⚠ 連帶：`CLAUDE.md` §9 寫的 `data -> /home/kyzen/data/MetaFind` **也已失效**（該路徑不存在）；
> 那份檔不在本輪可改範圍，已上呈 MASTER。
> 本節的吞吐量、容量與瓶頸結論**全部未在新硬體上重量，標 UNVERIFIED**。

**實測 2026-08-15**：

| | |
|---|---|
| `$METAFIND_DATA` 實際讀取吞吐 | **74 MB/s** |
| 46,052 個 GLB 總量 | **351 GB** |
| `n03` 單獨的讀取需求 | **70 MB/s**（平均 7.6 MB × 553 資產/分） |
| RAM | 62 GB（可用 37 GB，**快取不下 351 GB**） |

所以：

- **單一節點掃過一遍 GLB 的硬下限是 `351 GB ÷ 74 MB/s ≈ 79 分鐘`**，加多少 worker 都無效。
  `n03` 用 8 個 worker 量到 553/min，那正好就是讀取上限。
- **`n03` 與 `n04` 不能同時跑。** 試過一次：`n03` 一個就把頻寬吃滿，`n04`
  卡在 **`D` 狀態、wchan `folio_wait_bit_common`**（不可中斷的磁碟等待），
  **連啟動掃描都沒過、log 一行都沒印**。
- 循序跑等於**把 351 GB 讀兩次**（共 702 GB，約 110 分鐘）。

**「讀一次、同時產出兩種產物」可以把 I/O 砍半**（約 79 分鐘），
**刻意不做**：那會把 graph 為了不同失敗類別（`n03` 的壞網格 vs `n04` 的壞材質）
與不同 cacheability 而分開的兩個節點合成一個執行單元。
省 30 分鐘不值得動搖已經過四輪審查的節點邊界。

**這一格的教訓**：GPU 利用率 0%、CPU 只有 196%（20 核），
看起來像「機器沒被用滿、可以加更多平行度」——**而真正的天花板從頭到尾是磁碟**。
我為 `n04` 停擺猜了兩個原因（EGL context churn、`is_complete` 重算雜湊），
兩個都是真缺陷、也都修了，**但都不是主因**。
**第三次才去量，而量一次就結束了爭論。**

---

## F23. 前四個節點的全量實測結果

> 🔴 **[過期 2026-08-30 標記，數字保留不刪] `n04` 這一列是 pyrender 世代的產物，已被 Blender 重跑取代。**
> 2026-08-23 起 n04 改用 OpenShape 的 Blender／Cycles 腳本（12 視角／512px／perspective／
> transparent RGBA，`DL-024 A1/A2/A3`，USER_APPROVED），整個語料重渲染過。
> **現行實數（`data/outputs/logs/renders_index.jsonl`）：46,024 筆**，不是 45,955。
> 下游語料再經 n05 admission 收到 **45,692**（= 46,024 − 311 − 21，見 `splits.json` 的
> `admitted_total`）。本節其餘關於「11 張全空白」「部分空白分佈」的細節同樣是 pyrender 世代的，
> 保留為當時的證據，**不得當作現行渲染語料的統計**。
> `n02` 與 `n03` 兩列未受影響（它們讀的是 manifest 的 46,052）。

| 節點 | 完成 | 隔離 | 大小 |
|---|---|---|---|
| `n02_download` | 46,052 / 46,052 | 0 | 351 GB |
| `n03_sample_pointclouds` | **46,052 / 46,052** | **0** | 5.6 GB |
| `n04_render_views` | ~~45,955 / 46,052~~ **（過期，pyrender 世代；現行 46,024）** | 97（**0.21%**） | 7.3 GB |

G3 的隔離門檻是 2%，兩個節點都遠低於。

**`n04` 的隔離原因**（143 筆紀錄、99 個相異資產，數字不同是因為批次重試）：

```
104  every view is blank -- the asset never entered frame
 14  Eigenvalues did not converge
 17  only N distinct views of 11（相機沒有在移動）
```

**「11 張全空白」是主要原因，而那正是我加的偵測器攔下來的** —— 那些資產在單位球正規化後仍然不進畫面，通常是幾何退化到一個點或一條線。

**部分空白是真實現象，不是缺陷**：

```
 0 張空白   45,773 個資產（99.6%）
 1 張空白      135 個
2–9 張空白      47 個
```

**182 個資產有部分空白視圖** —— 薄片狀物件從側面看幾乎沒有東西，
與 21 個點雲合法地呈平面（`L1-PC-NONDEGENERATE`）是同一批幾何。
它們被接受，但**數量必須記錄**：一個開始把**普通**資產也算成空白的迴歸
只會移動這個分佈、不會動到別的 —— 因為每一項其他渲染檢查在
一張形狀正確的空白圖上都會通過。已新增 `L1-RENDER-PARTIAL-BLANK`。

---

## F15. MetaFind 拿掉了 EGNN 的正規化常數 `C = 1/(M−1)`

**這不是 UNKNOWN，是 MetaFind 與 EGNN 之間一個明確的差異。**

EGNN Eq. (4)：

```
x_i^{l+1} = x_i^l + C Σ_{j≠i} (x_i^l − x_j^l) φ_x(m_ij)
```

論文明講 *"C is chosen to be 1/(M − 1), which normalizes the sum"*。

MetaFind **兩處**都沒有 `C` —— §2.5 Eq. (3) 與 Appendix C Eq. (13) 都是純粹的 `Σ`。

**主線照 MetaFind 走**（Level 0 勝過 Level 1，而且這裡 MetaFind 不含糊，是 EGNN 不同）。
但要記下後果：**座標更新的量級會隨鄰居數成長**。EGNN 的 QM9 分子節點數相近，
ProcTHOR 房間的物件數不是 —— 一個 4 件家具的房間與一個 30 件的房間，
在同一組權重下位移尺度差一個量級。`C` 是純量，等變性兩邊都成立，
所以 Appendix C 的證明不受影響；受影響的是**訓練穩定性**，不是正確性。

`coords_agg = "sum"` 因此不是「我們的一種讀法」，而是**照著 MetaFind 抄**。

---

## F16. EGNN 在自己的圖級任務裡**不更新座標**

§5.3（QM9）：*"Since positions are static, we do not update coordinates x_i during
message passing, making our model functionally E(n) invariant."*

那正是 EGNN 唯一一個「把節點聚合成一個圖級向量」的實驗，與 `e_layout = Pooling({h_i^L})`
的用途相同 —— 而它**關掉了** Eq. (4)。

MetaFind 沒有關：Eq. (3) 與 Appendix C Eq. (13) 都保留座標更新，
而 `e_layout` 只讀 `h`。這正是 **F11**（最後一層 `f_x` 收不到梯度）的來源，
現在多了一個佐證：**EGNN 自己遇到同樣的情形時，選擇不更新座標**。

**不據此改主線** —— 依賴方的做法不能覆蓋 MetaFind 的明文。記錄下來，
因為它讓 F11 從「我們發現的怪事」變成「上游遇過並處理過的已知情形」。

---

## F17. ULIP-2 自己就把「物件級 → 場景級」列為未解問題

§6.2 Limitations：*"ULIP-2's pre-training primarily utilizes object-level 3D shape
datasets, which differ in distribution and complexity from scene-level 3D data.
Applying ULIP-2 to scene-level data represents a compelling avenue for future
research."*

MetaFind 做的就是這件事。這不改變任何實作決定，但它是**報告裡該講的一句話**：
MetaFind 的 backbone 論文明文說自己沒驗過場景級資料。

順帶兩點對得上的：

- **點雲 10k xyzrgb 是對的。** ULIP-2 Appendix A.1 的 `10k xyzrgb` = 50.6／79.1，
  正是摘要的 SOTA 數字（`8k xyz` 是 48.9）。我們的 `G2_pc_sanity` 驗 `(10000, 6)`、
  下載的 checkpoint 是 `ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g`，**一致**。
  先前這只是從 checkpoint 檔名推的，現在有論文。
- **視圖數不衝突。** ULIP-2 對 Objaverse 渲染 **12** 張（§4.1），MetaFind 是 **11** 張
  （摘要與 §2.3）。U-14 問的是「11 張怎麼變成一個 `e_image`」，
  跟 ULIP-2 的 12 無關 —— 不要把它拿來當證據。

---

## F18. I-Design 論文自己就把「放不下就失敗」列為已知限制 —— R-01 的解讀要改

R-01 記的是「Qwen2.5-7B 跑 5 次、0 個完成」，並註明**沒有基準所以不能斷定是缺陷**。
那個保留是對的。讀完 I-Design 原論文後，多了三件事：

**其一，失敗是論文寫明的已知失敗模式。** §5.2 Limitations 第一項：

> **Placement Termination Problems:** The pipeline may fail to find a solution for
> object placements when handling **many objects in a relatively small scene**.
> Spatial conflicts may persist, or there may not be enough space for furniture
> placements.

**而我們的 smoke 設定正好落在那個區間。** 先前用的是 `n=15` 放進 `[4.0, 4.0, 2.5]`
—— 16 m² 放 15 件，比論文 Table 1 任何一個臥室場景都密（臥室平均 12.7 件，
客廳 23.6 件但房間開到 `[6.0, 8.0, 3.0]` = 48 m²）。

這仍然**不構成完成率基準**，但它把結論從「跑不出來 ⇒ 有東西壞了」
改成「跑不出來**可能就是論文描述的行為**，而且我們挑了最容易觸發它的設定」。

**其二，prompt 與房間尺寸論文其實有給，是我編的。** Table 4 列了 20 條 minimal
prompt 與尺寸，Table 5 列了 40 條 elaborate prompt。而 `idesign_generate.py`
用的是「A creative vibrant livingroom」「An aged archive room」—— **我編的**，
而且同一個檔案上面還寫著「不在這裡編造」。已換成 Table 4 的原文。

**`n`（物件數）論文沒給。** Table 1 的 NObj 是**產出**不是輸入。我們填的值仍是我們的。

**其三，`JSON mode` 沒有被繼承。** 補充材料 §7：

> All agents utilize GPT-4's **"JSON mode"** to restrict outputs exclusively to
> valid JSON, which significantly reduces token consumption.

我們的 vLLM 沒有開任何 guided decoding，所以 Qwen **在結構上可能吐出不合法的 JSON，
而 GPT-4 在那個模式下不可能**。那會落進 Engineer 的 schema 驗證重試迴圈，
正是我們失敗的路徑之一。**這是 D-5 的一部分，讀論文前沒有記下來。**

對上的一件事：**temperature 0.7 / top_p 1.0 是論文 §4.1 明定的**，
而 I-Design 自己的 `agents.py` 就設了這兩個值，所以我們是**繼承**而不是假設。

---

## F19. I-Design 自己也做檢索，而那正是 MetaFind 換掉的那一段

§3.4：I-Design 用 **CLIP text encoder** 產生描述嵌入，靠 **OpenShape** 與 CLIP 的對齊，
從 **Objaverse** 取回最接近的資產。

這解釋了 R-01 裡那個「MinkowskiEngine／dgl 其實不需要」的觀察 —— 它們只被 `retrieve.py`
用到，而 `retrieve.py` 實作的就是 §3.4。**MetaFind 取代的正是這一步**，
所以我們呼叫到 `to_json()` 產出場景圖就停，不走 I-Design 的檢索。

記下來是因為它讓那個安裝決定從「試出來不需要」變成**有論文根據**：
我們要的是 I-Design 的 §3.2／§3.3（多代理人 → 場景圖 → backtracking 佈局），
不要 §3.4。

---

## F20. ULIP-2 的點雲 RGB 在 **[0, 1]** —— 已由官方檔案**實測**

MetaFind 沒說，**ULIP-2 的論文也沒說**。checkpoint 名字只寫 `10k-xyzrgb`。

**2026-08-15 實測，不再是推論。** 下載官方 `ULIP-2/objaverse_lvis/000-009.tar.gz`
（shard 內 4,999 個資產），讀其中 120 個：

```
rgb dtype = float16    全域 min = 0.0000    全域 max = 1.0000
超出 [0,1] 的資產：0 / 120
```

先前這裡寫的是「決定性證據在 ULIP 程式」，**那個推論過頭了**：
`0.4` fallback 那兩行在 **ModelNet** 路徑，而 `Objaverse_Lvis_Colored` 是直接
concat released `.npy` 的 rgb、**不除不夾**，繼承的是檔案本身的尺度。
結論碰巧是對的，但當時的證據撐不起「decisive」這個詞。原始論證保留於下：

資料集沒有顏色時它填

```python
rgb_data = np.ones_like(point_set) * 0.4      # dataset_3d.py:292, 297
```

—— 一個中灰。若尺度是 0–255，那個替代值會是 ~102 而不是 0.4。

**這一次用 Level 2（依賴方實作）作為答案是正當的**，與第 155 項那個錯誤不同：
當時我拿 ULIP-2 的**程式**去論證它的**設計**（該不該凍結 CLIP），那是設計問題；
這裡問的是「**這個 checkpoint 的輸入介面吃什麼尺度**」——
那是關於依賴方**介面**的事實，程式就是最終權威，論文反而答不了。

**弄錯不會有任何錯誤訊息**：xyz 經 `pc_norm` 後落在 [-1, 1]，
若 rgb 是 0–255，顏色通道會大兩個數量級地主宰輸入，
症狀只會是檢索數字偏低，而我們會跑去別處找原因。
每個 sidecar 都記 `rgb_scale: "unit"`。

**另外兩件同樣抄自 ULIP loader 的事**：

- **`pc_norm` 只作用在 xyz**（`dataset_3d.py:496`），rgb 是**之後**才 concat。
  六欄一起正規化會讓顏色被物體實際尺寸縮放——形狀一樣，張量不同。
- **不做 FPS**。`Objaverse_Lvis_Colored` 讀的是已經恰好 10,000 點的雲，
  再用打亂的 permutation 取；FPS 出現在 ShapeNet 那條路徑。

---

## F24. ProcTHOR 資產**可以**被單獨渲染 —— U-08b 的前提是錯的

**這條推翻了我自己寫過三次的一句話。** 我一直說「ProcTHOR 只提供座標與 metadata，
沒有渲染圖也沒有點雲」。對 JSONL 而言那是對的，但 **ProcTHOR 不等於它的 JSONL** ——
房子是給 AI2-THOR 載入的，而 AI2-THOR 是一個實際會渲染的 Unity 應用。

**實測（2026-08-16，ai2thor 5.0.0，CloudRendering headless，與 n05 共用同一張 4090）**：

| n03／n04 給 Objaverse 的 | AI2-THOR 給 ProcTHOR 的 | |
|---|---|---|
| 11 個環繞視角 | 11 個 third-party camera | ✅ |
| 正交投影 | `orthographic: true` + `orthographicSize` | ✅ |
| 224×224 | 任意解析度 | ✅ |
| 白底、畫面中只有該物件 | 需要三步：`doors`／`windows`／`walls`／`ceilings` 清空，物件抬到 `y=40`，`skyboxColor="white"` | ✅ |
| 從完整 mesh 均勻取樣 10,000 點 | 只能從 11 張深度圖反投影 | ⚠️ **這一項不等價** |

`assetId` 直接出現在 `event.metadata["objects"]` 裡，所以 ProcTHOR 的 `Fridge_19`
與 AI2-THOR 的實際 3D 資產是同一個東西、對得起來。深度圖 `(11, 224, 224)` 全 finite。

**唯一真正的缺口是點雲。** n03 從完整 mesh 表面均勻取樣，拿得到被遮蔽的面；
11 個環繞視角的深度反投影只拿得到**可見外殼**。對多數物件那涵蓋大部分表面，
但**不是同一種取樣**，不能宣稱等價。

### 過程中最值得記的一件事

第一版隔離渲染，我的數值檢查說「11 張全相異、沒有空白視角、背景比例 19–45%」——
**全部通過**。實際打開圖片看：畫面裡有一根 metadata 沒列出的灰色柱狀物、
橘色地板鋪滿背景、而且在三張裡把冰箱**切成兩半**。

**數字說通過，圖片說沒有。** 這與本專案 n03 顏色 bug、n04 空白視角是同一類錯誤，
而這次是在一個我原本打算只花二十分鐘查證的探針上發生的。
凡是產生影像的節點，驗收一定要包含「真的把圖打開看」。

## F25. ProcTHOR 的資產數有**四個互不相同的數字** —— 一個都不能寫死
MetaFind 2.3 轉述 ProcTHOR「over 10,000 generated houses constructed from a
curated collection of more than **3,000 unique assets**」。查下去之後，
這個數字四邊都對不上：

| 來源 | 資產數 | 類別數 | 怎麼得到的 |
|---|---|---|---|
| MetaFind §2.3 | **3,000+** | — | 論文轉述 |
| ProcTHOR 原論文 | **1,633** | **108** | 原文「1633 household assets across 108 categories」 |
| 本機 build 的資產庫 | **1,934** | — | `controller.step(action="GetAssetDatabase")` |
| 12,000 間房子實際用到 | **1,467** | **93** | 掃 train 10k + val 1k + test 1k 全部 `assetId` |

**注意**：`GetAssetDatabase` 屬於 **procedural API**，在 `FloorPlan1` 這種手工做的 iTHOR 場景上會回傳空的。n01 的驗證第一版用了 `FloorPlan1` 因而失敗 —— 那次失敗同時也證實了 1,934 是在正確的情境下量到的。

**可重現性**：以上兩個實測值綁定這個 build ——

```
ai2thor            5.0.0
CloudRendering     thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917
procthor-10k       allenai/procthor-10k @ 439193522244720b86d8c81cde2e51e3a4d150cf
                   （"update procthor-10k"；train 10,000 / val 1,000 / test 1,000）
```

`prior` 下載的其實是一個 **git repo**，所以資料集有精確的 commit 可引用，
不必只寫「10k/1k/1k」。它的 `README.md` 與 `main.py` 都**沒有提到資產數**，
`main.py` 只是解壓那三個 `.jsonl.gz`。

順帶一提，我先前拿來對照的舊 revision `ab3cacd0` 是 `4391935` 的**前兩個 commit**，
而 `4391935` 的訊息正是「update procthor-10k」——**資料確實被改過**。
但兩個版本的資產集合完全相同（1,467／93），所以那次更新沒有動到資產。

### 排除掉的三個解釋

**(a) 我們的資料來源不對** —— 排除。整份刪掉、清空 `~/.prior`、
完全照官方 tutorial 的 `prior.load_dataset("procthor-10k")` 重下一次，
**三個 split 的 sha256 與刪除前位元組相同**：

```
train  8f9313a75fd95ade…    val  2a4b9abeefd2f43b…    test  513b88f7eaa61a1b…
```

**(b) 版本差異** —— 排除。`prior` 提示有新舊兩個 revision，
舊版 `ab3cacd0fc17754d4c080a3fd50b18395fae8647` 也載下來比過：
**12,000 間房、1,467 個 assetId、93 個 category，與新版完全相同**。

**(c) 房子只用了資產庫的一部分** —— 排除。資產庫本身
（`GetAssetDatabase`）只有 1,934 個，也不到 3,000。

**所以「3,000+」不是我們這邊的問題，也不是版本或取樣造成的。**
ProcTHOR 原論文 PDF 過大無法直接抓取、摘要沒給資產數，
所以我無法確定它出自何處 —— 可能含材質變體、可能是不同的計數口徑、
也可能是轉述誤植。**沒有證據前不要替它編一個解釋。**
1,934 與 1,467 都比較接近 ProcTHOR 原論文的 1,633 而不是 3,000+。

**類別數 93 vs 108 的差距要小心解讀。** 我的 93 是從物件 id 前綴
（`CounterTop|2|0` → `CounterTop`）推出來的，那**不見得等同 ProcTHOR 官方的
category taxonomy**。這可能是量法差異而非矛盾，在確認 taxonomy 定義前不要當成發現。

### 這個數字**不是** Eq. 7 的分母 —— 我一度寫錯，這裡更正

我原本寫「1,467 就是 Eq. 7a/7b 分母的規模，也就是負樣本池」。**那是錯的。**
論文 Eq. 5 下方明寫「`B` denotes the gallery **batch**」，Eq. 7a/7b 的分母同樣是
`Σ_{e' ∈ B}` —— **in-batch negatives**。batch size 相同時，每一步看到的負樣本數
不會因為資產庫從 3,000 變成 1,467 而減半。

**而這個專案早就記錄過正確的事實**（見上文 F 系列：「Eq.(5) 的分母 `Σ_{A' ∈ B}`
是 in-batch negatives —— 對比學習的檢索品質高度依賴 batch size」）。
我不是遺漏了一個沒人查過的細節，是**寫出了與自己既有紀錄相反的話**，
而 1,949 項檢查沒有一項看得到這種矛盾 —— 它們驗結構，不驗論述。

**資產庫大小真正影響的是別的東西**，而且都無法定量推回 Table 2：

| | 影響 |
|---|---|
| 負樣本多樣性 | 1,467 個 identity 抽 batch，重複得比 3,000 個頻繁 |
| hard negative 出現率 | 資產越多，越容易出現視覺／語意相近的 chair A / chair B |
| 目標覆蓋率 | 每個 asset identity 被看到的次數 |
| sampling dynamics | epoch 長度、positive／negative 重複頻率 —— 而論文沒給 batch size、epoch 或 sampling recipe |

**所以正確的說法是**：在本復現採用的 U-08a protocol 下，1,467 是 Stage 2 實際
可觀測的**資產 identity universe**，不是 Eq. 7 每一步的負樣本數。
Table 2 仍可與論文對照，但**不能宣稱 protocol 完全相同** ——
影響程度標為 **MODERATE，方向未知**。

### 還有一步推論不能做

「MetaFind 說 3,000 → 所以作者的 Stage 2 gallery 就是 3,000」——**這一步不成立**。
論文只說了 ProcTHOR 上做 Stage 2、gallery encoder 凍結、loss 用 batch negatives；
**它從未說「把 ProcTHOR 全部 unique assets 建成一個 Stage 2 gallery」**。
那是 U-08a 的復現選擇，是我們的，不是它的。

## F26. AI2-THOR 的深度緩衝**看不見透明資產** —— 而 RGB 看得見

實測 n07b 全量時抓到的。**同一支探針、同一個場景、四個距離:**

| 資產 | 0.5 m | 1 m | 3 m | 6 m |
|---|---|---|---|---|
| `Alarm_Clock_1` | 深度 0.446 ✅ | 0.946 ✅ | 2.945 ✅ | 5.946 ✅ |
| `Bowl_11` | **全空** | **全空** | **全空** | **全空** |

而 RGB 兩個都正常(碗在 1 公尺時有 1,255 個非白像素)。
**所以這是資產的性質,不是相機參數的問題。**

前 250 個資產裡有 13 個這樣:`Bottle_1`、`CD_1`、**11 個 Bowl**
—— 全是玻璃／高光材質。Unity 的 depth prepass 只收不透明幾何,
透明物件不寫深度緩衝。而且**不是所有碗都這樣**(`Bowl_15` 就正常),
所以是特定材質,不是特定類別。

### 這改變了節點的失敗語意

第一版把它當成資產失敗直接隔離 —— **那會扔掉兩個好模態**。
正確的處理是「少了一個模態」:記錄照寫,帶著文字與 11 張視角，
`pointcloud_uri` 為明確的 `null` 並附上原因。

論文 §2.4 本來就說 query encoder 接受任意模態子集，
`stage2_protocol.query_pointcloud` 也已經是 `optional`。

**三種做法裡只有一種是對的:**

| 做法 | 結果 |
|---|---|
| 隔離整個資產 | 扔掉可用的文字與圖片 ❌ |
| 點雲填零 | 與真點雲無法區分 ❌ |
| **記錄明確的 null ＋ 原因** | 下游看得見、可統計、可決定要不要用 ✅ |

這與 `L1-SEMEDGE-NO-ZEROFILL` 是同一條原則:**降級必須是看得見的,不是安靜的。**

### 順帶修正的另一件事:bbox 檢查的參考值本身不精確

同一輪發現 31/163 個資產的 bbox 絕對誤差超過 5 公分,最差 0.916 m。
查下去**全是床** —— 而打開圖看,床的被子和枕頭確實垂在框外，
點雲抓到了它們,**是 AI2-THOR 回報的 `axisAlignedBoundingBox` 不含垂布**
（那是 collider 的框）。

所以檢查改用**比例**而非絕對值:

```
真實資料 192 個   0.94 – 1.53 倍（中位 1.005）
正交深度 bug 那次        69 倍
```

界線設 0.5–3.0。

### 然後比例界線也誤報了 —— 用鏡像的方式

全量跑到 1,050 個時,`Wall_Decor_Painting_6` 被隔離:

```
reported [0.797, 0.601, 0.007]   ← 一幅畫，7 公釐厚
measured [0.801, 0.599, 0.025]
```

x 和 y 吻合到 4 公釐以內,**只有厚度是 7 mm vs 25 mm** ——
絕對誤差 **18 公釐**,而比例是 3.6 倍。**分母趨近於零時,深度量化看起來像災難。**

所以兩種單一判準都會誤報,而且方向相反:

| 判準 | 誤報對象 | 原因 |
|---|---|---|
| 絕對誤差 | 床(31/163) | 大物件的正常外懸就有數十公分 |
| 比例 | 畫 | 極薄物件的分母趨近零 |

**改成兩個條件都要成立才算失敗**:比例超出 0.5–3.0 **而且**
絕對誤差超過最大邊長的 25%。三個真實案例:

| | 比例 | 絕對誤差 | 判定 |
|---|---|---|---|
| 畫 | 3.57 | 0.018 m | 通過 |
| 床 | 1.52 | 0.920 m | 通過 |
| **正交深度 bug** | **123** | **11.06 m** | **隔離** |

真 bug 兩個條件都超標一個數量級以上,兩個誤報各只超標一個。

**這輪的教訓**:一個判準只有在「正常變異」與「真故障」在**那個維度上**分得開時才有用。
床和畫證明了沒有單一維度能分開所有情況 —— 而我是分兩次、各被誤報打一次才學到的。

## F27. checkpoint 會不會塞爆磁碟，取決於一行 `torch.save` 怎麼寫

> 🔴 **[過期 2026-08-30 標記，內容保留不刪] 這一節分析的磁碟已經不是現在這台機器的磁碟。**
> 當時的前提：`/` 是 3.6 T 的 LVM、用到 98%，資料放在 `/dev/sda1` 的 `$METAFIND_DATA`。
> **現況實測（`df -h`、`readlink -f data`）**：
> `/` 是 **`/dev/nvme0n1p2`，937 G，用到 66%**；`/dev/sda1` 是 **3.6 T、只用 14%** 的 SMR 碟；
> 而 repo 的 `data` symlink 指向 **`/home/kyzen/metafind_data`，也就是那顆 NVMe**，不再是 `sda1`。
> ⚠ 連帶：`CLAUDE.md` §9 寫的 `data -> /home/kyzen/data/MetaFind` **也已失效**（該路徑不存在）；
> 那份檔不在本輪可改範圍，已上呈 MASTER。
> 本節的吞吐量、容量與瓶頸結論**全部未在新硬體上重量，標 UNVERIFIED**。

> 另註：本節的 `46,052` 是 manifest 數。實際落地的 embedding／gallery 索引分母是
> **45,692**（n05 admitted），估算值要照這個縮。

由「空間夠不夠」這個問題推導出來的，**在 n10 寫出來之前先關掉**。

剩下要產出的東西其實很小:

```
n06 embedding    46,052 × 32 KB = 1.53 GB
n11 gallery 索引 46,052 × 1280 × 4B = 0.24 GB
n05 剩餘標註     29 MB
n07b 剩餘        102 MB
                 合計 < 2 GB
```

**但 checkpoint 不小,而且差距是兩個數量級:**

| 存法 | 單個 | Table 3 十個變體 ＋ 主線 |
|---|---|---|
| 只存 `requires_grad=True` 的 | ~170 MB | **1.9 GB** |
| `model.state_dict()` 整包 | 10.2 GB | **112 GB** |

差在 **ViT-bigG-14 的 2.5B 參數** —— 它是**凍結的**，一個 step 都不會動，
而且可以從釘死的 OpenCLIP 權重完整重建。存進去等於把一份 10 GB 的唯讀資料
複製十一次。

`$METAFIND_DATA` 是**共用碟**，目前只剩 374 GB。112 GB 不是「有點浪費」，
是會影響到別人。

### 為什麼這值得在寫程式前就登記

`torch.save(model.state_dict())` 是**最自然的那一行**。它不會報錯、不會變慢、
訓練結果完全正確 —— 只是每個檔案大了 60 倍。等到跑完十個 ablation 才發現，
那時已經寫了 112 GB。

而且它還有第二層誤導:**那種 checkpoint 看起來像自足的成品**，
實際上仍然依賴同一份釘死的 OpenCLIP 權重才能載入。

已加 `L1-CKPT-TRAINABLE-ONLY`，注入是「整包存」，並在 channel 型別裡加了
`trainable_only` / `n_params_saved` / `size_bytes`。
**檢查的是 state dict 的 key，不是檔案大小** —— 大小門檻會被「剛好比較小」的
變體矇混過去，而且每次架構變動都要重調。

## F28. 一個「驗協定沒打錯」的檢查，本身什麼都沒驗

寫 n09b 時，我加了 `assert_matches_code()`：把協定餵進 `ESSGNNConfig.from_protocol`，
再逐欄比對建出來的 config 與協定是否一致。看起來很嚴謹。

**它接受 `distance="l2"`。**

原因是 `ESSGNNConfig` 用 `Literal` 標註型別：

```python
Distance = Literal["squared", "euclidean"]
```

而 **dataclass 不會在執行期強制標註** —— 給什麼存什麼。所以：

```
協定寫 "l2"  ->  config.distance == "l2"  ->  兩者相等  ->  檢查通過
```

**這個檢查數學上永遠通過。** 它比沒有檢查更糟，因為它讓人以為驗過了。

改成從標註本身讀出合法詞彙再比：

```python
hints = typing.get_type_hints(ESSGNNConfig)
allowed = typing.get_args(hints[field])   # ("squared", "euclidean")
```

### 為什麼這個特別值得記

**它是被「事後補的測試」抓到的，不是被程式碼審查抓到的。** 我寫完
`assert_matches_code` 時很有信心；是後來寫 `test_a_value_outside_the_configs_vocabulary_is_refused`
並看到它**沒有失敗**，才發現問題。

而且它跟 `mlp_structure` 是同一件事的兩面：那個欄位 `ESSGNNConfig` 根本沒有，
所以協定記了它、沒有任何程式讀它。**「記了但沒人讀」與「驗了但沒在驗」是同一個病**——
一個宣告存在、實際不起作用的東西，比明確缺席危險。

現在 `mlp_structure` 若不是 `linear_silu_linear` 直接拒絕，
並且檢查 `essgnn.py` 真的還在建 SiLU MLP。

## F29. 砍掉 n07b 會留下一個**孤兒 Unity 程序**，佔著 1.5 GB 的 GPU

由使用者發現的：「1294384 在幹嘛沒停？」

**實測**：

```
PID 1294384   PPID 1   已存活 2,761 秒（46 分鐘）
              RSS 2.1 GB   GPU 1,545 MiB
              thor-CloudRendering-f0825767... -screen-width 224 -screen-height 224
```

它是我第一次跑 n07b 時、為了修透明資產 bug 而砍掉的那個程序留下的。
**我砍的是 python，而 AI2-THOR 的 Unity 二進位是它 fork 出來的子程序** ——
父程序死掉之後它被 init 收養，繼續跑。

### 諷刺的地方

那次停 n07b 的**理由**就是「把 GPU 讓給別的工作」。
結果孤兒抱著 1.5 GB 的 GPU 不放，整整 46 分鐘 —— **停下來的成本比不停還高**，
而且沒有任何東西會講。

`main()` 結尾有 `renderer.stop()`，但**那只涵蓋 `main()` 跑到結尾的路徑**。

### 兩層修法，只有第二層是可靠的

**第一層**：`atexit` ＋ `SIGTERM`／`SIGINT` handler。涵蓋乾淨的終止。

**第二層（真正重要的）**：**啟動時掃掉 PPID == 1 的 thor 程序**。
訊號處理擋不住 `SIGKILL`、OOM killer、或直接崩潰的直譯器 ——
而那個孤兒正是這樣活下來的。啟動掃描**不需要上一輪配合**。

判準是 `PPID == 1`：這個節點不會同時開兩個 Controller，
所以一個被 init 收養的 ProcTHOR Unity 程序不屬於任何人。

寫完立刻掃到 1 個（我測試時留下的）。

### 測試的寫法

**沒有真的殺程序來測。** 用假造的 `ps` 輸出 + monkeypatch 掉 `os.kill`，
驗它「殺什麼、不殺什麼」：孤兒殺、有活父程序的不殺、python 自己不殺、grep 那行不殺。
拿真程序測一個殺手不是測試，是危險 —— 跑 pytest 的機器上剛好 PPID 是 1 的東西會遭殃。

## F30. 逐式比對：MetaFind 到底怎麼接 ULIP-2 與 EGNN

寫 n13 前重讀三份原文並排比對。

### 最核心的一句：接點是 Eq. 6

整篇論文的數學核心可以縮成一條式子：

```
e_query = Fusion( E_T(T), E_I(I), E_P(P) )  +  λ · Pooling( ESSGNN(G) )
          └────────── ULIP-2 那一支 ──────────┘     └──── EGNN 那一支 ────┘
```

**ULIP-2 提供的是 object-level 的共同語意空間,不是場景模型。**
**EGNN 提供的是等變的訊息傳遞,不是檢索模型。**
MetaFind 的創新在於用一個**可學的純量 λ 把兩者殘差相加**。

### 一、Stage 1 vs ULIP-2 —— 對齊的東西不同，方向數也不同

| | ULIP-2（Eq. 1–3） | MetaFind Stage 1（Eq. 5） |
|---|---|---|
| 對齊什麼 | 點雲↔圖、點雲↔文字，**兩個成對損失** | 融合後的 query ↔ 融合後的 gallery，**一個損失** |
| 融合 | 沒有 —— 三個特徵各自存在 | 融合模組把每一側壓成一個向量 |
| 方向 | **雙向**（Eq.1/2 各含兩項取 ½） | **單向**（Eq.5 只有一項） |
| 訓練誰 | `min_{E_P}` —— 只有 3D 編碼器 | 兩座塔 |

**所以程式上絕不能寫成 `loss = loss_pc_text + loss_pc_image`** —— 那是 ULIP-2，不是 MetaFind Stage 1。

#### 論文自己沒提的一件事

```
ULIP-2            雙向
MetaFind Stage 1  單向   ← 唯一的例外
MetaFind Stage 2  雙向
```

**Stage 1 照字面實作，會是一個比它所依賴的 backbone 更弱的目標函數**，
而論文沒有一句解釋為什麼第一階段退成單向、第二階段又改回來。
這不是矛盾（三個式子各自都清楚），所以不列 RA；但它是報告該指出的設計選擇。

### 二、ESSGNN vs EGNN —— **論文自己有兩個版本**

**[更正] 本節第一版只引了 2.5，把它當成「MetaFind 的公式」。那是錯的：
正文與 Appendix C 描述的是兩個不同的架構。** 這已登記為 **U-26**。

正文 2.5（Eq. 2–3）——**兩個獨立 MLP，`f_x` 吃更新後的 `h^{l+1}`**：

```
h_i^{l+1} = h_i + Σ f_h(d_ij, h_i, h_j, e_ij; θ_h)
x_i^{l+1} = x_i + Σ (x_i−x_j) · f_x(d_ij, h_i^{l+1}, h_j^{l+1}, e_ij; θ_x)
```

Appendix C（Eq. 10, 13, 14）——**一個共用訊息，和原始 EGNN 同構**：

```
m_ij      = φ_e(h_i, h_j, ‖x_i−x_j‖², e_ij)      (10)
x_i^{l+1} = x_i + Σ (x_i−x_j) · φ_x(m_ij)        (13)
h_i^{l+1} = h_i + Σ φ_h(m_ij)                    (14)
```

而原始 EGNN（Eq. 3–6）與 Appendix C 的差別只剩兩處：
`h` 的更新 EGNN 是 `φ_h(h, m_i)`、Appendix C 是**殘差和**；EGNN 有 `C = 1/(M−1)`、兩者都沒有。

| | EGNN | Appendix C | 正文 2.5 |
|---|---|---|---|
| 共用訊息 `m_ij` | ✅ | ✅ | ❌ 兩個獨立 MLP |
| `f_x` 看到什麼 | `m_ij` | `m_ij` | **`h^{l+1}`** |
| h 更新 | `φ_h(h, m_i)` | `h + Σ φ_h(m_ij)` | `h + Σ f_h(...)` |
| 正規化 `C` | ✅ | ❌ | ❌ |

~~**實作依 2.5**（兩個獨立 MLP），記錄為選擇而非推導。~~

> 🔴 **[已更正 2026-08-30 —— 這句寫反了]**
> 現行協定 `data/outputs/essgnn_arch_protocol.json` 是
> **`architecture_family: "appendix_shared_msg"`**（`coord_feat: "current"`、
> `distance: "squared"`、`use_io_projections: true`、hidden 128 / n_layers 4），
> 也就是**附錄 C 的共用訊息版**，不是正文 2.5 的兩個獨立 MLP。
> `metafind/models/essgnn.py:189-190` 的註解直接寫 `appendix_shared_msg ... <- primary`，
> `ESSGNNConfig.from_protocol` 只接受 `appendix_shared_msg` 或 `sec25_two_mlp` 兩個值，
> 而協定檔給的是前者。
> 記為 **U-26 的判定：附錄 C 的 shared-message 版，標 `[INFERENCE]`，2.5 版留作對照假設**
> （與 `README.md` 開頭 U registry 那段一致）。
> 下面「三、逐項驗證程式」那張表裡的 `Eq. 3 f_x 吃 h^{l+1}` ✅「依 2.5」同樣過期：
> `coord_feat` 現在是 `"current"`，`f_x` 吃的是**舊的** `h^l`。

### 三、逐項驗證程式（不是讀註解，是讀實作）

複習之後把每一條都對回 `metafind/models/essgnn.py` 與 `dual_tower.py`：

| 論文 | 程式 | |
|---|---|---|
| Eq. 2 `h + Σ f_h(...)` | `h_next = h + unsorted_segment_sum(m_h, row, ...)` | ✅ 殘差和 |
| Eq. 3 `(x_i−x_j)·f_x(...)` | `trans = coord_diff * w`，`coord_diff = x[row] − x[col]` | ✅ 方向正確 |
| Eq. 3 `f_x` 吃 `h^{l+1}` | `h_for_x = h_next if coord_feat == "updated"` | ✅ 依 2.5 |
| `Pooling({h^{(L)}})` | `pooling` 設定，主線 `mean` | ✅ |
| Eq. 6 `+ λ·e_layout` | `contribution = self.lam * layout` | ✅ 殘差相加 |
| λ 可學 | `nn.Parameter(init_lambda)` | ✅ |
| Eq. 8 兩方向 **取平均** | `0.5 * (loss_q2g + loss_g2q)` | ✅ 不是相加 |

#### 驗出來的一個真錯誤：參數名 `log_lambda`

它被**直接**當 λ 用（`contribution = self.lam * layout`），初始值 1.0，**沒有任何 exp**。
2.6 只說「a learnable scalar」，沒有正值約束，所以**行為是對的、名字是錯的**。

危險在於：讀到 `log_lambda` 的人會合理地「修正」成 `torch.exp(...)`，
而那會讓 λ 從 1.0 變成 e ≈ 2.718，**悄悄改掉每一個 Table 2／Table 3 的數字**，
不報錯、不變慢。

已改名為 `layout_weight`。**現在改是安全的、之後就不是了** ——
參數名是 checkpoint 的 key，而目前還沒有任何 checkpoint 存在。
`stage2.py` 裡用字串比對挑可訓練參數的那一行也一併更新（它原本比對 `"lambda"`）。

### 四、三個已登記的矛盾，程式各站哪一邊

| | 正文說 | Appendix C／EGNN 說 | 程式 |
|---|---|---|---|
| **U-17** `d_ij` | `‖x_i−x_j‖_2` | `‖x_i−x_j‖²` | **平方**（跟證明走） |
| **RA-2** `f_x` 值域 | `→ ℝ³` | `φ_x(m_ij) ∈ ℝ¹` | **純量**（`_mlp(..., 1)`）—— `ℝ³` 會讓 `Q` 提不出來，證明就不成立 |
| **RA-1** `h⁰` | `Concat(x_i, t_i)` | 證明前提要求 `h⁰` 對 `x` 不變 | **只用 `t_i`**（`h0_mode=semantic`） |

**這三個都不准當成「論文事實」寫進報告**，必須標為矛盾並說明採用哪一讀法。

## 由 F1–F13 推導出的三個架構決策

### D1 — 不重訓 ULIP-2 本身，用官方 released checkpoint 當起點

**理由**：F5（單卡）。ULIP-2 在 ensembled Objaverse+ShapeNet 上的預訓練需要 8×A100 等級資源。

論文 §2.6 Stage 1 訓練的是 **MetaFind 自己的雙塔**（"Both query and gallery encoders are
trained"），ULIP-2 是它的起點（§2.2 "both leveraging the ULIP-2 embedding backbone"）。
所以「不重訓 ULIP-2」指的是不從頭做 ULIP-2 的預訓練，**不等於凍結它的全部權重**。

### D2 — Stage 1 訓練 point encoder + fusion；只有 CLIP 側凍結

> **2026-08-15 修正。先前的草稿在這裡是錯的，而且是最嚴重的一個錯。**
>
> 先前寫成「凍結全部 backbone、把三個模態的 embedding 全部預先快取、
> Stage 1／Stage 2 都只在 1280-d 向量上訓練」。
>
> 那正是論文 Table 3 的 **`Train fuser only`** 那一列 —— 論文明確報告它較差
> （**8.7** vs Full **11.4**），§3.4 也直接寫
> "Fine-tuning the entire encoder outperformed training the fuser only"。
> 把它裝成主線等於一開始就跑錯實驗。

**現在的做法**，`train_scope` 三個等級：

> ⚠ **[2026-08-30] 這張表的可行性欄位是在 24GB 前提下寫的，前提已經不成立。**
> 本機實測 `nvidia-smi`：**NVIDIA GeForce RTX 5090, 32,607 MiB**（driver 595.84）。
> **但顯存變大不等於這些格子變成「可行」** —— 三個等級**沒有一個在 5090 上重量過**，
> 所以下表每一格的可行性判斷一律標 **UNVERIFIED**，`full` 那格仍由 RA-3 量。
> （`point_encoder+fuser` 有一次 25 epoch 的 ladder 執行紀錄，
> 但 `data/outputs/ladder/e25_500w/stage1_ckpt.json` 與 `train_stage1.jsonl`
> **都沒有記錄 GPU 型號或顯存**，所以連「那次跑在 5090 上」都無法從產物證實。）

| 等級 | 訓練什麼 | 單卡可行（本機 RTX 5090 32GB） | 定位 |
|---|---|---|---|
| `fuser_only` | 只有 fusion 層 | **UNVERIFIED**（24GB 下曾標 ✅） | **Table 3 的 ablation 列** |
| `point_encoder+fuser` | PointBERT (32.5M) + fusion + 投影 | **UNVERIFIED**（24GB 下曾標 ✅） | **目前選定的 `actual=frozen` 執行方式**（不是「論文必然如此」，見 U-34） |
| `full` | 再加 ViT-bigG-14 (2.5B) | ❓ **未量測** | `actual_clip_train_scope=trainable` 的執行對象，由 RA-3 量測 |

**快取的範圍也跟著改**：

```
actual=frozen     text / image  →  CLIP 側凍結，可以預先快取
                  point cloud   →  point encoder 可訓練，不可預先快取
actual=trainable  三者皆不可快取；Stage 1 之後由 n10b 用訓練後的 encoder 重編
```

先前「三個模態全部快取」正是讓設計退化成 `fuser_only` 的直接原因。

**[D-1 —— 條件式偏離，已判定不啟用]** ViT-bigG-14 的 text/image 端保持凍結。

**U-34 已於 2026-08-16 判定為 `frozen`，D-1 因此不啟用。** 理由不是「4090 塞不下所以偏離」，而是：MetaFind 明確建立於 ULIP-2，ULIP-2 §3.3 明文凍結 OpenCLIP，而 MetaFind 全文從未聲明改變此策略。§2.6「Both query and gallery encoders are trained」講的是**塔**（point encoder／projection／fuser 本來就在 optimizer 裡），§3.4「entire encoder」對比的是 fuser-only ablation，§2.4「gallery frozen after pretraining」與 §2.6 是 Stage 1／Stage 2 的界線，不是矛盾。**不得寫成「論文明文說 CLIP 凍結」** —— 論文沒有這句。若日後取得官方 code 或作者回覆證實 optimizer 更新到 OpenCLIP，重開 U-34 並啟用 D-1。

**它不是偏離**：ULIP-2 §3.3 明文凍結 OpenCLIP，MetaFind 建立在 ULIP-2 之上且從未
聲明改變此策略，所以主線就是忠實做法。報告陳述的是這個判讀與它的依據，
以及實際採用的 `actual_clip_train_scope = frozen`；`trainable` 那條路仍記錄為
RA-3 的 alternative 稽核對象，不是一個並列的未決讀法。

### D3 — 原始 mesh 保留，不刪除

**先前的草稿寫成「下載 → 編碼 → 存向量 → 刪原始檔」，那是錯的。**

論文的 Algorithm 1 是 iterative scene composition：檢索出資產之後要**放進場景**。
放置需要真實幾何，只有 embedding 不夠。刪掉 mesh 之後 Table 2／3 就得重新下載一次。

現在的做法：

```
datasets/objaverse-lvis/glbs/   46,052 個 GLB，保留（~216 GB）
outputs/pointclouds/            從 mesh 取樣
outputs/renders/                從 mesh 渲染 11 視角
```

`datasets/` 與 `models/` 只下載不寫入；`outputs/` 全部可從它們重新生成。
刪掉 outputs 只損失算力，刪掉 datasets 才要重新下載。

**連帶的未定項**：ULIP-2 的 checkpoint 是在**它自己取樣**的點雲上訓練的。
我們從 mesh 自行取樣，取樣方式若不一致，embedding 會偏離分布而且不會報錯。

> **⚠️ 這一段的結論已修改。** 原本寫「由 G2 gate 擋住」。
> 現行 `G2_pc_sanity` 擋的是**結構有效性**（形狀、有限、`pc_norm`、非退化、自我可辨識）；
> 與 ULIP 官方點雲的比較降為 `L2-PC-ULIP-REF` 診斷，不擋。
> 理由：論文從未說 MetaFind 沿用 ULIP 預取樣的點雲，而 Stage 1 會 fine-tune
> point encoder —— 「和官方雲不一樣」推不出「復現無效」。
> 詳見 `01_GRAPH_SPEC.md` §11。

---

## 這些發現如何改變 graph 的形狀

> ⚠️ **這張表寫於第一輪，之後四輪的決策把其中三格推翻了。** 保留是為了記錄推翻本身。

| 若沒做這次調查，會畫出的圖 | 實際該畫的圖 | 現況 |
|---|---|---|
| `prep → pretrain → finetune → eval` 四段循序 | 前面多一整段 Stage 0 可行性（環境修補 + 硬體 preflight） | ✅ 成立（**預算 pilot 節點已不存在** —— GPT-4o 換成本地 Qwen，D-2） |
| 昂貴的是「訓練」，gate 擋在訓練前 | 昂貴的是標註與編碼（一次性） | ✅ 成立 |
| Ablation 是最貴的尾巴 | 最貴且不可逆的尾巴是 5 位專家 × 200 場景的人工評分 | ❌ **推翻** —— 人工評分不做（偏離 D-4） |
| 等變性是「訓練完看一下」的指標 | 等變性是 **G-INVALID 判準** | ❌ **推翻** —— 等變性**明確被拒絕當 gate**（`validation_plan.yaml` 的 `rejected_gate_candidates`：設成 gate 會誘使放寬容差去遷就論文自己的矛盾），改為 SC-4/5/6 + RA-1/RA-2 |
| 資料準備是一次 fan-out | 是帶 disk backpressure 的 shard 串流，且含**不可逆的刪檔動作** | ❌ **推翻** —— GLB **保留不刪**（D3），沒有不可逆刪檔 |

---

## 待解 UNKNOWN — **SUPERSEDED**

> 本節原有一張自己的 `U-01`…`U-05` 表，**編號與 `01_GRAPH_SPEC.md` §15 的登記表衝突**
> （相同 id、不同意義），且內容已過時，其中一條還是已知錯誤：
> 它主張「用 baseline 的 PC-Only ≈ 98–99% **反推** gallery 分母」。
> **那不可能成立** —— PC-Only 的 query embedding 與它自己的 gallery 條目相同，
> 無論分母多大，自我檢索都趨近 100%，反推不出任何資訊。
> 它引用的 `G2_corpus_validity` 這道 gate 也已不存在。
>
> 整節刪除而非保留註記，是因為「本文件權威最低」擋不住關鍵字搜尋。
>
> **唯一的 UNKNOWN 登記表：[`01_GRAPH_SPEC.md` §15](01_GRAPH_SPEC.md)**
> （機器可讀版：`graph_spec.yaml` 的 `risks_unknowns`；
> 由 `tools/check_graph.py` 檢查兩者對齊，並檢查沒有任何文件引用登記表以外的 `U-nn`）。
