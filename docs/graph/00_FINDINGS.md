# 前置調查發現（在設計 graph 之前必須先知道的事）

> 這份文件記錄實際檢查論文與程式碼後得到的**實測事實**（F 系列）與由此推導的**架構決策**（D 系列）。
>
> **權威順序**：`docs/metafind_paper.md`（論文本身）> `02_BUILD_STEPS.md`（最新決策）
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
- 官方 EGNN 實作 [`egnn_clean.py:95-103`](/home/kyzen/egnn/models/egnn_clean/egnn_clean.py) 的 `forward(h, edge_index, coord, ...)`
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

**證據**：[`ULIP_models.py`](/home/kyzen/ULIP/models/ULIP_models.py) `ULIP2_WITH_OPENCLIP.__init__`

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

## F5. 單張 RTX 4090 (24GB) vs 官方腳本假設 **8 張 GPU**

**證據**

```
scripts/pretrain_pointbert.sh:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 ...
本機: 1× NVIDIA GeForce RTX 4090, 24564 MiB
```

ULIP-2 用 open_clip **ViT-bigG-14**（~2.5B 參數）當 text/image backbone，
其論文 §3.3 明文 **"freeze it during the pre-training"**。

實作上有個對不上的地方：`ULIP2_PointBERT_Colored` 只呼叫 `eval()`，**沒有**設
`requires_grad = False`，而 `main.py` 的 optimizer 用 `if not p.requires_grad: continue`
挑參數 —— 所以照公開碼跑，那 2.5B 參數會進 optimizer。
同檔的 **ULIP-1 五個 factory 都有明確凍結**，只有 ULIP-2 的沒有。

> 這是 **ULIP-2 程式與它自己論文的落差**，不是「官方設計不凍結」。
> 先前本文用這個落差論證「凍結是我們的偏離」，那個推論已撤回（見 U-34）。
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
| 46,052 點雲 @10000 pts × (xyz+rgb) × float32 | 240KB × 46,052 ≈ **11 GB** |
| 46,052 × 11 views 渲染圖 @~100KB | 507K 張 ≈ **51 GB** |
| ProcTHOR-10K | ~10 GB |
| 快取 embedding（**只有 text/image 兩種**，見 D2） | 46,052 × 2 × 1280 × 4B ≈ **0.5 GB** |
| **合計** | **~88 GB**（**不含 GLB 原檔 ~216 GB**，見 D3） |

> **2026-08-14 更新**：所有大型資料改放 `/mnt/data1/kyzen/MetaFind`（779GB 可用），
> repo 內以 `./data` symlink 指向。90GB 對 779GB 綽綽有餘。
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

- 標註 prompt 明說渲染圖是 scale-normalised，要模型依物件類別估計，
  而不是假裝在量測（`test_prompt_explains_that_renders_are_scale_normalised`）。
- **原始 bounding box 記進 render sidecar**（`extents_m`、`volume_m3`）。
  我們手上本來就有，記下來幾乎零成本，而且讓模型的估計**可稽核**
  —— 之後可以直接比對「模型猜的尺寸」與「mesh 真實尺寸」的相關性。
- 尺寸欄位不得被當成量測值使用。

---

## F14. 其他實作細節（原編號 F9，與 F8→F13 的順序衝突，改編號）

| 項目 | 事實 | 影響 |
|---|---|---|
| `unsorted_segment_sum` 用 `scatter_add_`（`egnn_clean.py:149-154`） | GPU atomics → **非確定性** | 列入 `nondeterminism_sources`；等變性測試容差不能設成 exact |
| `coords_agg` 預設 `'mean'`（`egnn_clean.py:17`） | 論文 Eq.(3) 是 **sum** | 設定必須明寫 `coords_agg='sum'`，列 L1 |
| `E_GCL.forward` 回傳的第三個值是**原封不動的** `edge_attr`（`egnn_clean.py:103`） | 語意邊不會被逐層更新 | 與論文一致（`e_ij` 為常量），寫進 postcondition |
| `get_edges()` 建**全連接**圖（`egnn_clean.py:167-176`） | 我們要的是稀疏的 physical+semantic 邊 | 必須自寫 `edge_index` 建構；不可複用 |
| `Objaverse_Lvis_Colored.__getitem__` 只回傳 `(pc, label, name)` | **沒有圖片、沒有 caption** | tri-modal 資料要自己組，不能複用這個 Dataset |
| `Objaverse_Lvis_Colored` 需要 `data/objaverse-lvis/lvis.json` 等 | 檔案不存在 | 列入 `n02_download` 的明確產出 |

---

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

| 等級 | 訓練什麼 | 24GB 可行 | 定位 |
|---|---|---|---|
| `fuser_only` | 只有 fusion 層 | ✅ | **Table 3 的 ablation 列** |
| `point_encoder+fuser` | PointBERT (32.5M) + fusion + 投影 | ✅ | **主線** |
| `full` | 再加 ViT-bigG-14 (2.5B) | ❌ | 硬體限制，由 RA-3 記錄 |

**快取的範圍也跟著改**：

```
text / image  →  CLIP 側凍結，可以預先快取
point cloud   →  point encoder 主線可訓練，不可預先快取
```

先前「三個模態全部快取」正是讓設計退化成 `fuser_only` 的直接原因。

**[D-1 —— 條件式偏離，取決於 U-34]** ViT-bigG-14 的 text/image 端保持凍結。
**這未必是偏離**：ULIP-2 §3.3 明文凍結 OpenCLIP，而 MetaFind 建立在 ULIP-2 之上，
所以主線可能正是忠實做法。是否偏離取決於 MetaFind 的 "entire encoder" 是否涵蓋 CLIP（U-34）。
兩種讀法都會在報告中陳述，並附上實際採用的 `clip_train_scope`。

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
