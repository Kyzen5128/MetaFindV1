# P1s 系列 Stage 1 —— 架構與訓練配方寫成公式，每塊解釋（v2，2026-09-05）

Kyzen 2026-09-05：「你把你做完的 stage 1 架構轉成公式 每塊解釋」；同日轉來 GPT 的逐塊審查並要求「不要全信 逐步對照」。
v2 = 逐點對照 `2methdology.tex`、`3experiments.tex` 與程式碼後的修正版；改了什麼見文末「v1→v2 修正表」。

**這份描述的是 P1s 那一族 checkpoint 實際跑的東西**（P1s、sel20、s1_scratchbb 都是這個架構；差別只在 backbone 起點與選模池），
不是「正式 canonical 配方」：`stage1_hyperparameters.json` 的 base 值是 lr 5e-4、5 epoch；P1s 系列用 `--lr 1e-4 --epochs 10` 覆蓋。

標籤：**PAPER** = MetaFind 論文原文；**PAPER→INFER** = 論文有一句，但細節是我們推的；**ULIP-2** = 上游事實；**OURS** = 論文沒寫、我們定的；**OBS** = 程式裡實際如此。
程式：`metafind/models/ulip_backbone.py`、`fusion.py`、`dual_tower.py`、`losses.py`、`metafind/train/stage1.py`。

記號：資產 $A$ 有文字 $T$、渲染圖 $I_1..I_{12}$、點雲 $P\in\mathbb{R}^{10000\times6}$（xyz+rgb）。$D=1280$。
**視角數：論文 11 個正交視角；我們 12 個 —— DEVIATION（已記；實驗顯示影響小，非主因）。**

---

## 塊 1：三個模態編碼器

$$
\mathbf{z}^T = E_T(T),\qquad \mathbf{z}^I_k = E_I(I_k),\qquad \mathbf{z}^P = W_{pc}^{\top}\,\mathrm{PointBERT}(P),\qquad \text{皆}\in\mathbb{R}^{1280}
$$

| 項 | 我們 | 標籤 |
|---|---|---|
| 兩座塔都用 ULIP-2 backbone 編碼三模態 | 是 | **PAPER**（§2.4 "both leveraging the ULIP-2 embedding backbone"） |
| $E_T,E_I$ = OpenCLIP ViT-bigG-14，**凍結**，向量從 n06 快取讀 | 是 | **ULIP-2**（ULIP-2 凍結 CLIP）+ **OBS**；MetaFind 本身**沒明寫** CLIP 凍不凍 → **UNRESOLVED** |
| PointBERT（18 層，[CLS ; max] 拼接 → 768）+ $W_{pc}$（768→1280）**可訓** | 是 | **PAPER→INFER**：論文只說 "both query and gallery encoders are trained"、消融 "full encoder fine-tuning yields better performance"；哪些子模組可訓沒列。「= PointBERT + 投影」是我們的解讀 |
| PointBERT 起點 | P1s：釋出 ULIP-2 權重；s1_scratchbb：DL-095 從頭訓的 | 兩個不同 arm，**OURS** |
| 影像：gallery 用 $\bar{\mathbf{z}}^I=\tfrac{1}{12}\sum_k\mathbf{z}^I_k$；query 用單張 $\mathbf{z}^I_{k(A)}$ | 是 | **OURS**（論文沒寫 query 影像怎麼取） |

## 塊 2：前融合正規化 + 遮罩向量

$$
\hat{\mathbf{z}}^m=\frac{\mathbf{z}^m}{\lVert\mathbf{z}^m\rVert_2},\qquad
\mathbf{x}^m=\begin{cases}\hat{\mathbf{z}}^m&\text{模態存在}\\ \mathbf{u}^m_{tower}&\text{被遮／缺}\end{cases}
$$

- 「缺的模態不補零，用 masked embedding」：**PAPER**（§2.6 "Rather than zero-padding, we apply masked embeddings"；Table 3 補零 10.5 vs Full 11.4）。
- $\mathbf{u}^m$ **可學**、每模態一顆、$\mathcal{N}(0,0.02)$ 初始化、進 attention、參與讀出：**OURS**（論文只說 masked embeddings）。
- query 塔與 gallery 塔各有一套 $\mathbf{u}^m$（兩個 `ModalityFusion`）；gallery 永遠三模態齊全，所以 gallery 的 $\mathbf{u}^m$ 在 Stage 1 從不被用到：**OBS**。
- 前融合 L2 正規化：**OURS**（`prefusion_norm: true`）。

## 塊 3：訓練時的隨機遮罩（只在 query 塔）

$$
s_m\sim\mathrm{Bernoulli}(1-p),\quad p=0.30,\quad m\in\{T,I,P\}\ \text{各自獨立}
$$

- 30%、各模態獨立：**PAPER**（§2.6；Table 3 消融 10%→7.3、50%→13.2、Full 11.4）。
- 三個全遮（機率 2.7%）也允許：**OURS**（U-23）。
- gallery 塔永遠 $s\equiv1$（modality-complete）：**PAPER**。

## 塊 4：融合 Transformer

$$
X=\begin{bmatrix}\mathbf{x}^T+\mathbf{p}^T\\\mathbf{x}^I+\mathbf{p}^I\\\mathbf{x}^P+\mathbf{p}^P\end{bmatrix}\in\mathbb{R}^{3\times1280},\qquad
H=\mathrm{TransformerEncoder}_\theta(X),\qquad
f(A)=\frac{1}{3}\sum_{m}H_m
$$

- 融合用 Transformer：**PAPER**（§3.4 最終選 Transformer；Table 3 Mean 9.4、MLP 9.9 vs 11.4）。
- 2 層、8 頭、FFN 2048、dropout 0、pre-norm、可學模態位置 $\mathbf{p}^m$、**三格平均讀出（被遮的格也算進平均）**：**OURS**——論文一項都沒給。這是 GPT 與我都認為要升級為 unresolved 的一塊。
- 雙塔：**PAPER**。query 塔 $\theta_q$ 與 gallery 塔 $\theta_g$ **不共用**：**OBS**（`tower_sharing: shared_backbone_separate_fusion`）；論文的參數共用方式**沒寫**（U-16；圖一寫 ULIP-2 (Shared) 只講 backbone）。backbone 共用一份：**OURS**。
- **訓練時 gallery 是 live 的**：每個 batch 重算 $\mathrm{PointBERT}(P)$ → $\theta_g$ → $g_j$，因為 PointBERT 與 $\theta_g$ 每步都在變（只有 CLIP 文字／影像向量是快取）。「預先算好存起來」只發生在**訓完之後**（n11 建索引 → G4 → n12），供評估與 Stage 2 用。

## 塊 5：相似度

$$
\mathrm{sim}(Q,A)=\frac{f_{query}(Q)^{\top}f_{gallery}(A)}{\lVert f_{query}(Q)\rVert\,\lVert f_{gallery}(A)\rVert}
$$

cosine。論文 Eq.1 寫 $\mathrm{sim}$ 沒明說 cosine：**ULIP-2／OURS**。評估用 float64。

## 塊 6：Stage 1 損失（論文 Eq. 5）

$$
\mathcal{L}_{pre}=\frac{1}{B}\sum_{i=1}^{B}-\log\frac{\exp(\mathrm{sim}(q_i,g_i)/\tau)}{\sum_{j=1}^{B}\exp(\mathrm{sim}(q_i,g_j)/\tau)},\qquad B=64,\ \tau=0.5
$$

- 單向 query→gallery、負例 = 同 batch 其他 gallery 向量：**PAPER**（Eq.5）。
- $\tau=0.5$：**PAPER**（§3.1）。固定不學：**OURS**。
- 程式：`cross_entropy((1/τ)·q̂ĝᵀ, arange(B))`，取 batch 平均。**τ 只在訓練用；評估排名直接用 cosine，不除 τ**（除以常數不改名次）。
- Stage 2 才改雙向 $\tfrac12(L^{q2g}+L^{g2q})$：**PAPER**。

## 塊 7：正例 = 同一資產

$g_i$ 是 $q_i$ **同一件家具**的 gallery 向量；query 的 $T$、$P$ 是它自己那份，$I$ 是它 12 張中的一張。
- 同資產配對：**PAPER→INFER**（§2.6 "each asset has full modality inputs" + 遮罩模擬部分模態；論文沒寫 query 觀測從哪來，U-09）。
- 這是 Table 1 形狀對不上的地方：query 用自己那份 → pc 格 98；換成同類別另一件的文字＋圖 → full 50.4 ≈ 論文 51.7（DL-094）。

## 塊 8：訓什麼、怎麼更新

$$
\Theta_{train}=\{\mathrm{PointBERT},W_{pc},\theta_q,\theta_g,\mathbf{u}^m_{q,g},\mathbf{p}^m_{q,g}\},\qquad
\Theta_{frozen}=\{E_T,E_I\}
$$

- 這對應 `train_scope = point_encoder_and_fuser`：**OBS**；與論文 "full encoder fine-tuning" 的對應是我們的解讀（**PAPER→INFER**）。
- **發現（GPT 指出，程式核實）**：我們的 `fuser_only` 凍結 backbone 後，optimizer 仍吃 `model.parameters()`，即 **query 與 gallery 兩套 Fusion 都訓**（`stage1.py:2117`）；論文 Table 3 的 "training only the fusion module **in the query encoder**" 只訓 query 端。兩者不同。這個 arm 沒進過任何報告，但對應關係要修正後才能拿來對 Table 3 的 8.7。
- 優化：AdamW $(0.9,0.98)$、$\epsilon=10^{-8}$、wd 0.1（bias／LN／一維不衰減）、lr $10^{-4}$、1 epoch 暖身 $10^{-6}\to$ lr、cosine 至 $10^{-5}$、10 epoch、batch 64、seed 20260816、fp32：**OURS**（P1s 系列的 arm 值；論文一個字都沒給；排程形狀照 ULIP `main.py`）。

## 塊 9：選 checkpoint

$$
e^*=\arg\max^{\mathrm{lex}}_{e}\big(\overline{R@1}^{(e)}(\mathcal{V}\to\mathcal{V}),\ \overline{R@5}^{(e)}\big),\quad\text{同分保留較早的 epoch}
$$

- 平均是七種條件的 R@1；先比 R@1、同分比 R@5、再同分留早的：**OBS**（`better_checkpoint`）。論文沒寫怎麼選：**OURS**。
- 選模池 $\mathcal{V}$：**D-3b 的規定是 val 4,569**（test 4,569 只給最終報告）。
- **例外（Kyzen 2026-09-04 「20%選啦」，DL-093）**：sel20 與 s1_scratchbb 用整個 20% holdout 9,138 選。這代表 test 那 4,569 也參與了選模，A／A20 的 test 數字帶 selection-side bias，**不再是 held-out**。GPT 的審查也指出這點。兩種都有紀錄；哪一種當最終版，見文末待決。

## 塊 10：評估（論文 Eq. 1）

對每個 query 資產、每個條件 $c$：query 只放 $c$ 內的模態，其餘放 $\mathbf{u}^m$（固定，不隨機）；gallery 用**訓完後**預先算好、G4 檢查、n12 發布的 $f_{gallery}$。

$$
A^*=\arg\max_{A'\in\mathcal{G}}\mathrm{sim}\big(f_{query}(Q_c(A)),f_{gallery}(A')\big),\qquad
\mathrm{R@k}=\frac{1}{|\mathcal{Q}|}\sum_A\mathbb{1}[\mathrm{rank}(A)\le k]
$$

名次 = 嚴格高於正解的個數 + 同分個數 + 1。七條件、R@1／R@5：**PAPER**。gallery 是哪一池論文沒寫（U-09），現在的正式協定：

| 協定 | query | gallery | 備註 |
|---|---|---|---|
| A | test 4,569 | test 4,569 | |
| A20 | test 4,569 | holdout 9,138 | gallery 含 val（選模用過）→ selection-side bias 已記 |
| B | test 4,569 | full 45,692 | 含 36,554 訓練資產當干擾 |
| A20_holdout_vs_holdout | holdout 9,138 | holdout 9,138 | Kyzen 9/4 的圖；選模池 = 報告池 |

---

## 資料流（一頁）

```
訓練（每個 batch，live）：
T ──E_T(凍結,快取)──► z^T ─/‖‖─► x^T ─┐
I_k ─E_I(凍結,快取)──► z^I ─/‖‖─► x^I ─┼ +p^m ─► Transformer_q(2層) ─► mean ─► q_i
P ──PointBERT(可訓)─W_pc─► z^P ─/‖‖─► x^P ─┘   （遮掉的格 = u^m_q）
                                              InfoNCE：cos(q_i, g_j)/τ，τ=0.5，單向
T, mean(I_1..12), P ──同上、Transformer_g──────────────────────────────► g_j

評估（訓完後）：gallery 全部編一次存索引；query 依條件 c 放模態；cosine 排名（不除 τ）。
```

---

## v1 → v2 修正表（對 GPT 審查逐點核對後）

| GPT 指出 | 核對結果 | 處置 |
|---|---|---|
| PointBERT 可訓標 PAPER | 論文只有 "encoders are trained"、"full encoder fine-tuning"；子模組沒列 | 改 PAPER→INFER |
| CLIP 凍結標 MetaFind PAPER | MetaFind 沒寫；凍結來自 ULIP-2 | 改 ULIP-2 + OBS，MetaFind UNRESOLVED |
| 可學 mask token 標 PAPER | 論文只說 masked embeddings | 改 OURS；補兩套 $\mathbf{u}^m$ |
| 補零 10.5 vs 13.8 | 13.8 是 Table 1 的 text-only；Table 3 的 Full 是 11.4 | **改 10.5 vs 11.4**；Train fuser only 8.7 vs 11.4 |
| Stage 1 gallery「預先算好」 | 訓練時 live，訓完才快取 | 塊 4 補寫兩階段 |
| 12 視角未標 deviation | 論文 11 | 開頭標 DEVIATION |
| 選模池「4,569 或 9,138」 | D-3b 是 val；9,138 含 sealed test；用它選是 Kyzen 9/4 的口令 | 塊 9 寫清規定、例外、後果；待 Kyzen 決定最終版 |
| best 只寫 R@1 | 程式是 (R@1, R@5) 字典序、同分留早 | 改公式 |
| Block 10 只寫兩種 gallery | 現有 A／A20／B／A20_holdout | 列表 |
| Q/G 兩套 Fusion 標 PAPER | 論文沒寫共用方式（U-16） | 改 OBS |
| 圖上 cosine/τ 混評估 | τ 只在訓練 | 拆開 |
| 標題「現在 Stage 1」 | canonical 是 5e-4／5 epoch，1e-4／10 是 P1s arm | 改標題與塊 8 |
| `fuser_only` 訓兩套 Fusion vs 論文只訓 query fusion | **程式核實為真**（`stage1.py:2117`） | 記為發現；arm 未進報告 |

GPT 說「數學資料流約 80% 正確、證據標籤不合格」——對照後我接受。上表每一項都改了。
