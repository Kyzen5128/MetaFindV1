# Stage 1 架構 —— 我們實際跑的版本，寫成公式，每塊解釋

Kyzen 2026-09-05：「你把你做完的 stage 1 架構轉成公式 每塊解釋」。
每條公式後面標：**PAPER** = MetaFind 論文有寫；**ULIP-2** = 繼承上游；**OURS** = 論文沒寫、我們定的。
對應程式：`metafind/models/ulip_backbone.py`（編碼器）、`metafind/models/fusion.py`（融合）、
`metafind/models/dual_tower.py`（雙塔）、`metafind/models/losses.py`（損失）、`metafind/train/stage1.py`（訓練迴圈）。

記號：一個資產 $A$ 有文字 $T$、渲染圖 $I_1..I_{12}$、點雲 $P\in\mathbb{R}^{10000\times 6}$（xyz + rgb）。$D = 1280$。

---

## 塊 1：三個模態編碼器（ULIP-2 backbone）

$$
\mathbf{z}^T = E_T(T)\in\mathbb{R}^{1280},\qquad
\mathbf{z}^I_k = E_I(I_k)\in\mathbb{R}^{1280},\qquad
\mathbf{z}^P = W_{pc}^{\top}\,\mathrm{PointBERT}(P)\in\mathbb{R}^{1280}
$$

- $E_T$、$E_I$：OpenCLIP ViT-bigG-14，**凍結**。向量事先算好存快取（n06），訓練時直接讀。**PAPER**（用 ULIP-2 backbone）+ **ULIP-2**（CLIP 凍結）。
- $\mathrm{PointBERT}(P)\in\mathbb{R}^{768}$：18 層 Point-Transformer，輸入先 $\mathrm{pc\_norm}$（去質心、除最大半徑）再接 rgb；輸出 = [CLS token ; 其餘 token 的 max] 拼接。$W_{pc}\in\mathbb{R}^{768\times1280}$ 是 ULIP-2 的投影。這一塊**可訓**（「full encoder fine-tuning」，PAPER 消融）；起點是釋出的 ULIP-2 權重（P1s），或我們從頭訓的那顆（DL-095）。
- 影像用哪張：gallery 用 12 張平均 $\bar{\mathbf{z}}^I=\frac{1}{12}\sum_k \mathbf{z}^I_k$；query 用單一視角 $\mathbf{z}^I_{k(A)}$（視角由 uid 決定）。**OURS**（論文沒寫 query 影像怎麼取）。

## 塊 2：前融合正規化 + 遮罩 token

每個模態 $m\in\{T,I,P\}$，先化成單位向量：

$$
\hat{\mathbf{z}}^m = \frac{\mathbf{z}^m}{\lVert \mathbf{z}^m\rVert_2}
$$

再依「有／沒有」決定這一格放什麼：

$$
\mathbf{x}^m = \begin{cases}\hat{\mathbf{z}}^m & \text{模態存在} \\ \mathbf{u}^m & \text{模態被遮／缺}\end{cases},\qquad \mathbf{u}^T,\mathbf{u}^I,\mathbf{u}^P\in\mathbb{R}^{1280}\ \text{可學（mask token）}
$$

- 「用可學的 mask token、不補零」：**PAPER**（§2.6 masked embeddings rather than zero-padding；消融補零 10.5 vs 13.8）。
- 前融合 L2 正規化：**OURS**（`prefusion_norm: true`；讓三個模態進 Transformer 前同尺度）。

## 塊 3：訓練時的隨機遮罩（只在 query 塔）

$$
s_m \sim \mathrm{Bernoulli}(1-p),\quad p = 0.30,\quad m\in\{T,I,P\}\ \text{各自獨立}
$$

$s_m=1$ 表示該模態給 query；$s_m=0$ 就換 mask token。三個都被遮（機率 $0.3^3=2.7\%$）也允許。
- 30%、各自獨立：**PAPER**（§2.6；消融 10%／50%）。
- 「三個都遮也允許」：**OURS**（論文沒說怎麼處理）。
- gallery 塔永遠 $s_T=s_I=s_P=1$：**PAPER**（modality-complete）。

## 塊 4：融合 Transformer（query 塔一套、gallery 塔另一套）

把三格排成序列並加上「我是哪個模態」的可學位置向量 $\mathbf{p}^m$：

$$
X = \begin{bmatrix}\mathbf{x}^T+\mathbf{p}^T\\ \mathbf{x}^I+\mathbf{p}^I\\ \mathbf{x}^P+\mathbf{p}^P\end{bmatrix}\in\mathbb{R}^{3\times1280}
$$

過 $L=2$ 層 pre-norm Transformer encoder（8 頭，FFN 2048，dropout 0）：

$$
H = \mathrm{TransformerEncoder}_{\theta}(X)\in\mathbb{R}^{3\times1280}
$$

讀出 = 三個輸出 token 的平均：

$$
f(A) = \frac{1}{3}\sum_{m\in\{T,I,P\}} H_m \in\mathbb{R}^{1280}
$$

- 「Transformer 融合」：**PAPER**（§3.4 最終選 Transformer）。
- 層數 2、8 頭、FFN 2048、pre-norm、位置向量、平均讀出、被遮的格也參與讀出（`include_absent_slots`）：**OURS**（論文沒給任何一項）。
- query 塔 $f_{query}$ 與 gallery 塔 $f_{gallery}$ 是**兩套不同的** $\theta$；backbone（塊 1）共用一份（P13 試過兩份 Point-BERT，形狀不變）。「雙塔、各自 encoder」：**PAPER**；backbone 共用：**OURS**（圖一寫 ULIP-2 (Shared)，U-16）。

## 塊 5：相似度

$$
\mathrm{sim}(Q,A) = \frac{f_{query}(Q)^{\top} f_{gallery}(A)}{\lVert f_{query}(Q)\rVert\,\lVert f_{gallery}(A)\rVert}
$$

cosine，評估時 float64。**PAPER**（Eq.1 用 sim；cosine 是 CLIP／ULIP 的慣例，論文沒明寫 cosine → 這一點算 **OURS/ULIP-2**）。

## 塊 6：Stage 1 損失（論文 Eq. 5，單向）

batch 有 $B=64$ 個資產。query 端第 $i$ 個資產經遮罩後的向量 $q_i = f_{query}(Q_i)$，gallery 端完整向量 $g_j = f_{gallery}(A_j)$。

$$
\mathcal{L}_{pre} = \frac{1}{B}\sum_{i=1}^{B} -\log\frac{\exp(\mathrm{sim}(q_i,g_i)/\tau)}{\sum_{j=1}^{B}\exp(\mathrm{sim}(q_i,g_j)/\tau)},\qquad \tau = 0.5\ \text{固定}
$$

- 單向 query→gallery、負例 = 同 batch 其他 63 個 gallery 向量：**PAPER**（Eq.5，$B$ = gallery batch）。
- $\tau=0.5$：**PAPER**（§3.1「temperature is 0.5 for all experiments」）。固定不學：**OURS**（論文說 hyperparameter，沒說可學）。
- 程式裡是 `cross_entropy(scale · q̂ gᵀ, arange(B))`，`scale = 1/τ = 2`，取 batch 平均。（Stage 2 才開雙向 `0.5(L_q2g + L_g2q)`，PAPER §2.6 Stage 2。）

## 塊 7：正例定義 = 同一個資產（uid）

$g_i$ 是 $q_i$ **同一件家具**的 gallery 向量。query 的 $T$、$P$ 就是這件家具自己的那份；$I$ 是它 12 張裡的一張。**PAPER**（§2.6「each asset has full modality inputs」+ 遮罩模擬部分模態）；「query 觀測就是 gallery 自己那份」是**我們的讀法**，論文沒明寫 query 從哪來（U-09；DL-094 指出 Table 1 形狀暗示論文的 query 文字／影像不是 gallery 自己那份）。

## 塊 8：訓練哪些參數、怎麼更新

$$
\Theta_{train} = \{\mathrm{PointBERT},\ W_{pc},\ \theta_{query},\ \theta_{gallery},\ \mathbf{u}^m,\ \mathbf{p}^m\},\qquad
\Theta_{frozen} = \{E_T, E_I\}
$$

- AdamW，$\beta=(0.9,0.98)$，$\epsilon=10^{-8}$，weight decay 0.1（bias／LayerNorm／一維參數不衰減），lr $10^{-4}$，第 1 個 epoch 線性暖身 $10^{-6}\to10^{-4}$，之後 cosine 降到 $10^{-5}$，10 個 epoch，batch 64，seed 20260816，fp32。全部 **OURS**（論文一個字都沒給；優化器分組與排程形狀照 ULIP 的 `main.py`，數值是我們掃出來的）。
- 參數量：backbone 可訓 33,483,394；兩個融合 Transformer 47,255,552。

## 塊 9：選 checkpoint

每個 epoch 結束，在選模池 $\mathcal{V}$（val 4,569，或 Kyzen 9/4 要求的整個 20% 9,138）上，用塊 5 對七種條件 $c$ 各算 R@1，取平均最高的 epoch：

$$
\text{best} = \arg\max_{e}\ \frac{1}{7}\sum_{c}\ \mathrm{R@1}_c^{(e)}(\mathcal{V}\to\mathcal{V})
$$

**OURS**（論文沒寫怎麼選；機制照 ULIP 每 epoch 驗證選最好）。

## 塊 10：評估（Table 1，論文 Eq. 1）

對測試池每個資產 $A$、每個條件 $c\in\{T, I, P, T{+}I, T{+}P, I{+}P, T{+}I{+}P\}$：query 只放 $c$ 裡的模態，其餘放 mask token（固定，不隨機）；gallery 用預先算好、發布過的 $f_{gallery}$（G4 檢查、n12 發布）。

$$
A^* = \arg\max_{A'\in\mathcal{G}} \mathrm{sim}(f_{query}(Q_c(A)),\ f_{gallery}(A')),\qquad
\mathrm{R@k} = \frac{1}{|\mathcal{Q}|}\sum_{A}\mathbb{1}\big[\mathrm{rank}(A) \le k\big]
$$

名次 = 分數嚴格高於正解的個數 + 同分的個數 + 1（同分算輸）。七種條件、R@1／R@5：**PAPER**。gallery 是哪一池（20% 或全部）：**OURS**（U-09），兩種都跑。

---

## 一頁總結（資料流）

```
T ──E_T(凍結)──► z^T ──/‖‖──► x^T ─┐
I_k ─E_I(凍結)──► z^I ──/‖‖──► x^I ─┼─ +p^m ─► Transformer(2層) ─► mean ─► f_query(Q)
P ──PointBERT─W_pc► z^P ──/‖‖──► x^P ─┘         （遮掉的格 = mask token u^m）
                                                                    │ cosine / τ=0.5
T, mean(I_1..I_12), P ──同上、另一套 Transformer──────────────────► f_gallery(A)
```
