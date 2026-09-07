# MetaFind 論文的完整流程（訓練 + 評估，含 Stage 2）—— 逐步列表

來源：`docs/paper/metafind_source/{2methdology.tex, 3experiments.tex, appendix.tex}`（arXiv 2510.04057v1）。
標籤：**PAPER** = 論文原文；**UNKNOWN** = 論文沒寫；**OURS** = 我們目前的做法（只在需要對照時列）。
Kyzen 2026-09-05 00:0x：「你幫我列出論文的全部流程步驟 訓練 包含 stage 2 完整的」。

---

## 0. 資料準備（§2.3 Data Preparation）

| 步 | 內容 | 標籤 |
|---|---|---|
| 0.1 | 物件級資料：Objaverse-LVIS，約 48,000 個資產 | PAPER |
| 0.2 | 每個資產從 **11 個正交視角**渲染 | PAPER（相機參數、解析度、背景：UNKNOWN） |
| 0.3 | 用 **GPT-4o** 標註，產生結構化描述：類別、尺寸、材質、擺放限制 | PAPER（prompt、輸入是哪幾張圖、輸出格式：UNKNOWN） |
| 0.4 | 場景級資料：ProcTHOR-10K，>10,000 間程序生成的房子，由 >3,000 個資產組成；每間房有每個資產的精確座標與語意 metadata | PAPER |
| 0.5 | 從房間抽「結構化場景圖」：節點 = 已放置的物件；邊有兩種：(i) 物理關係邊（adjacency、support，如「杯子在桌上」）；(ii) 語意關係邊（LLM 對物件對生成的關係句，如「顯微鏡–實驗桌」） | PAPER（LLM 型號與 prompt、邊的門檻：UNKNOWN） |
| 0.6 | 切分：兩個資料集都 **80% 訓練、20% 測試** | PAPER（seed、切在資產還是房間、是否分層：UNKNOWN） |

## 1. 架構（§2.4、§2.5）

| 步 | 內容 | 標籤 |
|---|---|---|
| 1.1 | 雙塔：query encoder、gallery encoder，**兩座都用 ULIP-2 embedding backbone** 各自編碼可用模態（文字、影像、點雲） | PAPER |
| 1.2 | 每個模態的向量經 **fusion layer** 合成一條向量；候選有 mean pooling、MLP、masked MLP、gated fusion、Transformer；**最終選 Transformer**（§3.4） | PAPER（Transformer 層數、寬度、token 形式：UNKNOWN） |
| 1.3 | gallery 塔是 modality-complete（三模態都給），預先算好所有資產的向量存起來；query 塔接受任意子集，並可加 layout 向量 | PAPER |
| 1.4 | ESSGNN：場景圖節點 h_i^(0) = Concat(x_i, t_i)，x_i 為 3D 位置，t_i 為文字衍生特徵；邊嵌入 e_ij = LLM 關係句經**凍結的文字編碼器（CLIP 或 BERT）** | PAPER（t_i 用哪個編碼器、維度 d：UNKNOWN） |
| 1.5 | 訊息傳遞 = 修改版 EGCL：h_i^{l+1} = h_i^l + Σ_j f_h(d_ij, h_i, h_j, e_ij)；x_i^{l+1} = x_i^l + Σ_j (x_i − x_j)·f_x(d_ij, h_i^{l+1}, h_j^{l+1}, e_ij)；f_h、f_x 為 MLP；L 層後 e_layout = Pooling({h_i^(L)})；SE(3) 等變（附錄有證明） | PAPER（層數 L、Pooling 是哪種、MLP 大小：UNKNOWN；上游 EGNN QM9 用 7 層與 sum） |

## 2. Stage 1：跨模態對齊預訓練（§2.6 Stage 1）

| 步 | 內容 | 標籤 |
|---|---|---|
| 2.1 | 資料：Objaverse-LVIS 的 80%，每個資產有完整的文字、影像、點雲 | PAPER |
| 2.2 | 訓練對象：「both query and gallery encoders are trained」；消融：「full encoder fine-tuning」優於「train fuser only」（text-only R@1 13.8 vs 8.7） | PAPER（哪些層可訓——Point-BERT？CLIP 文字/影像塔？——UNKNOWN；OURS：Point-BERT 從 ULIP-2 釋出權重微調，CLIP 兩塔凍結） |
| 2.3 | 隨機模態遮蔽：query 的每個模態**各自 30% 機率整個被遮**；被遮的用 **masked embedding**（不是補零）；消融：10%→7.3、50%→13.2、補零→10.5 | PAPER |
| 2.4 | gallery 塔訓成 modality-complete | PAPER |
| 2.5 | 損失 L_pre = −log [ exp(sim(f_query(Q), f_gallery(A))/τ) / Σ_{A'∈B} exp(sim(f_query(Q), f_gallery(A'))/τ) ]；B = gallery batch；**單向 query→gallery** | PAPER |
| 2.6 | τ = **0.5**（「for all experiments」，§3.1） | PAPER |
| 2.7 | 優化器、學習率、batch、epoch、warmup、schedule、seed、選 checkpoint 的規則 | **全部 UNKNOWN** |
| 2.8 | 每步 query 看到的觀測是資產「自己的」文字/影像/點雲，還是別的來源 | UNKNOWN（Table 1 的形狀暗示 query 文字/影像不等於 gallery 自己那份；見 DL-094） |

## 3. Stage 2：Layout-aware 微調（§2.6 Stage 2；§3.2 末段）

| 步 | 內容 | 標籤 |
|---|---|---|
| 3.1 | 資料：ProcTHOR 的 80%（房間級，資產分布與 Objaverse-LVIS 不同） | PAPER |
| 3.2 | 對每個 query：e_query = Fusion(e_text, e_img, e_pc) + **λ · e_layout**；λ 是**可學的純量**；殘差式加法 | PAPER（λ 初值：UNKNOWN） |
| 3.3 | e_layout 由 ESSGNN 從**當前場景圖**算出 | PAPER（場景圖裡放哪些物件——已放置的全部？排除目標？——UNKNOWN） |
| 3.4 | **Scene dropout 30%**：30% 的 batch 不給 e_layout | PAPER |
| 3.5 | 只更新 **query 端的 fusion 層 + ESSGNN**；gallery encoder 凍結（省成本、保持資產向量一致） | PAPER |
| 3.6 | §3.2 補充：報告的結果是「single shared head：Stage 2 **凍結兩座 encoder**，只更新 ESSGNN 與 fusion，加 30% scene dropout」；另一種做法是保留兩個 fusion head（layout-free 的 Stage 1 head 與 scene-aware 的 Stage 2 head），用 Stage 1 head 就重現「w/o ESSGNN」數字 | PAPER |
| 3.7 | 損失：**雙向** InfoNCE，L = ½(L^{q2g} + L^{g2q})；τ 同上 0.5 | PAPER |
| 3.8 | 正例定義（query 對應哪個 gallery 資產）、query 的模態怎麼來、gallery 是 ProcTHOR 的 3,000 資產還是 Objaverse、優化超參、epoch | **UNKNOWN** |

## 4. 推論與場景組裝（§2.6 Inference；Algorithm 1）

| 步 | 內容 | 標籤 |
|---|---|---|
| 4.1 | 所有 gallery 資產向量預先算好、快取 | PAPER |
| 4.2 | 迭代組裝：給初始場景圖 G_0 與 N 個資產 query；for i=1..N：e_layout ← EGNN(G)；編碼 Q_i 的可用模態；e_query ← Fusion + λ·e_layout；A*_i ← argmax sim(e_query, e_gallery)；放進場景、更新 G | PAPER |
| 4.3 | 效率選項：全序列（品質最好）、平行、或按區域分解（區內序列、區間平行） | PAPER |

## 5. 評估

### 5a 物件級（Table 1，§3.2）

| 步 | 內容 | 標籤 |
|---|---|---|
| 5a.1 | Objaverse-LVIS 的 20% 測試 | PAPER（gallery 是這 20% 還是全部 48K：UNKNOWN，U-09） |
| 5a.2 | 七種 query 條件：Text / Image / PC / T+I / T+PC / I+PC / T+I+PC | PAPER |
| 5a.3 | 指標 R@1、R@5 | PAPER |
| 5a.4 | 基線（ULIP、OpenShape、SCA3D、Uni3DL、Uni3D、OmniBind）：用預訓練單塔編碼器，加一層 **mean pooling** 合併可用模態，對預先編碼的 gallery 檢索；另有「本模型 + mean fusion、無 layout」當消融基線 | PAPER |
| 5a.5 | 論文數字（R@1）：基線 pc 97.9–99.0、text 0.1–6.9、T+I 0–0.5、full 5.5–11.9；**MetaFind w/o ESSGNN 13.8 / 11.7 / 75.1 / 17.2 / 44.5 / 45.8 / 51.7**；w/ ESSGNN 11.3 / 10.5 / 63.2 / 15.9 / 41.2 / 42.0 / 48.2 | PAPER |
| 5a.6 | query 的文字/影像/點雲各自從哪來 | **UNKNOWN**（見 DL-094：所有方法加文字/影像都變差，只有 query 觀測不是 gallery 自己那份才會如此） |

### 5b 場景級（Table 2，§3.3）

| 步 | 內容 | 標籤 |
|---|---|---|
| 5b.1 | 用 **I-Design** 的場景生成管線（給房間描述 → 設計、檢索、擺放），原本用 OpenShape 檢索，這裡換成 MetaFind（有/無 ESSGNN） | PAPER |
| 5b.2 | 不算檢索準確率；評四個維度各 1–5 分：整體美感、色彩與材質、場景一致性、真實感與幾何合理性 | PAPER |
| 5b.3 | 評分者：GPT-4o（給場景 layout 與渲染圖）+ **5 位專家**；**200 個隨機場景**；分數對人與樣本平均 | PAPER（prompt、渲染視角：UNKNOWN） |
| 5b.4 | 數字：ULIP 2.7–3.0、OpenShape 2.95–3.28、w/o ESSGNN 3.22–3.55、w/ ESSGNN 4.04–4.25 | PAPER |

### 5c 消融（Table 3，§3.4；Text-only R@1）

Full（雙向、迭代檢索、ESSGNN）11.4；w/o 迭代 11.3；w/o layout 13.5；GAT 11.0；Fusion=Mean 9.4；MLP 9.9；dropout 10% 7.3；50% 13.2；只訓 fuser 8.7；補零 10.5。 —— PAPER

---

## 6. 我們現在對到哪（OURS，一句話一項）

- 0.1–0.3：45,692 個資產（46,024 減排除）、**12** 視角（非 11）、Gemma-4-12B 標註（非 GPT-4o）——DEVIATION 已記。
- 0.6：80/20 seed 20260816；20% 再對半 val/test（D-3b）；Kyzen 2026-09-04 又下「20% 選、20% 報」（DL-093，已停）。
- 1.1–1.3：釋出 ULIP-2 backbone + 兩個 Transformer fusion；gallery 預編碼、G4 檢查、n12 發布。
- 2.2：Point-BERT 微調、CLIP 凍結；**2026-09-04 23:5x 起另跑 ULIP-2 從頭訓**（DL-095）。
- 2.3–2.6：30%、mask token、單向 InfoNCE、τ 0.5 —— 照論文。
- 2.7：lr 1e-4、10 epoch、batch 64 —— 我們量的。
- 2.8：query = 資產自己的觀測（影像單視角）→ pc 格 98，跟論文 75 不同；換件評估（P1 partner）才出論文形狀。
- 3.x：Stage 2 有初步結果（`exp_stage2_procthor_retrieval*.json`），ESSGNN 依 EGNN QM9 設定；ProcTHOR 切分與 query 定義仍待定。
- 5b：GPT-4o 場景評分未做。
