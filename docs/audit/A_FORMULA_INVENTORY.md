# A. FORMULA_INVENTORY

三份文獻中**所有** display equation，一條不漏。**由修復版抽取** —— 
`ulip2_paper.md` 與 `egnn_paper.md` 的 LaTeX 跳脫序列在原檔中被寫成了控制字元
（`\frac`→`<FF>rac`、`\tau`→`<TAB>au`、`\right`→`<LF>ight`），共 21 與 122 處。
修復版在 `docs/audit/repaired/`，**原檔未動**。`metafind_paper.md` 未受損。

**總計 67 條**：MetaFind 20、ULIP-2 3、EGNN 44。

`relationship_to_metafind` 只能是 `DIRECTLY_USED` / `MODIFIED` / `CONCEPTUAL_SOURCE` / `NOT_USED` / `UNKNOWN`。
**EGNN 的預設是 `NOT_USED`** —— 存在於 EGNN 論文不等於 MetaFind 用了它。

## MetaFind — `metafind_paper.md`（20 條）

| 行 | 節 | Eq | 公式 | relationship | 註 |
|---|---|---|---|---|---|
| 61 | 2.1 Task Definition | **1** | `A^* = \operatorname{argmax}_{A \in \mathcal{A}} \operatorname{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A))` | `DIRECTLY_USED` |  |
| 99 | 2.5 ESSGNN: Scene-Aware Equivariant Graph Enco | **-** | `h_i^{(0)} = \operatorname{Concat}(x_i, t_i)` | `DIRECTLY_USED` **未編號但關鍵**：h⁰ 的定義。矛盾 C3 的來源 —— Concat(x,t) 使 h⁰ 隨 x 變，而 Appendix C 的證明前提要求 h⁰ 對 x 不變 |
| 105 | 2.5 ESSGNN: Scene-Aware Equivariant Graph Enco | **2** | `h_i^{l+1} = h_i^l + \sum_{j \in \mathcal{N}(i)} f_h(d_{ij}^l, h_i^l, h_j^l, e_{ij}; \theta_h)` | `DIRECTLY_USED` |  |
| 107 | 2.5 ESSGNN: Scene-Aware Equivariant Graph Enco | **3** | `x_i^{l+1} = x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \cdot f_x(d_{ij}^l, h_i^{l+1}, h_j^{l+1}, e_{ij}; \theta_x)` | `DIRECTLY_USED` |  |
| 111 | 2.5 ESSGNN: Scene-Aware Equivariant Graph Enco | **-** | `e_{\text{layout}} = \operatorname{Pooling}(\{h_i^{(L)}\})` | `DIRECTLY_USED` **未編號**：e_layout 的定義。Pooling 種類未指定 → UNKNOWN，見文件 C |
| 117 | 2.5 ESSGNN: Scene-Aware Equivariant Graph Enco | **4** | `(R x^{l+1} + T, h^{l+1}) = \text{ESSGNN}(R x^l + T, h^l, E)` | `DIRECTLY_USED` |  |
| 125 | Stage 1: Cross-Modal Alignment Pretraining | **5** | `\mathcal{L}_{\text{pre}} = -\log \frac{\exp(\operatorname{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A)) / \tau)}{\sum_{A' \in \mathcal{B}} \exp(…` | `DIRECTLY_USED` |  |
| 132 | Stage 2: Layout-Aware Fine-Tuning | **6** | `e_{\text{query}} = \operatorname{Fusion}(e_{\text{text}}, e_{\text{image}}, e_{\text{pc}}) + \lambda \cdot e_{\text{layout}}` | `DIRECTLY_USED` |  |
| 138 | Stage 2: Layout-Aware Fine-Tuning | **7a** | `\mathcal{L}_{\text{layout}}^{q2g} = -\log \frac{\exp(\operatorname{sim}(e_{\text{query}}, e_{\text{gallery}}) / \tau)}{\sum_{e'_{\text{gallery}} \i…` | `DIRECTLY_USED` |  |
| 140 | Stage 2: Layout-Aware Fine-Tuning | **7b** | `\mathcal{L}_{\text{layout}}^{g2q} = -\log \frac{\exp(\operatorname{sim}(e_{\text{gallery}}, e_{\text{query}}) / \tau)}{\sum_{e'_{\text{query}} \in …` | `DIRECTLY_USED` |  |
| 144 | Stage 2: Layout-Aware Fine-Tuning | **8** | `\mathcal{L}_{\text{layout}} = \frac{1}{2} \left( \mathcal{L}_{\text{layout}}^{q2g} + \mathcal{L}_{\text{layout}}^{g2q} \right)` | `DIRECTLY_USED` |  |
| 322 | Appendix C: Equivariance Proof of ESSGNN - Ext | **9** | `(Q x^{l+1} + g, h^{l+1}) = \text{ESSGNN}(Q x^l + g, h^l, E)` | `DIRECTLY_USED` |  |
| 326 | Appendix C: Equivariance Proof of ESSGNN - Ext | **10** | `m_{ij} = \phi_e \left( h_i^l, h_j^l, \\|x_i^l - x_j^l\\|^2, e_{ij} \right)` | `DIRECTLY_USED` |  |
| 330 | Appendix C: Equivariance Proof of ESSGNN - Ext | **11** | `\\|Q x_i^l + g - (Q x_j^l + g)\\|^2 = \\|Q(x_i^l - x_j^l)\\|^2 = \\|x_i^l - x_j^l\\|^2` | `DIRECTLY_USED` |  |
| 334 | Appendix C: Equivariance Proof of ESSGNN - Ext | **12** | `m'_{ij} = \phi_e \left( h_i^l, h_j^l, \\|Q x_i^l + g - Q x_j^l - g\\|^2, e_{ij} \right) = m_{ij}` | `DIRECTLY_USED` |  |
| 338 | Appendix C: Equivariance Proof of ESSGNN - Ext | **13** | `x_i^{l+1} = x_i^l + \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij})` | `DIRECTLY_USED` |  |
| 342 | Appendix C: Equivariance Proof of ESSGNN - Ext | **-** | `Q x_i^l + g + \sum_{j \neq i} (Q x_i^l + g - Q x_j^l - g) \cdot \phi_x(m_{ij}) = Q x_i^l + g + Q \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij})` | `DIRECTLY_USED` Appendix C 證明的代數步驟（把 Q 提出） |
| 344 | Appendix C: Equivariance Proof of ESSGNN - Ext | **-** | `= Q \left[ x_i^l + \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij}) \right] + g = Q x_i^{l+1} + g` | `DIRECTLY_USED` 同上，證明收尾 |
| 350 | Appendix C: Equivariance Proof of ESSGNN - Ext | **14** | `h_i^{l+1} = h_i^l + \sum_{j \neq i} \phi_h(m_{ij})` | `DIRECTLY_USED` |  |
| 356 | Appendix C: Equivariance Proof of ESSGNN - Ext | **15** | `(Q x^{l+1} + g, h^{l+1}) = \text{ESSGNN}(Q x^l + g, h^l, E)` | `DIRECTLY_USED` |  |

## ULIP-2 — `ulip2_paper.md`（3 條）

| 行 | 節 | Eq | 公式 | relationship | 註 |
|---|---|---|---|---|---|
| 78 | 3.3 Tri-modal Pre-training | **1** | `\mathcal{L}_{P2I} = -\frac{1}{2} \sum_{i} \left[ \log \frac{\exp(f_{P,i} \cdot f_{I,i} / \tau)}{\sum_{j} \exp(f_{P,i} \cdot f_{I,j} / \tau)} + \log…` | `CONCEPTUAL_SOURCE` | L_P2I -- symmetric point<->image. MetaFind Stage 1 aligns FUSED towers instead |
| 83 | 3.3 Tri-modal Pre-training | **2** | `\mathcal{L}_{P2T} = -\frac{1}{2} \sum_{i} \left[ \log \frac{\exp(f_{P,i} \cdot f_{T,i} / \tau)}{\sum_{j} \exp(f_{P,i} \cdot f_{T,j} / \tau)} + \log…` | `CONCEPTUAL_SOURCE` | L_P2T -- symmetric point<->text. Same |
| 86 | 3.3 Tri-modal Pre-training | **3** | `\min_{E_P} \mathcal{L}_{P2I} + \mathcal{L}_{P2T}` | `CONCEPTUAL_SOURCE` | min_{E_P} -- ULIP-2 trains ONLY the 3D encoder; MetaFind trains both towers |

## EGNN — `egnn_paper.md`（44 條）

| 行 | 節 | Eq | 公式 | relationship | 註 |
|---|---|---|---|---|---|
| 40 | 2.1 Equivariance | **1** | `\phi(T_g(x)) = S_g(\phi(x))` | `CONCEPTUAL_SOURCE` |  |
| 46 | 2.1 Equivariance | **-** | `y + g = \phi(x + g)` | `NOT_USED` |  |
| 48 | 2.1 Equivariance | **-** | `Qy = \phi(Qx)` | `NOT_USED` |  |
| 50 | 2.1 Equivariance | **-** | `P(y) = \phi(P(x))` | `NOT_USED` |  |
| 58 | 2.2 Graph Neural Networks | **-** | `m_{ij} = \phi_e(h_i^l, h_j^l, a_{ij})` | `NOT_USED` |  |
| 59 | 2.2 Graph Neural Networks | **-** | `m_i = \sum_{j \in \mathcal{N}(i)} m_{ij}` | `NOT_USED` |  |
| 60 | 2.2 Graph Neural Networks | **2** | `h_i^{l+1} = \phi_h(h_i^l, m_i)` | `CONCEPTUAL_SOURCE` |  |
| 71 | 3. Equivariant Graph Neural Networks | **3** | `m_{ij} = \phi_e \left( h_i^l, h_j^l, \\|x_i^l - x_j^l\\|^2, a_{ij}  \right)` | `MODIFIED` |  |
| 72 | 3. Equivariant Graph Neural Networks | **4** | `x_i^{l+1} = x_i^l + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})` | `MODIFIED` |  |
| 74 | 3. Equivariant Graph Neural Networks | **5** | `m_i = \sum_{j 
eq i} m_{ij}` | `MODIFIED` |  |
| 76 | 3. Equivariant Graph Neural Networks | **6** | `h_i^{l+1} = \phi_h(h_i^l, m_i)` | `MODIFIED` |  |
| 87 | 3.1 Analysis on E(n) Equivariance | **14** | `Qx^{l+1} + g, h^{l+1} = \text{EGCL}(Qx^l + g, h^l)` | `CONCEPTUAL_SOURCE` |  |
| 95 | 3.2 Extending EGNNs for Vector Type Representa | **7a** | `v_i^{l+1} = \phi_v(h_i^l) v_i^{init} + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})` | `NOT_USED` | velocity-type update; MetaFind has no velocity channel |
| 97 | 3.2 Extending EGNNs for Vector Type Representa | **7b** | `x_i^{l+1} = x_i^l + v_i^{l+1}` | `NOT_USED` | velocity-type update |
| 104 | 3.3 Inferring the Edges | **8** | `m_i = \sum_{j \in \mathcal{N}(i)} m_{ij} = \sum_{j 
eq i} e_{ij} m_{ij}` | `NOT_USED` | edge inference / e_ij gating -- MetaFind's e_ij is an LLM sentence, unrelated |
| 162 | 5.2 Graph Autoencoder | **9** | `\hat{A}_{ij} = g_e(z_i, z_j) = \frac{1}{1 + \exp(w \\|z_i - z_j\\|^2 + b)}` | `NOT_USED` | edge inference |
| 259 | Appendix A: Equivariance Proof | **15** | `Qx^{l+1} + g, h^{l+1} = \text{EGCL}(Qx^l + g, h^l)` | `CONCEPTUAL_SOURCE` |  |
| 265 | Appendix A: Equivariance Proof | **11** | `\\|(Qx_i^l + g) - (Qx_j^l + g)\\|^2 = \\|Q(x_i^l - x_j^l)\\|^2 = (x_i^l - x_j^l)^T Q^T Q (x_i^l - x_j^l) = \\|x_i^l - x_j^l\\|^2` | `CONCEPTUAL_SOURCE` |  |
| 269 | Appendix A: Equivariance Proof | **12** | `m'_{ij} = \phi_e \left( h_i^l, h_j^l, \\|Qx_i^l + g - (Qx_j^l + g)\\|^2, a_{ij}  \right) = \phi_e \left( h_i^l, h_j^l, \\|x_i^l - x_j^l\\|^2, a_{ij…` | `NOT_USED` | proof step for the velocity variant |
| 273 | Appendix A: Equivariance Proof | **-** | `Qx_i^l + g + C \sum_{j 
eq i} (Qx_i^l + g - [Qx_j^l + g]) \phi_x(m_{ij}) = Qx_i^l + g + Q C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})` | `NOT_USED` |  |
| 276 | Appendix A: Equivariance Proof | **-** | `= Q \left[ x_i^l + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})  \right] + g = Qx_i^{l+1} + g` | `NOT_USED` |  |
| 281 | Appendix A: Equivariance Proof | **-** | `h_i^{l+1} = \phi_h(h_i^l, m_i)` | `NOT_USED` |  |
| 290 | Appendix B: Re-Formulation for Velocity Type I | **-** | `m_{ij} = \phi_e \left( h_i^l, h_j^l, \\|x_i^l - x_j^l\\|^2, a_{ij}  \right)` | `NOT_USED` |  |
| 291 | Appendix B: Re-Formulation for Velocity Type I | **-** | `v_i^{l+1} = \phi_v(h_i^l) v_i^{init} + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})` | `NOT_USED` |  |
| 293 | Appendix B: Re-Formulation for Velocity Type I | **-** | `x_i^{l+1} = x_i^l + v_i^{l+1}` | `NOT_USED` |  |
| 294 | Appendix B: Re-Formulation for Velocity Type I | **-** | `m_i = \sum_{j 
eq i} m_{ij}` | `NOT_USED` |  |
| 296 | Appendix B: Re-Formulation for Velocity Type I | **-** | `h_i^{l+1} = \phi_h(h_i^l, m_i)` | `NOT_USED` |  |
| 301 | B.1 Equivariance Proof for Velocity Type Input | **-** | `h^{l+1}, Qx^{l+1} + g, Qv^{l+1} = \text{EGCL}[h^l, Qx^l + g, Qv^{init}, E]` | `NOT_USED` |  |
| 305 | B.1 Equivariance Proof for Velocity Type Input | **-** | `\phi_v(h_i^l) Qv_i^{init} + C \sum_{j 
eq i} (Qx_i^l + g - [Qx_j^l + g]) \phi_x(m_{ij})` | `NOT_USED` |  |
| 307 | B.1 Equivariance Proof for Velocity Type Input | **-** | `= Q \phi_v(h_i^l) v_i^{init} + Q C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})` | `NOT_USED` |  |
| 309 | B.1 Equivariance Proof for Velocity Type Input | **12** | `= Q \left[ \phi_v(h_i^l) v_i^{init} + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})  \right] = Qv_i^{l+1}` | `NOT_USED` | proof step for the velocity variant |
| 314 | B.1 Equivariance Proof for Velocity Type Input | **-** | `Qx_i^l + g + Qv_i^{l+1} = Q(x_i^l + v_i^{l+1}) + g = Qx_i^{l+1} + g` | `NOT_USED` |  |
| 323 | Appendix C: Implementation Details | **-** | `\text{Input} \to \text{Linear} \to \text{Swish} \to \text{Linear} \to \text{Swish} \to \text{Output}` | `NOT_USED` |  |
| 325 | Appendix C: Implementation Details | **-** | `m_{ij} \to \text{Linear} \to \text{Swish} \to \text{Linear} \to \text{Output}` | `NOT_USED` |  |
| 327 | Appendix C: Implementation Details | **-** | `[h_i^l, m_i] \to \text{Linear} \to \text{Swish} \to \text{Linear} \to \text{Addition}(h_i^l) \to h_i^{l+1}` | `NOT_USED` |  |
| 369 | E.1 Invariance of Distance Norms under E(n) | **-** | `\ell_2(Qx_i + t, Qx_j + t) = \sqrt{(Qx_i + t - [Qx_j + t])^T (Qx_i + t - [Qx_j + t])}` | `NOT_USED` |  |
| 370 | E.1 Invariance of Distance Norms under E(n) | **13** | `= \sqrt{(Q(x_i - x_j))^T Q(x_i - x_j)} = \sqrt{(x_i - x_j)^T Q^T Q (x_i - x_j)} = \ell_2(x_i, x_j)` | `NOT_USED` | E.1 distance-norm invariance under E(n) |
| 382 | E.2 Uniqueness of Distance Norms Representatio | **-** | `\tilde{x}_i^T \tilde{x}_i - 2 \tilde{x}_i^T \tilde{x}_j + \tilde{x}_j^T \tilde{x}_j = \ell_2(\tilde{x}_i, \tilde{x}_j)^2 = \ell_2(\tilde{y}_i, \til…` | `NOT_USED` |  |
| 386 | E.2 Uniqueness of Distance Norms Representatio | **14** | `\langle \tilde{x}_i, \tilde{x}_j 
angle = \langle \tilde{y}_i, \tilde{y}_j 
angle` | `CONCEPTUAL_SOURCE` |  |
| 394 | E.2 Uniqueness of Distance Norms Representatio | **-** | `\\| \sum_i c_i \tilde{x}_i \\|^2 = \\| \sum_i c_i \tilde{y}_i \\|^2 \quad (*)` | `NOT_USED` |  |
| 398 | E.2 Uniqueness of Distance Norms Representatio | **-** | `\\| \tilde{y}_i - A\tilde{x}_i \\|_2^2 = \\| \tilde{y}_i - \sum_j c_j y_{i_j} \\|_2^2` | `NOT_USED` |  |
| 399 | E.2 Uniqueness of Distance Norms Representatio | **-** | `= \langle \tilde{y}_i, \tilde{y}_i 
angle - 2 \langle \tilde{y}_i, \sum_j c_j y_{i_j} 
angle + \langle \sum_j c_j y_{i_j}, \sum_j c_j y_{i_j} 
angle` | `NOT_USED` |  |
| 403 | E.2 Uniqueness of Distance Norms Representatio | **15** | `\stackrel{(*)}{=} \langle \tilde{x}_i, \tilde{x}_i 
angle - 2 \langle \tilde{x}_i, \sum_j c_j x_{i_j} 
angle + \langle \sum_j c_j x_{i_j}, \sum_j c…` | `CONCEPTUAL_SOURCE` |  |
| 410 | E.2 Uniqueness of Distance Norms Representatio | **16** | `\langle A x_{i_j}, A x_{i_k} 
angle = \langle y_{i_j}, y_{i_k} 
angle = \langle x_{i_j}, x_{i_k} 
angle` | `NOT_USED` | graph autoencoder / QM9 task head |