> # DERIVED DOCUMENT — NON-AUTHORITATIVE FOR FORMULAS
>
> Converted from the published PDF. The converter read LaTeX backslash sequences
> as C string escapes, so `\frac` arrived as a form feed plus "rac", `\rangle` as
> a carriage return plus "angle", and `\neq` as a **real newline** — which is a
> legal character, so a control-byte census cannot find it. Equation numbering,
> dimensions and symbols are all unreliable here.
>
> **Authority for every formula, dimension, symbol and equation number is the
> authors' arXiv TeX source** under `docs/paper/*_source/`, inventoried in
> `docs/audit/A_FORMULA_INVENTORY.md`.
>
> This file is kept only as a convenience copy for prose search.

# MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation

**Zhenyu Pan**  
Northwestern University  
*zhenyupan@u.northwestern.edu*  

**Yucheng Lu**  
New York University  
*yuchenglu@nyu.edu*  

**Han Liu**  
Northwestern University  
*hanliu@northwestern.edu*  

---

## Abstract
We present **MetaFind**, a scene-aware tri-modal compositional retrieval framework designed to enhance scene generation in the metaverse by retrieving 3D assets from large-scale repositories. MetaFind addresses two core challenges:
1. **Inconsistent asset retrieval** that overlooks spatial, semantic, and stylistic constraints.
2. **The absence of a standardized retrieval paradigm** specifically tailored for 3D asset retrieval, as existing approaches mainly rely on general-purpose 3D shape representation models.

Our key innovation is a flexible retrieval mechanism that supports arbitrary combinations of text, image, and 3D modalities as queries, enhancing spatial reasoning and style consistency by jointly modeling object-level features (including appearance) and scene-level layout structures. Methodologically, MetaFind introduces a plug-and-play equivariant layout encoder **ESSGNN** that captures spatial relationships and object appearance features, ensuring retrieved 3D assets are contextually and stylistically coherent with the existing scene, regardless of coordinate frame transformations. The framework supports iterative scene construction by continuously adapting retrieval results to current scene updates. Empirical evaluations demonstrate the improved spatial and stylistic consistency of MetaFind in various retrieval tasks compared to baseline methods.

---

## 1. Introduction
This work introduces **MetaFind**, a novel scene-aware 3D retrieval framework designed to facilitate coherent scene generation within the metaverse by retrieving 3D assets from extensive repositories. Effective scene generation heavily relies on retrieving relevant, consistent, and contextually appropriate 3D assets; however, current methods face significant limitations, primarily due to two key challenges.

1. **Neglect of Contextual Constraints:** Existing retrieval frameworks often overlook critical factors such as spatial relationships, semantic coherence, and stylistic consistency, leading to retrieved assets that are visually and contextually incongruous when integrated into complex scenes.
2. **Lack of Standardized Paradigms:** Unlike well-established retrieval paradigms in natural language processing (NLP), such as Dense Passage Retrieval (DPR)—which introduced a generalizable dual-encoder architecture—there is currently no standardized retrieval paradigm explicitly tailored to the requirements and characteristics of 3D asset retrieval. Recent retrieval depends on generic 3D shape representation models, which fail to capture scene-specific contextual and stylistic nuances essential for coherent scene layout.

Recent approaches try to address these challenges by introducing various strategies. Early efforts enhance retrieval through 3D representations, focusing on object-level geometric features. Subsequent studies address cross-domain retrieval limitations through advanced techniques. Methods like SPL leverage domain alignment strategies, minimizing inter-domain discrepancies. UCD proposes sample-level weighting combined with domain and class alignment mechanisms, achieving improved performance but still relying on labeled data and introducing prediction bias. More recently, S2Mix and SCA3D introduce style fusion layers and cross-modal data augmentation techniques to enhance retrieval performance.

Despite these improvements, the current approaches are limited as they mainly consider object-centric features without adequately capturing crucial spatial, contextual, and scene-level relationships. Furthermore, they only support single-modality queries (3D-to-3D, text-to-3D, or image-to-3D), lacking the flexibility to handle compositional queries across multiple modalities.

To address these limitations, MetaFind introduces a dual-tower retrieval framework that integrates fine-grained object-level semantics with global scene-level spatial reasoning to enable context-aware, multimodal 3D asset retrieval. Unlike prior methods that only rely on object-centric cues (images or 3D shapes or text descriptions), MetaFind incorporates the spatial background by modeling the current scene layout as a structured graph. This layout-aware design allows the retriever to reason about placement constraints, positional dependencies, and contextual fit, enhancing spatial, semantic, and stylistic consistency. Moreover, MetaFind supports flexible multimodal queries, where the input can be any combination of text, image, point cloud, and layout context. This compositional design ensures robustness under missing modality conditions and adaptability to diverse use cases, including interactive scene editing, layout-conditioned asset generation, and large-scale virtual environment construction.

MetaFind builds upon ULIP-2, a tri-modal learning framework that aligns text, image, and point cloud into a shared embedding space. We adopt a dual-encoder architecture, where the query encoder flexibly encodes any user-provided modality combination, and the gallery encoder precomputes embeddings for all 3D assets to enable efficient retrieval. To supervise this alignment, we annotate 48K 3D assets from the Objaverse-LVIS subset, each rendered from 11 views and processed with GPT-4o to generate structured text descriptions.

For layout-level reasoning, we introduce the **Equivariant Spatial-Semantic Graph Neural Network (ESSGNN)**, an EGNN-based encoder designed to model rooms as graphs where nodes represent existing objects with 3D coordinates and text features and edges reflect spatial-semantic relationships. Unlike standard GNNs, ESSGNN maintains equivariance to rotation and translation by separating spatial and semantic channels, ensuring that scene embeddings remain stable across coordinate shifts and alignments—an essential property for robust layout modeling in unnormalized or dynamic environments. This encoder is trained on ProcTHOR, which contains over 10,000 generated houses. The ESSGNN outputs a layout context vector, which is fused with the query embedding to produce a layout-aware retrieval representation.

We adopt a two-stage training strategy:
1. **Pretraining** on object-level data for cross-modal grounding.
2. **Fine-tuning** on room-level scenes for layout-aware adaptation.

This architecture ensures strong generalization, modularity, and robustness across complex retrieval conditions.

### Key Contributions:
1. We present **MetaFind**, a novel layout-aware multimodal 3D asset retrieval framework tailored for coherent scene generation, which jointly considers object-level features and scene-level spatial context.
2. We introduce a plug-and-play **ESSGNN** layout encoder that models the evolving scene as a structured graph, capturing spatial relationships, contextual dependencies, and semantic attributes to guide retrieval decisions, with built-in $SE(3)$ equivariance to prevent degradation under arbitrary scene rotations or global shifts in coordinate systems.
3. We design MetaFind to support **flexible and robust multimodal querying**, allowing arbitrary combinations of multi-modalities as input, enabling strong performance under diverse and incomplete input conditions.
4. We demonstrate through comprehensive experiments that MetaFind outperforms baselines in both standard retrieval and layout-aware scene construction, and that our proposed **iterative retrieval pipeline** enhances contextual consistency and realism compared to current methods.

---

## 2. Methodology

### 2.1 Task Definition
We aim to accurately retrieve contextually coherent 3D assets from a large-scale repository, given a user query and optional existing scene layout information. Formally, our retrieval task is defined as follows: given an input query $Q = \{q_{\text{text}}, q_{\text{img}}, q_{\text{pc}}, q_{\text{layout}}\}$, which may include text $q_{\text{text}}$, images $q_{\text{img}}$, 3D point clouds $q_{\text{pc}}$, and optionally layout context $q_{\text{layout}}$, the system retrieves the asset $A^*$ from a pre-encoded asset database $\mathcal{A}$:

$$A^* = \operatorname{argmax}_{A \in \mathcal{A}} \operatorname{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A)) \quad (1)$$

where $f_{\text{query}}$ and $f_{\text{gallery}}$ represent the query and gallery encoders, and $\operatorname{sim}(\cdot, \cdot)$ denotes the similarity function. The task is challenging due to the multimodal nature of user queries, partial modality absence, and the necessity for accurate layout awareness to ensure spatial coherence and realism.

### 2.2 Method Overview
MetaFind is a dual-tower retrieval framework consisting of a query encoder and a gallery encoder, both leveraging the ULIP-2 embedding backbone.
* **Gallery Encoder:** Precomputes embeddings for assets using three available modalities, which are then stored for efficient retrieval.
* **Query Encoder:** Designed to flexibly handle arbitrary combinations of modalities and, optionally, layout information—accommodating partial modality absence through a modality-aware fusion strategy.

Each available modality is independently encoded using the ULIP-2 backbone, and these modality embeddings are subsequently integrated via a fusion layer (e.g., mean pooling, an MLP, or a Transformer-based module), generating a unified representation. Furthermore, the query encoder optionally integrates a layout encoder (ESSGNN) to capture spatial context from the existing scene layout. The layout is modeled as a structured graph with nodes representing placed objects (each described by spatial coordinates and semantic embeddings) and edges capturing spatial relationships. The layout encoder processes this graph to produce a context-aware layout vector, enhancing the spatial reasoning capability of the retrieval process. Its equivariant property ensures stable and generalizable scene embeddings under varying coordinate frames and unnormalized layouts common in open-world environments.

Our training protocol involves two stages:
1. **Cross-Modal Alignment Pretraining:** Train the query and gallery encoders to learn fundamental multimodal embedding alignment without spatial context.
2. **Layout-Aware Fine-Tuning:** Fine-tune the query encoder—particularly the fusion module and the layout encoder—using layout-aware room-level datasets with adaptive freezing strategies.

### 2.3 Data Preparation
Our methodology requires prepared datasets at both object and scene levels to support multimodal and layout-aware retrieval tasks:
* **Object-Level Dataset:** We utilize the **Objaverse-LVIS** dataset, which comprises approximately 48,000 distinct 3D assets. Each asset is rendered from 11 orthogonal viewpoints and annotated using GPT-4o. These annotations provide rich textual descriptions detailing attributes such as object category, size dimensions, materials, and placement constraints.
* **Scene-Level Dataset:** We leverage the **ProcTHOR** dataset, which includes over 10,000 generated houses constructed from a curated collection of more than 3,000 unique assets. Each room configuration provides precise spatial coordinates and comprehensive semantic metadata for each asset, enabling the extraction of structured graphs representing object-level placements and spatial relationships.

The extracted structured scene graphs include two types of edges:
1. **Physical-relation edges** that capture spatial dependencies (e.g., "cup on table").
2. **Semantic-relation edges** that capture functional or contextual associations (e.g., "microscope–lab bench"), obtained by prompting an LLM on object pairs.

This dual-edge design encodes both physical layout and high-level semantics, enhancing retrieval and layout reasoning.

### 2.4 Dual-Tower Architecture and Fusion Design
While prior works typically align 3D encoders to a fixed CLIP embedding space by freezing pretrained text and image encoders, MetaFind adopts a more flexible dual-tower design. It enables context-aware, multi-modal queries by training a dedicated query encoder that fuses arbitrary modality subsets.

Each tower leverages ULIP-2 to independently encode available modalities (text, images, and point clouds). A modality-aware fusion module combines these modality embeddings via one of several strategies, such as mean pooling, MLP, masked MLP, gated fusion, or Transformer-based fusion. The gallery encoder is modality-complete and frozen after pretraining, while the query encoder remains flexible: it accepts any subset of modalities and can be augmented with a layout-aware vector extracted using ESSGNN.

### 2.5 ESSGNN: Scene-Aware Equivariant Graph Encoder
The **Equivariant Spatial-Semantic Graph Neural Network (ESSGNN)** is proposed to encode 3D scene layouts in a way that is both spatially grounded and semantically expressive. ESSGNN is designed to maintain equivariance to $SE(3)$ transformations during message passing while incorporating semantic relationships between objects through learned edge representations.

Standard Graph Attention Networks (GATs) are highly sensitive to global translation and scaling variations across scenes, resulting in unstable layout embeddings and poor generalization. These issues are especially prominent in open-world or metaverse environments, where object positions are defined in large and often unnormalized coordinate systems, with no guarantee that scenes are aligned or centered.

Motivated by Equivariant Graph Neural Networks (EGNNs) used in drug design, ESSGNN extends the EGNN formulation to incorporate semantic edge features in addition to geometric ones, allowing message passing to be informed not only by spatial proximity but also by functional or compositional relationships between objects. Given a scene graph $G = (V, E)$, each node $v_i \in V$ represents an object with 3D position $x_i \in \mathbb{R}^3$ and a text-derived feature $t_i \in \mathbb{R}^d$. The node feature is initialized as:

$$h_i^{(0)} = \operatorname{Concat}(x_i, t_i)$$

Edges in the graph include both spatial and semantic relationships. Spatial edges are extracted from physical layout constraints (e.g., adjacency, support), while semantic edges are generated by prompting an LLM with object descriptions to produce natural language relation sentences. These sentences are then encoded into dense vectors using a frozen text encoder (e.g., CLIP or BERT), resulting in edge embeddings $e_{ij}$ that carry functional and relational meaning.

The message-passing mechanism in ESSGNN follows a modified Equivariant Graph Convolutional Layer (EGCL) structure. For each layer $l$, node features and positions are updated as:

$$h_i^{l+1} = h_i^l + \sum_{j \in \mathcal{N}(i)} f_h(d_{ij}^l, h_i^l, h_j^l, e_{ij}; \theta_h) \quad (2)$$

$$x_i^{l+1} = x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \cdot f_x(d_{ij}^l, h_i^{l+1}, h_j^{l+1}, e_{ij}; \theta_x) \quad (3)$$

where $d_{ij}^l = \|x_i^l - x_j^l\|_2$ is the Euclidean distance between nodes, and $f_h : \mathbb{R}^{(2d+1+e)} \to \mathbb{R}^d$, $f_x : \mathbb{R}^{(2d+1+e)} \to \mathbb{R}^3$ are two learnable functions parameterized by $\theta_h$ and $\theta_x$, respectively, approximated using MLPs (where $e$ denotes the dimension of the semantic edge embedding $e_{ij}$). After $L$ layers, the node features are aggregated into a global layout embedding:

$$e_{\text{layout}} = \operatorname{Pooling}(\{h_i^{(L)}\})$$

This embedding is integrated into the query encoder to provide scene-aware conditioning. ESSGNN generalizes the original EGNN by introducing semantic-aware edge modulation, enabling it to operate on multi-relational graphs with heterogeneous object types.

Our model retains full $SE(3)$-equivariance concerning input transformations. Specifically, for any rotation operator $R \in SO(3)$ and translation vector $T \in \mathbb{R}^3$, the following condition holds:

$$(R x^{l+1} + T, h^{l+1}) = \text{ESSGNN}(R x^l + T, h^l, E) \quad (4)$$

### 2.6 Training Strategy
We adopt a two-stage training strategy:

#### Stage 1: Cross-Modal Alignment Pretraining
Both query and gallery encoders are trained on large-scale object-level data from Objaverse-LVIS, where each asset has full modality inputs. We introduce stochastic modality masking to simulate partial-modality queries: each modality in the query has a 30% probability of being independently masked. Rather than zero-padding, we apply masked embeddings to ensure flexibility. The gallery encoder is trained to be modality-complete, and both towers share the contrastive retrieval objective:

$$\mathcal{L}_{\text{pre}} = -\log \frac{\exp(\operatorname{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A)) / \tau)}{\sum_{A' \in \mathcal{B}} \exp(\operatorname{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A')) / \tau)} \quad (5)$$

where $\tau$ is a temperature hyperparameter and $\mathcal{B}$ denotes the gallery batch.

#### Stage 2: Layout-Aware Fine-Tuning
In the second training stage, we enhance the query encoder with spatial context derived from the current scene layout. Given available modality embeddings for text $e_{\text{text}}$, image $e_{\text{image}}$, and point cloud $e_{\text{pc}}$, along with the optional layout embedding $e_{\text{layout}}$ produced by the ESSGNN module, the final fused query representation is computed as:

$$e_{\text{query}} = \operatorname{Fusion}(e_{\text{text}}, e_{\text{image}}, e_{\text{pc}}) + \lambda \cdot e_{\text{layout}} \quad (6)$$

where $\lambda$ is a learnable scalar controlling the contribution of layout information. This residual design allows layout reasoning to enhance retrieval without disrupting the original embedding space.

To ensure robustness, we introduce stochastic scene dropout (30%) during training: the layout vector $e_{\text{layout}}$ is omitted in 30% of batches. Only the query-side fuser and the ESSGNN module are updated; the gallery encoder is frozen. We adopt a bidirectional contrastive learning objective to symmetrically align query and gallery embeddings:

$$\mathcal{L}_{\text{layout}}^{q2g} = -\log \frac{\exp(\operatorname{sim}(e_{\text{query}}, e_{\text{gallery}}) / \tau)}{\sum_{e'_{\text{gallery}} \in \mathcal{B}} \exp(\operatorname{sim}(e_{\text{query}}, e'_{\text{gallery}}) / \tau)} \quad (7a)$$

$$\mathcal{L}_{\text{layout}}^{g2q} = -\log \frac{\exp(\operatorname{sim}(e_{\text{gallery}}, e_{\text{query}}) / \tau)}{\sum_{e'_{\text{query}} \in \mathcal{B}} \exp(\operatorname{sim}(e_{\text{gallery}}, e'_{\text{query}}) / \tau)} \quad (7b)$$

The final loss is the average of the two directions:

$$\mathcal{L}_{\text{layout}} = \frac{1}{2} \left( \mathcal{L}_{\text{layout}}^{q2g} + \mathcal{L}_{\text{layout}}^{g2q} \right) \quad (8)$$

### 2.7 Inference and Iterative Composition
At inference time, all gallery asset embeddings are precomputed and cached. Given an input query, the query encoder generates a layout-aware embedding to retrieve the most contextually suitable asset.

To construct complete scenes, we deploy an iterative composition strategy shown in **Algorithm 1**.

```
Algorithm 1: Iterative Layout-Aware Scene Composition
---------------------------------------------------------------------------------------------------------
Require: Precomputed gallery embeddings E_gallery, initial scene graph G_0, asset query list {Q_1, Q_2, ..., Q_N}
1: Initialize scene graph G <- G_0
2: for i = 1 to N do
3:   Extract current layout embedding: e_layout <- ESSGNN(G)
4:   Encode available modalities of query Q_i: e_text, e_img, e_pc
5:   Fuse into layout-aware query: e_query <- Fusion(e_text, e_img, e_pc) + \lambda * e_layout
6:   Retrieve best-matching asset: A*_i <- argmax_{A in E_gallery} sim(e_query, e_gallery(A))
7:   Place A*_i into the scene, update scene graph: G <- G U {A*_i}
8: end for
9: return Final composed scene G
---------------------------------------------------------------------------------------------------------
```

Instead of retrieving all required objects independently in a single step, we retrieve and place one object at a time. This step-by-step process improves spatial coherence and contextual alignment, resulting in more realistic and visually harmonious scenes. When efficiency is prioritized, we use parallel retrieval or region-based decomposition (partitioning a room into regions, retrieving sequentially within each region, and processing regions in parallel).

---

## 3. Experiments

### 3.1 Experimental Setup
* **Datasets:** Object-level experiments are conducted on **Objaverse-LVIS** (48K unique assets). Scene-level layout-aware retrieval is conducted on **ProcTHOR-10K** (over 10,000 house layouts with 3,000+ unique assets). We allocate 80% of the data for training and reserve 20% for testing.
* **Baselines:** We compare MetaFind against:
  * **ULIP** (tri-modal single-tower model)
  * **OpenShape** (dual-tower contrastive model)
  * **SCA3D** (point cloud-text retrieval model)
  * **Uni3DL** and **Uni3D** (unified 3D-language-image models)
  * **OmniBind** (omni-modality space model)
* **Metrics:** We report top-$k$ retrieval accuracy (R@1, R@5) on Objaverse-LVIS. For scene-level performance, we evaluate structural coherence and stylistic consistency using a GPT-4o-based aesthetic and alignment evaluator, validated through expert human preference studies.

---

### 3.2 Retrieval Performance on Objaverse-LVIS
The evaluation focuses on the capability of MetaFind to support flexible, modality-compositional retrieval. All methods are evaluated under seven query conditions: text-only, image-only, point cloud-only (PC), text+image (T+I), text+point cloud (T+PC), image+point cloud (I+PC), and full (T+I+PC).

#### Table 1: Retrieval accuracy (R@1 / R@5) on Objaverse-LVIS under different query modality combinations.
| Method | Text Only | Image Only | PC Only | T + I | T + PC | I + PC | T + I + PC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ULIP** [30] | 0.1 / 0.9 | 0.1 / 1.3 | 97.9 / 99.4 | 0 / 0.3 | 33.9 / 58.0 | 22.6 / 41.6 | 6.4 / 15.9 |
| **OpenShape** [10] | 0.6 / 1.7 | 0.3 / 1.1 | 98.4 / 99.7 | 0 / 0.5 | 35.1 / 61.4 | 25.0 / 44.3 | 7.0 / 17.2 |
| **SCA3D** [19] | 6.9 / 10.4 | – | 98.1 / 99.3 | – | 39.7 / 65.2 | – | – |
| **Uni3DL** [9] | 4.5 / 9.2 | – | 98.5 / 99.8 | – | 37.4 / 63.9 | – | – |
| **Uni3D** [32] | 1.7 / 3.9 | 1.2 / 2.5 | 98.3 / 99.4 | 0.5 / 1.1 | 36.3 / 63.6 | 26.1 / 44.8 | 8.2 / 19.1 |
| **OmniBind (Base)** | 1.2 / 2.8 | 0.6 / 1.4 | 98.3 / 99.6 | 0 / 0.4 | 34.0 / 55.9 | 21.5 / 38.7 | 5.5 / 13.8 |
| **OmniBind (Large)** | 2.7 / 4.0 | 0.9 / 1.8 | 98.2 / 99.3 | 0.1 / 0.4 | 35.2 / 56.7 | 23.4 / 40.9 | 6.0 / 16.7 |
| **OmniBind (Full)** [28] | 5.3 / 11.7 | 2.3 / 3.5 | 99.0 / 99.7 | 0.5 / 1.2 | 37.5 / 60.8 | 27.5 / 46.4 | 11.9 / 23.4 |
| **MetaFind w/o ESSGNN** | **13.8 / 23.1** | **11.7 / 19.2** | 75.1 / 78.0 | **17.2 / 21.8** | **44.5 / 71.3** | **45.8 / 73.1** | **51.7 / 76.5** |
| **MetaFind w/ ESSGNN** | 11.3 / 21.5 | 10.5 / 15.9 | 63.2 / 66.5 | 15.9 / 20.3 | 41.2 / 68.8 | 42.0 / 70.4 | 48.2 / 74.9 |

*Note: In baseline models, "PC Only" performance reflects retrieval using identical embeddings for both query and gallery, leading to inflated accuracy. MetaFind's dual-tower framework introduces cross-modality retrieval, causing more realistic "PC Only" numbers.*

Integrating ESSGNN introduces a temporary and explainable trade-off between object-level precision and scene-level coherence due to feature-attribution mismatch when evaluating on layout-free datasets. A practical mitigation is to maintain two fusion heads, selected at inference by context availability.

---

### 3.3 Scene-Level Retrieval with Layout Context
We evaluate MetaFind on the scene generation pipeline of I-Design on a set of 200 randomly sampled scenes, rated on a scale from 1 (poor) to 5 (excellent) by GPT-4o and five expert human annotators.

#### Table 2: Scene-level quality comparison across four evaluation dimensions.
| Method | Aesthetic (GPT-4o / Human) | Color & Material (GPT-4o / Human) | Scene Coherence (GPT-4o / Human) | Realism & Geometry (GPT-4o / Human) |
| :--- | :---: | :---: | :---: | :---: |
| **ULIP** [30] | 2.91 / 3.02 | 2.84 / 2.97 | 2.76 / 2.89 | 2.70 / 2.81 |
| **OpenShape** [10] | 3.14 / 3.28 | 3.08 / 3.19 | 3.01 / 3.11 | 2.95 / 3.06 |
| **MetaFind w/o ESSGNN** | 3.42 / 3.55 | 3.31 / 3.41 | 3.26 / 3.33 | 3.22 / 3.30 |
| **MetaFind w/ ESSGNN** | **4.13 / 4.25** | **4.04 / 4.17** | **4.10 / 4.21** | **4.06 / 4.18** |

* **Room 1 ("classical-style lounge"):** Without ESSGNN, the scene suffers from inconsistent styles (e.g., metallic fireplace, mismatched furniture). With ESSGNN, the scene adopts a unified classical aesthetic with a dark-toned fireplace, matching sofa, and a layout suitable for group interaction.
* **Room 2 ("aged archive room"):** Without ESSGNN, modern office furniture and cluttered seating break the archive theme. With ESSGNN, compact wooden chairs are arranged around the table, fitting the aged archive context and improving usability.

---

### 3.4 Ablation Studies
We analyze the contribution of core architectural components and training strategies under a Text-Only setting.

#### Table 3: Ablation study (Text Only).
| Variant | R@1 (%) | Aesthetic (GPT-4o) | Scene Coherence (GPT-4o) |
| :--- | :---: | :---: | :---: |
| **MetaFind (Full, bidirectional) w/ iterative & ESSGNN** | 11.4 | 4.1 | 4.2 |
| *w/o iterative retrieval* | 11.3 | 4.0 | 4.1 |
| *w/o Layout Context* | 13.5 | 3.4 | 3.3 |
| *w/ Layout Context (GAT)* | 11.0 | 3.4 | 3.7 |
| **Fusion = Mean** | 9.4 | 3.2 | 3.5 |
| **Fusion = MLPs** | 9.9 | 3.3 | 3.5 |
| **Modality Dropout = 10%** | 7.3 | 3.4 | 3.5 |
| **Modality Dropout = 50%** | 13.2 | 3.1 | 3.2 |
| **Train fuser only** | 8.7 | 3.3 | 3.2 |
| **Padding missing modalities with 0** | 10.5 | 3.1 | 3.1 |

Key takeaways:
* **ESSGNN vs GAT:** GAT is sensitive to translation/scaling, whereas ESSGNN's equivariant property provides substantial improvements in aesthetic and scene coherence scores.
* **Iterative Retrieval:** Sequential retrieval improves scene-level coherence compared to parallel/non-iterative retrieval.
* **Modality Dropout:** A 30% dropout rate strikes the best balance. Lower rates lead to overfitting, whereas higher rates introduce instability.
* **Fusion Strategy:** Masked modality fusion outperformed zero-padding. Fine-tuning the entire encoder outperformed training the fuser only.

---

## 4. Summary, Limitations, and Future Work
We present MetaFind, a scene-aware, multimodal 3D asset retrieval framework that unifies object-level semantics and scene-level spatial reasoning through a dual-tower design and a plug-and-play ESSGNN layout encoder. MetaFind significantly improves scene coherence and realism. 

**Limitations:** Asset annotations rely on GPT-4o, which can introduce language bias, hallucinations, and occasional mislabeling, potentially affecting training and evaluation.

**Future Work:** We plan to incorporate real-world human-in-the-loop feedback for adaptive scene refinement and scale to open-world settings with dynamic object catalogs.

---

## 5. Acknowledgments
We gratefully acknowledge support from the NVIDIA Academic Grant ("Interactive Spatial Reasoning and 3D Scene Generation with RL-Enhanced VLMs") and the provision of cloud computing resources, which enabled systematic training and evaluation of our MetaFind and other baselines. This paper is a core component of that project. The views expressed are those of the authors and do not necessarily reflect those of NVIDIA.

---

## References
1. Ata Çelen, Guo Han, Konrad Schindler, Luc Van Gool, Iro Armeni, Anton Obukhov, and Xi Wang. I-design: Personalized llm interior designer. *arXiv preprint arXiv:2404.02838*, 2024.
2. Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. *In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 13142–13153, 2023.
3. Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Procthor: Large-scale embodied ai using procedural generation. *Advances in Neural Information Processing Systems*, 35:5982–5994, 2022.
4. Chuan Fang, Yuan Dong, Kunming Luo, Xiaotao Hu, Rakesh Shrestha, and Ping Tan. Ctrl-room: controllable text-to-3d room meshes generation with layout constraints. *arXiv preprint arXiv:2310.03602*, 2023.
5. Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. *Advances in Neural Information Processing Systems*, 36:18225–18250, 2023.
6. Xinwei Fu, Dan Song, Yue Yang, Yuyi Zhang, and Bo Wang. S2mix: Style and semantic mix for cross-domain 3d model retrieval. *Journal of Visual Communication and Image Representation*, 107:104390, 2025.
7. Haiyun Guo, Jinqiao Wang, Min Xu, Zheng-Jun Zha, and Hanqing Lu. Learning multi-view deep features for small object retrieval in surveillance scenarios. *In Proceedings of the 23rd ACM international conference on Multimedia*, pages 859–862, 2015.
8. Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. *In EMNLP (1)*, pages 6769–6781, 2020.
9. Xiang Li, Jian Ding, Zhaoyang Chen, and Mohamed Elhoseiny. Uni3dl: Unified model for 3d and language understanding. *arXiv preprint arXiv:2312.03026*, 2023.
10. Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao Su. Openshape: Scaling up 3d shape representation towards open-world understanding. *Advances in neural information processing systems*, 36:44860–44879, 2023.
11. Zhenyu Pan, Rongyu Cao, Yongchang Cao, Yingwei Ma, Binhua Li, Fei Huang, Han Liu, and Yongbin Li. Codev-bench: How do llms understand developer-centric code completion?, 2024.
12. Zhenyu Pan and Han Liu. Metaspatial: Reinforcing 3d spatial reasoning in vlms for the metaverse. *arXiv preprint arXiv:2503.18470*, 2025.
13. Zhenyu Pan, Haozheng Luo, Manling Li, and Han Liu. Chain-of-action: Faithful and multimodal question answering through large language models. *arXiv preprint arXiv:2403.17359*, 2024.
14. Zhenyu Pan, Haozheng Luo, Manling Li, and Han Liu. Conv-coa: Improving open-domain question answering in large language models via conversational chain-of-action, 2024.
15. Zhenyu Pan, Xuefeng Song, Yunkun Wang, Rongyu Cao, Binhua Li, Yongbin Li, and Han Liu. Do code llms understand design patterns? *arXiv preprint arXiv:2501.04835*, 2025.
16. Zhenyu Pan, Yiting Zhang, Zhuo Liu, Yolo Yunlong Tang, Zeliang Zhang, Haozheng Luo, Yuwei Han, Jianshu Zhang, Dennis Wu, Hong-Yu Chen, Haoran Lu, Haoyang Fang, Manling Li, Chenliang Xu, Philip S. Yu, and Han Liu. Advevo-marl: Shaping internalized safety through adversarial co-evolution in multi-agent reinforcement learning, 2025.
17. Zhenyu Pan, Yiting Zhang, Yutong Zhang, Jianshu Zhang, Haozheng Luo, Yuwei Han, Dennis Wu, Hong-Yu Chen, Philip S. Yu, Manling Li, and Han Liu. Evo-marl: Co-evolutionary multi-agent reinforcement learning for internalized safety, 2025.
18. Zhenyu Pan, Yutong Zhang, Jianshu Zhang, Haoran Lu, Haozheng Luo, Yuwei Han, Philip S. Yu, Manling Li, and Han Liu. Fairreason: Balancing reasoning and social bias in mllms, 2025.
19. Junlong Ren, Hao Wu, Hui Xiong, and Hao Wang. Sca3d: Enhancing cross-modal 3d retrieval via 3d shape and caption paired data augmentation. *arXiv preprint arXiv:2502.19128*, 2025.
20. Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, and Kamal Rahimi Malekshan. Clip-forge: Towards zero-shot text-to-shape generation. *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18603–18613, 2022.
21. Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. *In International conference on machine learning*, pages 9323–9332. PMLR, 2021.
22. Jonas Schult, Sam Tsai, Lukas Höllein, Bichen Wu, Jialiang Wang, Chih-Yao Ma, Kunpeng Li, Xiaofang Wang, Felix Wimbauer, Zijian He, et al. Controlroom3d: Room generation using semantic proxy rooms. *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6201–6210, 2024.
23. Erzhuo Shao, Yifang Wang, Yifan Qian, Zhenyu Pan, Han Liu, and Dashun Wang. Sciscigpt: Advancing human-ai collaboration in the science of science. *arXiv preprint arXiv:2504.05559*, 2025.
24. Dan Song, Tian-Bao Li, Wen-Hui Li, Wei-Zhi Nie, Wu Liu, and An-An Liu. Universal cross-domain 3d model retrieval. *IEEE Transactions on Multimedia*, 23:2721–2731, 2020.
25. Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. Multi-view convolutional neural networks for 3d shape recognition. *In Proceedings of the IEEE international conference on computer vision*, pages 945–953, 2015.
26. Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. Layoutvlm: Differentiable optimization of 3d layout via vision-language models. *arXiv preprint arXiv:2412.02193*, 2024.
27. Qian Wang and Toby Breckon. Unsupervised domain adaptation via structured prediction based selective pseudo-labeling. *In Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 6243–6250, 2020.
28. Zehan Wang, Ziang Zhang, Hang Zhang, Luping Liu, Rongjie Huang, Xize Cheng, Hengshuang Zhao, and Zhou Zhao. Omnibind: Large-scale omni multimodal representation via binding spaces, 2024.
29. Hao Wu, Ruochong Li, Hao Wang, and Hui Xiong. Com3d: Leveraging cross-view correspondence and cross-modal mining for 3d retrieval. *In 2024 IEEE International Conference on Multimedia and Expo (ICME)*, pages 1–6. IEEE, 2024.
30. Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, et al. Ulip-2: Towards scalable multimodal pre-training for 3d understanding. *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 27091–27101, 2024.
31. Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. *In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8552–8562, 2022.
32. Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. *arXiv preprint arXiv:2310.06773*, 2023.

---

## Appendices

### Appendix A: Broader Impacts
MetaFind facilitates accessible and coherent 3D scene generation, which can benefit fields like virtual reality, education, and game design. By supporting flexible multimodal queries, it lowers the barrier for non-experts to build rich virtual environments. However, risks include potential misuse in generating misleading content, propagation of bias from training data, and intellectual property concerns tied to retrieved assets. We recommend responsible dataset curation and human oversight to ensure ethical deployment.

### Appendix B: Related Work
3D scene generation serves as the broader task context of our work, encompassing both generative and retrieval-based approaches to assembling realistic virtual environments. Within this paradigm, 3D object retrieval plays a critical role by providing high-quality assets that satisfy semantic, stylistic, and spatial constraints.

#### B.1 3D Scene Generation Paradigms
Recent progress in 3D scene generation follows two directions:
1. **Generative Synthesis:** Methods that synthesize entire 3D scenes in mesh, voxel, or neural field formats. While promising, these methods struggle with ensuring object-level realism or semantic fidelity.
2. **Layout Composition via Retrieval:** Frame scene generation as a layout composition task using retrieved assets from large-scale 3D repositories. Methods like LayoutGPT and I-Design employ LLMs as planners to generate layouts from text descriptions. LayoutVLM improves physical plausibility through differentiable rendering optimization and layout supervision. MetaSpatial addresses VLM spatial reasoning and data limitations via a reinforcement learning-based framework optimizing layouts in real time.

While MetaSpatial improves layout generation, MetaFind bridges the gap in the retrieval mechanism itself by explicitly incorporating layout context into the retrieval loop, supporting arbitrary modality combinations, and iteratively selecting assets via ESSGNN.

#### B.2 3D Object Retrieval
3D object retrieval historically focused on visual and geometric alignment with semantic queries. Early models like PointCLIP and CLIP-Forge aligned 2D/3D pairs, while ULIP and OpenShape extended this to tri-modal alignment (text, image, point cloud) in a shared latent space. However, they lack explicit mechanisms to handle arbitrary modality combinations under missing input scenarios, or to incorporate spatial constraints. MetaFind's ESSGNN-based layout encoder resolves this, offering robust, iterative, and context-aware selection that supports spatial realism.

### Appendix C: Equivariance Proof of ESSGNN - Extension to Semantic Embedding
We prove that our ESSGNN maintains $SE(3)$ equivariance in 3D space. While the original EGNN formulation allows discrete, task-specific edge features, ESSGNN introduces edge embeddings $e_{ij}$ derived from LLM-generated natural language relation descriptions, encoded via a frozen text encoder. These semantic edge embeddings are invariant to the input node positions $x$, as they depend solely on object-level text descriptions. Thus, the mathematical property required for equivariance—the independence of $e_{ij}$ from $x$—remains satisfied, and the original proof structure holds.

Specifically, we show that for any translation vector $g \in \mathbb{R}^3$ and any orthogonal rotation transformation $Q \in \mathbb{R}^{3 \times 3}$, the model satisfies:

$$(Q x^{l+1} + g, h^{l+1}) = \text{ESSGNN}(Q x^l + g, h^l, E) \quad (9)$$

where $x^l$ and $h^l$ are the positions and features of all nodes at layer $l$, and $E$ contains edge features including learned semantic embeddings $e_{ij}$. Assuming that $h^0$ is invariant to $SE(3)$ transformations on $x$, and that semantic edge embeddings $e_{ij}$ are derived solely from object-level textual descriptions and thus independent of spatial coordinates, the pairwise edge message is:

$$m_{ij} = \phi_e \left( h_i^l, h_j^l, \|x_i^l - x_j^l\|^2, e_{ij} \right) \quad (10)$$

Under translation and rotation $x_i^l \mapsto Q x_i^l + g$, the Euclidean distance becomes:

$$\|Q x_i^l + g - (Q x_j^l + g)\|^2 = \|Q(x_i^l - x_j^l)\|^2 = \|x_i^l - x_j^l\|^2 \quad (11)$$

Hence, the edge message is preserved:

$$m'_{ij} = \phi_e \left( h_i^l, h_j^l, \|Q x_i^l + g - Q x_j^l - g\|^2, e_{ij} \right) = m_{ij} \quad (12)$$

The coordinate position update in ESSGNN is:

$$x_i^{l+1} = x_i^l + \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij}) \quad (13)$$

Applying the rotation and translation transformation:

$$Q x_i^l + g + \sum_{j \neq i} (Q x_i^l + g - Q x_j^l - g) \cdot \phi_x(m_{ij}) = Q x_i^l + g + Q \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij})$$

$$= Q \left[ x_i^l + \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij}) \right] + g = Q x_i^{l+1} + g$$

Thus, the coordinate update is $SE(3)$ equivariant.

For the feature update:

$$h_i^{l+1} = h_i^l + \sum_{j \neq i} \phi_h(m_{ij}) \quad (14)$$

Since $m_{ij}$ is invariant to transformations of $x$, and both $h_i^l, h_j^l$ and $e_{ij}$ are independent of the global pose, the feature update is invariant to $SE(3)$ transformations of positions.

Therefore, the ESSGNN update satisfies:

$$(Q x^{l+1} + g, h^{l+1}) = \text{ESSGNN}(Q x^l + g, h^l, E) \quad (15)$$

This completes the proof that ESSGNN preserves $SE(3)$ equivariance despite the inclusion of semantic edge embeddings.

### Appendix D: Experimental Analysis
* **Room 1:** Without ESSGNN, the room lacks stylistic coherence—the metallic fireplace and mismatched furniture deviate from the classical theme. With ESSGNN, the scene adopts a unified classical aesthetic with a dark-toned fireplace, matching sofa, and bookshelf.
* **Room 2:** Without ESSGNN, modern office furniture and cluttered seating break the archive theme and hinder functionality. With ESSGNN, compact wooden chairs are arranged around the table, better fitting the aged archive context and improving usability.
