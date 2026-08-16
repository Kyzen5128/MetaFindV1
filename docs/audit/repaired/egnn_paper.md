# E(n) Equivariant Graph Neural Networks

**Victor Garcia Satorras**  
University of Amsterdam, Netherlands  
*v.garciasatorras@uva.nl*  

**Emiel Hoogeboom**  
University of Amsterdam, Netherlands  
*e.hoogeboom@uva.nl*  

**Max Welling**  
University of Amsterdam, Netherlands  
*m.welling@uva.nl*  

---

## Abstract
This paper introduces a new model to learn graph neural networks equivariant to rotations, translations, reflections, and permutations called **E(n)-Equivariant Graph Neural Networks (EGNNs)** [1]. In contrast with existing methods, our work does not require computationally expensive higher-order representations in intermediate layers while it still achieves competitive or better performance [1]. In addition, whereas existing methods are limited to equivariance on 3-dimensional spaces, our model is easily scaled to higher-dimensional spaces [1]. We demonstrate the effectiveness of our method on dynamical systems modelling, representation learning in graph autoencoders, and predicting molecular properties [1].

---

## 1. Introduction
Although deep learning has largely replaced hand-crafted features, many advances are critically dependent on inductive biases in deep neural networks [2]. An effective method to restrict neural networks to relevant functions is to exploit the symmetry of problems by enforcing equivariance with respect to transformations from a certain symmetry group [2]. Notable examples are translation equivariance in Convolutional Neural Networks and permutation equivariance in Graph Neural Networks [2].

Many problems exhibit 3D translation and rotation symmetries, such as point clouds, 3D molecular structures, or N-body particle simulations [3]. The group corresponding to these symmetries is named the Euclidean group: $SE(3)$ or when reflections are included $E(3)$ [3]. It is often desired that predictions on these tasks are either equivariant or invariant with respect to $E(3)$ transformations [3].

Recently, various forms and methods to achieve $E(3)$ or $SE(3)$ equivariance have been proposed [4]. Many of these works achieve innovations in studying types of higher-order representations for intermediate network layers [4]. However, the transformations for these higher-order representations require coefficients or approximations that can be expensive to compute [4]. Additionally, in practice for many types of data, the inputs and outputs are restricted to scalar values (referred to as type-0 in literature) and 3D vectors (referred to as type-1 in literature) [4].

In this work, we present a new architecture that is translation, rotation, and reflection equivariant ($E(n)$), and permutation equivariant with respect to an input set of points [5]. Our model is simpler than previous methods in that it does not require spherical harmonics while it can still achieve competitive or better results [5]. In addition, equivariance in our model is not limited to 3-dimensional space and can be scaled to larger dimensional spaces without a significant increase in computation [5].

We evaluate our method in modelling dynamical systems, representation learning in graph autoencoders, and predicting molecular properties in the QM9 dataset, reporting the best or very competitive performance in all three experiments [6].

---

## 2. Background

### 2.1 Equivariance
Let $T_g : X \to X$ be a set of transformations on $X$ for the abstract group $g \in G$ [6]. We say a function $\phi : X \to Y$ is equivariant to $g$ if there exists an equivalent transformation on its output space $S_g : Y \to Y$ such that [6]:

$$\phi(T_g(x)) = S_g(\phi(x)) \quad (1)$$

As a practical example, let $\phi(\cdot)$ be a non-linear function, $x = (x_1, \dots, x_M) \in \mathbb{R}^{M \times n}$ an input set of $M$ point clouds embedded in an $n$-dimensional space, $\phi(x) = y \in \mathbb{R}^{M \times n}$ the transformed set of point clouds, $T_g$ a translation on the input set $T_g(x) = x + g$, and $S_g$ an equivalent translation on the output set $S_g(y) = y + g$ [7]. If our transformation $\phi$ is translation equivariant, translating the input set and then applying the function delivers the same result as first running the function and then applying an equivalent translation to the output: $\phi(x+g) = \phi(x) + g$ [7].

We explore three types of equivariance on a set of particles $x$ [7]:
1. **Translation equivariance:** Translating the input by $g \in \mathbb{R}^n$ results in an equivalent translation of the output [8]. Letting $x+g$ be shorthand for $(x_1+g, \dots, x_M+g)$, then:
   $$y + g = \phi(x + g)$$
2. **Rotation (and reflection) equivariance:** For any orthogonal matrix $Q \in \mathbb{R}^{n \times n}$, letting $Qx$ be shorthand for $(Qx_1, \dots, Qx_M)$, rotating the input results in an equivalent rotation of the output [8]:
   $$Qy = \phi(Qx)$$
3. **Permutation equivariance:** Permuting the input results in the same permutation of the output [8]:
   $$P(y) = \phi(P(x))$$
   where $P$ is a permutation on the row indexes [8].

Velocities $v \in \mathbb{R}^{M \times n}$ are unaffected by translations, but they transform equivalently under rotation and permutation [9].

### 2.2 Graph Neural Networks
Graph Neural Networks are permutation equivariant networks that operate on graph structured data [9]. Given a graph $G = (V, E)$ with nodes $v_i \in V$ and edges $e_{ij} \in E$, we define a graph convolutional layer following the notation from Gilmer et al. (2017) as [9, 10]:

$$m_{ij} = \phi_e(h_i^l, h_j^l, a_{ij})$$
$$m_i = \sum_{j \in \mathcal{N}(i)} m_{ij}$$
$$h_i^{l+1} = \phi_h(h_i^l, m_i) \quad (2)$$

Where $h_i^l \in \mathbb{R}^{n_f}$ is the $n_f$-dimensional embedding of node $v_i$ at layer $l$, $a_{ij}$ represent the edge attributes, $\mathcal{N}(i)$ represents the set of neighbors of node $v_i$, and $\phi_e$ and $\phi_h$ are the edge and node operations approximated by Multilayer Perceptrons (MLPs) [10].

---

## 3. Equivariant Graph Neural Networks
We consider a graph $G = (V, E)$ with nodes $v_i \in V$ and edges $e_{ij} \in E$ [10]. In addition to the feature node embeddings $h_i \in \mathbb{R}^{n_f}$, we now also consider an $n$-dimensional coordinate $x_i \in \mathbb{R}^n$ associated with each of the graph nodes [10]. Our model preserves equivariance to rotations and translations on these coordinates $x_i$ and equivariance to permutations on the set of nodes $V$ [10].

Our **Equivariant Graph Convolutional Layer (EGCL)** takes as input the set of node embeddings $h^l = \{h_0^l, \dots, h_{M-1}^l\}$, coordinate embeddings $x^l = \{x_0^l, \dots, x_{M-1}^l\}$, and edge information $E = (e_{ij})$, and outputs a transformation on $h^{l+1}$ and $x^{l+1}$ [11]. Concisely, $h^{l+1}, x^{l+1} = \text{EGCL}[h^l, x^l, E]$ [11]. The equations defining this layer are [11]:

$$m_{ij} = \phi_e \left( h_i^l, h_j^l, \|x_i^l - x_j^l\|^2, a_{ij} \right) \quad (3)$$
$$x_i^{l+1} = x_i^l + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij}) \quad (4)$$
$$m_i = \sum_{j 
eq i} m_{ij} \quad (5)$$
$$h_i^{l+1} = \phi_h(h_i^l, m_i) \quad (6)$$

The main differences from standard GNNs are found in Equations (3) and (4) [11]. In Equation (3), we input the relative squared distance $\|x_i^l - x_j^l\|^2$ into the edge operation $\phi_e$ [11].

In Equation (4), we update the position of each particle $x_i$ as a vector field in a radial direction [12]. The position is updated by the weighted sum of all relative differences $(x_i - x_j)$ [12]. The weights are provided by $\phi_x : \mathbb{R}^{n_f} \to \mathbb{R}^1$, which takes as input the edge embedding $m_{ij}$ and outputs a scalar value [12]. $C$ is chosen to be $1/(M - 1)$, which normalizes the sum [12]. This equation preserves translation and rotation equivariance (proof in Appendix A) [12].

Equations (5) and (6) perform standard GNN aggregation and node updates, where we can limit the neighborhood to $j \in \mathcal{N}(i)$ if desired [13].

### 3.1 Analysis on E(n) Equivariance
Our model satisfies the following relation under $E(n)$ transformations (for any translation vector $g \in \mathbb{R}^n$ and orthogonal matrix $Q \in \mathbb{R}^{n \times n}$) [13]:

$$Qx^{l+1} + g, h^{l+1} = \text{EGCL}(Qx^l + g, h^l) \quad (14)$$

Intuitively, if $h^l$ is already $E(n)$ invariant, the resultant edge embedding $m_{ij}$ from Equation (3) will also be $E(n)$ invariant, because it only depends on $h^l$ and the squared distances $\|x_i^l - x_j^l\|^2$, which are $E(n)$ invariant [14]. Consequently, the coordinate update in Equation (4) transforms as a type-1 vector, and the feature update in Equation (6) remains invariant [14].

### 3.2 Extending EGNNs for Vector Type Representations
In scenarios where we must explicitly keep track of a particle's momentum or initial velocity $v^{init} 
eq 0$, we can include vector type representations by replacing Equation (4) with [15, 16]:

$$v_i^{l+1} = \phi_v(h_i^l) v_i^{init} + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij}) \quad (7a)$$
$$x_i^{l+1} = x_i^l + v_i^{l+1} \quad (7b)$$

where $\phi_v : \mathbb{R}^{n_f} \to \mathbb{R}^1$ scales the initial velocity [16]. If $v_i^{init} = 0$, both Equations (4) and (7) become mathematically equivalent [16].

### 3.3 Inferring the Edges
When an adjacency matrix is not provided, we can assume a fully connected graph or infer relations dynamically by modifying the aggregation operation as [17, 18]:

$$m_i = \sum_{j \in \mathcal{N}(i)} m_{ij} = \sum_{j 
eq i} e_{ij} m_{ij} \quad (8)$$

where $e_{ij} \approx \phi_{inf}(m_{ij})$ represents a soft estimation of edge existence [18]. $\phi_{inf} : \mathbb{R}^{n_f} \to [0, 1]^1$ is approximated by a linear layer followed by a sigmoid function [18]. This preserves the $E(n)$ properties because $m_{ij}$ is already $E(n)$ invariant [18].

---

## 4. Related Work
Group equivariant neural networks have shown substantial performance across geometric and physical tasks [21]. Many prior works achieve $E(3)$ or $SE(3)$ equivariance using spherical harmonics to construct a basis for transformation in higher-order representations (e.g., Thomas et al., 2018; Fuchs et al., 2020) [21]. However, spherical harmonics need to be recomputed and can be computationally expensive [21]. Other methods map kernels on Lie Algebra (Finzi et al., 2020) or restrict to positional data without feature propagation (Köhler et al., 2019) [21].

### Relationship with Existing Methods
Table 1 outlines the equations of EGNN alongside related methods under Gilmer's message passing framework [23].

#### Table 1: Comparison of different works under the message passing framework notation.
| Method | Edge Message ($m_{ij}$) | Aggregation ($m_i$, $\hat{m}_i$) | Node Update ($h_i^{l+1}$, $x_i^{l+1}$) | Equivariance Class |
| :--- | :---: | :---: | :---: | :---: |
| **GNN** [10] | $\phi_e(h_i^l, h_j^l, a_{ij})$ | $m_i = \sum_{j \in \mathcal{N}(i)} m_{ij}$ | $h_i^{l+1} = \phi_h(h_i^l, m_i)$ | Non-equivariant |
| **Radial Field** [24] | $\phi_{rf}(\|r_{ij}^l\|) r_{ij}^l$ | $m_i = \sum_{j 
eq i} m_{ij}$ | $x_i^{l+1} = x_i^l + m_i$ | $E(n)$-Equivariant |
| **TFN** [24] | $\sum_k W^{l,k}(r_{ij}^l) h_i^{l,k}$ | $m_i = \sum_{j 
eq i} m_{ij}$ | $h_i^{l+1} = w_{ll} h_i^l + m_i$ | $SE(3)$-Equivariant |
| **Schnet** [25] | $\phi_{cf}(\|r_{ij}^l\|) \phi_s(h_j^l)$ | $m_i = \sum_{j 
eq i} m_{ij}$ | $h_i^{l+1} = \phi_h(h_i^l, m_i)$ | $E(n)$-Invariant |
| **EGNN** [25] | $m_{ij} = \phi_e(h_i^l, h_j^l, \|r_{ij}^l\|^2, a_{ij})$<br>$\hat{m}_{ij} = r_{ij}^l \phi_x(m_{ij})$ | $m_i = \sum_{j 
eq i} m_{ij}$<br>$\hat{m}_i = C \sum_{j 
eq i} \hat{m}_{ij}$ | $h_i^{l+1} = \phi_h(h_i^l, m_i)$<br>$x_i^{l+1} = x_i^l + \hat{m}_i$ | **$E(n)$-Equivariant** |

*Note: $r_{ij}^l = (x_i^l - x_j^l)$ is the spatial difference vector [20].*

---

## 5. Experiments

### 5.1 Modelling a Dynamical System — N-Body System
We model a 3D extension of the Charged Particles N-body experiment (with $M=5$ particles) [27]. Particles carry positive or negative charges and move in a 3D space according to physical laws [27].

* **Dataset:** 3,000 training trajectories, 2,000 validation, and 2,000 testing [28]. Each trajectory runs for 1,000 timesteps sliced from a 5,000-timestep generation to avoid transient phase initialization issues [74]. Positions $p(0)$, initial velocities $v(0)$, and charges $c_i c_j$ are given [28]. The task is to predict the positions after 1,000 timesteps by optimizing the Mean Squared Error (MSE) [28].
* **Implementation:** We stack 4 layers of our velocity-augmented EGNN (Section 3.2) with a hidden dimension of 64 [29, 30]. We compare against GNN, Radial Field, TFN, and SE(3) Transformer baselines [29].

#### Table 2: Position estimation Mean Squared Error (MSE) and forward pass time (seconds) for batch size 100 on a GTX 1080 Ti.
| Method | MSE | Forward Time (s) |
| :--- | :---: | :---: |
| **Linear** | 0.0819 | **0.0001** |
| **SE(3) Transformer** | 0.0244 | 0.1346 |
| **Tensor Field Network** | 0.0155 | 0.0343 |
| **Graph Neural Network** | 0.0107 | 0.0032 |
| **Radial Field** | 0.0104 | 0.0039 |
| **EGNN** (Ours) | **0.0071** | 0.0062 |

EGNN significantly outperforms other equivariant and non-equivariant alternatives, reducing the error of the second-best method by **32%** while running highly efficiently without spherical harmonics [31].

* **Data Efficiency Analysis:** Sweeping training samples from 100 to 50,000 reveals that EGNN consistently outperforms Radial Field and GNNs [32]. It achieves superior data efficiency compared to GNNs by avoiding the need to generalize over global rotations and translations manually [32].

---

### 5.2 Graph Autoencoder
We construct an Equivariant Graph Autoencoder to embed featureless graphs into continuous latent node embeddings $z \in \mathbb{R}^{M \times n}$ while reconstructing the adjacency matrix $A$ [33, 35]. The decoder is defined as [36]:

$$\hat{A}_{ij} = g_e(z_i, z_j) = \frac{1}{1 + \exp(w \|z_i - z_j\|^2 + b)} \quad (9)$$

* **The Symmetry Problem:** Standard GCN encoders on featureless graphs (e.g., a cycle graph) produce identical node embeddings for topologically symmetric nodes, making adjacency reconstruction impossible [37]. Adding input Gaussian noise $h_i^0 \sim \mathcal{N}(0, \sigma I)$ breaks symmetry [38]. In our model, we input this noise as initial coordinates $x_0 \sim \mathcal{N}(0, \sigma I)$ and set $h^0 = \mathbf{1}$ [40, 42].
* **Datasets:** Community Small (12 to 20 nodes) and Erdos-Renyi (7 to 16 nodes, $p_e=0.25$) random graphs [41]. 5,000 training, 500 validation, and 500 testing graphs [41].

#### Table 4: Graph Autoencoding performance on Community Small and Erdos-Renyi datasets.
| Dataset | Metric | Baseline | GNN | Noise-GNN | Radial Field | EGNN (Ours) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Community Small** | BCE Loss | 31.79 | 6.75 | 3.32 | 9.22 | **2.14** |
| | % Error | - | 1.29 | 0.44 | 1.19 | **0.06** |
| | F1 Score | 0.000 | 0.980 | 0.993 | 0.981 | **0.999** |
| **Erdos & Renyi** | BCE Loss | 25.13 | 14.15 | 4.56 | 6.78 | **1.65** |
| | % Error | - | 4.62 | 1.25 | 1.63 | **0.11** |
| | F1 Score | 0.000 | 0.907 | 0.975 | 0.968 | **0.998** |

EGNN's equivariance to the input noise distribution enables almost optimal reconstruction ($0.06\%$ error on Community Small and $0.11\%$ on Erdos & Renyi) [44].

---

### 5.3 Molecular Data — QM9
We predict 12 chemical properties of small molecules in the QM9 dataset, containing up to 29 atoms per molecule [49]. Atoms are represented by static 3D coordinate positions $x_i^0$ and a 5D one-hot atom type embedding [49].

* **Dataset:** 100K training, 18K validation, and 13K testing molecules [50].
* **Implementation:** We deploy a 7-layer EGNN with 128 hidden features and the edge-inference module (Section 3.3) [51]. Since positions are static, we do not update coordinates $x_i$ during message passing, making our model functionally $E(n)$ invariant [51]. Node embeddings are aggregated using sum pooling followed by an MLP [51].

#### Table 3: Mean Absolute Error (MAE) for molecular property prediction on QM9.
| Property | Units | NMP | Schnet | Cormorant | L1Net | LieConv | DimeNet++* | TFN | SE(3)-Tr. | EGNN (Ours) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\alpha$ | $\text{bohr}^3$ | 0.092 | 0.235 | 0.085 | 0.088 | 0.084 | **0.044** | 0.223 | 0.142 | 0.071 |
| $\Delta \epsilon$ | $\text{meV}$ | 69 | 63 | 61 | 68 | 49 | **33** | 58 | 53 | 48 |
| $\epsilon_{\text{HOMO}}$ | $\text{meV}$ | 43 | 41 | 34 | 46 | 30 | **25** | 40 | 35 | 29 |
| $\epsilon_{\text{LUMO}}$ | $\text{meV}$ | 38 | 34 | 38 | 35 | 25 | **20** | 38 | 33 | 25 |
| $\mu$ | $\text{D}$ | 0.030 | 0.033 | 0.038 | 0.043 | 0.032 | 0.030 | 0.064 | 0.051 | **0.029** |
| $C_
u$ | $\text{cal/mol K}$| 0.040 | 0.033 | 0.026 | 0.031 | 0.038 | **0.023** | 0.101 | 0.054 | 0.031 |
| $G$ | $\text{meV}$ | 19 | 14 | 20 | 14 | 22 | **8** | - | - | 12 |
| $H$ | $\text{meV}$ | 17 | 14 | 21 | 14 | 24 | **7** | - | - | 12 |
| $R^2$ | $\text{bohr}^3$ | 0.180 | 0.073 | 0.961 | 0.354 | 0.800 | 0.331 | - | - | **0.106** |
| $U$ | $\text{meV}$ | 20 | 19 | 21 | 14 | 19 | **6** | - | - | 12 |
| $U_0$ | $\text{meV}$ | 20 | 14 | 22 | 13 | 19 | **6** | - | - | 11 |
| $\text{ZPVE}$ | $\text{meV}$ | 1.50 | 1.70 | 2.03 | 1.56 | 2.28 | **1.21** | - | - | 1.55 |

EGNN delivers highly competitive or superior performance across molecular properties without introducing computationally expensive higher-order representations, angles, or spherical harmonics [52].

---

## 6. Conclusions
We presented a new $E(n)$ equivariant deep architecture for graphs that is computationally efficient, easy to implement, and significantly improves over the current state-of-the-art on a wide range of tasks [53]. We believe these properties make it ideally suited to make a direct impact on topics such as drug discovery, protein folding, material design, and 3D computer vision [53].

---

## Acknowledgments
We would like to thank Patrick Forré for his support to formalize the invariance features identification proof [54].

---

## References
1. Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) Equivariant Graph Neural Networks. *In International Conference on Machine Learning (ICML)*, 2021.
2. Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. *arXiv preprint arXiv:1312.6203*, 2013.
3. Michael Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. *Advances in Neural Information Processing Systems*, 2016.
4. Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*, 2016a.
5. Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. *In International Conference on Machine Learning (ICML)*, 2017.
6. Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick F. Riley. Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds. *arXiv preprint arXiv:1802.08219*, 2018.
7. Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, and Max Welling. SE(3)-Transformers: 3D roto-translation equivariant attention networks. *Advances in Neural Information Processing Systems*, 2020.
8. Marc Finzi, Samuel Stanton, Pavel Izmailov, and Andrew Gordon Wilson. Generalizing convolutional neural networks for equivariance to Lie groups on arbitrary continuous data. *arXiv preprint arXiv:2002.12880*, 2020.
9. Jonas Köhler, Leon Klein, and Frank Noé. Equivariant flows: sampling configurations for multi-body systems with symmetric energies. *CoRR, abs/1910.00753*, 2019.
10. Jonas Köhler, Leon Klein, and Frank Noé. Equivariant flows: exact likelihood generative learning for symmetric densities. *arXiv preprint arXiv:2006.02425*, 2020.
11. Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. Neural relational inference for interacting systems. *arXiv preprint arXiv:1802.04687*, 2018.
12. Thomas N. Kipf and Max Welling. Variational graph auto-encoders. *arXiv preprint arXiv:1611.07308*, 2016b.
13. Risi Kondor, Hy Truong Son, Horace Pan, Brandon Anderson, and Shubhendu Trivedi. Covariant compositional networks for learning graphs. *arXiv preprint arXiv:1801.02144*, 2018.
14. Kristof T. Schütt, Farhad Arbabzadah, Stefan Chmiela, Klaus-Robert Müller, and Alexandre Tkatchenko. Quantum-chemical insights from deep tensor neural networks. *Nature Communications*, 8(1):1–8, 2017a.
15. Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *arXiv preprint arXiv:1706.08566*, 2017b.
16. Johannes Klicpera, Shankari Giri, Johannes T. Margraf, and Stephan Günnemann. Fast and uncertainty-aware directional message passing for non-equilibrium molecules. *arXiv preprint arXiv:2011.14115*, 2020a.
17. Johannes Klicpera, Janek Groß, and Stephan Günnemann. Directional message passing for molecular graphs. *arXiv preprint arXiv:2003.03123*, 2020b.
18. Brandon Anderson, Truong-Son Hy, and Risi Kondor. Cormorant: Covariant molecular neural networks. *arXiv preprint arXiv:1906.04015*, 2019.
19. Benjamin K. Miller, Mario Geiger, Tess E. Smidt, and Frank Noé. Relevance of rotationally equivariant convolutions for predicting molecular properties. *arXiv preprint arXiv:2008.08461*, 2020.
20. H. Serviansky, N. Segol, J. Shlomi, K. Cranmer, E. Gross, H. Maron, and Y. Lipman. Set2Graph: Learning graphs from sets. *Advances in Neural Information Processing Systems*, 2020.
21. Martin Simonovsky and Nikos Komodakis. GraphVAE: Towards generation of small graphs using variational autoencoders. *In International Conference on Artificial Neural Networks*, 2018.
22. Jenny Liu, Aviral Kumar, Jimmy Ba, Jamie Kiros, and Kevin Swersky. Graph normalizing flows. *In Advances in Neural Information Processing Systems*, 2019.
23. Balasubramaniam Srinivasan and Bruno Ribeiro. On the equivalence between positional node embeddings and structural graph representations. *arXiv preprint arXiv:1910.00452*, 2019.
24. Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, and Jure Leskovec. GraphRNN: Generating realistic graphs with deep autoregressive models. *arXiv preprint arXiv:1802.06616*, 2018.
25. Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. *In Proceedings of the IEEE International Conference on Computer Vision*, 2019.
26. Raghunath Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*, 1(1):1–7, 2014.
27. Danilo Jimenez Rezende, Sébastien Racanière, Irina Higgins, and Peter Toth. Equivariant Hamiltonian flows. *CoRR, abs/1909.13739*, 2019.
28. David W. Romero and Jean-Baptiste Cordonnier. Group equivariant stand-alone self-attention for vision. *In International Conference on Learning Representations (ICLR)*, 2021.
29. Prajit Ramachandran, Barret Zoph, and Quoc V. Le. Searching for activation functions. *arXiv preprint arXiv:1710.05941*, 2017.
30. Nicholas Watters, Daniel Zoran, Theofanis Weber, Peter Battaglia, Razvan Pascanu, and Andrea Tacchetti. Visual interaction networks: Learning a physics simulator from video. *Advances in Neural Information Processing Systems*, 2017.
31. Maurice Weiler and Gabriele Cesa. General E(2)-equivariant steerable CNNs. *In Advances in Neural Information Processing Systems*, 2019.
32. Béla Bollobás and Béla Bollobás. Random graphs. *Number 73. Cambridge University Press*, 2001.

---

## Appendices

### Appendix A: Equivariance Proof
We formally prove that the Equivariant Graph Convolutional Layer (EGCL) satisfies:

$$Qx^{l+1} + g, h^{l+1} = \text{EGCL}(Qx^l + g, h^l) \quad (15)$$

for any translation vector $g \in \mathbb{R}^n$ and orthogonal transformation matrix $Q \in \mathbb{R}^{n \times n}$ ($Q^T Q = I$) [66].

We assume $h^0$ is invariant to $E(n)$ transformations on $x$ [67]. Under rotation and translation of coordinates $x_i^l \mapsto Qx_i^l + g$, the squared distance term is invariant [67]:

$$\|(Qx_i^l + g) - (Qx_j^l + g)\|^2 = \|Q(x_i^l - x_j^l)\|^2 = (x_i^l - x_j^l)^T Q^T Q (x_i^l - x_j^l) = \|x_i^l - x_j^l\|^2 \quad (11)$$

Thus, the edge message $m_{ij}$ in Equation (3) is invariant [68]:

$$m'_{ij} = \phi_e \left( h_i^l, h_j^l, \|Qx_i^l + g - (Qx_j^l + g)\|^2, a_{ij} \right) = \phi_e \left( h_i^l, h_j^l, \|x_i^l - x_j^l\|^2, a_{ij} \right) = m_{ij} \quad (12)$$

Now, applying the transformation to Equation (4) [68, 69]:

$$Qx_i^l + g + C \sum_{j 
eq i} (Qx_i^l + g - [Qx_j^l + g]) \phi_x(m_{ij}) = Qx_i^l + g + Q C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})$$
$$= Q \left[ x_i^l + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij}) \right] + g = Qx_i^{l+1} + g$$

Since $m_{ij}$ is invariant, $m_i$ in Equation (5) is invariant [69]. Consequently, node feature updates $h_i^{l+1}$ in Equation (6) are invariant [69]:

$$h_i^{l+1} = \phi_h(h_i^l, m_i)$$

This completes the proof [69].

---

### Appendix B: Re-Formulation for Velocity Type Inputs
When velocity inputs are given, the EGCL layer computes $h^{l+1}, x^{l+1}, v^{l+1} = \text{EGCL}[h^l, x^l, v^{init}, E]$ as follows [70]:

$$m_{ij} = \phi_e \left( h_i^l, h_j^l, \|x_i^l - x_j^l\|^2, a_{ij} \right)$$
$$v_i^{l+1} = \phi_v(h_i^l) v_i^{init} + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})$$
$$x_i^{l+1} = x_i^l + v_i^{l+1}$$
$$m_i = \sum_{j 
eq i} m_{ij}$$
$$h_i^{l+1} = \phi_h(h_i^l, m_i)$$

#### B.1 Equivariance Proof for Velocity Type Inputs
We prove that under translation $g \in \mathbb{R}^n$ and orthogonal transformation $Q \in \mathbb{R}^{n \times n}$, the model satisfies [71]:

$$h^{l+1}, Qx^{l+1} + g, Qv^{l+1} = \text{EGCL}[h^l, Qx^l + g, Qv^{init}, E]$$

Applying transformations to the velocity update equation [72]:

$$\phi_v(h_i^l) Qv_i^{init} + C \sum_{j 
eq i} (Qx_i^l + g - [Qx_j^l + g]) \phi_x(m_{ij})$$
$$= Q \phi_v(h_i^l) v_i^{init} + Q C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij})$$
$$= Q \left[ \phi_v(h_i^l) v_i^{init} + C \sum_{j 
eq i} (x_i^l - x_j^l) \phi_x(m_{ij}) \right] = Qv_i^{l+1} \quad (12)$$

And the coordinate update [72]:

$$Qx_i^l + g + Qv_i^{l+1} = Q(x_i^l + v_i^{l+1}) + g = Qx_i^{l+1} + g$$

Thus, velocity and coordinate equivariance are preserved [73].

---

### Appendix C: Implementation Details
Across all experiments, the three learnable functions share a consistent structure [73]:
* **Edge function $\phi_e$:** A 2-layer MLP with Swish activations:
  $$\text{Input} \to \text{Linear} \to \text{Swish} \to \text{Linear} \to \text{Swish} \to \text{Output}$$
* **Coordinate function $\phi_x$:** A 2-layer MLP with one activation:
  $$m_{ij} \to \text{Linear} \to \text{Swish} \to \text{Linear} \to \text{Output}$$
* **Node function $\phi_h$:** A 2-layer MLP with one activation and a residual connection:
  $$[h_i^l, m_i] \to \text{Linear} \to \text{Swish} \to \text{Linear} \to \text{Addition}(h_i^l) \to h_i^{l+1}$$

#### C.1 Implementation Details for Dynamical Systems
We customize the Charged Particle's N-body code from Kipf et al. (2018) in 3D [74].
* **Model specifications:** 4 layers, hidden dimension of 64 [75]. $\phi_v$ consists of two linear layers with a Swish activation in between [75].
* **Baselines:** We implement standard GNN, Radial Field with final Tanh scaling (to prevent divergence), TFN, and SE(3)-Transformers [75].

#### C.2 Implementation Details for Graph Autoencoders
* **Datasets:** Community Small generated using You et al. (2018) code [76]. Erdos-Renyi generated with NetworkX `gnp_random_graph(M, p)` [76].
* **Training setup:** Learning rate $10^{-4}$, batch size 1, Adam optimizer, early stopping based on validation BCE [78].

#### C.3 Implementation Details for QM9
We use the dataloader and preprocessing from Cormorant [78].
* **Model specifications:** 7 layers, 128 hidden features, Adam optimizer, cosine learning rate decay starting at $5 \cdot 10^{-4}$ ($10^{-3}$ for Homo/Lumo/Gap) [79, 80]. Sum-pooling followed by a 2-layer MLP maps embeddings to predictions [79].

---

### Appendix D: Further Experiments

#### D.1 Graph Autoencoder
We study the reconstruction error when the embedding dimension $n$ is reduced ($n \in \{4, 6, 8\}$) [80].

##### Table 5: Wrong edge percentage (%) and F1 Score for different embedding sizes ($n$).
| Dataset | Method | $n=4$ (% Err / F1) | $n=6$ (% Err / F1) | $n=8$ (% Err / F1) |
| :--- | :--- | :---: | :---: | :---: |
| **Community Small** | GNN | 1.45 / 0.977 | 1.29 / 0.980 | 1.29 / 0.980 |
| | Noise-GNN | 1.94 / 0.970 | 0.44 / 0.993 | 0.44 / 0.993 |
| | EGNN (Ours) | 2.19 / 0.966 | 0.42 / 0.993 | **0.06 / 0.999** |
| **Erdos & Renyi** | GNN | 7.92 / 0.844 | 5.22 / 0.894 | 4.62 / 0.907 |
| | Noise-GNN | 3.80 / 0.925 | 2.66 / 0.947 | 1.25 / 0.975 |
| | EGNN (Ours) | 3.09 / 0.939 | 0.58 / 0.988 | **0.11 / 0.998** |

For very small sizes ($n=4$), all methods struggle, but as $n$ grows, EGNN achieves nearly perfect reconstruction [82].

---

### Appendix E: Sometimes Invariant Features are All You Need
We prove that when only static positional coordinates $x_i$ are provided (and no velocity-type inputs), the overall geometry of a point collection $\{x_i\}_{i=1}^M$ up to $E(n)$ transformations is completely uniquely identified by the pairwise Euclidean distances [82, 83].

#### E.1 Invariance of Distance Norms under E(n)
Let $x \mapsto Qx + t$ be an arbitrary $E(n)$ transformation, where $Q$ is orthogonal ($Q^T Q = I$) and $t$ is a translation vector [84]. The transformed pairwise distance is [84]:

$$\ell_2(Qx_i + t, Qx_j + t) = \sqrt{(Qx_i + t - [Qx_j + t])^T (Qx_i + t - [Qx_j + t])}$$
$$= \sqrt{(Q(x_i - x_j))^T Q(x_i - x_j)} = \sqrt{(x_i - x_j)^T Q^T Q (x_i - x_j)} = \ell_2(x_i, x_j) \quad (13)$$

Thus, Euclidean distances are $E(n)$ invariant [84].

#### E.2 Uniqueness of Distance Norms Representation
Let $\{x_i\}_{i=1}^M$ and $\{y_i\}_{i=1}^M$ be two point collections satisfying $\ell_2(x_i, x_j) = \ell_2(y_i, y_j)$ for all $i, j$ [84]. We prove there exists an orthogonal matrix $Q$ and translation $t$ such that $x_i = Qy_i + t$ for all $i$ [84].

**Proof:**  
We subtract $x_0$ and $y_0$ to align the origins: $\tilde{x}_i = x_i - x_0$ and $\tilde{y}_i = y_i - y_0$ [85]. Distances are preserved under translation, so $\ell_2(\tilde{x}_i, \tilde{x}_j) = \ell_2(\tilde{y}_i, \tilde{y}_j)$ [85]. Assuming without loss of generality that $x_0 = y_0 = 0$, we have $\|\tilde{x}_i\|_2 = \|\tilde{y}_i\|_2$ [85].

Expanding the squared distance expression [85]:

$$\tilde{x}_i^T \tilde{x}_i - 2 \tilde{x}_i^T \tilde{x}_j + \tilde{x}_j^T \tilde{x}_j = \ell_2(\tilde{x}_i, \tilde{x}_j)^2 = \ell_2(\tilde{y}_i, \tilde{y}_j)^2 = \tilde{y}_i^T \tilde{y}_i - 2 \tilde{y}_i^T \tilde{y}_j + \tilde{y}_j^T \tilde{y}_j$$

Since $\|\tilde{x}_i\|^2 = \|\tilde{y}_i\|^2$ and $\|\tilde{x}_j\|^2 = \|\tilde{y}_j\|^2$, we obtain:

$$\langle \tilde{x}_i, \tilde{x}_j \rangle = \langle \tilde{y}_i, \tilde{y}_j \rangle \quad (14)$$

which proves that angles between all pairs are identical [85]. 

Thus, for any linear combination of coefficients $c_i$, we have [86]:

$$\| \sum_i c_i \tilde{x}_i \|^2 = \| \sum_i c_i \tilde{y}_i \|^2 \quad (*)$$

Let $V_x = \text{span}(\{\tilde{x}_i\})$ with a basis $\{x_{i_j}\}_{j=1}^d$ ($d \le n$) [86]. We define a linear map $A : V_x \to V_y$ by mapping the basis vectors $x_{i_j} \mapsto y_{i_j}$ [86]. For any point $\tilde{x}_i = \sum_j c_j x_{i_j}$, we show $A\tilde{x}_i = \tilde{y}_i$ [86]:

$$\| \tilde{y}_i - A\tilde{x}_i \|_2^2 = \| \tilde{y}_i - \sum_j c_j y_{i_j} \|_2^2$$
$$= \langle \tilde{y}_i, \tilde{y}_i \rangle - 2 \langle \tilde{y}_i, \sum_j c_j y_{i_j} \rangle + \langle \sum_j c_j y_{i_j}, \sum_j c_j y_{i_j} \rangle$$
$$\stackrel{(*)}{=} \langle \tilde{x}_i, \tilde{x}_i \rangle - 2 \langle \tilde{x}_i, \sum_j c_j x_{i_j} \rangle + \langle \sum_j c_j x_{i_j}, \sum_j c_j x_{i_j} \rangle = 0 \quad (15)$$

Thus $A\tilde{x}_i = \tilde{y}_i$ for all $i$ [87]. Orthogonality of $A$ on $V_x$ is guaranteed because:

$$\langle A x_{i_j}, A x_{i_k} \rangle = \langle y_{i_j}, y_{i_k} \rangle = \langle x_{i_j}, x_{i_k} \rangle \quad (16)$$

We can extend $A$ to an orthogonal matrix $Q$ over the whole space $\mathbb{R}^n$ using the orthogonal complement [87]. Finally, incorporating the translations, we obtain $x_i = Qy_i + t$, which completes the uniqueness proof [87].
