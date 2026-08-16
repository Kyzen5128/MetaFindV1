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

# ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding

**Le Xue**¹ *, **Ning Yu**¹ , **Shu Zhang**¹, **Artemis Panagopoulou**¹,³, **Junnan Li**¹, **Roberto Martín-Martín**⁴, **Jiajun Wu**², **Caiming Xiong**¹, **Ran Xu**¹, **Juan Carlos Niebles**¹,², **Silvio Savarese**¹,²  
¹ *Salesforce AI Research*  
² *Stanford University*  
³ *University of Pennsylvania*  
⁴ *University of Texas at Austin*  
*Corresponding contact: lxue@salesforce.com*

---

## Abstract
Recent advancements in multimodal pre-training have shown promising efficacy in 3D representation learning by aligning multimodal features across 3D shapes, their 2D counterparts, and language descriptions [156]. However, the methods used by existing frameworks to curate such multimodal data, in particular language descriptions for 3D shapes, are not scalable, and the collected language descriptions are not diverse [156]. To address this, we introduce **ULIP-2**, a simple yet effective tri-modal pre-training framework that leverages large multimodal models to automatically generate holistic language descriptions for 3D shapes [156]. It only needs 3D data as input, eliminating the need for any manual 3D annotations, and is therefore scalable to large datasets [156]. 

ULIP-2 is also equipped with scaled-up backbones for better multimodal representation learning [156]. We conduct experiments on two large-scale 3D datasets, **Objaverse** and **ShapeNet**, and augment them with tri-modal datasets of 3D point clouds, images, and language for training ULIP-2 [156]. Experiments show that ULIP-2 demonstrates substantial benefits in three downstream tasks: zero-shot 3D classification, standard 3D classification with fine-tuning, and 3D captioning (3D-to-language generation) [156]. It achieves a new SOTA of **50.6%** (top-1) on Objaverse-LVIS and **84.7%** (top-1) on ModelNet40 in zero-shot classification [156]. In the ScanObjectNN benchmark for standard fine-tuning, ULIP-2 reaches an overall accuracy of **91.5%** with a compact model of only 1.4 million parameters [156]. ULIP-2 sheds light on a new paradigm for scalable multimodal 3D representation learning without human annotations and shows significant improvements over existing baselines [156]. 

The code and datasets are released at [https://github.com/salesforce/ULIP](https://github.com/salesforce/ULIP) [156].

---

## 1. Introduction
3D visual understanding has seen a surge of interest in recent years due to its growing applications in augmented reality and virtual reality (AR and VR), autonomous driving, and robotics [157, 159]. Despite this, the collection and annotation of 3D data remain a costly and labor-intensive process [159]. In response to this challenge, researchers have turned to other more abundantly available modalities, such as image and natural language, to provide supervisory signals for learning 3D representations [159]. This approach has not only led to improved unimodal representation but also cultivated a richer multimodal representation capability, alleviating the need for single-modal dense annotations in the 3D domain [159].

However, multimodal learning frameworks in this direction commonly face the challenge of assembling scalable, high-quality, and well-aligned multimodal data for 3D applications [160]. We identify the **language modality** for 3D data as the critical bottleneck in this process [160]. Existing frameworks tend to utilize manually annotated category names and short descriptions derived from metadata as the language counterparts for the 3D data [160]. Those approaches lack scalability as they always rely on some degree of human annotations during the dataset collection process, which is hard to scale up [160]. Furthermore, existing methods are not comprehensive enough as the derived language information might not provide sufficient details, lacks variations, or appears to be noisy [160]. This highlights the need for an innovative paradigm to provide language counterparts for 3D data that are both scalable and comprehensive [160].

To address these issues, we first reconsider what the 2D image counterpart modality for a 3D shape should be [161]. Semantically, if we can render 2D images of a 3D shape from any viewpoint, the collection of all these rendered images should approximately encapsulate all information about this 3D shape, thus forming an appropriate image counterpart modality for 3D [161]. By analogy, if we can linguistically describe a 3D shape from any viewpoint, the compilation of all these language descriptions from all perspectives should also approximately encompass all linguistically expressible information about this shape, thus forming an appropriate language modality for the 3D shape [161]. In practice, for efficiency, we sample a finite fixed set of holistic viewpoints instead of "any viewpoint" [161]. If we apply the same set of viewpoints for creating the language modality as we render the images, this task naturally boils down to describing the rendered 2D image for a given viewpoint [161]. 

Leveraging the advances in large multimodal models, we utilize their ability to generate detailed language descriptions for the rendered images [161]. This method allows us to automate the process in a scalable way as it only needs 3D data itself, while the rich knowledge from the large multimodal models is distilled into the language descriptions [161]. As a result, this automated and scalable strategy enriches the language modality with detailed, holistic descriptions, further aiding multimodal 3D representation learning [162].

We introduce **ULIP-2**, a novel framework that encompasses an innovative approach to generate well-aligned, holistic multimodal data for 3D understanding, coupled with an efficient multimodal pre-training architecture capable of robustly aligning this multimodal data [162]. Given a 3D shape, our initial step involves extracting 3D point cloud data [163]. We then render this shape into a series of images from a fixed set of holistic viewpoints [163]. For each rendered image, we employ a large multimodal model to generate a list of detailed descriptions, thereby establishing the language modality (as illustrated in Figure 2) [163]. 

ULIP-2 advances beyond its predecessor, ULIP, by:
1. Proposing a **manual-effort-free data creation paradigm** for comprehensive multimodal learning [164].
2. Leveraging this scalable paradigm to **extend multimodal 3D learning to larger datasets**, while scaling up both the vision-language and 3D backbones [164].
3. Delivering **impressive improvements over ULIP on all downstream tasks** when pre-trained on the same datasets [164].

---

## 2. Related Work

### 2.1 Multimodal Representation Learning
Multimodal representation learning has emerged as a popular research topic [169]. Most research works focus on learning representation for language and image modalities [169]. One line of research emphasizes the interaction between image regions and caption tokens using Transformer-based architectures [169]. Alternatively, methods such as CLIP and SLIP target generating single features for image and text independently and subsequently aligning these two modalities, promoting robust and efficient large-scale pre-training [169].

Recent works have extended multimodal representation learning to the 3D modality [170]. **ULIP** is one of the pioneering works in creating (3D point cloud - image - language) triplets [170]. By aligning these three modalities together, ULIP enhances 3D representation learning and mitigates the need for single-modal dense 3D annotations [170]. Concurrent work OpenShape further extends ULIP's framework but still relies on manual annotations of 3D data and a complicated data engineering framework [170, 171]. In contrast, ULIP-2 overcomes these limitations by leveraging the power and knowledge of state-of-the-art large multimodal models, fundamentally diminishing data requirements and enriching the pre-trained multimodal data [171].

### 2.2 Generative Large Multimodal Models
The expansion of transformer models demonstrates the effectiveness of scale in multimodal tasks [172]. This approach has seen considerable advancements in text generation from images [172]. Our study leverages **BLIP-2** to generate diverse annotations for 3D shapes, facilitating learning richer multimodal 3D representations [172]. Our ablation study indicates that ULIP-2 benefits from the advancements in large multimodal models, synergizing with rapid improvements in the field [172].

### 2.3 3D Point Cloud Understanding
PointNet is a pioneering work that processes 3D point clouds directly [173]. Building on this, PointNeXt emerges as a lightweight, high-performance variant [173]. In the realm of self-supervised pre-training, Point-BERT moves a significant step forward with its transformer-based architecture, showcasing notable performance in zero-shot classification [173]. In ULIP-2, we leverage both Point-BERT and PointNeXt as our 3D encoders to harness their strong capabilities [173].

---

## 3. Methodology

### 3.1 Preliminary: ULIP
ULIP [52] presents an efficient multimodal pre-training framework that constructs triplets encompassing three modalities: 
1. **3D modality**: obtained by extracting 3D point cloud data [175].
2. **Image modality**: generated by rendering images from 3D shapes across multiple viewpoints [175].
3. **Language modality**: derived by prompting dataset metadata such as descriptive terms and category names into cohesive sentences [175].

ULIP utilizes the ViT-B encoders from SLIP, a pre-trained vision-language model, to learn 3D representations by aligning 3D modality features to the pre-aligned language-image feature space [175].

### 3.2 Scalable Triplet Creation
In ULIP-2, the model similarly utilizes three input modalities, though **it only requires the 3D object data itself** [176]. Given a 3D object:
1. We extract 3D point clouds from the surface as the input to the 3D encoder [176].
2. We generate images from various viewing angles [176].
3. We leverage **BLIP-2**, a cutting-edge large multimodal model, to generate descriptive texts for each rendered 2D image [176]. For each image, we generate a set of sentences, rank them using CLIP similarities, and aggregate the top-1 description to form the language modality in the triplet [176].

This scalable triplet creation approach facilitates dataset scaling, eliminating the need for dataset metadata collection and necessitating only the 3D data itself [177].

### 3.3 Tri-modal Pre-training
ULIP-2 aligns the triplet of 3D point clouds, 2D rendered images, and comprehensive descriptions to a unified feature space [177]. We adopt the largest version of encoders from OpenCLIP (**ViT-G/14**) [13] for most of our experiments and freeze it during pre-training [180]. The feature space, already pre-aligned by OpenCLIP, serves as the target space where we aim to integrate the 3D modality [180].

Given a 3D shape $O$, we extract its 3D point cloud $P$, randomly sample its 2D rendered image $I \sim 	ext{render}(O)$, and its BLIP-2 generated language description $T \sim 	ext{blip2}(I)$, where $	ext{render}$ is the 3D-to-2D rendering operation and $	ext{blip2}$ is the image description query [180]. We extract the image feature $f_I = E_I(I)$ and text feature $f_T = E_T(T)$ based on the frozen encoders [180]. We train the 3D encoder $E_P$ to align the 3D feature $f_P = E_P(P)$ with the image and text features [180].

The **3D-to-image contrastive loss** is formulated as:
$$\mathcal{L}_{P2I} = -rac{1}{2} \sum_{i} \left[ \log rac{\exp(f_{P,i} \cdot f_{I,i} / 	au)}{\sum_{j} \exp(f_{P,i} \cdot f_{I,j} / 	au)} + \log rac{\exp(f_{P,i} \cdot f_{I,i} / 	au)}{\sum_{j} \exp(f_{P,j} \cdot f_{I,i} / 	au)} 
ight] \quad (1)$$

where $i, j$ are the sampling indices, and $	au$ is a learnable temperature parameter [181].

Similarly, the **3D-to-text contrastive loss** is formulated as:
$$\mathcal{L}_{P2T} = -rac{1}{2} \sum_{i} \left[ \log rac{\exp(f_{P,i} \cdot f_{T,i} / 	au)}{\sum_{j} \exp(f_{P,i} \cdot f_{T,j} / 	au)} + \log rac{\exp(f_{P,i} \cdot f_{T,i} / 	au)}{\sum_{j} \exp(f_{P,j} \cdot f_{T,i} / 	au)} 
ight] \quad (2)$$

Our final training objective is to minimize the sum of the two contrastive alignment losses:
$$\min_{E_P} \mathcal{L}_{P2I} + \mathcal{L}_{P2T} \quad (3)$$

### 3.4 Scaling Up the 3D Multimodal Learning
Recognizing the benefits of more powerful image and text encoders, we extend our exploration beyond the smaller ViT-B model utilized in ULIP [182]. Our experiments focus on upgrading this vision-language backbone to **ViT-G** in the tri-modal alignment framework [180, 182]. Additionally, we investigate scaling up the model size of the 3D backbone while keeping other settings unchanged, evaluating effectiveness through zero-shot classification [182].

---

## 4. Experiments

### 4.1 Triplet Creation Datasets
We extract triplets based on two large-scale datasets of 3D shapes [183]:
* **ULIP-Objaverse Triplets**: Derived from Objaverse, containing $\sim$**800K** real-world 3D shapes [183]. We render **12 images** per shape, spaced equally by 30 degrees [183]. For each image, we employ BLIP-2-opt6.7B to generate **10 independent descriptions**, ranking them using CLIP-ViT-Large similarities and selecting the top-1 [183]. We sample 10k, 8k, and 2k point clouds per shape [183].
* **ULIP-ShapeNet Triplets**: Derived from ShapeNet, containing $\sim$**52.5K** synthetic CAD models spanning 55 categories [184]. We sample **30 equally spaced view angles** to render RGB and depth images, using the same BLIP-2 description generation and CLIP ranking workflow [184].

#### Table 2: Triplet Dataset Statistics
| Dataset | Point Clouds | Images | Language |
| :--- | :---: | :---: | :---: |
| **ULIP-Objaverse** | $\sim$ 800k | $\sim$ 10 million | $\sim$ 100 million |
| **ULIP-ShapeNet** | $\sim$ 52.5k | $\sim$ 3 million | $\sim$ 30 million |

---

### 4.2 Downstream Task Evaluations

#### Zero-Shot 3D Classification
We benchmark zero-shot performance on **ModelNet40** (synthetic CAD, $\sim$2.5k test samples) and **Objaverse-LVIS** (challenging open-world benchmark, $\sim$46k samples, $\sim$1.2k categories) [185].

##### Table 1: Zero-Shot 3D Classification Accuracy (R@1 / R@5)
| Method | Pre-train Dataset | Manual Captions? | Objaverse-LVIS (Top-1 / Top-5) | ModelNet40 (Top-1 / Top-5) |
| :--- | :--- | :---: | :---: | :---: |
| **PointCLIP** [58] | – | – | 1.9 / 5.8 | 19.3 / 34.8 |
| **PointCLIPv2** [62] | – | – | 4.7 / 12.9 | 63.6 / 85.0 |
| **ReCon** [34] | ShapeNet | Yes | 1.1 / 3.7 | 61.2 / 78.1 |
| **CLIP2Point** [11] | ShapeNet | No | 2.7 / 7.9 | 49.5 / 81.2 |
| **OpenShape** [22] | ShapeNet | Yes | 10.8 / 25.0 | 70.3 / 91.3 |
| **OpenShape** [22] | Objaverse (no LVIS) + ShapeNet | Yes | 38.8 / 68.8 | 83.9 / 97.6 |
| **OpenShape** [22] | Objaverse + ShapeNet | Yes | 46.5 / 76.3 | 82.6 / 96.9 |
| **OpenShape** [22] | Objaverse + ShapeNet + 2 Extra | Yes | 46.8 / 77.0 | 84.4 / 98.0 |
| **ULIP** [52] | ShapeNet | Yes | 2.6 / 8.1 | 60.4 / 84.0 |
| **ULIP-2** (Ours) | ShapeNet | **No** | **16.4 / 34.3** | **75.2 / 95.0** |
| **ULIP** [52] | Objaverse (no LVIS) + ShapeNet | Yes | 21.4 / 41.9 | 68.6 / 86.4 |
| **ULIP-2** (Ours) | Objaverse (no LVIS) + ShapeNet | **No** | **46.3 / 75.0** | **84.0 / 97.2** |
| **ULIP** [52] | Objaverse + ShapeNet | Yes | 34.9 / 61.0 | 69.6 / 85.9 |
| **ULIP-2** (Ours) | Objaverse + ShapeNet | **No** | **50.6 / 79.1** | **84.7 / 97.1** |

*Note: Highlighted lines represent SOTA comparisons. ULIP-2 achieves SOTA performance, outperforming OpenShape by 3.8% on the Objaverse-LVIS top-1 benchmark without manual captions [179].*

---

#### Standard 3D Classification (Fine-tuning)
Evaluated on **ScanObjectNN (Hardest set)** [185, 188].

##### Table 3: Fine-Tuned 3D Classification on ScanObjectNN
| Model Backbone | Pre-training | # Params (M) | Overall Accuracy (%) | Class-Average Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: |
| **PointNet** [32] | From Scratch | 3.5 | 68.2 | 63.4 |
| **PointNet++** [33] | From Scratch | 1.5 | 77.9 | 75.4 |
| **DGCNN** [49] | From Scratch | 1.8 | 78.1 | 73.6 |
| **MVTN** [9] | From Scratch | 11.2 | 82.8 | – |
| **RepSurf-U** [38] | From Scratch | 1.5 | 84.6 | – |
| **Point-MAE** [31] | From Scratch | 22.1 | 85.2 | – |
| **PointMLP** [26] | From Scratch | 12.6 | 85.7 | 84.4 |
| **Point-M2AE** [57] | From Scratch | 15.3 | 86.4 | – |
| **PointCMT** [53] | From Scratch | 12.6 | 86.7 | 84.8 |
| **ACT** [7] | From Scratch | 22.1 | 88.2 | – |
| **P2P** [47] | From Scratch | – | 89.3 | – |
| **Recon-s** [34] | From Scratch | 19.0 | 89.5 | – |
| **I2P-MAE** [59] | From Scratch | 12.9 | 90.1 | – |
| **Point-BERT** [55] | Official Initialization | 22.1 | 83.1 | – |
| **Point-BERT** [55] | w/ ULIP | 22.1 | 88.7 | – |
| **Point-BERT** [55] | w/ **ULIP-2** | 22.1 | **89.7** | – |
| **PointNeXt** [36] | From Scratch | 1.4 | 87.5 | 85.9 |
| **PointNeXt** [36] | w/ ULIP | 1.4 | 90.1 | 89.2 |
| **PointNeXt** [36] | w/ **ULIP-2** | 1.4 | **91.1** | **90.3** |
| **PointNeXt** [36] | w/ **ULIP-2**\* (with voting) | 1.4 | **91.5** | **90.9** |

---

#### 3D-to-Language Generation (3D Captioning)
We adopt the **X-InstructBLIP** methodology [40] to plug our frozen pre-trained Point-BERT encoder into a frozen LLM [192].

##### Table 4: 3D Captioning performance (CIDEr Score) on X-InstructBLIP
| Multimodal Framework | 3D Encoder | Pre-training Dataset | CIDEr Score |
| :--- | :---: | :--- | :---: |
| **X-InstructBLIP** [40] | Point-BERT (w/ ULIP) | Objaverse + ShapeNet | 132.2 |
| **X-InstructBLIP** [40] | Point-BERT (w/ **ULIP-2**) | Objaverse + ShapeNet | **160.5** |

*Note: ULIP-2 improves the captioning score by **28.3%**, producing significantly more accurate and descriptive captions [192].*

---

## 5. Ablation Studies

### 5.1 Ablation on the Effect of Generated Captions
We compared the manual captions of ULIP to our BLIP-2 generated top-1 holistic captions [193].

##### Table 5: Captions Ablation (Point-BERT on ModelNet40)
| Pre-train Language Modality | ModelNet40 Top-1 | ModelNet40 Top-5 |
| :--- | :---: | :---: |
| **Manual captions** (ULIP) | 60.4 | 84.0 |
| **Top-1 holistic BLIP-2 captions** (Ours) | **69.7** | **88.1** |

---

### 5.2 Different Large Multimodal Models
We compared BLIP-2 with its predecessor BLIP for description generation [194].

##### Table 6: Multi-modal Generator Ablation (ModelNet40 Zero-Shot)
| Large Multimodal Model | ModelNet40 Top-1 | ModelNet40 Top-5 |
| :--- | :---: | :---: |
| **BLIP** [17] | 67.7 | 88.6 |
| **BLIP-2** [18] | **69.7** | **88.8** |

---

### 5.3 Number of 2D Views Per 3D Object
Ablation on the number of rendered views used in pre-training [195].

##### Table 7: Number of Rendered Views Ablation
| # Holistic Views | ModelNet40 Top-1 | ModelNet40 Top-5 |
| :--- | :---: | :---: |
| **1** | 54.8 | 77.9 |
| **2** | 58.1 | 80.5 |
| **15** | 69.3 | 88.6 |
| **30** | **69.7** | **88.8** |

---

### 5.4 Top-k CLIP Ranked Captions Per 2D View
Ablation on selecting the top-k ranked captions from the 10 independently generated descriptions [196].

##### Table 8: Top-k Caption Selection Strategy
| Top-k Captions Selected | ModelNet40 Top-1 | ModelNet40 Top-5 |
| :--- | :---: | :---: |
| **Top-1** (Ours) | **69.7** | **88.8** |
| **Top-3** | 66.7 | 87.2 |
| **Top-5** | 66.4 | 87.7 |
| **Top-10** | 66.3 | 85.1 |

---

### 5.5 Scaling Up the Backbone Models
Evaluation of increasing CLIP sizes and 3D backbones [197].

##### Table 9: Scaling Encoders (Pre-trained on Objaverse-no-LVIS)
| CLIP Encoder Size | 3D Encoder Params (M) | ModelNet40 Top-1 | ModelNet40 Top-5 | Objaverse-LVIS Top-1 | Objaverse-LVIS Top-5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-B | 21.9 | 71.4 | 89.7 | 28.3 | 52.6 |
| **ViT-G** | 21.9 | 76.3 | 94.1 | 35.0 | 62.5 |
| ViT-G | 5.3 | 75.0 | 94.7 | 34.1 | 61.1 |
| **ViT-G** | **32.5** | **77.0** | **94.0** | **35.7** | **62.9** |
| ViT-G | 43.1 | 76.8 | 94.8 | 35.9 | 62.6 |
| ViT-G | 85.7 | 76.5 | 94.7 | 35.9 | 62.7 |

*Note: Gray highlight (32.5M 3D encoder + ViT-G CLIP) balances performance and parameter footprint, and represents our chosen default scaled-up setup [197, 198].*

---

## 6. Conclusion, Limitations, and Broader Impact

### 6.1 Conclusion
We present **ULIP-2**, a novel framework for multimodal 3D representation learning [199]. By leveraging large multimodal models for language description generation and scaling up pre-training backbones, ULIP-2 addresses dataset quality and scalability challenges, achieving significant improvements in zero-shot classification, fine-tuned classification, and 3D-to-language generation [199]. We also release "ULIP-Objaverse" and "ULIP-ShapeNet" triplets to foster future research [199].

### 6.2 Limitations
ULIP-2’s pre-training primarily utilizes object-level 3D shape datasets, which differ in distribution and complexity from scene-level 3D data [200]. Applying ULIP-2 to scene-level data represents a compelling avenue for future research [200].

### 6.3 Broader Impact
ULIP-2 aims to minimize human annotation in 3D pre-training, reducing labor but potentially impacting low-skilled job markets, highlighting the need for ethical AI considerations [200].

---

## References
1. Peter Anderson, Xiaodong He, et al. Bottom-up and top-down attention for image captioning and visual question answering. *CVPR*, 2018.
2. Iro Armeni, Ozan Sener, et al. 3D semantic parsing of large-scale indoor spaces. *CVPR*, 2016.
3. Cesar Cadena, Anthony R Dick, et al. Multi-modal auto-encoders as joint estimators for robotics scene understanding. *Robotics: Science and Systems*, 2016.
4. Angel X Chang, Thomas Funkhouser, et al. ShapeNet: An information-rich 3D model repository. *arXiv:1512.03012*, 2015.
5. Jaemin Cho, Jie Lei, et al. Unifying vision-and-language tasks via text generation. *ICML*, 2021.
6. Matt Deitke, Dustin Schwenk, et al. Objaverse: A universe of annotated 3D objects. *arXiv:2212.08051*, 2022.
7. Runpei Dong, Zekun Qi, et al. Autoencoders as cross-modal teachers: Can pretrained 2D image transformers help 3D representation learning? *arXiv:2212.08320*, 2022.
8. Benjamin Graham, Martin Engelcke, et al. 3D semantic segmentation with submanifold sparse convolutional networks. *CVPR*, 2018.
9. Abdullah Hamdi, Silvio Giancola, et al. MVTN: Multi-view transformation network for 3D shape recognition. *ICCV*, 2021.
10. Qingyong Hu, Bo Yang, et al. RandLA-Net: Efficient semantic segmentation of large-scale point clouds. *CVPR*, 2020.
11. Tianyu Huang, Bowen Dong, et al. Clip2point: Transfer CLIP to point cloud classification with image-depth pre-training. *ICCV*, 2023.
12. Zhicheng Huang, Zhaoyang Zeng, et al. Seeing out of the box: End-to-end pre-training for vision-language representation learning. *CVPR*, 2021.
13. Gabriel Ilharco, Mitchell Wortsman, et al. OpenCLIP, 2021.
14. Brian R Kent. 3D scientific visualization with Blender. *Morgan & Claypool*, 2015.
15. Wonjae Kim, Bokyung Son, et al. ViLT: Vision-and-language transformer without convolution or region supervision. *ICML*, 2021.
16. Junnan Li, Ramprasaath Selvaraju, et al. Align before fuse: Vision and language representation learning with momentum distillation. *NeurIPS*, 2021.
17. Junnan Li, Dongxu Li, et al. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. *ICML*, 2022.
18. Junnan Li, Dongxu Li, et al. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *ICML*, 2023.
19. Wei Li, Can Gao, et al. UNIMO: Towards unified-modal understanding and generation via cross-modal contrastive learning. *ACL*, 2021.
20. Yingwei Li, Adams Wei Yu, et al. Deepfusion: Lidar-camera deep fusion for multi-modal 3D object detection. *CVPR*, 2022.
21. Zhichao Li, Feng Wang, et al. Lidar R-CNN: An efficient and universal 3D object detector. *CVPR*, 2021.
22. Minghua Liu, Ruoxi Shi, et al. OpenShape: Scaling up 3D shape representation towards open-world understanding. *NeurIPS*, 2023.
23. Yongcheng Liu, Bin Fan, et al. DensePoint: Learning densely contextual representation for efficient point cloud processing. *ICCV*, 2019.
24. Ze Liu, Zheng Zhang, et al. Group-free 3D object detection via transformers. *ICCV*, 2021.
25. Jiasen Lu, Jianwei Yang, et al. Neural baby talk. *CVPR*, 2018.
26. Xu Ma, Can Qin, et al. Rethinking network design and local geometry in point cloud: A simple residual MLP framework. *arXiv:2202.07123*, 2022.
27. Ishan Misra, Rohit Girdhar, et al. An end-to-end transformer model for 3D object detection. *ICCV*, 2021.
28. Norman Mu, Alexander Kirillov, et al. SLIP: Self-supervision meets language-image pretraining. *ECCV*, 2022.
29. OpenAI. GPT-4 technical report. *OpenAI Blog*, 2023.
30. Artemis Panagopoulou, Le Xue, et al. X-instructblip: A framework for aligning x-modal instruction-aware representations to LLMs. *arXiv:2311.18799*, 2023.
31. Yatian Pang, Wenxiao Wang, et al. Masked autoencoders for point cloud self-supervised learning. *arXiv:2203.06604*, 2022.
32. Charles R Qi, Hao Su, et al. PointNet: Deep learning on point sets for 3D classification and segmentation. *CVPR*, 2017.
33. Charles R Qi, Li Yi, et al. PointNet++: Deep hierarchical feature learning on point sets in a metric space. *NeurIPS*, 2017.
34. Zekun Qi, Runpei Dong, et al. Contrast with reconstruct: Contrastive 3D representation learning guided by generative pretraining. *ICML*, 2023.
35. Zhangyang Qi, Ye Fang, et al. GPT4Point: A unified framework for point-language understanding and generation. *arXiv:2312.02980*, 2023.
36. Guocheng Qian, Yuchen Li, et al. PointNeXt: Revisiting PointNet++ with improved training and scaling strategies. *arXiv:2206.04670*, 2022.
37. Alec Radford, Jong Wook Kim, et al. Learning transferable visual models from natural language supervision. *ICML*, 2021.
38. Haoxi Ran, Jun Liu, et al. Surface representation for point clouds. *CVPR*, 2022.
39. Manli Shu, Le Xue, et al. Model-agnostic hierarchical attention for 3D object detection. *arXiv:2301.02650*, 2023.
40. Anonymous Submission. X-InstructBLIP: A framework for aligning X-modal instruction-aware representations to LLMs. *X-InstructBLIP paper*, 2023.
41. Mikaela Angelina Uy, Quang-Hieu Pham, et al. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. *ICCV*, 2019.
42. Mikaela Angelina Uy, Quang-Hieu Pham, et al. Revisiting point cloud classification. *ICCV*, 2019. (duplicate citation in text)
43. Ashish Vaswani, Noam Shazeer, et al. Attention is all you need. *NeurIPS*, 2017.
44. Ramakrishna Vedantam, C Lawrence Zitnick, et al. CIDEr: Consensus-based image description evaluation. *CVPR*, 2015.
45. Thang Vu, Kookhoi Kim, et al. SoftGroup for 3D instance segmentation on point clouds. *CVPR*, 2022.
46. Zirui Wang, Jiahui Yu, et al. SimVLM: Simple visual language model pretraining with weak supervision. *arXiv:2108.10904*, 2021.
47. Ziyi Wang, Xumin Yu, et al. P2P: Tuning pre-trained image models for point cloud analysis with point-to-pixel prompting. *arXiv:2208.02812*, 2022.
48. Christian Wojek, Stefan Walk, et al. Monocular 3D scene understanding with explicit occlusion reasoning. *CVPR*, 2011.
49. Bo Wu, Yang Liu, et al. DGCNN: Disordered graph convolutional neural network based on the gaussian mixture model. *Neurocomputing*, 2018.
50. Zhirong Wu, Shuran Song, et al. 3D ShapeNets: A deep representation for volumetric shapes. *CVPR*, 2015.
51. Runsen Xu, Xiaolong Wang, et al. PointLLM: Empowering large language models to understand point clouds. *arXiv:2308.16911*, 2023.
52. Le Xue, Mingfei Gao, et al. ULIP: Learning unified representation of language, image and point cloud for 3D understanding. *CVPR*, 2023.
53. Xu Yan, Heshen Zhan, et al. Let images give you more: Point cloud cross-modal training for shape analysis. *arXiv:2210.04208*, 2022.
54. Tianwei Yin, Xingyi Zhou, et al. Center-based 3D object detection and tracking. *CVPR*, 2021.
55. Xumin Yu, Lulu Tang, et al. Point-BERT: Pre-training 3D point cloud transformers with masked point modeling. *CVPR*, 2022.
56. Yan Zeng, Xinsong Zhang, et al. Multi-grained vision language pre-training: Aligning texts with visual concepts. *arXiv:2111.08276*, 2021.
57. Renrui Zhang, Ziyu Guo, et al. Point-m2ae: Multi-scale masked autoencoders for hierarchical point cloud pre-training. *arXiv:2205.14401*, 2022.
58. Renrui Zhang, Ziyu Guo, et al. PointCLIP: Point cloud understanding by CLIP. *CVPR*, 2022.
59. Renrui Zhang, Liuhui Wang, et al. Learning 3D representations from 2D pre-trained models via image-to-point masked autoencoders. *arXiv:2212.06785*, 2022.
60. Luowei Zhou, Yannis Kalantidis, et al. Grounded video description. *CVPR*, 2019.
61. Luowei Zhou, Hamid Palangi, et al. Unified vision-language pretraining for image captioning and vqa. *AAAI*, 2020.
62. Xiangyang Zhu, Renrui Zhang, et al. Point-CLIP v2: Prompting CLIP and GPT for powerful 3D open-world learning. *ICCV*, 2023.

---

## Appendices

### Appendix A.1: Ablation on 3D Input (Point Cloud Preprocessing)
We investigate how point cloud size and color channels (RGB) affect representation learning [225]. 

##### Table 10: 3D Input Preprocessing Ablation on Objaverse-LVIS Zero-Shot R@1
| 3D Encoder Input | Objaverse-LVIS Top-1 (%) | Objaverse-LVIS Top-5 (%) |
| :--- | :---: | :---: |
| **8k xyz** | 48.9 | 77.1 |
| **10k xyzrgb** | **50.6** | **79.1** |

*Note: Incorporating color channels (xyzrgb) with 10k points provides a 1.7% boost but ULIP-2 maintains excellent performance even without color information [225].*

### Appendix A.2: Backbone Agnostic Generalization
To verify that ULIP-2's improvements generalize across different point cloud encoders, we compared PointNeXt with Point-BERT [226].

##### Table 11: 3D Encoder Architecture Generalization (ModelNet40 Zero-Shot)
| 3D Encoder Backbone | Pre-training Method | ModelNet40 Top-1 (%) | ModelNet40 Top-5 (%) |
| :--- | :--- | :---: | :---: |
| **PointNeXt** [36] | ULIP [52] | 56.2 | 77.0 |
| **PointNeXt** [36] | **ULIP-2** (Ours) | **72.8** | **95.7** |
| **Point-BERT** [55] | ULIP [52] | 60.4 | 84.0 |
| **Point-BERT** [55] | **ULIP-2** (Ours) | **75.2** | **95.0** |

*Note: Regardless of the underlying architecture (PointNeXt vs. Point-BERT), ULIP-2 improves zero-shot classification performance significantly (+16.6% for PointNeXt, +14.8% for Point-BERT) [226, 227].*
