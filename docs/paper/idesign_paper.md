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

# I-Design: Personalized LLM Interior Designer

**Ata Çelen¹**, **Guo Han¹**, **Konrad Schindler¹**, **Luc Van Gool¹**, **Iro Armeni²\***, **Anton Obukhov¹\***, and **Xi Wang¹\***  
¹ ETH Zürich, ² Stanford University  
\* *Equal supervision*  
[Project Page](https://atcelen.github.io/I-Design/)

---

## Abstract
Interior design allows us to be who we are and live how we want – each design is as unique as our distinct personality. However, it is not trivial for non-professionals to express and materialize this since it requires aligning functional and visual expectations with the constraints of physical space; this renders interior design a luxury. To make it more accessible, we present **I-Design**, a personalized interior designer that allows users to generate and visualize their design goals through natural language communication. 

I-Design starts with a team of large language model (LLM) agents that engage in dialogues and logical reasoning with one another, transforming textual user input into feasible scene graph designs with relative object relationships. Subsequently, an effective placement algorithm determines optimal locations for each object within the scene. The final design is then constructed in 3D by retrieving and integrating assets from an existing object database. Additionally, we propose a new evaluation protocol that utilizes a vision-language model (VLM) and complements the design pipeline. Extensive quantitative and qualitative experiments show that I-Design outperforms existing methods in delivering high-quality 3D design solutions and aligning with abstract concepts that match user input, showcasing its advantages across detailed 3D arrangement and conceptual fidelity.

---

## 1. Introduction
Our lives are intimately connected to the spaces we inhabit [89]. Whether renting or buying, these spaces become the backdrop for our memories, hobbies, and time with loved ones [89]. However, finding or creating the perfect space to match our lifestyle, aspirations, and needs is not always straightforward, and professional assistance can be considered a luxury [89]. Even when seeking help from experts, the gap between what inhabitants truly desire and their ability to convey it in the language of professionals often leads to unsatisfactory results [89]. This discrepancy can leave people with ill-fitting living spaces, impacting their physical and mental well-being [89]. Designing interior spaces better suited for individual needs should be accessible to everyone [89].

Toward this, we tackle the challenging task of **3D Indoor Scene Synthesis (3DISS)** [91]. Given a user’s unstructured textual description of preferences, we aim to deliver 3D design solutions that align with the latter [91].

Specialized generative models [11, 21, 49] and data-driven approaches [48, 56, 66] have demonstrated a remarkable ability to produce diverse and realistic interior layouts [91]. Yet, their performance is governed by the closed-set and limited datasets used to train them, which inevitably is a biased and incomplete sample of the world [91]. In addition, such methods that accept textual input from the user can only utilize structured text with predefined grammar and rules [91]. Consequently, it is challenging to guide these models toward producing practical designs for real-world, unseen interiors that align with user preferences [91].

The 3DISS task requires reasoning abilities beyond any specialized data-driven model [92]. This includes understanding design principles and concepts like object selection, styles, and spatial arrangement [92]. However, a potential solution must succeed at several fundamental steps to become successful in 3DISS [92]. Firstly, the underlying generator must interpret the abstract input proficiently, i.e., identifying the objects to incorporate into the scene to meet user preferences [92]. Simultaneously, it should know about the typical items associated with specific room types, incorporating high-frequency and common-sense objects even if the user does not explicitly mention them [92]. In addition, it requires an awareness of plausible spatial relationships among objects originating from an unrestricted vocabulary [92]. Addressing these challenges requires a system with a comprehensive understanding of diverse human interior design preferences and an extensive database of 3D objects, encompassing their functionality, dimensions, and stylistic attributes [92].

Given recent technological advancements, the most viable option to overcome these obstacles is **large language models (LLMs)** [1, 64]. LLMs are trained on internet-scale data and encode a “world model” that can be probed and interacted with using natural language [93]. Still, they cannot use the language as humans do to solve complex tasks without resorting to specialized techniques, such as Chain-of-Thought [55]. Technical aspects (e.g., limited context window length, number of parameters) and artifacts (e.g., hallucinations) additionally hamper practical applications [93]. Attempts to produce structured output with LLMs without incorporating hard constraints, communication interfaces, and cross-checks are exciting at first glance but leave much room for improvement [93]. LayoutGPT [16], a recent work that utilizes LLMs for 3DISS, directly predicts the absolute positions of objects in the scene [93]. Although this technique may be effective when dealing with small-scale arrangements of a few objects, it proves inadequate in generating realistic scenes containing dozens of objects interlinked in intricate ways [93]. Moreover, the single-step scene generation in LayoutGPT cannot provide interpretability regarding the resulting objects and their arrangement [93].

Recent research [40, 63] has also demonstrated that when multiple LLM agents with diverse responsibilities communicate with each other, they can collectively tackle complex tasks with which a single LLM instance may struggle [94]. Hence, we utilize LLM agents to address the challenges posed by 3DISS, leading to the creation of scenes that are spatially more plausible and diverse [94]. Furthermore, drawing inspiration from other works [2, 7, 51, 66], we employ **scene graphs** as scene representations since they offer a high-level abstraction by focusing on the objects and their relationships, which can be creatively developed with LLMs, refined through rule-based feedback, and visualized [94]. Additionally, employing the scene graph representation as the interface between LLM agents and algorithms enables interpretable object arrangements [94].

To this end, we present **I-Design**, a personalized interior designer for 3D Indoor Scene Synthesis [95]. Starting from the user specification of the design preferences in plain text, I-Design queries LLM agents to come up with room items, their properties, and relative relationships in the form of a scene graph [95]. It solves for absolute object placement in the scene graph using the proposed backtracking algorithm, retrieves 3D assets according to the functional and stylistic specifications, and composes the final result in 3D [95]. To evaluate the proposed design pipeline, and following in the footsteps of [60], we propose a novel evaluation protocol based on a **vision-language model (VLM)** [95]. Extensive quantitative and qualitative experiments show that I-Design outperforms existing methods in delivering high-quality 3D design solutions that align with abstract concepts in the user input [96].

### Key Contributions:
* **Unstructured Input Processing:** A novel method that takes an unstructured, grammar-free natural language user input and provides 3D design solutions that align with user preferences [96].
* **Multi-agent Reasoning:** A new approach to the 3DISS task through the reasoning and conversation of multiple LLM agents [96].
* **Procedural Layout Generation:** A procedural scene graph layout transformation, converting scene graphs with relative node relationships into final absolute 3D representations [97].
* **Interpretable Pipeline:** An interpretable pipeline, providing flexibility and enabling iterative design without redoing the entire process [97].
* **VLM-based Evaluation:** A VLM-based evaluation protocol for 3D scenes [97].

---

## 2. Related Work

### 2.1 3DISS via LLMs
Advancements in LLMs have already impacted 3D scene synthesis, even though the connections between the two have not been fully explored [97]. Initial scene synthesis methods integrated LLMs to encode user textual input into vector representations that were subsequently used in the object placement process [30, 31, 48, 53]. Feng et al. [16] expanded the scope with LayoutGPT by exclusively employing GPT models to generate 3D indoor scene representations [97]. It functions like a retrieval system, employing a strategy based on absolute coordinates to position objects in the scene [97]. Wen et al. [58] further advance this line of research by developing AnyHome, a dataset-free, open-vocabulary approach for generating 3D home layouts [97]. Such endeavors often encounter challenges stemming from the limitations of existing GPT models regarding geometric reasoning, resulting in scenes with objects that overlap or are placed outside the scene boundaries [97].

The concept of specialized multi-agents, introduced through projects like AutoGPT [63] and ChatDev [40], has already found diverse applications in data analysis [24], interactive reasoning [29, 54], software development [40], and planning [9, 44, 45]. Our work addresses the above 3DISS challenges by reasoning with LLMs in a multi-agent setting [98].

### 2.2 3DISS via Generative Models
One formulation of 3DISS involves generating multi-view consistent image sets or panoramas according to user-defined specifications and converting these 2D representations into 3D scenes [98]. Approaches like Text2Room [21] and LucidDreamer [11] use a virtual camera to navigate the space and iteratively generate images through image inpainting, monocular depth estimation, 3D lifting, and stitching [98]. Alternatively, MVDiffusion [49] utilizes correspondence-aware attention modules to generate multi-view consistent panoramas or image sets in a single pass [98]. Notably, efforts focusing on consistent novel view synthesis are also adaptable for 3DISS [98].

Despite leveraging the capabilities of 2D generative models [14, 20, 42, 68], current methodologies struggle to integrate 3D geometric constraints that are essential for practical interior design applications, such as floor plans and walls [99]. Ctrl-Room [15] attempts to address this challenge, but artifacts persist [99]. Besides, monocular image-based depth estimation [26] and 3D lifting introduce uncertainty into the final 3D mesh output [21, 99]. Furthermore, the loose semantic coupling across views poses challenges in indoor scene synthesis, leading to unrealistic scenarios like multiple beds in one bedroom [15, 49, 99]. The LLM-based approach we employ can also generalize to different settings, such as room type or object descriptions, while simultaneously ensuring the satisfaction of geometric constraints [99].

### 2.3 3DISS through Prior Learning
Another data-driven approach to tackling 3DISS divides the process into two distinct steps: 3D asset selection and layout synthesis, i.e., determining the furniture sets for a room and their placements [100].

For **3D asset selection**, two primary approaches emerge:
1. **Dataset-based retrieval:** With the availability of extensive and high-quality asset datasets [8, 12, 19, 61], one can retrieve suitable models, for example, using semantic embedding [32, 100].
2. **Synthesis via generative models:** Generative models can generate appropriate 3D assets based on text or image inputs [22, 33, 34, 37, 100]. Given the richness of existing large-scale 3D asset repositories within our context and the direct real-world applicability of retrieving objects from product databases, we use the retrieval-based method [100].

For **furniture arrangement**, classical approaches apply specific rules to guide the placement of furniture [13, 57, 62], construct grammars for procedural modeling [25, 38, 39], or use human interaction for editing [23, 35, 101]. However, with the advent of large-scale 3D indoor datasets [19, 28, 46, 69], recent works have shifted towards learning from expert-designed layouts by employing generative models [101]. Pioneering efforts like ATISS [36] and SceneFormer [53] employ transformer-based models to synthesize indoor environments autonomously in an autoregressive manner, selecting and placing objects sequentially [101]. LEGO-NET [56] refines initial coarse room layouts by learning human criteria for regularity through a transformer-based diffusion-like pipeline [101]. Recent works like DiffuScene [48] and Commonscenes [67] utilize diffusion models to generate interior scenes, representing the scene as a scene graph [101]. To untie layout synthesis from dataset constraints, we choose to employ LLM-based methods [101]. The commonsense knowledge learned by LLMs can assist in creating a wide range of reasonable designs [101].

---

## 3. Method

### 3.1 Task & Problem Formulation
Our method for proposing design solutions from user input in plain text is summarized in Fig. 1. Initially, I-Design examines unstructured textual user input and transforms it into a viable design proposal, represented as a scene graph, through querying LLM agents (Sec. 3.2). We then introduce the scene graph layout module for producing object placement proposals represented by a 2D room floor plan (Sec. 3.3). Finally, we retrieve 3D assets from existing databases (Sec. 3.4) and assemble the final design within a 3D environment, producing functional and stylistic design solutions that reflect user input (Sec. 3.5).

Given a non-structured textual user input $T_{	ext{user}}$, the dimensions of a room $(l_{	ext{room}}, w_{	ext{room}}, h_{	ext{room}}) \in \mathbb{R}^3$, and the number of objects to include in the scene $n \in \mathbb{R}$, the objective is to create a 3D scene that aligns with the free-form user’s requests in the textual input, while ensuring functionality, coherent design, and 3D consistency [103].

We use a scene graph $G = (O, E)$ to represent the spatial relationship between objects, where nodes $O = [o_1, \dots, o_n]$ represent object instances in the scene [103]. An additional node type $o^{(r)}$ is included for room layout elements such as walls, ceiling, or the floor [103]. Each object $o_i$ is associated with a set of properties:
$$o_i = \{lpha_i, m_i, s_i, r_i, p_i, cs_i\}$$
where $lpha_i$ denotes the object's name, and $m_i$ describes its material and architectural style [103]. The associated geometric properties are:
* $s_i \in \mathbb{R}^3$: the dimensions of its bounding box [103].
* $r_i \in \mathbb{R}$: the rotation angle along the z-axis pointing upwards [104].
* $p_i \in \mathbb{R}^3$: the position of the object in the scene [104].
* $cs_i = (x_{	ext{neg}}, x_{	ext{pos}}, y_{	ext{neg}}, y_{	ext{pos}}) \in \mathbb{R}^4$: the size of the subgraph's bounding box, indicating the overall space that an object, along with its children, would occupy in each direction [104].

Edges in the graph are represented by $E$ where $e_{ij} = (	ext{adj}_{ij}, 	ext{prep}_{ij})$ is the directed edge from $o_i$ to $o_j$ that comprises details regarding the adjacency of the two objects and the prepositional connection between them [104]. $o_i$ is considered as the parent node to the child node $o_j$ [104]. These edges describe the spatial locations between objects (e.g., left/right, in front/behind, on, and above/under) as well as the connections to the room layout (e.g., on and in the corner) [104].

### 3.2 LLM Multi-agents Pipeline
Generating a 3D indoor scene based on unstructured user input is a challenging task that demands detailed planning [105]. Two essential steps are shared among common methods: selecting the objects to populate the scene based on user input and arranging them in a meaningful and coherent configuration [105]. We employ a multi-agent approach to tackle this complexity effectively, allocating the tasks involved in these steps across multiple LLM agents [105]. 

As shown in Fig. 2, interpreting and transforming user input into a functional, personalized scene graph involves five distinct agents: 
1. **Interior Designer**
2. **Interior Architect**
3. **Engineer**
4. **Layout Corrector**
5. **Layout Refiner**

Due to the specialized nature of each agent, the generated scene graph reflects user specifications and feasibility constraints, such as topological correctness and semantic plausibility [106].

```
+--------------+     +-------------------+     +--------------+     +------------------+     +----------------+
|  User Prompt | --> | Interior Designer | --> |  Int. Arch.  | --> |     Engineer     | --> | Layout Correc. |
+--------------+     +-------------------+     +--------------+     +------------------+     +----------------+
                                                                                                     |
                                                                                                     v
                                                                                             +----------------+
                                                                                             | Layout Refiner |
                                                                                             +----------------+
                                                                                                     |
                                                                                                     v
                                                                                             [Final Scene Graph]
```

#### Multi-agent Roles:
* **Interior Designer:** Receives inputs comprising the free-form user input $T_{	ext{user}}$, the room dimensions $(l_{	ext{room}}, w_{	ext{room}}, h_{	ext{room}})$, and the desired number of objects $n$, and proposes a selection of objects $\{o_i\}$ tailored to the user’s preferences, while also ensuring that their functionality matches the room type [107]. Specifically, for each object, the interior designer suggests the name $lpha_i$, material $m_i$, 3D size $s_i$, orientation $r_i$, and position $p_i$ [107]. It may propose several identical instances (e.g., four instances of "chair" around a table) [107].
* **Interior Architect:** Primary responsibility is to establish object-to-object connections, as well as object-to-room layout relations [108]. In other words, its role is to establish the edge connections $e_{ij}$ between the object nodes [108]. The agent is not constrained on the number of edges each object can have, allowing for diverse configurations where an object may have edges to room layout elements but none to other objects, and vice versa [108]. Additionally, it determines information on the rotation $r_i$ of each object around the vertical axis [108].
* **Engineer:** This agent transforms the relative scene graph into a JSON object structured according to a specified schema [109]. Each entry in the JSON file encompasses the details for $o_i = \{lpha_i, m_i, s_i, r_i, p_i\}$ [109]. Moreover, the agent employs a JSON schema validator that assesses the validity of the generated file based on the provided schema [109]. In cases of non-compliance, a modification from the Engineer is required until the output aligns with the specified schema, ensuring a valid JSON representation [109].
* **Layout Corrector:** The responsibility of this agent is to fix invalid connections in the graph, which involves removing spatially implausible edges and eliminating ambiguities between nodes [110]. We preemptively examine three types of spatial implausibilities:
  1. *Room Boundaries Check:* Checks children nodes of objects allocated along walls or in room corners [111]. For example, if object A is positioned behind object B, which is placed alongside a wall, this would result in object A being positioned out of bounds [111].
  2. *Spatially Impossible Object-to-Object Connections:* Verifies whether adjacency relations between objects could be violated via conflicting edges in the scene graph (e.g., checking if any object has been positioned between two adjacent objects) [111].
  3. *Size Compatibility Check:* Operates under the assumption that, in a valid scene graph, the bounding box of the parent object is sufficiently spacious to accommodate its children [112]. For example, if a table has two chairs positioned on its left side, it should be wide enough to accommodate them [112]. The Layout Corrector determines the relocation of objects experiencing spatial conflicts, either suggesting alternative edges for these objects or removing them entirely from the scene [110].
* **Layout Refiner:** Aims to eliminate ambiguities between children nodes that share the same edge from the parent [112]. A notable example of this phenomenon is observed with ornaments placed on a desk [112]. As a desk typically accommodates several objects on its surface, a scene graph designating the desk as the parent with the preposition `on` as the edge fails to convey the relative orientation of the objects on the desk to each other [112]. To eliminate this ambiguity, we establish edges between children nodes, ensuring a distinct ordering among them [112]. The agent also verifies that the scene graph maintains its acyclic property [112]. If the graph contains cycles, we must remove edges contributing to these cycles to preserve a distinct hierarchy within the scene graph [112].

### 3.3 Scene Graph Layout

#### Computing Cluster Dimensions
In the final postprocessing phase, we compute cluster dimensions for each object to facilitate the positioning in the subsequent backtracking algorithm, thereby strategically constraining the search space [113]. Leveraging that $G$ is a Directed Acyclic Graph, we perform a topological sort on the nodes to establish a hierarchical ordering [113]. This approach enables us to determine the clearance – the distance an object must spare in each direction to ensure the successful placement of its children within the scene [113]. Taking such precaution becomes crucial in scenes where the search space for each object is extensive, yet the solution space is relatively constrained [114]. Without this precaution, the randomized nature of the backtracking algorithm may lead to extended processing times, particularly when dealing with many object relationships in the scene [114, 115].

#### The Backtracking Algorithm
The backtracking algorithm serves to convert the relative representation of the scene graph $G$ into absolute object positions $P$ [115]. The room is initially populated with the root nodes of the scene graph $G$, representing the fundamental elements of the room layout, such as walls and ceiling [115]. The positions of these root nodes remain fixed while other objects are arranged around them [115]. Conceptually, the algorithm represents the plausible position of each object as a bounding box $B_i$ and samples a point $p_i$ for each $o_i$ from $B_i$ [115].

The objects are placed after being topologically sorted, prioritizing nodes higher in the scene graph hierarchy for placement first [116]. To avoid placement issues later in the algorithm, each object $o_i$ is positioned alongside its children according to $G_i$ [116]. The plausible positions bounding box for the objects is defined through predetermined placement functions $f(o_i, o_j), orall o_j, o_i \in G_j$ [116].

The nodes are organized into depth groups, where a node’s depth $d = |o_i 	o o^{(r)}|$ represents its distance to the closest $o^{(r)}$ [116]. If an object $o_i$ with depth $d$ cannot be placed into the scene due to either an empty bounding box $B_i = \emptyset$ or if $p_i$ consistently results in collisions with other objects, we remove $p_j$ for all objects with depth $\ge d$ [116]. Subsequently, we re-sample positions for objects with depth $d-1$ [116]. Once all objects with depth $d$ are positioned successfully, we increment the depth counter of the algorithm [116].

### 3.4 3D Asset Retrieval
We generate textual descriptions for each object by utilizing the object name, style, and material information [117]. These textual descriptions are then transformed into text embeddings using the CLIP [41] text encoder [117]. The alignment between OpenShape [32] encodings and CLIP embeddings enables us to measure the distance between our text embeddings and the learned object representations from the database of choice [117]. This facilitates the retrieval of a 3D asset that is closest to our textual description of each object from the database (such as Objaverse) [117, 119]. After the retrieval, the assets can be adjusted/rescaled to fit the bounding box provided to these objects within the scene graph [117].

### 3.2 3D Composer
Having produced the final scene graph and retrieved the 3D assets, we employ an off-the-shelf 3D renderer (Blender [4]) to visualize the room interior in 3D [118, 119]. The output of the I-Design pipeline is a bundle of entities, including:
1. The scene graph [118].
2. The floor plan [118].
3. Preconfigured rendered views [118].
4. Interpretability artifacts, such as the input user prompt and the communication log between the agents [118].

This output bundle allows for the inspection of every pipeline stage and, if necessary, enables the replay of select stages to introduce variations (e.g., swapping a piece of furniture or rendering the scene from a novel viewpoint) [118].

---

## 4. Experiments

### 4.1 Implementation Details
We use Microsoft’s **AutoGen** [59] framework to enable multi-agent conversation, with each agent equipped with a **GPT-4** model [1, 119]. The temperature is configured at 0.7, while `top_p` is set to 1.0 [119]. For object retrieval, we rely on **OpenShape** [32], utilizing text embeddings to retrieve objects from **Objaverse** [12, 119]. The scene is then visualized and rendered using **Blender** [4, 119].

### 4.2 Quantitative Evaluation

#### Metrics:
* **Average Number of Proposed Objects (NObj):** Measures the diversity/richness of the generated rooms. Computed separately for bedrooms and living rooms [120].
* **Out-of-Boundary Rates (OOB):** Frequency of objects extending beyond room boundaries. If any bounding boxes are outside the designated room boundaries within a scene, that scene is deemed invalid. OOB is calculated as the ratio of invalid scenes to the total number of generated scenes [120].
* **Bounding Box Loss (BBL):** Evaluates the degree of overlap between proposed furniture bounding boxes, calculated as the average volume of bounding box intersections across generated scenes [120].
* **GPT-4V ratings:** Inspired by GPT-4V acting as a human-aligned evaluator for 3D content, we employ GPT-4V to assess synthesized rooms based on their renderings [120]. Ratings range from 0 to 10 across four dimensions:
  1. *Functionality and activity-based alignment (Func.)* [120].
  2. *Layout and furniture (Layout.)* [120].
  3. *Color scheme and material choice (Scheme.)* [120].
  4. *Overall aesthetic and atmosphere (Atmos.)* [120].
  Renderings from two viewpoints are horizontally concatenated and passed along with the user input to the evaluator [120].

#### Baseline:
We compare against **LayoutGPT** [16], which generates room layouts in a CSS-like format through few-shot learning with examples from the 3D-FRONT dataset [121]. For a fair comparison, we extended LayoutGPT to retrieve and place objects from Objaverse using the same rendering settings [121].

#### Table 1: Quantitative comparison with LayoutGPT [16].
Both methods generate 10 "Bedroom" and 10 "Living room" scenes of varying room sizes [123].

| Method | Room Type | NObj ↑ | OOB ↓ (%) | BBL ↓ | Func. ↑ | Layout. ↑ | Scheme. ↑ | Atmos. ↑ | Avg. ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LayoutGPT** [16] | Bedroom | 5.5 | 51.06 | 14.09 | 4.8±0.4 | 4.8±0.8 | 4.6±0.8 | 4.9±0.4 | 4.8 |
| | LivingRoom | 6.9 | 64.15 | 1.06 | 4.8±1.3 | 4.8±0.8 | 4.8±1.4 | 4.6±0.8 | 4.8 |
| | **Avg.** | **6.2** | **57.6** | **7.58** | **4.8** | **4.8** | **4.7** | **4.8** | **4.8** |
| **Ours (I-Design)** | Bedroom | 12.7 | 0.0 | 0.34 | 5.2±0.4 | 5.5±0.2 | 5.6±0.7 | 5.5±0.2 | 5.5 |
| | LivingRoom | 23.6 | 0.0 | 0.31 | 5.8±2.6 | 5.6±2.1 | 5.9±1.1 | 5.7±1.3 | 5.8 |
| | **Avg.** | **18.2** | **0.0** | **0.33** | **5.5** | **5.6** | **5.8** | **5.6** | **5.7** |

Compared to LayoutGPT, our approach stands out in proposing larger furniture sets and more physically plausible layouts, exhibiting **zero OOB** and minimal **BBL** values [126]. Our method outperforms LayoutGPT across all GPT-4V metrics [126].

#### Prompt Diversity Evaluation
We analyze the performance of I-Design under different prompt categories (Atmospheric, Scheme, Layout, and Functional) [122, 125]. We generate 10 prompts for each category and report the GPT-4V grades [122, 125].

#### Table 2: Quantitative evaluation of I-Design with various prompt types [124].
| Prompt Type | Func. | Layout. | Scheme | Atmos. | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Atmospheric** | 4.9±0.7 | 4.9±0.6 | 4.3±0.6 | 4.4±0.5 | 4.9 |
| **Scheme** | 4.6±0.5 | 4.8±0.6 | 5.0±0.5 | 4.8±0.5 | 5.0 |
| **Layout** | 6.2±0.6 | 6.0±0.7 | 5.2±0.7 | 5.6±0.6 | 5.8 |
| **Functional** | **7.0±0.7** | **6.5±0.7** | **5.8±0.8** | **6.2±0.7** | **6.4** |

Our method yields the best results in all four grading aspects when incorporating functional descriptions [127]. Functionality prompts provide cues regarding room type and potential activities [127]. The usage of atmosphere and scheme information somewhat diminishes the quality of synthesis [127]. We attribute this to our pipeline's lack of consideration for the materials and textures of the walls, ceiling, and floor, and the absence of an object re-texturing step [127].

### 4.3 User Study
We conducted a user study to reinforce our GPT-4V evaluation [129]. From 20 scenes created by our method and 20 by LayoutGPT, we randomly sampled 5 scenes to compose 100 pairs [129]. Subjects were instructed to vote on the most realistic room in each pair (order randomized) [129]. Two control questions with predefined correct answers were included for verification (detecting object collisions and level of detail) [128, 129]. We collected 1,254 answers for bedroom pairs and 660 for living room pairs, converting votes into Bradley-Terry preference scores and probabilities [129].

#### Table 3: Subjective Study of user preferences.
| Method | Room Type | Bradley-Terry Score ↑ | Probability ↑ |
| :--- | :--- | :---: | :---: |
| **LayoutGPT** | Bedroom | 0.12 | 0.42 |
| | LivingRoom | 0.17 | 0.38 |
| **Ours (I-Design)** | Bedroom | **0.40** | **0.58** |
| | LivingRoom | **0.69** | **0.62** |

The results affirm our findings with the GPT-4V evaluator, with I-Design outperforming LayoutGPT for both room types [129].

### 4.4 Qualitative Study
The gallery showcasing rooms generated with I-Design provided with various prompts is available in Fig. 4. Each synthesized room adeptly reflects specifications from user textual inputs [130]. When users explicitly mention furniture preferences and desired positions, the resulting room seamlessly aligns [130]. When specifications are implicit (e.g., a bedroom tailored for a family with a toddler), the synthesized room's functionality satisfies preferences, with a crib placed next to the king-sized bed [130].

---

## 5. Conclusion & Discussion

### 5.1 Summary
We presented **I-Design**, an interior design assistant designed to simplify the interior design process for individuals without expertise [131]. By leveraging an LLM multi-agent architecture, our framework interprets user preferences expressed through text, transforming unstructured text into structured scene graphs [132]. Subsequently, spatial relationships are established, integrating common-sense knowledge into the agents' decision-making process [132]. Crucially, our framework ensures interpretability through a transparent breakdown of each step, facilitating an accessible design process [132].

### 5.2 Limitations & Future Work
* **Placement Termination Problems:** The pipeline may fail to find a solution for object placements when handling many objects in a relatively small scene [133]. Spatial conflicts may persist, or there may not be enough space for furniture placements [133].
* **Asset Retrieval Orientation & Scale Discrepancies:** The quality and default canonical orientations of retrieved assets may vary, leading to resizing artifacts or incorrect facing directions (e.g., determining the front side of a desk can be subjective) [133, 167, 168].
* **Aesthetic Texture Incoherence:** The original textures of retrieved objects cannot be guaranteed to align seamlessly [168]. Achieving precise control over texture remains challenging, and future work will incorporate a re-texturing step [168].
* **Simplistic Bounding Box Assumption:** We place objects assuming quadratic surfaces [169]. While this simplifies spatial representation, it can lead to floating child objects on top of parents [169]. Future work will explore using mesh surfaces instead of bounding boxes for precise placement [169].
* **Placement Optimization:** We plan to explore an automated, learning-based approach to replace the current backtracking algorithm that relies on trial and error [134]. We also plan to adopt a generative approach to complement object retrieval, improving the diversity and flexibility of the overall pipeline [134].

---

## References
1. Achiam, J., et al.: GPT-4 technical report. *arXiv preprint arXiv:2303.08774* (2023).
2. Armeni, I., et al.: 3d scene graph: A structure for unified semantics, 3d space, and camera. *In: ICCV* (2019).
3. Bautista, M.A., et al.: GAUDI: A neural architect for immersive 3d scene generation. *In: NeurIPS* (2022).
4. Blender Online Community: Blender - a 3D modelling and rendering package. *Blender Foundation* (2018).
5. Bradley, R.A., Terry, M.E.: Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika* (1952).
6. Cai, S., et al.: Diffdreamer: Towards consistent unsupervised single-view scene extrapolation with conditional diffusion models. *In: ICCV* (2023).
7. Chang, A., et al.: Text to 3d scene generation with rich lexical grounding. *arXiv preprint arXiv:1505.06289* (2015).
8. Chang, A.X., et al.: Shapenet: An information-rich 3d model repository. *arXiv preprint arXiv:1512.03012* (2015).
9. Chen, G., et al.: AutoAgents: A framework for automatic agent generation. *arXiv preprint arXiv:2309.17288* (2023).
10. Ching, F.D., Binggeli, C.: Interior design illustrated. *John Wiley & Sons* (2018).
11. Chung, J., et al.: LucidDreamer: Domain-free generation of 3d gaussian splatting scenes. *arXiv preprint arXiv:2311.13384* (2023).
12. Deitke, M., et al.: Objaverse: A universe of annotated 3d objects. *In: CVPR* (2023).
13. Deitke, M., et al.: ProcTHOR: Large-scale embodied ai using procedural generation. *In: NeurIPS* (2022).
14. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. *In: NeurIPS* (2021).
15. Fang, C., et al.: Ctrl-Room: Controllable text-to-3d room meshes generation with layout constraints. *arXiv preprint arXiv:2310.03602* (2023).
16. Feng, W., et al.: Layoutgpt: Compositional visual planning and generation with large language models. *arXiv preprint arXiv:2305.15393* (2023).
17. Fridman, R., et al.: SceneScape: Text-driven consistent scene generation. *In: NeurIPS* (2024).
18. Fu, H., et al.: 3D-FRONT: 3d furnished rooms with layouts and semantics. *In: ICCV* (2021).
19. Fu, H., et al.: 3D-FUTURE: 3d furniture shape with texture. *International Journal of Computer Vision* (2021).
20. Ho, J., et al.: Denoising diffusion probabilistic models. *In: NeurIPS* (2020).
21. Höllein, L., et al.: Text2Room: Extracting textured 3d meshes from 2d text-to-image models. *arXiv preprint arXiv:2303.11989* (2023).
22. Hong, Y., et al.: LRM: Large reconstruction model for single image to 3d. *arXiv preprint arXiv:2311.04400* (2023).
23. Huang, I., et al.: Aladdin: Zero-shot hallucination of stylized 3d assets from abstract scene descriptions. *arXiv preprint arXiv:2306.06212* (2023).
24. Jin, Q., et al.: GeneGPT: Augmenting large language models with domain tools for improved access to biomedical information. *arXiv preprint arXiv:2304.09667* (2023).
25. Kar, A., et al.: Meta-Sim: Learning to generate synthetic datasets. *In: ICCV* (2019).
26. Ke, B., et al.: Re-purposing diffusion-based image generators for monocular depth estimation. *In: CVPR* (2024).
27. Lee, K.T., et al.: Conceptual framework to support personalized indoor space design decision-making: A systematic literature review. *Buildings* (2022).
28. Li, W., et al.: InteriorNet: Mega-scale multi-sensor photo-realistic indoor scenes dataset. *In: BMVC* (2018).
29. Lin, B.Y., et al.: Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks. *arXiv preprint arXiv:2305.17390* (2023).
30. Lin, C., Mu, Y.: Instructscene: Instruction-driven 3d indoor scene synthesis with semantic graph prior. *arXiv preprint arXiv:2402.04717* (2024).
31. Liu, J., et al.: Clip-layout: Style-consistent indoor scene synthesis with semantic furniture embedding (2023).
32. Liu, M., et al.: OpenShape: Scaling up 3d shape representation towards open-world understanding. *In: NeurIPS* (2023).
33. Liu, M., et al.: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. *In: NeurIPS* (2023).
34. Long, X., et al.: Wonder3D: Single image to 3d using crossdomain diffusion. *arXiv preprint arXiv:2310.15008* (2023).
35. Merrell, P., et al.: Interactive furniture layout using interior design guidelines. *ACM TOG* (2011).
36. Paschalidou, D., et al.: ATISS: Autoregressive transformers for indoor scene synthesis. *In: NeurIPS* (2021).
37. Poole, B., et al.: DreamFusion: Text-to-3d using 2d diffusion. *arXiv preprint arXiv:2209.14988* (2022).
38. Purkait, P., et al.: SG-VAE: Scene grammar variational autoencoder to generate new indoor scenes. *In: ECCV* (2020).
39. Qi, S., et al.: Human-centric indoor scene synthesis using stochastic grammar. *In: CVPR* (2018).
40. Qian, C., et al.: Communicative agents for software development. *arXiv preprint arXiv:2307.07924* (2023).
41. Radford, A., et al.: Learning transferable visual models from natural language supervision. *arXiv preprint arXiv:2103.00020* (2021).
42. Rombach, R., et al.: High-resolution image synthesis with latent diffusion models. *In: CVPR* (2022).
43. Shen, L., et al.: Make-it-4d: Synthesizing a consistent long-term dynamic scene video from a single image. *In: ACM MM* (2023).
44. Shen, Y., et al.: HuggingGPT: Solving AI tasks with ChatGPT and its friends in Hugging Face. *arXiv preprint arXiv:2303.17580* (2023).
45. Song, C.H., et al.: LLM-Planner: Few-shot grounded planning for embodied agents with large language models. *In: CVPR* (2023).
46. Song, S., et al.: Semantic scene completion from a single depth image. *In: CVPR* (2017).
47. Subjectify.us: Crowd-sourced subjective quality evaluation platform (Accessed: Jan 2024).
48. Tang, J., et al.: Diffuscene: Scene graph denoising diffusion probabilistic model for generative indoor scene synthesis. *arXiv preprint arXiv:2303.14207* (2023).
49. Tang, S., et al.: MVDiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. *arXiv* (2023).
50. Tseng, H.Y., et al.: Consistent view synthesis with pose-guided diffusion models. *In: CVPR* (2023).
51. Wang, K., et al.: PlanIT: Planning and instantiating indoor scenes with relation graph and spatial prior networks. *ACM TOG* (2019).
52. Wang, T., et al.: Breathing new life into 3d assets with generative repainting. *In: BMVC* (2023).
53. Wang, X., et al.: Sceneformer: Indoor scene generation with transformers. *In: 3DV* (2021).
54. Wang, Z., et al.: Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models. *arXiv preprint arXiv:2311.05997* (2023).
55. Wei, J., et al.: Chain-of-thought prompting elicits reasoning in large language models. *In: NeurIPS* (2022).
56. Wei, Q.A., et al.: LEGO-Net: Learning regular rearrangements of objects in rooms. *In: CVPR* (2023).
57. Weiss, T., et al.: Fast and scalable position-based layout synthesis. *IEEE TVCG* (2018).
58. Wen, Z., et al.: AnyHome: Open-vocabulary generation of structured and textured 3d homes. *arXiv preprint arXiv:2312.06644* (2023).
59. Wu, Q., et al.: AutoGen: Enabling next-gen LLM applications via multi-agent conversation framework. *arXiv preprint arXiv:2308.08155* (2023).
60. Wu, T., et al.: GPT-4V (ision) is a human-aligned evaluator for text-to-3d generation. *arXiv preprint arXiv:2401.04092* (2024).
61. Wu, Z., et al.: 3d shapenets: A deep representation for volumetric shapes. *In: CVPR* (2015).
62. Xu, K., et al.: Constraint-based automatic placement for scene composition. *In: Graphics Interface* (2002).
63. Yang, H., et al.: Auto-GPT for online decision making: Benchmarks and additional opinions. *arXiv preprint arXiv:2306.02224* (2023).
64. Yang, J., et al.: Harnessing the power of llms in practice: A survey on chatgpt and beyond. *ACM TKDD* (2023).
65. Yu, H.X., et al.: WonderJourney: Going from anywhere to everywhere. *arXiv preprint arXiv:2312.03884* (2023).
66. Zhai, G., et al.: CommonScenes: Generating commonsense 3d indoor scenes with scene graphs. *arXiv preprint arXiv:2305.16283* (2023).
67. Zhai, G., et al.: CommonScenes: Generating commonsense 3d indoor scenes with scene graphs. *In: NeurIPS* (2024).
68. Zhang, L., et al.: Adding conditional control to text-to-image diffusion models. *In: ICCV* (2023).
69. Zheng, J., et al.: Structured3D: A large photo-realistic dataset for structured 3d modeling. *In: ECCV* (2020).

---

## Appendices / Supplementary Material

### Section 6: Additional Qualitative Results
Figures 6, 7, and 8 of the original paper showcase various qualitative outcomes of our framework:
* **Fig. 6 (Living Room Renders with Generic Prompting):** Showcases room renders used for comparison with LayoutGPT [16], generated using the generic prompt "Design me a living room." [159]
* **Fig. 7 (Generating Renders through Elaborate Prompting):** Shows samples evaluated across different prompt categories, such as Atmosphere, Scheme, Layout, and Functionality [159].
* **Fig. 8 (Synthesized Scene Graph and Corresponding Renders):** Illustrates the end-to-end transformation of a user prompt into a scene graph, and subsequently into a beautifully composed 3D layout [160].

### Section 7: Implementation of the Multi-agent Pipeline

#### Divide & Conquer Approach
The primary motivation for distributing the tasks among multiple specialized LLM agents is the observed performance boost when each LLM instance focuses on solving a more trivial sub-problem [161]. The agents iteratively refine the scene by attending to the outputs of preceding agents in the communication flow [161]. Introducing a **feedback loop** to the communication process effectively helps eliminate unwanted behaviors and mitigates hallucinations [161].

#### GPT-4 Token Limit Constraints
Task distribution also addresses the hard token output limit of current LLMs [161]. GPT-4 supports an output limit of **4,000 tokens**, which constrains the volume of object information an LLM can generate in a single turn [161]. Approaching this token limit degrades the LLM's attention to instructions, causing object suggestions or placements to drift from user preferences [161]. Distributing information processing across multiple agents successfully mitigates this issue, enabling the stable generation of complex layouts with dozens of objects [161].

#### Execution Flow
The Interior Designer and Interior Architect agents generate and process all objects together in a single pass to allow the agents to attend to each object effectively [162]. In contrast, the Engineer agent processes objects individually and iteratively, structuring them into the target JSON format independently [163]. All agents utilize GPT-4's **"JSON mode"** to restrict outputs exclusively to valid JSON, which significantly reduces token consumption [162].

### Section 8: Implementation of the GPT-4V Evaluator
The GPT-4V evaluation scheme provides a reliable metric for measuring and comparing abstract concepts (e.g., atmosphere, scheme) using rendered views [164]. We horizontally concatenate two corner renders of a scene and prompt GPT-4V to grade the scene across multiple dimensions [164]. The evaluator is executed three times per scene, and grades are averaged to reduce stochasticity [164]. 

To stimulate a **Chain-of-Thought (CoT)** process, we instruct the evaluator to comment on the visual aspects of the scenes before outputting the final numerical grades, which enhances grading accuracy and consistency [165].

#### Section 8.1 GPT-4V Evaluator System Prompt
```
Give a grade from 0 to 10 to the following room renders based on how well they correspond together to the user preference (in triple backquotes) in the following aspects:
- Realism and 3D Geometric Consistency
- Functionality and Activity-based Alignment
- Layout and furniture
- Color Scheme and Material Choices
- Overall Aesthetic and Atmosphere

User Preference: "{prompt}"
Return the results in the following JSON format: "{example_json}"
```

### Section 9: System Prompts for Agents
Creating the scene graph involves various agents, each playing a distinct role within the overall generation process [170].Verbatim system prompts for each agent are provided below:

#### 9.1 Interior Agent System Prompt (Interior Designer)
> *"Interior Designer. Suggest {n} essential new objects to be added to the room based on the user preference, the general functionality of the room, and the room size. The suggested objects should contain the following information: [171]*
> *1. Object name (ex., bed, desk, chair, monitor, bookshelf, etc.)*
> *2. Architecture style (ex., modern, classic, etc.)*
> *3. Material (ex., wood, metal, etc.)*
> *4. Bounding box size in meters (ex., Length: 1.0m, Width: 1.0m, Height: 1.0m). Only use “Length”, “Width”, and “Height” as keys for the bounding box size.*
> *5. Quantity (ex., 1, 2, 3, etc.)*
> *IMPORTANT: Do not suggest any objects related to doors or windows, such as curtains, blinds, etc.*
> *Follow the JSON schema below: {json_schema}" [171]*

#### 9.2 Interior Architect System Prompt
> *"Interior Architect. Your role is to analyze user preferences, consider the optimal placement for each object that the Interior Designer suggests, find a place for this object in the room, and give a detailed description of it. If the quantity of an object is greater than one, you have to find a place for each instance of this object separately but give all this information in one list item. Give explicit answers for EACH object on the following three aspects: [172]*
> 
> *Placement: Find a relative place for the object (e.g., in the middle of the floor, in the northwest corner, on the east wall, right of the desk, on the bookshelf). For relative placement with other objects in the room, use the prepositions “on”, “left of”, “right of”, “in front”, “behind”, “under”. For relative placement with the room layout elements (walls, the middle of the room, ceiling), use the prepositions “on” and “in the corner”. You are not allowed to use any prepositions other than the ones above. Explicitly state the placement for each instance (ex., one is on the left of desk_1, one is on the south_wall). [173]*
> 
> *Proximity : Proximity of this object to the relative placement objects: 1. Adjacent: The object is physically contacting the other object, or the other object supports it, or they are touching, or they are close to each other. 2. Not Adjacent: The object is not physically contacting the other object and is distant from it. [174]*
> 
> *Facing : Think about which wall (west/east/north/south_wall) this object should be facing and explicitly state this (ex., one is facing the south_wall, one is facing the west_wall). [174]*
> 
> *If the quantity of an object is greater than one, you have to find a place for each instance of this object separately but give all this information in one list item. [175]*
> *Follow the JSON schema below: {json_schema}"*

#### 9.3 Engineer System Prompt
> *"Engineer. You listen to the input by the Admin and create a JSON file. [175]*
> *When the Admin outputs objects to be in the room, you will save ALL of them in the given schema. For the scene graph, you can use the IDs for the objects already in the room but only output the objects to be placed. If an object has a quantity higher than one, save each instance of this object separately. [175]*
> 
> *IMPORTANT: The inputted “Placement” key should be used for the “placement” key in the JSON object. Follow exactly the prepositions stated; do not use the information in the “Facing” key for the room layout elements. [176]*
> *IMPORTANT: For object quantities greater than one, the “placement” key gives separately the relative placement of each instance of that object in the room; make the distinction for each instance accordingly.*
> *Use only the following JSON Schema to save the JSON object: {json_schema}" [176]*

#### 9.4 Layout Corrector System Prompt
> *"Layout Corrector Agent. Whenever a user provides an object that doesn’t fit the room for various spatial conflicts, you will change its “scene_graph” and “facing_object” keys to resolve these conflicts. [177]*
> *You will use the JSON Schema to validate the user’s JSON object.*
> *For relative placement with other objects in the room, use the prepositions “on”, “left of”, “right of”, “in front”, “behind”, “under”. For relative placement with the room layout elements (walls, the middle of the room, ceiling), use the prepositions “on”, and “in the corner”. [177]*
> *Use only the following JSON Schema to save the JSON object: {json_schema}" [178]*

#### 9.5 Layout Refiner System Prompt
> *"Layout Refiner. Whenever the Admin speaks, you will look at the parent object and children objects, the first preposition that connects these objects, and find a second suitable relative placement for the children objects while considering the initial positioning of the object. Give the relative placement of the children objects with each other and with the parent object. For example, if there are five children objects “on” the parent object, give the relative positions of the children objects to one another and the second preposition to the parent object (“on” is the first preposition). [178]*
> *Use only the following JSON Schema to save the JSON object: {json_schema}" [179]*

---

### Section 10: Room Synthesis Prompt Generation

#### Section 10.1 Prompts for Room Preferences Text Generation
* **Minimal Preference Generation Prompt (Bedroom):** *"You are a helpful assistant. Could you please provide a list of common dimensions for a bedroom? Please list ten potential dimensions, including width, length, and height in meters. Format your response as follows: [[width, length, height], [width, length, height], ...]" [182]*
* **Minimal Preference Generation Prompt (Living Room):** *"You are a helpful assistant. Could you please provide a list of common dimensions for a living room? Please list ten potential dimensions, including width, length, and height in meters. Format your response as follows: [[width, length, height], [width, length, height], ...]" [183]*
* **Functionality-related Preference Generation Prompt:** *"You are a helpful assistant who is designed to output JSON. Please provide ten interior design instructions emphasizing functionality. Begin by specifying the room type and desired room dimensions in meters, including width, length, and height. Describe the room’s intended functionality succinctly. Besides, common suggestions also extend to more diverse requirements, such as creating a reading corner or a movie area. Keep descriptions brief, within three sentences, and exclude details involving windows and doors. Aim for diversity in interior room types. Provide the results in JSON format: {“room type”: {“dimension”: [width, length, height], “functionality”: “functionality description”}, ...} Note: you need to replace the key “room type” with specified room type, such as bedroom, living room etc." [183]*
* **Layout-related Preference Generation Prompt:** *"You are a helpful assistant designed to output JSON. Could you outline ten descriptions of potential bedroom interior design layouts? Please do not include descriptions about style, theme, etc, and only focus on layout. Please present the results in JSON format as follows: {“1”: “layout and furniture description”, “2”: “layout and furniture description”, ...}. Please keep each description concise with two to three sentences." [184]*
* **Color-scheme-and-material-related Preference Generation Prompt:** *"You are a helpful assistant designed to output JSON. Can you list ten interior bedroom design requirements or ideas with different color themes (e.g., room with pink color colors or beige tones) and material emphasis (e.g., room with wooden elements or metallic elements)? Please do not use windows, ceilings, floors, and walls in the descriptions. Provide the results in JSON format: {“1”: “requirement description”, “2”: “requirement description”, ...}. Please keep the description short, with no more than four sentences." [184]*
* **Overall-atmosphere-related Preference Generation Prompt:** *"You are a helpful assistant designed to output JSON. Can you provide ten concise descriptions of the overall atmosphere for potential bedroom interior designs? Please present the results in JSON format as follows: {“1”: “atmosphere description”, “2”: “atmosphere description”, ...}. Please keep each description concise within two to three sentences." [185]*

#### Section 10.2 Complete Lists of Prompts

#### Table 4: List of Minimal Prompts for Evaluation (Tab. 1) [186]
| Index | Prompt | Room Dimension |
| :--- | :--- | :---: |
| 1 | Design me a bedroom. | [3.0, 4.0, 2.4] |
| 2 | Design me a bedroom. | [2.5, 3.0, 2.4] |
| 3 | Design me a bedroom. | [3.5, 4.5, 2.4] |
| 4 | Design me a bedroom. | [4.0, 5.0, 2.4] |
| 5 | Design me a bedroom. | [2.4, 3.5, 2.4] |
| 6 | Design me a bedroom. | [3.2, 4.2, 2.4] |
| 7 | Design me a bedroom. | [2.8, 3.6, 2.4] |
| 8 | Design me a bedroom. | [3.6, 4.8, 2.4] |
| 9 | Design me a bedroom. | [4.2, 5.2, 2.4] |
| 10 | Design me a bedroom. | [3.0, 3.5, 2.4] |
| 11 | Design me a living room. | [4.0, 5.0, 2.8] |
| 12 | Design me a living room. | [3.5, 4.5, 2.8] |
| 13 | Design me a living room. | [3.0, 4.0, 2.8] |
| 14 | Design me a living room. | [4.5, 6.0, 3.0] |
| 15 | Design me a living room. | [5.0, 7.0, 3.0] |
| 16 | Design me a living room. | [3.6, 4.8, 2.8] |
| 17 | Design me a living room. | [4.2, 5.2, 2.8] |
| 18 | Design me a living room. | [5.5, 6.5, 3.0] |
| 19 | Design me a living room. | [3.2, 4.2, 2.8] |
| 20 | Design me a living room. | [6.0, 8.0, 3.0] |

#### Table 5: Complete List of Elaborate Prompts for Evaluation (Tab. 2) [187-205]
*(Showing a representative sample of indices 1-40)*

| Index | Category | Prompt | Room Dimension |
| :--- | :--- | :--- | :---: |
| 1 | Functionality | Could you please design a Living Room for me? Designed for relaxation and socializing, the living room should feature comfortable seating areas, ample lighting for different activities, and a designated movie area with a large screen and surround sound for an immersive experience. | [6.0, 8.0, 2.5] |
| 2 | Functionality | Could you please design a Home Office for me? The home office should prioritize a clutter-free workspace with ergonomic furniture, ample storage for office supplies, and a small area for breaks with a comfortable chair and a coffee machine. | [3.0, 4.0, 2.5] |
| 3 | Functionality | Could you please design a Kitchen for me? Efficiency and ease of movement are key, with a triangular layout between the stove, refrigerator, and sink. Include a central island for additional workspace and seating for casual dining. | [5.0, 7.0, 2.5] |
| 4 | Functionality | Could you please design a Bedroom for me? A cozy and restful environment with a comfortable bed, soft lighting, and ample storage for personal items. A small reading nook with a comfy chair and a bookshelf should be included. | [4.0, 5.0, 2.5] |
| 5 | Functionality | Could you please design a Bathroom for me? Focus on practicality and tranquility, incorporating water-saving fixtures, good ventilation, and storage for toiletries. A separate shower and bathtub area can enhance the spa-like experience. | [3.0, 4.0, 2.5] |
| 6 | Functionality | Could you please design a Dining Room for me? Designed for meal gatherings, it should have a large table with comfortable seating for the family and guests, along with ambient lighting to enhance the dining experience. | [4.0, 6.0, 2.5] |
| 7 | Functionality | Could you please design a Playroom for me? A vibrant and flexible space that encourages play, creativity, and learning, with durable, easy-to-clean surfaces, storage for toys, and a comfortable area for reading and crafts. | [4.0, 6.0, 2.5] |
| 8 | Functionality | Could you please design a Fitness Room for me? Equipped with a range of exercise equipment, the room should have good ventilation, durable flooring, and a mirrored wall to check form during workouts. | [4.0, 6.0, 2.5] |
| 9 | Functionality | Could you please design a Laundry Room for me? A functional space with efficient appliances, a fold-out ironing board, and storage for cleaning supplies to make laundry tasks easier and organized. | [3.0, 3.0, 2.5] |
| 10 | Functionality | Could you please design a Home Theater for me? A dedicated space for cinematic experiences with tiered seating, blackout curtains for controlled lighting, and a high-quality sound system for an immersive audio experience. | [5.0, 7.0, 2.5] |
| 11 | Layout | Design my bedroom with following layout: This layout features a queen-sized bed against the main wall, with two nightstands on either side. Opposite the bed, there’s a dresser with a mirror above it, creating a functional dressing area. | [4.0, 4.0, 2.5] |
| 12 | Layout | Design my bedroom with following layout: In this setup, a single bed is placed in the corner, maximized for space efficiency. A small desk and chair fit snugly in the opposite corner, with a tall bookshelf beside it, making it ideal for a student. | [4.0, 4.0, 2.5] |
| 13 | Layout | Design my bedroom with following layout: This layout utilizes a king-sized bed centered on the main wall, with a bench at its foot. A large wardrobe is placed on the adjacent wall, providing ample storage space without cluttering the room. | [4.0, 4.0, 2.5] |
| 14 | Layout | Design my bedroom with following layout: A twin bed is positioned against one wall, leaving space for a play area on the opposite side of the room. Toy storage and a small table with chairs are included in the play area, perfect for children. | [4.0, 4.0, 2.5] |
| 15 | Layout | Design my bedroom with following layout: The room features a full bed flanked by a desk on one side and a nightstand on the other. Across from the bed, a low media console serves as a place for entertainment equipment, optimizing the layout for relaxation and study. | [4.0, 4.0, 2.5] |
| 16 | Layout | Design my bedroom with following layout: In this compact layout, a murphy bed is installed to maximize floor space when not in use. A fold-down desk is mounted on the opposite wall, creating a multipurpose space that can easily transition from bedroom to home office. | [4.0, 4.0, 2.5] |
| 17 | Layout | Design my bedroom with following layout: A loft bed dominates this layout, with a desk and wardrobe positioned underneath it. This efficient use of vertical space is ideal for small bedrooms, allowing for work and storage areas without sacrificing floor space. | [4.0, 4.0, 2.5] |
| 18 | Layout | Design my bedroom with following layout: This spacious layout includes a queen-sized bed positioned centrally with a vanity table and stool set against the adjacent wall. A comfortable reading chair and floor lamp are placed in one corner, creating a cozy reading nook. | [4.0, 4.0, 2.5] |
| 19 | Layout | Design my bedroom with following layout: Featuring a platform bed with storage drawers beneath, this layout optimizes storage. Along the opposite wall, a long, low dresser doubles as a display surface for personal items, with a mirror above it to enhance natural light. | [4.0, 4.0, 2.5] |
| 20 | Layout | Design my bedroom with following layout: In this innovative layout, the bed is centrally located with a headboard that doubles as a room divider. Behind the headboard, a workspace is created with a desk and shelving, effectively separating the sleeping and working areas. | [4.0, 4.0, 2.5] |
| 21 | Color & Material | Design a bedroom with color theme Minimalist with White Tones for me. Use white and light grey hues to emphasize cleanliness and simplicity. Incorporate sleek, modern furniture with straight lines. Material focus on matte finishes and textiles like cotton or linen for a soft touch. | [4.0, 4.0, 2.5] |
| 22 | Color & Material | Design a bedroom with color theme Boho Chic with Earthy Tones for me. Focus on mixing patterns, colors, and textures. Use materials like rattan, bamboo, and unfinished woods. Incorporate plants and macrame textiles for a cozy, natural feel. | [4.0, 4.0, 2.5] |
| 23 | Color & Material | Design a bedroom with color theme Industrial with Metallic Elements for me. Incorporate exposed steel, iron, or brushed nickel finishes in decor items and furniture. Use a neutral palette with bold accents in art or textiles. Emphasize raw and unfinished looks. | [4.0, 4.0, 2.5] |
| 24 | Color & Material | Design a bedroom with color theme Modern Glam with Gold Accents for me. Use a base of neutral colors with pops of bold color. Integrate gold-trimmed furniture and gold accent decor for a touch of luxury. Velvet and silk fabrics add texture and opulence. | [4.0, 4.0, 2.5] |
| 25 | Color & Material | Design a bedroom with color theme Nautical with Blue Colors for me. Incorporate various shades of blue with crisp white for a sea-inspired look. Use striped patterns and nautical decor items. Materials include weathered wood and rope accents for a maritime feel. | [4.0, 4.0, 2.5] |
| 26 | Color & Material | Design a bedroom with color theme Scandinavian with Pastel Colors for me. Use pale blues, pinks, and greens on a backdrop of white or grey. Furniture is minimalist and functional, with natural light wood materials. Add cozy textiles like wool or mohair to enhance comfort. | [4.0, 4.0, 2.5] |
| 27 | Color & Material | Design a bedroom with color theme Rustic with Wooden Elements for me. Emphasize natural, unfinished woods in furniture and decor for a warm, earthy feel. Use leathers and woven textiles to add depth. Incorporate organic, handmade items to underscore the rustic theme. | [4.0, 4.0, 2.5] |
| 28 | Color & Material | Design a bedroom with color theme Art Deco with Rich Jewel Tones for me. Combine deep greens, blues, and purples with metallic accents in gold or brass. Use geometric patterns in textiles and art. Furniture and decor pieces should evoke the luxury and opulence of the 1920s and 1930s. | [4.0, 4.0, 2.5] |
| 29 | Color & Material | Design a bedroom with color theme Contemporary with Monochromatic Scheme for me. Stick to a monochromatic color scheme throughout, using varying shades of the same color for depth. Focus on sleek furniture with minimalist designs. Utilize textures like glass, polished metals, and smooth fabrics to add interest. | [4.0, 4.0, 2.5] |
| 30 | Color & Material | Design a bedroom with color theme Vintage with Floral Patterns for me. Incorporate floral patterns in textiles, art, and wallpaper. Use a mix of antique or vintage-style furniture with rich wood tones. Embrace lace and embroidered textiles for a delicate, classic touch. | [4.0, 4.0, 2.5] |
| 31 | Atmosphere | Design my bedroom with atmosphere: Minimalist and serene, with clean lines and a monochrome palette. Accentuated by natural light and a lack of clutter. | [4.0, 4.0, 2.5] |
| 32 | Atmosphere | Design my bedroom with atmosphere: Bohemian and eclectic, featuring a mix of patterns, colors, and textures. Plants and vintage finds add personality and warmth. | [4.0, 4.0, 2.5] |
| 33 | Atmosphere | Design my bedroom with atmosphere: Modern and sleek, characterized by bold geometric shapes and a neutral color scheme. Innovative lighting and high-tech elements are key. | [4.0, 4.0, 2.5] |
| 34 | Atmosphere | Design my bedroom with atmosphere: Cozy and rustic, emphasizing natural wood, stone, and warm, earthy tones. Chunky knits and a fireplace complete the inviting ambiance. | [4.0, 4.0, 2.5] |
| 35 | Atmosphere | Design my bedroom with atmosphere: Nautical and breezy, with a color palette of blues, whites, and sandy tones. Maritime accessories and striped patterns evoke the seaside. | [4.0, 4.0, 2.5] |
| 36 | Atmosphere | Design my bedroom with atmosphere: Glamorous and luxurious, marked by opulent fabrics, metallic finishes, and a touch of sparkle. Elegant furniture and plush textiles dominate. | [4.0, 4.0, 2.5] |
| 37 | Atmosphere | Design my bedroom with atmosphere: Industrial and edgy, with exposed brick, metal details, and raw concrete elements. A neutral color scheme is offset by vibrant art. | [4.0, 4.0, 2.5] |
| 38 | Atmosphere | Design my bedroom with atmosphere: Traditional and elegant, featuring classic furniture, rich textures, and symmetrical arrangements. Deep wood tones and luxurious fabrics prevail. | [4.0, 4.0, 2.5] |
| 39 | Atmosphere | Design my bedroom with atmosphere: Scandinavian and bright, with a focus on simplicity, functionality, and minimalism. Pale woods, muted colors, and hygge accents are key. | [4.0, 4.0, 2.5] |
| 40 | Atmosphere | Design my bedroom with atmosphere: Contemporary and dynamic, with a mix of textures and materials. Clean lines, pops of color, and versatile pieces adapt to changing trends. | [4.0, 4.0, 2.5] |
