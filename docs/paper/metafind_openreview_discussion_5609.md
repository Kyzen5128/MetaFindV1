# OpenReview discussion, NeurIPS 2025 Submission 5609 (MetaFind)

Source: pasted verbatim by Kyzen on 2026-09-07 from the OpenReview page. The paste was cut by
the chat's 50,000-character limit inside Reviewer cY7o's review ("the entire architecture is
largely built upo"); the rest of cY7o's review and the authors' reply to it are MISSING here.
Author statements below are primary-source evidence about the method (authority tier 1,
alongside the paper); reviewer statements are not authority about what the authors did unless
the authors confirm them in a reply. Kept outside `metafind_source/` because that directory is
write-guarded; it is a paper-authority document all the same.

Statements that bear on the reproduction (all AUTHOR statements unless marked):

- Table 1 "w/o ESSGNN" = Stage-1 fusion head: "using the stage-one fusion layer yields results
  identical to the 'w/o ESSGNN' variant; thus, we omitted this configuration from our reported
  results for conciseness."
- Reported "w/ ESSGNN" results = single shared fusion head, Stage 2 with "the query and gallery
  encoders" frozen, "only the ESSGNN and fusion layer" updated, "stochastic scene dropout,
  randomly omitting layout context in 30% of the training samples ... The results we report
  correspond to this configuration."
- Cause of the R@1 drop: "feature attribution mismatch"/"partial feature attribution drift in
  the fusion layer" after Stage 2 on ProcTHOR, evaluated on layout-free Objaverse-LVIS.
- Table 1 is "evaluated on Objaverse-LVIS, which also serves as the training source for the
  first stage".
- Temperature 0.5, "following commonly used defaults in prior works. We did not perform
  dedicated tuning".
- Gallery point clouds: "downsampled representations (e.g., 10,000 points per object)".
- Scene graph: "sparse", two edge types: "(1) Physical-relation edges from ProcTHOR, capturing
  spatial dependencies like support or containment (e.g., 'cup on table'); and (2)
  Semantic-relation edges ... by prompting an LLM on object pairs."
- Reviewer ZhAY (reviewer statement, thanked and not corrected by the authors): "using text
  descriptions for procThor as the objects are synthetic and cannot be encoded using their
  point cloud"; authors: "the use of textual descriptions in ProcTHOR to address the
  limitations of synthetic object point clouds".
- Bidirectional vs unidirectional Stage 2 loss (text-only R@1): 11.4 vs 11.0.
- Inference placement comes from I-Design's LLM planners; ground-truth positions not assumed.
- Retrieval latency about 1.8 s for one object or 10 in parallel; 20 s per scene iteratively.
- Annotation: 11 views per object; LayoutVLM/Holodeck cited as the practice followed
  (they use 4 views).
- OOD scene study (conference room, gym, bar, laboratory; 5 prompts each) scored by GPT-4o and
  humans; table in Reviewer YspC's thread.
- No code link was given; Reviewer ZhAY asked the authors to publish code.

---

## Verbatim paste (truncated at the end as noted)

Paper Decision
Decisionby Program Chairs17 Sept 2025, 20:41 (modified: 29 Oct 2025, 13:52)EveryoneRevisions
Decision: Accept (poster)
Comment:
This paper presents MetaFind, a novel and highly compelling framework for scene-aware, multi-modal retrieval of 3D assets, a critical challenge for metaverse scene generation. The work is timely and addresses a significant gap in existing methods, which often lack the context-awareness necessary for coherent and flexible scene composition.

The core contributions of this paper are substantial. First, the proposed dual-tower retrieval architecture is a creative and robust solution that effectively supports arbitrary combinations of text, image, and 3D queries. This flexibility is a key strength, moving beyond the limitations of fixed-modality retrieval systems. Second, the introduction of the Equivariant Spatial-Semantic Graph Neural Network (ESSGNN) is a technically sound innovation. Its ability to maintain SE(3) equivariance while reasoning about semantic relationships is crucial for generating plausible and transformation-invariant scene layouts, a clear advancement over prior art. Third, the move to align with ULIP-2 instead of a fixed CLIP model is a well-justified design choice that likely contributes to the model's superior performance and generalization.

The experimental validation is thorough and convincing. The use of large-scale, standard datasets (Objaverse-LVIS/10K) provides a strong foundation for benchmarking. The comprehensive ablation studies effectively isolate the contributions of key components like the ESSGNN and the bidirectional loss function, which itself is a strength that ensures robust retrieval performance from both directions.

While the paper could potentially be strengthened by a more detailed discussion of computational complexity or performance on even more diverse and complex multi-modal queries, these are minor considerations that do not detract from the paper's significant contributions. The work is methodologically novel, technically sophisticated, and rigorously evaluated. It represents a clear step forward in the field of 3D scene understanding and retrieval and is highly suitable for acceptance. Its "plug-and-play" nature also suggests immediate utility and potential for broad adoption in downstream applications.

Author Final Remarks by Authors
Author Final Remarksby Authors12 Aug 2025, 10:40 (modified: 29 Oct 2025, 13:51)EveryoneRevisions
Author Final Remarks:
We thank all for the insightful feedback and the area chair for the time and diligent evaluation. We appreciate all reviewers' recognition of our proposed MetaFind in this community. Our MetaFind framework offers core contributions: (1) a scene-aware tri-modal retrieval integrating object-level cues with scene context; (2) ESSGNN, a theoretically SE(3)-equivariant GNN for robust contextual/spatial relations; and (3) flexible querying from any text/image/3D mix, with optional layout.

We've addressed all concerns:

R-GpRz (Rating 5): The "quality↑ but R@1↓" was due to stage-2 fusion head adaptation to layout; a dual-head resolves this. Latency is configurable (sequential/partial/parallel). Key hyperparameters (e.g., temp=0.5) were added.

R-YspC (Rating 4): An OOD study on four unseen scene types showed MetaFind+ESSGNN maintained coherence/aesthetics. MetaFind also acts as a prior selector for generation and supports scene completion.

R-ZhAY (Rating 4 ↑ to 5): Sparse scene graph details were added (physical/semantic edges). We clarified single-room training and outlined multi-room/outdoor extensions via SE(3)-equivariant ESSGNN. Dual-tower design decouples noisy queries from stable galleries.

R-cY7o (Rating 2 ↑ to 3 or 4):

We appreciate the reviewer's 2 concerns; after our clarifications resolved the concerns, the reviewer thoughtfully raised score.

Motivation of ESSGNN and R@1 gap: ESSGNN is vital for injecting SE(3)-equivariant scene context, ensuring coherent asset composition. It fills the gap left by object-centric retrievers, crucial for consistent layout representations in dynamic metaverse environments. Our two-stage training is cost-effective. Stage 1 learns object features; Stage 2 adds scene awareness by tuning ESSGNN + fusion. The R@1 dip on layout-free tests is not due to ESSGNN itself, but a distribution shift in the shared fusion head. Remedies: using the Stage-1 fusion head or a lightweight dual-head restores object-level accuracy. This trade-off and its resolution are now explicitly discussed.

Addressing Contribution & Community Impact: MetaFind is a significant advancement for the community, providing a practical blueprint for scene-aware tri-modal retrieval. We directly resolve prior limitations in coherent 3D scene generation. We ensured fair comparisons by adhering to the strongest official settings of all available open-source baselines, making MetaFind a valuable and actionable contribution to the field.

Official Review of Submission5609 by Reviewer GpRz
Official Reviewby Reviewer GpRz03 Jul 2025, 16:40 (modified: 29 Oct 2025, 10:55)EveryoneRevisions
Summary:
This paper presents MetaFind, a novel scene-aware tri-model compositional retrieval framework, for enhancing scene generation in metaverse.

Core contributions:

A dual-tower retrieval framework that integrates object-level semantics with global scene-level spatial reasoning, supporting arbitrary combinations of text, image, and 3D queries, allowing flexible and robust multi-modal querying.
A plug-and-play Equivariant Spatial-Semantic Graph Neural Network (ESSGNN) layout encoder that captures the spatial relationships and object attributes.
A two-stage training strategy that aligns with the dual-tower architecture and the flexible, multimodal nature of the retrieval task.
Strengths And Weaknesses:
Strengths:
This paper resolves a challenge of existing retrieval methods: context-aware retrieval of 3D assets.
ESSGNN layout encoder helps maintain equivariance to SE(3) transformations during message passing while combining semantic relationships between objects.
This paper proposes a creative dual-tower architecture with ULIP-2 instead of aligning to a fixed CLIP like in prior works.
The bidirectional loss function (considering both query-to-gallery and gallery-to-query) helps ensuring the robustness and generalization of the model.
There are comprehensive experiments on Objaverse-LVIS and -10K datasets, as well as a complete ablation study.
Weaknesses:
No explanation why Incorporating ESSGNN slightly degrades R@1 accuracy on object-level retrieval.
Iterative retrieval pipeline improves scene quality but adds latency and compute cost compared to one-shot methods.
Asset annotations are generated using GPT-4o, which may introduce language bias or inaccuracies.
Quality: 3: good
Clarity: 2: fair
Significance: 3: good
Originality: 3: good
Questions:
"After integrating the ESSGNN, while the overall scene quality is improved, we observe a drop in accuracy due to the added encoded information." Could this section be more specific of why there is a drop in accuracy? Why adding encoded information decreases the accuracy?
Same, in the ablation study in table 3, adding layout context improves aesthetic and scene coherence, while it significantly decreases the R@1 (13.5% compared without layout context compared with 11% with layout context). Please explain why this happens.
Please specify the value of the temperature hyperparameter in the experiments.
Appendix D, figure 4, look like this is a duplicate of figure 3 in a clearer way. Is this duplicate intended?
Limitations:
No negative societal impact.

Rating: 5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
No formatting concern.

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Final Justification:
This paper presents a well-motivated and technically solid contribution for context-aware retrieval of 3D assets. While some limitations remain, the overall contributions are substantial, the experiments are thorough, and the results demonstrate improvements over baselines.

Rebuttal by Authors
Rebuttalby Authors29 Jul 2025, 15:20 (modified: 29 Oct 2025, 12:03)EveryoneRevisions
Rebuttal:
We sincerely thank the reviewer for the thoughtful and encouraging feedback. We are especially grateful for your recognition of our core contributions, including the novel dual-tower retrieval architecture with ULIP-2, the flexible multimodal query design, and our two-stage training strategy. In particular, we deeply appreciate your acknowledgment of the ESSGNN layout encoder's ability to maintain SE(3) equivariance during message passing while capturing rich semantic relationships between objects. Your comments strongly validate our motivation to incorporate spatial-functional layout reasoning into retrieval, and we are encouraged by your recognition of the framework's robustness and comprehensiveness.

1. Reviewer Comment:
No explanation why incorporating ESSGNN slightly degrades R@1 accuracy on object-level retrieval.

Response:
Thank you for pointing out this problem. We believe this reflects a temporary and explainable trade-off between object-level retrieval precision and enhanced scene-level coherence.

As shown in Table 1, the reported results are evaluated on Objaverse-LVIS, which also serves as the training source for the first stage of our framework. In this stage, the model is trained on isolated assets without layout context, and ESSGNN is not involved. In the second stage, we fine-tune the model with ESSGNN using a different dataset—ProcTHOR—which provides structured room layouts and a distinct asset distribution. Although the retrieval objective remains the same, the input structure becomes richer and layout-aware, shifting the model's attention from asset-specific details toward spatial and semantic consistency within the scene.

However, during evaluation on Objaverse-LVIS, which lacks scene context, ESSGNN cannot be applied, and the model must rely solely on the backbone and fusion layer that is co-trained with ESSGNN. This creates a feature attribution mismatch: the fusion layer, originally trained to operate on standalone object embeddings, has now been partially adapted to consume layout-conditioned features, leading to a moderate performance drop in layout-free retrieval.

To mitigate this, the simplest solution is to maintain two separate fusion layers—one trained in stage one on layout-free data, and another fine-tuned in stage two with layout context. During inference, the system can dynamically select the appropriate fusion head depending on whether layout information is available. In fact, using the stage-one fusion layer yields results identical to the "w/o ESSGNN" variant; thus, we omitted this configuration from our reported results for conciseness. We thank the reviewer for pointing this out and will make this distinction clearer in the revised manuscript to avoid confusion.

That said, we are interested in exploring whether a single shared fusion layer can simultaneously support both scene-aware and layout-free retrieval. To this end, we freeze the query and gallery encoders during stage two and update only the ESSGNN and fusion layer. Moreover, as mentioned in line 202, we introduce stochastic scene dropout, randomly omitting layout context in 30% of the training samples to encourage generalization to layout-free inputs. The results we report correspond to this configuration. Despite these efforts, some performance drop remains, particularly in layout-free retrieval, due to partial feature attribution drift in the fusion layer. We sincerely appreciate the reviewer's insightful comment. We have incorporated a more detailed explanation into the revised manuscript and highlighted it as a valuable direction for future work.

2. Reviewer Comment:
Iterative retrieval pipeline improves scene quality but adds latency and compute cost compared to one-shot methods.

Response:
We agree with the reviewer that the iterative retrieval pipeline introduces additional latency and compute cost compared to one-shot methods, especially when retrieving multiple objects for a full scene. However, we believe this trade-off is highly use-case dependent and can be flexibly adjusted in real-world deployments. For applications where global scene coherence and stylistic consistency are critical, a fully sequential pipeline offers the best quality. In contrast, when efficiency is prioritized, simplified variants—such as parallel retrieval or partial scene decomposition—can be employed. For example, a room can be divided into semantically or spatially grouped regions (e.g., seating area, storage area), where objects within each region are retrieved sequentially to preserve local coherence, while regions are processed in parallel to improve overall efficiency. We believe this design flexibility makes our method practical for a range of scenarios and have clarified this point in the updated version.

3. Reviewer Comment:
Asset annotations are generated using GPT-4o, which may introduce language bias or inaccuracies.

Response:
We sincerely thank the reviewer for pointing out this important limitation. We agree that potential language bias and annotation inaccuracies introduced by GPT-generated descriptions are meaningful concerns. This challenge is common across large-scale 3D datasets, and our annotation pipeline follows the prevailing practice in recent works such as LayoutVLM [1] and Holodeck [2], which also generate asset-level descriptions via VLM based on multi-view rendered images. While these works typically use 4 rendered views per object, we increase coverage by rendering each asset from 11 distinct viewpoints to provide GPT-4o with a more complete understanding of 3D geometry and appearance. We further perform manual spot checks on a substantial subset of the generated annotations and find GPT-4o's descriptions to be largely accurate and semantically rich. However, we acknowledge that language bias was not explicitly addressed in this work and agree that it represents a valuable direction for future research. We have added this clarification to the revised manuscript. Thank you again for raising this important point.

References:
【1】 LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models, CVPR 2025.
【2】 Holodeck: Language Guided Generation of 3D Embodied AI Environments, CVPR 2024.

4. Reviewer Comment:
"After integrating the ESSGNN, while the overall scene quality is improved, we observe a drop in accuracy due to the added encoded information." Could this section be more specific of why there is a drop in accuracy? Why adding encoded information decreases the accuracy?

Response:
Please refer to Answer 1 above for a detailed explanation of this trade-off and its cause.

5. Reviewer Comment:
Same, in the ablation study in Table 3, adding layout context improves aesthetic and scene coherence, while it significantly decreases the R@1 (13.5% without layout context vs. 11% with layout context). Please explain why this happens.

Response:
Please refer to Answer 1 for our explanation regarding the observed trade-off due to dataset shift between stages.

6. Reviewer Comment:
Please specify the value of the temperature hyperparameter in the experiments.

Response:
Thank you for the question. We used a temperature value of 0.5 in our experiments, following commonly used defaults in prior works. We did not perform dedicated tuning for this hyperparameter and will consider exploring more optimal values in future work. The value has been added to the revised manuscript.

7. Reviewer Comment:
Appendix D, figure 4, looks like this is a duplicate of figure 3 in a clearer way. Is this duplicate intended?

Response:
Yes, this duplication is intentional. Due to space limitations in the main paper, we included a smaller version of the visualization in Figure 3 to provide a quick overview. The full-resolution version is placed in Appendix D (Figure 4) for clearer detail. Thank you for pointing this out—we have added a note in the main text referencing "see Appendix Figure 4 for full-resolution details."

Official Comment by Reviewer GpRz (08 Aug 2025): Thank you for the rebuttal of the authors. I keep my original score.

Official Review of Submission5609 by Reviewer YspC
Official Reviewby Reviewer YspC02 Jul 2025, 04:46 (modified: 29 Oct 2025, 10:55)EveryoneRevisions
Summary:
This paper introduces MetaFind, a retrieval framework for 3D asset selection designed to support coherent scene generation in metaverse applications. Unlike prior retrieval approaches that largely ignore spatial and stylistic consistency, MetaFind integrates a dual-tower architecture with ULIP-2 embeddings for text, images, and 3D point clouds, while also incorporating a layout-aware equivariant graph encoder (ESSGNN) to model the scene's spatial context.

Strengths And Weaknesses:
Overall i think this paper is interesting, but i am not an expert in retrieval.

Strength:

the paper identifies a clear gap in 3D retrieval research: the lack of methods for scene-aware, multimodal retrieva, and the authors proposed a principled solution: ESSGNN
the ESSGNN module is a technical contribution
the strategy to iteratively retrieve and update the scene graph is well thought out, improving context-awareness over naive one-shot retrieval.
Weakness:

the reliance on synthetic layouts from ProcTHOR and object-level data from Objaverse-LVIS annotated with VLMs raised concerns about its generalization to real-world data .
Quality: 3: good
Clarity: 3: good
Significance: 2: fair
Originality: 3: good
Questions:
Do you see MetaFind as purely retrieval-based, or could it also be integrated with generative scene modeling approaches to fill in missing assets or generate novel variations
I would love to see more visual results to validate the generalization of the method
Limitations:
No

Rating: 4: Borderline accept
Confidence: 2
Final Justification:
The authors have addressed most of my concerns, and I really would love to see this work generalized to scene generation. And I find the visual comparison results by human and GPT convincing. Therefore, I would love to keep my original rating.

Rebuttal by Authors
Rebuttalby Authors29 Jul 2025, 16:45 (modified: 29 Oct 2025, 12:03)EveryoneRevisions
Rebuttal:
We sincerely thank the reviewer for the positive feedback and recognition of our work. We are especially grateful for the acknowledgment of our effort in identifying a key gap in 3D asset retrieval—namely, the lack of scene-aware, multimodal retrieval frameworks—and in proposing a principled solution via ESSGNN. We also appreciate the reviewer's recognition of the technical contribution of ESSGNN and the thoughtful design of our iterative retrieval strategy, which enhances context-awareness beyond one-shot methods.

1. Reviewer Comment:
the reliance on synthetic layouts from ProcTHOR and object-level data from Objaverse-LVIS annotated with VLMs raised concerns about its generalization to real-world data .

Response 1:
Thank you for raising this important concern. We address it from three perspectives:

Synthetic Data as a Foundation for Scene Learning
Synthetic datasets like ProcTHOR offer precise spatial and semantic annotations that are crucial for training layout-aware models. This setup enables controlled experiments and consistent supervision, which are currently difficult to obtain at scale from real-world 3D environments. While ProcTHOR is synthetic, the diversity of object arrangements and realistic layouts it contains provides a strong proxy for general spatial reasoning.

Use of VLM-based Annotation is Common and Scalable For object-level descriptions in Objaverse-LVIS, we follow recent best practices used in LayoutVLM [1] and Holodeck [2], both of which leverage vision-language models (VLMs) such as GPT-4 to annotate 3D assets based on multi-view renderings. To further improve annotation quality, we extend the rendering pipeline to 11 views per object (compared to 4 in prior work), ensuring more complete geometric coverage. Manual checks on a subset of our assets confirm that the resulting captions are accurate and semantically rich. We acknowledge the limitations of LLM-generated annotations and consider improving them—e.g., via human-in-the-loop verification—as a promising future direction.

Demonstrated Generalization to Unseen Scene Types
Our core contribution is the proposal of a flexible, scene-aware dual-tower retrieval framework that generalizes across varying spatial contexts. To assess this, we construct an additional evaluation set containing four scene categories not seen during training—conference rooms, gyms, bars, and laboratories—each with 5 unique prompts. Using the same retrieval pipeline described in Section 3.3, we observe consistent retrieval quality and scene coherence. These results suggest that MetaFind generalizes well to previously unseen spatial environments. Please refer to Response 3 for the details.

We sincerely appreciate the reviewer's suggestion. We have included this discussion, along with the new generalization experiment, in the revised manuscript. As our primary goal is to highlight the importance of scene-aware retrieval and advocate for a flexible, standardized retrieval framework, we initially focused on a single structured dataset in this submission to better isolate this contribution. In future work, we plan to incorporate more diverse scene styles and real-world datasets to further enhance generalization.

References:
【1】 LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models, CVPR 2025.
【2】 Holodeck: Language Guided Generation of 3D Embodied AI Environments, CVPR 2024.

2. Reviewer Comment:
Do you see MetaFind as purely retrieval-based, or could it also be integrated with generative scene modeling approaches to fill in missing assets or generate novel variations.

Response 2:
Thank you for the insightful question. While MetaFind is designed as a retrieval framework, our motivation extends far beyond retrieval alone. Our long-term research vision is to enable prompt-to-world generation, where retrieval plays a critical role within a broader pipeline for constructing interactive 3D environments.

In many practical scenarios, such as expanding existing virtual scenes, it is necessary to retrieve relevant, high-quality assets to fill in missing elements or extend the world coherently. MetaFind offers a structured and flexible retrieval backbone for such tasks.

Moreover, with the rapid progress in 3D generative models—especially text-to-3D generation—we see great potential in combining retrieval with generation. For instance, MetaFind can serve as a prior selector, retrieving semantically and stylistically aligned assets to guide or condition the generation process. These retrieved assets can act as base meshes, reference points, or fine-tuning anchors to improve quality and control in generative pipelines.

In particular, once an initial 3D scene is generated, it can be segmented into individual object-level regions using instance-level segmentation. These objects can then be rendered from multiple viewpoints and described via vision-language models. MetaFind can leverage these descriptions to retrieve missing or higher-quality replacement assets, enhancing both the completeness and realism of the scene. For novel variation generation, these retrieved assets can also be passed as conditioned priors into a generative model to produce more diverse or controllable outputs.

Our modular architecture allows for seamless integration with generative components, making MetaFind a strong foundation for both retrieval-based and hybrid retrieval-generation pipelines.

3. Reviewer Comment:
I would love to see more visual results to validate the generalization of the method

Response 3:
Thank you for your interest in the generalization ability of our method. While we are unable to include additional visualizations such as pdf file or links due to the rebuttal policy, we have conducted a targeted internal evaluation to assess this aspect.

To further verify the generalization ability of MetaFind beyond the ProcTHOR distribution, we designed an additional evaluation protocol. Specifically, we constructed a new test set containing four out-of-distribution scene categories not present in ProcTHOR: conference room, gym, bar, and laboratory. For each category, we designed 5 distinct prompts describing varied scene configurations and object compositions.

Following the exact same pipeline in Section 3.3 of our main paper, we generated scenes based on these prompts and conducted evaluations using both GPT-4o and human annotators across four dimensions: Aesthetic, Color & Material, Scene Coherence, and Realism & Geometry. The results are presented below.

Method	Aesthetic (GPT-4o)	Aesthetic (Human)	Color & Material (GPT-4o)	Color & Material (Human)	Scene Coherence (GPT-4o)	Scene Coherence (Human)	Realism & Geometry (GPT-4o)	Realism & Geometry (Human)
ULIP	2.85	3.05	2.85	2.90	2.70	2.80	2.90	2.95
OpenShape	3.05	3.25	3.10	3.20	3.15	3.05	2.90	2.95
MetaFind w/o ESSGNN	3.35	3.45	3.25	3.35	3.20	3.35	3.20	3.30
MetaFind w/ ESSGNN	4.05	4.10	3.90	4.05	3.95	4.15	3.95	4.10
These results strongly support MetaFind's generalization to novel scene domains and reinforce its effectiveness in capturing spatial and stylistic coherence even under out-of-distribution prompts：

Conference Room
A modern conference room with a long table and several chairs, set up for a team meeting.
A small meeting space featuring a round table, whiteboard, and overhead lighting.
A high-end boardroom with glass walls and a screen for video calls.
A minimalistic meeting area with wooden furniture and soft ambient lighting.
A collaborative workspace with movable chairs, a digital display, and an open layout.
Gym
A compact gym with basic cardio machines and a free weights section.
A functional training space with yoga mats, kettlebells, and mirrors on one wall.
A home-style gym setup with a treadmill and resistance bands.
A brightly lit gym room equipped with strength machines and floor mats.
A fitness zone for group workouts, with speakers and a clear open space.
Bar
A cozy bar corner with a counter, bar stools, and shelves for bottles.
A modern bar with minimalistic furniture and hanging lights.
A rooftop-style bar with high tables and open-air seating.
A small cocktail bar with soft lighting and decorative shelves.
A themed bar space with neon signs and a casual layout.
Laboratory
A standard lab with counters, storage cabinets, and basic lab tools.
A clean laboratory room with workstations and scientific posters.
A small research space with a microscope table and labeled storage.
A functional lab environment with digital equipment and power outlets.
A compact experiment room with white surfaces and labeled containers.

Official Comment by Reviewer YspC (06 Aug 2025): Thanks for your response. I will keep my original rating

Official Review of Submission5609 by Reviewer ZhAY
Official Reviewby Reviewer ZhAY01 Jul 2025, 11:33 (modified: 29 Oct 2025, 10:55)EveryoneRevisions
Summary:
This paper presents a method for retrieval of 3D objects based on a tri-modal format. The pipeline include 2 encoders for query and gallery, both of which are initialized with ULIP2. The authors also use a fuser module to fuse different modalities at the query encoder. The query encoder is also equipped with a layout encoder that encodes the objects, their position, and text description into a scene graph. Through an iterative construction methodology, the pipeline can retrieve and add objects to the scene. The authors demonstrate perofrmance of the model in terms of single object retrieval as well as coherency in the generated scene.

Strengths And Weaknesses:
Strengths:

The layout encoder is new and interesting to have. I like the idea of using text descriptions for procThor as the objects are synthetic and cannot be encoded using their point cloud.
The encode is multi-modal and gives flexibility for inference.
Authors conduct a comprehensive analysis of the model.
Weaknesses:

One ambugity for me is the way the scene is initialized. is it initialized from an empty scene?
ProcThor is a synthetic dataset and training the layout encoder on this dataset forces it to follow ProcThor layout generation. This questions the generalizability of the layout encoder.
The model uses two encoders (query and gallery) while they both perform the same thing (layout encoder of query encoder aside). Why don't you use one encoder for everything?
This is related to the previous comment. While tri-modal encoding gives flexibility at query, the gallery doesn't need all modalities as the 3D point cloud capture most of the necessary information encoded by the other two modalities such as shape, color, and even more. So why do we need a multi-modal encoder for gallery?
The semantic edge generation relies on LLM-prompted text, but this is only described abstractly. Plus, how are the edges connected between different components? Is the graph fully-connected?
While the iterative construction of the scene seems to narrowly increase the performance, there are no discussions on the computational complexity of this part. How does the time differ between a iterative and non-iterative model?
Strong layout-aware baselines such as ControlRoom3D are missing from the comparisons.
This work only applies to indoor scenes due to its training strategy while this has not been clarified in the text. How does the model work on outdoor scenes? Also, how are scenes with multiple rooms processed in this pipeline?
Authors need to perform analysis on the diversity of the generated scenes. Most likely the diversity is limited by the procThor layouts.
Lots of components like gallery encoder and fuser don't show up in the overview figure, making it hard to understand the work in one-go.
additionally, at inference time, how do you notify the model regarding the location of the query object? is it already included in the scene graph?
There are no ablations on the bidirectional and unidirectional layout loss.
line 300 to 302 are qualitivative and very broad. You have to guide the reader with specific details such as t"he painting is unrealistically behind the shelf", etc.
Quality: 2: fair
Clarity: 3: good
Significance: 2: fair
Originality: 3: good
Rating: 5 (raised from 4)
Confidence: 4
Final Justification:
I'm convinced by the authors responses. While I think there should be more explanation regarding graph construction at inference in more details, I don't find it a barrier for the acceptance of the paper.

Rebuttal by Authors
Rebuttalby Authors30 Jul 2025, 03:52 (modified: 29 Oct 2025, 12:03)EveryoneRevisions
Rebuttal:
Thank you for your thoughtful and encouraging feedback. We are glad that you found our proposed layout encoder novel and effective, and we appreciate your recognition of its design motivation—specifically, the use of textual descriptions in ProcTHOR to address the limitations of synthetic object point clouds. We are also pleased to hear that you value the flexibility brought by our multi-modal encoding strategy and the comprehensiveness of our analysis.

1. Comment:
what the way the scene is initialized.

Response 1:
We supports flexible scene graph initialization—starting from an empty layout, a partial scene, or a furnished template—depending on the use case. As shown in Algorithm 1, scene composition proceeds iteratively: at each step, the model retrieves an object coherent with the current scene, places it appropriately, and updates the scene graph. This design enables both scene generation from scratch and context-aware scene editing.

2. Comment:
ProcThor is a synthetic dataset. The generalizability of the layout encoder.

Response 2:
To further assess the generalizability, we do an additional evaluation containing four out-of-distribution scene categories not present in ProcTHOR: conference room, gym, bar, and laboratory. For each category, we designed 5 distinct prompts describing varied scene configurations and object compositions. Due to the characters limitation of rebuttal, we can not post contents in current chanel. Please refer to response 3 of Reviewer YspC. The results support MetaFind's generalization to novel scene domains even under out-of-distribution settings. While we initially focused on a single structured dataset to isolate the impact of our scene-aware retrieval framework, future work will extend to more diverse styles and real-world datasets to further validate generalization and flexibility.

3. Comment:
Why don't you use one encoder for everything?

Response 3:
While two encoders share same backbone, we deliberately adopt a dual-encoder (two-tower) design to enhance flexibility, modularity, and training stability. In practice, query inputs are often diverse and incomplete—varying across scenes and applications—while gallery assets typically come from curated datasets with complete annotations. Using a shared encoder would require balancing generalization for noisy queries with stability for clean gallery data, which can lead to suboptimal performance on both sides.

The dual-encoder decouples these concerns. We can fine-tune the query encoder to handle different modality combinations and tasks, while keeping the gallery encoder fixed and efficient. This not only improves training stability but also allows gallery features to be precomputed and reused, enabling fast and cost-effective inference. Moreover, this structure provides a scalable foundation for future extensions, such as adding more complex modules on the query side without affecting the gallery side.

4. Comment:
why need multi-modal encoder for gallery?

Response 4:
While point clouds capture key geometric and visual features, they are still downsampled representations (e.g., 10,000 points per object) and lose fine-grained details. More importantly, they lack high-level semantic cues such as typical usage, real-world scale, and functional context. To address this, we include image and text modalities in the gallery, which provide complementary information and help the model reason about object utility and scene relevance. This multi-modal encoding supports more contextual, user-aligned retrieval—crucial for realistic 3D scene construction.

5. Comment:
How edges connected ? fully-connected?

Response 5:
Our scene graph is sparse and semantically structured shown in Figure 2. It includes two types of edges: (1) Physical-relation edges from ProcTHOR, capturing spatial dependencies like support or containment (e.g., "cup on table"); and (2) Semantic-relation edges, which capture functional or contextual associations (e.g., "microscope and lab bench") by prompting an LLM on object pairs. This dual-edge encodes both physical layout and high-level semantics, enhancing retrieval and layout reasoning. We revised the paper to clarify the graph construction and define these edge types more explicitly.

6. Comment:
Time differ between a iterative and non-iterative pipeline?

Response 6:
Retrieving 10 objects in parallel takes roughly the same time as retrieving a single object—about 1.8 seconds total. In contrast, the fully iterative version performs one-by-one retrieval and computes a layout position for each object, resulting in an average total time of 20 seconds per scene.

This fully iterative mode is suitable for use cases like interactive scene authoring, where fine-grained control and global coherence are prioritized. For time-sensitive applications, a hybrid strategy—such as region-wise parallel retrieval—can be used to improve efficiency while preserving consistency. We clarified this trade-off and its practical implications in the revised paper.

7. Comment:
Strong layout-aware baselines (ControlRoom3D) are missing from comparisons.

Response 7:
Thank you for the valuable suggestion. We agree that ControlRoom3D is an excellent work and included it in our Related Work. As its code&models are not publicly available, we could not fairly include it in our comparisons. Nonetheless, its mesh-based structural prior is highly compatible with our pipeline and could guide object placement. We appreciate suggestion and consider such integration in future work.

8. Comment:
How model work on outdoor scenes? Also, how multiple rooms in pipeline?

Response 8:
We clarified in the revised manuscript that our model is trained on single-room indoor scenes due to dataset limitations. However, the framework is designed to generalize to open-world settings. ESSGNN ensures SE(3) equivariance—preserving layout representations under translation and rotation—which is essential for large-scale or dynamic open world (see Appendix C for proof). For multi-room scenes, as noted in Response 6, our pipeline supports both sequential and parallel region-wise processing. Its modular design allows scene graphs to be composed from multiple subgraphs, enabling flexible multi-room generation. We appreciate comment and plan to explore these further.

9. Comment:
diversity of the generated scenes.

Response 9:
Please refer to Response 2 for new experiments.

10. Comment:
hard to understand overview figure

Response 10:

Thanks for suggestion. We revised the figure for better clarity. Both query and gallery encoders share the ULIP-2 backbone, which supports tri-modalities. The query encoder additionally integrates an ESSGNN to encode contextual information from already-placed objects. All features are fused and compared with pre-encoded gallery embeddings via similarity search. Due to rebuttal policy, we cannot submit updated figures, but we clarified the architecture and included the new diagram in the revised version.

11. Comment:
at inference time, how do you notify the model regarding the location of the query object?

Response 11:

In training, locations are from ProcTHOR. At inference time of testing, we do not assume access to ground-truth positions. Instead, we use I-DESIGN framework [2], where multi-agent LLM planners iteratively determine object placement. Notably, layout generation is an active research area, with growing interest in enhancing LLM/VLM spatial reasoning [1].

References:
[1] LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models [2] I-Design: Personalized LLM Interior Designer

12. Comment:
no ablations on the bidirectional and unidirectional layout loss.

Response 12:
Thanks for suggestion. We added an ablation, the unidirectional one performs slightly worse than bidirectional one.

Variant	R@1 (%)	Aesthetic (GPT-4o)	Scene Coherence (GPT-4o)
MetaFind (Full, bidirectional)	11.4	4.1	4.2
MetaFind (Full, unidirectional)	11.0	3.9	4.1
13. Comment:
line 300 are broad.

Response 13:
Thank you for suggestion. We revised our qualitative analysis to more clearly highlight the impact of ESSGNN. As shown in Figure 3:

Room 1: Without ESSGNN, the room lacks stylistic coherence—the metallic fireplace and mismatched furniture deviate from the classical theme. With ESSGNN, the scene adopts a unified classical aesthetic with a dark-toned fireplace, matching sofa, and bookshelf.
Room 2: Without ESSGNN, modern office furniture and cluttered seating break the archive theme and hinder functionality. With ESSGNN, compact wooden chairs are arranged around the table, better fitting the aged archive context and improving usability.
14. Comment:
the limitation of procthor, outdoor scene, and multi-room generation.

Response 14:

As noted in Response 8, we now clarify in the paper that our training is limited to single-room scenes. We also outline preliminary strategies (e.g., region-wise retrieval) for extending to multi-room and outdoor scenes, which we plan to explore in future work.

15. Comment:
Seems good but lots of details could be added to the supplementary.

Response 15: We appreciate your recognition. We integrated relevant content in the previous responses in the supplementary.

Official Comment by Reviewer ZhAY (09 Aug 2025): I thank the authors for their rebuttal. I'm convinced with the responses and will raise my score. That said, I think adding more details regarding the scene graph generation at inference time would be beneficial for the reader. Hope you plan to publish the code as well so that they community can benefit from an additional baseline.

Official Review of Submission5609 by Reviewer cY7o
Official Reviewby Reviewer cY7o09 Jun 2025, 20:30 (modified: 29 Oct 2025, 10:55)EveryoneRevisions
Summary:
This paper presents MetaFind, a multimodal 3D asset retrieval framework designed to support scene-aware retrieval and composition in metaverse scenarios. The proposed system builds on the ULIP-2 architecture, with the main addition being an Equivariant Spatial-Semantic Graph Neural Network (ESSGNN) used to encode scene layout context. The paper shows scene-level composition quality with ESSGNN and reports improvements over baseline retrieval models on both object-level and scene-level tasks.

Strengths And Weaknesses:
Strengths

The paper tackles a problem for metaverse scene generation: retrieving 3D assets that are spatially and stylistically coherent with the current scene, rather than purely object-centric retrieval.

The framework supports arbitrary combinations of text, image, point cloud, and layout context as queries, which is valuable for practical flexibility.

Weaknesses

1. Confused Role and Unclear Motivation of ESSGNN

The role of ESSGNN in the proposed framework remains somewhat confusing and insufficiently justified. On the object-level retrieval task (Table 1), MetaFind without ESSGNN already achieves state-of-the-art performance across all modality combinations, outperforming strong baselines. However, adding ESSGNN degrades object-level retrieval accuracy. In contrast, on the scene-level evaluation (Table 2), ESSGNN brings significant gains in scene coherence and realism. This discrepancy raises a key question: is ESSGNN designed purely for enhancing scene-level composition, at the cost of degrading general-purpose retrieval? If so, this should be made clearer, as the current narrative presents ESSGNN as a broadly applicable enhancement. Moreover, the motivation for introducing ESSGNN is not fully convincing—if the core retrieval pipeline already outperforms baselines without it, the paper should better articulate why introducing additional complexity is warranted, and in which scenarios this trade-off is acceptable.

2. Questionable Contribution

The paper substantially overstates its contributions relative to the actual novelty and technical content. Despite claiming MetaFind as a new retrieval framework, the entire architecture is largely built upo

[PASTE TRUNCATED HERE by the chat's 50,000-character limit. Missing: the rest of Reviewer cY7o's review, the authors' rebuttal to cY7o, and any later comments.]
