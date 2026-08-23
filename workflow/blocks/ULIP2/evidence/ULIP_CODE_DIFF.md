# ULIP block vs official ULIP implementation — file by file

Compared 2026-08-23 against `/home/kyzen/upstream/ULIP` @ `95d480f`
and `docs/paper/ulip2_source/main.tex`.

Legend: **=** identical · **≈** equivalent, different form · **≠** differs ·
**∅** upstream has no counterpart · **?** unverified

---

## 0. Vendored upstream — `metafind/vendor/ulip/`

`cmp` over all 18 shared files: **byte-identical, no exceptions.**

```
dataset_3d.py  losses.py  ULIP_models.py  point_encoder.py  dvae.py  misc.py
checkpoint.py  logger.py  ULIP_2_PointBERT_10k_colored_pointclouds.yaml
build.py  config.py  io.py  logger.py  registry.py  utils.py  __init__.py
pointnet2.py  pointnet2_utils.py
```

Not vendored (unused): `main.py`, `templates.json`, `labels.json`,
`dataset_catalog.json`, the ShapeNet/ModelNet yamls, `tokenizer.py`,
pointmlp / pointnext / customized_backbone.

**Consequence:** every architectural number below is upstream's own code, not a
re-implementation. `trans_dim 384`, `depth 18`, `num_heads 6`, `group_size 32`,
`num_group 512`, `encoder_dims 256`, `drop_path_rate 0.1`, `pc_feat_dims 768`,
`pc_projection (768,1280)`, `Encoder(input_dim=6)`, the `[cls ‖ max]`
concatenation — all executed from upstream bytes.

---

## 1. Compat shims — `metafind/compat/ulip_patch.py`  ∅

Upstream has no counterpart; these exist because upstream does not import on
PyTorch 2.x.

| shim | why | risk |
|---|---|---|
| `torch._six` module | removed in PyTorch 2.0; `dataset_3d.py` imports it | none — pure name binding |
| `knn_cuda` stub | `dvae.py` builds `KNN(k=4)` at import, never calls it (call site commented out) | stub **raises** if reached |
| `pointnet2_ops` stub | `misc.py` imports at module level | stub **raises** if reached |
| `misc.fps` → pure torch | the one real consumer of `pointnet2_ops` | **? see below** |

**? UNVERIFIED — `fps`.** Our `farthest_point_sample_idx` mirrors the CUDA
kernel's algorithm and seed (index 0), but the two have **never been compared
numerically**, because `pointnet2_ops` is not installed on this machine. Ties in
the argmax could break differently. `Group()` calls it for 512 centres per
cloud, so any divergence changes every embedding slightly.
**This is a genuine open item, not a claim of equivalence.**

---

## 2. n03 point clouds — `metafind/data/pointclouds.py`

Upstream counterpart: `Objaverse_Lvis_Colored` (`dataset_3d.py:456-540`)
and the released `.npy` triplets (`SFXX/ULIP`, `ULIP-2/objaverse_lvis/`).

### 2a. Measured directly from the official release (2026-08-23, n=40)

Extracted `000-009.tar.gz` and read the arrays:

```
xyz         float16  (10000, 3)   max radius from ORIGIN   = 1.0000  (0.9998-1.0002)
                                  max radius from CENTROID = 1.0679  (0.9657-1.2541)
                                  ||centroid||             = 0.1332 mean
rgb         float16  (10000, 3)   min 0.0000   max 1.0000
image_feat  float32  (12, 1280)
blip_caption  ONE string per asset
```

**This settles `U-02`.** rgb in the official release is **unit [0,1]**, measured,
not inferred. `RGB_SCALE = "unit"` is correct.

**It also explains the normalisation.** The released file is unit-radius about
the **origin**; upstream's `pc_norm` at load time re-centres on the **centroid**
and rescales. So the tensor the model actually sees is centroid-centred,
unit-radius-from-centroid.

### 2b. Line by line

| behaviour | upstream `Objaverse_Lvis_Colored` | ours | |
|---|---|---|---|
| point count | 10,000, loaded whole | 10,000 | **=** |
| `pc_norm` | centroid → 0, max radius → 1 | identical formula, `pointclouds.py:123-128` | **=** |
| when normalised | at **load** (`__getitem__`) | at **write** (`pointclouds.py:497`), Stage 1 loads as-is | **≈** idempotent; same model input |
| rgb scale | inherits released file → [0,1] | `/255` → [0,1] | **=** (measured) |
| rgb normalised? | **no** — concat after `pc_norm(xyz)` only | same | **=** |
| layout | `concat([xyz, rgb], axis=1)` → (10000,6) | same | **=** |
| `use_color` | `True` | always | **=** |
| `use_height` | `False` | not implemented | **=** |
| FPS | **none** on this path | none | **=** |
| `random_sample` | **none** on this path | none | **=** |
| augmentation | **none** on this path | none | **=** |
| dtype on disk | float16 | float32 | **≠** ours is higher precision |
| where points come from | pre-existing released `.npy` | area-weighted `trimesh.sample_surface` from the GLB | **≠** — see below |
| frame | as released | +180° yaw about Y (`meshload.py`) | **≠** correction, see below |
| COLOR_0 | n/a (baked into release) | recovered from glTF; multiplier on `flat`/`gltf_default`, withdrawn from `texture` | **≠** |

**≠ Sampling source.** Upstream never samples a mesh on this path — it reads
clouds Salesforce already made. Their sampler is not in the repo. Ours is
area-weighted surface sampling, which is the closest honest reconstruction of
"10,000 points from this mesh".

**≠ Frame.** Our GLBs sit 180° yawed relative to the released clouds. Measured:
median Chamfer **0.0903 as-is → 0.0230 after the yaw**. `det = +1`, a rotation.

### 2c. Agreement, measured today (5 overlapping uids)

```
uid          chamfer   ULIP rgb mean   ours    diff
867dfc95e9    0.0188       0.085       0.078   0.007
8ad044978b    0.0261       0.329       0.332   0.003
93b4b53d09    0.0251       0.704       0.706   0.001
aa191cdb73    0.0277       0.634       0.634   0.000
b751333a51    0.0088       0.171       0.216   0.045
             median 0.0251                median 0.0025
```

Our clouds are **not** their clouds and never will be — different sampler. They
agree closely in shape and colour.

---

## 3. n04 renders — `metafind/data/renders.py`

**Upstream counterpart: ∅ — the ULIP repository contains NO rendering code.**
Grep for blender/render/camera across the repo returns nothing executable. The
only rendering statement anywhere is the paper sentence:

> `main.tex:677` — "we use **Blender** to render **12 images**, spaced equally
> by **360/12 degrees**"

Every other rendering parameter is unpublished. Corroborated independently by
the release: `image_feat` is `(12, 1280)`.

| parameter | ULIP | ours | class |
|---|---|---|---|
| view count | **12** | **11** | **≠** follows MetaFind, not ULIP |
| azimuth spacing | 360/N equal | 360/11 equal | **=** same rule |
| renderer | **Blender** | **pyrender** | **≠ UNREGISTERED DEVIATION** |
| elevation | unpublished | 20.0° | fitted |
| projection | unpublished | orthographic | fitted |
| framing half-width | unpublished | 1.10 | fitted |
| ambient light | unpublished | 0.5 | fitted |
| directional light | unpublished | 1.5 | fitted |
| resolution | unpublished | 224 | chosen for CLIP |
| background | black in released figures | **white** | **≠** `D-11`, USER decision |

"Fitted" = tuned until our renders matched ULIP's released images, e.g. lighting
chosen on over-exposed-pixel fraction and median brightness against their figures
(0.7% / 111 vs their 0.2% / 119). Best available check: **R@1 = 97.2% (n=286)**
retrieving our renders against ULIP's own released `image_feat`.

**Fitting to an output is not the same as knowing the parameter.** These six stay
IMPLEMENTATION CHOICE.

---

## 4. n05 annotation — `annotate.py` / `annotate_run.py` / `describe_rank.py`

**Upstream counterpart: ∅ for generation.** The repo has **no captioning code and
no caption-loading dataset class.** What upstream code actually feeds the text
tower:

* `Objaverse_Lvis_Colored.__getitem__` → `lvis_metadata["value_to_key_mapping"][sample]` — the **category name**
* `ShapeNet.__getitem__` → `synset_id_map[taxonomy_id]["name"]`, `random.choice` of the comma-separated synonyms
* `test_zeroshot_3d_core` (`main.py:350`) → category names × the 64 `templates.json` strings, embeddings averaged

**But the released `.npy` DOES carry captions** (read today):

```
text            ['PB127 Shoe Hi']              <- Objaverse metadata "name"
blip_caption    'a 3d model of a pink shoe'    <- ONE per asset, not 12
msft_caption    'a pink shoe with white sole'
retrieval_text  14 scraped metadata strings
```

**`blip_caption` is a single string per asset.** The 12 views × 10 captions ×
CLIP-rank pipeline in the paper collapses to one stored sentence.

| step | ULIP-2 paper (`main.tex:677`) | ours | |
|---|---|---|---|
| captioner | BLIP-2-opt6.7B | Qwen3.8-27B (4-bit) | **≠** USER decision |
| input per call | **one** image | **all 11** views at once | **≠** measured: per-view named one shovel 4 different things |
| candidates | **10** per image | **5** per asset | **≠** `E-10`, cost |
| ranker | CLIP-ViT-Large | CLIP-ViT-Large | **=** |
| score | image-text similarity | mean cosine over 11 views | **≈** no single "its own image" |
| kept | top-1 | top-1 | **=** (their ablation: 69.7 / 66.7 / 66.4 / 66.3) |
| identity given to the model? | n/a — BLIP-2 is unprompted | LVIS category supplied as fact | **≠** `v8` |
| schema fields | none | MetaFind Figure 2 schema | **∅** MetaFind, not ULIP |

---

## 5. n06 encoding — `encode_text_image.py` / `ulip_backbone.py`

| behaviour | upstream `ULIP2_WITH_OPENCLIP` | ours | |
|---|---|---|---|
| CLIP model | `open_clip ViT-bigG-14`, `laion2b_s39b_b160k` | same, via upstream factory | **=** |
| embed dim | 1280 | 1280 | **=** |
| `encode_pc` | `point_encoder(pc) @ pc_projection` | upstream's, unmodified | **=** |
| `encode_text` | `open_clip_model.encode_text` | same | **=** |
| `encode_image` | `open_clip_model.encode_image` | same | **=** |
| CLIP frozen? | paper 3.3 says frozen; **code only calls `.eval()`**, never `requires_grad=False` | explicit `requires_grad_(False)` + `.eval()` | **≈** we follow the paper; upstream code is looser than its own paper |
| point encoder | trainable | trainable (`train_scope="point_encoder_and_fuser"`) | **=** |
| text prompt | 64 `templates.json` strings, embeddings L2-normed → mean → re-normed | **one** MetaFind template sentence | **≠** MetaFind speaks |
| image aggregation | training samples **one random** view; eval uses none | all 11 encoded, **mean** pooled, all 11 stored | **≠** `U-14` |
| image transform | ULIP-1: `RandomResizedCrop(224, scale=(0.5,1.0))` + ImageNet norm. **ULIP-2 has no training path → no transform in the release** | `open_clip` preprocess for ViT-bigG-14 | **≈** the only defined choice |
| 77-token overflow | silently truncated | **refused** (`P-4`) | **≠** stricter |
| checkpoint | — | `ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt` | official |

---

## 6. Training — `metafind/train/stage1.py` / `metafind/models/losses.py`

### 6a. The finding that matters most

**The released ULIP repo has no ULIP-2 training path.**

`train()` at `main.py:283` reads `inputs[2] tokenized_captions`, `inputs[3] pc`,
`inputs[4] image`, and `customized_collate_fn` filters on `example[4]`. Only
**`ShapeNet` returns a 5-tuple.** `Objaverse_Lvis_Colored` returns
`(data, label, name)` — 3 items — and is only reachable through
`--validate_dataset_name`, i.e. `test_zeroshot_3d_core`.

`scripts/` contains ten shell scripts; the only two ULIP-2 ones are both
`test_*`. There is **no** `pretrain_ulip2_*.sh`.

**So ULIP-2's Objaverse training loop, its augmentation, its epochs, its
learning rate and its batch size are not in the release, and the paper states
none of them.** Anything we write there cannot be compared to upstream code.

### 6b. Where upstream code does exist (ULIP-1 / ShapeNet)

| | upstream | ours | |
|---|---|---|---|
| loss | `ULIPWithImageLoss`: **4** cross-entropies — pc↔text and pc↔image, both directions, `(a+b)/2 + (c+d)/2` | `MetaFindContrastiveLoss`: query↔gallery, `0.5*(q2g + g2q)` | **≠** MetaFind Eq. 5/7b/8 |
| temperature | learnable, init `log(1/0.07)` | learnable, init `0.07`; MetaFind paper says **fixed 0.5** | **≠** flagged DEVIATION in code |
| scale clamp | `logit_scale.data.clamp_(0, 4.6052)` — in place, **after** each step, floor **and** ceiling | `logit_scale.exp().clamp(max=100)` — at use, **ceiling only, no floor** | **≠ see below** |
| normalisation | `F.normalize(..., p=2, dim=-1)` on all three | same on both | **=** |
| gather across GPUs | `all_gather_batch` | single GPU | **≈** n/a |
| optimizer | `AdamW`, split so `ndim<2`/bias/ln/bn get `weight_decay=0` | `AdamW`, **no split** | **≠** |
| lr schedule | custom cosine, `lr 3e-3 → 1e-5`, warmup 1 epoch from `1e-6` | `CosineAnnealingLR` | **≠** |
| betas / eps | `(0.9, **0.98**)` / `1e-8` | PyTorch defaults `(0.9, **0.999**)` / `1e-8` — never set | **≠** |
| AMP | `amp.autocast` + `GradScaler`, on by default | **none** — no autocast, no scaler | **≠** |
| grad clipping | `grad_norm_clip: 10` in the PointBERT yaml (unused by `main.py`) | none | **≈** upstream's loop does not clip either |
| loss finiteness guard | `sys.exit(1)` if `not isfinite(loss)` | none | **≠** |
| ShapeNet augmentation | dropout → scale(0.8,1.25) → shift(±0.1) → rot-perturb → yaw | **none** | **≠** but ULIP's own Objaverse path also has none |

**≠ The clamp difference is real.** Upstream clamps the *parameter* in place with
no grad, so the scale can come back down. Ours clamps the *exponentiated value*,
where `clamp(max=)` has zero gradient above the ceiling — the parameter can drift
up and stick. Ours also has **no floor**, so the scale may fall below 1.
Low-severity, but it is a difference and it is not written down anywhere else.

---

## 7. Open items

| id | item | status |
|---|---|---|
| — | pyrender vs Blender | **DEVIATION, unregistered.** My omission. USER's call. |
| — | pure-torch `fps` vs `pointnet2_ops` | **never compared numerically** — extension not installed |
| — | AdamW no-weight-decay split for bias/norm params | ours does not do it; upstream does |
| — | `logit_scale` clamp form and missing floor | ours differs |
| — | AdamW `betas` 0.999 vs upstream 0.98 | ours never sets it |
| — | no AMP, no non-finite-loss guard | upstream has both |
| `U-02` | rgb scale | **RESOLVED 2026-08-23 → unit [0,1], measured on the official release** |
| — | 6 fitted render parameters | IMPLEMENTATION CHOICE, fitted to output, not known |
| — | 11 views vs 12 | MetaFind's number, not ULIP's |

---

# ADDENDUM 2026-08-23 — after reading the paper source in full

The first pass **grepped** `docs/paper/ulip2_source/` instead of reading it.
`appendix.tex` (74 lines) was never opened. Corrected below. Provenance:
`SOURCE_MANIFEST.json` → arXiv `2305.08275`, archive sha256
`7d274247d248…`, `main.tex` sha256 `fa8d1685…`, `appendix.tex` sha256
`06c1896d…`. The five `sec/*.tex` files are **orphans** — CVPR template
boilerplate, not included by `main.tex`, no ULIP-2 content.

## A1. The one that changes a decision — `appendix.tex:10`

> "In order to fairly compare to OpenShape on Objaverse-LVIS benchmark, which
> utilizes 10k colored point clouds as the 3D input, **we adopt the same 3D
> input preprocessing as in OpenShape**."

**UPSTREAM FACT.** The authority for how a 10k coloured cloud is built is
**OpenShape**, not ULIP. Reinforced twice more:

* `main.tex:677` — "Following ULIP and **OpenShape**, we use 10k, 8k, and 2k points"
* `main.tex:704` — "We follow the same dataset setup and preparation protocols used in ULIP and **OpenShape**"

**OpenShape has never been read in this project.** Our n03 uses
`trimesh.sample.sample_surface`, area-weighted — an IMPLEMENTATION CHOICE made
while the named upstream source sat unexamined. The only paper support for
surface sampling at all is `main.tex:601` ("we extract 3D point clouds **from
the surface**"), which does not specify the method.

**Open item, mine, not previously listed.**

## A2. The caption is PER VIEW at training time — `main.tex:612`

> "given a 3D shape **O**, we extract its 3D point cloud **P**, **randomly
> sample its 2D rendered image I ∼ render(O)**, with **its** BLIP-2 generated
> language description **T ∼ blip2(I)**"

One step = one point cloud + **one** randomly chosen view + **that view's own
caption**. Not a whole-asset caption, not pooled views.

This contradicts what the release actually stores — measured yesterday, each
`.npy` carries a single `blip_caption` string, not 12. Both statements are
evidence; they disagree, and the disagreement stays open.

**Ours differs from both:** one description written from all 11 views at once,
and 11 image embeddings mean-pooled.

## A3. The loss — upstream code matches the paper exactly

`main.tex:614/620/626`:

```
L_P2I = -1/2 Σ [ log softmax over images  +  log softmax over point clouds ]
L_P2T = -1/2 Σ [ log softmax over texts   +  log softmax over point clouds ]
objective:  min over E_P of   L_P2I + L_P2T          <- SUM, not average
```

`ULIPWithImageLoss` computes `(ce+ce)/2 + (ce+ce)/2` — **the same thing**.
τ is stated learnable (`main.tex:616`), which is what the code does. Only
`E_P` is minimised over — the CLIP halves are not in the objective.

## A4. `ViT-G/14` vs `ViT-bigG-14` — resolved by the checkpoint

The paper writes **ViT-G/14** (`main.tex:609`); the code loads
**`ViT-bigG-14`, `laion2b_s39b_b160k`**. In `open_clip` these are different
models — `ViT-g-14` is 1024-d, `ViT-bigG-14` is 1280-d. The released
checkpoint's `pc_projection` is **(768, 1280)**, so **`ViT-bigG-14` is the
correct reading** and the paper's notation is loose. Ours loads bigG. **=**

## A5. Model size — exact match, measured

`tab:ablate_scale_up` sweeps the 3D encoder at 5.3 / 21.9 / **32.5** / 43.1 /
85.7 M and highlights **32.5 M** as "the model setting we use to scale ULIP-2
to the larger Objaverse dataset".

Constructed `PointTransformer_Colored` from the vendored yaml and counted:

```
point encoder                32,499,072  = 32.5 M   <- exact
+ pc_projection (768x1280)      983,040  =  1.0 M
```

**We are running the paper's chosen configuration.** Confirmed, not assumed.

## A6. Ablations I had not reported

**Number of holistic views** (`tab:ablate_num_views`, ShapeNet/SLIP ViT-B, 30
view angles — *not* the Objaverse 12):

```
views    1      2      15     30
top-1  54.8   58.1   69.3   69.7
top-5  77.9   80.5   88.6   88.8
```

15 → 30 buys 0.4. Our **11** sits below the flat part of that curve.

**Top-k captions** (`tab:ablate_top_k`) — and the caption defines what top-k
means: *"top-5 BLIP-2 captions selected means that in the pre-training, we will
**ensemble** the top-5 CLIP ranked captions as the language modality"*. It is an
embedding ensemble, not k separate samples.

```
k        1      3      5      10
top-1  69.7   66.7   66.4   66.3
```

**Captions vs manual** (`tab:ablate-captions`): manual 60.4 → BLIP-2 top-1 **69.7**.

**Captioner** (`tab:ablate_large_multimodal_model`): BLIP 67.7 → BLIP-2 **69.7**.
The paper's own claim is that a stronger captioner lifts the whole method —
which is the argument for our Qwen3.8-27B substitution, and it is the paper's
argument, not mine.

**3D input** (`appendix.tex` `tab:ablate_3d_input`): 8k xyz 48.9/77.1 →
**10k xyzrgb 50.6/79.1**. Colour is worth ~1.7 top-1.

**3D backbone** (`appendix.tex` `tab:ablate_different_3d_backbone`): PointNeXt
ULIP 56.2 → ULIP-2 72.8; Point-BERT ULIP 60.4 → ULIP-2 75.2.

## A7. Other statements now on record

* `main.tex:704` — Objaverse-LVIS is **~46k samples, ~1.2k categories**. Our
  corpus is 46,052 clouds / 45,973 render sets. **Consistent.**
* `main.tex:1425`, Limitations — ULIP-2 is trained on **object-level** shapes and
  the authors flag scene-level as unexplored. MetaFind applies it to scenes; the
  depth-shell path inherits that gap **by the authors' own statement**.
* Checkpoint `…objaverse_shapenet-pretrained.pt` corresponds to Table 1's
  `Objaverse + ShapeNet` row: **50.6 / 79.1**.
* The paper states **no** epochs, learning rate, batch size, optimizer, warmup,
  augmentation or rendering parameters anywhere. Confirmed by reading all 1,474
  lines of `main.tex` and all 74 of `appendix.tex`, not by grep.

## A8. Corrections to the first pass

| first pass said | correct |
|---|---|
| "the ablation compares how many to USE" | it **ensembles** the top-k embeddings |
| n03 sampling is an unconstrained implementation choice | the appendix names **OpenShape** as the source we should have followed |
| release stores one caption ⇒ method uses one caption | the **method** is per-view (`main.tex:612`); the **release** stores one. They disagree |
| — | the 32.5 M configuration is now **verified by parameter count** |
| — | `ViT-G/14` vs `ViT-bigG-14` resolved by `pc_projection` shape |

## A9. Open items added by this pass

| item | status |
|---|---|
| **OpenShape point-cloud preprocessing never read** | our area-weighted surface sampling is unjustified against the named source |
| per-view caption+view pairing vs our whole-asset description | **DEVIATION**, previously unstated in these terms |
| 11 views vs the 15-view knee in `tab:ablate_num_views` | quantified now: 15→30 buys 0.4 top-1; below 15 is unmeasured |
