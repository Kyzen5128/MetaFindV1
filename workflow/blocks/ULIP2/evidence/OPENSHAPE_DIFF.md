# OpenShape vs our ULIP block — file by file

**Clone:** `/home/kyzen/upstream/OpenShape`
**Origin:** `https://github.com/Colin97/OpenShape_code.git`
**Commit:** `abe5aa42b7c99c037c286ad54313f695a114bf0d` (2025-03-16)
**Size:** 7.6 MB, 33 files

**Why OpenShape is in scope at all** — ULIP-2's own appendix:

> `docs/paper/ulip2_source/appendix.tex:10` — "we adopt **the same 3D input
> preprocessing as in OpenShape**"

plus `main.tex:677` ("Following ULIP and **OpenShape**, we use 10k, 8k, and 2k
points") and `main.tex:704` ("the same dataset setup and preparation protocols
used in ULIP and **OpenShape**").

**And the release confirms it.** OpenShape's README documents its `.npy` format
as 15 named keys. The ULIP-2 Objaverse tarball I opened yesterday
(`SFXX/ULIP`, `ULIP-2/objaverse_lvis/000-009.tar.gz`) has **exactly those 15
keys, same names**: `dataset group id text text_feat blip_caption
blip_caption_feat msft_caption msft_caption_feat thumbnail_feat retrieval_text
retrieval_text_feat xyz rgb image_feat`. The ULIP-2 release IS OpenShape's data
format. [INFERENCE, strongly supported]

---

## 1. What OpenShape's paper says — the missing specification

Fetched arXiv **2305.10764**, §3.2 "Ensembling 3D Datasets":

> "for each shape, we **sample 10,000 points from the mesh surface** and
> **interpolate the point colors according to the mesh textures**."

> "We also render **12 color images** from the **preset camera poses that
> uniformly cover the whole shape**."

§6.5 Evaluation Details: "10,000 sampled points **with** point colors" for
Objaverse-LVIS; "10,000 sampled points **without** color" for ModelNet40;
"official 2,048 points without color" for ScanObjectNN.

**The sampling ALGORITHM is not named** (uniform / area-weighted / FPS / Poisson
— none stated). Normalisation of the stored file is not described.

---

## 2. n03 point clouds — `pointclouds.py` vs OpenShape

### 2a. Generation — the paper sentence is the only source; no code exists

**OpenShape's repo has NO point-cloud generation code either.** The clouds ship
pre-made in `openshape-training-data`. So the comparison is against one sentence.

| | OpenShape §3.2 | ours | |
|---|---|---|---|
| count | **10,000** | 10,000 | **=** |
| where from | "**from the mesh surface**" | `trimesh.sample.sample_surface`, area-weighted | **≈** surface sampling; the exact algorithm is unstated so area-weighted is a legitimate reading |
| colour | "**interpolate the point colors according to the mesh textures**" | `to_color()` → per-VERTEX sample, then the triangle's 3 vertex colours **averaged** | **≠ REAL DIFFERENCE — see 2b** |

### 2b. ≠ The colour difference, precisely

```
OpenShape:  texture image --> barycentric lookup AT THE SAMPLED POINT
ours:       texture image --> per-vertex sample (trimesh to_color())
                          --> mean of the face's 3 vertex colours
                          --> ONE colour per face
```

`pointclouds.py:432` — `cols = _vertex_rgb(part)[tri].mean(axis=1)`.

Every point that lands on the same triangle gets the **same** colour. OpenShape
interpolates per point. On a coarse mesh with a detailed texture ours loses that
detail; on a dense mesh the two converge.

**Scope: the `texture` class is 23,675 of 46,052 assets — 51% of the corpus.**

`trimesh.sample.sample_surface` already returns `face_idx`, and barycentric
coordinates plus a UV lookup is the operation OpenShape describes, so this is
implementable rather than blocked. **Previously unrecorded. Mine.**

Our other colour classes — `flat` (13,524), `gltf_default` (8,853),
`vertex`/`face`, `fallback_grey` — have **no OpenShape counterpart**: their
sentence covers textures only. Those remain our IMPLEMENTATION CHOICE.

### 2c. Load-time preprocessing — this IS in code, `src/data.py` + `src/utils/data.py`

This is what ULIP-2's appendix means by "3D input preprocessing".

```python
n   = data['xyz'].shape[0]
idx = random.sample(range(n), self.num_points)     # random subsample, NO FPS
xyz = data['xyz'][idx];  rgb = data['rgb'][idx]
if self.y_up:      xyz[:, [1, 2]] = xyz[:, [2, 1]]     # SWAP y and z
if self.normalize: xyz = normalize_pc(xyz)
if train and self.augment:         xyz = augment_pc(xyz)
if train and self.random_z_rotate: xyz = random_rotate_z(xyz)
if train and rand() < self.rgb_random_drop_prob:  rgb = np.ones_like(rgb) * 0.4
features = np.concatenate([xyz, rgb], axis=1)
```

`train.yaml`: `num_points 10000` · `y_up True` · `normalize True` ·
`random_z_rotate True` · `use_color True` · `rgb_random_drop_prob 0.5` ·
`augment True`

| behaviour | OpenShape | ULIP-2's `Objaverse_Lvis_Colored` | ours | |
|---|---|---|---|---|
| subsample | `random.sample` to 10,000 | none (reads all 10,000) | none | **=** ULIP-2 |
| `normalize_pc` | mean-centre, ÷ max‖·‖, **with a `< 1e-6` degenerate guard** | same formula, **no guard** | same formula, **raises** on degenerate | **≈** |
| **y↔z swap** | **YES** (`y_up: True`) | **NO** | **NO** | **≠ see below** |
| rgb scale | **`[0, 1]`, stated in README** | inherits file | `/255` → `[0,1]` | **=** |
| rgb → 0.4 grey | **augmentation, p = 0.5** | fallback when a dataset has no colour | fallback `DEFAULT_GREY` | **≠ see below** |
| train augment | dropout → scale(.8,1.25) → shift(.1) → rot-perturb | (ShapeNet path only) | none | **≠** |
| random z rotation | **YES, uniform 0-2π** | no | no | **≠** |
| concat | `[xyz, rgb]` | same | same | **=** |

**≠ The 0.4 grey.** Our code comments call `DEFAULT_GREY = 0.4` "ULIP's stand-in
for a dataset with no colour channel at all". Its actual origin is
**OpenShape's RGB-dropout augmentation fired at p = 0.5 during training**
(`data.py:63`). ULIP inherited the constant into its ModelNet fallback. Our
comment attributes it to the wrong mechanism. **Cosmetic — the value is the same
— but the provenance in the code is wrong.**

**≠ The y↔z swap.** OpenShape's README has a `gravity-axis` column: its main
checkpoints are **z-axis**, two demo ones are **y-axis**. OpenShape swaps y↔z at
load so gravity lands on z. **ULIP-2 does not swap** — `Objaverse_Lvis_Colored`
feeds the stored orientation straight in. So the two projects train PointBERT on
**different gravity conventions from the same stored files**. We follow ULIP-2
(no swap) plus our own 180° yaw about Y. Note a y↔z swap has **det = −1** — it
is a reflection, not a rotation.

### 2d. Measured agreement (yesterday, 5 overlapping uids)

median Chamfer **0.0251**, median |rgb mean difference| **0.0025**.

---

## 3. n04 renders — `renders.py` vs `render_single_glb.py`

**This is the only rendering code in the entire ULIP/OpenShape lineage.**
254 lines, BlenderProc.

```python
render.engine            = CYCLES (or BLENDER_EEVEE)
render.resolution_x/y    = 512, 512
image_settings           = PNG, RGBA
scene.render.film_transparent = True            # TRANSPARENT background
cycles.samples           = 32
cycles.diffuse_bounces   = 1
cycles.glossy_bounces    = 1
cycles.transparent_max_bounces = 3
cycles.transmission_bounces    = 3
cycles.filter_width      = 0.01
cycles.use_denoising     = True

add_lighting():  ONE AREA light, energy = 30000, z = 0.5, scale 100x100x100

normalize_scene():  scale = 1 / max(bbox_max - bbox_min) * 0.8
                    then translate so the BBOX CENTRE sits at the origin

setup_camera():  lens = 35 mm, sensor_width = 32       -> PERSPECTIVE
                 TRACK_TO an empty at the origin, up_axis = UP_Y
                 --camera_dist default 1.5; the header comment uses 1.2

load_object():   bpy.ops.import_scene.gltf(..., merge_vertices=True)

outputs: colors, normals, depth
```

**The 12 camera poses are NOT one azimuth ring.** They are three polar rings of
four, staggered in azimuth:

```
phi = pi/3   ( 60 deg, ABOVE)   theta =  30, 120, 210, 300
phi = pi/2   ( 90 deg, LEVEL)   theta =  60, 150, 240, 330
phi = 2pi/3  (120 deg, BELOW)   theta =   0,  90, 180, 270
```

`phi = 2pi/3` is **30° below the equator** — four views look UP at the object.
This matches OpenShape's own paper wording, "preset camera poses that
**uniformly cover the whole shape**", and it matches what MetaFind's Figure 2
shows. It does **not** match ULIP-2's sentence "spaced equally by 360/12
degrees", nor ULIP-2's own Figure 2, which draws the 12 views in a single ring
around a 360° camera icon.

### 3a. Side by side

| | OpenShape code | ours | |
|---|---|---|---|
| renderer | **Blender / BlenderProc, CYCLES** | **pyrender** | **≠ unregistered** |
| views | 12, **3 rings x 4** | 11, **1 ring at 20° elevation** | **≠** count is MetaFind's; layout is ours |
| sees underside? | **yes**, 4 views at −30° | **never** | **≠** |
| projection | **perspective**, 35 mm / 32 mm sensor | **orthographic** | **≠** |
| resolution | **512 x 512** | **224 x 224** | **≠** |
| background | **transparent** (`film_transparent`, RGBA) | **opaque white** | **≠ `D-11`** |
| normalisation | `1 / max(bbox extent) * 0.8`, **bbox centre** to origin | fit to `ORTHO_HALF_WIDTH = 1.10` | **≠** |
| lighting | one area light, energy 30000, scale 100 | ambient 0.5 + directional 1.5 | **≠** |
| samples | 32, denoised, 1 diffuse + 1 glossy bounce | rasteriser, no ray tracing | **≠** |
| glTF import | `merge_vertices=True` | `meshload.load_scene`, **merge disabled** (it breaks COLOR_0 alignment) | **≠**, ours deliberate |
| extra outputs | normals + depth | none | **≠**, unused |

### 3b. The background question, resolved

Three descriptions that looked contradictory are one file read three ways:

* **OpenShape code** writes **transparent RGBA** (`film_transparent = True`,
  `set_output_format(enable_transparency=True)`).
* **Our registry** records "ULIP-2's released renders measure corner luminance
  0, pure black". **A transparent PNG opened and `.convert("RGB")`'d has black
  corners** — alpha is discarded and the RGB underneath is `(0,0,0)`.
* **ULIP-2's own Figure 2** shows the renders composited on **light grey** — and
  BLIP-2's own caption in that figure reads *"statue of king benjamin on a
  **grey background**"*.
* **MetaFind's Figure 2** shows them on the figure's cream hatch — i.e. alpha.

**Caveat:** the render images that "corner luminance 0" was measured on are **no
longer on this machine** (`find` over the data root returns nothing), so I
cannot re-measure it. The reconciliation above is INFERENCE from the code, not a
re-measurement.

And the number that already exists (`S-5`, retrieval against ULIP's own
`image_feat`, n = 286):

```
white + xmag 1.10    R@1 97.2%   matched cos 0.9160   gap 0.3734
black + xmag 1.20    R@1 95.8%   matched cos 0.8782   gap 0.3404
```

---

## 4. What is NOT in OpenShape

* **No captioning code.** `blip_caption` ships in the data; nothing generates it.
* **No point-cloud generation code.** Ships pre-made.
* Its training (`src/train.py`, `train.yaml`) is its own — MinkResNet34,
  `logit_scale_init 14.28`, `lr 0.001`, decay 0.95 / 10k steps, warmup 10k,
  `max_epoch 1000`, batch 200, `lambda_img_contras 1`, `lambda_text_contras 1`.
  **MetaFind does not adopt OpenShape's training**, so this is context only.

---

## 5. Open items this reading adds

| item | status |
|---|---|
| **texture colour: per-face average vs OpenShape's per-point interpolation** | **REAL, unrecorded, 51% of the corpus.** Mine. |
| `DEFAULT_GREY = 0.4` provenance comment names the wrong mechanism | cosmetic; value unaffected |
| y↔z gravity swap: OpenShape yes, ULIP-2 no, ours no | we follow ULIP-2; worth stating explicitly |
| single 20° ring vs OpenShape's 3 rings incl. −30° | **the biggest render difference; larger than background** |
| perspective vs orthographic | OpenShape perspective 35 mm; ULIP-2/MetaFind text says "orthogonal" |
| 512 vs 224 resolution | ours chosen for CLIP input |
| "corner luminance 0" cannot be re-measured | source images gone from this machine |
