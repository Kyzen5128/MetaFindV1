# MetaFind paper — read verbatim, 2026-08-23

Every line of `docs/paper/metafind_source/` read, not grepped:
`neurips_2025.tex` 127 · `2methdology.tex` 134 · `3experiments.tex` 143 ·
`appendix.tex` 125 · `4backgound.tex` 12 — **541 lines total**, plus both
referenced figures opened as images.

Provenance: `SOURCE_MANIFEST.json` → arXiv **2510.04057**, archive sha256
`61351d34…`. No orphan tex files — everything in the directory is included by
`neurips_2025.tex`.

---

## 1. Everything the paper says about the ULIP block

### 1.1 Renders — PAPER FACT

> `neurips_2025.tex:100` — "we annotate **48K** 3D assets from the Objaverse-LVIS
> subset, each rendered from **11 views** and processed with **GPT-4o**"

> `2methdology.tex:28` — "approximately **48,000** distinct 3D assets. Each asset
> is rendered from **11 orthogonal viewpoints** and annotated using **GPT-4o**."

> `2methdology.tex:24` (Fig. 2 caption) — "rendered from multiple **orthogonal
> views** and passed through a **VLM**"

**That is the complete rendering specification.** 11 views, the word
"orthogonal" twice. No resolution, camera, lens, elevation, framing, lighting,
background, renderer, or file format.

"Orthogonal" is ambiguous — it can mean orthographic projection, or mutually
perpendicular directions. **Eleven directions cannot be mutually perpendicular
in 3-space** (six is the maximum), so the mutually-perpendicular reading is
impossible at n=11. That leaves orthographic projection or loose wording, and
the paper never disambiguates. It is weak support for `PROJECTION =
"orthographic"` — more than nothing, less than a specification.

### 1.2 Annotation — PAPER FACT, schema read off Figure 2

The schema is in the **figure image**, not the text. Transcribed from
`data-preprocess.png`:

```json
{"annotations": {
  "category": "robot",  "synset": "robot.n.01",
  "width": 30, "length": 30, "height": 40, "volume": 36000, "mass": 2.5,
  "description": "A small cubic-shaped robot with a smiling screen face, two
                  antennae on top, and rounded side arms and feet with
                  spring-like connectors.",
  "materials": ["metal", "glass", "plastic"],
  "onCeiling": false, "onWall": false, "onFloor": true, "onObject": true}}
```

Read directly off the figure:

* **`volume` = 36000 = 30 × 30 × 40 exactly.** The identity holds in the paper's
  own example.
* **`onFloor` and `onObject` are BOTH true.** The placement flags are
  multi-label, not mutually exclusive. Our four-boolean handling is correct.
* **No units anywhere.** 30/30/40 and mass 2.5 carry no unit. Centimetres and
  kilograms are our **INFERENCE**, not a paper fact.
* **`synset` is in the schema.** We keep it in the record and drop it from the
  encoder text — that omission is ours.
* **`identity_confirmed` is NOT in the schema.** It is ours, an
  IMPLEMENTATION CHOICE, and should be labelled as one.
* The prose names the same fields: "object category, size dimensions, materials,
  and placement constraints" (`2methdology.tex:28`).

**No prompt, no generation procedure, no per-view-vs-per-asset rule, no
candidate count, no ranking step.** The paper says only "processed with GPT-4o".

### 1.3 Point clouds — **the paper says NOTHING**

Read all 541 lines. "Point cloud" appears only as the name of a modality
(`q_pc`, `e_pc`, "3D Encoder"). There is **no** statement of:

point count · sampling method · normalisation · colour · file format ·
frame/orientation · which ULIP-2 checkpoint · which CLIP · backbone size

**n03 has zero MetaFind authority.** Under `U-O` precedence it falls through to
ULIP-2, whose appendix (`appendix.tex:10`) then defers to **OpenShape**.

### 1.4 Training — PAPER FACT

| | value | where |
|---|---|---|
| split | **80% train / 20% test**, both datasets | `3experiments.tex:8` |
| temperature | **"The temperature is 0.5 for all experiments."** | `3experiments.tex:15` |
| Stage 1 | both towers trained; modality masking **30%** independent per modality; **masked embeddings**, not zero-padding | `2methdology.tex:75` |
| Stage 1 loss | **query→gallery only**, single direction | `2methdology.tex:77` |
| Stage 2 | scene dropout **30% of batches**; only fusion + ESSGNN updated; **gallery frozen** | `2methdology.tex:89` |
| Stage 2 loss | **bidirectional**, `½(L_q2g + L_g2q)` | `2methdology.tex:94-102` |
| fusion | candidates: mean pooling / MLP / masked MLP / gated / Transformer; **"the final selected Transformer"** | `2methdology.tex:34`, `3experiments.tex:143` |
| layout fusion | `e_query = Fusion(e_text, e_img, e_pc) + λ·e_layout`, **λ learnable scalar** | `2methdology.tex:85` |
| gallery | "modality-complete and **frozen after pretraining**" | `2methdology.tex:34` |

Table 3 (`3experiments.tex:94-108`), Text-Only R@1:

```
Full (bidirectional) w/ iterative retrieval & ESSGNN   11.4
w/o iterative retrieval                                11.3
w/o Layout Context                                     13.5
w/ Layout Context (GAT)                                11.0
Fusion = Mean                                           9.4
Fusion = MLPs                                           9.9
Modality Dropout = 10%                                  7.3
Modality Dropout = 50%                                 13.2
Train fuser only                                        8.7
Padding missing modalities with 0                      10.5
```

`3experiments.tex:143` — "full encoder fine-tuning yields better performance by
allowing earlier layers to adapt". **The prior citations in
`metafind/models/ulip_backbone.py` to "paper 2.6" and "paper 3.4" both check
out**: §2.6 is Training Strategy, §3.4 is Ablation Studies.

**Not stated anywhere:** epochs, learning rate, batch size, optimizer, scheduler,
warmup, weight decay, seed, hardware, ULIP-2 checkpoint identity.

### 1.5 Figure 1 (`MetaFind.drawio.png`)

The ULIP-2 block is labelled **"ULIP-2 (Shared)"** — one encoder stack serving
both towers. The text qualifies it: the gallery is frozen after Stage 1 while
the query side keeps training, so "shared" is architectural, not
weight-identical after Stage 2.

Text input shown as `Platform Bed {size:.........}` — category plus structured
fields, which is the shape of our `TEXT_TEMPLATE`.

---

## 2. What Figure 2's renders actually show — and where we differ

The eight render thumbnails were cropped and upscaled 3× and inspected.

**① Background is TRANSPARENT.** The figure's cream hatched pattern shows
through behind every robot. Not black, not white — alpha.

Independent corroboration: OpenShape's `render_single_glb.py` sets
`scene.render.film_transparent = True` and
`bproc.renderer.set_output_format(enable_transparency=True)`, and writes RGBA.

Ours is opaque white (`BACKGROUND_RGBA = [255,255,255,255]`, `D-11`). Flattening
alpha onto white reproduces white, so the choice is compatible — but the source
is alpha and the registry should say so.

**② The views span MULTIPLE ELEVATIONS, including from below.** In the crop,
one view looks up at the robot's underside and foot; others look down from
above; others are level. This is not a single-elevation orbit.

**Ours is a single ring at `ORBIT_ELEVATION_DEG = 20.0` for all 11 views.**

Independent corroboration: OpenShape's 12 views are **three rings of four**,
staggered in azimuth —

```
phi = pi/3   (60 deg,  above)   theta =  30, 120, 210, 300
phi = pi/2   (90 deg,  level)   theta =  60, 150, 240, 330
phi = 2pi/3  (120 deg, BELOW)   theta =   0,  90, 180, 270
```

`phi = 2pi/3` is 30° **below** the equator. That is exactly the family of views
Figure 2 shows, and it is **not** what ULIP-2's sentence ("spaced equally by
360/12 degrees") describes either.

**This is a discrepancy our render registry does not currently carry**, and it
is larger than the background one: a single 20° ring never sees the underside
of an asset, and the annotation model is shown only what the renders show.

**③ Perspective vs orthographic.** At this resolution I will not claim either
way from the figure. OpenShape's camera is `lens = 35mm`, `sensor_width = 32` —
**perspective**. The MetaFind text says "orthogonal viewpoints". Unresolved.

---

## 3. Where our ULIP block stands against MetaFind specifically

| our choice | MetaFind says | verdict |
|---|---|---|
| 11 views | **11 views** | **PAPER FACT, match** |
| annotation fields category/synset/w/l/h/volume/mass/description/materials/4 placement flags | same 12 fields, Figure 2 | **match** |
| `identity_confirmed` extra field | not in schema | **ours — IMPLEMENTATION CHOICE** |
| centimetres, kilograms | no units given | **INFERENCE** |
| Qwen3.8-27B annotator | **GPT-4o** | **DEVIATION from a PAPER FACT** (USER-approved) |
| 5 candidates + CLIP rank | not mentioned | ours, inherited from ULIP-2 |
| single 20° elevation ring | figure shows multiple elevations incl. below | **DISCREPANCY, unregistered** |
| opaque white background | figure shows transparent | **DEVIATION `D-11`, source is alpha not black** |
| orthographic projection | "orthogonal viewpoints" ×2 | weak support, still ambiguous |
| 10,000 xyzrgb points, area-weighted surface sampling, `pc_norm` | **nothing at all** | **no MetaFind authority; falls to ULIP-2 → OpenShape** |
| temperature 0.07 learnable | **0.5, fixed, all experiments** | **DEVIATION, already flagged in `losses.py`** |
| Stage 1 q2g only | q2g only | **match** |
| Stage 2 bidirectional ½(q2g+g2q) | same | **match** |
| 30% modality masking | 30% | **match** |
| masked embeddings not zero-pad | same | **match** |
| gallery frozen in Stage 2 | same | **match** |
| fusion `masked_mlp` (default, `U-13` open) | **"the final selected Transformer"** | **MISMATCH — and `U-13` can now be CLOSED** |
| `TRAIN_FRACTION = 0.8`, seed 20260816 | 80/20 | **match** (verified `splits.py:65`) |
| `p_mask = 0.30` | 30% | **match** (verified `fusion.py:89`) |

---

## 4. Open items this reading adds

| item | status |
|---|---|
| **single-elevation 20° ring vs multi-elevation views in Fig. 2** | unregistered discrepancy, mine |
| background: source is **transparent**, not black | `D-11` wording is wrong about the reference |
| units for width/length/height/mass | **UNKNOWN in the paper**; cm/kg is INFERENCE |
| `identity_confirmed` | not a paper field; label as IMPLEMENTATION CHOICE |
| **fusion: ours defaults to `masked_mlp`, paper selected Transformer** | verified `stage1_config.py:13`. `U-13` is open in our code and the paper answers it: `3experiments.tex:143` names Transformer as the final selection. **U-13 can be closed as PAPER FACT.** |
| 80/20 split, 30% masking | **verified, both match** |
