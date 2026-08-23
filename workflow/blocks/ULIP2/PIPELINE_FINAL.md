# ULIP block — final settings and verification plan
2026-08-23. Every number below is MEASURED on this machine unless marked otherwise.

Hardware: Intel Ultra 7 265KF (20 threads) · 61 GB RAM · RTX 5090 32 GB (OptiX) ·
NVMe 937 GB · SMR HDD 3.6 TB.

---

## 0. Code changes to make (none applied yet)

| # | file | change | evidence |
|---|---|---|---|
| 1 | `data/pointclouds.py` | texture colour: per-face average -> `sample_surface(sample_color=True)` per-point bilinear texture lookup | OpenShape paper §3.2 "interpolate the point colors according to the mesh textures" |
| 2 | `data/pointclouds.py` | multiply `baseColorFactor` into the texture | glTF 2.0; USER decision 2026-08-23 |
| 3 | `data/pointclouds.py` | `SAMPLER_VERSION 6 -> 7` | forces regeneration |
| 4 | `data/renders.py` | pyrender -> Blender/BlenderProc wrapper | USER decision |
| 5 | `data/renders.py` | `RENDERER_VERSION 4 -> 5` | forces regeneration |
| 6 | `data/annotate_run.py` | 5 sequential `generate()` -> `num_return_sequences=4` + 1 | **implementation only, method unchanged** |
| 7 | new `data/view_io.py` | one `load_view_rgb()` flattening RGBA onto BLACK | USER decision 1 |
| 8 | `models/losses.py` | `learnable_temperature=False`, `init_temperature=0.5` | `3experiments.tex:15` PAPER FACT |
| 9 | `models/fusion.py` | `kind="transformer"`, close `U-13` | `3experiments.tex:143` PAPER FACT |

Docs-only: register pyrender->Blender and 12-vs-11 as deviations; fix `D-11`
wording (source is transparent, not black); label `identity_confirmed` and the
cm/kg units as IMPLEMENTATION CHOICE / INFERENCE; correct the `DEFAULT_GREY`
provenance comment.

---

## 1. n03 point clouds

```
points per asset      10,000
sampling              trimesh.sample.sample_surface, area-weighted
colour  (texture)     sample_color=True -> per-point bilinear UV lookup   [CHANGED]
        (flat)        baseColorFactor
        (gltf_default) [1,1,1,1]
        (vertex/face) geometry colours, barycentric interpolated          [CHANGED]
        (fallback)    0.4 grey
baseColorFactor       multiplied into the texture                         [CHANGED]
COLOR_0               multiplier on flat / gltf_default; texture -> RE-MEASURE (D-12)
rgb range             [0, 1] float32
normalisation         pc_norm: centroid -> 0, max radius -> 1, at WRITE time
frame                 +180 deg yaw about Y (FRAME_CORRECTION_ID yaw180_about_y@ulip2_frame)
seed                  uid_seed(uid) = sha256(uid)[:8]
storage               .npz {xyz (10000,3) f32, rgb (10000,3) f32} + .json sidecar
parallelism           ThreadPool, 8 workers  (measured 986 assets/min)
SAMPLER_VERSION       7
```
**Measured cost: 46,052 assets in ~47 minutes.**

## 2. n04 renders

```
renderer              Blender 4.2.1 + BlenderProc 2.8.0, CYCLES, OptiX on the 5090
script                metafind/vendor/openshape/render_single_glb.py, byte-identical
                      except two commented lines (normals + depth passes)
views                 12 = three polar rings of four
                        phi  60 deg (above)  theta  30 120 210 300
                        phi  90 deg (level)  theta  60 150 240 330
                        phi 120 deg (below)  theta   0  90 180 270
resolution            512 x 512
format                PNG RGBA, film_transparent = True
projection            perspective, lens 35 mm, sensor_width 32
camera_dist           1.2
lighting              one AREA light, energy 30000, z = 0.5, scale 100 x 100 x 100
cycles                samples 32, denoising on,
                      diffuse 1 / glossy 1 / transparent 3 / transmission 3 bounces
                      filter_width 0.01
normalisation         scale = 1 / max(bbox extent) * 0.8, then bbox centre -> origin
glTF import           merge_vertices = True (upstream default)
parallelism           8 concurrent processes  (P>=10 dies: CUDA context OOM)
RENDERER_VERSION      5
```
**Measured: 3.234 s/asset at P=8 -> 45,973 assets in 41.3 h = 1.72 days, 77 GB.**

Parallel scaling actually measured (16 assets, 192 images each config):
```
P= 1  8.12 s/asset  1.00x  192/192      P= 8  3.36 s/asset  2.41x  192/192
P= 2  4.54 s/asset  1.78x  192/192      P= 9  3.28 s/asset         192/192
P= 4  3.56 s/asset  2.28x  192/192      P=10       FAILS            48/192
P= 6  3.42 s/asset  2.37x  192/192      P=12       FAILS            96/192
```
Resolution alternatives, same asset: 512 -> 8.00 s, 384 -> 5.43 s, 256 -> 3.69 s.
**512 kept**: only Gemma consumes detail above 224, and its vision tower upscales
to 896, so a larger render is real information to it.

## 3. n05 annotation

```
model                 gemma-4-12B-it, bf16, NO quantisation, loaded from NVMe
                      (NVMe 12.6 s vs SMR 150.2 s to load)
input                 all 12 views in ONE conversation turn
                      RGBA flattened onto BLACK before the model sees them
prompt                PROMPT_VERSION 8, LVIS category supplied as fact
description candidates 5, sampled independently, CLIP-ranked, top-1 kept   [ULIP-2 method KEPT]
  implementation      num_return_sequences=4 in one call + 1 separate call  [CHANGED, method identical]
ranker                openai/clip-vit-large-patch14, mean cosine over the 12 views
structured fields     category, synset, height, width_axis, mass, materials,
                      onCeiling, onWall, onFloor, onObject
env                   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   [REQUIRED, else OOM]
```
**Measured per asset:**
```
                          current      optimised
5 description candidates   10.94 s       6.36 s   (4.46 batched + 1.90 single)
structured JSON call        2.98 s       2.98 s
CLIP ranking                0.14 s       0.14 s
                          --------      -------
                           14.06 s       9.48 s
full corpus                 7.48 d       5.04 d
```
GPU memory measured with the ranker resident: n=5 OOM, n=4 30.1 GB, n=3 28.6 GB
(fp16 ranker: n=5 still OOM, n=4 29.3 GB).

## 4. n06 encoding

```
model                 open_clip ViT-bigG-14, laion2b_s39b_b160k, FROZEN
image                 all 12 views encoded, mean-pooled, ALL 12 stored (U-14)
                      same load_view_rgb() -> black flatten
text                  MetaFind TEXT_TEMPLATE; >77 true BPE tokens is REFUSED, not truncated
output                float16 per-asset sidecar
```
Estimated ~2 hours. Not measured.

## 5. Training

```
temperature           0.5, FIXED            [PAPER FACT 3experiments.tex:15]
fusion                transformer           [PAPER FACT 3experiments.tex:143]
split                 80 / 20, seed 20260816
modality masking      30% independent per modality, masked embeddings not zero-pad
Stage 1 loss          query -> gallery, one direction
Stage 2 loss          bidirectional, 0.5 * (q2g + g2q); gallery frozen
epochs / lr / batch / optimizer / scheduler   NOT STATED BY THE PAPER -> ours
```

---

# Verification plan

`SOFT` = record and report, keep going. `HARD` = stop.

## Phase 0 — code

| gate | check | level |
|---|---|---|
| V0.1 | `ContrastiveConfig()` -> tau 0.5, not learnable, no DEVIATION warning | HARD |
| V0.2 | `ModalityFusion(FusionConfig(dim=1280))` : (2,3,1280) -> (2,1280), finite under masking | HARD |
| V0.3 | `load_view_rgb()` on a synthetic 50%-alpha PNG returns exactly `round(fg*a)` | HARD |
| V0.4 | `pytest tests/` green | HARD |
| V0.5 | vendored `render_single_glb.py` differs from upstream only in the two commented blocks | HARD |

## Phase 1 — point clouds

| gate | check | level |
|---|---|---|
| V1.0 | **D-12 re-measure**: 37 texture+COLOR_0 assets that ULIP also publishes, new sampler, COLOR_0 on vs off, cosine through the frozen ULIP-2 encoder against ULIP's own cloud. Pre-registered rule, fixed BEFORE seeing the numbers: a margin inside the standard error of a coin flip prints WITHIN NOISE and `R-11` keeps `D-12` as it stands. **[ADDED by ULIP2 Reviewer 2026-08-23 -- selection bias]** the effect size depends on texture detail (averaging is a low-pass filter, so the two rules diverge most where the texture has detail), but those 37 were selected by *overlapping with ULIP*, not by detail. So the run must ALSO report the sampled-RGB variance of the 37 against all 23,675 texture assets: without it, "no difference" cannot be told apart from "this sample happens to be low-detail". n=37 is the ceiling available from the 11 shards on disk and was already inconclusive (16/37) under the old rule. | HARD (decides a setting) |
| V1.1 | 20-asset smoke: shape (10000,3)+(10000,3) f32, rgb in [0,1], centroid ~0, max radius 1.000, no NaN | HARD |
| V1.2 | full run: 46,052 npz + 46,052 sidecars, `sampler_version == 7` everywhere, quarantine 0 | HARD |
| V1.3 | colour_source histogram vs the old run (texture 23,675 / flat 13,524 / gltf_default 8,853) | SOFT |
| V1.4 | vs ULIP released clouds, 3,706 overlap: median Chamfer (was 0.0251), median rgb-mean gap (was 0.0025). Texture class should get CLOSER. | SOFT |

## Phase 2 — renders

| gate | check | level |
|---|---|---|
| V2.0 | 10-asset timing; extrapolate and report the real day count | SOFT |
| V2.1 | contact sheet of 5 assets x 12 views, inspected by eye: no blank views, the three rings visibly differ, the below-ring really sees the underside | SOFT |
| V2.2 | every view non-empty; object does not touch the frame edge; alpha channel present and not all-255 | HARD |
| V2.3 | full run: 45,973 sidecars, `renderer_version == 5`, exactly 12 PNGs each; failure rate <1% SOFT, >5% HARD | both |
| V2.4 | `S-5` against ULIP's released `image_feat`. **[REDESIGNED by ULIP2 Reviewer 2026-08-23 -- a threshold was the wrong instrument.]** The engineer proposed "R@1 >= 90% or stop". The Reviewer declined to bless any number, on evidence already in this repo's own HANDOFF: encoding **ULIP's own released PNGs** with our transform scores its diagonal (0.68, 0.76) BELOW its off-diagonal (0.70, 0.78) -- `image_feat` cannot be reproduced from the pixels upstream itself published. An uncalibrated target makes 97.2% and 90% equally meaningless. **Replaced by a control arm**: run ULIP's released PNGs through the same harness to get the ceiling our transform can reach, then report ours as a percentage OF THAT. Ceiling >> ours means a real render gap worth chasing; ceiling ~= ours means `S-5` is a diagnostic and must be demoted from gate, and a drop then needs no defence because it was never calibrated. This substitutes a measurable reference for a number the engineer chose, and removes the failure mode where a criterion is quietly dropped once it stops agreeing. | SOFT, plus the demotion decision |
| V2.5 | CYCLES is NOT bit-reproducible: two identical runs differ by <=1/255. Record this; do not treat re-render equality as a check. | note |

## Phase 3 — annotation

| gate | check | level |
|---|---|---|
| V3.0 | 5-asset smoke: all 10 structured fields present, English, valid JSON, 5 distinct candidates, CLIP scores in range | HARD |
| V3.1 | **candidate-quality parity**, NOT an equivalence proof. 20 assets both ways: mean CLIP of the winner, distinct-candidate count, non-English rejection rate. **[CORRECTED by ULIP2 Reviewer 2026-08-23]** This gate cannot prove equivalence, because the two paths are *not* expected to emit the same tokens: batch shape changes bf16 matmul reduction order and a max logit difference of 1.25e-1 between batch=1 and batch=5 was measured, which is enough to flip a token. What licenses "implementation-only" is the **source**, not this gate: `transformers 5.15.0` `_expand_inputs_for_generation` (generation/utils.py:929) repeat_interleaves the prompt into n identical rows and `_sample` (2921-2923) draws with `torch.multinomial` over `(n, vocab)`, which samples each row independently. Valid ONLY for `do_sample=True, num_beams=1`, now asserted in `Annotator.generate`. | HARD |
| V3.2 | 100 frozen assets (`sample_100.jsonl`, sha256 `3dbb2ec8de8d8710`): category hit rate, mean CLIP, violation rate, vs the old-render run | SOFT |
| V3.3 | full run: completion count, failure classes, field-validation pass rate; <2% SOFT, >10% HARD | both |

## Phase 4 — encoding

| gate | check | level |
|---|---|---|
| V4.1 | pre-flight: every serialized text <= 77 true BPE tokens BEFORE any GPU work | HARD |
| V4.2 | embeddings finite, unit-normalisable, 1280-d; 12 view vectors stored per asset | HARD |

## Reproducibility recorded per phase

git SHA · exact command · seeds · version constants · Blender/BlenderProc/torch/CUDA
versions · output paths · wall time · failure counts.
