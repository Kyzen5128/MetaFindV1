# n03 / n04 upstream verification — findings

> **Evidence document. Kept verbatim as produced.** Its measurements are current and
> load-bearing; its cross-references name work packages that no longer route anywhere.
> Read it for the measurements, not for project state.

**Executed directly by Master 2026-08-21 at the user's instruction** ("先不要管角色了 先解決"),
not through the contract in `TASK.md`. Read-only: no code, test, or data was modified.

**Sources**
- `/home/kyzen/upstream/ULIP` @ `95d480fe2b16c06d0558c60b5cfea981b4cdc8eb`
- `/home/kyzen/upstream/egnn` @ `e9ca6c0c3e1d30a7598efbd66034121b4af8dccc`
- Official ULIP-2 Objaverse-LVIS clouds, **already on disk**:
  `data/models/hf-cache/datasets--SFXX--ULIP/blobs/ef8ffe19…` (1.1 GB, `ULIP-2/objaverse_lvis/000-009.tar.gz`, 4,999 `.npy`, downloaded 2026-08-15)
- `docs/paper/metafind_source/{2methdology,3experiments,appendix}.tex`
- Repo @ `468bbac`

---

## Headline

**No re-run is required.** Both nodes are verified functional against the official upstream
artifacts. One real geometric defect was found (a systematic 180° yaw) and measured to have
**no effect** on the embeddings the pipeline actually consumes.

---

## FIND-1 — `pc_norm` is identical to upstream. `UPSTREAM FACT`

`dataset_3d.py` holds three normalisation functions — `pc_normalize()` at `:33`, and
`pc_norm()` methods at `:381` and `:496`. **All three are byte-identical in behaviour:**
subtract the centroid, divide by the largest radius.

`metafind/data/pointclouds.py:93-98` is line-for-line the same. Its docstring cites
`dataset_3d.py:496-502`; that citation is correct.

**Correction to my own earlier claim.** `D15/TASK.md` §5 says "which one the Objaverse/ULIP-2
path actually uses is the first thing to determine — they may differ". They do not differ.
That question was answerable in under a minute and should never have become a contract item.

## FIND-2 — Point count, colour flag and normalisation order all match. `UPSTREAM FACT`

`Objaverse_Lvis_Colored.__init__` (`dataset_3d.py:456-505`):

| | upstream | ours |
|---|---|---|
| points | `self.npoints = 10000` | 10,000 |
| colour | `self.use_color = True` | on |
| order | `pc_norm(xyz)` **then** `concatenate([data, rgb])` — rgb never normalised | same |
| dtype (released) | `float16` | we store `float32` |

## FIND-3 — No farthest-point sampling in the Objaverse path. `UPSTREAM FACT`

`Objaverse_Lvis_Colored.__getitem__` does no subsampling: it loads a pre-made `.npy`
holding exactly 10,000 points. `farthest_point_sample` appears only in the **ShapeNet**
path (`dataset_3d.py:400`). `pointclouds.py`'s "What this deliberately does not do" section
states exactly this and is correct.

Consequence: **upstream ships pre-sampled clouds and this repository contains no mesh-sampling
procedure at all.** Our area-weighted surface sampler has no upstream counterpart to match.
It stays an `IMPLEMENTATION CHOICE` — but see FIND-6, which measures whether it works.

## FIND-4 — RGB scale is `[0, 1]`. **`UPSTREAM FACT` — the U-02 open question is closed.**

`pointclouds.py:14` recorded this as *"STRONGLY INDICATED, not proven"* and said settling it
"needs one official .npy read directly". Measured over **300 official files**:

```
rgb global min   0.00000000
rgb global max   1.00000000
files > 1.0      0 / 300
files < 0.0      0 / 300
dtype float16, shape (10000, 3)
```

Per-uid distributions also agree closely with ours (e.g. `06f26c0afd10` ours
`[0.004, 0.404] mean 0.022` vs official `[0.004, 0.402] mean 0.023`).

**Our `[0, 1]` choice is correct.** The factor-of-255 risk to all 46,052 clouds does not exist.

The `0.4` grey evidence quoted in the docstring is verified at `dataset_3d.py:291,297`, and the
docstring is right that those lines sit in the ModelNet path — but the direct measurement now
supersedes that inference.

## FIND-5 — Released clouds are radius-normalised but **not** centred. `UPSTREAM FACT`

Over 300 official files: `max radius` 0.9997–1.0003, but `centroid offset` median **0.104**,
max 0.588. The dataloader's `pc_norm` centres them at load time. We centre at write time.
**Equivalent at the point the model sees the data.** No action.

## FIND-6 — `L2-PC-ULIP-REF`, run for the first time. `OBSERVED DATA`

The registry note on `G2_pc_sanity` states this diagnostic "has never been run and the
reference clouds are not on disk". **The second half is false** — they have been on disk since
2026-08-15. 286 uids overlap our corpus in shard `000-009`.

Chamfer distance, both clouds normalised (scale: 0.01 = 1% of object size):

```
                 as-is      +180° yaw about Y
median          0.0903          0.0230
95th pct        0.3483          0.0385
worst           1.2829          0.3781

180° better on:            269 / 286   (94.1%)
180° better by ≥2×:        200 / 286   (69.9%)
assets with chamfer > 0.1:  137  →  7  (47.9% → 2.4%)
```

A continuous yaw sweep (5° steps) lands **14 of 18** spot-checked assets at exactly 180°; the
four exceptions improve by only 1.0–1.1×, i.e. they are rotationally near-symmetric and the
angle is unconstrained for them.

**Our point clouds are systematically rotated 180° about Y relative to ULIP-2's released
clouds.** `(x, y, z) → (−x, y, −z)`, determinant +1 — a rotation, not a reflection. Consistent
with a glTF `−Z forward` versus `+Z forward` convention difference.

**Our renders share the frame.** `pointclouds.py:118` and `renders.py:150` both use
`trimesh.load(path, force="scene", process=False)` and then apply the same
`scene.graph.nodes_geometry` transforms. The two modalities agree with each other and with the
GLB as authored.

## FIND-7 — The 180° yaw does **not** move the ULIP-2 embedding. `OBSERVED DATA`

Encoded through the **frozen** ULIP-2 point encoder and scored against the official
`image_feat` shipped inside the same `.npy`. 200 assets, mismatched pairs as the control:

```
                matched cos  mismatched cos    gap      R@1     R@5   median rank
ours 0°            0.4513        0.1283      0.3230   98.0%   99.5%      1.0
ours 180°          0.4512        0.1284      0.3228   97.5%   99.5%      1.0
ULIP's own cloud   0.4587        0.1259      0.3328   99.0%  100.0%      1.0

random-chance R@1 over a 200-asset pool = 0.5%
```

The measurement is alive: matched 0.451 against mismatched 0.128, R@1 98% against 0.5%.

- **Yaw makes no difference:** 98.0% vs 97.5%, cosine identical to 4 decimals.
- **Our clouds work:** 98.0% against ULIP's own 99.0% — 2 assets out of 200.

**Conclusion: the point-cloud corpus does not need regenerating.**

Residual note — the yaw still matters for **`n16_compose_scenes`**, which places retrieved
assets into a scene with real geometry. A 180°-yawed asset is placed backwards. That is a
downstream concern, not a corpus defect.

## FIND-8 — There is no upstream rendering procedure to compare against. `UPSTREAM FACT`

The ULIP repository contains **no rendering code**. `README.md:36-37` ships the images as a
dataset (`only_rgb_depth_images`, ~420 GB subset, ~1 TB full). What the code does reveal:

- ULIP-1 / ShapeNet: `picked_rotation_degrees = list(range(0, 360, 12))` (`dataset_3d.py:326`)
  — **30 views, 12° azimuth apart, single-axis orbit**, filenames `_r_000` … `_r_348`.
- ULIP-2 / Objaverse: `image_feat` has shape **`(12, 1280)`** — **12 views**, i.e. 30° apart.
  This corroborates the ULIP-2 paper's "spaced equally by 30 degrees" that `renders.py:89`
  cites, and confirms a single-axis orbit rather than a sphere lattice.
- Image transform (`main.py:176-181`): `RandomResizedCrop(224, scale=(0.5,1.0))` +
  **ImageNet** mean/std, *not* OpenCLIP's. Measured in FIND-9; not material.

Our layout, `ulip2_azimuth_orbit_11` with 11 azimuths at 360/11 ≈ 32.7° and elevation 20°, is
therefore an `IMPLEMENTATION CHOICE` informed by upstream, exactly as
`camera_layout_source: upstream_informed_choice` in every sidecar already says.

**U-03 (camera placement) and U-03a (projection) remain UNKNOWN.** Upstream cannot resolve them
because upstream ships pixels, not code. The elevation of 20° is ours and is not derivable.

## FIND-9 — Our renders land on ULIP-2's own renders. `OBSERVED DATA`

Our 11 views per asset, encoded through our own pipeline, scored against the official 12-view
`image_feat`. 200 assets, mismatched pairs as control:

```
preprocessing            matched  mismatched    gap      R@1     R@5   median rank
ours (open_clip norm)     0.8369     0.5561   0.2809   83.5%   94.0%      1.0
ULIP's (ImageNet norm)    0.8336     0.5613   0.2723   82.5%   94.5%      1.0

random-chance R@1 = 0.5%
```

- **Our renders work:** R@1 83.5%, median rank 1, against 0.5% chance.
- **The normalisation difference is not material:** 83.5% vs 82.5%, within noise. The
  ImageNet-vs-OpenCLIP concern raised by FIND-8 is closed.
- The gap to the point-cloud tower's 98% is expected: different renderer, lighting, background
  and elevation. Median rank 1 says the renders are unambiguously the same object.

## FIND-10 — Render corpus provenance is internally consistent. `OBSERVED DATA`

5,000 sidecars sampled, all identical on every provenance field:

```
camera_layout        ulip2_azimuth_orbit_11
orbit_elevation_deg  20.0
projection           orthographic
resolution           224
renderer_version     2
camera_layout_source upstream_informed_choice
projection_source    implementation_choice
n_views_source       paper
```

No asset was rendered under the older `fibonacci` layout (`renderer_version: 1`). Every sidecar
labels the authority of each choice correctly.

`view_paths` stores `/home/kyzen/MetaFindV1/data/...` — the symlink side — so the corpus
survived the 2026-08-21 move to `/mnt/data1` without rewriting.

## FIND-11 — Upstream reads `value_to_key_mapping` as the label. `UPSTREAM FACT`

`Objaverse_Lvis_Colored.__getitem__`, `dataset_3d.py:539-540`:

```python
name  = self.lvis_metadata["value_to_key_mapping"][sample]
label = self.lvis_metadata["key_to_id"][name]
```

That is the file `metafind/data/download.py:70` fetches and **nothing in this repository reads**
(`workflow/MIF_n05_diagnosis.md`, Evidence 4).

The n05 remediation was previously argued from the prior-collapse statistics alone. This is
stronger and independent: **the official implementation treats the LVIS category as the
ground-truth label for these assets.** It strengthens `D14` and should be cited there.

## FIND-12 — ESSGNN follows the Appendix, and the Appendix requires the scalar `f_x`. `PAPER FACT`

`2methdology.tex:54` states `f_x: R^(2d+1+e) → R^3` and `d_ij = ‖x_i − x_j‖_2`.
`appendix.tex:32,50,65` uses a shared message `m_ij = φ_e(h_i^l, h_j^l, ‖x_i^l − x_j^l‖², e_ij)`,
then `x_i^{l+1} = x_i^l + Σ (x_i^l − x_j^l) · φ_x(m_ij)` and `h_i^{l+1} = h_i^l + Σ φ_h(m_ij)`.

The main text and the appendix disagree on three points, and the recorded protocol
(`essgnn_arch_protocol.json`, decided by Kyzen 2026-08-19) follows the appendix on all three —
`distance: squared`, `coord_feat: current`, `architecture_family: appendix_shared_msg`. That is
internally coherent.

**§2.5's `R^3` is a typo in the paper, not a choice we made.** Two independent reasons:

1. Dimensional. If `f_x → R^3`, then `(x_i − x_j) · f_x` is a dot product yielding a scalar, and
   `x_i^{l+1} = x_i^l + scalar` is not well-formed. Only a scalar `f_x` makes Eq. 3 typecheck.
2. The appendix's own equivariance proof. `appendix.tex:56-57` factors `Q` out of
   `Σ (Qx_i + g − Qx_j − g) · φ_x(m_ij)` to reach `Q(Σ (x_i − x_j) · φ_x(m_ij))`. **That
   factoring holds only if `φ_x` is a scalar.**

`essgnn.py:311` implements the scalar. Correct, and `D0-009` already adjudicated it.

## FIND-13 — τ = 0.5 is implemented. My own earlier report was stale. `OBSERVED IMPLEMENTATION`

`3experiments.tex:15` — "The temperature is 0.5 for all experiments" — is a `PAPER FACT`.

`workflow/MASTER.md:273` still says "There is currently no supported way to produce τ = 0.5
through n05b". **That is out of date.** `resolve_stage1.py:272-273` now defaults to
`init_temperature: 0.5, learnable_temperature: False`; `stage1_hyperparameters.json` records
0.5 / false; `stage1.py:335-336` reads those values. `D2a` fixed this on 2026-08-21.

The `tau deviates from the paper` warning visible in `pytest` output comes from **tests**
constructing `ContrastiveConfig` with library defaults (0.07 / learnable), not from the
training path. `MASTER.md` needs correcting.

Also verified: Stage 1's loss is query→gallery only (`2methdology.tex:76-78`) and Stage 2's is
bidirectional (`:93+`); `tests/test_train_stage1.py` asserts exactly this.

---

## Status after this pass

| item | node | verdict |
|---|---|---|
| 04 | `n03_sample_pointclouds` | **BEHAVIOR-VERIFIED against upstream.** R@1 98.0% vs ULIP's own 99.0% |
| 06 | `n04_render_views` | **BEHAVIOR-VERIFIED against upstream artifacts.** R@1 83.5%, median rank 1 |

## Open, and deliberately not resolved here

1. **U-03 / U-03a** — camera placement and projection. Upstream ships pixels, not code; these
   cannot be closed from the sources available. The 20° elevation is ours.
2. **The 180° yaw's effect on `n16_compose_scenes`.** Embeddings are unaffected; asset
   *placement* is not. Needs a decision before scene composition is implemented.
3. **`G2_pc_sanity`'s `self_retrieval_rank` criterion** — the machinery now exists
   (FIND-7 is that measurement), so `D16`'s OQ-2 cost estimate is answerable cheaply.
4. **Two documentation corrections required**, both factual:
   - `node_registry.yaml`, `G2_pc_sanity` note: "the reference clouds are not on disk" — false.
   - `workflow/MASTER.md:273`: the τ = 0.5 claim — superseded by `D2a`.
5. **`pointclouds.py:14`'s U-02 rgb caveat** is now answered by FIND-4 and could be updated —
   left alone here because this pass modified no code.

## Reusable artifacts

- `scratchpad/ulip_ref/000-009/` — 4,999 official clouds, extracted, 1.4 GB
- `scratchpad/orient_test2.py` — point-tower retrieval harness (FIND-7)
- `scratchpad/render_test.py` — image-tower retrieval harness (FIND-9)
