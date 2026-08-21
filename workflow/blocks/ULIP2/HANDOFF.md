# HANDOFF — ULIP2

> **Binding USER rule:** everything goes through this file. In-conversation reporting does not count.
> Write here **before** calling Master, not after.
> Append newest at the top. Never delete an entry — mark it `RESOLVED` and say by whom.

## When to write here

- You need Master's permission, a contract change, or a ruling.
- You are stuck and it touches the other block or the Integrator.
- You found something that changes shared architecture, a dependency, or an accepted assumption
  (`MASTER-IMPACTING FINDING`) — report it, do not act on it.
- You finished a work item and it needs integration review.

## Entry format

```
### <date> · <FROM role> → <TO role> · <BLOCKING | INFO | FINDING>

FINDING     what is true, with evidence (file:line, paper section, measurement, population)
DECISION    what you propose — kept separate from the finding, never merged
EVIDENCE    how it was verified, and what remains unverified
IMPACT      which tasks, artifacts, stages
ASK         exactly what you need from the recipient
STATE       can this block safely continue meanwhile? yes / no, and why
```

Classify every claim: PAPER FACT · UPSTREAM FACT · OBSERVED IMPLEMENTATION · OBSERVED DATA ·
INFERENCE · IMPLEMENTATION CHOICE · DEVIATION · UNKNOWN. Never promote an inference to a fact.

---

### 2026-08-22 · ULIP2 ENGINEER → MASTER · **MASTER-IMPACTING FINDING**

**`n04`'s camera orbit uses `+Z` as the up axis. The meshes are `Y`-up. Every render is
rotated, and the orbit tumbles the asset end-over-end instead of circling it.**

Repo `adf3df2`. Read-only: no code, no data, no render was modified.

---

#### FINDING

`OBSERVED IMPLEMENTATION` — `metafind/data/renders.py`

- `azimuth_orbit_directions()` (`:84-102`) places the elevation on the **third** component:
  `[cos(el)·cos(az), cos(el)·sin(az), sin(el)]`. Printing the 11 directions: index 0 and
  index 1 vary; **index 2 is constant at `sin(20°) = +0.342`**. Index 2 is the orbit axis.
- `look_at()` (`:120`) defaults to `up=(0,0,1)` and the single call site
  `look_at(d * 3.0)` (`:277`) **does not override it**.
- The only transform applied to the scene is `fit` (`:180-184`) — an isotropic scale plus a
  translation. **There is no rotation and no Y→Z conversion anywhere in the file.**

`OBSERVED DATA` — the meshes are `Y`-up. Measured over the render corpus, using LVIS
categories as an **external** ground truth (not the implementation under test):

```
unambiguously TALL   (lamppost/bottle/candle/person/…)  n=1,195
    mean normalised extent   x 0.542   y 0.962   z 0.488
    longest edge on index 1: 1,060 / 1,195   (88.7%)

unambiguously FLAT   (rug/plate/pizza/place mat/…)      n=481
    mean normalised extent   x 0.944   y 0.304   z 0.824
    longest edge on index 1:    36 / 481    (7.5%)
```

This reproduces, independently, the Y-up convention already recorded in
`annotate_run.load_proportions()`'s docstring, and is consistent with
`evidence/n03_n04_upstream_verification.md` FIND-6, whose measured defect is a yaw **about Y**.

---

#### EVIDENCE — a falsifiable prediction, tested against pixels

Code reading alone does not establish runtime behaviour, so two mutually exclusive
hypotheses were predicted and tested against the rendered PNGs. For `view_00` (`az = 0`,
camera looking along `−X`):

- **H-A — camera up is `+Z` (the current code).** Image vertical is `+Z`, image horizontal
  is `+Y`, so the measured image bounding box ratio should be `extent_z / extent_y`.
- **H-B — camera up is `+Y` (a correct horizontal orbit).** The ratio should track
  `extent_y / extent_x`.

120 assets drawn with `numpy.default_rng(20260822)`, object bounding box measured from
non-background pixels (`L < 250`):

| hypothesis | log-log correlation | median multiplicative error | within ±20% |
|---|---|---|---|
| **H-A — `+Z` up (current code)** | **+0.893** | **1.12×** | **63%** |
| H-B — `+Y` up (correct orbit) | **−0.671** | 2.64× | 15% |

**H-A is confirmed. H-B is refuted** — its correlation is negative.

Illustration. Asset `03b315a8…`, LVIS `lamppost`, mesh aspect **7.2× along Y**. Its eleven
views have image height/width ratios:

```
0.14  0.34  0.82  1.63  0.50  0.22  0.22  0.50  1.63  0.82  0.33
```

A 7.2×-tall object under a correct orbit renders ≈7× tall in **every** view. `0.14` is
`1/7.2` — the object lying on its side. The sequence is a tumble, not an orbit.

Across 12 tall assets × 11 views, the median image height/width ratio is **0.73**:
**upright objects render wider than tall more often than not.**

---

#### What this does **NOT** overturn

**`FIND-9` stands.** Our renders scored R@1 83.5%, median rank 1, against ULIP-2's official
`image_feat` (200 assets, mismatched pairs as control, chance 0.5%). Eleven views cover the
object, and CLIP still identifies it when it is sideways. **The renders are not useless, and
this finding must not be reported as "the render corpus is invalid".**

#### What this **does** put in question

1. **Sidecar provenance is factually wrong for all 45,955 assets.**
   `camera_layout: "ulip2_azimuth_orbit_11"` claims a ULIP-2-style azimuth orbit. ULIP-2's
   documented method is "12 images per shape, spaced equally by 30 degrees" — an orbit about
   the asset's **vertical** axis. Ours is about a horizontal axis. The field does not
   describe what was produced. `renders.py`'s own header makes the same claim.

2. **A settled item now has new evidence against it.**
   `BLOCK.md` §7 and `evidence/n05_annotation_defect.md` Evidence 2 record that framing does
   not drive annotation quality — `correlation(best-view occupancy, LVIS agreement) = +0.054`
   over a ~100× range of effective object pixels. **That measurement is about occupancy — how
   much of the frame the object fills. It is not about orientation — whether the object is
   upright.** Orientation has never been measured. Per `BLOCKS.md`, new evidence against a
   settled decision is reported, not acted on.

---

#### IMPACT

| | |
|---|---|
| `n04` | 45,955 sidecars carry a `camera_layout` value that does not describe the output |
| `n05` v5 | The prompt asks for `width_axis` — *"which of the two horizontal axes reads as left-to-right WIDTH in these views"*. On a tumbled render this question has no well-defined answer, so the response is noise |
| `annotate.derive_dimensions()` | Consumes `width_axis` to assign `width` vs `length` |
| `DL-001` | Those two numbers are serialized into the ratified Stage 1 text template — `roughly {W} by {L} by {H} centimetres` |
| Stage 1 | Therefore every text embedding |
| **Table 1** | Therefore every text-conditioned column |
| `identity_confirmed` | A model shown a lamppost on its side is more likely to answer `false`, which would be read as an LVIS labelling error rather than a rendering artefact |
| `n16` (ESSGNN) | Not assessed here — out of this block's scope. Flagged only because `Q-YAW-PLACEMENT` already concerns asset orientation at composition time |

---

#### DECISION — proposed, kept separate from the finding

**None proposed. This block does not decide it.**

Re-rendering is prohibited by `BLOCK.md` §7, and overturning that is a USER decision.
The engineer's position is that the finding be adjudicated **before** the M1 bake-off runs
— see STATE. No remedy is recommended here, and no re-render was performed or scheduled.

#### ASK

1. Master to route the sidecar-provenance falsity and the challenge to the "framing is not
   the cause" evidence. Both are `MASTER-IMPACTING`.
2. A ruling on whether `n05` may be run at all while `width_axis` is under question.

#### STATE

**Can this block safely continue? — Partially, and M1 is paused by USER decision.**

- **USER decision, 2026-08-22, in conversation: M1 (the annotator bake-off) is PAUSED**
  pending adjudication of this finding. Recorded here because conversation is not storage.
- The engineer's reason, accepted by the USER: two of the bake-off's quality criteria
  (`width_axis` agreement and `identity_confirmed`) are contaminated by this finding. All
  three arms would return noise on them, and a **bug would be read as "this criterion has no
  discriminating power"**.
- **Safe to continue meanwhile:** W-3 (rebuild `annotation_provenance.json`), W-8 (stale
  `24 GB` comments), and the `G1` gate work below. None depends on render orientation.

---

#### Not verified — stated so it is not assumed

- **Whether upright renders would actually improve annotation quality has NOT been measured.**
  Establishing it needs a small A/B re-render, which is not authorised.
- The effect on the *image* tower is not re-measured here; `FIND-9` is the only evidence and
  it was taken on the current renders.
- No claim is made about which orbit MetaFind used. `U-03` (camera placement) and `U-03a`
  (projection) remain `UNKNOWN`; upstream ships pixels, not rendering code (`FIND-8`).

---

### 2026-08-22 · ULIP2 ENGINEER → MASTER · INFO — grill decisions taken in conversation

Recorded because in-conversation reporting does not count. **These are USER decisions; the
ledger is Master's to write.**

| # | Item | USER decision |
|---|---|---|
| 1 | **`G1_sources_valid` has never run.** `01_GRAPH_SPEC.md:410` requires: manifest complete · GLB coverage ≥98% · ULIP-2 checkpoint **behaviourally** verified ("not sha256 alone") · ProcTHOR's three splits present | **Run it now**, and define with it the gate-record schema the five later gates inherit |
| 2 | Does running `G1` now count as **evaluation** or **backfill**? (`BLOCK.md` M3c's open question) | **Evaluation** — it may FAIL, and a FAIL stops work. *(Taken from the engineer's recommendation; the USER answered "A" without addressing the sub-question, and was told a one-word veto would change it.)* |
| 3 | **M1 bake-off** vs the `n04` finding above | **M1 paused** until the finding is adjudicated |

**Engineer's supporting measurements for item 1** — `OBSERVED DATA`, repo `adf3df2`:

```
manifest uids                             46,052
GLBs on disk (>1 KiB)                     46,052      coverage 100.00%   (G1 needs >=98%)
manifest \ GLB                                 0
GLB \ manifest                                 0
glb_failures.json                         absent
ProcTHOR train/val/test.jsonl             all present
n03 produced a point cloud for            46,052 / 46,052 GLBs
```

The last line is the strongest available integrity evidence: **every GLB actually parsed.**
It is a by-product of `n03`, not a deliberate check, and `G1` does not currently claim it.

**Two gaps `G1` will have to close:**

- **`U-01`'s own recorded resolution is only half done.** `graph_spec.yaml:1583` reads
  `resolution: "use len(manifest); record its sha256"`. `len(manifest)` is honoured
  throughout. **The sha256 has never been recorded anywhere in the repository.** Measured now:

  ```
  lvis.json                      ba1bb191e1d98252e19e59aa18185ea46b83d9955b28ba895133536974e8bf9f
  objaverse_lvis_metadata.json   38e66d5b6cf38f19b1bd174943caa14c2f8952006d943e2f99ba0826b82d095e
  ```

- **The ULIP-2 checkpoint has no recorded behavioural verification.** The engineer's reading
  is that `evidence/n03_n04_upstream_verification.md` FIND-7 already satisfies it — our clouds
  through the frozen ULIP-2 encoder reach R@1 98.0% against official `image_feat`, versus
  0.5% chance. `01_GRAPH_SPEC.md:1282` states the check exists to catch a checkpoint that
  "loads and emits plausible-looking garbage"; a checkpoint yielding 98% retrieval is not
  that. **This is the engineer's INFERENCE and needs Master's ruling, not a self-award.**

---

#### `n02` — reviewed, with its limit stated

Acquisition is complete and clean (table above). **"n02 is fine" is not claimed**: its gate
has never run, and two of the four things that gate checks have no record. `UPSTREAM FACT`
established this session: ULIP's own `Objaverse_Lvis_Colored.__init__`
(`/home/kyzen/upstream/ULIP` @ `95d480fe`, `data/dataset_3d.py:463-470`) reads the same two
files and takes `list(npy_file_map.keys())` as its instance list — identical to
`download.py:330`. **Our acquisition is faithful to ULIP-2's own usage.**

**That does not transfer to MetaFind.** In the ULIP repository `objaverse_lvis_colored`
appears **only** as `--validate_dataset_name` in `scripts/test_ulip2_pointbert_objaverse_lvis.sh`;
`--pretrain_dataset_name` defaults to `shapenet` (`main.py:38`). **ULIP evaluates on this set.
MetaFind trains on it** — `2methdology.tex:75` ("In the first stage, both query and gallery
encoders are trained on large-scale object-level data from Objaverse-LVIS") and
`3experiments.tex:24` ("Stage-1 pretraining on Objaverse-LVIS"), both `PAPER FACT`, with an
80/20 train/test split (`3experiments.tex:8`).

Consequence for `U-01`: the 46,052 vs "approximately 48,000" gap is **not** merely a different
evaluation set. Under the paper's split it is ~1,558 fewer training assets and ~390 fewer test
assets. `U-01` remains `UNKNOWN` and **unanswered by the USER**.

---

#### Open grill items, not yet decided by the USER

| | Item |
|---|---|
| **O-1** | A **circuit breaker** for the multi-day `n05` full run — stop automatically if the cumulative failure rate exceeds a threshold **measured by the bake-off**, rather than a number invented now. Engineer recommends yes. *Asked; not answered.* |
| **O-2** | `U-01` — resolve now, or carry it as a stated Table 1 limitation. Engineer recommends carrying it, because **the answer changes the reporting, not the work**: the paper's exact 48K set is unobtainable either way. *Asked; not answered.* |
| **O-3** | `n03` has not been reviewed in this grill. |

**Prior-art note for `O-1`, `OBSERVED DATA`:** the v3 full run lost **3** assets
(`n05_v3_full.log`: *"45,553 annotated this run, 45,955 complete on disk, 3 quarantined"*).
All 5 records in `quarantine_n05_annotate.jsonl` are `MODEL_RECOVERABLE` / `repair_budget`.
**v5 will lose at least 103 before any model speaks** — 5 assets whose up-axis extent is 0
(`derive_dimensions` raises) and 98 whose feasible height band under
`MIN_DIM_CM=0.1 / MAX_DIM_CM=10,000` is empty. That failure mode is created by v5's own
exact-proportions handover; v3 did not have it.

---

### 2026-08-22 · ULIP2 ENGINEER → MASTER · INFO — `n03` reviewed

Read-only. Repo `adf3df2`. Swept the **whole** `pointclouds_index.jsonl`, not a sample.

#### `n03` is in good shape. `OBSERVED DATA`, all 46,052 records

```
records                  46,052
n_points                 10000        uniform, no exceptions
sampler_version          3            uniform
rgb_scale                "unit"       uniform
coloured_point_fraction  1.000        for all 46,052
max_radius               1.000000     min == max
centroid_offset          <= 1.35e-05
seed                     46,052 distinct values, none missing
per_axis_variance == 0   21 assets, exactly one axis each
```

The 21 zero-variance assets **reproduce the figure recorded at
`validation_plan.yaml:113`** independently. That note establishes they are planes, decals and
2D cut-outs whose mesh bounding box is 0 to 1.8e-12 thick on that axis, with all 10,000 points
distinct — flat assets, not a flattening bug. Confirmed, not merely repeated.

Per-asset seeds being 46,052 distinct and complete means the sampling is individually
reproducible. `npz` payload is `xyz (10000,3) float32` + `rgb (10000,3) float32`.

#### `F-N03-1` — 8,853 assets (19.2%) carry glTF-default **white** as their colour

`OBSERVED IMPLEMENTATION` — `pointclouds.py:147` ranks colour sources worst-first
`("fallback_grey", "gltf_default", "flat", "vertex", "face", "texture")` and the sidecar
records **the worst source across an asset's parts**. `:161-163`: `gltf_default` means a
material carrying neither a `baseColorTexture` nor a `baseColorFactor`, for which glTF 2.0
defines the default `[1,1,1,1]` — **white**.

`OBSERVED DATA` — measured on the `.npz` files themselves, 150 assets sampled per class with
`numpy.default_rng(20260822)`:

| `colour_source` | assets | median fraction of pure-white points | median RGB std | assets ≥99.9% white |
|---|---|---|---|---|
| `texture` | 23,675 | 0.000 | 0.187 | 0 / 150 |
| `flat` | 13,524 | 0.000 | 0.200 | 2 / 150 |
| **`gltf_default`** | **8,853** | **0.698** | 0.127 | **66 / 150 (44%)** |

Extrapolated: roughly **3,900 assets (8.5% of the corpus) are entirely white point clouds**,
and the median `gltf_default` asset is ~70% white.

**Why it may matter.** ULIP-2's Objaverse path is `xyzrgb` with `use_color = True`
(`dataset_3d.py:456-505`, `UPSTREAM FACT` recorded in `FIND-2`). The colour channel is
consumed by the point tower. For ~19% of the corpus that channel carries a spec default
rather than the asset's appearance.

**Why this is NOT yet called a defect.** The code follows the glTF 2.0 specification
correctly: those assets genuinely carry no colour in the file. The open question is whether
ULIP-2's **released** cloud for the same uid is also white. If it is, our corpus matches
upstream and this is a property of Objaverse. If ULIP's is coloured, we have a bug.

**`FIND-4` does not settle it.** It measured 300 official files and reported that per-uid
distributions "agree closely with ours", but it did not stratify by `colour_source`, so the
`gltf_default` class was never isolated. The comparison is available — the official clouds
are on disk at `data/models/hf-cache/datasets--SFXX--ULIP/blobs/` — and is **proposed, not
performed** (open item `O-4`).

`STATE` — no action taken, nothing modified. This does not block M1 and does not block the
`n04` finding above.

---

### 2026-08-22 · USER DIRECTIVES taken in conversation · **BINDING**

Recorded verbatim because conversation is not storage. **The USER is the final authority and
stated that this thread's decisions carry it.** Master must ledger these; the block does not.

| # | USER decision | Wording / basis |
|---|---|---|
| **U-A** | **No post-processing repairs.** A defect is fixed at its source, once, properly. The codebase is meant to be used by other people and must be written to that standard | 「任何問題 我不希望透過後處理來修改 要修就一次修好 後續這個程式是需要被人拿來用的 你必須寫好來」 |
| **U-B** | **`n04` re-render is AUTHORISED** — the `+Z`-up camera is fixed in the source, then the corpus is regenerated. A 100-asset A/B runs **first** as the pre-flight; its numbers reach the USER before any full run | answered `A` to "重渲 46,045 個資產，我拿到授權了嗎" |
| **U-C** | **O-5 approved** — a 100-asset A/B render, corrected axis vs current axis, written to an isolated directory. It does **not** touch `data/outputs/renders/` | answered `A` |
| **U-D** | **O-6: Master is not called yet.** The `n04` finding waits for the A/B numbers, then goes to Master with evidence rather than as a bare claim | answered `B` |

#### ⚠️ What `U-A` and `U-B` overturn — flagged, not acted on beyond the authorisation

`BLOCK.md` §7 states: *"Point clouds and renders are read-only here. **No re-render** — framing
was measured not to drive annotation agreement."*

**`U-B` overrides that prohibition for `n04`.** The engineer's position, on record: the §7
prohibition rests on `evidence/n05_annotation_defect.md` Evidence 2, which measured
**occupancy** (`correlation = +0.054`). **Orientation was never measured.** The prohibition was
not wrong when written; it was answering a different question.

**`U-A` also forecloses the cheaper remedy.** Correcting the `camera_layout` sidecar text to
describe the tumbling orbit honestly is exactly the "post-processing repair" `U-A` rejects, and
it would additionally record a defect as if it were a design choice. It is off the table.

**Master must ledger `U-A` as a standing project rule**, not only as an `n04` decision.

`STATE` — nothing has been rendered, regenerated, or modified under these authorisations. No
GPU job has run. Implementation waits on the SPEC and on the remaining open items below.

---

### 2026-08-22 · ULIP2 ENGINEER → MASTER · **FINDING CONFIRMED AGAINST THE UPSTREAM ARTIFACT**

**ULIP-2's own Objaverse renders were located, downloaded and compared. The `n04` up-axis
finding is confirmed visually and numerically, and two further differences were measured.**

#### How the primary source was reached — `FIND-8` was too pessimistic

`evidence/n03_n04_upstream_verification.md` FIND-8 concluded *"There is no upstream rendering
procedure to compare against… ULIP ships pixels, not code"*. **The first half is correct and
the second half made us stop too early.** Every file in `/home/kyzen/upstream/ULIP` @ `95d480fe`
was enumerated (≈350 files): there is no render script. **But the pixels themselves are
published**, and pixels are enough to measure a camera.

```
SFXX/ulip  →  ULIP_Objaverse_Triplets/render_images_resized_224/
              objaverse_rgb_chunk_0000 … 0191      474 GB total, 2.47 GB per chunk
```

Chunk 0000 downloaded: 50,000 PNGs, 4,167 assets × 12 views, already 224×224 — **the same
resolution we produce.** 217 of those uids overlap our corpus.

#### Measured, on assets present in BOTH corpora

`OBSERVED DATA`. Foreground segmented by luminance against each image's own background.

**`9f1335d8…` — LVIS `vase`, rotationally symmetric, mesh `x 3.639 · y 7.480 · z 3.639`**

| | image height/width, median | range over the views | background | longest side ÷ 224 | frame occupancy |
|---|---|---|---|---|---|
| **ULIP official (12)** | **1.94** | **1.94 – 1.94** | **0** (black) | **0.61** | 12.1% |
| ours (11) | 0.59 | 0.47 – 1.15 | 255 (white) | 0.91 | 21.2% |

**`75fdc43a…` — LVIS `person`, asymmetric, mesh `x 60.56 · y 74.49 · z 19.50`**

| | median | range | background | longest ÷ 224 | occupancy |
|---|---|---|---|---|---|
| **ULIP official (12)** | 1.54 | 1.14 – 2.98 | 0 | 0.60 | 10.9% |
| ours (11) | 0.53 | 0.40 – 0.60 | 255 | 0.75 | 12.1% |

#### Three separate differences, each independently established

1. **Orbit axis — the confirmed bug.** ULIP's vase holds `h/w = 1.94` across **all twelve
   views, to two decimals**. An invariant silhouette under rotation is the signature of a
   rotationally symmetric object orbited **about its own vertical axis**. The asymmetric
   `person` behaves as that model predicts instead — `1.14` broadside, `2.98` edge-on. Ours
   sits at `0.59` / `0.53` and swings, i.e. **the object is lying down and tumbling.** This
   corroborates the `+Z`-up finding from an entirely independent direction: upstream pixels
   rather than our own code.

2. **Background. ULIP renders on BLACK (0); we render on WHITE (255).** Not previously
   recorded anywhere in this project. It is consumed directly by the CLIP image tower.

3. **Framing. ULIP fits the object to ≈0.60 of the frame, both assets, tight agreement.**
   Ours reaches 0.75–0.91. A constant 0.60 across two very different shapes indicates a fixed
   fit rule upstream, not incidental scaling.

#### The elevation becomes solvable rather than guessable

The vase is rotationally symmetric with a mesh aspect of `7.480 / 3.639 = 2.056`. ULIP images
it at `1.94`, **identically in every view**. That figure is a function of the elevation alone.
Rendering the same mesh through a corrected camera across a sweep of elevations and matching
`1.94` **solves** for ULIP's elevation. `U-03` stops being permanently `UNKNOWN` for the
ULIP-conformance question — though it stays `UNKNOWN` for what *MetaFind* used.

#### An independent verification target already exists

`FIND-9` measured our current renders against ULIP's official `image_feat`: **R@1 83.5%**.
`FIND-7` measured our point clouds against the same target: **R@1 98.0%**.

**The 14.5-point gap is the hypothesis.** Re-running FIND-9's harness after the three fixes is
a falsifiable check whose expected-truth source is an **official upstream artifact**, not our
own code. If R@1 climbs toward 98%, the fixes are confirmed; if it does not, the hypothesis is
wrong and must be reported as such.

---

### 2026-08-22 · USER DIRECTIVES, second batch · **BINDING**

| # | USER decision |
|---|---|
| **U-E** | **11 views stands.** The engineer objected once to a proposed change to 12 — MetaFind states 11 twice (`2methdology.tex:28`, `neurips_2025.tex:100`) and it is one of the few hard numbers the paper gives for this node. Objection accepted; no deviation is incurred |
| **U-F** | Orbit layout is the **single horizontal ring**, not the Fibonacci sphere. Chosen because ULIP-2's *"spaced equally by 360/12 degrees"* (`ulip2_source/main.tex:677`) describes a circle and therefore has provenance; the sphere was the engineer's preference and had none |
| **U-G** | The **180° yaw is fixed in the same pass**, at the **mesh-load layer**, so point clouds and renders both land in the corrected frame from one change. Requires `n03` (46,052) to re-run as well as `n04` |
| **U-H** | The white-point-cloud question (`F-N03-1`) is **settled by differential comparison against ULIP's official clouds BEFORE the re-run**, so that if it is a bug it is fixed in the same pass |
| **U-I** | **The bake-off sample is 100 assets per arm**, not the 300–500 the engineer proposed |
| **U-J** | Annotation **fields follow the paper's Figure 2** — all 13. The four provenance-source deviations (`DL-007`) stand |
| **U-K** | A **few-shot exemplar** is added, using the paper's own Figure 2 record. Text-only for now |
| **U-L** | **Two-turn identity check.** Turn 1 shows the views with **no** anchor and asks for a blind identification; turn 2 reveals the LVIS label and requests the full record. Images are encoded once and the cache reused. `identity_confirmed` becomes **computed** from turn 1 rather than asked — which is a direct answer to `IC-1`, the rubber-stamp risk |
| **U-M** | **Multiple description candidates, re-ranked by an independent CLIP.** Adopted from ULIP-2's own method (`main.tex:677`: 10 BLIP-2 captions ranked by CLIP-ViT-Large, top-1 kept; its `top_k` ablation reports 69.7 / 66.7 / 66.4 / 66.3 for k = 1 / 3 / 5 / 10). **The ranking CLIP must NOT be `ViT-bigG-14`**, which n06 uses to encode — ranking and encoding with one model is circular. ULIP itself ranked with one model and trained with another |
| **U-N** | **All three render differences are fixed**: orbit axis, black background, ≈0.60 framing |
| **U-O** | **Standing rule, project-wide** — see below |

#### `U-O` — the source-precedence rule the USER adopted

```
MetaFind states it          →  follow MetaFind
MetaFind is silent          →  follow ULIP-2
```

**Basis, `PAPER FACT`:** MetaFind names ULIP-2 as its embedding backbone four times —
`2methdology.tex:14` (*"ULIP-2 embedding backbone"*), `:34` (*"leverage ULIP-2 to
independently encode available modalities"*), `neurips_2025.tex:90` (*"ULIP-2 backbone"*),
`:100` (*"MetaFind builds upon ULIP2"*).

**This does not make ULIP-2 a substitute for MetaFind.** It resolves silence, and only silence.
Where MetaFind speaks — 11 views, the Figure 2 field set, GPT-4o, τ = 0.5 — MetaFind governs and
ULIP-2 is irrelevant. **A choice made under `U-O` is an `IMPLEMENTATION CHOICE` with upstream
provenance. It is never a `PAPER FACT`, and it must never be written up as one.**

**Master is asked to ledger `U-O` as a standing rule**, since it will govern every future
"the paper does not say" decision in this project.

#### Still open — asked, not answered

| | Item |
|---|---|
| **O-1** | A circuit breaker for the multi-day `n05` run — stop automatically if the cumulative failure rate exceeds a threshold **measured by the bake-off**, never a number invented now. Engineer recommends yes. Asked three times; still unanswered |
| **O-2** | `U-01` — resolve the 46,052 vs "approximately 48,000" gap now, or carry it as a stated Table 1 limitation. Engineer recommends carrying it |

`STATE` — **still nothing implemented, rendered, or regenerated.** One 2.47 GB upstream chunk
was downloaded to the session scratchpad for measurement; no project artifact was touched.

---

### 2026-08-22 · USER DIRECTIVES, third batch · **BINDING** — `n05` protocol, `n05b`, `n06`, `n10`

| # | USER decision | Basis established this session |
|---|---|---|
| **U-P** | Bake-off sample is **4 strata × 25 = 100** per arm: ordinary · rare LVIS class · low visibility · extreme aspect. The 12-cell design was scaled down with the sample | 100 assets over 16 cells is ~6 per cell and measures nothing |
| **U-Q** | **20 assets hand-adjudicated by the USER**, once, shared across all three arms. **The only ground truth in the whole bake-off** | Every other quality signal traces back to LVIS, to the model, or to the code under test |
| **U-R** | Install `compressed-tensors`, `llmcompressor`, and a CLIP ViT-Large for description re-ranking | `compressed_tensors` is **absent**, so the "READY" `gemma-4-31B-it-qat-w4a16` arm cannot load today. On-disk ≠ loadable |
| **U-S** | **Widen the validator's dimension floor** so genuinely flat assets pass | 103 of 45,955 currently have an EMPTY feasible height band under `MIN_DIM_CM = 0.1`: 5 with a zero up-axis extent (`derive_dimensions` raises) and 98 whose derived width/length cannot land in range for **any** height. They are posters, decals and 2D cut-outs — real data, and a sub-millimetre thickness is correct for them. The floor is what is wrong |
| **U-T** | **Fusion default becomes `transformer`** | `fusion.py:22-29` records `U-13` as *"the paper never says which"*. **It does.** `3experiments.tex:143`: *"MLP and **the final selected Transformer** outperform others"*. A false `UNKNOWN` is corrected back to a `PAPER FACT`. `variant_registry.json`'s `full` variant also carries `"fusion": null` and must be filled |
| **U-U** | **Do not invent `lr`, `epochs`, `batch_size`. Measure them.** Carve a validation split (**72 / 8 / 20**, the paper's 20% test untouched), size the batch to what this GPU actually holds, sweep the learning rate, and stop on validation R@1 | The paper has **no implementation-details paragraph at all** — `3experiments.tex` §Experimental Setup contains only Datasets, Baselines and Metrics. Zero mentions of optimizer, learning rate, epochs, warmup, schedule or hardware. Every current value is invention |
| **U-V** | **`n06` stores all 11 per-view vectors alongside the mean** | GPU cost **zero** — all 11 are already encoded and 10 are discarded. Storage cost 45,955 × 11 × 1280 × 4 B ≈ **2.6 GB**. Recovering them later costs a full re-encode. `encode_text_image.py:25` (`U-14`) already recorded the trap |

#### `U-U` — what ULIP's own settings do and do not transfer

`/home/kyzen/upstream/ULIP` `main.py:47-60` and `scripts/pretrain_pointbert.sh`:

```
epochs 250 · warmup-epochs 1 · batch-size 64 per GPU × 8 GPUs · lr 3e-3
lr-start 1e-6 · lr-end 1e-5 · wd 0.1 · betas (0.9, 0.98)
```

Ours records `lr 1e-3 · epochs 50 · batch 64 · wd 0.1 · cosine`, and **does not record `betas` or
`warmup` at all** — those currently take PyTorch's defaults `(0.9, 0.999)` and none. **An
unrecorded optimizer parameter is still an experimental condition.**

**Transferable under `U-O`:** `betas`, `warmup`, `wd`, the shape of the schedule.
**NOT transferable:** `lr` and `epochs`. ULIP pretrains **from scratch** on ~800K shapes across
8 GPUs; we **fine-tune a released checkpoint** on ~36.8K assets on one. `3e-3` into a converged
checkpoint is a different experiment, not the same one.

#### `F-N10-1` — the contrastive negative count is an unavoidable DEVIATION

`OBSERVED IMPLEMENTATION`, `/home/kyzen/upstream/ULIP/models/losses.py:38-40`: ULIP's loss calls
`utils.all_gather_batch([pc_embed, text_embed, image_embed])` — **features are gathered across
all 8 GPUs before the contrastive matrix is formed.**

```
ULIP    in-batch negatives per step = 64 × 8 = 512
ours    in-batch negatives per step = batch_size on one GPU
```

**Gradient accumulation does not close this.** Accumulation sums gradients across micro-batches;
each micro-batch still forms its own contrastive matrix, so the negative count is unchanged. Only
a larger real batch — or an explicitly implemented gradient-cache — raises it, and 512 clouds of
10,000 points with backprop through PointBERT will not fit 32,607 MiB.

**Negative count is a first-order term in a contrastive objective, not a tuning detail.** It must
be registered as a DEVIATION with its measured value once the batch size is determined, and it is
a candidate explanation for any shortfall against the paper's Table 1.

#### `F-N09-1` — Table 1's own numbers give a free sanity check

`PAPER FACT`, `3experiments.tex` Table 1, MetaFind w/o ESSGNN:

```
Text only  13.8 / 23.1     Image only 11.7 / 19.2     PC only  75.1 / 78.0
T+I        17.2 / 21.8     T+PC       44.5 / 71.3     I+PC     45.8 / 73.1
T+I+PC     51.7 / 76.5
```

Every single-tower baseline scores **~98% on PC-only**; MetaFind scores **75.1**. The paper states
the reason at `:24` — the baselines retrieve "using **identical embeddings for both query and
gallery**, leading to **inflated accuracy**".

**Therefore: if our reproduction returns ~98% on PC-only, that is evidence the two towers are
not actually distinct**, regardless of what the config says. A number near ~75 is the regime the
paper describes. This is a cheap, paper-anchored check available the moment Table 1 runs, and it
should be written into the `n15` spec rather than discovered afterwards.

`U-09` (whether the gallery is the 20% test split or the full corpus) **cannot be settled from
these values** — 13.8% sits far above chance for either pool. `splits.py:104-125` already emits
both protocols, `A_test_gallery` and `B_full_gallery`. **Both must be reported, always labelled**,
and no claim of agreement with the paper may be made until the paper's own scope is known.

`STATE` — nothing implemented. `splits.py:90-101` verified to shuffle under an explicit seed before
cutting, so the partition is reproducible.

---

### 2026-08-22 · ULIP2 ENGINEER · **DELEGATED DECISIONS** — the remainder of the block

**USER, 2026-08-22:** 「算了 全部給你決定吧 記得遵守論文 多參考 ulip 論文 跟我講你剩下的決定」

Authority: **explicit USER delegation.** Constraint carried with it: follow the MetaFind paper;
consult the ULIP-2 paper where MetaFind is silent (`U-O`). **Every decision below names its
evidence class.** Where the delegation would require inventing research-critical information,
the item is left OPEN rather than decided — see the stop-safe list at the end.

| # | Decision | Class | Basis |
|---|---|---|---|
| **E-1** | **`n15` is designed and unit-tested NOW**, against synthetic inputs, before training | IMPLEMENTATION CHOICE | Needs no GPU and no checkpoint. `MASTER.md` §7: *"The longest pole is not on the GPU."* A protocol defect found after training costs a training run |
| **E-2** | Table 1 reports **both gallery protocols, always labelled** — `A_test_gallery` and `B_full_gallery` | PAPER-UNDERDETERMINED → recorded | `node_registry.yaml` `n15_eval_retrieval`: *"7 modality conditions x 2 gallery protocols"*. `splits.py:104-125` already emits both. `U-09` cannot be resolved from Table 1's values — 13.8% text-only sits far above chance for either pool. **No agreement with the paper may be claimed until the paper's own scope is known** |
| **E-3** | Metrics are **R@1 and R@5, instance-level** — the target is the query's own asset | PAPER FACT | `3experiments.tex:18` (*"top-k retrieval accuracy (R@1, R@5)"*) and `:24`, whose complaint that baselines use *"identical embeddings for both query and gallery, leading to inflated accuracy"* is only coherent if the retrieval target is the asset itself |
| **E-4** | **Seven query conditions**, exactly the paper's columns: Text · Image · PC · T+I · T+PC · I+PC · T+I+PC | PAPER FACT | Table 1's own column headers, `3experiments.tex:24` |
| **E-5** | **`n15` carries a hard sanity assertion: PC-only must NOT land near ~98%** | PAPER FACT (basis) | Every single-tower baseline scores ~98 on PC-only; MetaFind scores **75.1**, and `:24` gives the reason. **~98 from our dual tower is evidence the towers are not actually distinct**, whatever the config claims. Cheap, paper-anchored, and worth more as an assertion than as a post-hoc observation |
| **E-6** | **Gates G1–G4 are implemented to the criteria already written in `01_GRAPH_SPEC.md` §11. Nothing is invented and no threshold is loosened** | OBSERVED SPEC | The criteria, classes and `on_fail` behaviour are already specified. This is implementation, not design. G2's note is explicit: *"不得放寬門檻"* |
| **E-7** | **G2's comparison against ULIP's official clouds stays a DIAGNOSTIC, not a blocker — but its result is always reported** | follows the spec | `01_GRAPH_SPEC.md:141` already demoted it, correctly: MetaFind never claims to reuse ULIP's pre-sampled clouds, and Stage 1 fine-tunes the point encoder. **However**, now that the 180° yaw is being corrected, this diagnostic becomes genuinely informative and must not be silently skipped |
| **E-8** | **`n10b` becomes conditional, and certifies rather than assumes** | INFERENCE, recorded | `node_registry` gives `n10b` as *"the text/image embeddings … using the FINAL Stage 1 encoders"*. But `stage1_encoding_protocol.json` records `actual_clip_train_scope: frozen`, and `stage1.py`'s header states OpenCLIP stays frozen — **so the text and image encoders do not change during Stage 1 and `n06`'s vectors are already final.** Under `frozen`, a re-encode is pure waste; under `trainable` (deviation `D-1`) it is mandatory. `n10b` therefore **compares the checkpoint's CLIP weights against the pretrained ones and re-encodes only if they differ**, recording which branch it took. A silent no-op and a silent stale-embedding bug look identical otherwise |
| **E-9** | **`n11` → `G4` → `n12` implemented to the registry's preconditions verbatim** | OBSERVED SPEC | `n12` precondition: `G4 verdict == PASS with is_terminal=true`; postcondition: *"a second differing write is an error"*. `G4` criterion: dimensions, count, no NaN, **target similarity == max similarity and within the argmax tie set** |
| **E-10** | **Five description candidates per asset**, re-ranked, top-1 kept | IMPLEMENTATION CHOICE | ULIP-2 generates **10** and keeps CLIP-top-1 (`ulip2_source/main.tex:677`); its ablation compares *ensembling* k, not generating k, and top-1 wins (69.7 vs 66.7 / 66.4 / 66.3). Ten generations from a 27B model over 45,955 assets is not affordable here. **Five is a cost choice, not an upstream figure.** The winner's CLIP score and the spread across candidates are recorded, so whether more candidates would still be improving is **measured in the bake-off rather than assumed** |
| **E-11** | **Re-ranking model is `openai/clip-vit-large-patch14`** | UPSTREAM FACT | The exact family ULIP-2 ranked with — *"ranked using CLIP-VIT-Large"*. It is also **not** `ViT-bigG-14`, which `n06` encodes with, so ranking and encoding stay independent. ULIP likewise ranked with one model and trained with another |

#### Stop-safe — delegated, but deliberately NOT decided

The delegation does not extend to inventing research-critical information that no available source
determines. These stay open and are named rather than quietly filled:

| | Item | Why it is not mine to close |
|---|---|---|
| **`Q-TOWER`** | `tower_sharing` ∈ {`shared_backbone_separate_fusion`, `fully_shared`, `fully_separate`} | Master holds it under `DL-009`, and it **hard-blocks `n09` → `n10`**. `dual_tower.freeze_gallery()` already refuses `fully_shared`, so the paper's frozen-gallery requirement and a single shared module cannot both hold. This decides whether Stage 2 can freeze the gallery at all — it crosses into the other block |
| **`Q-BUILDMODEL`** | The Stage 1 trainer builds from raw protocol dicts and passes ONE fusion object to both towers, so `fully_separate` is unimplementable as written | Same owner, same seam |
| **`Q-CATEGORY`** | Assigned to the Reviewer as `W-5`/`W-6`, and it gates `M2` | Not the engineer's to self-award |
| **`U-09`** | Whether the paper's gallery is the 20% test split or the full corpus | **Not determinable from any available source.** Handled by reporting both (`E-2`), never by picking one |
| **`U-01`** | 46,052 vs "approximately 48,000" | USER decided to carry it as a stated Table 1 limitation |
| **`F-N10-1`** | The contrastive negative count, 512 upstream vs one GPU's batch here | An unavoidable DEVIATION. Its magnitude is **measured** once the batch size is sized to the card; it is not chosen |

`STATE` — **still nothing implemented.** These are decisions of record, to be carried into the
SPEC. No code, data, render or checkpoint has been created, modified or deleted in this session.

---

### 2026-08-22 · ULIP2 ENGINEER · **`F-N03-1` DIAGNOSED AND CLOSED** — root cause found, no residual

**`trimesh` discards the glTF `COLOR_0` vertex-colour attribute whenever a `PBRMaterial` is
present. Our point clouds therefore render those assets white. It accounts for 100% of the
discrepancy against ULIP's official clouds — nothing else is wrong.**

Read-only. Repo `adf3df2`. Method: `diagnosing-bugs` — reproduce, minimise, form a falsifiable
hypothesis, split the population by it.

#### The differential, and then the split that closes it

286 uids overlap ULIP's `ULIP-2/objaverse_lvis/000-009` shard. Restricted to the 62 our sidecars
label `gltf_default`, comparing the fraction of pure-white points in each cloud:

| | ULIP official | ours |
|---|---|---|
| median white-point fraction | **17.9%** | **95.9%** |
| RGB std | 0.2297 | 0.1123 |
| entirely white (>99%) | 20 / 62 | 31 / 62 |
| **only ours entirely white** | — | **11 / 62** |
| only ULIP entirely white | **0 / 62** | — |

**The hypothesis:** the discrepancy is exactly the assets carrying `COLOR_0`.
**The prediction:** split by it, and the `COLOR_0`-absent half must agree; the present half must not.

| population | n | ULIP median white | ours median white | ULIP all-white | ours all-white |
|---|---|---|---|---|---|
| `gltf_default` **with** `COLOR_0` | 12 | **0.0%** | **100.0%** | 1 / 12 | **12 / 12** |
| `gltf_default` **without** `COLOR_0` | 50 | **35.3%** | **35.1%** | **19 / 50** | **19 / 50** |

**Where the attribute is absent we match upstream to within 0.2 points and identically on the
all-white count, 19 / 50 both sides. Where it is present we are wrong on every single asset.**
The hypothesis has no residual: **`COLOR_0` explains the entire gap.**

#### Mechanism, `OBSERVED IMPLEMENTATION`

Worked example `b3f86ea0972c4358a764225be1ef069f`, whose ULIP caption reads
*"a 3d model of a shoe with a **green** sole"* and whose cloud is 100% white on our side.

Raw glTF JSON chunk, parsed directly from the GLB:

```
materials[0]  = {"doubleSided": true, "name": "Scene_-_Root",
                 "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.6}}
meshes[0].primitives[0].attributes = {"COLOR_0": 2, "NORMAL": 1, "POSITION": 0}
embedded PNG: 0 · embedded JPEG: 0 · extensionsUsed: none
```

**The colour is in `COLOR_0`, and it is the asset's only colour.** The material carries no
`baseColorTexture` and no `baseColorFactor`, so `pointclouds._colourise()` correctly classifies it
`gltf_default` and applies glTF 2.0's `[1,1,1,1]` — white.

`trimesh.load(path, force="scene", process=False)` yields `TextureVisuals` with
`vertex_attributes == {}`. **`process=True` behaves identically.** The attribute never reaches us.

**`_colourise`'s source order at `pointclouds.py:147` already lists `vertex`** — that branch is
simply unreachable, because trimesh never surfaces the data. The classification logic is right;
its input is incomplete.

#### Scale, `OBSERVED DATA` — 200 assets sampled per class, `default_rng(20260822)`

| `colour_source` | carry `COLOR_0` | corpus size | affected, estimated |
|---|---|---|---|
| `gltf_default` | **17.0%** | 8,853 | **≈ 1,505 — white where real colour exists** |
| `flat` | 7.5% | 13,524 | ≈ 1,014 — material colour used where vertex colour exists |
| `texture` | 2.0% | 23,675 | ≈ 474 — same |

The 17.0% measured over the corpus and the 11 / 62 = **17.7%** measured against ULIP agree, from
two independent samples.

**Severity.** ULIP-2's Objaverse path is `xyzrgb` with `use_color = True`
(`dataset_3d.py:456-505`), so this channel is consumed by the point tower. For ~1,505 assets we
feed a specification default where upstream feeds the object's own colours.

#### Remedy — in scope, not yet implemented

Read `COLOR_0` from the glTF accessor directly rather than waiting for trimesh to expose it, and
place it in `_colourise`'s existing `vertex` slot so the ordering already written at
`pointclouds.py:147` takes effect. **Lands in the same `n03` regeneration as the 180° yaw
correction** — `U-A`, fix once at the source.

**Verification available:** re-run this exact differential after the fix. The `COLOR_0`-present
group must move to ULIP's numbers, and **the `COLOR_0`-absent group must not move at all** — it
already agrees, and any movement there would mean the fix broke something that was correct.

`STATE` — nothing modified. `F-N03-1` moves from OPEN to **DIAGNOSED**; it closes when the
regeneration reproduces ULIP's numbers.

---

### 2026-08-22 · ULIP2 ENGINEER · **`S-5` MET** — the render correction verified against upstream

**Against ULIP-2's released `image_feat`, over the 286 assets present in both corpora: R@1 rises
from 83.2% to 95.8%. The corrected renders land beside the point tower's 98.0%.**

Tool: `tools/verify_renders_against_ulip.py` (new). Repo `2a8ded4` plus the renderer changes.

#### The measurement

Same 286 assets, same frozen ViT-bigG-14, same aggregation. The only difference is which pixels
the encoder is shown: **the v2 renders still on disk**, or the same assets re-rendered under v3.
The target is ULIP's own `image_feat`, which we did not produce and cannot tune.

| | R@1 | R@5 | median rank | matched cos | mismatched cos | gap |
|---|---|---|---|---|---|---|
| **v2 — on-disk renders** | **83.2%** | 92.7% | 1 | 0.8371 | 0.5565 | 0.2806 |
| **v3 — corrected** | **95.8%** | 98.6% | 1 | 0.8782 | 0.5378 | **0.3404** |

Chance R@1 over this pool is **0.35%**.

#### Three things this establishes

**1. The harness reproduces `FIND-9` independently.** FIND-9 measured **83.5%** over 200 assets
with a script that no longer exists. This tool, written from scratch against a different sample of
286, measures **83.2%** for the same corpus. The agreement is 0.3 points.

**2. The stated hypothesis is confirmed.** It was recorded before the fix: *"the 14.5-point gap
between the render tower's 83.5% and the point tower's 98.0% is what the renderer-v2 defects would
be expected to cost."* Correcting the up axis, the background and the framing **recovers 12.6 of
those 14.5 points.** The prediction was made first and the measurement was not free to land
anywhere.

**3. The signal sharpened, it did not merely shift.** Matched cosine rose 0.8371 → 0.8782 while
mismatched **fell** 0.5565 → 0.5378. A change that only brightened or rescaled the images would
move both together.

#### Correction of record — an alarm the engineer raised and then disproved

Mid-investigation this block reported that the image tower's weights were **random** and that the
measurement was void. **That was wrong, and it is recorded rather than quietly dropped.**

- The trigger was real: `open_clip` logs *"No pretrained weights loaded for model 'ViT-bigG-14'.
  Model initialized randomly"*, and the ULIP-2 checkpoint turns out to contain **228 tensors, all
  `point_encoder` / `pc_projection` / `logit_scale`, and zero CLIP tensors** — so the checkpoint is
  not where the image tower's weights come from.
- The warning comes from `ulip_backbone.py:195-196`, which constructs a **throwaway**
  `create_model_and_transforms("ViT-bigG-14", pretrained=None)` solely to obtain `self.preprocess`.
  The model actually used is built by the vendored path at `ULIP_models.py:354-355` with
  `pretrained='laion2b_s39b_b160k'`, and those weights are on disk (11 GB under
  `data/models/hf-cache/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k`).
- **Disproved independently of ULIP.** Five assets whose LVIS category is unambiguous, encoded
  against `"a 3d model of a {category}"`: matched cosine **0.400**, mismatched **0.298**, and the
  argmax lands on the correct text **5 / 5**. Random weights cannot do that.

**What actually misled the engineer** was a different test: encoding ULIP's *own* released PNGs
and comparing to their `image_feat` gave a diagonal (0.68, 0.76) *below* the off-diagonal
(0.70, 0.78). That is explained by `FIND-8`, already on record: ULIP's transform is
`RandomResizedCrop(224, scale=(0.5,1.0))` with **ImageNet** mean/std, not OpenCLIP's centre crop,
and the released images are a resized re-release. **Their features are not reproducible from their
published pixels with our transform, and that is a property of their pipeline, not of ours.**

The A/B above is unaffected: both arms use the same transform, so the comparison is internally
consistent, and `FIND-9` measured the transform difference as immaterial (83.5% OpenCLIP norm vs
82.5% ImageNet norm).

#### Still not established

- **The elevation is NOT solved.** R@1 over a 60-asset pool read 100.0 / 98.3 / 98.3 at 5° / 15° /
  25°, which is a 1-asset difference and cannot separate them. `ORBIT_ELEVATION_DEG` stays at 20°
  as an IMPLEMENTATION CHOICE, **not** as a solved value, and `U-03` stays `UNKNOWN`.
- **`U-03a` (projection) is untouched.** The silhouette sweep that appeared to favour orthographic
  over perspective by 0.89 to 0.57 **did not normalise the framing between them**, so it measured
  object size, not projection. That comparison is withdrawn.
- **95.8% is agreement with ULIP-2, not with MetaFind.** MetaFind never states a camera. Under
  `U-O` this is the best available target, and it remains an `IMPLEMENTATION CHOICE` with upstream
  provenance.

`STATE` — the corpus has **not** been regenerated. 286 assets were re-rendered in memory for this
measurement and their PNGs deleted; `data/outputs/renders/` still holds v2.
