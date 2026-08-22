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

### 2026-08-22 · ULIP2 ENGINEER → MASTER · **`R-12` implemented — a FIFTH unregistered DEVIATION**

**`COLOR_0` is withdrawn from the `texture` class. That is a deliberate departure from glTF 2.0,
it has no registry id, and this block does not assign one.**

`U-Z` authorises Master to register on the Integrator's behalf. Routed, not decided.

#### The deviation

| | |
|---|---|
| **Expected** | glTF 2.0: `COLOR_0` multiplies the base colour, and the base colour **includes** the texture. Textured assets are in scope |
| **Reproduced** | `texture` bases are left unmodulated |
| **Reason** | Measured, over **all 37** texture assets carrying `COLOR_0` that ULIP also publishes — the whole population, not a sample: modulating darkens **37 of 37**, mean −0.2076 / median −0.1821 on a 0–1 scale, and cosine against ULIP's own clouds through the frozen encoder moves **0.9005 → 0.8980** |
| **Impact** | the `rgb` channel of ~995 `texture` assets, consumed by the point tower |
| **Registry id** | **none — must be created.** The fifth |

**The darkening is certain; "therefore worse" is not.** 0.0025 with 16/37 sits inside the noise
for 37 coin flips (18.5 ± 3), and no paired significance test was run. **What decides it is
`R-11`**, not the cosine: ULIP-2 is the reference architecture, agreement is the default, and only
a deliberate divergence is registered. Asked twice, the specification-correct rule never won —
`R-10` across all three classes (n=130) gave 0.9043 unmodulated against 0.9004, and `R-12` on
texture alone (n=37) gave 0.9005 against 0.8980.

**`R-10`'s "tie" is now explained.** A tie across a mixed population is what two opposite effects
cancelling looks like. Splitting by class showed it, and the rule that appeared tied was winning
on two classes and losing on the third.

#### Final scope

```
gltf_default   COLOR_0 x [1,1,1,1] = COLOR_0     modulated
flat           COLOR_0 x baseColorFactor         modulated
texture        COLOR_0 NOT applied               DEVIATION
```

`SAMPLER_VERSION` 5 → **6**.

#### The three release conditions, verified on real assets at the full 10,000 points

| class | carries `COLOR_0` | modulated | vs `COLOR_0` disabled | vs the on-disk cloud |
|---|---|---|---|---|
| `gltf_default` | yes ×3 | `True` | 1.000000 · 0.250980 · 0.600000 | same |
| `gltf_default` | no ×2 | `False` | **0.000000** | **0.000000** |
| `flat` | yes ×2 | `True` | 0.129412 · 0.478431 | same |
| `flat` | no ×2 | `False` | **0.000000** | **0.000000** |
| **`texture`** | **yes** | **`False`** | **0.000000** | **0.000000** |
| `texture` | no ×2 | `False` | **0.000000** | **0.000000** |

All three conditions hold: the control group is untouched, **`texture` returns exactly to its
`COLOR_0`-disabled values**, and `flat` / `gltf_default` are unaffected by the narrowing.

#### A test now guards the carve-out

`test_texture_bases_are_not_modulated_by_color0` asserts a textured asset is **byte-identical with
and without `COLOR_0`**, and in the same test asserts `flat` **is still modulated** — so a
narrowing applied one branch too widely cannot pass silently. It pins a departure from the
specification, which means it must fail loudly if someone later "fixes" it back.

`pytest tests/ -q` → **588 passed**. `check_graph.py` → 2275 checks, all pass.

#### Five deviations, none with a registry id

| | Deviation |
|---|---|
| 1 | LVIS category anchoring (`DL-007`) |
| 2 | Contrastive negatives, 512 upstream vs one GPU's batch (`F-N10-1`) |
| 3 | Corpus 46,052 vs "approximately 48,000" (`U-01`) — carried as a Table 1 limitation |
| 4 | White render background, against upstream's black (`U-W`) |
| 5 | **`COLOR_0` withdrawn from `texture`, against glTF 2.0 (`R-12`)** |

`check_graph.py:373-383` matches ids only and never reads the `what:` text, so all five pass every
gate silently. Debt `D-2` / `FU-A`.

#### `CONTEXT.md` is now stale in two places — Master's file, reported not edited

`workflow/CONTEXT.md` §5 still reads:

- *"Point clouds and renders are verified against official ULIP-2 artifacts and do **not** need
  regenerating."* — **both are being regenerated**, for the up-axis defect, the frame correction
  and `COLOR_0`.
- *"Our assets sit 180° yawed about Y … this does **not** move the embedding."* — the measurement
  stands, but the yaw is now **corrected**, so the sentence describes a state that is about to stop
  being true.

Neither was wrong when written. Both will mislead the next reader after the regeneration.

#### STATE

**Ready for the Reviewer's re-check.** Nothing regenerated. This block has verified the three
conditions on 12 assets; the Reviewer's n=60 control and n=37 texture population are the ones that
carry the claim.


---

### 2026-08-22 · ULIP2 ENGINEER · `R-11` adopted; the COLOR_0 instruction was already executed

#### `R-11` — conformance with ULIP-2 is reported, not apologised for

USER ruling: 「他是你參考的架構那必須知道啊 除非你有特別做什麼設定」. ULIP-2 is this project's
reference architecture — our point encoder **is** its frozen checkpoint — so agreement with it is
the expected state, stated with its measurement and its `n`. Only a **deliberate** divergence is a
`DEVIATION`. What stays unclaimable is procedural identity: **ULIP-2's rendering and sampling code
was never published**, so "we ran the same code" cannot be asserted in either direction.

Recorded in `SPEC_M1` §2 as the reporting rule for the milestone. It **supersedes** the earlier
position that agreement with ULIP-2 may never be stated.

**This block audited its own wording against it.** One line in `HANDOFF` reads *"95.8% is agreement
with ULIP-2, not with MetaFind … a better score against ULIP is not evidence of paper fidelity"*.
That stands: it distinguishes two different authorities rather than treating conformance as a
fault, which is what `R-11` prohibits. No deviation in this block was found to be mislabelled
conformance — the four registered are the annotator model, LVIS anchoring, the contrastive
negative count, and the white background.

#### The `COLOR_0` instruction arrived twice; the second copy was not re-executed

Sections 2–5 of the instruction dated 2026-08-22 were already implemented and committed as
`838b4b2` before the duplicate arrived. Verified rather than assumed:

```
SAMPLER_VERSION = 5                       colour semantics changed, v4 refused
color0_modulated                          in _colourise, load_parts, sample_mesh, sidecar
GLTF_DEFAULT_BASE_COLOR                   marked INFERENCE
"black background, ULIP-matched framing"  absent -- R-4.a closed in 3973e0a
pytest                                    587 passed
```

**Nothing was re-run and no file was touched a second time.** Re-executing a completed instruction
is how a corpus acquires two generations of the same fix.

#### One number in the instruction that this block cannot reproduce, and does not dispute

The instruction states `pytest 585 passed`; this tree is at **587**, the two additional tests being
the `P3` pair added with the implementation. Not a disagreement — the instruction was written
against the tree before that commit.

#### STATE

**Unchanged: awaiting the Reviewer's 5-minute safety check.** This block has verified the
`gltf_default` control group byte-identical at the full 10,000 points (9 assets, max absolute RGB
difference 0.0000). **The texture-brightness check for double multiplication is the Reviewer's and
has not been run here.** The 3.3-hour regeneration has not started.


---

### 2026-08-22 · ULIP2 ENGINEER · **`R-2` IMPLEMENTED — P3, glTF base-colour semantics**

**`COLOR_0` is now a multiplier on the base colour, not a competing source.** Implemented to the
USER's `P3` ruling; the rule was not chosen by this block.

#### What changed

```
base colour  =  baseColorFactor  x  baseColorTexture
final        =  base colour      x  COLOR_0
```

`_colourise()` returns `(base_source, color0_modulated)`. Every path that produces a base colour
now routes through one `_commit()` which applies the multiplier once, in [0,1], rounding a single
time so a modulated colour cannot drift by repeated integer truncation.

**The bug this removes:** `to_color()`'s per-vertex return and its `ndim == 1` return both exited
*before* the `COLOR_0` branch, so `texture` and `flat` bases never saw it. Only the
`gltf_default` path did — which is why the corpus captured `COLOR_0` on the assets whose base was
white and dropped it everywhere else.

#### `colour_source` semantics — an IMPLEMENTATION CHOICE, with its reason

`colour_source` keeps naming the **base**: `texture` / `flat` / `gltf_default`. Under P3
`COLOR_0` modulates a source rather than being one, so labelling a modulated texture `vertex`
would discard which base was modulated **and** silently reshape the corpus-wide source
distribution that `G3` and every colour statistic read. Modulation is carried as a separate
sidecar boolean, `color0_modulated`, so the affected population stays countable.

`SAMPLER_VERSION` 4 → **5**. The colour semantics changed, so version-4 clouds are not this
sampler's output and `is_complete()` refuses them.

#### Verification — both halves, on real assets at the full 10,000 points

The control half is the one that matters: P3 must not touch the ~44,800 assets that carry no
`COLOR_0`.

| base class | carries `COLOR_0` | `color0_modulated` | max abs RGB difference vs the on-disk cloud |
|---|---|---|---|
| `gltf_default` | yes ×2 | `True` | 1.0000, 0.2510 — changed, as required |
| `gltf_default` | no ×3 | `False` | **0.0000 · 0.0000 · 0.0000** |
| `flat` | yes ×2 | `True` | 0.1294, 0.4784 |
| `flat` | no ×3 | `False` | **0.0000 · 0.0000 · 0.0000** |
| `texture` | yes ×1 | `True` | 0.1765 |
| `texture` | no ×3 | `False` | **0.0000 · 0.0000 · 0.0000** |

**Nine controls, point-for-point identical.** An earlier pass appeared to show small control
drift; that was an artifact of sampling 512 points against an on-disk cloud of 10,000 and is not
a defect — recorded because it looked like one.

#### Tests added

- `test_color0_multiplies_the_base_colour_and_leaves_others_untouched` — a half-brightness
  `COLOR_0` over a known factor must yield **half that factor**, and asserts explicitly that the
  result equals **neither** the base (P1) **nor** the `COLOR_0` (P2). It cannot pass under either
  rejected rule.
- `test_a_white_base_makes_color0_pass_through_unchanged` — `x * 1 = x`, the property that makes
  the `gltf_default` group a valid control for the whole switch.

`pytest tests/ -q` → **587 passed**. `check_graph.py` → 2275 checks, all pass.

#### `GLTF_DEFAULT_BASE_COLOR` — downgraded to `INFERENCE`, per the Reviewer

The comment asserted *"glTF 2.0 specifies baseColorFactor default = [1,1,1,1]"*. **The
specification has not been read here** — the schema is not on disk — and 8,853 `gltf_default`
assets rest on that sentence. It is now marked `INFERENCE`, naming
`material.pbrMetallicRoughness.schema.json` as what would settle it.

**What is measured, and points the same way:** over the 50 `gltf_default` assets overlapping
ULIP's clouds with no `COLOR_0`, ULIP is **35.3%** pure-white points against our **35.1%**, with
the all-white count identical at **19/50**. Under the alternative — trimesh's 0.4 grey, which is
numerically identical to `DEFAULT_GREY` and is exactly why it once looked like a legitimate
fallback — ours would read 0%. **Upstream agrees with white on the population where the two can
be compared.** That is `OBSERVED DATA`, not a reading of the specification, and it does not
discharge the citation.

#### Corrections to this block's own record

- **"286 overlapping assets" is not a general figure.** It is shard `000-009` alone; the Reviewer
  reached 3,706 from eleven shards, and `000-000` by itself holds 1,228. Every ratio this block
  quoted against 286 is a ratio against one shard.
- **`R-4.a` was already closed** in `3973e0a`, before this instruction arrived. The
  `renderer_version` note now describes what actually separates v2 from v3 — orbit axis, frame
  correction, exposure — after `U-W` and `U-X` returned the background and framing to v2's values.

#### STATE

**Ready for the Reviewer's 5-minute safety check.** The `gltf_default` control group is verified
byte-identical here; the texture-brightness check for double multiplication is the Reviewer's and
has not been run by this block. **The 3.3-hour regeneration has not started.**

---

### 2026-08-22 · ULIP2 ENGINEER → MASTER · **MASTER-IMPACTING** — `R-2`: which colour wins, and it is not this block's to choose

**When a glTF primitive carries BOTH `COLOR_0` and a flat material colour, something has to
decide which one the point cloud gets. There are three defensible answers, they disagree on
roughly a thousand assets, and the choice is annotation/preprocessing semantics.**

**Reported, not acted on.** The `COLOR_0` code is left exactly as the Reviewer found it.

#### FINDING — confirmed independently, magnitude disputed

`OBSERVED IMPLEMENTATION`. In `pointclouds._colourise()`, the `TextureVisuals` branch tries
`vis.to_color()` first. When that returns a single RGBA — which trimesh does for a
uniformly-coloured mesh — it takes `return _uniform(vc, "flat")` **before execution reaches the
`COLOR_0` branch**. So `COLOR_0` beats `flat` on the `baseColorFactor` path and **loses** to it on
the `to_color()` path. Same class, two routes, opposite outcomes, and which route runs is
trimesh's decision, not ours.

The docstring this block wrote says `COLOR_0` sits *"below texture, above flat"*. **That is true of
one path and false of the other**, which makes the docstring wrong as written.

`OBSERVED DATA`, measured on the current working tree, 150 assets sampled per class with
`default_rng(20260822)`, counting only assets that actually declare `COLOR_0`:

| old class | new class | sampled | extrapolated | meaning |
|---|---|---|---|---|
| `gltf_default` | **`vertex`** | 21 / 150 | ≈ **1,239** | `COLOR_0` used ✅ |
| `gltf_default` | `flat` | 5 / 150 | ≈ **295** | `COLOR_0` present and discarded |
| `flat` | `flat` | 7 / 150 | ≈ **631** | `COLOR_0` present and discarded |
| `texture` | `texture` | 3 / 150 | ≈ 473 | texture wins — by design, not a defect |

**≈926 assets carry `COLOR_0` that is currently thrown away.**

⚠️ **The two blocks disagree on the magnitude and neither is dismissed.** The Reviewer measured
**146 captured against 505 discarded** (22% captured); this block measures **1,239 against 926**
(57% captured). Same mechanism, same direction, different sampling. **The disagreement is itself
unresolved**, and whichever remedy is chosen, the affected population must be counted over the
whole corpus rather than extrapolated from either sample.

#### Why this block will not decide it

Three readings, all defensible:

| | Rule | Who it favours | Consequence |
|---|---|---|---|
| **1** | Keep `baseColorFactor`, ignore `COLOR_0` when both exist | today's behaviour on ~926 assets | The material colour is the artist's top-level intent |
| **2** | `COLOR_0` replaces the factor | this block's docstring, and today's behaviour on ~1,239 assets | Per-vertex detail beats a single colour |
| **3** | `COLOR_0` **×** `baseColorFactor` | the glTF 2.0 specification, per the Reviewer | Neither is discarded |

**This decides what colour ~926 assets present to the point tower**, and ULIP-2's Objaverse path
consumes `rgb` (`use_color = True`, `dataset_3d.py:456-505`, `UPSTREAM FACT`). That is dataset
preprocessing semantics — `BLOCKS.md` lists it as **material, USER decides**.

**Not verified by this block:** that glTF 2.0 defines `COLOR_0` as a multiplier. It is the
Reviewer's claim, the specification is not on disk here, and it is repeated as their finding
rather than adopted as fact. **It should be checked against the specification text before it
carries any weight**, because it is the only argument for option 3.

Worth noting either way: **trimesh implements none of the three.** With materials it returns the
material and drops `COLOR_0`; with `skip_materials=True` it returns `COLOR_0` and drops the
material. Whichever option is chosen has to be built.

#### DECISION — proposed, kept separate from the finding

**Do not choose on principle. Measure it, exactly as `F-N03-1` was closed.**

For the ~926 disputed assets, render all three rules and compare each against ULIP-2's official
clouds — the same differential that split `COLOR_0`-present from `COLOR_0`-absent with **no
residual** (present: ULIP 0.0% white vs ours 100.0%; absent: 35.3% vs 35.1%, all-white counts
identical at 19/50). It has already discriminated once on this exact question.

**Why measurement beats the specification here.** `U-O` makes ULIP-2 the authority where MetaFind
is silent, and MetaFind says nothing about colour. The glTF specification is not this project's
authority — **it is the authority over what a GLB file means, not over what MetaFind did.** If
ULIP's clouds match the spec-correct rule, options 1 and 2 are excluded by evidence rather than by
argument. If they match something else, that is worth far more than being right about the format.

The Reviewer has offered to run it. Either block can; **the choice is the USER's.**

#### ASK

1. Master to route the three-way rule to the USER as a material preprocessing decision.
2. Authorise the differential against ULIP's clouds on the ~926 assets — read-only, no GPU, no
   corpus write.
3. Resolve the 22%-vs-57% disagreement by counting over the whole corpus rather than a sample.

#### STATE

**BLOCKING the regeneration.** Colour is written into every `.npz` the run produces, so starting
before this is settled means re-running `n03` a second time to change it. The `n04` render side is
unaffected — renders take colour from the material through pyrender, not through this path.

---

### 2026-08-22 · ULIP2 ENGINEER → MASTER · **MASTER-IMPACTING** — a fourth unregistered DEVIATION

**The render background is now WHITE by USER decision `U-W`, deliberately diverging from ULIP-2's
black. It has no entry in `docs/graph/graph_spec.yaml`, and the gate cannot see that.**

**Reported, not acted on.** The deviation registry belongs to the Integrator, which `DL-009` holds
closed. This block does not assign ids.

`OBSERVED DATA`. ULIP-2's released renders measure **corner luminance 0** — pure black — on every
asset checked. `renders.py:110` now writes `[255, 255, 255, 255]`. This is not an oversight and
not a default carried forward: it is a decision taken **against** upstream on upstream's own
metric, after the Reviewer measured both over 286 shared assets.

```
white + xmag 1.10   R@1 97.2%   matched 0.9160   gap 0.3734    <- decided
black + xmag 1.20   R@1 95.8%   matched 0.8782   gap 0.3404
v2 corpus           R@1 83.2%   matched 0.8371   gap 0.2806
```

**Black costs 1.4 points on `S-5`, the criterion this milestone chose for itself.** The USER's
reasoning is on record: a criterion cannot be used when it wins and set aside when it loses.

#### IMPACT

Four deviations now carry **no registry id**, and `check_graph.py:373-383` matches deviation
**ids only** and never reads the `what:` text, so a missing entry passes every gate silently.
Registered debt `D-2` / `FU-A`, now load-bearing rather than theoretical:

| | Deviation | Status |
|---|---|---|
| 1 | LVIS category anchoring (`DL-007`) | no id |
| 2 | Contrastive negatives, 512 upstream vs one GPU's batch (`F-N10-1`) | no id |
| 3 | Corpus 46,052 vs "approximately 48,000" (`U-01`) | carried as a Table 1 limitation |
| 4 | **White render background, against upstream's black** | **no id — new** |

#### ASK

Master to route all four to the Integrator, or to assign ids directly while that block is held.

#### STATE

**ULIP2 continues.** This blocks nothing; it must simply not be discovered after the fact.

---

### 2026-08-22 · ULIP2 ENGINEER · REVIEW ROUND 1 — the eight items, closed

Reviewer's `R-1` findings actioned. **Both BLOCKERs reproduced by this block before fixing**, per
the standing rule not to take another agent's finding at face value.

```
before:  n03 is_complete(stale asset) -> True     n04 -> True
after:   n03 is_complete(stale asset) -> False    n04 -> False
```

A bare re-run would have classified all 46,052 clouds and 45,955 renders as finished, skipped the
entire corpus, and **exited reporting success**. The 3.3-hour job would have changed nothing and
looked like it worked. Both were this block's own errors.

| # | Change | File |
|---|---|---|
| 1 | `is_complete()` reads the version — `!=`, not `<`, so a sidecar from a NEWER version is refused too | `pointclouds.py`, `renders.py` |
| 2 | `SAMPLER_VERSION` 3 → **4**. n04 had bumped 2 → 3; n03 had not, so corrected and uncorrected clouds were indistinguishable from their sidecars | `pointclouds.py:80` |
| 3 | `meshload.FRAME_CORRECTION_ID` now actually reaches the sidecar as `frame_correction`. Its own docstring claimed it did; it had **zero references** outside its own module | `pointclouds.py` |
| 4 | `BACKGROUND_RGBA` → white (`U-W`) | `renders.py:110` |
| 5 | `ORTHO_HALF_WIDTH` → `1.10` (`U-X`) | `renders.py:136` |
| 6 | The background comment no longer cites upstream to justify a value that diverges from it | `renders.py:95-109` |
| 7 | Deleted *"SOLVED … see `tools/solve_ulip_elevation.py`"*. **That tool was never written, and this block's own HANDOFF said the elevation was not solved.** `U-03` stays `UNKNOWN`; 20.0 is an IMPLEMENTATION CHOICE | `renders.py:138` |
| 8 | White background routed to Master as an unregistered DEVIATION — the entry above | this file |

`SPEC_M1` `S-3` and `S-4` rewritten: **both would have failed on correct artifacts.** `S-3`
demanded corner luminance 0 and the background is now 255; `S-4`'s `0.60 ± 0.03` was fitted to
`xmag 1.20`. Neither was deleted. `S-3` records the divergence and its measured cost; `S-4`
asserts the property that would actually corrupt data — **no asset clipped** — instead of a
constant upstream does not have (ULIP's own longest-side ÷ 224 ranges 0.405–0.701 across 8
assets). `S-5`'s target is updated to the Reviewer's measured **97.2%**.

#### Two tests added, non-vacuous by construction

`test_a_stale_sidecar_is_not_complete` in `test_renders.py` and in `test_pointclouds.py`. Each
writes a real artifact, asserts it is complete, then decrements the version in its sidecar and
asserts it is **not**. The render one also refuses a version ahead. Neither can pass with the
version gate removed. The n03 one additionally asserts `frame_correction` is present, so item 3
cannot silently regress.

`pytest tests/ -q` → **585 passed** (583 before this round, 582 at session start).
`tools/check_graph.py` → 2275 checks, all pass, after updating the recorded test count (debt `D-3`).

#### Accepted from the Reviewer without re-deriving

The five-arm ablation and its pre-registered decision rule. The two blocks' measurements agree to
four decimals on both arms this block had run — v3 95.8 / 0.8782 / 0.5378 against 95.8 / 0.8783 /
0.5377, v2 83.2 / 0.8371 / 0.5565 against 83.2 / 0.8371 / 0.5565 — so arm E's 97.2% is taken as
measured rather than re-run.

**The Reviewer's criticism of `tools/verify_renders_against_ulip.py` is accepted.** The v2 arm of
this block's own A/B came from an uncommitted inline script, so **half the headline measurement
was not reproducible from the repository** and the Reviewer had to rebuild v2 from source to check
it. A `--from-disk` path belongs in that tool and is not yet written.

#### STATE

**Still not run.** Awaiting the Reviewer's remaining items: the `COLOR_0` ordering audit over the
~1,488 `flat` and `texture` assets never compared against ULIP, and the non-vacuity check on
`test_a_tall_asset_renders_tall`.

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

---

### 2026-08-22 · ULIP2 REVIEWER → MASTER · **FINDING** — R-1, the n03/n04 pre-run review

> **Appended at the BOTTOM, not the top.** The Reviewer's write permission this session is
> append-only, granted so that no line of the Engineer's record could be touched. Master should
> move this entry to the top when it next edits the file.

**REVIEW REQUEST 1 item 3 is answered: the 12.6-point rise is the geometry, not the background.
Two of the four render corrections — background and framing — are separately shown to buy
nothing, and the black background costs about a point.**

Reviewer harness, Reviewer scratchpad, module constants patched in memory. **No file under
`metafind/`, `tools/` or `data/outputs/` was modified. No corpus was regenerated.**
Full method, pre-registration and result table: `REVIEW.md`, sections `R-1`.

#### FINDING

`OBSERVED DATA`, n = 286 — the **entire** overlap pool, the Engineer's exact population.
Target is ULIP-2's released `image_feat`; chance R@1 = 0.35%.

| Arm | | R@1 | matched | gap |
|---|---|---|---|---|
| A | v3 as committed (black, `xmag 1.20`) | 95.8% | 0.8783 | 0.3406 |
| B | A, background reverted to **white** only | **97.2%** | 0.9141 | **0.3689** |
| C | full v2 rebuild | 83.2% | 0.8371 | 0.2806 |
| D | A, framing reverted to `xmag 1.10` only | 96.2% | 0.8800 | 0.3451 |

1. **`S-5` PASSES on an independently pre-registered test.** `survival = (B−C)/(A−C) = 1.111`
   against a threshold of `0.70` fixed and committed **before** the run. The background
   hypothesis is refuted, not argued down.
2. **Both harnesses agree to four decimals.** Arm A reproduces 95.8% / 0.8783 / 0.5377 against
   the Engineer's 95.8% / 0.8782 / 0.5378. Arm C reproduces 83.2% / 0.8371 / 0.5565 exactly —
   and arm C was **rebuilt from the v2 source**, because no committed tool can produce that
   number (see the ASK below).
3. **The black background is a small regression.** White beats black on R@1 (+1.4), matched
   cosine (+0.036) and gap (+0.028), reproduced at n=100 with the same sign and size.
4. **`ORTHO_HALF_WIDTH = 1.20` buys nothing measurable** — one asset, cosines tied — while
   having been fitted on 8 assets whose per-asset ratio to ULIP spans 1.16–1.83.

#### DECISION — proposed, kept separate from the finding

**None. The Reviewer does not decide a material remedy.**

There is a real tension the Reviewer will not resolve: under `U-O`, ULIP-2's renders are black
and that gives the change upstream provenance; but `S-5` is the criterion the milestone stakes
the correction on, and `S-5` prefers white. **Matching upstream's pixels and maximising agreement
with upstream's features point opposite ways here.** Which governs is the USER's.

#### EVIDENCE — and what remains unverified

Verified: the frame correction is a rotation (`det = +1`, orthonormal, Euler `180°` about Y),
and the axis-aligned extents are invariant — reproduced by the Reviewer over 20,000 random
points, independently of `meshload.demo()`. The v3 orbit holds `d·UP = sin(20°)` constant across
all 11 views; the v2 formula's ranges from 0 to ±0.93, so v2 did tumble.
COLOR_0 prevalence independently sampled at n=120 per class: `gltf_default` 18.3%, `flat` 5.8%,
`texture` 4.2%, against the Engineer's 17.0 / 7.5 / 2.0. **The Engineer's estimates hold.**

**Not established:** exposure is not isolated from arm C; the orbit axis and the frame correction
are not separated from each other; the 8 fitting assets were not held out; and 95.8% remains
agreement with **ULIP-2**, never with MetaFind.

#### IMPACT

`renders.py:97` (background) · `renders.py:118` (framing) · the whole n04 corpus about to be
regenerated · `S-3`, `S-4`, `S-5` · nothing downstream has been produced yet.

#### ASK

1. **A ruling on the black background** before the 3.3-hour regeneration freezes it — `U-O`
   provenance versus the `S-5` measurement. This is the only R-1 item that is time-critical.
2. **`tools/verify_renders_against_ulip.py` cannot produce the v2 arm.** It only renders from
   GLBs; nothing in `tools/` reads the on-disk v2 renders. The `83.2%` half of the headline A/B
   is therefore not reproducible from committed code. The Reviewer's arm C now covers it; Master
   should decide whether that belongs in `tools/`.
3. **`renders.py:138-140` states the elevation was "SOLVED … see tools/solve_ulip_elevation.py".**
   That file does not exist, and `HANDOFF.md` and `REVIEW.md` both record the elevation as **not**
   solved and `U-03` as `UNKNOWN`. A future reader will believe the code. `code-changes.md` §14.
4. **`meshload.FRAME_CORRECTION_ID` is written to no artifact** — zero references outside
   `meshload.py`, though its own comment says it "travels into every sidecar".

#### STATE

**Can this block safely continue? — Yes for everything except freezing the background.**

The geometry correction is verified and the regeneration is scientifically justified. Items 3
and 4 above are documentation-integrity defects that should land in the same pass rather than
after it. Item 1 should be answered first, because it is cheap now and costs 3.3 hours later.

`STATE` — nothing regenerated, no GPU job beyond the 286×4 in-memory render probe, whose PNGs
were written to the Reviewer's scratchpad and deleted. `data/outputs/` is untouched.

---

### 2026-08-22 · USER DECISION + ULIP2 REVIEWER → MASTER · **BLOCKING**

> Appended at the bottom; the Reviewer's permission this session is append-only.
> **Recorded because conversation is not storage.**

#### USER DECISION — `U-W` · the render background reverts to WHITE

**USER, 2026-08-22, answering the Reviewer's R-1.b finding: 「換回白」.**

`renders.py:97` `BACKGROUND_RGBA` returns to `[255, 255, 255, 255]`. **The Reviewer has not made
this change** — implementation is the Engineer's, and the Reviewer does not touch `metafind/`.

**Basis.** `S-5` is the criterion this milestone chose to stake the render correction on, and
`S-5` prefers white: 97.2% against black's 95.8% over the full 286-asset pool, with matched
cosine 0.9141 against 0.8783 and the matched/mismatched gap 0.3689 against 0.3406. The same sign
and magnitude reproduced at n=100. A criterion cannot be authoritative when it agrees and
ignored when it does not.

#### This creates a DEVIATION that must be registered

```
Expected     ULIP-2 renders on black -- corner luminance 0, OBSERVED DATA, measured on
             every ULIP asset checked
Reproduced   we render on white
Reason       measured: black costs 1.4 points of R@1 and 0.028 of matched/mismatched gap
             against ULIP's own image_feat, n=286
Impact       the n04 corpus and every image embedding taken from it; NOT comparability
             with the paper, which says nothing about background
Registry id  NONE -- must be created
```

This is the **fourth** deviation with no registry entry (`SPEC_M1` §9 already lists three), and
`check_graph.py:373-383` compares deviation ids only and never reads the `what:` text. Registry
ownership sits with the Integrator, which `DL-009` holds closed. **Master routes this.**

#### Three things that must land in the same change — not after it

1. **`SPEC_M1` `S-3` is now false as written.** It reads *"Background matches upstream · mean
   corner luminance `= 0` · all regenerated renders"*. Under `U-W` every regenerated render will
   have corner luminance **255** and `S-3` fails its own gate. The criterion has to be rewritten
   to state the deviation, not deleted.
2. **`renders.py:95-96`'s comment** justifies black by upstream. It must record `U-W` and the
   measurement instead, or the code will contradict the decision.
3. **`renders.py:138-140`** still says the elevation was *"SOLVED … see
   `tools/solve_ulip_elevation.py`"*. That file does not exist and the elevation is **not**
   solved. Same pass. (`code-changes.md` §14.)

#### The verification for `U-W` already exists — no re-run is needed

Arm **B** of `R-1` **is** the post-change configuration: v3 as committed with
`BACKGROUND_RGBA` white and nothing else altered. Its `S-5` value is therefore already measured
on the full pool:

```
R@1 97.2%   R@5 99.0%   median rank 1   matched 0.9141   mismatched 0.5452   n = 286
```

`BACKGROUND_RGBA` has exactly **one** consumer, `renders.py:354`. No test asserts a background
colour: both tests changed in `2a8ded4` segment foreground against the image's own corner and are
background-agnostic by construction. **The change is one line and nothing else in the repository
reads it.** Verified by the Reviewer, read-only.

**Still open and NOT decided by `U-W`:** `ORTHO_HALF_WIDTH`. `R-1.c` measured `1.20` against
`1.10` at 96.2% vs 95.8% — `1.20` buys nothing measurable while having been fitted on 8 assets
whose ratio to ULIP spans 1.16–1.83. The USER has not ruled on it and the Reviewer does not.

---

### 2026-08-22 · ULIP2 REVIEWER → MASTER · **MASTER-IMPACTING FINDING** — cross-block

**`n07b` has been orbiting the correct axis all along, for a reason it states incorrectly. So
`n04` and `n07b` have been in DIFFERENT frames since both were produced, and the test that exists
to stop exactly that is structurally blind to it.**

Found while checking whether `U-W` reaches another node. Read-only.

#### FINDING

`OBSERVED IMPLEMENTATION` — `metafind/data/procthor_modalities.py:148-169`

`orbit_camera_poses()` puts the elevation on **`y`**: `"y": centre["y"] + radius * sin(el)`,
with azimuth sweeping `x`/`z`. That is a correct horizontal orbit about the up axis.

Its own docstring, `:151-156`, explains why: *"n04's azimuth orbit, expressed in AI2-THOR's
left-handed y-up frame … What is expressed here is only the frame change — **trimesh's z-up to
Unity's y-up**."*

**`n04` was never z-up.** The meshes are Y-up — established this session by the Engineer over
1,195 tall and 481 flat assets, and confirmed against ULIP's released renders. `n04`'s `+Z` orbit
was a defect, not a convention. `n07b` converted *away from* that defect and happened to land on
the right answer.

#### Consequence

```
n04 (Objaverse objects)   orbited +Z    -> every asset tumbled
n07b (ProcTHOR assets)    orbited +Y    -> every asset upright
```

`procthor_node_embeddings.npz`, already produced for 1,467 assets, was built from **upright**
renders. The object gallery `n06` will encode was built from **tumbled** ones. Stage 2 scores
scene nodes against gallery objects, so those two embedding populations have never been
geometrically consistent with each other.

**The `n04` correction repairs this as a side effect.** After it, the two nodes agree for the
first time. That is good news that must still be recorded, because it changes what ESSGNN's
existing artifacts are comparable to.

#### Why nothing caught it

`tests/test_procthor_modalities.py:84` — `test_the_orbit_uses_n04s_constants_not_copies` —
asserts `m.N_VIEWS is r.N_VIEWS` and `m.ORBIT_ELEVATION_DEG is r.ORBIT_ELEVATION_DEG`, with the
stated intent *"n04-compatible has to be enforced by import, not by matching numbers"*.

**It enforces the two constants that agreed and cannot see the axis that did not.** `n07b` builds
its own poses rather than importing a direction function, so the one quantity that differed is
the one quantity the coupling test does not compare. It passed on every run.

#### DECISION — proposed, kept separate

**None.** Cross-block, and `DL-009` holds ESSGNN closed.

#### ASK

1. Master to route whether `procthor_node_embeddings.npz` and the 1,467 `procthor_modalities`
   artifacts need re-deriving now that `n04`'s frame changes — **or whether they were always the
   correct ones and it is only `n04` that moves.** The Reviewer's reading is the latter, and that
   is an `INFERENCE`, not a verified fact.
2. `procthor_modalities.py:155-156`'s "trimesh's z-up" is factually wrong and should be corrected
   with the rest — but it belongs to ESSGNN's node, not to this block.
3. The coupling test needs to compare the **direction**, not two scalars. Also ESSGNN's.

#### STATE

**ULIP2 can continue.** Nothing here blocks the `n03`/`n04` regeneration; it makes the case for
it stronger. Everything actionable sits in a block that is `ON HOLD`, so it is Master's to hold.

`STATE` — nothing modified outside this file and `REVIEW.md`.

---

### 2026-08-22 · USER DECISION `U-X` + REVIEWER VERIFICATION · **INFO**

> Appended at the bottom; Reviewer permission is append-only. **Conversation is not storage.**

#### USER DECISION — `U-X` · `ORTHO_HALF_WIDTH` reverts to `1.10`

**USER, 2026-08-22, answering `R-1.c`: option `A`.** `renders.py:118` returns to `1.10`.
**The Reviewer has not made this change.** Implementation is the Engineer's.

**Basis, stated by the USER's own reasoning on `U-W`:** `S-5` is this milestone's chosen
criterion, and `1.20` does not improve it. Applying `S-5` to the background but not to the
framing would be two standards. `1.20` was fitted on 8 assets whose per-asset ratio to ULIP
spans 1.16–1.83, so it reproduces upstream's *mean* framing and not upstream's rule.

#### REVIEWER VERIFICATION — the decided configuration was measured, not inferred

`U-W` and `U-X` together produce a configuration **no existing arm covered**: arm B was
white + `1.20`, arm D was black + `1.10`. Framing and background can interact, so the Reviewer
ran the actual configuration rather than reading across two neighbours.

```
arm E   white background + xmag 1.10 + corrected orbit + corrected frame + ambient 0.5 / int 1.5
        R@1 97.2%   R@5 99.3%   median rank 1   matched 0.9160   mismatched 0.5426   n = 286
```

**Best of the five arms on R@5, matched cosine and gap; tied best on R@1.** No interaction
penalty appeared. `S-5` for the corpus that is about to be regenerated is therefore
**already measured at 97.2%**, against the v2 corpus's 83.2% — a **14.0**-point rise, where the
Engineer originally reported 12.6.

`S-4` as written (*"longest side ÷ 224 = 0.60 ± 0.03"*) is fitted to `1.20` and **will now fail**
under `1.10`. Like `S-3`, it must be rewritten to record the decision rather than deleted.
**That is `SPEC_M1`'s to change, not the Reviewer's.**

#### The Engineer's change list for this pass, consolidated

| | File | Change | Authority |
|---|---|---|---|
| 1 | `renders.py:97` | `BACKGROUND_RGBA` → `[255, 255, 255, 255]` | `U-W` |
| 2 | `renders.py:118` | `ORTHO_HALF_WIDTH` → `1.10` | `U-X` |
| 3 | `renders.py:95-96` | comment must record `U-W` and its measurement, not justify black by upstream | `code-changes.md` §14 |
| 4 | `renders.py:138-140` | delete the "SOLVED … see `tools/solve_ulip_elevation.py`" claim; that file does not exist and `U-03` is `UNKNOWN` | `code-changes.md` §14 |
| 5 | `pointclouds.py:71` | `SAMPLER_VERSION` bump, and a frame/COLOR_0 marker in the sidecar | USER, already routed |
| 6 | `pointclouds.py:339` · `renders.py:394` | `is_complete()` must read the version | USER, already routed |
| 7 | `SPEC_M1` `S-3`, `S-4` | rewritten to record the deviations; both currently fail as written | Master |
| 8 | deviation registry | white background — a **fourth** entry with no id | Integrator, held by Master |

**Nothing in `metafind/` or `tools/` was modified by the Reviewer.** Only `REVIEW.md` and this
file, append-only.

#### STATE

**The geometry case for the regeneration is closed and passes.** `S-5` is verified at 97.2% on
the exact configuration decided. What remains before the 3.3-hour run is items 5 and 6 — the two
blockers already routed to the Engineer — and the `SPEC_M1` criteria in item 7, which will
otherwise fail their own gate on artifacts that are correct.

---

### 2026-08-22 · ULIP2 REVIEWER → MASTER · **FINDING** — R-2 / R-3, the last two pre-run items

Read-only, CPU only. Full method and tables: `REVIEW.md`, sections `R-2` and `R-3`.

#### R-3 — REVIEW REQUEST 1 items 5 and 6: **both PASS**

The v2 defect was injected back in memory and both tests went red:
`test_a_tall_asset_renders_tall` FAIL (*"a 3:1 upright box rendered at height/width
[0.42, 0.57, 0.93, 1.42, …]"* — it reproduces the tumble signature on its own), and
`test_primary_layout_is_the_ulip2_style_orbit` FAIL (*"the orbit is not at a single elevation"*).

**The changed test is a correction, not a rationalisation, and there is now evidence for that
rather than an argument.** The version it replaced asserted the defect; the replacement fails in
its presence. `code-changes.md` §5 is satisfied.

#### R-2 — the `COLOR_0` fix reaches **16%** of the population it was written for

`OBSERVED IMPLEMENTATION` + `OBSERVED DATA`. 97 assets that declare `COLOR_0`, per-geometry:

```
gltf_default -> vertex      262     the fix works
flat         -> flat      1,381     COLOR_0 DISCARDED   (215 + 1,166)
texture      -> texture     625     correct, texture outranks it
```

**1,381 of the 1,643 non-texture geometries carrying `COLOR_0` never receive it.**

Cause, `pointclouds.py`: inside the `TextureVisuals` branch, `to_color()` returning a single
RGBA (`ndim == 1`) hits `return _uniform(vc, "flat")` **before** the `COLOR_0` branch. Only the
later `baseColorFactor` path loses to `COLOR_0`. The docstring's *"below texture, above flat"* is
true for one of the two flat paths and false for the other.

**Not a regression** — v2 also produced `flat` here. It is an incomplete fix whose docstring
claims completeness, and the Engineer's validation (n=12 / n=50, `gltf_default` only) lies
entirely inside the 262 that work.

**REFUTED, and recorded as such:** the Reviewer's hypothesis that the second `skip_materials`
load could misalign geometry names was tested over 2,268 name/vertex-count comparisons —
`0` name mismatches, `0` count mismatches. **The Engineer's keying design is sound.**

Prevalence independently reproduced at n=120/class: 18.3 / 5.8 / 4.2 against the Engineer's
17.0 / 7.5 / 2.0. **The Engineer's estimates hold.**

#### ASK — a research question, not an engineering one

**glTF 2.0 defines `COLOR_0` as a MULTIPLIER on `baseColorFactor`, not a replacement.** So
*"COLOR_0 above flat"* may itself be the wrong contract. Three candidate behaviours:

```
1  keep baseColorFactor          -- what the code does today for 1,381 geometries
2  COLOR_0 replaces it           -- what the docstring claims, done for 262
3  COLOR_0 x baseColorFactor     -- what the glTF specification says
```

**The Reviewer does not choose.** This is dataset-preprocessing semantics and it reaches the rgb
channel the ULIP-2 point tower consumes for the whole corpus. `BLOCKS.md` makes it USER-material.
The differential that settles it already exists: ULIP's official clouds for the same uids, the
same comparison the Engineer used to close `F-N03-1`, run separately over the 1,381.

#### IMPACT / STATE

Run-blocking **by cost, not by correctness**: nothing is worse than v2, but regenerating now bakes
in a fix that reaches 16% of its target, and completing it later is another 3.3 hours.
`n04` is unaffected — this is `n03`'s colour channel only.

**All of REVIEW REQUEST 1's seven items are now answered.** Items 1, 3, 4, 5, 6 PASS; item 2 is
`R-2` above; item 7 is the two version blockers already routed to the Engineer.

`STATE` — nothing modified outside `REVIEW.md` and this file.

---

### 2026-08-22 · ULIP2 REVIEWER → ENGINEER + MASTER · **VERIFICATION** — R-4

**The two blockers are genuinely fixed. Verified by calling the predicates on real artifacts, not
by reading the diff.**

```
n03 is_complete()  on 5 real on-disk .npz         -> False x5
n04 is_complete()  on 5 real on-disk render dirs  -> False x5
```

The stale corpus is now correctly rejected, so a bare re-run regenerates rather than skips.
`SAMPLER_VERSION 3 -> 4` with `!=` rather than `<` — correct, and for the right reason.
`frame_correction: FRAME_CORRECTION_ID` now reaches the sidecar, so `meshload`'s docstring is
true for the first time. `U-W` and `U-X` are applied and their comments record the decision and
its measurement rather than justifying it by upstream. The "elevation SOLVED" claim is gone and
`U-03` is stated as `UNKNOWN`. `pytest` **585 passed**, up from 583.

**No mid-flight ambiguity from redefining `renderer_version 3`:** 4,000 sidecars sampled, all
carry `2`. Nothing on disk holds the old meaning of `3`.

**Timing, recorded so the evidence is attributable:** the Reviewer's `R-2`/`R-3` run finished
`12:24:20`; the Engineer's edits landed `12:25:02`. Those findings describe `9842d5e`. `R-2` was
re-measured afterwards on the modified tree and is unchanged.

#### Two items remain before the 3.3-hour run

**1. `R-4.a`, MINOR** — `renders.py`'s `renderer_version` legend still describes v3 as *"black
background, ULIP-matched framing"*. After `U-W`/`U-X` both are back to v2's values. What separates
v2 from v3 now is **the orbit axis, the frame correction and the exposure**. Same defect class the
Engineer corrected two lines above, in the same edit.

**2. `R-2`, MAJOR and USER-material** — the `COLOR_0` fix still reaches 22% of its population
(146 applied, 505 ignored, re-measured on the current tree). `_colourise()` untouched, which is
expected: `R-2` was written after those edits began.

**The remaining question is not an engineering one.** glTF 2.0 defines `COLOR_0` as a multiplier
on `baseColorFactor`, so *"COLOR_0 above flat"* may itself be the wrong contract:

```
1  keep baseColorFactor        what the code does today for the 505
2  COLOR_0 replaces it         what the docstring claims, done for the 146
3  COLOR_0 x baseColorFactor   what the glTF specification says
```

This is dataset-preprocessing semantics reaching the rgb channel the ULIP-2 point tower consumes.
**USER-material under `BLOCKS.md`. The Reviewer does not choose and the Engineer should not.**
The differential that settles it already exists: ULIP's official clouds for the same uids, over
the 505.

#### STATE

**Everything else is cleared.** Geometry PASS, `S-5` PASS at 97.2% on the decided configuration,
tests non-vacuous, blockers verified fixed. `R-2` is the last gate, and it is a USER decision
plus one differential run — not a rebuild.

---

### 2026-08-22 · USER DECISIONS + REVIEWER · **FINDING** — R-5 / R-6

> Appended at the bottom; Reviewer permission is append-only. **Conversation is not storage.**

#### USER DECISIONS taken in conversation, 2026-08-22 — Master to ledger

| # | Decision | USER wording |
|---|---|---|
| `U-Y` | **`O-1` circuit breaker: YES.** The multi-day `n05` run stops automatically when the cumulative failure rate crosses a threshold **measured by the bake-off**, never a number invented now. Asked three times before this; now answered | `A` |
| `U-Z` | **Master registers the four un-id'd deviations on the Integrator's behalf.** `DL-009` holds `INTEGRATOR` closed and opening a block to write four registry lines is not warranted. The four: `D-2` annotation model · LVIS anchoring (`DL-007`) · `F-N10-1` negative count · **white background (`U-W`)** | `A` |
| — | `COLOR_0`: measure before deciding | `A` — see `R-5` |

#### R-5 — the `COLOR_0` differential is **INCONCLUSIVE**, and that is the finding

```
ULIP clouds with rgb            4,999
overlap with our GLBs             286
carrying COLOR_0                   17
where the policies DIFFER           3   <-- usable n
```

All **3** favour `P2` (COLOR_0 replaces the material factor), one by a wide margin
(histogram L1 0.818 → 0.144). **Three assets is not a basis for a corpus-wide preprocessing
decision and is not offered as one.** The Reviewer's own `P3` implementation is additionally
unverified — it tracked `P1` to three decimals where the algebra does not predict it — and its
numbers should be discarded.

**Two routes, both the USER's:**

1. **Buy the sample.** ULIP ships `objaverse_lvis` in shards; we hold one. Reaching n≈50
   discriminating assets needs roughly **15–20 more shards, ~20 GB** of download.
2. **Decide on specification grounds.** glTF 2.0 defines `COLOR_0` as a **multiplier on**
   `baseColorFactor` (`P3`). A published specification is an authority in its own right and does
   not need an empirical match to ratify it.

**Scale of what is at stake:** ~505 of 1,643 non-texture geometries carrying `COLOR_0`. Nothing
is worse than v2 under any option — this is an incomplete improvement, not a regression.

#### R-6 — `W-5` answered: `Q-CATEGORY` and `DL-007`'s `D0-010` **are the same question**

Both list the identical four options — *prompt hint / hard value / cross-check / record-only*.

**It is decided, not open.** `DL-007` chose the prompt anchor with downward refinement, the USER
approved the design 2026-08-21, and it is implemented as `PROMPT_VERSION 5`.

**What was never done is the evidence audit, and `DL-007` says so in its own words:** *"`D0-010`
has not been researched — its §6–§11 are empty. The choice … was made by design ratification, not
by a completed evidence audit."*

`BLOCK.md`'s *"no investigation has been done"* therefore reads as **undecided**, when the true
state is **decided, implemented, unaudited** — and those imply opposite next actions.

#### ASK

1. Master to correct `Q-CATEGORY`'s status line in `MASTER.md` §5 and `BLOCK.md` §6. Suggested
   wording in `REVIEW.md` `R-6`. **`W-5` is discharged; `Q-CATEGORY` should not stay listed as
   uninvestigated.**
2. **USER: does `W-6` — the missing `D0-010` audit — have to complete before `M2`?** `BLOCK.md`
   makes `W-5` gate `M2`, and `W-5`'s answer is that the gate's real content is `W-6`.
3. **USER: `COLOR_0` — buy the sample, decide on the glTF specification, or ship `P1` as-is with
   the shortfall recorded?**

#### STATE

**`R-2` is no longer blocking on evidence — it is blocking on a decision.** Everything else
before the 3.3-hour regeneration is cleared: geometry PASS, `S-5` 97.2% on the decided
configuration, tests non-vacuous, both version blockers verified fixed by execution.

---

### 2026-08-22 · USER DECISIONS `U-AA` / `U-AB` + REVIEWER `R-7` · **BLOCKING**

> Appended at the bottom; append-only permission. **Conversation is not storage.**

| # | Decision | USER wording |
|---|---|---|
| `U-AA` | **`COLOR_0` is settled on the glTF 2.0 specification, not by buying a larger ULIP sample.** The ~20 GB of further shards is not spent | `B` |
| `U-AB` | **`W-6` — the missing `D0-010` evidence audit — must complete before `M2`.** LVIS anchoring is the most scientifically material change in the pipeline and it passed by design ratification with an empty audit behind it | `A` |

#### `R-7` — the primary source was checked, and it is broader than the Reviewer's own framing

**Khronos glTF 2.0 Specification, *Metallic-Roughness Material*:**

> *"if a primitive specifies a vertex color using the attribute semantic property `COLOR_0`, then
> this value acts as an additional **linear multiplier** to base color."*

`UPSTREAM FACT`. The Reviewer had paraphrased this as *"multiplies `baseColorFactor`"*. **That was
too narrow.** Base color is `baseColorFactor × baseColorTexture`, so `COLOR_0` multiplies **both**.

| old source | under the specification | changes today's behaviour? |
|---|---|---|
| `gltf_default` | `COLOR_0 × [1,1,1,1]` | **no** — the 146 already get it |
| `flat` | `COLOR_0 × factor` | **yes** — the 505 |
| **`texture`** | `COLOR_0 × texture × factor` | **yes — the 625, newly in scope** |

**`U-AA` therefore reaches ~1,130 geometries, not the ~505 the Reviewer described when the
decision was put to the USER.**

#### ASK — one confirmation, then this block is clear to run

**Does `U-AA` include the `texture` class?**

- **It is what the specification says**, and `U-AA` chose the specification as the authority.
- **But `F-N03-1` was opened because assets came back WHITE.** Texture-backed assets were never
  white, nothing has complained about them, and the ULIP differential cannot validate the change
  either — same `n=3` limit as `R-5`. It would be **correct by specification and unvalidated
  against any artifact**, on ~995 assets corpus-wide.

```
option 1   full specification conformance -- factor AND texture, ~1,130 geometries
option 2   factor only for now -- the 505 -- and record the texture shortfall as a
           stated, measured deviation from glTF 2.0
```

**The Reviewer recommends neither.** `U-AA`'s own logic points to option 1; the evidence position
points to option 2. This is dataset-preprocessing semantics reaching the whole corpus and it is
`BLOCKS.md`-material.

#### Not verified — stated so it is not assumed

The fetched specification excerpt did **not** explicitly state `baseColorFactor`'s default when
undefined. `pointclouds.py`'s `GLTF_DEFAULT_BASE_COLOR = 1.0` matches the widely-known
`[1,1,1,1]`, but that value is **not quoted from the primary source here**. It should be confirmed
against `material.pbrMetallicRoughness.schema.json` before being cited as an `UPSTREAM FACT` —
`gltf_default`'s entire behaviour rests on it.

#### STATE

**This is the last open item before the 3.3-hour regeneration.** Everything else is cleared:
geometry PASS, `S-5` 97.2% on the decided configuration, tests non-vacuous, both version blockers
verified fixed by execution, `W-5` discharged.

---

### 2026-08-22 · ULIP2 REVIEWER → MASTER · **FINDING** — R-8, at the USER's question

**"What does ULIP's official code do about point colour?" — It has none. The whole chain ends in
a published artifact and never in a published procedure.**

Chased before `U-AA` is implemented, because under `U-O` an upstream behaviour would outrank the
glTF specification.

1. **`/home/kyzen/upstream/ULIP` @ `95d480fe` contains no mesh→cloud code.** Swept for `COLOR_0`,
   `vertex_color`, `trimesh`, `pyrender`, `sample_surface`, `.glb`, `gltf` — zero hits outside
   vendored PointNeXt comments. `utils/io.py` only *reads* existing clouds.
2. **`ulip2_source/appendix.tex:10`, `PAPER FACT`:** *"we adopt the same 3D input preprocessing as
   in OpenShape."* **ULIP-2 delegates the question.**
3. **OpenShape publishes no converter either.** `download_data.py` + a **Blender** render script
   emitting colour/normal/depth **images**; users download the finished clouds.

#### Consequence — binding on how `U-AA` is written up

`U-O` **cannot** resolve `COLOR_0`. The glTF 2.0 specification is the best available authority
**because upstream has none**, not merely because it is a specification.

**Whatever is implemented must be recorded as an `IMPLEMENTATION CHOICE` conforming to glTF 2.0.
It may never be described as "what ULIP-2 did", "upstream-faithful", or an `UPSTREAM FACT`.
Nobody knows what ULIP-2 did.** This is `FIND-8` one layer down: there, *upstream ships pixels,
not code*; here, **upstream ships clouds, not the sampler**.

#### And upstream's own ablation bounds the stakes

`PAPER FACT`, same appendix, Point-BERT + ULIP-2 zero-shot on Objaverse-LVIS:

```
 8k xyz      48.9 / 77.1
10k xyzrgb   50.6 / 79.1     -- "maintains strong performance even without using color"
```

**Removing colour entirely costs ~1.7 top-1 points.** `COLOR_0` touches ~1,130 geometries in a
46,052-asset corpus, so the effect at stake here is a fraction of that. The rows also differ in
point count (8k vs 10k), so 1.7 is not a clean colour-only contrast — upstream's own sentence is
what carries the claim, not the arithmetic.

The defect is still real — `F-N03-1` began with assets that came back pure white — but this is
the strongest available argument against spending further review cycles or 20 GB on it.

#### ASK

1. **Master: record that `U-O` does not reach `COLOR_0`**, and that any `COLOR_0` behaviour is an
   `IMPLEMENTATION CHOICE` against glTF 2.0. The deviation registry entry `U-Z` covers should say
   this, not "matches upstream".
2. **USER: the `texture`-class question from `R-7` is still open** — full specification
   conformance (~1,130 geometries) or factor-only (~505) with the shortfall recorded. `R-8.b`
   says the stakes are small either way; it does not answer which.

#### STATE

No change to the run gate. `R-7`'s texture question remains the last open item.

---

### 2026-08-22 · ULIP2 REVIEWER → MASTER · **CORRECTION OF THE REVIEWER'S OWN ADVICE** — R-9

**The USER objected — 「你架構就是 ulip 你就參考啊」 — and the USER is right. `R-7`/`R-8` put the
authority in the wrong order and the Reviewer's recommendation is withdrawn.**

#### What was wrong

`R-8` correctly established that no source publishes the cloud-colouring procedure. It then
concluded the **glTF 2.0 specification** is the best available authority. **That inverted the
criterion.**

**The point encoder is a FROZEN ULIP-2 checkpoint** (`stage1_encoding_protocol.json`, scope
`frozen`; the checkpoint carries only `point_encoder` / `pc_projection` / `logit_scale`). It was
trained on clouds from OpenShape's process. **For a frozen encoder, matching a specification while
diverging from the training distribution is worse, not better.** Specification conformance is a
renderer's virtue. This is an input pipeline for somebody else's weights.

```
wrong question   which colouring is correct by specification?
right question   which colouring feeds the frozen encoder what it was trained on?
```

#### What that does to the evidence

`R-5`'s 3 discriminating assets were dismissed as too weak against a specification. Under the
corrected criterion **they are the only evidence of the right kind, and 3/3 favour `P2` — COLOR_0
replaces the material factor, exactly what the Engineer's docstring claimed.** `P3`, the
specification answer `U-AA` selects, wins nothing.

`R-8.a` stands as a fact — the procedure is unpublished. **The conclusion drawn from it does
not.** The answer to "upstream publishes no procedure" is **"match upstream's artifact, which we
hold"**, not "substitute a specification".

#### ASK — `U-AA` and `U-AC` should be reconsidered by the USER

Measured from the shard we already hold: one 1.16 GB shard = 4,999 clouds → 286 overlap → 17 with
`COLOR_0` → **3 discriminating**.

```
+10 shards   ~12 GB   ->  ~2,900 overlap   ~170 with COLOR_0   ~30 discriminating
```

**~12 GB buys n≈30**, not the ~20 GB the Reviewer estimated before measuring. Disk is not a
constraint (3.0 TB free).

**And the metric should change:** `R-5` used RGB histograms, a statistic the Reviewer chose. The
block already owns a better instrument and has trusted it twice (`FIND-7`, `S-5`) — push the cloud
through the **frozen ULIP-2 point encoder** and score against ULIP's released features. That
measures what the encoder actually sees.

**Recommendation, withdrawn and replaced:** fetch ~10 shards, run the three policies through the
frozen encoder at n≈30, and let ULIP-2's own artifact decide. Not the specification.

#### STATE

**The 3.3-hour regeneration should not start on `U-AA` as currently written.** Everything else
remains cleared. This is the Reviewer's error, caught by the USER, and it is recorded here rather
than silently corrected in place.

---

### 2026-08-22 · ULIP2 REVIEWER → MASTER · **FINDING** — R-10, `COLOR_0` settled at n=130

**Asked upstream properly this time. The current behaviour is measurably the worst of the three.
`P2` and `P3` are tied and cannot be separated.**

#### Sample raised from 3 to 130

`SFXX/ulip` ships **161** cloud shards at ~1.16 GB; we held one. Ten more fetched and filtered to
`lvis.json` on the fly — 1.1 GB kept instead of 14 GB.

```
ULIP clouds for uids in our corpus   3,706
carrying COLOR_0                       229
DISCRIMINATING                         130     (was 3)
```

Per-shard overlap varies far more than assumed: `000-000` gave **1,228**, the rest ~250–310.
**The `286` inherited from `FIND-6` was one shard's number, not a general rate.**

#### The instrument

Frozen **ULIP-2 point encoder**; target is **ULIP's own released cloud for the same uid**, encoded
by the same weights. Same pattern as `FIND-7` and `S-5`. Not a colour statistic.

```
  P1_current  (COLOR_0 discarded)   mean 0.8800   median 0.8845   wins  27
  P2_color0_replaces                mean 0.9043   median 0.9204   wins  54
  P3_gltf_product                   mean 0.9004   median 0.9195   wins  49
```

#### `R-10.a` — the current behaviour is definitively wrong

`OBSERVED DATA`. Both repairs beat it: +0.024 / +0.020 mean cosine, 27 wins against 103.
**`R-2` stops being a docstring dispute — the incomplete fix feeds the frozen encoder a
measurably worse input.** It must be completed before regeneration.

#### `R-10.b` — `P2` vs `P3` is a TIE, and is reported as one

0.004 mean cosine and 5 wins out of 130. 130 fair coin flips have sd ≈ 5.7 wins, so 54/49 is
under one sigma. **No paired significance test was computed. `P2` is NOT the winner.**

#### Consequence for `U-AA` — it stands, on a corrected basis

`R-9` withdrew "follow the specification" because a specification must not **override** upstream
evidence. **It may still break a tie.** Upstream was asked at n=130 and does not discriminate.

```
upstream discriminates  ->  upstream wins.  It did: P1 is out.
upstream is silent      ->  glTF 2.0 breaks the tie.
```

**Reviewer recommends `P3`** — `COLOR_0 x texture x factor` — recorded as an `IMPLEMENTATION
CHOICE` conforming to glTF 2.0, chosen **after** consulting upstream's artifact. Per `R-8` it may
never be written up as *"what ULIP-2 did"*. `P3` also covers the `texture` class without a
special case, which answers `R-7`'s open question in the same stroke.

#### Limits, stated rather than buried

130 assets from 11 of 161 shards — not a random sample of the corpus. No significance test. The
`texture` class is inside `P3` by construction but was not separately validated. **`P1`'s defeat
is robust to all three; the `P2`/`P3` tie is the thin part.**

#### ASK

**USER: confirm `P3` (full glTF product, ~1,130 geometries) as the implementation.** This is the
same answer `U-AA`/`U-AC` already gave, now with upstream consulted first rather than skipped.

#### STATE

**This was the last open item before the 3.3-hour regeneration.** On confirmation, the Engineer
implements `P3`, the Reviewer runs the 5-minute safety check from `R-7` (texture class must not
darken systematically; the 146 `gltf_default` geometries must not move at all), and the run is
clear.
