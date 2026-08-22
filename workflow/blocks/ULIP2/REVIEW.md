# REVIEW — ULIP2

**Reviewer:** unassigned · **Mode:** independent, read-only, **synchronous with the engineer**

## What the reviewer is for

Not a second engineer. Attacks the engineer's **contract**, not only its output:

1. Is the contract the engineer defined itself wrong?
2. Was an upstream source-of-truth missed?
3. Do the generated artifacts actually match the source data?
4. Could a schema PASS still be semantically wrong?
5. Could all tests PASS and the science still be wrong?
6. Is there silent corruption?
7. Are the block's work items consistent with each other?
8. Could this block's output contaminate downstream?
9. Did the engineer state an INFERENCE as a FACT?
10. Which failure modes do the engineer's tests not cover?

**Review early.** Before any long GPU job, corpus generation, full training, or expensive
evaluation, the reviewer must already have audited the sources, the contract, a real sample,
and the semantic consistency. A review that starts after the run is a post-mortem.

## Boundaries

Read-only by default. To execute a check, use a read-only command, an isolated output
directory, or a separate git worktree — never the engineer's production files.

**The reviewer may not decide a material remedy.** Findings go to Master via `HANDOFF.md`;
only the USER makes anything FINAL.

## Skills the Reviewer uses

Policy: `workflow/SKILLS.md` §10. Skills are method tools, never authority.

| Skill | Use it for | Claude may invoke |
|---|---|---|
| `mattpocock-skills:research` | source-of-truth and contract audit against primary sources | yes |
| `mattpocock-skills:diagnosing-bugs` | contradictions, conflicting measurements, suspected silent failure | yes |
| `mattpocock-skills:code-review` | an independent 4-axis pass, separate from the Owner's | yes |
| `improve-codebase-architecture` | milestone-only architecture survey | **no — ask the USER to run `/improve-codebase-architecture`** |

### Four axes, reported separately

`STANDARDS` · `SPEC` · `SOURCE / EVIDENCE` · `SCIENTIFIC / SEMANTIC`.
Axis 4 assumes the code runs, the tests pass and the SPEC is met, then asks how the result could
**still** be scientifically wrong: wrong units, coordinate or frame mismatch, generated artifact
disagreeing with its source, label noise, silent corruption, downstream contamination, evaluation
leakage. Never merge the four into one verdict.

### Differential testing is the sharpest tool here

Compare two things that should agree — official upstream artifact vs ours, source metadata vs
generated annotation, before vs after, configuration A vs B. Build a red-capable loop, reproduce,
minimise, form **falsifiable** hypotheses, instrument, then fix. **Do not read the code and guess
the cause first.** This is how the 180 degree yaw was found, and how it was then shown not to move
the embedding (`workflow/blocks/ULIP2/evidence/n03_n04_upstream_verification.md`).

### Review early, not at the end

Before full annotation, corpus generation, n06 full encode, any multi-hour GPU run, full training,
or full evaluation, the sources, the contract, a real sample and the semantic consistency must
**already** have been audited. A review that starts after the run is a post-mortem.

---

## Finding format

```
FINDING          what is true
EVIDENCE         file:line / paper section / measurement + the population it was measured over
CLASSIFICATION   PAPER FACT · UPSTREAM FACT · OBSERVED IMPLEMENTATION · OBSERVED DATA ·
                 INFERENCE · IMPLEMENTATION CHOICE · DEVIATION · UNKNOWN
IMPACT           tasks, artifacts, stages
SEVERITY         BLOCKER · MAJOR · MINOR · NOTE
```

A material finding must carry real evidence. "Looks wrong" is not a finding.

---

## Findings

_None yet._

---

# REVIEW REQUEST 1 — the `n03` / `n04` corpus correction

**Raised by the ULIP2 Engineer, 2026-08-22. Commits `bc863b7`, `2a8ded4`, `b1a927f`.**

**This is a pre-run review, and it is blocking.** The next step is regenerating the whole object
corpus — `n03` 46,052 clouds at ~564/min and `n04` 46,045 assets at ~400/min, **about 3.3 hours of
GPU**. `BLOCKS.md` requires an independent synchronous review before an expensive run, and the
Reviewer seat was empty when this work was done. **Nothing has been regenerated.**

Full evidence with measurements and populations: `HANDOFF.md`, entries dated 2026-08-22.
The contract this is judged against: `SPEC_M1_corpus_and_annotator.md`.

## What changed, and why the Engineer believes it

Four defects were found and corrected. **All four were found by comparing against artifacts we did
not produce**, which is the one thing the Engineer could not have tuned:

| | Defect | Correction | Evidence the Engineer offers |
|---|---|---|---|
| 1 | `n04`'s camera orbited **+Z** while the meshes are **Y-up**, so every asset tumbled end-over-end instead of turning | `azimuth_orbit_directions()` builds a basis around `UP_AXIS` | A 7.2× tall lamppost rendered at image h/w **0.14** = 1/7.2. Falsifiable test: hypothesis "camera up is +Z" predicts `extent_z/extent_y`, correlation **+0.893**; "up is +Y" predicts `extent_y/extent_x`, correlation **−0.671** |
| 2 | Background **white**; ULIP's is **black** | `BACKGROUND_RGBA = [0,0,0,255]` | Corner luminance 0 on every ULIP asset checked |
| 3 | Framing threw away half the frame | `ORTHO_HALF_WIDTH` 1.1 → **1.20** | Longest side ÷ 224 over 8 shared assets: ULIP **0.574**, v2 0.418, v3 **0.574** |
| 4 | Lighting clipped light-coloured assets | ambient 0.4→**0.5**, intensity 3.0→**1.5** | Clipped foreground px: ULIP 0.2%, v2 7.5%, v3 0.7% |

Plus two `n03` corrections sharing one loader (`metafind/data/meshload.py`):

| | Defect | Evidence |
|---|---|---|
| 5 | 180° yaw about Y vs ULIP's clouds (known, `FIND-6`, never corrected) | Symmetric Chamfer median **0.0338 → 0.0120**, >0.1 count **8/50 → 1/50**, better on 48/50 |
| 6 | **trimesh 5.0.0 silently discards glTF `COLOR_0`** when a material is present, so ~1,505 assets got white clouds | With `COLOR_0` (n=12): ULIP 0.0% white, ours **was 100.0%**. Without (n=50): ULIP 35.3%, ours 35.1%, and all-white counts identical at 19/50. **No residual** |

## The claim the Engineer stakes the whole thing on

`tools/verify_renders_against_ulip.py`, 286 assets in both corpora, frozen ViT-bigG-14,
target = ULIP's released `image_feat`:

```
v2 on-disk renders   R@1 83.2%   matched 0.8371   mismatched 0.5565
v3 corrected         R@1 95.8%   matched 0.8782   mismatched 0.5378
chance R@1 0.35%     FIND-9 measured v2 at 83.5%; FIND-7 point clouds at 98.0%
```

## What the Reviewer is asked to attack

**Reproduce, do not read.** Every number above is the Engineer's, and the Engineer already got one
thing wrong in this session — see below.

1. **Is `meshload.FRAME_CORRECTION` a rotation and not a reflection**, and does it truly leave
   `raw_bbox_extents` invariant? If it does not, **every dimension in the n05 corpus moves** and
   the proportions anchor moves with it.
2. **Does the `COLOR_0` recovery touch anything it should not?** The Engineer's control is
   3 assets. The claim is that assets *without* `COLOR_0` are byte-identical before and after.
   That is checkable across thousands, cheaply, with no GPU.
3. **Is the 95.8% real, or is it the black background?** The v2→v3 change altered the background
   from white to black, and ULIP's is black. **A random-looking feature that keys on background
   colour would produce exactly this improvement.** The Engineer's counter-evidence is that
   mismatched cosine *fell* while matched rose. **Test it directly**: re-render v3 geometry with a
   white background and re-score. If most of the 12.6 points survive, the geometry fix is real; if
   they vanish, the Engineer measured the background.
4. **Does the framing number generalise?** `ORTHO_HALF_WIDTH = 1.20` was fitted on **8** assets,
   and the per-asset ratio to ULIP ranged 1.16–1.83 — so upstream's rule is *not* a rescale of
   ours. Is 1.20 the right compromise, and does any asset now get clipped?
5. **`test_a_tall_asset_renders_tall` is new. Is it non-vacuous?** Injecting the old `+Z` orbit
   should turn it red. If it does not, it proves nothing.
6. **Two existing tests were changed to pass.** `code-changes.md` §5 forbids exactly that. The
   Engineer's position is that both tests had encoded the v2 bug — one asserted "every direction
   shares a **z** component" with a comment saying so, the other segmented foreground as "darker
   than white" and counted all 50,176 pixels on a black background. **Judge whether that is a
   correction or a rationalisation.**
7. **What did the Engineer not measure?** `S-7` — that point count, `rgb_scale`,
   `coloured_point_fraction`, `max_radius`, `centroid_offset` and the 21 zero-variance assets are
   unchanged — **cannot be confirmed until the corpus is regenerated.** Is that acceptable, or must
   it be sampled first?

## Where the Engineer is already known to have erred

Recorded so it is not taken on trust. Mid-session this block reported that the image tower's
weights were **random** and the measurement void. That was **wrong**, and it was disproved
in-session: image-to-text matching scores 0.400 matched against 0.298 mismatched with argmax
correct **5/5**. The trigger was a real `open_clip` warning from a throwaway preprocess-only model
construction at `ulip_backbone.py:195-196`.

**Two things that are still NOT established, and must not be reported as settled:**

- **The elevation is not solved.** `ORBIT_ELEVATION_DEG` stays 20° as an IMPLEMENTATION CHOICE.
  R@1 over 60 assets read 100.0 / 98.3 / 98.3 at 5° / 15° / 25° — a one-asset spread. `U-03`
  remains `UNKNOWN`. Our vase reads h/w **2.10** where ULIP reads **1.94**, an 8% residual nobody
  has explained.
- **`U-03a` (projection) is untouched.** A sweep that appeared to favour orthographic 0.89 to 0.57
  **did not normalise framing between the two**, so it measured object size. **That comparison is
  withdrawn**, and the Reviewer should not treat orthographic as evidenced.

## The Engineer's own answer to "how could this still be wrong"

Ranked, and offered as targets rather than as reassurance:

1. **95.8% measures agreement with ULIP-2, not with MetaFind.** MetaFind never states a camera.
   Every render decision here is an `IMPLEMENTATION CHOICE` under `U-O`, and a better score against
   ULIP is **not** evidence of paper fidelity.
2. **Framing and lighting were fitted to ULIP on 8 assets and validated on the same 8.** No
   held-out set. The retrieval score used 286, but those 8 are inside it.
3. **`COLOR_0` changes what the point tower is fed for ~1,505 assets.** Verified against ULIP's
   clouds, never against a downstream metric.

---

# REVIEWER PRE-REGISTRATION — R-1 · the background ablation

**Reviewer, 2026-08-22. Written and committed BEFORE the measurement was run.**
Repo `9842d5e`, working tree clean. Authorised by the USER this session (question A).

## Why this is pre-registered

`REVIEW REQUEST 1` item 3 asks whether the `83.2% -> 95.8%` R@1 rise is the geometry fix or
the white-to-black background change. A decision rule written **after** seeing the numbers is
not a test. The rule below is fixed now and will not be edited; the result is appended under it
whatever it says.

## What is being separated

`v2 -> v3` changed **five** things at once: orbit axis, background, framing, exposure, and the
`meshload` 180-degree frame correction. One aggregate number was reported for all five.

The Engineer's counter-argument — that mismatched cosine FELL (0.5565 -> 0.5378) while matched
rose, which a pure brightness change would not do — is suggestive and **not decisive**.
Tightening the framing from 0.418 to 0.574 of the frame raises object-specific signal and
lowers background-shared signal, producing the same signature. Background and framing are two
suspects with the same fingerprint.

## Arms — same 100 uids, same seed, same frozen ViT-bigG-14, same target

| Arm | Configuration | Purpose |
|---|---|---|
| **A** | v3 as committed: corrected orbit, corrected frame, black, `xmag 1.20`, ambient 0.5 / int 1.5 | must reproduce ~95.8% |
| **B** | A, with **`BACKGROUND_RGBA` white** and nothing else changed | **the ablation** |
| **C** | full v2 rebuild: old `+Z` orbit formula, `look_at` up `(0,0,1)`, **no** frame correction, white, `xmag 1.10`, ambient 0.4 / int 3.0 | must reproduce ~83.2%, and restores the v2 arm that no committed tool can produce |

Target is ULIP-2's released `image_feat`, which we did not produce and cannot tune.
Control is mismatched pairs. Chance R@1 at n=100 is 1.00%.

## Decision rule — fixed before the data

Let `dA`, `dB`, `dC` be the arms' R@1. The quantity of interest is how much of `A - C`
survives when only the background is reverted:

```
survival  =  (dB - dC) / (dA - dC)
```

| Outcome | Reading |
|---|---|
| `survival >= 0.70` | The background is **not** the driver. The geometry correction is real. `S-5` stands |
| `survival <= 0.30` | The Engineer measured the **background**. The geometry claim is unsupported by this evidence and `S-5` must be re-argued |
| `0.30 < survival < 0.70` | **Both contribute.** Reported as two effects, never merged into one PASS |

**Additional pre-committed conditions:**

- If arm **C** does not land within 5 points of the Engineer's reported 83.2%, the harness
  disagrees with the Engineer's and **no arm may be interpreted** until that is resolved.
- If arm **A** does not land within 5 points of 95.8%, likewise.
- `matched_cos` and `mismatched_cos` are reported per arm alongside R@1. A change that moves
  both in the same direction is a common-mode effect and is labelled as one.
- `n = 100` is a pre-scout at the USER's instruction. It cannot separate differences smaller
  than a few points, and any conclusion resting on such a difference will be reported as
  **inconclusive at this sample size**, not as a result.

## Boundaries observed

Read-only with respect to the Engineer's work. No file under `metafind/`, `tools/`, or
`data/outputs/` is modified. The arms are produced by patching module constants **in memory**
inside the Reviewer's own harness, which lives in the Reviewer's scratchpad and not in `tools/`.
Rendered PNGs are written to the Reviewer's scratchpad, never to
`data/outputs/renders/` and never to `data/outputs/_render_probe/`.

_Result appended below when the run completes._

---

## R-1 RESULT — measured 2026-08-22, Reviewer, repo `9842d5e`

Harness: the Reviewer's own, in the Reviewer's scratchpad, **not** `tools/`. Module constants
patched in memory; no file under `metafind/`, `tools/` or `data/outputs/` was modified.
`n = 286` — the **entire** overlap pool, i.e. the Engineer's exact population, not a sample of it.
Target: ULIP-2's released `image_feat`. Control: mismatched pairs. Chance R@1 = 0.35%.

| Arm | Configuration | R@1 | R@5 | matched | mismatched | gap |
|---|---|---|---|---|---|---|
| **A** | v3 as committed — corrected orbit + frame, **black**, `xmag 1.20`, 0.5 / 1.5 | **95.8%** | 98.6% | 0.8783 | 0.5377 | 0.3406 |
| **B** | A with the background reverted to **white**, nothing else | **97.2%** | 99.0% | 0.9141 | 0.5452 | **0.3689** |
| **C** | full v2 rebuild — `+Z` orbit, no frame correction, white, `xmag 1.10`, 0.4 / 3.0 | **83.2%** | 92.7% | 0.8371 | 0.5565 | 0.2806 |
| **D** | A with the framing reverted to `xmag 1.10`, nothing else | **96.2%** | 99.0% | 0.8800 | 0.5349 | 0.3451 |

### Gate conditions, both pre-committed, both met

| | pre-committed | measured | |
|---|---|---|---|
| Arm A within 5 pts of the Engineer's 95.8% | 90.8 – 100.0 | **95.8%**, matched 0.8783 vs 0.8782, mismatched 0.5377 vs 0.5378 | ✅ |
| Arm C within 5 pts of the Engineer's 83.2% | 78.2 – 88.2 | **83.2%**, matched 0.8371 vs 0.8371, mismatched 0.5565 vs 0.5565 | ✅ |

**Two independent harnesses agree to four decimal places on both arms.** Arm C additionally
**restores the v2 measurement no committed tool can produce** — rebuilt from the v2 source rather
than read off disk, and it lands on the Engineer's on-disk number exactly.

---

### FINDING R-1.a — the background is NOT what produced the gain

```
FINDING          Reverting v3's background from black to white loses NOTHING of the
                 12.6-point rise. survival = (B - C) / (A - C) = 14.0 / 12.6 = 1.111,
                 against a pre-registered PASS threshold of 0.70.
EVIDENCE         Arms A, B, C above. n = 286, the entire overlap pool. Reproduced at
                 n = 100 with the same sign and magnitude (97.0 / 99.0 / 88.0,
                 survival 1.222).
CLASSIFICATION   OBSERVED DATA
IMPACT           REVIEW REQUEST 1 item 3; success criterion S-5
SEVERITY         NOTE  -- this is a PASS
```

**The Engineer's geometry correction is real.** The alternative hypothesis — that a feature
keying on background colour produced the improvement — is refuted, not merely argued against.

### FINDING R-1.b — the black background is a small REGRESSION, not a correction

```
FINDING          Black scores LOWER than white against ULIP's own image_feat, on every
                 metric, on two independent samples.
                     n=286   A black 95.8%  vs  B white 97.2%   (+1.4 pts)
                             matched 0.8783 vs 0.9141           (+0.036)
                             gap     0.3406 vs 0.3689           (+0.028)
                     n=100   A black 97.0%  vs  B white 99.0%   (+2.0 pts)
                             matched 0.8798 vs 0.9156           (+0.036)
EVIDENCE         Arms A and B differ in `BACKGROUND_RGBA` and in nothing else.
CLASSIFICATION   OBSERVED DATA
IMPACT           renders.py:97; the whole n04 corpus; S-3 ("mean corner luminance = 0")
SEVERITY         MAJOR -- a corpus decision about to be frozen by a 3.3-hour run
```

**The R@1 difference is 4 assets and is small on its own.** The finding does not rest on it:
matched cosine and the matched/mismatched gap move in the same direction, by the same amount,
on both samples. Three metrics, two populations, one sign.

**This is a genuine tension the Reviewer does not resolve.** Under `U-O`, ULIP-2's released
renders are black and that gives the change upstream provenance. But *matching upstream's pixels*
and *maximising agreement with upstream's features* point in opposite directions here, and `S-5`
is the criterion the milestone actually stakes the correction on. **Which one governs is a USER
decision.** The finding stands either way.

### FINDING R-1.c — `ORTHO_HALF_WIDTH = 1.20` is not an improvement either

```
FINDING          Reverting the framing to v2's 1.10 and changing nothing else scores
                 96.2% against the committed 1.20's 95.8%.
EVIDENCE         Arms A and D. n = 286. matched 0.8800 vs 0.8783.
CLASSIFICATION   OBSERVED DATA
IMPACT           renders.py:118; REVIEW REQUEST 1 item 4
SEVERITY         MINOR -- 1 asset on R@1; the cosines are effectively tied
```

The honest reading is **not** "1.10 is better". It is that **`1.20` buys nothing measurable**,
while having been fitted on 8 assets whose per-asset ratio to ULIP spans 1.16–1.83. A parameter
fitted on 8 assets that moves the criterion by one asset is not evidenced by that criterion.

### What the 12.6 points actually came from

Background costs ~1.4 and framing costs ~0.4. **The entire gain, and then some, is the orbit-axis
correction, the 180-degree frame correction, and the exposure change.** Those three are not
separated from each other by this experiment.

### Not established by R-1

- **Exposure is not isolated.** `AMBIENT 0.5 / INTENSITY 1.5` was changed inside arm C only. A
  fifth arm would separate it.
- **The orbit axis and the frame correction are not separated from each other.** Both entered
  through `meshload`; arm C reverts both together.
- **95.8% is agreement with ULIP-2, not with MetaFind.** Unchanged by anything above.
- **The 8 fitting assets were not held out** of the 286. Arms B and D make that largely moot for
  framing and background — both were reverted and nothing was lost — but it still stands for
  exposure.

---

## R-1 EXTENSION — arm E, the configuration actually decided

**Reviewer, 2026-08-22.** The USER decided `U-W` (white background) and then `U-X`
(`ORTHO_HALF_WIDTH` 1.10). **Neither arm B (white + 1.20) nor arm D (black + 1.10) is that
configuration.** A decision verified on a neighbouring configuration is not verified. Arm E was
run for exactly this reason: framing and background can interact — a tighter crop leaves less
background, so their effects are not guaranteed to add.

`n = 286`, same pool, same seed, same frozen ViT-bigG-14, same target.

| Arm | background | `xmag` | R@1 | R@5 | matched | mismatched | gap |
|---|---|---|---|---|---|---|---|
| C | white | 1.10 | 83.2% | 92.7% | 0.8371 | 0.5565 | 0.2806 |
| A | **black** | 1.20 | 95.8% | 98.6% | 0.8783 | 0.5377 | 0.3406 |
| D | **black** | 1.10 | 96.2% | 99.0% | 0.8800 | 0.5349 | 0.3451 |
| B | white | 1.20 | 97.2% | 99.0% | 0.9141 | 0.5452 | 0.3689 |
| **E** | **white** | **1.10** | **97.2%** | **99.3%** | **0.9160** | 0.5426 | **0.3734** |

*(Arm C is white + 1.10 like E, and differs from it in the orbit axis, the frame correction and
the exposure — the three changes that actually carry the gain.)*

### FINDING R-1.d — the decided configuration is verified and is the best of the five

```
FINDING          white + xmag 1.10 scores R@1 97.2%, and is the best arm measured on
                 R@5 (99.3%), matched cosine (0.9160) and matched/mismatched gap
                 (0.3734). No interaction penalty between the two reverts appeared.
EVIDENCE         Arm E, n = 286, the entire overlap pool. Chance R@1 0.35%.
CLASSIFICATION   OBSERVED DATA
IMPACT           S-5; the n04 corpus about to be regenerated
SEVERITY         NOTE -- this is a PASS
```

**Total against the v2 corpus on disk: 83.2% → 97.2%, a 14.0-point rise**, against the Engineer's
originally reported 12.6. The extra 1.4 points are the two reverts.

**What this does NOT say.** E and B tie exactly on R@1 and differ by 0.002 on matched cosine.
`1.10` is **not** shown to be better than `1.20`; `1.20` is shown to buy nothing, which is the
same finding as `R-1.c` and no stronger. The decision rests on not applying `S-5` selectively,
not on a measured advantage for `1.10`.

**Still not isolated:** the exposure change (`AMBIENT 0.4/3.0 → 0.5/1.5`) is bundled inside arm C
together with the orbit axis and the frame correction. Those three carry ~14 points between them
and this experiment does not divide them.

---

## R-2 — the `COLOR_0` recovery reaches 16% of the population it is written for

**Reviewer, 2026-08-22.** Read-only, CPU only. 97 assets that actually declare `COLOR_0`,
drawn from a 2,500-sidecar sample stratified by the OLD `colour_source`. `_colourise()` was
called twice per geometry — once as v2 (`color0=None`), once as v3 — and the transition recorded.

| OLD source | transition | geometries | |
|---|---|---|---|
| `gltf_default` | `gltf_default -> vertex` | **262** | the fix works |
| `gltf_default` | **`flat -> flat`** | **215** | **COLOR_0 discarded** |
| `gltf_default` | `texture -> texture` | 40 | correct — texture outranks it |
| `flat` | **`flat -> flat`** | **1,166** | **COLOR_0 discarded** |
| `flat` | `texture -> texture` | 361 | correct |
| `texture` | `texture -> texture` | 224 | correct |

**Of the 1,643 non-texture geometries that carry `COLOR_0`, 262 receive it and 1,381 do not —
84% are silently skipped.**

### FINDING R-2.a — the code contradicts its own stated ordering

```
FINDING          `_colourise()` states COLOR_0 sits "below texture, above flat". It does
                 not. `to_color()` returning a single RGBA (`ndim == 1`) hits
                 `return _uniform(vc, "flat")` BEFORE the COLOR_0 branch is reached, so
                 flat wins on that path. Only the later `baseColorFactor` path loses to it.
EVIDENCE         pointclouds.py:221-236 (the early return) vs :227-233 (the COLOR_0
                 branch). Measured: 1,381 of 1,643 non-texture COLOR_0 geometries keep
                 `flat`, over 97 assets that declare the attribute.
CLASSIFICATION   OBSERVED IMPLEMENTATION + OBSERVED DATA
IMPACT           n03 for a large part of the ~1,505-asset population F-N03-1 was opened
                 for; the rgb channel the ULIP-2 point tower consumes
SEVERITY         MAJOR
```

**This is not a regression** — v2 produced `flat` for these too, so nothing gets worse. It is an
**incomplete fix whose docstring claims completeness**, and the Engineer's own validation
(n=12 with / n=50 without, `gltf_default` only) sits entirely inside the 262 that work. The 1,381
were never looked at.

**It is nevertheless run-blocking by cost, not by correctness.** Regenerating now bakes in a fix
that reaches 16% of its target; completing it afterwards is another 3.3 hours.

**A question the Reviewer cannot answer and does not decide:** glTF 2.0 defines `COLOR_0` as a
**multiplier on** `baseColorFactor`, not a replacement for it. So "COLOR_0 above flat" may itself
be the wrong contract, and the correct answer may be the product of the two. **Which of the three
— factor, COLOR_0, or their product — is right is a research decision for the USER**, and it
changes what the fix should do, not merely where it is placed.

### FINDING R-2.b — REFUTED: the geometry-name mapping is sound

```
HYPOTHESIS       `color0_by_geometry()` keys on geometry NAME from a SECOND trimesh load
                 (`skip_materials=True`). If that load names or merges geometry
                 differently, a same-length array could be applied to the wrong mesh --
                 silent corruption that `len(color0) == n` cannot catch.
RESULT           REFUTED. 2,268 name/vertex-count comparisons across 97 assets:
                 name_mismatch 0, count_mismatch 0, ok 2,268.
CLASSIFICATION   OBSERVED DATA
SEVERITY         NOTE -- the Engineer's design is sound here
```

### Prevalence, independently reproduced

120 assets sampled per class: `gltf_default` **18.3%** carry `COLOR_0`, `flat` **5.8%**,
`texture` **4.2%** — against the Engineer's 17.0 / 7.5 / 2.0. **The Engineer's estimates hold.**

---

## R-3 — the changed and new tests are NOT vacuous

**Reviewer, 2026-08-22.** The v2 defect was injected back in memory (old `+Z` orbit formula,
`look_at` up `(0,0,1)`) and each test re-run. A test that would have caught the bug must go red.

| Test | v3 code | v2 bug injected |
|---|---|---|
| `test_primary_layout_is_the_ulip2_style_orbit` **(changed)** | PASS | **FAIL** — *"the orbit is not at a single elevation"* |
| `test_a_tall_asset_renders_tall` **(new)** | PASS | **FAIL** — *"a 3:1 upright box rendered at height/width [0.42, 0.57, 0.93, 1.42, 0.71, …]"* |

### FINDING R-3.a — REVIEW REQUEST 1 items 5 and 6 both PASS

```
FINDING          The new test is non-vacuous: it goes red under exactly the defect it
                 was written for, and its failure message reproduces the tumble
                 signature independently. The changed layout test is a CORRECTION and
                 not a rationalisation: it also goes red under the injected bug, which
                 the version it replaced could not do -- that version asserted the bug.
EVIDENCE         Injection run above, both directions, both tests.
CLASSIFICATION   OBSERVED DATA
IMPACT           REVIEW REQUEST 1 items 5 and 6; code-changes.md §5
SEVERITY         NOTE -- PASS
```

`code-changes.md` §5 forbids changing a test to make it pass. **The evidence says this was not
that.** The old assertion `d[:, 2] == d[0, 2]` held *because* of the defect; the replacement
fails when the defect is present. That is the definition of a correction.

**Not covered:** `test_orthographic_size_does_not_change_with_distance` was not injection-tested.
Its change was to the foreground predicate, not to a geometric assertion, and the defect it would
need to catch (a projection setting not reaching the camera) was not reintroduced here.

---

## R-4 — verification of the Engineer's corrections, 2026-08-22 12:25

**The Reviewer's `R-2` / `R-3` measurements finished at `12:24:20`; the Engineer's edits to
`pointclouds.py` and `renders.py` landed at `12:25:02`.** No race: `R-2` and `R-3` describe the
clean tree at `9842d5e`. `R-2` was then **re-measured on the modified tree** — see below.

Verified by execution, not by reading the diff.

| | Item | Method | Result |
|---|---|---|---|
| ✅ | `n03 is_complete()` reads the version | called on **5 real on-disk `.npz`** | `False` ×5 — the stale corpus is now correctly rejected |
| ✅ | `n04 is_complete()` reads the version | called on **5 real on-disk render dirs** | `False` ×5 |
| ✅ | `SAMPLER_VERSION` bumped | `3 → 4`, `!=` not `<` | correct: a newer sampler is also not this one's output |
| ✅ | `FRAME_CORRECTION_ID` reaches an artifact | `pointclouds.py` sidecar now writes `frame_correction` | the promise in `meshload`'s docstring is now true |
| ✅ | `U-W` white background | `BACKGROUND_RGBA == [255,255,255,255]` | applied, and the comment records the decision and its measurement instead of justifying it by upstream |
| ✅ | `U-X` framing | `ORTHO_HALF_WIDTH == 1.10` | applied |
| ✅ | the "elevation SOLVED" claim | `renders.py` | removed; replaced with an explicit `U-03 remains UNKNOWN` and the failed attempt recorded |
| ✅ | no ambiguity from redefining v3 mid-flight | 4,000 render sidecars sampled | **all `renderer_version: 2`**; nothing on disk carries the old meaning of `3` |
| ✅ | test suite | `pytest tests/ -q` | **585 passed** (was 583; two tests added) |

### FINDING R-4.a — `RENDERER_VERSION`'s own comment is now false

```
FINDING          renders.py's version legend still reads
                   "3 = azimuth orbit about the mesh up axis, BLACK BACKGROUND,
                    ULIP-matched framing, frame-corrected mesh"
                 After U-W and U-X, v3's background is WHITE and its framing is
                 xmag 1.10 -- both IDENTICAL to v2. The legend is wrong on two of
                 its three clauses.
EVIDENCE         renders.py version legend, immediately below ORBIT_ELEVATION_DEG,
                 against BACKGROUND_RGBA == [255,255,255,255] and
                 ORTHO_HALF_WIDTH == 1.10 in the same file.
CLASSIFICATION   OBSERVED IMPLEMENTATION
IMPACT           provenance only -- no artifact is wrong. But `renderer_version` is
                 the field a later reader uses to know what a render IS
SEVERITY         MINOR
```

**Same defect class the Engineer had just corrected two lines above.** What actually separates v2
from v3 now is: **the orbit axis, the frame correction, and the exposure.** Background and framing
are back to v2's values. The legend should say that.

### FINDING R-4.b — `R-2` survives the Engineer's changes, unaddressed

Re-measured on the modified tree, 50 assets carrying `COLOR_0`:

```
gltf_default -> vertex     146    applied
flat         -> flat       505    COLOR_0 STILL IGNORED
                                  -> 22% reached  (16% on the larger clean-tree sample)
```

`_colourise()` was not touched. **This is expected, not a criticism** — `R-2` was written after
the Engineer's edits began. It remains the one open item before the regeneration.

---

## R-5 — the `COLOR_0` policy differential: **INCONCLUSIVE at the available overlap**

**Reviewer, 2026-08-22.** USER-authorised (decision 1, option A). Read-only, CPU only.
`PC._colourise` patched in memory; three policies sampled through the **production**
`sample_mesh()` path, compared against ULIP-2's own released clouds for the same uid.

```
P1_current               what the code does today
P2_color0_replaces       what the docstring claims
P3_color0_times_factor   what glTF 2.0 defines
```

### The measurement collapsed on sample size, not on method

```
ULIP clouds with rgb                    4,999
overlap with our GLB corpus               286
of those, carrying COLOR_0                 17
of those, where the policies DIFFER         3    <-- the usable n
```

Per-asset RGB-histogram L1 distance to ULIP's cloud, the 3 discriminating assets:

| uid | P1 | P2 | P3 | winner |
|---|---|---|---|---|
| `557bad085856` | 0.818 | **0.144** | 0.818 | P2 |
| `501dc84286bf` | 1.999 | **1.369** | 1.792 | P2 |
| `8b18b4289ab0` | 1.751 | **1.519** | 1.752 | P2 |

### FINDING R-5.a — 3 / 3 for P2, and n=3 cannot decide this

```
FINDING          On every asset in the ULIP overlap where the policies produce
                 different clouds, P2 (COLOR_0 replaces the material factor) is
                 closest to ULIP's own cloud -- unanimously, and by a large margin
                 on one (0.818 -> 0.144).
                 The population is THREE ASSETS. That is not a result.
EVIDENCE         Table above. Production sample_mesh() path, uid_seed() seeds,
                 ULIP shard 000-009.
CLASSIFICATION   OBSERVED DATA -- and explicitly INCONCLUSIVE
IMPACT           the n03 colour channel for ~505 of 1,643 non-texture COLOR_0
                 geometries
SEVERITY         NOTE -- reported as inconclusive, NOT as evidence for P2
```

**`REVIEW.md`'s own rule applies to the Reviewer too: a measurement must state the population,
because n=3 and n=286 give opposite conclusions.** Three unanimous assets are a hint. They are not
a basis for a corpus-wide preprocessing decision, and this finding must not be cited as one.

**`P3` is additionally NOT TRUSTED.** It matched `P1` to three decimals on two of the three
assets, which the algebra does not obviously predict. The Reviewer's `P3` implementation is
unverified and its numbers should be discarded rather than read. `P1` and `P2` are sound —
`P1` is the production code unmodified, `P2` a single-branch override.

### What would actually settle it

The overlap is the binding constraint, and it is purchasable: ULIP publishes
`ULIP-2/objaverse_lvis` in shards and we hold **one** (`000-009`, 4,999 clouds → 286 overlap →
3 usable). Reaching n≈50 discriminating assets needs roughly **15–20 further shards, ~20 GB**.

**Alternatively the question does not need ULIP at all.** `P3` is what the glTF 2.0 specification
defines — `COLOR_0` multiplies `baseColorFactor` — and conformance to a published specification is
an authority in its own right, not something an empirical match has to ratify.

**Both routes are the USER's. The Reviewer recommends neither from n=3.**

---

## R-6 — `W-5`: is `Q-CATEGORY` the same question `DL-007` already answered?

**Answer: YES, they are the same question — and NO, it is not closed. `DL-007` records a
decision AND records that the evidence audit behind it was never performed.**

`Q-CATEGORY`, `MASTER.md` §5, lists four options:

> *prompt input · cross-check · the value itself · recorded but unused*

`DL-007`, "What remains UNRESOLVED", lists the same four:

> *"`D0-010` has not been researched — its §6–§11 are empty. The choice between **prompt hint /
> hard value / cross-check / record-only** was made by design ratification, **not** by a
> completed evidence audit."*

**Same four options, same decision, one written as open and one as approved.**

### FINDING R-6.a — `Q-CATEGORY`'s status line is misleading, in the direction that matters

```
FINDING          BLOCK.md describes Q-CATEGORY as "no investigation has been done",
                 which reads as "undecided". It IS decided -- DL-007, prompt-anchor
                 with downward refinement, design approved by the USER 2026-08-21,
                 already implemented as PROMPT_VERSION 5.
                 What was never done is the EVIDENCE AUDIT behind that choice, and
                 DL-007 says so itself.
EVIDENCE         DECISION_LEDGER.md DL-007 "What remains UNRESOLVED" vs
                 MASTER.md §5 Q-CATEGORY vs BLOCK.md §6 item 4.
CLASSIFICATION   OBSERVED IMPLEMENTATION (documentation state)
IMPACT           M2 -- BLOCK.md makes W-5 gate the full annotation run
SEVERITY         MAJOR -- the two readings imply opposite next actions
```

**Why the direction matters.** "Undecided" invites someone to decide it and move on. The true
state is worse: **it is decided, implemented, and the most scientifically material change in the
pipeline, with an empty audit behind it.** `DL-007` is explicit that the choice was ratified
rather than evidenced, and it also carries two unresolved sub-questions — whether anchoring merely
substitutes LVIS's errors for Qwen's, and whether `identity_confirmed` detects that or rubber-
stamps the anchor (`IC-1`).

**Recommended wording, for Master — not a decision the Reviewer may take:**

> `Q-CATEGORY` — **decided** by `DL-007` (prompt anchor, downward refinement only), design
> approved 2026-08-21, implemented as `PROMPT_VERSION 5`. **The evidence audit (`D0-010`
> §6–§11) was never performed.** Open: whether that audit must complete before `M2`.

**The one thing this does NOT settle** is whether the audit is required before the full run. That
is `W-6`, and it is the USER's call, not the Reviewer's.

---

## R-7 — the glTF 2.0 primary source, and what `U-AA` actually implies

**Reviewer, 2026-08-22.** USER chose option `B`: decide `COLOR_0` on specification grounds rather
than buy a larger ULIP sample. The Reviewer therefore fetched the **primary source** rather than
relying on recollection.

**Source:** Khronos glTF 2.0 Specification, `specification/2.0/Specification.adoc`, section
*Metallic-Roughness Material*.

> *"In addition to the material properties, if a primitive specifies a vertex color using the
> attribute semantic property `COLOR_0`, then this value acts as an additional **linear
> multiplier** to base color."*

`UPSTREAM FACT.` The Reviewer's earlier paraphrase — *"COLOR_0 multiplies baseColorFactor"* — was
**too narrow**. The specification multiplies it into **base color**, and base color is itself the
product of `baseColorFactor` **and** `baseColorTexture`.

### FINDING R-7.a — the specification reaches the `texture` class too, which the Reviewer's own framing excluded

```
FINDING          Under the specification COLOR_0 multiplies base color, i.e. factor
                 AND texture. The Reviewer presented this decision as affecting the
                 ~505 `flat` geometries. It also reaches the ~625 `texture`
                 geometries that carry COLOR_0 and currently ignore it entirely.
                 The blast radius is roughly 1,130 geometries, not 505.
EVIDENCE         glTF 2.0 Specification, Metallic-Roughness Material, quoted above.
                 Reviewer's own measurement, 97 assets carrying COLOR_0:
                     gltf_default -> vertex   146   unchanged under the spec
                     flat         -> flat     505   CHANGES
                     texture      -> texture  625   CHANGES  <-- newly in scope
CLASSIFICATION   UPSTREAM FACT (the rule) + OBSERVED DATA (the counts)
IMPACT           n03's rgb channel; ~995 texture-class assets corpus-wide that were
                 never suspected of being wrong
SEVERITY         MAJOR -- it materially enlarges a decision already taken
```

**Per class, under the specification:**

| old `colour_source` | base color is | `COLOR_0` effect | changes vs today? |
|---|---|---|---|
| `gltf_default` | the default `[1,1,1,1]` | `COLOR_0 × 1 = COLOR_0` | **no** — already applied to those 146 |
| `flat` | `baseColorFactor` | `COLOR_0 × factor` | **yes** — the 505 |
| `texture` | `texture × factor` | `COLOR_0 × texture × factor` | **yes** — the 625, newly |

**Why this deserves a separate confirmation rather than silent inclusion.** `F-N03-1` was opened
because ~1,505 assets came back **white**. Texture-backed assets were never white and nobody has
complained about them. The specification says they are nonetheless wrong today. **Correct by the
spec, and entirely unvalidated against any artifact** — the ULIP differential cannot check it
either, for the same n=3 reason as `R-5`.

**Not verified, stated so it is not assumed.** The fetched excerpt did **not** contain an explicit
sentence giving `baseColorFactor`'s default when undefined. `pointclouds.py`'s
`GLTF_DEFAULT_BASE_COLOR = 1.0` matches the widely-known default `[1,1,1,1]`, but that default is
**not quoted from the primary source here** and should be confirmed against the schema
(`material.pbrMetallicRoughness.schema.json`) before it is cited as a `UPSTREAM FACT`.

**The Reviewer does not decide the scope.** `U-AA` says "follow the specification"; the
specification says texture is included. Whether the reproduction follows it that far is the
USER's, and it is asked in `HANDOFF.md` rather than assumed either way.

---

## R-8 — what ULIP's official code actually does about point colour: **NOTHING. It has none.**

**Reviewer, 2026-08-22, at the USER's question.** This was chased before `U-AA` is implemented,
because under `U-O` an upstream behaviour would outrank the glTF specification for this project —
we are reproducing ULIP-2's pipeline, not writing a glTF renderer.

### The authority chain, followed to its end

**1. ULIP's own repository has no mesh→point-cloud code at all.** `OBSERVED IMPLEMENTATION`,
`/home/kyzen/upstream/ULIP` @ `95d480fe`, swept for `COLOR_0`, `vertex_color`, `trimesh`,
`pyrender`, `sample_surface`, `.glb`, `gltf`: **zero hits outside vendored PointNeXt comments.**
The only point-cloud I/O is `utils/io.py`, which *reads* an existing file
(`np.load`, `open3d.io.read_point_cloud`). This re-confirms `FIND-3` from the other direction.

**2. ULIP-2's paper says where the preprocessing came from.** `PAPER FACT`,
`ulip2_source/appendix.tex:10`:

> *"In order to fairly compare to OpenShape on Objaverse-LVIS benchmark, which utilizes 10k
> colored point clouds as the 3D input, **we adopt the same 3D input preprocessing as in
> OpenShape**."*

**So the question is not "what does ULIP do" — ULIP delegates it to OpenShape.**

**3. OpenShape does not publish it either.** `UPSTREAM FACT`, `github.com/Colin97/OpenShape_code`:
the repository ships `download_data.py` and `render_single_glb.py` and directs users to
**download the pre-made** point clouds. `render_single_glb.py` is a **Blender** script
(`bpy.ops.import_scene.gltf`) that emits colour, normal and depth **images** — it does not sample
surface points and assigns no colour to any point.

### FINDING R-8.a — `U-O` cannot resolve `COLOR_0`. The chain ends in a published artifact, not a published procedure

```
FINDING          No source in the authority chain states how the point colours were
                 produced. MetaFind is silent; ULIP publishes no converter and
                 defers to OpenShape; OpenShape publishes no converter either and
                 ships the finished clouds.
EVIDENCE         ULIP repo sweep (zero hits) · ulip2_source/appendix.tex:10 ·
                 OpenShape_code repository contents
CLASSIFICATION   UPSTREAM FACT (that the procedure is unpublished), not an inference
IMPACT           U-AA's justification: the glTF specification is the best available
                 authority BECAUSE upstream has none, not merely because it is a
                 specification
SEVERITY         MAJOR -- it changes how U-AA must be written up
```

**This is `FIND-8` repeating one layer down.** There it was *"upstream ships pixels, not code"*.
Here it is **"upstream ships clouds, not the sampler"**.

**Consequence for the write-up, and it is binding.** Whatever `U-AA` implements must be recorded
as an **`IMPLEMENTATION CHOICE` conforming to the glTF 2.0 specification**. It may **never** be
described as *"what ULIP-2 did"*, *"upstream-faithful"*, or as an `UPSTREAM FACT`. **Nobody knows
what ULIP-2 did.** The released clouds are the only trace, and `R-5` showed they discriminate on
**3 assets**.

### INFERENCE, offered as a hypothesis and explicitly not established

OpenShape's one published mesh tool renders **colour + depth** through Blender, whose glTF
importer builds the full material graph. **If** the clouds were produced by unprojecting those
renders, the point colours are *rendered pixels* — which would already carry
`COLOR_0 × texture × factor` **and** lighting. That would point the same way as full
specification conformance.

**It is a hypothesis. It is not evidence, it was not tested, and it must not be cited as support
for `U-AA`.** Testing it needs the larger ULIP sample the USER declined in `U-AA`.

### FINDING R-8.b — upstream's own ablation says point colour is close to immaterial

`PAPER FACT`, `ulip2_source/appendix.tex`, Table *Ablation on 3D Input*, Point-BERT + ULIP-2
zero-shot on Objaverse-LVIS:

```
 8k xyz      top-1 48.9   top-5 77.1
10k xyzrgb   top-1 50.6   top-5 79.1
```

> *"ULIP-2 maintains strong performance on Objaverse-LVIS zero-shot classification tasks, **even
> without using color information**."*

**Dropping colour entirely costs 1.7 top-1 points.** `COLOR_0` affects ~1,130 geometries of a
46,052-asset corpus, so the effect available to this whole question is a **fraction** of that 1.7.

**This does not make the question unimportant** — a silently discarded input is still a defect,
and `F-N03-1` began with assets that came back pure white. **It does bound the stakes**, and it is
the strongest available argument against spending 20 GB and further review cycles on it.

*(The two rows differ in point count as well as colour, `8k` against `10k`, so 1.7 is not a clean
colour-only contrast. Upstream's own sentence, not the arithmetic, is what carries the claim.)*

---

## R-9 — **CORRECTION.** The Reviewer put the authority in the wrong order

**Reviewer, 2026-08-22, at the USER's objection: 「你架構就是 ulip 你就參考啊」.**
**The USER is right and `R-7`/`R-8`'s framing was wrong. Recorded, not quietly amended.**

### What the Reviewer got wrong

`R-8` established that no source publishes the cloud-colouring procedure, and concluded that the
**glTF 2.0 specification** is therefore the best available authority. **That conclusion does not
follow, and it inverted the actual criterion.**

**Our point encoder is a FROZEN ULIP-2 checkpoint.** `OBSERVED IMPLEMENTATION` —
`stage1_encoding_protocol.json` records the CLIP scope as `frozen`, and the ULIP-2 checkpoint
supplies all 228 `point_encoder` / `pc_projection` tensors. That encoder was trained on clouds
produced by OpenShape's unpublished process, whatever it was.

```
the question is NOT   "which colouring is correct by specification?"
the question IS       "which colouring feeds the frozen encoder the input
                       distribution it was actually trained on?"
```

**For a frozen encoder, being right by specification while differing from the training
distribution is not better. It is worse.** Specification conformance is a virtue for a renderer.
This is not a renderer — it is an input pipeline for somebody else's frozen weights.

### What that does to the evidence already collected

`R-5`'s three discriminating assets were dismissed as too weak to matter against a specification.
**Under the corrected criterion they are the only evidence of the right kind**, and they do not
point where `U-AA` sends us:

| uid | P1 current | **P2 COLOR_0 replaces** | P3 glTF product |
|---|---|---|---|
| `557bad085856` | 0.818 | **0.144** | 0.818 |
| `501dc84286bf` | 1.999 | **1.369** | 1.792 |
| `8b18b4289ab0` | 1.751 | **1.519** | 1.752 |

**3 / 3 for `P2` — which is exactly what the Engineer's docstring claimed all along.** `P3`, the
specification answer `U-AA` selects, wins nothing.

`R-8.a` still stands as a fact: the procedure is unpublished. **What changes is the conclusion
drawn from it.** The right response to "upstream publishes no procedure" is not "substitute a
specification" — it is **"match upstream's artifact, which we hold."**

### FINDING R-9.a — `U-AA`'s premise should be revisited by the USER

```
FINDING          U-AA chose the glTF specification over buying a larger ULIP sample.
                 That trade was presented by the Reviewer with the specification framed
                 as an authority for THIS pipeline. It is not: the encoder is ULIP-2's
                 and frozen, so ULIP-2's own clouds define the target distribution.
                 The evidence that exists points to P2, not to the specification's P3.
EVIDENCE         R-5 table above (n=3) · stage1_encoding_protocol.json frozen scope ·
                 ULIP-2 checkpoint contains only point_encoder/pc_projection tensors
CLASSIFICATION   the criterion is INFERENCE; the 3/3 result is OBSERVED DATA, n=3
IMPACT           U-AA, U-AC, and the colour of the whole regenerated corpus
SEVERITY         MAJOR -- it reverses the Reviewer's own recommendation
```

### The sample is purchasable, and cheaper than the Reviewer previously said

`SFXX/ulip` `ULIP-2/objaverse_lvis` ships **~1.16 GB shards** (`000-000` … at least `000-049`).
We hold **one**. Measured: one shard = 4,999 clouds → **286** overlapping our corpus → **17**
carrying `COLOR_0` → **3** discriminating.

```
+10 shards   ~12 GB   ->  ~2,900 overlap   ~170 with COLOR_0   ~30 discriminating
```

**~12 GB buys n≈30.** The earlier "~20 GB" figure was an estimate; this one is measured from the
shard we hold. Disk is not a constraint — 3.0 TB free.

### And the metric should change too

`R-5` compared RGB histograms. **The project already has a better instrument and has used it
twice** — `FIND-7` and `S-5`: push the cloud through the **frozen ULIP-2 point encoder** and score
against ULIP's released features. That measures the thing that actually matters — what the
encoder sees — rather than a colour statistic chosen by the Reviewer.

**Recommended, and it is a recommendation, not a decision:** fetch ~10 shards, then run the three
policies through the frozen encoder at n≈30. That is the same harness pattern this block has
already trusted twice.

---

## R-10 — `COLOR_0` settled against ULIP's own clouds, with the frozen ULIP-2 encoder. **n = 130**

**Reviewer, 2026-08-22.** USER-authorised after `R-9`. This replaces `R-5` entirely: right
instrument, right reference, usable sample.

### How the sample was raised from 3 to 130

`R-5` was limited by holding **one** ULIP shard. `SFXX/ulip` `ULIP-2/objaverse_lvis` ships
**161 shards** at ~1.16 GB. Ten more were fetched and filtered to our manifest on the fly — only
uids in `lvis.json` were kept, so the footprint is 1.1 GB rather than 14 GB.

```
shard 000-000  kept 1,228      000-005    292      (overlap per shard varies widely;
      000-001       307        000-006    266       000-009's 286 was not typical)
      000-002       247        000-007    273
      000-003       289        000-008    276
      000-004       274        000-010    254

ULIP clouds for uids in our corpus     3,706
of those carrying COLOR_0                229
DISCRIMINATING (policies differ)         130    <-- n, was 3
```

### The instrument — the one this block already trusts twice

Not a colour statistic. Each cloud is pushed through the **frozen ULIP-2 point encoder** and
scored by cosine against the embedding of **ULIP's own released cloud for the same uid**, encoded
by the same weights. Same pattern as `FIND-7` and `S-5`. It measures what the encoder actually
sees, which is the only thing that matters for a frozen encoder.

```
cosine to ULIP's OWN cloud, frozen ULIP-2 point encoder, n = 130

  P1_current  (COLOR_0 discarded)     mean 0.8800   median 0.8845   wins  27
  P2_color0_replaces                  mean 0.9043   median 0.9204   wins  54
  P3_gltf_product                     mean 0.9004   median 0.9195   wins  49
```

### FINDING R-10.a — doing nothing is definitively wrong. `R-2` is settled by evidence

```
FINDING          The current behaviour -- discarding COLOR_0 wherever the material
                 path returns a flat colour first -- is the WORST of the three by a
                 clear margin. Both repairs beat it: mean cosine +0.024 / +0.020,
                 and it wins on 27 of 130 assets against the other two's 103.
EVIDENCE         Table above. n = 130 discriminating assets, frozen ULIP-2 point
                 encoder, target = ULIP's own released cloud for the same uid.
CLASSIFICATION   OBSERVED DATA, measured against an upstream artifact
IMPACT           n03's rgb channel; ~1,130 geometries; the frozen point tower's input
SEVERITY         MAJOR -- and now decided by evidence rather than by argument
```

**`R-2` is no longer a contract-versus-docstring dispute.** The incomplete fix measurably feeds
the frozen encoder a worse input than either completion. It must be completed before the corpus
is regenerated.

### FINDING R-10.b — `P2` and `P3` are NOT separable at this sample size

```
FINDING          P2 leads P3 by 0.004 mean cosine and 5 wins out of 130. That is
                 inside the noise: 130 fair coin flips have a standard deviation of
                 ~5.7 wins, so a 54/49 split is well under one sigma.
                 NO paired significance test was computed. The two are reported as
                 TIED, and P2 must not be called the winner.
EVIDENCE         Table above.
CLASSIFICATION   OBSERVED DATA -- explicitly inconclusive between P2 and P3
SEVERITY         NOTE
```

### What this means for `U-AA`, and it rehabilitates it

`R-9` withdrew the Reviewer's "follow the specification" recommendation because a specification
must not **override** upstream evidence. **It does not follow that a specification may never
break a tie.** Upstream's own artifact has now been asked, at n=130, and it **cannot separate
`P2` from `P3`**.

```
where upstream evidence discriminates   ->  upstream wins.  It did: P1 is out.
where upstream evidence is silent       ->  glTF 2.0 is a legitimate tie-breaker.
```

**`U-AA` therefore stands, for a reason it did not have when it was taken.** `P3` also handles the
`texture` class coherently — `COLOR_0 x texture x factor` — which `P2` has to special-case.

**Recommendation: `P3`.** Recorded as an `IMPLEMENTATION CHOICE` conforming to glTF 2.0, chosen as
a tie-break **after** upstream's artifact was consulted and found not to discriminate. Per `R-8`
it may still never be described as *"what ULIP-2 did"*.

### Limits, stated

- 130 assets, drawn from 11 of 161 shards. Not a random sample of the corpus — it is whatever
  those shards overlap.
- No paired significance test; the `P2`/`P3` tie is asserted from win counts and means only.
- The `texture` class is inside `P3` by construction but was **not** separately validated: the
  measurement mixes all classes.
- `P1`'s defeat is robust to all of the above. The `P2`/`P3` tie is the part that is thin.

---

## R-11 — **CORRECTION.** `R-8`'s blanket prohibition was wrong and is withdrawn

**Reviewer, 2026-08-22, at the USER's objection: 「他是你參考的架構那必須知道啊…除非你有特別做什麼設定」.**
**The USER is right. Recorded, not quietly amended.**

`R-8` and `R-10` both ended with: *"it may never be described as what ULIP-2 did."* Applied as a
blanket rule that is **wrong**, and it is the mirror image of over-claiming: refusing to state a
result the evidence supports is as much a reporting failure as stating one it does not.

### The distinction `R-8` collapsed

| Claim | Status |
|---|---|
| *"we run the same procedure as ULIP-2"* | **NOT claimable.** Unpublished — `R-8.a` stands |
| *"our clouds agree with ULIP-2's released clouds"* | **CLAIMABLE, and measured** — `R-10` |

**`R-10` is exactly that measurement**, and `R-8`'s rule would have suppressed it:

```
cosine to ULIP-2's own cloud, through ULIP-2's own frozen encoder
n = 130 assets where the COLOR_0 policies differ
    mean 0.90   median 0.92
```

**Procedure identity is unknown. Artifact agreement is measured.** They are different claims and
only the first is barred.

### FINDING R-11.a — the default direction was inverted

```
FINDING          ULIP-2 is this project's reference architecture: the point encoder
                 IS its frozen checkpoint. The default expectation is therefore
                 AGREEMENT with ULIP-2, and every DIVERGENCE is what must be
                 marked, registered and justified.
                 R-8 inverted it into "agreement may not be asserted", which would
                 leave the reproduction unable to report its own positive results.
EVIDENCE         USER instruction 2026-08-22 · R-10 (n=130, cosine 0.90/0.92) ·
                 CONTEXT.md authority order, upstream implementations at rank 3
CLASSIFICATION   the rule is a USER DECISION; the 0.90/0.92 is OBSERVED DATA
IMPACT           how every ULIP-2-facing result in this block is written up,
                 including Table 1
SEVERITY         MAJOR -- it changes the reporting posture of the whole block
```

### The rule that replaces it

```
DEFAULT      agree with ULIP-2. State the agreement, with its metric and its n.
DIVERGENCE   only where we deliberately chose differently -- THAT is the deviation,
             and it carries the registry entry.
BARRED       claiming a shared PROCEDURE where none is published.
```

### Applied to what is already on record

| Item | Correct wording |
|---|---|
| `COLOR_0` / `P3` | *"an implementation choice conforming to glTF 2.0, whose output agrees with ULIP-2's released clouds at cosine 0.90 / median 0.92, n=130, under ULIP-2's own frozen encoder."* **Not a deviation** — it moves toward upstream |
| **white background (`U-W`)** | **This one IS a real deviation.** ULIP renders black, we deliberately render white, and it is registered under `U-Z`. Divergence by choice, correctly marked |
| point clouds, frame | `FIND-7` already measured agreement — R@1 98.0% against ULIP's own `image_feat`. **State it** |
| renders, camera | `S-5` 97.2% against the same target, `R-1` arm E. **State it** |

**`R-8.a` survives unchanged as a fact** — the procedure is unpublished, and no claim of shared
*method* may be made. Everything `R-8` and `R-10` said about *forbidding claims of agreement* is
withdrawn.

---

## R-12 — the two safety checks owed on `P3`, at commit `53f0b99`

**Reviewer, 2026-08-22.** Baseline produced by patching `meshload.color0_by_geometry` to `{}`,
so the control runs the **production** code with no `COLOR_0` present anywhere. Nothing inside
`_colourise` was touched.

### B — control: PASS

Assets carrying **no** `COLOR_0` must be unchanged by the `P3` work.

```
n = 60 gltf_default assets without COLOR_0
max |delta| over 10,000 points x 6 channels  =  0.0000000000
```

**Bit-identical.** Independently reproduces the Engineer's `n=9` result at `n=60`. **PASS.**

### A — texture class: one clear signal, one ambiguous one

Every `texture`-class uid carrying `COLOR_0` that ULIP also holds — the full overlap, not a
sub-sample.

```
n = 37

luminance (P3 - COLOR_0 off)   mean -0.2076   median -0.1821   DARKER on 37/37
cosine to ULIP's own cloud     P3 0.8980   COLOR_0 off 0.9005   P3 better on 16/37
```

### FINDING R-12.a — `P3` darkens every texture-class asset it touches

```
FINDING          Modulating a texture-sampled colour by COLOR_0 lowers mean
                 luminance on 37 of 37 assets, by 0.21 on a 0-1 scale.
EVIDENCE         Table above, full ULIP overlap for the class.
CLASSIFICATION   OBSERVED DATA
IMPACT           ~995 texture-class assets corpus-wide
SEVERITY         MAJOR
```

**Unanimous and large.** This is the signature the safety check existed to look for.

### FINDING R-12.b — whether that darkening is *wrong* is NOT established

```
FINDING          P3 moves the texture class 0.0025 further from ULIP's own clouds
                 and wins on 16 of 37. 37 fair coin flips give 18.5 +/- 3, so
                 16/37 is inside the noise. The mean favours leaving COLOR_0 off;
                 the win count does not separate them.
CLASSIFICATION   OBSERVED DATA -- explicitly inconclusive
SEVERITY         NOTE
```

**The darkening is certain. "Therefore worse" is not.**

### What it does to `R-10`

`R-10` measured all three classes **mixed** at `n=130` and found `P2 ≈ P3`. `R-12` isolates the
texture class and finds `P3` behind there. **A tie across a mixture is consistent with two
opposite effects cancelling**, and `P2` — which leaves the texture class alone by construction —
is exactly the arm that avoids the negative half.

### Which arm is closer to ULIP's artifact — the only answerable form of "what did ULIP do"

`R-8.a` stands: the procedure is unpublished, so *"ULIP's official method"* has no answer.
**"Which of ours lands closer to ULIP's clouds"** does, and it has now been asked twice:

| measurement | population | closer to ULIP |
|---|---|---|
| `R-10`, all classes | n = 130 | **`P2` 0.9043** vs `P3` 0.9004 |
| `R-12A`, texture only | n = 37 | **COLOR_0 off 0.9005** vs `P3` 0.8980 |

**`P3` does not win either one.** Both margins are small and neither is significance-tested, but
**there is no measurement in this block in which full glTF conformance is closer to ULIP-2's own
clouds than leaving the texture class alone.**

**Reviewer recommends `A` (= `P2`): modulate `flat` and `gltf_default`, leave `texture`
untouched.** Recorded as an `IMPLEMENTATION CHOICE` that **deliberately departs from glTF 2.0 for
the texture class**, because ULIP-2's artifact is the higher authority for this pipeline (`R-9`,
`R-11`) and it does not support the spec-conforming variant. **That departure needs a registry
id** — a fifth, and it is the Integrator's under `U-Z`.

---

## R-13 — release checks on the `R-12` narrowing. **PASS.**

**Reviewer, 2026-08-22.** Working tree at `58637f3`, clean, `SAMPLER_VERSION = 6`.
Baseline produced by patching `meshload.color0_by_geometry` to `{}` — the production code with no
`COLOR_0` present. Nothing inside `_colourise` was touched.

| # | Condition | n | Result |
|---|---|---|---|
| 1 | `texture` **with** `COLOR_0` → must now be identical to `COLOR_0` off | **37** | `max abs delta = 0.0000000000`, changed 0/37 — **PASS** |
| 2 | control, **no** `COLOR_0` → must be identical | **60** | `max abs delta = 0.0000000000` — **PASS** |
| 3 | `flat` **with** `COLOR_0` → must **still** be modulated | 60 | 52/60 changed — **PASS**, see below |
| 4 | `gltf_default` **with** `COLOR_0` → still modulated | 60 | 59/60 changed — **PASS**, see below |

Condition 1 is the one that matters and it is measured on the **full** ULIP-overlapping texture
population, not a sample. The Engineer's own check was 12 assets; this is 37 + 60.

### The Reviewer's checks 3 and 4 first reported FAIL. **The criterion was wrong, not the code.**

The script required **every** asset to change. `52/60` and `59/60` therefore tripped it. Diagnosed
rather than assumed:

**a. The commit's code diff is confined to the texture branch.** `_commit()` gained
`modulate: bool = True`; only `return _commit(vc, "texture", modulate=False)` passes it. The
`flat` and `gltf_default` paths are unchanged.

**b. The unchanged assets are explained, and the explanation is correct behaviour.** Three of the
eight were traced per geometry:

```
04f51877ba25   graph geoms 31   names found in COLOR_0 31/31
                 texture base, modulated=False   x26
                 flat    base, modulated=True    x5
258373b08ba7   7 geoms, 7/7 matched     texture x6 (not modulated) · flat x1 (modulated)
c6968ff55e24   4 geoms, 4/4 matched     texture x2 (not modulated) · flat x2 (modulated)
```

The sidecar records the **worst** source across an asset's parts, so an asset labelled `flat`
can be mostly `texture`-backed. Its `COLOR_0` lives on the texture parts, which `R-12` correctly
declines to modulate; the few genuinely `flat` parts either carry neutral `COLOR_0` (measured
`min = max = 255`) or receive too little area-weighted sampling to move the cloud.

**c. Geometry-name matching between the two loads is perfect** — 31/31, 7/7, 4/4. The silent
mis-keying risk raised in `R-2.b` is refuted a second time, now on `flat`-class assets.

**Conditions 3 and 4 are satisfied.** The narrowing did not over-reach.

### FINDING R-13.a — the new test is NOT vacuous

```
FINDING          test_texture_bases_are_not_modulated_by_color0 goes RED when the
                 defect is injected back -- _colourise wrapped so a texture base is
                 multiplied by COLOR_0 again.
                 Injected result: FAIL, "COLOR_0 was applied to a texture base;
                 R-12 withdrew it from that class after measuring 37/37 assets darker."
                 Baseline: 1 passed.
CLASSIFICATION   OBSERVED DATA
SEVERITY         NOTE -- PASS
```

The failure message carries its own justification, so a future reader who "fixes" the carve-out
back to spec conformance is told why it exists. That is the right shape for a test pinning a
deliberate deviation.

### FINDING R-13.b — the R-12 code is not in the commit that claims it. **Provenance defect.**

```
FINDING          58637f3 -- "WIP UNACCEPTED: R-12 -- COLOR_0 withdrawn from the
                 texture class" -- changes ONE file: workflow/blocks/ULIP2/HANDOFF.md.
                 It contains no code.
                 The actual R-12 change -- SAMPLER_VERSION 5->6, the `modulate` flag,
                 the texture carve-out, and the new test -- is in 4e5053f, whose
                 message is "docs: DL-010 -- upstream is a source, not a forbidden
                 zone", and which also carries CONTEXT.md, DECISION_LEDGER.md and
                 docs/graph/README.md.
EVIDENCE         git show 58637f3 --stat  -> 1 file, HANDOFF.md, +96
                 git log -S"modulate=False" -- metafind/data/pointclouds.py -> 4e5053f
                 git show 4e5053f --stat -> pointclouds.py +57, test_pointclouds.py +56,
                 CONTEXT.md +27, DECISION_LEDGER.md +19, docs/graph/README.md
CLASSIFICATION   OBSERVED IMPLEMENTATION
IMPACT           experiment provenance -- .claude/rules/experiments.md §7 (code state)
                 and §10; code-changes.md §13 (diff discipline)
SEVERITY         MAJOR -- provenance only. No behavioural defect; the behaviour is
                 verified correct above
```

**Why it matters even though nothing is broken.** A later reader looking for when the texture
carve-out landed will open `58637f3`, find only documentation, and skip `4e5053f` because it is
labelled `docs:`. A research-critical dataset-semantics change is filed under a documentation
message, **bundled with three of Master's governance files.** `experiments.md` §7 requires results
to be attributable to the code state that produced them, and `code-changes.md` §13 requires the
diff to be inspectable.

**Recommended remedy, and it is a recommendation only:** record the correct commit id in
`HANDOFF.md` and in `SPEC_M1`, and keep code and governance-document changes in separate commits
from here on. **The Reviewer does not rewrite history and does not decide this.**

### VERDICT

**The `COLOR_0` work is cleared for the regeneration.** All four release conditions met, the new
test is non-vacuous, and the only open item is the provenance record in `R-13.b`, which does not
affect the artifacts the run will produce.

---

## R-14 — the Engineer's diagnosis of `R-13.b` is correct, and it reaches the Reviewer's own file

**Reviewer, 2026-08-22.** The Engineer reported that `R-13.b` is not a mislabelled commit but
**cross-session contamination via `git add -A` on a shared tree**. Verified independently rather
than accepted.

```
git show 4e5053f --stat

  docs/graph/README.md                 +2      Master's
  metafind/data/pointclouds.py        +57      the ENGINEER's R-12 code
  tests/test_pointclouds.py           +56      the ENGINEER's new test
  workflow/CONTEXT.md                 +27      MASTER's
  workflow/DECISION_LEDGER.md         +19      MASTER's
  workflow/blocks/ULIP2/REVIEW.md     +84      *** THE REVIEWER'S OWN FILE ***
  workflow/roles/ESSGNN_ENGINEER.md    +8      MASTER's
  workflow/roles/README.md            +90      MASTER's
```

**Three different roles' work is inside one commit whose message describes only one of them.**
`git log --format="%an <%ae>"` over the last six commits returns
`Kyzen5128 <legend2341528@gmail.com>` **six times** — git cannot attribute a role, so nothing in
the history distinguishes who wrote what.

### FINDING R-14.a — the Reviewer's own audit trail is not attributable either

```
FINDING          84 lines of REVIEW.md -- the Reviewer's findings, measurements and
                 populations -- were committed by a session that did not write them,
                 under a message about something else. The same mechanism that
                 misfiled the Engineer's code misfiled the Reviewer's evidence.
EVIDENCE         git show 4e5053f --stat, line 6
CLASSIFICATION   OBSERVED IMPLEMENTATION
IMPACT           experiments.md §7 code state · §10 artifact provenance ·
                 BLOCKS.md role separation. It affects every role, not one
SEVERITY         MAJOR -- process, not behaviour. No artifact is wrong
```

**The Engineer is right that "separate code from governance commits" does not fix this.** The
mechanism is `git add -A` against a tree three roles share; the failure is bidirectional and
silent in both directions.

**The Reviewer adopts explicit-path commits from now on, without waiting for a ruling** — same
position the Engineer took, same reason: it costs nothing and removes half the failure.
**Option 2 (a distinct git identity per role) is the only one of the three that makes the history
*attributable after the fact*;** the other two only prevent recurrence. That is Master's and the
USER's to decide, not the Reviewer's.

**The commit-to-content map the Engineer recorded at `c9ef702` was spot-checked and matches** what
`git log -S` returns for `modulate=False` and `SAMPLER_VERSION`. History is not rewritten; the map
is the remedy.

---

## R-15 — on archiving the v2 corpus before the regeneration

**The Engineer asks the USER to decide. The Reviewer's input, with one thing the proposal misses.**

### Measured

```
data/outputs/pointclouds   5.6 GB
data/outputs/renders       7.3 GB          total 12.9 GB, against 3.0 TB free
```

**`data/outputs` and `data/outputs/renders` are on the same device (`st_dev 2049`).** An archive
performed with `mv` is therefore a **rename**, not a copy: no bytes move, and **the SMR
small-file penalty in `CONTEXT.md` §6 does not apply.** The cost is effectively zero, which
removes the only argument against doing it.

### What the proposal misses — the two index files must move with the corpus

```
data/outputs/logs/pointclouds_index.jsonl    29 MB
data/outputs/logs/renders_index.jsonl       105 MB

renders_index.jsonl stores ABSOLUTE paths:
  "/home/kyzen/MetaFindV1/data/outputs/renders/000074a3.../view_00.png"
```

Move the corpus and leave the indexes, and every path in them dereferences into a directory that
no longer holds those files. `n03`/`n04` rebuild them, so the end state is fine — **but the window
between the move and the rebuild holds a v3 corpus described by a v2 index, and an interrupted run
freezes that state.** `n05` and `n06` both read `renders_index.jsonl`, and `annotate_run.py`
builds its entire work list from it.

**Archive the two index files alongside the two directories, or not at all.**

### Reviewer's recommendation

**Archive.** Not because the v2 corpus is irreplaceable — it is reproducible from source, and
`R-1` arm C did exactly that, landing on 83.2% to four decimals. Because:

1. Reproducing it costs a GPU run and depends on the code history staying readable, and `R-14`
   has just shown that history is **less trustworthy than assumed**.
2. The cost is a rename. There is nothing on the other side of the trade.

**Suggested, not performed — the Reviewer does not move the USER's data:**

```
data/outputs/_v2_archive/
    pointclouds/                     (mv)
    renders/                         (mv)
    logs/pointclouds_index.jsonl     (mv)
    logs/renders_index.jsonl         (mv)
    PROVENANCE.md                    produced by 58637f3's PARENT tree; the
                                     renderer-v2 / sampler-v3 corpus; R-1 arm C
                                     scored it at R@1 83.2% vs ULIP image_feat
```

Making it read-only afterwards would stop anything writing into it. **That is a destructive-class
operation on the USER's data and the Reviewer neither performs nor decides it.**

**This is a `code-changes.md` §9 data-safety decision and it belongs to the USER.**

---

## R-16 — USER decision `U-AE`: **no archive. The v2 corpus is not kept.**

**USER, 2026-08-22: 「若需要重作 舊的就全部刪掉 不要留」.** This overrides the Reviewer's `R-15`
recommendation. Recorded, not re-argued.

### The decision is defensible on this block's own evidence

`code-changes.md` §9 requires that a destructive operation identify **whether the data is
reproducible**. It is:

```
R-1 arm C rebuilt the renderer-v2 corpus from source -- old +Z orbit formula,
look_at up (0,0,1), no frame correction, white, xmag 1.10, ambient 0.4 / int 3.0 --
and scored it against ULIP's image_feat at

    R@1 83.2%   matched 0.8371   mismatched 0.5565   n = 286

against the Engineer's on-disk v2 measurement of

    R@1 83.2%   matched 0.8371   mismatched 0.5565
```

**Four decimal places, from source, without the corpus.** The v2 artifacts are a cache of
something the repository can regenerate, not a unique observation. **Keeping the corpus is not
what preserves the ability to reproduce it — the code history is.** That places the weight on
`R-14`'s attribution problem, which is being fixed separately.

### Two things that must still be cleared, or "deleted" is not what happens

**1. The two index files.** `R-15`'s hazard is unchanged by the decision not to archive: leave
them and a v3 corpus is described by a v2 index for the whole run, and an interruption freezes
that state. `n05`'s entire work list is built from `renders_index.jsonl`
(`annotate_run.py:485-502`). **They must be removed or rebuilt as part of the regeneration, not
left to be overwritten at the end.**

**2. The 90 quarantined render directories — unmarked v2 residue.** `OBSERVED DATA`:

```
render directories            46,045
with a sidecar / in the index 45,955
without either                    90   -- 11 blank PNGs each,
                                         quarantine reason "every view is blank"
```

**Those 90 carry no sidecar, so they carry no `renderer_version`.** `n04` will re-attempt them,
fail the same way, and re-quarantine — and their **renderer-v2 PNG bytes stay on disk, inside the
regenerated corpus, with nothing marking them as v2.** Every other stale artifact is now
distinguishable by a version field; these are the one class that is not.

The images are blank either way, so the scientific impact is nil. **But `U-AE` says nothing old is
kept, and this is the only thing that would survive unmarked.** Flagged for the Engineer to
remove; the Reviewer does not delete the USER's data.

### Reviewer's position

**No objection to `U-AE`.** The evidence supports it, and it was raised once and answered. The
regeneration is cleared to proceed on the conditions above.

---

## R-17 — audit of the finished `n03` corpus. All 46,052 sidecars, no sampling.

**Reviewer, 2026-08-22, while `n04` is still running.** Storage verified first: the five
`data/outputs/*` entries are symlinks into `/home/kyzen/metafind_out`, which is
`/dev/nvme0n1p2` — **`ROTA=0`, a genuine NVMe**, 953.9 GB with **816 GB free**. Capacity is not a
risk for this run or for `n06`'s embeddings.

| Field | Value | |
|---|---|---|
| `sampler_version` | `6` on all 46,052 | ✅ uniform |
| `frame_correction` | `yaw180_about_y@ulip2_frame` on all 46,052 | ✅ **`meshload`'s docstring is finally true** |
| `n_points` | `10000` on all | ✅ |
| `rgb_scale` | `unit` on all | ✅ |
| `coloured_point_fraction` | `1.000000` on all 46,052 | ✅ |
| `max_radius` | min = max = `1.000000` | ✅ |
| `centroid_offset` | max `9.869e-09` | ✅ tighter than the old `1.35e-05` |
| `seed` | 46,052 distinct, 0 missing | ✅ individually reproducible |

### `colour_source` did not shift, and the carve-out holds at corpus scale

```
texture       23,675      flat  13,524      gltf_default  8,853
```

**Identical to the pre-correction distribution**, so the base classification did not move.

```
color0_modulated = True
    gltf_default   1,451
    flat             806
    texture            0        <-- R-12's carve-out, corpus-wide
                   -------
    total          2,257 assets modulated
```

**`texture` is zero on all 23,675.** The `R-13` release check, measured on 37 assets, holds across
the whole corpus.

### FINDING R-17.a — `S-7`'s zero-variance criterion FAILS as written, and the criterion is what is wrong

```
FINDING          S-7 requires "the 21 zero-variance assets stay exactly 21".
                 Measured on the new corpus: 18.
                 This is NOT a geometry change. The criterion pins a floating-point
                 equality the corpus cannot hold stable.
EVIDENCE         Distribution of the minimum per-axis variance over all 46,052:
                     exactly 0.0            18
                     0 < v < 1e-20          66
                     0 < v < 1e-12          88
                 The values immediately above zero cluster at 1e-36 to 1e-33
                 (1.48e-36, 5.77e-35, 5.85e-35, 3.37e-33, 3.85e-33, ...).
CLASSIFICATION   OBSERVED DATA
IMPACT           SPEC_M1 success criterion S-7
SEVERITY         MAJOR -- a milestone criterion that fails on a correct corpus
```

**Why it is numerical.** `FRAME_CORRECTION` is `(x, y, z) -> (-x, y, -z)`. **Negating an axis
leaves that axis's variance unchanged in exact arithmetic**, so the transform cannot alter
per-axis variance by any geometric mechanism. What it does alter is the *arithmetic path*: the
correction is composed into the scene graph's node transforms, so vertex coordinates differ in
their last bits, and three assets sitting at the 1e-33 boundary crossed from exactly `0.0` to
approximately `0.0`.

**84 assets are flat to within double precision.** Which of them lands on exactly `0.0` is not a
stable property of the corpus, and a criterion written on `== 0` will keep failing.

**Recommended rewording — `SPEC_M1`'s, not the Reviewer's:** count assets whose minimum per-axis
variance is below a stated epsilon (`1e-20` gives 84) rather than exactly zero.

**One honest cost of `U-AE`.** The `21` came from `validation_plan.yaml:113` and was reproduced by
the Engineer on the old corpus. That corpus is now deleted, so **`21 -> 18` cannot be verified
asset-by-asset any more** — the explanation above is inference from the value distribution plus the
algebra, not a paired comparison. It is a small cost and it is stated rather than hidden.

### Still outstanding for `n03`

`S-6` — that the 180-degree yaw is gone, measured as Chamfer distance against ULIP's released
clouds — has **not** been re-run on the new corpus. The Reviewer now holds **3,706** ULIP clouds
for uids in this corpus rather than the 286 the original `FIND-6` used, so it can be measured at
roughly 13x the original population once `n04` releases the GPU.

---

## R-18 — the `n04` / `n07b` camera-phase question, settled on the `n04` side

**Reviewer, 2026-08-22.** `ESSGNN REVIEWER` raised a measurement the ULIP2 Reviewer had not made
and asked the `ULIP2 ENGINEER` for two artifacts to close it. **Both are computable from `n04`'s
own code, which is this Reviewer's node, so it is answered here rather than queued.**
Pure numpy, read-only, no GPU — `n04` was still rendering.

### Their claim, independently reproduced

```
d07[k] == R_y(180 deg) @ d04[k],  k = 0..10
max |d07[k] - R @ d04[k]|  =  1.110e-16        CONFIRMED

pre-fix set-to-set: every d07 sits 15.371 deg from the nearest d04
azimuth spacing 360/11 = 32.727 deg      180 / 32.727 = 5.500 steps
```

**Confirmed to the last bit, including the half-step interleave.** `ESSGNN REVIEWER`'s numbers are
correct.

### FINDING R-18.a — the mesh yaw fix resolves the phase offset EXACTLY

```
FINDING          Rotating the mesh by FRAME_CORRECTION is equivalent, in the
                 asset's own frame, to rotating the camera ring by R^T = R.
                 n04's effective in-mesh view directions after the fix are
                 therefore R @ d04, and R @ d04 == d07 to 1.11e-16.
                 The two nodes' camera phases COINCIDE after the correction --
                 not merely come close.
EVIDENCE         max |R@d04[k] - d07[k]| = 1.110e-16 over all 11 views.
                 Azimuth about UP = Y, sorted:
                   d04  16.364  49.091  81.818 ... 343.636
                   d07   0.000  32.727  65.455 ... 327.273
                   eff   0.000  32.727  65.455 ... 327.273   <- identical to d07
                 view_00:  d04[0] = [0, 0.342020143, -0.939692621]
                           R@d04[0] = [0, 0.342020143, +0.939692621]
                           d07[0]   = [0, 0.342020143, +0.939692621]
                           difference = 0.0 exactly
CLASSIFICATION   OBSERVED IMPLEMENTATION (both direction generators) + exact algebra
IMPACT           stage2_protocol.json `image_protocol: "n04_compatible"` ·
                 whether n07b must re-run · n11b / n13 / Stage 2 indexing
SEVERITY         MAJOR -- it removes a cross-block blocker, and it is the ESSGNN
                 side's call to accept, not this block's
```

**`R_y(180°)` is its own inverse, which is why the offset closes exactly rather than doubling.**
The concern that the fix might "differ by two rotations instead of none" does not arise for a
180-degree yaw.

### The two artifacts `ESSGNN REVIEWER` asked the Engineer for

```
(1) implementation location and sign convention

    metafind/data/meshload.py
        FRAME_CORRECTION = np.diag([-1.0, 1.0, -1.0])     # (x,y,z) -> (-x, y, -z)
        det = +1, orthonormal, Euler(xyz) = (180, 0, 180) == R_y(180 deg)
        applied in load_scene() as scene.apply_transform(FRAME_CORRECTION),
        i.e. to the SCENE, so every node transform is composed with it once.
    metafind/data/renders.py  normalised_scene()  calls meshload.load_scene(),
        so n03 and n04 inherit the same correction from one place.

(2) view_00 camera direction for a corrected asset, world frame, unit, about centre

    n04 camera direction        d04[0] = [0, 0.342020143, -0.939692621]
    camera eye = d04[0] * 3.0          = [0, 1.02606043, -2.819077862]
    effective in-mesh direction R@d04[0] = [0, 0.342020143, +0.939692621]
    n07b's d07[0]                        = [0, 0.342020143, +0.939692621]

    UP_AXIS = [0, 1, 0] ; ORBIT_ELEVATION_DEG = 20.0 ; N_VIEWS = 11
```

### What this does NOT establish — stated so it is not over-read

**It settles camera geometry, not asset orientation.** The two nodes now sample **corresponding
directions in each asset's own frame**. It does **not** follow that an Objaverse asset and a
ProcTHOR asset are oriented alike in the world: the canonical facing of an Objaverse mesh is
undetermined — recorded in `evidence/n03_n04_upstream_verification.md` and unchanged by this.

So `image_protocol: "n04_compatible"` is restored **at the level the claim is written at**. Any
stronger reading — that the two asset families share a canonical front — remains `UNKNOWN`.

**Whether `n07b` re-runs is `ESSGNN`'s decision on `ESSGNN`'s evidence.** This block supplies the
`n04` side and does not conclude for another block. **`ULIP2` must not write "n07b does not need
re-running" into any document**; that sentence belongs to `ESSGNN REVIEWER` if they accept this.
