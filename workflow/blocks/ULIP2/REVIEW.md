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
