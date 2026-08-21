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
