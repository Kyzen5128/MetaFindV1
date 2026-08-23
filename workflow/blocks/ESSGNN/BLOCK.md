# BLOCK — ESSGNN (scene chain)

**State:** `READY — code only, no GPU` · **Engineer:** unassigned · **Reviewer:** unassigned
**Verified:** 2026-08-22

---

## 1. Objective

Produce layout-conditioned scene representation, train Stage 2, compose scenes, and produce
**Table 2**. Every research-significant behaviour classified with evidence.

## 2. Scope

n07 scene graphs · n07b ProcTHOR modalities · n08 semantic edges · n09b Stage 2 protocol ·
n09c scene splits · n11b Stage 2 gallery index · n13 Stage 2 training · n14 equivariance
probe · **n15a / n15b / n15c → n16 compose → n17 judge (Table 2)** · gates G6, G7.

Open questions this block owns: **`Q-ESSGNN-AXIS`** (coord_feat / architecture_family
coupling), **`Q-NODETEXT`** (n08's node text is category-only), **`Q-TABLE2`** (the 200-scene
protocol and scale comparability), **`Q-JUDGE-MODEL`**, **`Q-N08-MODEL`**, **`Q-YAW-PLACEMENT`**.

## 3. Non-scope

Everything in the object chain. `Q-TOWER` and `Q-BUILDMODEL` belong to the Integrator.

## 4. **USER constraint — binding**

> 「ESSGNN 可以把該寫的程式碼寫好，任何會需要用到 GPU 的先等」

**Write code. Run no GPU job of any kind** without a fresh authorisation. The GPU belongs to
ULIP2 for the foreseeable future.

**CORRECTED 2026-08-22 — this sentence was wrong and an engineer acting on it would have
rewritten a trainer that already exists.** Found by the ESSGNN Reviewer, verified by Master:

```
IMPLEMENTED, NEVER EXECUTED   n11b   gallery_index.py:5 carries IMPLEMENTS-NODE
                              n13    stage2.py, 680 lines, the loop is complete.
                                     Its marker is deliberately OFF -- stage2.py:3-11
                                     says the marker is a claim and goes on when a
                                     smoke run passes, not when the code looks done.
                                     It cannot run: needs stage1_ckpt (n10) and
                                     sem_edge_cache (n08), neither of which exists.

NO IMPLEMENTATION AT ALL      n14 · n15a · n15b · n15c · n16 · n17
```

**"Never executed" and "unimplemented" imply opposite next actions.** Use the right one.

There is still plenty that needs no GPU: the six nodes above have no code, and six open
questions belong to this block.

## 5. Current state — measured 2026-08-22

```
scene_graphs                12,000   valid
procthor_modalities          1,467   valid  -- but the Stage 2 gallery is 1,439, see below
procthor_node_embeddings     present  valid
procthor_object_text.json    present  rule-based f"a {category}" - NOT LLM output (Q-NODETEXT)
scene_splits.json            present  9,600 / 2,400
stage2_protocol.json / essgnn_arch_protocol.json / essgnn_edge_protocol.json / stage2_positive_map.json   present
sem_edge_cache.json / sem_edge_sentences.jsonl / sem_edge_embeddings.npz   DELETED 2026-08-22
```

### ⚠️ `1,467` is NOT the Stage 2 denominator. It is **1,439**. Measured 2026-08-24.

`train/gallery_index.py:261` excludes every asset whose `pointcloud_uri` is `None`, so the
Stage 2 gallery — **and therefore Table 2's denominator** — is smaller than the modality count.

Measured by Master 2026-08-24, reading all 1,467 sidecars directly:

```
sidecars                        1,467
pointcloud_uri is None             28     <- excluded
pointcloud_uri set, file present 1,439    <- the gallery
pointcloud_uri set, file MISSING     0
key absent entirely                  0
```

All 28 carry one reason, verbatim and identical: *"every view was empty; the asset never entered
frame. AI2-THOR returned no depth for this asset at any distance; its material is not in the depth
prepass."* They have text and images and no point cloud, and `2.6`'s modality-complete gallery has
no way to admit them.

> **Master wrote "transparent materials" here first. That word is not in the record.** The reason
> string says *the material is not in the depth prepass*; **why** it is not is unverified.
> "Transparent" is a plausible explanation for glassware and it is still an INFERENCE — the exact
> `CONTEXT.md` §3 notch, committed by Master one hour after writing the rule. Caught by the ESSGNN
> Engineer. Withdrawn.

### It is not a scattered 1.9%. **Five families are 100% gone, and 70.6% of houses are affected.**

Measured by the ESSGNN Engineer 2026-08-24, reproduced exactly by Master:

```
family              excluded / corpus
  Vase_Open              3 /   3     100%   <- every one
  Bottle                 1 /   1     100%   <- every one
  CD                     1 /   1     100%   <- every one
  RoboTHOR_cup_ai2       1 /   1     100%
  Tabletop_Decor_1       1 /   1     100%
  Bowl                  11 /  30      37%
  Cup                   10 /  29      34%
```

Across all 12,000 scene graphs, 827,730 node instances:

```
instances referencing the 28   20,411   (2.47%)
houses with at least one        8,471 / 12,000   (70.6%)
most frequent   CD_1 6,074 · Bottle_1 3,573 · Vase_Open_1 840 · Vase_Open_3 836
```

**"1.9% of assets" and "no open vase, no bottle and no CD exists in the gallery" are the same fact,
and only the second predicts a failure.** `3experiments.tex:55` scores Table 2 partly on
*"Color Scheme and Material Choices … consistency in textures, colors, and materials"* — a gallery
that structurally cannot return glassware, judged on material consistency. It runs, it scores, and
nothing reports an error.

### Are they still CONTEXT nodes? **YES — traced by Master, 2026-08-24.**

The Engineer raised this and could not answer it while stopped, correctly refusing to guess. It is
the question that decides the severity, so Master traced it read-only:

```
stage2.py:131   if asset_id in eligible          -> TARGET selection only; the 28 are excluded here
stage2.py:194   keep = [n for n in graph["nodes"] if n["index"] != target_index]
                                                 -> no point-cloud filter. Every other node stays.
stage2.py:270   data.node_vectors[str(n["asset_id"])] for n in keep
                                                 -> needs a node vector for every kept node
procthor_node_embeddings  1,467 asset_ids, and all 28 are present (0 missing)
```

**So the layout ESSGNN reads is intact.** The 28 remain as context nodes with position, node text
and vector; only the point cloud is absent, and the context path never asks for one. **This is a
candidate-pool composition question, NOT the MASTER-IMPACTING case** the Engineer flagged as the
alternative.

**Still open, and NOT closed by the above:** the effect on `n16` / Table 2. `n16` retrieves and
then PLACES, and bottles, open vases and CDs are not in the pool at all. Whether that is a defect
depends on Table 2's gallery scope, which `U-21` leaves open — it fixed the scene source
(I-Design) and not the gallery. **Do not call it a defect before that is settled.**

**`gallery_index.py:244` and `:253` say "24 of 1,467, 1.6%". That comment is stale — it is 28, and
1.9%.** Found by the ESSGNN Engineer running the `CONTEXT.md` §3 notch check on his own earlier
statement; he had quoted the comment rather than counting. Reproduced exactly by Master. **The
comment is the ESSGNN Engineer's file and the block is paused — not corrected here, routed to him.**

**What is measured and what is not.** The 1,439 figure is the count of sidecars that pass the only
filter in the build loop, with their `.npz` present on disk. It is **not** an executed index —
`n11b` has never run, and `np.load` plus `prepare_depth_shell` at `:270-271` could still fail on
an individual file. `1,439` is therefore the **input-side ceiling**, exact as to the filter and
unverified as to execution. Do not restate it as the built index size until `n11b` has run.

---

## 6. **n08 must be rebuilt — USER directive 2026-08-22**

> 「不准使用舊模型產出的東西」

n08's semantic-edge sentences were generated by `Qwen2.5-7B-Instruct`
(`semantic_edges_run.py:77`). All three of its artifacts were deleted. **n08 must re-run.**

Four things this drags in — none may be decided locally:

1. **It needs a GPU.** Write the code now; the run queues behind ULIP2.
2. **It changes Stage 2's input.** Different sentences → different edge embeddings →
   different layout conditioning. That is a research condition, not maintenance.
3. **`Q-NODETEXT` should be settled first.** It is about n08's node text. Resolving it after
   the rerun means running n08 twice.
4. **Registry gap.** `semantic_edges_run.py:77` labelled its model as a stand-in under
   deviation `D-2`. Since the registry split `D-2` (annotation) from `D-8` (scene judging),
   **n08's model belongs to neither id.** It needs its own entry. Integrator's job.

**The model for n08 is a USER decision** (`Q-N08-MODEL`), not the engineer's.

## 7. What is settled and must not be re-litigated

- **`f_x` stays a scalar coordinate multiplier.** Verdict `PAPER-AMBIGUOUS` (`DL-004`).
  **Never write "the paper is wrong", and never cite upstream EGNN as settling it.**
  The `2.2e-16 vs 0.43` figures are **UNVERIFIED here and unreproducible** — no `R³`
  variant exists in code.
- `essgnn_arch_protocol.json`: `appendix_shared_msg`, `coord_feat: current`, `hidden_dim
  128`, `n_layers 4`, `distance: squared`. Confirmed internally coherent against the paper's appendix.
- Semantic edges are undirected.
- Stage 2 loss is symmetric (Eq. 7a/7b) — PAPER FACT.
- `fully_shared` cannot reach Stage 2 (`dual_tower.py:315-321` raises on `freeze_gallery()`).

## 7b. The `n04` / `n07b` frame question — SETTLED by Master, 2026-08-22

The ESSGNN Reviewer found that `n04` and `n07b` orbit in different frames and said it could
**not** determine from outside whether ULIP2's mesh-load yaw would align them or double the
offset. **Master measured it. It aligns them exactly.**

```
d07[k] == Ry(180) . d04[k]        pairwise, every k        max err 2.22e-16
as SETS, d07 vs Ry180(d04)                                 0.000000 deg
as SETS, d07 vs d04  (before the fix)                      15.3706 deg
d04 vs Ry180(d04)    -- the ring is NOT 180-symmetric      15.3706 deg
```

**Why the ring is not self-symmetric, and why that mattered:** 11 views at `360/11 = 32.7273`
apart puts 180° at **5.5 slots** — exactly half a step, because 11 is odd. So before the fix the
two nodes were not merely relabelled, they pointed at genuinely different directions.

**After the fix they agree exactly, pairwise and as sets.**

> **Evidence correction, 2026-08-22 — Master's first version of this paragraph had a gap and the
> ESSGNN Reviewer caught it.** Master cited *"`frame_correction` verified on all 46,052 `n03`
> sidecars"*. **`n03` is point clouds. It cannot prove `n04`'s camera is in the corrected frame.**
> The missing link, supplied by the Reviewer and re-verified here: **both nodes call the same
> loader** — `renders.py:272` and `pointclouds.py:153` each call `meshload.load_scene(path)`, and
> those are the only two `meshload` consumers in the repository. `meshload.py:177` applies
> `FRAME_CORRECTION = diag([-1, 1, -1, 1])`, `det = +1`, self-inverse. `renders.py:269-271` says so
> in its own comment. **Same loader, therefore same frame** — and only now is the conclusion
> evidence rather than inference.

ProcTHOR assets come through AI2-THOR and are not yawed. Net: an Objaverse asset under `n04` and a
ProcTHOR asset under `n07b` end up in the same relative viewing frame.

**Consequences:**

- **`procthor_node_embeddings.npz` does NOT need re-deriving for frame reasons.** ULIP2's
  "saves ~3 GPU hours" holds, and the Reviewer's hold on it is lifted. It may still be
  regenerated for the `n08` model rerun — a different reason.
- **`image_protocol: "n04_compatible"` in `stage2_protocol.json` was FALSE for everything
  produced before the fix**, and becomes true only once `n04` finishes. Do not read it as a
  historical guarantee.
- **View index `k` now means the same view in both nodes.** Before the fix it did not — they were
  5.5 slots apart. Any code that paired them by index was wrong and is now right, by accident.
- **`test_the_orbit_uses_n04s_constants_not_copies` is still structurally blind and this does not
  fix it.** It asserts `m.N_VIEWS is r.N_VIEWS` and `m.ORBIT_ELEVATION_DEG is r.ORBIT_ELEVATION_DEG`
  — two scalars that always agreed — while `n07b` builds its own poses instead of importing a
  direction function. **The one quantity that differed is the one it cannot see.** It must compare
  the direction, not the constants. **ESSGNN's to fix.**
- **`procthor_modalities.py:151-156` is factually wrong.** It calls the difference *"trimesh's
  z-up to Unity's y-up"*. `n04` was never z-up — its `+Z` orbit was a defect. The docstring records
  a bug as a convention. **ESSGNN's to fix, in the same pass as the test.**

## 7c. Three knobs in `n08`, not one — Master, 2026-08-22

The ESSGNN Reviewer reported that `Q-N08-MODEL` *"also decides the node features `t_i`"*.
**The shared-encoder fact is real; the attribution is not.** Verified in
`semantic_edges_run.py`:

```
LLM_MODEL     :77   Qwen2.5-7B-Instruct     GENERATES the edge sentences
TEXT_ENCODER  :102  CLIP ViT-B/32, 512-d    ENCODES both the edge sentences and t_i
node_texts    :371  from procthor_object_text.json -- rule-based "a {category}",
                    source: procthor_category.  NOT LLM output.
```

| Knob | Question | What it moves |
|---|---|---|
| `LLM_MODEL` | **`Q-N08-MODEL`** | the **edge** sentences only. `t_i`'s content is untouched |
| node text | **`Q-NODETEXT`** | `t_i`'s content. Today `"a {category}"`, which collapses distinct assets |
| `TEXT_ENCODER` | **`U-20` — and it is marked RESOLVED. See §7d** | **both `t_i` and `e_ij`, and the width.** This is the real coupling |

> **CORRECTED 2026-08-22.** This row first read *"unregistered — no question id"*. **Wrong.**
> `U-20` covers exactly this knob. The ESSGNN Engineer spotted it and the Reviewer verified it
> before Master could act — **had Master registered a new id, one knob would carry two ids, one
> `RESOLVED` and one open, which is worse than the gap it was meant to close.**

`:98-101` is explicit: *"This also pins `t_i`'s encoder … Whatever encodes `t_i` must be THIS
model"*, and `:366-370` gives the reason — the two vectors are **concatenated inside `f_h`**, so a
mismatch puts node and edge features in unrelated spaces with nothing to signal it.

**So the Reviewer's warning lands on the encoder, not on the LLM.** Changing `TEXT_ENCODER` moves
`t_i` and `e_ij` together and changes `EDGE_DIM = 512`; `essgnn_arch_protocol.json`'s
`hidden_dim: 128` does **not** follow automatically. Changing `LLM_MODEL` does none of that.

**The gap this exposes: `TEXT_ENCODER` has no question id at all**, while the two knobs that
matter less both have one. Master's, to register.

## 8. Carried finding — do not lose

**`Q-YAW-PLACEMENT`.** Our point clouds and renders sit 180 degrees yawed about Y
relative to ULIP-2's released clouds. It does **not** move the embedding, so the corpus
is fine — but **`n16_compose_scenes` places assets with real geometry, and a 180°-yawed
asset is placed backwards.** This block owns that decision before n16 is implemented.

## 9. Self-verification the engineer owns

Same standard as ULIP2 §8. Having a reviewer never excuses skipping it.

## 10. Milestone

Not reachable until Stage 2 trains and Table 2 exists. Requires engineer self-verification
+ reviewer independent verification + Codex adversarial review + Master integration review
+ the USER's item-by-item acceptance.
