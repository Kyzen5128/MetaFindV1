# BLOCK — ULIP2 (object chain)

**State:** `ACTIVE` · **Engineer:** unassigned · **Reviewer:** unassigned
**Verified:** 2026-08-22

---

## 1. Objective

Produce a trustworthy object corpus, train Stage 1, build the gallery index, and produce
**Table 1** — with every research-significant behaviour classified and cited.

## 2. Scope

```
n02 download → n03 pointclouds → n04 renders (11 views) → n05 annotate
→ n05b encoding protocol → n06 encode text+image → n09 splits
→ n10 Stage 1 training → n10b post-Stage-1 encode
→ n11 staging → G4 freeze → n12 promote
→ n15 retrieval evaluation (Table 1)

gates G1 G2 G3 G4
```

Open questions this block owns: **`Q-CATEGORY`**.

## 3. Non-scope

Scene graphs, semantic edges, ESSGNN, Stage 2, scene composition, the scene judge.
`Q-TOWER` and `Q-BUILDMODEL` belong to the Integrator — they cross into Stage 2.

## 4. Current state — measured 2026-08-22

```
pointclouds     46,052   verified against official ULIP-2 artifacts
renders         46,045   verified against official ULIP-2 artifacts
annotations          0   the corpus was deleted; none has been produced since
embeddings           0
checkpoints          0   nothing has ever been trained
splits.json · eval_protocols.json · stage1_protocol.json      absent
```

`pytest tests/ -q` → 582 passed. `tools/check_graph.py` → 2275 checks, all pass.
Neither is evidence of paper fidelity.

## 5. Settled — do not re-litigate

Decisions in force are recorded in `workflow/DECISION_LEDGER.md`:

- **Stage 1 text serialization** — the ratified centimetre template
- **Stage 1 encoding contract** — cache validity, protocol-mismatch rejection, the 77-token
  hard gate, the text pre-flight
- **τ = 0.5**, `learnable_temperature: false`
- **n03 and n04 are verified against upstream. The point-cloud corpus does not need
  regenerating.** Evidence: `evidence/n03_n04_upstream_verification.md`
- **The n05 v5 design** — LVIS anchor, `identity_confirmed`, exact mesh proportions, synset
  lookup. Evidence: `evidence/n05_v5_design.md`, `evidence/n05_annotation_defect.md`

New *evidence* against any of these is a MASTER-IMPACTING FINDING. Preference is not.

## 6. Open work

| # | Item | State |
|---|---|---|
| 1 | **Annotator bake-off** — pick the best annotator for this hardware. §11 | awaiting USER go |
| 2 | **n05 sample validation** — stratified 300–500 assets per candidate, then **HARD STOP** | blocked on 1 |
| 3 | **n05 full run** — 45,952 assets, runs **once** | blocked on the USER's explicit go |
| 4 | **`Q-CATEGORY`** — what role the LVIS ground-truth category plays in n05. No investigation has been done | OPEN |
| 5 | **n06 encode** — 45,952 embeddings | blocked on a corpus existing |
| 6 | **n09 splits** | blocked on `Q-TOWER` |
| 7 | **Gates G1 / G2** — implement, define the gate-record schema the five later gates inherit, produce the project's first gate records | not started. Two USER decisions first: does running a gate now count as evaluation or as backfill, and is the self-retrieval criterion in the first run, sampled, or deferred |
| 8 | **Stage 1 training** | blocked on 3, 5, 6 |
| 9 | **Gallery index** — staging → freeze → promote | blocked on 8 |
| 10 | **n15 — Table 1.** **Zero implementation code exists.** Can be designed and unit-tested against synthetic inputs today | not started |

## 7. Standing constraints

- **The GPU belongs to this block.** ESSGNN runs no GPU jobs.
- **The full annotation run must not happen twice.** It is a multi-day job. Its hold gate is
  absolute: good sample numbers are not permission.
- **LVIS category anchoring is a DEVIATION.** The paper has the VLM generate the category.
  Never describe it as paper-faithful.
- **The annotation model is not GPT-4o.** Also a DEVIATION, whichever candidate wins.
- Point clouds and renders are read-only here. **No re-render** — framing was measured not to
  drive annotation agreement.

## 8. Self-verification the engineer owns

implementation correctness · unit and integration tests · runtime verification · artifact
integrity · provenance · dataset consistency · upstream and downstream consistency · semantic
sanity · paper consistency · failure cases · resume and cache correctness · silent failure.

**Having a reviewer never excuses skipping this.**

Every test whose expected value encodes a claim about the world must name where that value came
from, and it must not be the implementation under test (`workflow/SKILLS.md` §7).

## 9. Evidence held by this block

```
evidence/n03_n04_upstream_verification.md   n03/n04 vs official ULIP-2 artifacts
evidence/n05_annotation_defect.md           why the previous annotator failed, measured
evidence/n05_category_vs_lvis.md            the divergence, full corpus
evidence/n05_v5_design.md                   the ratified v5 design
```

## 10. Milestone

Not reachable until Stage 1 trains and Table 1 exists. Requires engineer self-verification +
reviewer independent verification + 4-axis completion review + Codex adversarial review +
Master integration + the USER's item-by-item acceptance.

---

## 11. Annotator bake-off

> USER, 2026-08-22: 「我沒有要比較兩個 LLM，我只是要挑出最好最適合的。比較不是重點，
> 而是我想知道我挑誰對我比較好」

**This is a selection procedure, not a controlled model comparison.** That decides how it runs
and how it is reported:

- Arms need **not** be matched in size or precision. Each arm is the best that family can
  actually do on this hardware, because that is what gets deployed.
- Officially published quantization-aware-trained checkpoints are **admissible and preferred**.
- The result is an **IMPLEMENTATION CHOICE backed by a measurement** — never a general claim
  that one model beats another. Report it as "chosen on this hardware, on this sample, by these
  criteria".
- **The annotator is not GPT-4o whichever wins.** The deviation stands.

### Candidates — all verified to fit 32.6 GB

| Arm | Weights | Notes |
|---|---|---|
| `Qwen/Qwen3.8-27B`, self-quantized w4a16 | ~15 GB | **bf16 source already on disk**, 18/18 shards, 56 GB. No official quantized release fits: the FP8 build is 30.9 GB, leaving 1.7 GB — it will not run with vision |
| `google/gemma-4-31B-it-qat-w4a16-ct` | 23.3 GB | official quantization-aware-trained. Largest Gemma that fits |
| `google/gemma-4-12B-it` | 24.0 GB | **bf16, no quantization loss at all**, and materially faster. The throughput candidate |

`google/gemma-4-26B-A4B-it` (MoE, 4B active) is held as a fourth candidate if throughput turns
out to be the binding constraint.

The Qwen3.8 line has only two members — the 27B vision model and a 2.4T text-only model — so
there is no small Qwen arm and no bf16-vs-bf16 pairing.

### Protocol

**Same sample, same prompt, same seed, one arm at a time.** Record per arm:

```
identity_confirmed == false            n and %
category vs LVIS       exact / refined / divergent, counted separately
category top-20 concentration          against the previous corpus's 22.3%
the previous corpus's failure modes    how often each recurs
dimension plausibility against the exact mesh proportions
JSON parse failures and repair-budget exhaustions
seconds per asset, and the projection to 45,952
VRAM peak with 11 images
model id · quantization method and bit-width · seed · git commit · hardware
```

### Standing rule

**Only the winner runs the full corpus.** 45,952 assets is a multi-day job at 27B scale.
Running every arm at full scale is the one thing this bake-off exists to prevent.
