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
pointclouds     46,052   .npz + .json each; verified against official ULIP-2 artifacts
renders         46,045   directories, 506,495 PNG = 46,045 x 11 exactly
    of which    45,955   USABLE -- in renders_index.jsonl, with a sidecar
    and             90   directories of 11 blank PNGs, no index entry, no sidecar
annotations          0   the corpus was deleted; none has been produced since
embeddings           0   the directory does not exist
checkpoints          0   nothing has ever been trained
annotations_index.jsonl  ABSENT -- splits.py:170 reads it unconditionally, so n09
                         raises FileNotFoundError until n05 runs
splits.json · eval_protocols.json · stage1_protocol.json      absent
```

**Your corpus denominator is 45,955, not 45,952.** `DL-006` (2026-08-22) folded the 3 deleted
residuals back in. Code and tools already agree — `annotate_run.py:142`, `tools/status.sh:55`,
`tools/chain_after_n05.sh:56`. The 97-asset gap from 46,052 is n04's quarantine
(`quarantine_n04_render_views.jsonl`, 143 lines / 99 unique uids, all `DETERMINISTIC_INPUT`,
*"every view is blank -- the asset never entered frame"*). **Do not re-render them** (§7).

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

### Milestone 1 — the annotator bake-off. Everything here is M1.

Assigned by Master 2026-08-22 under `DL-008`. **`E` = Engineer · `R` = Reviewer.**
Report through `HANDOFF.md`. Report **FINDING** and **DECISION** separately, always.

| # | Owner | Item | State |
|---|---|---|---|
| **W-1** | E | **Multi-arm runner.** `annotate_run.py:72` hardcodes one `MODEL_ID`. The bake-off needs arm selection, per-arm output isolation, and the §11 metric block emitted per arm. `DL-008`: that hardcoded value is **leftover state, not a decision** | first |
| **W-2** | E | **Run the two ready arms first** — `gemma-4-31B-it-qat-w4a16`, then `gemma-4-12B-it`. Both are on disk and need zero preparation, so the USER gets numbers soonest. Same sample, same prompt, same seed, one arm at a time | after W-1 |
| **W-3** | E | **Rebuild `annotation_provenance.json`** via `tools/declare_annotation_provenance.py`. It declares 45,955 records that do not exist (debt `D-1`). Never by hand | any time |
| **W-4** | E | **Qwen arm — produce the w4a16 build.** §11 claims `~15 GB`; **measured, it does not exist.** Only 55.56 GB bf16 is on disk. If quantizing proves expensive or degrades the model, **report it, do not drop the arm** | parallel with W-2 |
| **W-5** | R | **Audit `Q-CATEGORY` against `DL-007`.** `DL-007` says the LVIS-anchor design was **approved 2026-08-21**, yet `Q-CATEGORY` is listed as *"no investigation has been done"*. **Are these the same question?** If yes, `Q-CATEGORY` is answered and should be closed; if no, name precisely what is still open. This changes whether M2 may start | first, blocks M2 |
| **W-6** | R | **`DL-007` admits `D0-010` was never researched** — its §6–§11 are empty, so LVIS anchoring passed by *design ratification, not evidence audit*. It is the most scientifically material change in the pipeline. Produce the missing audit: prompt hint / hard value / cross-check / record-only, with evidence for each | before M2 |
| **W-7** | R | **`IC-1` — is `identity_confirmed` a rubber stamp?** We feed the LVIS label in, then ask the model to confirm it. Design a check the bake-off can actually run. **Naming the false-rate is not enough — there is no ground truth telling us a `true` is really true.** State plainly if no sound check exists at this sample size | with W-2 |
| **W-8** | E | **Stale `24 GB` comments** — `ulip_backbone.py:9`, `:101`, `resolve_stage1.py:31`, `:645`. This machine is 32,607 MiB. Comment-only; they are feasibility premises, so correct them, change no behaviour | low, any time |

### Later milestones — not started, do not begin without USER scope approval

| # | Item | Blocked on |
|---|---|---|
| M2 | **n05 full run** — **45,955** assets, runs **once** | the USER's explicit go, plus W-5 |
| M3a | **n06 encode** — 45,955 embeddings | a corpus existing |
| M3b | **n09 splits** | `Q-TOWER` — **Master holds it** while INTEGRATOR is on hold (`DL-009`) |
| M3c | **Gates G1 / G2** — implement, and define the gate-record schema the five later gates inherit | two USER decisions first: does running a gate now count as evaluation or as backfill, and is the self-retrieval criterion in the first run, sampled, or deferred |
| M4 | **Stage 1 training** | M2, M3a, M3b |
| M5a | **Gallery index** — staging → freeze → promote | M4 |
| M5b | **n15 — Table 1.** **Zero implementation code exists** — there is no `metafind/eval/` directory at all. Designable and unit-testable against synthetic inputs **today**, with no trained model | nothing technical; only sequencing |

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

### Candidates — measured on disk by Master, 2026-08-22

**GPU: RTX 5090, 32,607 MiB.** Sizes below are the actual weight files, not estimates. **No arm
has been loaded.** Fitting on paper is not evidence that it runs with 11 images in context.

| Arm | Measured on disk | State |
|---|---|---|
| `google/gemma-4-31B-it-qat-w4a16` | **22,188 MiB**, single `model.safetensors`, `quant_method: compressed-tensors`, 0 `.incomplete` | ✅ **READY.** Official quantization-aware-trained. Largest Gemma that fits |
| `google/gemma-4-12B-it` | **22,812 MiB**, bf16, **no quantization at all**, 0 `.incomplete` | ✅ **READY.** No quantization loss, and materially faster. The throughput candidate |
| `Qwen/Qwen3.8-27B`, self-quantized w4a16 | **55.56 GB bf16 only** — 18/18 shards verified against `model.safetensors.index.json` `total_size` | ❌ **NOT READY. The w4a16 build does not exist and has never been produced** (work item W-4). bf16 does not fit. No official quantized release fits either: the FP8 build is 30.9 GB, leaving 1.7 GB — it will not run with vision |

**Run the two ready arms first (W-2), quantize Qwen in parallel (W-4).** Do not drop the Qwen
arm to save time — if quantization proves costly or lossy, that is a **finding to report**, and
the USER decides whether the arm still runs.

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
seconds per asset, and the projection to 45,955
VRAM peak with 11 images
model id · quantization method and bit-width · seed · git commit · hardware
```

### Standing rule

**Only the winner runs the full corpus.** 45,955 assets is a multi-day job at 27B scale.
Running every arm at full scale is the one thing this bake-off exists to prevent.
