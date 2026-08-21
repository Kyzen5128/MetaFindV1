# BLOCK — ULIP2 (object chain)

**State:** `ACTIVE` · **Engineer:** unassigned · **Reviewer:** unassigned
**Opened:** 2026-08-22

---

## 1. Objective

Produce a trustworthy object corpus, train Stage 1, build the gallery index, and produce
**Table 1**. Every research-significant behaviour classified with evidence.

## 2. Scope

n02 download · n03 pointclouds · n04 renders · n05 annotate · n05b encoding protocol ·
n06 encode text+image · n09 splits · n10 Stage 1 training · n10b post-Stage-1 encode ·
n11 / G4 / n12 gallery index · **n15 retrieval eval (Table 1)** · gates G1–G4.

Tasks: `D14`, `D15`, `D16`, `D1`, `D2`, `D3`, `D4`, `D7`.
Decisions owned: `D0-003` (the 3 legacy-v1 residuals), `D0-010` (LVIS category source).

## 3. Non-scope

ESSGNN, scene graphs, semantic edges, Stage 2, scene composition, the n17 judge.
`D0-002` and `D0-005` belong to the Integrator — they cross into Stage 2.

## 4. Current state — measured 2026-08-22

```
pointclouds     46,052   BEHAVIOR-VERIFIED vs upstream (D15 FINDINGS FIND-6/7)
renders         46,045   BEHAVIOR-VERIFIED vs upstream (FIND-9)
annotations          3   only the legacy-v1 residuals remain. The 45,952 v3 records
                         were DELETED 2026-08-22 on the USER's order (old-model output)
embeddings           0   deleted
checkpoints          0   nothing has ever been trained
splits.json / eval_protocols.json / stage1_protocol.json   ABSENT
```

`pytest tests/ -q` → 582 passed. `tools/check_graph.py` → 2275 checks, all pass.
Neither is evidence of paper fidelity.

## 5. What is settled and must not be re-litigated

- `DL-001` Stage 1 text template (centimetres, ratified form)
- `DL-002` Stage 1 encoding contract, cache validity, pre-flight
- `DL-003` τ = 0.5, `learnable_temperature: false`, protocol refresh, AC-1
- `D15 FINDINGS` — n03/n04 verified; **the point-cloud corpus does not need regenerating**
- n05 v5 design decisions 1–4 (LVIS anchor, `identity_confirmed`, exact proportions, synset lookup)

## 6. Open work items

| # | Item | State |
|---|---|---|
| 1 | **Annotator bake-off** — pick the best annotator for this hardware. See §11 | awaiting USER go |
| 2 | `D14` Phase 2 — stratified 300–500 asset sample, then **HARD HOLD** | blocked on 1 |
| 3 | `D14` Phase 3 — full re-annotation | blocked on the USER's explicit go |
| 4 | `D0-010` — how the LVIS category enters n05. §6–§11 empty, no research done | OPEN |
| 5 | `D0-003` — the 3 legacy-v1 residuals: admit, drop, or re-annotate | OPEN. **They are the only annotations left on disk** |
| 6 | `D1` n06 re-encode | blocked on a corpus existing |
| 7 | `D2` n09 splits | blocked on `D0-002`, `D0-003` |
| 8 | `D15` — findings written, but `TASK.md` still says `PLANNED`. Reconcile | needs Master |
| 9 | `D16` gates G1/G2 — `OQ-2`'s cost is now answerable from FIND-7 | blocked on `D15` acceptance |
| 10 | `n15` Table 1 — **zero implementation code** | not started |

## 7. Standing constraints

- The GPU belongs to this block. ESSGNN does not run GPU jobs.
- `D14` Phase 3 costs ~19.6+ GPU-hours and **must not run twice**. Its HOLD GATE is absolute.
- The 3 legacy-v1 residuals are **PROTECTED**. Do not touch them; `D0-003` is unresolved.
- LVIS category anchoring is a **DEVIATION**. Never describe it as paper-faithful.
- The annotation model is not GPT-4o (`D-2`). Never describe it as paper-faithful.

## 8. Self-verification the engineer owns

Implementation correctness · unit and integration tests · runtime verification · artifact
integrity · provenance · dataset consistency · upstream/downstream consistency · semantic
sanity · paper consistency · failure cases · resume and cache correctness · silent failure.

**Having a reviewer never excuses skipping this.**

## 11. Annotator bake-off — USER decision 2026-08-22

> 「我沒有要比較兩個 LLM，我只是要挑出最好最適合的。比較不是重點，而是我想知道我挑誰對我比較好」

**This is a selection procedure, not a research comparison.** That distinction decides how it is
run and how it is reported:

- The arms do **not** have to be matched in size or precision. Each arm is **the best that family
  can actually do on this hardware**, because that is what would be deployed.
- Google's official QAT checkpoints are therefore **admissible and preferred** for the Gemma arm.
  An earlier Master note called that an unacceptable confound; that framing applied to a
  controlled model comparison, which this is not.
- The result is an **IMPLEMENTATION CHOICE backed by a measurement**, never a claim that one model
  is better than another in general. Report it as "chosen on this hardware, on this sample, by
  these criteria".
- Whatever wins, the annotator is still not GPT-4o. **Deviation `D-2` stands** either way.

### Candidates — all fit 32.6 GB

| Arm | Weights | Notes |
|---|---|---|
| `Qwen/Qwen3.8-27B`, self-quantized w4a16 | ~15 GB | **bf16 source already on disk**, 18/18 shards. No official quantized release exists that fits (FP8 is 30.9 GB — only 1.7 GB headroom, will not run with vision) |
| `google/gemma-4-31B-it-qat-w4a16-ct` | 23.3 GB | Official quantization-aware-trained. Largest Gemma that fits |
| `google/gemma-4-12B-it` | 24.0 GB | **bf16, no quantization loss at all**, and materially faster. The throughput candidate |

`google/gemma-4-26B-A4B-it` (MoE, 4B active) is a fourth candidate if throughput turns out to be
the binding constraint. Not proposed for the first round.

### Protocol

**Same sample, same prompt, same seed, one arm at a time.** This *is* `D14` Phase 2 — the
stratified 300–500 asset sample — run once per arm rather than once.

Record per arm, per `.claude/rules/experiments.md`:

```
identity_confirmed == false            n and %
category vs LVIS       exact / refined / divergent, counted separately
category top-20 concentration          vs the 22.3% baseline
toy / bookshelf / pillow frequency     vs 3.4% / 2.4% / 2.1%
dimension plausibility against the exact mesh proportions
JSON parse failures and repair-budget exhaustions   <- this is what stranded the 3 residuals
seconds per asset, and the projection to 45,952
VRAM peak with 11 images
model id, quantization method and bit-width, seed, git commit, hardware
```

### Standing rule — the reason this exists

**Only the winner runs the full corpus.** 45,952 assets is a multi-day job on a 27B-class model
(the old 7B took 19.6 h). Running every arm at full scale is the one thing this bake-off exists to
prevent. USER agreed 2026-08-22.

The HOLD GATE (`D14` R-A) is unchanged and applies to the winner's full run.

---

## 9. Milestone

Not reachable until Stage 1 trains and Table 1 exists. Requires engineer self-verification
+ reviewer independent verification + Codex adversarial review + Master integration review
+ the USER's `APPROVE`.
