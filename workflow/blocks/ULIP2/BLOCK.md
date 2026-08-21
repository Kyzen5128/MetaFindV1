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
| 1 | **Two-model annotation comparison** — Qwen3.8-27B vs Gemma 4. USER decision 2026-08-22 | model variants awaiting USER choice |
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

## 9. Milestone

Not reachable until Stage 1 trains and Table 1 exists. Requires engineer self-verification
+ reviewer independent verification + Codex adversarial review + Master integration review
+ the USER's `APPROVE`.
