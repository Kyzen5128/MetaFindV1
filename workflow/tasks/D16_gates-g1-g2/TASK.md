# D-Task Execution Contract

> This file is the authoritative execution contract for one bounded work package.
> The task owner must stay within scope, satisfy the Definition of Done, perform verification, obtain Codex review, and return a HANDOFF to Master.

---

## Task ID

`D16_gates-g1-g2`

---

## Status

Current:

`BLOCKED` — on `D15_n03-n04-code-audit`. Contract written by Master 2026-08-21. **Not approved. Do not start.**

> Finishing execution does not make a task `DONE`. See Section 17 of `workflow/tasks/TEMPLATE.md`.

---

## 1. Objective

Implement and execute `G1_sources_valid` and `G2_pc_sanity`, producing the project's **first** `gate_records` entries.

Two gates, one task, because they share a single new write surface (`gate_records`) and a single new module. Splitting them would put two conversations into the same new file.

---

## 2. Why This Task Exists

**No gate has ever run. `gate_records` has never existed.**

Verified 2026-08-21: `data/outputs/gate_records.jsonl` is absent; nothing under `data/outputs/logs/` matches `gate|degrad|audit`; `grep -rln "gate_record"` across `metafind/` returns only `pointclouds.py` (comments describing what G2 *will* check) and `tools/check_graph.py` (a spec-consistency checker, not a gate).

All seven gates — items 03, 05, 14, 17, 21, 29, 37 in the pipeline inventory — are unimplemented and unrun.

The graph spec states the consequence directly (`graph_spec.yaml:1074`):

> **GD — a gate may never backfill its own record. Missing record == not passed.**

So `n03_sample_pointclouds` and `n04_render_views`, whose precondition is `G1 verdict == PASS` (`graph_spec.yaml:1171-1172`), ran against a gate that had not passed. **That ordering violation is a fact of this project's history and must be recorded, not erased.** See §6, Open Question OQ-1 — it is the user's to resolve, not this task's.

**Why now.** `D3_stage1-train` will consume the 46,052 clouds. G2 exists precisely to refuse malformed, degenerate or wrongly-normalised clouds before training. Training first and gating afterwards inverts the gate's only purpose.

**Why after D15.** G2's criteria include "pc_norm'd exactly as ULIP's dataset does". `D15` determines whether that is what `pointclouds.py` actually does. Writing G2's thresholds before that answer would encode an unverified assumption into the artifact that certifies the corpus.

---

## 3. Required Shared Context

Before execution, read:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. `/home/kyzen/MetaFindV1/.claude/rules/research-rigor.md` and `.claude/rules/code-changes.md`
3. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
4. this `TASK.md`
5. **`workflow/tasks/D15_n03-n04-code-audit/FINDINGS.md`** — mandatory. G2's criteria derive from it

Then read only the sources listed under Section 5.

---

## 4. Dependencies

### Required Before Start

- `D15_n03-n04-code-audit` — execution `COMPLETE` **and** integration `USER_APPROVED`
- **OQ-1 resolved by the user** (§6). Execution may not begin while it is open

### Blocks

- `D3_stage1-train` — should not train on an ungated corpus
- All later gates inherit whatever `gate_record` shape and runner this task establishes

### Parallel Safety

`NO`

Reason: this task creates `gate_records`, a channel six other gates will later write to, and it is the first thing to define its schema. Concurrent work touching `metafind/data/pointclouds.py` or the sidecars would race the criteria.

---

## 5. Authoritative Inputs

| # | Source | Why it matters |
|---|---|---|
| 1 | `docs/graph/node_registry.yaml` — `G1_sources_valid`, `G2_pc_sanity` | The gates' purpose, reads, writes, postcondition (`gate_record with is_terminal=true`), failure policy (`CONTRACT_VIOLATION` → `fail_closed`, `max_attempts: 1`), and `record_fields` |
| 2 | `docs/graph/graph_spec.yaml:1068-1075` | The `gate_records` channel: `list[gate_record]`, `merge: append`, `lifetime: persistent`, and rule **GD** |
| 3 | `docs/graph/graph_spec.yaml:1076-1082` | `audit_records` is a **separate** channel "so an audit can never be misread as a gate verdict". Do not merge them |
| 4 | `docs/graph/graph_spec.yaml:1169-1172` | Edges `e02`, `e03`, `e04` — the `verdict == PASS` guards |
| 5 | **`workflow/tasks/D15_n03-n04-code-audit/FINDINGS.md`** | Determines G2's normalisation criterion and thresholds |
| 6 | `metafind/data/pointclouds.py:365-392` | The sidecar fields G2 routes on, and the float64/float32 tolerance interaction |
| 7 | `metafind/paths.py` | Where artifacts live. Do not hardcode paths |
| 8 | Sidecars + `pointclouds_index.jsonl` (46,052) | G2's actual input |
| 9 | `metafind/data/download.py`, `metafind/models/ulip_backbone.py` | G1's inputs: manifest, GLB hashes, ProcTHOR, and the checkpoint **behavioural** probe |

---

## 6. Current Relevant State

**G1 — what it must check** (registry `reads`: `asset_manifest`, `asset_glb`, `procthor_dataset`, `pretrained_weights`)

Registry note, verbatim:

> Checkpoint validity is a BEHAVIOURAL check, not just sha256: a hash proves the file is not corrupt, not that it is the right file.

`pretrained_weights` carries `behavioural_check: {probe, expected, observed, passed}` (`graph_spec.yaml:310`). The registry records that this channel was **added** because G1 "routes on `ckpt_behaviour` and had nowhere to derive it", and that a sha256 is deliberately not enough — "the failure this guards against is a checkpoint that loads and produces plausible garbage, which hashes fine."

`procthor_dataset` was **added to G1's reads by correction**: an earlier draft mentioned ProcTHOR in the criterion text while giving the gate no channel to see it, "so a run with Objaverse complete and ProcTHOR absent could PASS."

Current observed state: manifest 46,052 · GLB 46,052 · ProcTHOR 12,000 · ULIP-2 checkpoint present (`md5 b47ca224557b284537e7e698650b1b8a`) · CLIP ViT-bigG-14 present. **`ulip_backbone.py` already performs load-time structural verification** (`._load_and_verify()`, `._check()`) — whether that constitutes G1's behavioural probe, or whether a separate probe is required, is in scope.

**G2 — what it must check** (registry `record_fields`: `shape`, `centroid`, `max_radius`, `min_axis_var`, `self_retrieval_rank`)

Registry note, condensed: this gate was **narrowed on purpose**. It used to require our clouds to match ULIP's *released* clouds through the same encoder. That tested a proposition the paper never makes. What remains is genuinely an invalidity condition: **shape, finiteness, pc_norm (centroid at origin, max radius 1), non-degeneracy, and that a sampled cloud retrieves its own asset well above chance.** The ULIP comparison survives as `L2-PC-ULIP-REF`, a **diagnostic, not a gate** — and it has never been run, because the reference clouds are not on disk.

The registry is explicit that the narrowing was not a retreat from a failing check: "It is NOT demoted because it was failing — it has never been run … It is demoted because it tested the wrong claim."

**The tolerance interaction — carry this into the criteria.** `pointclouds.py:380-389` records that float32 summation over 10,000 values accumulates ~1e-5 of error, exceeding a 1e-5 centroid tolerance, and that **8 of 46,052 assets were recorded as failing a check their data passes**; one re-verified in float64 gave `5.2e-09`, not `1.15e-05`. The comment states the rule: *"The gate's threshold is not the problem and must not be widened; the measurement was reporting the summation method rather than the cloud."* `D15` §12 re-measures this over ≥200 assets. **G2 must honour that rule: fix the measurement, never the threshold.**

`self_retrieval_rank` requires running clouds through the ULIP-2 point encoder. Cost is not yet estimated. **Estimating it is in scope; deciding whether to spend it is the user's** — see OQ-2.

---

### Open Questions — the user must resolve these before this task starts

**OQ-1 — What does running a gate today, over artifacts produced months ago, actually mean?**

Rule GD says a gate may never backfill its own record. Two readings:

- **(a) Running the gate now is not backfilling.** The gate evaluates the artifacts as they exist and stamps a verdict dated now. GD forbids *fabricating* a past verdict, not evaluating late. The historical ordering violation is recorded separately as a DEVIATION.
- **(b) Any record written after its downstream consumers already ran is a backfill in substance**, and the honest outcome is to record the violation and leave the gate unpassed until the corpus is regenerated in correct order.

Reading (b) implies re-running n03 and n04 — roughly 2.5 hours of I/O-bound work over 351 GB of GLB, on an SMR volume, for a corpus that may be byte-identical. Reading (a) accepts a documented ordering deviation permanently in the reproduction record.

**Master does not choose between these.** It is a reproduction-fidelity decision. `.claude/rules/research-rigor.md` §2 requires stopping here.

**OQ-2 — Is `self_retrieval_rank` in G2's first run?**

It is the one G2 criterion requiring GPU work: encode clouds through the ULIP-2 point encoder and check each retrieves its own asset above chance. Options: full corpus / a sample with a stated confidence / defer to a later gate run. Cost must be estimated by this task before the user decides; the estimate is in scope, the spending is not.

---

## 7. Scope

### In Scope

**Shared**

1. Define the `gate_record` schema — gate id, verdict, `is_terminal`, timestamp, criteria evaluated, observed values, populations, code commit, and the `record_fields` each registry entry names. This schema will be inherited by five later gates; design it for them.
2. Implement a gate runner that **appends** to `gate_records` and can never overwrite or rewrite an existing record (rule GD, `merge: append`).
3. `fail_closed`, `max_attempts: 1` for `CONTRACT_VIOLATION` on both gates — a failing gate halts, it does not retry and does not degrade.
4. Record the historical ordering violation explicitly, in whatever form OQ-1's resolution dictates.

**G1**

5. Manifest completeness against `asset_manifest`.
6. GLB presence and `sha256` for every admitted uid.
7. ProcTHOR presence and the counts the channel declares.
8. A **behavioural** check on the ULIP-2 checkpoint — determine first whether `ulip_backbone.py`'s existing verification satisfies this, and say so explicitly either way.
9. A behavioural check on ViT-bigG-14 to the extent the channel's `behavioural_check` shape requires.
10. Emit one terminal `gate_record`.

**G2**

11. Shape `(10000, 6)` and finiteness over all 46,052 clouds.
12. `pc_norm` — centroid at origin, max radius 1 — using the criterion `D15` establishes, and measuring in the dtype that reports the cloud rather than the summation method.
13. Non-degeneracy via `min_axis_var`.
14. `self_retrieval_rank` **per OQ-2's resolution**, with the cost estimate produced regardless.
15. Emit one terminal `gate_record`.
16. Report, per criterion, how many of the 46,052 pass and the full identity of every failure.

### Explicit Non-Scope

- **Do not modify `pointclouds.py`, `renders.py`, or any existing sidecar, index or `.npz`.** G2 reads; it does not repair. A cloud that fails is reported, not fixed.
- **Do not widen a threshold to make assets pass.** `pointclouds.py:387` states the rule and it is binding.
- **Do not run `L2-PC-ULIP-REF`.** Diagnostic, not gate, reference clouds absent.
- **Do not implement G3, G4, G5, G6 or G7.** Schema design must anticipate them; implementation must not.
- **Do not re-run n03 or n04** unless OQ-1 resolves that way, in which case Master reissues the contract.
- **Do not touch n05, n06 or anything in `D14`'s scope.**
- **Do not delete or move any artifact.**

---

## 8. Expected Deliverables

1. Gate runner + G1 + G2 implementation, carrying `IMPLEMENTS-NODE: G1_sources_valid` and `IMPLEMENTS-NODE: G2_pc_sanity` markers (`tools/check_graph.py` enforces these).
2. `data/outputs/gate_records.jsonl` — the project's first two gate records.
3. Unit tests for the runner and both gates, including a test that a gate **cannot** overwrite an existing record.
4. `workflow/tasks/D16_gates-g1-g2/RESULTS.md` — per-criterion pass counts over 46,052, every failure identified, the `self_retrieval_rank` cost estimate, and the float64/float32 measurement decision with its evidence.
5. `HANDOFF.md` per `workflow/tasks/HANDOFF_TEMPLATE.md`.

---

## 9. Likely Files / Areas

- new: `metafind/gates/` or equivalent — placement is the task's call, argued in the handoff
- new: `tests/test_gates.py`
- read-only: `metafind/data/pointclouds.py`, `metafind/models/ulip_backbone.py`, `metafind/paths.py`
- write: `data/outputs/gate_records.jsonl`, `data/outputs/logs/run_progress.jsonl`
- `docs/graph/node_registry.yaml` — read only; the spec is not edited to match the implementation

**Declared write surface — exactly these:**

```
metafind/gates/**                       (new)
tests/test_gates.py                     (new)
data/outputs/gate_records.jsonl         (new, append-only)
data/outputs/logs/run_progress.jsonl    (append)
workflow/tasks/D16_gates-g1-g2/**       (new)
```

Anything else written is a scope deviation and must be reported before completion, not after.

---

## 10. Execution Requirements

1. Read `D15`'s `FINDINGS.md` before writing G2's normalisation criterion.
2. Confirm OQ-1 and OQ-2 are resolved. If either is open, **stop**.
3. Smallest coherent implementation. No gate framework, no plugin registry, no abstraction for the five gates not being built.
4. A gate that fails must halt and say which criterion and what it observed. It must not degrade, retry, or continue.
5. Never edit the spec to match the implementation. If the spec is wrong, that is a `MASTER-IMPACTING FINDING`.
6. Record any Master-impacting discovery immediately.
7. Stop if a required authority decision is missing.

---

## 11. Master-Impacting Finding Rule

Report as `MASTER-IMPACTING FINDING`, and do not act on:

- G2 failing a material number of the 46,052 clouds — that bears on `D3` feasibility and possibly on regenerating the corpus
- the ULIP-2 checkpoint failing its behavioural check — that invalidates far more than this task
- a registry criterion that cannot be evaluated as written
- any finding implying n03 or n04 must re-run

Include: finding, evidence, affected tasks, whether this task can safely continue.

---

## 12. Verification Requirements

### Required Checks

- `pytest tests/ -q` — **no `--ignore` flag.** Expected baseline **582 passed** plus this task's new tests. A different total must be explained, not assumed normal.
- `python tools/check_graph.py` — must stay at 2275 checks passing, with the two new `IMPLEMENTS-NODE` markers recognised.
- `git status` shows only the declared write surface.

### Required Tests

- Runner appends and cannot overwrite an existing record (rule GD).
- G1 fails closed when the manifest is short, when a GLB hash mismatches, when ProcTHOR is absent, and when the checkpoint behavioural probe fails — one test each, on fixtures, not on the real corpus.
- G2 fails closed on wrong shape, on a non-finite value, on a cloud whose centroid is off origin beyond tolerance, and on a degenerate axis.
- A regression test pinning the float64/float32 measurement decision, so a later refactor cannot silently reintroduce the 8-asset false negative.

### Runtime / Artifact Checks

- Both gates executed over the **real** corpus; `gate_records.jsonl` contains exactly two terminal records.
- Per-criterion pass counts over all 46,052 reported in `RESULTS.md`.
- Every failing asset identified by uid, with the observed value.

### Research Fidelity Check

Required. State, per criterion, whether it implements the registry's stated criterion or deviates, and on what evidence. **A passing gate is not evidence that the criterion is the right criterion** — that question belongs to `D15` and to the registry, and must not be answered by this task's own output.

---

## 13. Definition of Done

- [ ] OQ-1 and OQ-2 resolved by the user before start.
- [ ] `D15` `USER_APPROVED` and its `FINDINGS.md` read.
- [ ] Gate runner, G1 and G2 implemented with `IMPLEMENTS-NODE` markers.
- [ ] Both gates executed over the real corpus; two terminal records in `gate_records.jsonl`.
- [ ] `RESULTS.md` produced with per-criterion counts and every failure identified.
- [ ] `self_retrieval_rank` cost estimated.
- [ ] `pytest tests/ -q` and `tools/check_graph.py` both pass, totals explained.
- [ ] `git status` matches the declared write surface exactly.
- [ ] No existing artifact modified.
- [ ] Historical ordering violation recorded per OQ-1's resolution.
- [ ] Codex review completed.
- [ ] Material Codex findings independently verified by Claude.
- [ ] `HANDOFF.md` written.

The task owner does not mark the project stage DONE. Master reviews; the **user** accepts.

---

## 14. Codex Review Requirement

Scope Codex to this task. Provide this `TASK.md`, the diff, `RESULTS.md`, and the two registry entries.

Ask Codex specifically to attack: (a) whether the runner can be made to overwrite a record under any path, including crash-and-resume; (b) whether either gate can return PASS on input it should refuse; (c) whether the float64 measurement decision is a measurement fix or a disguised threshold widening.

If Codex is unavailable, that fact must reach the user review brief. It may not be silently omitted.

---

## 15A. Required Completion Reporting — Finding vs Decision

This task may implement and execute within the scope above. It may **not** resolve OQ-1 or OQ-2, alter a registry criterion, widen a threshold, repair a failing cloud, or declare the corpus valid. A gate verdict is an observation; accepting it as project state is the user's.

---

## 17. Master Recommendation and the User Review Gate

Execution `COMPLETE` is not acceptance. Master reviews and issues a **MASTER RECOMMENDATION**; only the user's `APPROVE` makes it project state. See `workflow/WORKFLOW.md` §13A/§13B.

---

## Contract provenance

Written by Master 2026-08-21 at the user's instruction ("03 沒做指派任務 … 05 也是需要驗證 … 將這幾個驗證指派任務"), against artifact items **03** (`G1_sources_valid`) and **05** (`G2_pc_sanity`).
Baseline commit `468bbac`. Gate absence verified the same day by direct filesystem inspection.
