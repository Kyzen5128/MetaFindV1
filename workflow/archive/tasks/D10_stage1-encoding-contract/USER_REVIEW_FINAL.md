# D10 FINAL USER REVIEW BRIEF

**Task ID:** `D10_stage1-encoding-contract`
**Review basis:** D10's `TASK.md`, `HANDOFF.md`, `CODEX_REVIEW.md`, `USER_REVIEW.md` (Rev 2), plus `AC-1` evidence produced by `D2a_stage1-protocol-refresh` (`DL-003`, `USER_APPROVED` 2026-08-21).
**Not re-run:** D10's implementation, its Codex review, its verification. No new contradictory evidence was found, so none was re-run.
**Integration status:** **`USER_APPROVED` 2026-08-21. FINAL ACCEPTED.**

**OUTCOME — `APPROVE`, 2026-08-21.** The user accepted D10 as FINAL. `AC-1` cleared as the sole blocking condition. **`G-7` is explicitly NOT ratified by this approval** — it remains an independent OPEN follow-up (§5).

---

## 1. DECISION REQUIRED

You returned `MODIFY` on D10 (2026-08-21): the implementation was approved in principle, but acceptance was **withheld** pending one blocking acceptance condition, **`AC-1`** — legacy-corpus rerun protection.

`AC-1` has since been satisfied by `D2a`, which you accepted as FINAL on 2026-08-21.

**The question now is only this: does `AC-1`'s satisfaction clear the one condition that was withholding D10's acceptance?**

Nothing else about D10 has changed.

---

## 2. WHAT D10 ESTABLISHED

Approved in principle by you on 2026-08-21, unchanged since:

| | |
|---|---|
| Ratified serializer | Emits the D0-008 template byte-for-byte. Identity `metafind_v2_cm@8e4b1fcc66c7f48c` |
| `text_serialization_id` / cache validity | `is_complete()` binds to the serialized text. **5,276 stale sidecars → 0 cache-valid** |
| `load_protocol()` mismatch rejection | Refuses a foreign serializer rather than encoding under a false label |
| Stage 1 text pre-flight | Full-corpus gate, read-only, ~40 s, no GPU |
| >77 true-token hard gate | Over-limit text is quarantined, not encoded |
| P-1 … P-5 | Prompt states language · validator refuses non-English · repair loop repairs language · token budget enforced · provenance schema declared |
| Annotation contract versioning | `PROMPT_VERSION 4` / `VALIDATOR_VERSION 2` / `SCHEMA_VERSION 2` + fingerprint |
| 3 manual translations | User-authorised, backed up, reversible |

**Measured effect, re-verified by Master today:** 161 zero-dimension renders → **0**. 3,643 ungrammatical articles → **0**. Over-77 records 1 → **0** (max now 72).

**Master integration check, run today:** D2a did **not** disturb D10's mechanism. 5,276 sidecars on disk, **0 still cache-valid**. B-1 holds after D2a's changes.

---

## 3. AC-1 FINAL EVIDENCE

All five sub-conditions. Master reproduced each **directly**, read-only, with no model load and no GPU.

| | Requirement | Evidence | Status |
|---|---|---|---|
| **AC-1.a** | Bare `annotate_run` queues **0 records TOTAL** | `build_work_list(force=False)` — `main()`'s own predicate (`annotate_run.py:439`) → **todo TOTAL = 0** · accepted legacy-v3 queued **0** · legacy-v1 residuals queued **0** | **SATISFIED** |
| **AC-1.b** | Re-annotation still reachable through explicit force / named migration intent | `build_work_list(force=True)` → **45,955**. Capability removed nothing | **SATISFIED** |
| **AC-1.c** | Three states explicit in the record **or a declared registry**, never inferred from a missing field | Declared registry `data/outputs/annotation_provenance.json`. State histogram: `{accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}` | **SATISFIED** |
| **AC-1.d** | Demonstrated without loading the annotation model or consuming GPU | Master's reproduction loaded no model and used no GPU | **SATISFIED** |
| **AC-1.e** | The **45,952 only** as legacy-v3 / `VALIDATOR_VERSION 2`; the 3 residuals **not** legacy-v3; nothing presents `D0-003` as resolved | Registry declares the 45,952 as `accepted_legacy_v3`; all 3 residuals explicitly `legacy_v1_residual_unresolved`. `annotation_contract`: 45,952 **absent**, 3 explicitly `null` — **no fake v4 id anywhere** | **SATISFIED** |

**One note on how this was verified.** Master's first reproduction used `is_complete()` and reported 45,955 — a false negative. The mechanism relocated the predicate to `build_work_list()`, exactly the case D2a's contract anticipated with its adaptation clause. Re-run through the real predicate, the result is 0. **The 0 is real; the 45,955 was Master checking a path the production run no longer takes.**

**`D0-003` remains UNRESOLVED.** The 3 legacy-v1 residuals are labelled `legacy_v1_residual_unresolved` in the registry. `AC-1`'s satisfaction protects them from silent mutation — **it does not decide their disposition.** Nothing in D10, D2a, or this brief claims `D0-003` is resolved.

---

## 4. MATERIAL FINDINGS

### The blocking condition — now cleared

`AC-1` was D10's **only** blocking acceptance condition. All five sub-conditions are satisfied and independently verified.

### D10's original findings — none is a remaining material blocker

| Finding | Disposition |
|---|---|
| MIF-D10-1 — E-3's remedy was a deterministic no-op; fixed by hand translation overriding TASK §8 | **You ratified it** 2026-08-21 (decision #2). Recorded as a DEVIATION. Backups intact |
| MIF-D10-2 — the Stage 1 critical path was inverted; `D1 → D2` was unexecutable | **You resolved it** 2026-08-21 (decision #5). `D2a` executed the corrected order and is FINAL ACCEPTED |
| MIF-D10-3 — the identity is enforced at n06 **only** | **Still open.** Correctly escalated by D10, out of its scope, Codex round 2 agreed. Re-verified today: `stage1.py:110` and `gallery_index.py:215` load NPZ directly, and **neither file mentions `text_serialization`**. Routed to `D3` / `D4`. **Not a D10 blocker** — n10 cannot run today (`splits.json` absent, `checkpoints/` empty) |
| MIF-D10-4 — `WORKFLOW.md` modified by another session | **That was Master**, at your instruction, concurrently. D10 correctly refused to touch it. Disclosed |
| Codex: 12 findings over 2 rounds | 10 fixed, 2 confirmed-and-escalated (C-2 → MIF-D10-3; C-8 → disclosed file scope). **Claude rejected none.** All reproduced before being acted on |

### Disclosed and unchanged

D10's residual risks stand as disclosed and none was elevated: `--force` bypasses B-1 by design · B-1 catches text drift, not encoder drift · NPZ *contents* unvalidated (Codex's full-validation proposal declined on cost, zero observed instances) · `_dim()` inherits round-half-to-even, no such value exists today · non-finite dimensions caught at the gate, not in the serializer.

---

## 5. REMAINING UNKNOWNS / FOLLOW-UPS

### Owed by Master before D10 closes cleanly

| ID | Item |
|---|---|
| **G-7** | Ratify or revert D10's **Decisions §5** — four in-code comment corrections outside R-1/R-2/R-3, zero behavioural impact — and **Decisions §6** — whether the `math.isfinite()` guard belongs in `serialize_annotation()` rather than only in the pre-flight. Corpus impact today: 0 records. **Still open** |

### Routed onward, not D10 blockers

| ID | Item | Owner |
|---|---|---|
| MIF-D10-3 | Centralise protocol validation and require a sidecar identity match before any NPZ is consumed | `D3`, `D4` |
| MIF-D10-2 residual | 2 remaining non-truncated CJK records (10 non-ASCII total) | `D0-003` scope |
| FU-8 / FU-9 (from D0-008) | Reproduce or retire `r = 0.52-0.62`; latent zero-impact `_cap()` defects | deferred |

### Explicitly NOT D10 blockers

`D2a`'s follow-ups **F-2**, **F-3**, **F-5** are **not** D10 blockers, and no evidence suggests they break D10 correctness:

- **F-2** — `sidecar_path()` performs no uid validation. **Pre-existing**, confirmed absent from both D10's and D2a's diffs. Not exploitable in this pipeline without a hand-written malicious `--uids-file`. Does not touch D10's B-1/B-2/B-3/B-4.
- **F-3** — Master under-declared `tools/` scope in D2a's contract. A defect in **Master's contract**, not in D10.
- **F-5** — Master holds no pre-task corpus fingerprint for D2a. A limitation of **Master's** verification of D2a, not of D10.

### Genuinely unknown

- **Retrieval impact of the new template — UNKNOWN**, as D0-008 states. Zero token cost is not zero embedding impact. What is established is that 161 false statements and 3,643 ungrammatical constructions no longer reach the encoder.
- `D0-003` — **UNRESOLVED**, by design.

---

## 6. DOWNSTREAM IMPACT

**If you approve:**

```
D2a USER_APPROVED ✅
   → D10 USER_APPROVED
      → D1_n06-reencode UNBLOCKS   (~4 GPU-hours, 45,952 embeddings expected)
```

`D1` becomes the first task in the chain that is actually runnable: `load_protocol()` passes, the pre-flight passes, the corpus is protected, and every stale embedding is invalidated without being deleted.

**Still blocked after `D1`:** `D2` (n09 half) needs `D0-002` and `D0-003`. `D3` needs `D1` + n09, and inherits MIF-D10-3. `D4`, `D7` follow.

**Artifacts affected by D10:** `resolve_stage1.py`, `encode_text_image.py`, `annotate.py`, `annotate_run.py`, `tools/preflight_stage1_text.py`, 3 annotation records (backed up), `docs/graph/README.md`.

**Not affected:** embeddings still 5,276 and untouched; `checkpoints/` still empty; the corpus still 45,952 v3 + 3 v1.

---

## 7. MASTER RECOMMENDATION

## `ACCEPT WITH FOLLOW-UP`

**Reasoning.** `AC-1` was the single blocking condition you named, and all five sub-conditions are now satisfied and independently verified by Master rather than accepted on report. D10's own findings are each either ratified by you, resolved by `D2a`, or correctly escalated out of scope. No new contradictory evidence emerged during this integration review, so nothing was re-run.

The one item genuinely owed is **G-7** — your ratification or reversal of D10's four comment corrections and the `math.isfinite()` guard placement. Both have zero behavioural impact today and neither justifies withholding acceptance; they need a decision, not a gate.

**Master does not mark D10 `USER_APPROVED`.** That is yours.

---

## 8. USER ACTION

**Does `AC-1`'s satisfaction clear the condition that was withholding D10's acceptance — and do you accept `D10_stage1-encoding-contract` as FINAL, with G-7 and the routed follow-ups outstanding?**

- `APPROVE` — D10 becomes `USER_APPROVED`; `D1_n06-reencode` unblocks
- `REJECT`
- `MODIFY` — e.g. accept but rule on G-7 now, or attach conditions
- `INVESTIGATE MORE`

`D1` has not been started and will not be until you decide.
