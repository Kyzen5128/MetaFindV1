# USER REVIEW BRIEF

**Task ID:** `D10_stage1-encoding-contract`
**Master Recommendation:** `ACCEPT WITH FOLLOW-UP`
**Integration status:** `AWAITING_USER_REVIEW / MODIFIED` — **not** `USER_APPROVED`
**Migration case** — `WORKFLOW.md` §19. Executed pre-gate. Implementation, Codex review, and verification were **not** re-run.

Your seven decisions (2026-08-21) are recorded in §5 and folded into the follow-ups.

**Revision 2 — 2026-08-21. User returned `MODIFY`.** The core implementation is approved in principle. **FIND-8 is elevated from a follow-up to a blocking acceptance condition (`AC-1`, §7.0).** D10 must not be marked `USER_APPROVED` until `AC-1` is satisfied.

---

## 1. What was found

| # | Finding |
|---|---|
| FIND-1 | All four blocker exit criteria hold. `is_complete()` now binds to the serialized text: **5,276 stale sidecars → 0 cache-valid.** No encoder was run to prove it |
| FIND-2 | The ratified template is implemented and emits byte-for-byte. **161 zero-dimension renders → 0. 3,643 ungrammatical articles → 0. Over-77 records 1 → 0** (max now 72) |
| FIND-3 | `load_protocol()` now **refuses the on-disk protocol**. This is not theoretical — n06 is hard-blocked today until n05b re-runs |
| FIND-4 | **The recorded Stage 1 critical path is wrong and now unexecutable.** `CONTEXT.md` §7 says `D1 → D2`; D0-008 item 6 says `n05b → n06`. The contradiction predates D10; D10 made it fatal rather than latent |
| FIND-5 | **τ = 0.5 has no code path.** `resolve_stage1.py:243` hardcodes `0.07`; there are **0 CLI flags** for it. Nothing currently owns that change |
| FIND-6 | E-3's specified remedy was a **deterministic no-op** — re-annotation produced a byte-identical record (`md5 aeaea2fd…` unchanged), because decoding is greedy. Fixed by hand translation on your directive, overriding TASK §8 |
| FIND-7 | Four annotation-pipeline gaps were open at once (prompt did not state language, validator did not check it, repair loop could not repair it, token budget was recorded not enforced). P-1…P-5 close them |
| FIND-8 | **BLOCKING — conflicts with your decision #3.** `annotate_run.is_complete()` keys on `annotation_contract`; **all 45,955 records carry none.** Master reproduced the work-list computation read-only: a bare `python -m metafind.data.annotate_run` (no `--force`) would queue **45,955 records** — the whole corpus, v1 residuals included — for re-annotation. At n05's observed 39/min that is ~19.6 GPU-hours **overwriting the legacy-v3 corpus you decided to preserve** |
| FIND-9 | The new identity is enforced **at n06 only**. `stage1.py` and `gallery_index.py` still load NPZ with no sidecar check (MIF-D10-3) |

---

## 2. Evidence / provenance

| Finding | Source | Master verified? |
|---|---|---|
| FIND-1, FIND-2 | runtime — ran `tools/preflight_stage1_text.py` myself | **direct.** Reproduced every number: 45,955 work list · 45,952 valid v3 · 3 residuals · 0 violations · 0 mismatches · 0 zero-dims · 0 over-77 (max 72) · 5,276 sidecars → **0 cache-valid**, independent recount 0 |
| FIND-3 | runtime — called `load_protocol()` | **direct.** `ValueError: … records 'metafind_v1_natural', but this process's serializer is 'metafind_v2_cm@8e4b1fcc66c7f48c'` |
| FIND-4 | decision file + `CONTEXT.md` §7 | **direct** — the two documents contradict each other |
| FIND-5 | code | **direct** — `resolve_stage1.py:243`; grep for temperature CLI flags returns **0** |
| FIND-6 | data | **direct** — compared all 3 records against `annotations_v3_pre_D10/`. CJK gone, pure ASCII, `prompt_version` still 3, backups intact |
| FIND-8 | code + runtime | **direct** — `annotate_run.py:98` requires `annotation_contract == annotation_contract_id()`; `annotate_run.py:250` builds `todo` from it. Master recomputed the work list without loading any model: **45,955 of 45,955 would be queued.** No script auto-invokes n05 (`grep` over `tools/`: monitoring only), so the exposure is a manual bare invocation |
| FIND-7, FIND-9 | code + D10's report | **on report** — not re-read line by line |
| test suite / graph | runtime | **direct** — `525 passed`, `check_graph` 2275 all pass |
| artifact integrity | filesystem | **direct** — embeddings still **5,276**, checkpoints still **empty**, 3 backups present |

---

## 3. Claude ↔ Codex disagreement

Two adversarial rounds, both completed. **Not** `CODEX REVIEW UNAVAILABLE`.

**12 findings. Claude rejected none.** Every one was reproduced before being acted on.

Two were **confirmed but deliberately not fixed**, both on scope grounds, and Codex round 2 agreed with both escalations:
- **C-2** → became MIF-D10-3 (the identity is enforced at n06 only). Out of D10's scope.
- **C-8** → `docs/graph/README.md` and `TASK.md` fall outside the task's file list; both were forced by the contract itself and disclosed before review.

One was **partially accepted**: Codex wanted every NPZ opened and validated. Declined on cost (1.3 GB per pass, zero observed instances). Recorded as a residual risk.

Worth noting: Codex defeated Claude's *first* fix for two findings (C-1 and C-7), forcing a second, stronger pass each time. That is round 2 earning its cost.

> No material disagreement remains open.

---

## 4. Verified conclusion

```
CONFIRMED:   FIND-1 … FIND-9. All 12 Codex findings
             (10 fixed, 2 confirmed-and-escalated)
REJECTED:    none
UNVERIFIED:  FIND-7 and FIND-9 accepted on D10's report, not re-read line by line
             by Master. Neither is load-bearing for this recommendation
```

---

## 5. Your decisions, and what each one binds

| # | Your decision | Status |
|---|---|---|
| 1 | Ratify P-1…P-5 scope extension | **RATIFIED.** The scope extension exceeded TASK §7; you authorised it |
| 2 | Ratify the 3 manual translations | **RATIFIED as a recorded DEVIATION.** TASK §8 forbade hand-editing; you overrode it after being shown that re-annotation is a deterministic no-op. Must appear in the reproduction report's annotation-pipeline section |
| 3 | **No full v4 re-annotation.** Keep v3, record formally as **legacy-v3 corpus validated under VALIDATOR_VERSION 2** — not disguised as v4-generated | **BINDING.** No document, field, or report may describe the corpus as v4-generated. The code currently contradicts this decision — see `AC-1` |
| 4 | Do not start `D1_n06-reencode` | **BINDING.** D1 stays BLOCKED |
| 5 | New protocol artifact must precede n06 | **BINDING.** Resolves FIND-4 in favour of D0-008 item 6. `CONTEXT.md` §7's `D1 → D2` is wrong and will be corrected |
| 6 | Minimal prerequisite task for C-001 + C-002 only — do not pull D2 forward | **BINDING.** `D2` splits. Proposal below |
| 7 | D0-003's 3 legacy v1 records stay **unresolved** | **BINDING.** Pre-flight PASS does **not** resolve them. They remain a `FileNotFoundError` waiting in `stage1.py`'s loader |

### Classification of what D10 decided within scope

| Item | Classification |
|---|---|
| B-1 as text binding rather than a version bump | IMPLEMENTATION CHOICE, within B-1's stated menu |
| B-3 as content-addressed identity (`metafind_v2_cm@8e4b1fcc66c7f48c`) | IMPLEMENTATION CHOICE |
| Late template binding in `serialize_annotation()` | **bug fix** — the drift detector reported the new template while the function still emitted the old one |
| P-1…P-5 + contract versioning | IMPLEMENTATION CHOICE, **user-ratified (your #1)** |
| 3 hand translations | **DEVIATION**, user-ratified (your #2) |
| Legacy-v3 corpus retained | **USER DECISION (your #3)** |

---

## 6. Impact

- **Unblocked:** nothing yet. D1 remains blocked by your #4 and by FIND-3.
- **Blocked:** `D1` → `D2` → `D3` → `D4` → `D7`.
- **Dependency correction:** `n05b (C-001 + C-002) → n06`. `D0-002` and `D0-003` gate **n09 only** — verified: `splits.py` never reads an embedding, `encode_text_image.py` never reads anything n09 writes.
- **Artifacts:** `resolve_stage1.py`, `encode_text_image.py`, `annotate.py`, `annotate_run.py`, 3 annotation records (backed up), 1 new tool. **No embeddings, no checkpoints, no protocol artifact written.**

---

## 7. Remaining UNKNOWN / follow-up

### 7.0 `AC-1` — BLOCKING ACCEPTANCE CONDITION

**User decision, 2026-08-21.** FIND-8 is **not** an ordinary follow-up. It contradicts a standing USER DECISION, so it gates acceptance.

> **`AC-1`.** Before `D10_stage1-encoding-contract` may be marked `USER_APPROVED`, a safety mechanism must exist and be **demonstrated** such that, absent explicit force or a named migration intent, **no** existing annotation record — neither the accepted legacy-v3 corpus nor the legacy-v1 residuals — is automatically treated by `annotate_run` as requiring re-annotation.

**Why it is an acceptance condition and not a follow-up.** D10's contract-versioning change is correct in itself, but combined with your decision #3 it puts the code in active opposition to a decision you already made. A follow-up describes work that remains; this describes a state in which a single ordinary command destroys an artifact the project has formally decided to preserve. Those are not the same class of item.

**Current measured state — the condition is not met:**

```
rendered assets                                        45,955
would be queued by a bare `annotate_run` (no --force)  45,955
  of those, accepted legacy-v3                         45,952
  of those, legacy-v1 residuals (D0-003 unresolved)         3
current contract id      metafind_annot_v4@52f6b2c72fce2950
records carrying that contract                              0
```

**Satisfying `AC-1` requires all of:**

**Amended by the user 2026-08-21** — two corrections, both binding.

#### The three populations. Never conflate them.

| count | population | status |
|---|---|---|
| **45,952** | **accepted legacy-v3 corpus** | validated under `VALIDATOR_VERSION 2` |
| **3** | **legacy-v1 residuals** — `6c7db00c…`, `8a0192ee…`, `a397b648…` | **`D0-003` UNRESOLVED.** Not legacy-v3. Not migrated |
| **45,955** | total render / work population | 45,952 + 3 |

| | Requirement |
|---|---|
| AC-1.a | **A bare `python -m metafind.data.annotate_run` queues 0 records TOTAL** — not merely 0 legacy-v3. Verified today: it queues **45,955**, and **all 3 v1 residuals are in that queue**. If the 45,952 were protected but the 3 slipped through, `annotate_run` would rewrite them into the current schema and thereby **resolve `D0-003` by mutation**, before any decision was taken |
| AC-1.b | Re-annotation remains reachable through **explicit** force or a **named migration intent**. The mechanism removes the accident, never the capability |
| AC-1.c | Three states are **explicit in the record or in a declared registry**, not inferred from a missing field: annotated-under-current-contract · accepted-legacy-v3 · legacy-v1-residual-unresolved.<br><br>**Corrected by Master 2026-08-21 on the user's decision.** This row previously said only "in the record", which was narrower than intended and stale. `data/outputs/annotation_provenance.json` as a declared registry is **permitted**. `D2a`'s `TASK.md` §7.1 carries the authoritative wording |
| AC-1.d | Demonstrated **without** loading the annotation model or consuming GPU time |
| AC-1.e | The **45,952 only** are recorded as **legacy-v3 validated under `VALIDATOR_VERSION 2`**. They must not be relabelled as v4-generated and must not be given a v4 contract id they did not earn. **The 3 residuals must not be labelled legacy-v3**, and nothing may present `D0-003` as resolved |

**Where it is fixed:** proposed task `D2a_stage1-protocol-refresh`. **D10's own implementation is not reopened.**

**Until `AC-1` is satisfied:** D10 integration status is `AWAITING_USER_REVIEW / MODIFIED`. It is **not** `USER_APPROVED`, and no ledger entry may record it as FINAL ACCEPTED.

---

**Requires action before n06:**

| ID | Item | Owner |
|---|---|---|
| G-1 | C-001 (τ = 0.5) — **needs a code change**, no path exists | proposed task below |
| G-2 | C-002 protocol refresh through n05b | proposed task below |
| **AC-1** | **Legacy-v3 rerun protection — BLOCKING D10's acceptance**, §7.0 | `D2a` |
| **AC-1.e** | Record the corpus formally as **legacy-v3 validated under VALIDATOR_VERSION 2** | `D2a` |

**Master's own follow-ups:**

| ID | Item |
|---|---|
| G-5 | Correct `CONTEXT.md` §7 and `INDEX.md` per your #5 — done at acceptance |
| G-6 | Record the 3 translations as a DEVIATION; route MIF-D10-3 to `D3`/`D4` |
| G-7 | Ratify or revert D10's Decisions §5 (4 comment corrections, zero behavioural impact) and §6 (`math.isfinite()` guard placement) |

**Deferred / open:**

- **D0-003 remains UNRESOLVED** (your #7). 3 records, still a live `FileNotFoundError` for the trainer.
- Retrieval impact of the new template — **UNKNOWN**, as D0-008 states.
- `_dim()` inherits round-half-to-even; no `0.25` exists today. A future batch needs this revisited.
- NPZ *contents* unvalidated; `--force` bypasses B-1 by design.
- FU-8 (`r = 0.52-0.62`) and FU-9 from D0-008 remain deferred.

---

## 8. USER ACTION REQUIRED

Master's disclosure, since D10 flagged it and could not attribute it: **MIF-D10-4 — the +321-line change to `workflow/WORKFLOW.md` was mine**, made in the Master session at your instruction while D10 was running. D10 correctly refused to touch it.

### Current state after your `MODIFY`

| | |
|---|---|
| D10 implementation | **approved in principle** — serializer, `text_serialization_id` / cache validity, `load_protocol` rejection, pre-flight, >77 hard gate, P-1…P-5, contract versioning, the 3 authorised translations |
| D10 integration status | **`AWAITING_USER_REVIEW / MODIFIED`** |
| Blocking | **`AC-1`** — legacy-v3 rerun protection, §7.0 |
| `USER_APPROVED` | **withheld** until `AC-1` is satisfied |

No further action is required from you on D10 right now. The next decision point is approving `D2a_stage1-protocol-refresh`, which carries `AC-1`.

D10 returns for final acceptance once `D2a` demonstrates `AC-1`. At that point the action requested will be `APPROVE` / `REJECT` / `MODIFY` / `INVESTIGATE MORE` on D10's final state.
