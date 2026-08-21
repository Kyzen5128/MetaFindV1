# USER REVIEW BRIEF

**Task ID:** `D2a_stage1-protocol-refresh`
**Master Recommendation:** `ACCEPT WITH FOLLOW-UP`
**Integration status:** **`USER_APPROVED` 2026-08-21. FINAL ACCEPTED.**

**OUTCOME — `APPROVE`, 2026-08-21.** The task is accepted and **MIF-2 is ratified by the user**: `n05b`'s rewrite of `data/outputs/variant_registry.json` is an accepted, disclosed scope deviation with no content effect. Recorded in `workflow/DECISION_LEDGER.md` `DL-003`. The contract was **not** retroactively edited to pretend the write was pre-authorised.

---

## 1. What was found

| # | Finding |
|---|---|
| FIND-1 | Every acceptance condition holds. **`AC-1.a`: a bare `annotate_run` queues 0 records TOTAL** — 0 legacy-v3, 0 residuals. Proven with no model load and no GPU |
| FIND-2 | **`AC-1.b` capability survives**: `--force` still queues all 45,955. The accident was removed, not the ability |
| FIND-3 | **`AC-1.c` is satisfied by explicit state, not by absence.** The registry declares `accepted_legacy_v3: 45,952` and `legacy_v1_residual_unresolved: 3` |
| FIND-4 | τ artifacts correct: `init_temperature 0.5`, `learnable_temperature False`. `load_protocol()` returns `metafind_v2_cm@8e4b1fcc66c7f48c` |
| FIND-5 | Corpus untouched: 45,952 v3 + 3 v1, **no v4 contract id anywhere**, embeddings 5,276, checkpoints 0 |
| FIND-6 | Codex found **two real AC-1 holes** — a JSON-`null` sidecar could be re-queued, and unchanged `prompt_version` with changed content could leave provenance stale. Both fixed, both with regression tests. **Retained in the finding history; not deleted because they were fixed** |
| **MIF-2** | **n05b writes three artifacts, not two.** `variant_registry.json` was also rewritten. My contract declared two. **This is a defect in Master's contract, not in the executor's conduct** |
| **FIND-4′** | `sidecar_path()` performs no uid validation — a uid containing `../` escapes `paths.ANNOTATIONS`. **Pre-existing, not introduced by D2a** |

---

## 2. Evidence / provenance

Master re-ran the load-bearing checks rather than accepting the report.

| Claim | Master verified? | Result |
|---|---|---|
| τ = 0.5, learnable false | **direct** | Read the artifact. `0.5` / `False` |
| `load_protocol()` passes | **direct** | Returned `metafind_v2_cm@8e4b1fcc66c7f48c` |
| **AC-1.a** | **direct, and it took two attempts** | My first reproduction used `is_complete()` and reported **45,955** — a false negative. The mechanism relocated the filter to `build_work_list()` (`annotate_run.py:439`), exactly the case TASK §12.1's **adaptation clause** anticipated. Re-run through `main()`'s own predicate: **todo TOTAL = 0**, legacy-v3 0, residuals 0 |
| **AC-1.b** | **direct** | `build_work_list(force=True)` → **45,955** |
| **AC-1.c** | **direct** | State histogram: `{accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}`. All 3 residuals explicitly `legacy_v1_residual_unresolved` |
| No v4 id on v3 | **direct** | `annotation_contract`: 45,952 absent, 3 explicitly `null`. Nothing carries the current id |
| Corpus intact | **direct** | `prompt_version` = `{3: 45952, 1: 3}`; embeddings 5,276; checkpoints 0 |
| **MIF-2 byte-identical** | **direct, proven not assumed** | `VARIANTS` does not appear in the `resolve_stage1.py` diff, and `_write()` is a deterministic `json.dump(indent=1)`. An unchanged constant through a deterministic writer **must** reproduce identical bytes |
| **FIND-4′ pre-existing** | **direct** | `sidecar_path` does not appear in D2a's diff |
| Suites and gates | **direct** | `547 passed` · `check_graph` 2275 all pass · `PRE-FLIGHT PASSED` |

---

## 3. Claude ↔ Codex disagreement

Two adversarial rounds. **No material disagreement remains.**

Codex found two genuine AC-1 defects that Claude's own verification had missed, both of which would have let a record be re-queued or its provenance go stale. Both were fixed with regression tests, and both are preserved in the finding history at your instruction.

> Convergence is not confirmation — but here Codex did not merely converge. It broke the mechanism twice before it held.

---

## 4. Verified conclusion

```
CONFIRMED:   FIND-1 … FIND-6, MIF-2, FIND-4′. Every Codex finding reproduced
REJECTED:    none
UNVERIFIED:  the pre/post checksum equality of the 45,955 annotation records was
             not re-derived by Master against a pre-task baseline — Master has no
             pre-task fingerprint. The population counts and prompt_version spread
             ARE directly confirmed, and no annotation file appears in the diff
```

---

## 5. Proposed / implemented decisions

| # | Decision | Authority | Classification |
|---|---|---|---|
| 1 | τ = 0.5 written to the artifact | `3experiments.tex:15` | **PAPER FACT** |
| 2 | `learnable_temperature = false` | your ratification 2026-08-21 | **USER-RATIFIED IMPLEMENTATION CHOICE.** The paper *does* use "learnable" (`2methdology.tex:54`, `:87`) but calls τ a "temperature hyperparameter" twice (`:79`, `:99`). Strong inference, **not** a paper fact |
| 3 | AC-1 mechanism = declared registry + relocated work-list predicate | TASK §7.1, your approval | **IMPLEMENTATION CHOICE**, within the approved menu |
| 4 | `data/outputs/annotation_provenance.json` as the registry | your approval 2026-08-21 | **USER DECISION** |
| 5 | Legacy-v3 provenance formalized for the 45,952 only | your decision #3 | **USER DECISION** |
| 6 | **MIF-2 — `variant_registry.json` rewritten** | **not previously authorised** | **DISCLOSED SCOPE DEVIATION.** Zero content effect, but the write surface exceeded the declared contract. **Needs your explicit ratification** |
| 7 | `docs/graph/README.md:270` counts 413→435, 525→547 | your narrow exception | Documentation. Master verified both figures |

---

## 6. Impact

- **Unblocked:** nothing yet. `D1` stays blocked by design.
- **Next in chain:** `D10` returns for final USER REVIEW carrying this task's AC-1 evidence. Only **D10's** approval unblocks `D1`.
- **Artifacts:** `stage1_hyperparameters.json`, `stage1_encoding_protocol.json`, `variant_registry.json` (see MIF-2), new `annotation_provenance.json`, new `tools/declare_annotation_provenance.py`.
- **Corpus:** unchanged. `D0-003` remains **unresolved** — the 3 residuals are explicitly labelled as such and nothing claims otherwise.

---

## 7. Remaining UNKNOWN / follow-up

| ID | Item | Owner |
|---|---|---|
| **F-1** | **Ratify or reverse MIF-2** — the `variant_registry.json` write | **you** |
| **F-2** | **FIND-4′** — `sidecar_path()` uid validation. Pre-existing robustness gap; no evidence it breaks AC-1 or any Stage 1 prerequisite | follow-up, unassigned |
| F-3 | `tools/declare_annotation_provenance.py` is new. §9.1 allowed "a new file under `tools/` only if the AC-1 proof needs a runnable **check**" — this is a **declaration tool**, not a check. **Second under-declaration in my contract**, same class as MIF-2 | Master |
| F-4 | TASK §12.1's AC-1.a snippet is now **stale** and yields a false negative. It must not be reused as-is | Master |
| F-5 | Master has no pre-task corpus fingerprint, so the executor's checksum-equality claim is accepted on report | Master |
| F-6 | Pre-flight still WARNS that all 45,955 records predate the declared annotation contract. Expected under your decision #3; not a failure | — |

---

## 8. USER ACTION REQUIRED

**Do you accept `D2a_stage1-protocol-refresh` as complete, and do you ratify the MIF-2 scope deviation?**

Two things are being asked at once:

1. **Acceptance of the task.** Every acceptance condition was independently confirmed by Master.
2. **Ratification of MIF-2.** `n05b` rewrote a third artifact my contract did not declare. The content is provably unchanged, but the write surface was not authorised. I have **not** edited the contract to pretend otherwise.

- `APPROVE` — accept the task and ratify MIF-2
- `REJECT`
- `MODIFY` — e.g. accept the task but withhold MIF-2 ratification, or attach conditions
- `INVESTIGATE MORE`
