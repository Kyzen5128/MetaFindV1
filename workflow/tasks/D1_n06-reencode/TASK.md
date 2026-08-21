# D-Task Execution Contract

> Authoritative execution contract for one bounded work package.
> Stay within scope, satisfy the Definition of Done, verify, obtain Codex review, return a HANDOFF to Master.

---

## Task ID

`D1_n06-reencode`

---

## Status

`PROPOSED` — TASK.md written 2026-08-21. **Awaiting user review and approval.**

The D1 execution conversation sets this to `ACTIVE` at start. That is the only edit to this file the executor may make.

---

## 1. Objective

**Run `n06_encode_text_image` to completion over the admitted corpus, producing the Stage 1 text and image embedding cache under the ratified serializer.**

One correctness boundary: when this task ends, every admitted asset has an embedding produced by `metafind_v2_cm@8e4b1fcc66c7f48c`, and no embedding from any other serializer survives in the cache.

**This is the first GPU-bearing task in the Stage 1 chain.** Roughly 4 hours.

---

## 2. Why This Task Exists

Three prerequisites are now `USER_APPROVED`, and D1 was blocked on all of them:

| | Ledger | What it gave D1 |
|---|---|---|
| `D0-008_stage1-text-template` | `DL-001` | The ratified template — what the encoder is allowed to emit |
| `D10_stage1-encoding-contract` | `DL-002` | Cache validity, `load_protocol()` binding, the pre-flight gate, the >77 hard gate |
| `D2a_stage1-protocol-refresh` | `DL-003` | τ = 0.5 written through n05b, the refreshed protocol, and corpus rerun protection |

**The cache is currently unusable and known to be so.** 5,276 sidecars exist; **0** are cache-valid under the ratified serializer. They are invalidated, not deleted — deliberately, so the evidence of the old text distribution survives.

**Downstream:** `D3_stage1-train` needs this cache **and** n09's splits. n09 is a separate half, still blocked on `D0-002` and `D0-003`.

---

## 3. Required Shared Context

Read, in order:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. `/home/kyzen/MetaFindV1/.claude/rules/code-changes.md`
3. `/home/kyzen/MetaFindV1/.claude/rules/experiments.md`
4. `/home/kyzen/MetaFindV1/workflow/WORKFLOW.md` §13A, §13B
5. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
6. this `TASK.md`

Then only the files named in §5 and §9. Do not re-read the repository.

---

## 4. Dependencies

### Required Before Start — all satisfied

- `D0-008` **`USER_APPROVED`** 2026-08-21 (`DL-001`)
- `D10` **`USER_APPROVED`** 2026-08-21 (`DL-002`)
- `D2a` **`USER_APPROVED`** 2026-08-21 (`DL-003`)

### Blocks

- `D3_stage1-train` — needs this cache **and** n09's three protocol files. n09 is not this task.

### Parallel Safety

`PARALLEL SAFE: NO` for anything touching `data/outputs/embeddings/`, `metafind/data/`, or `metafind/models/`.

A **read-only** task that writes only `docs/` or `workflow/decisions/` may run concurrently — `D0-009_essgnn-fx-codomain` is the current example. Master confirms at approval time; do not assume it.

---

## 5. Authoritative Inputs

| # | Source | Why |
|---|---|---|
| 1 | `workflow/DECISION_LEDGER.md` `DL-001`, `DL-002`, `DL-003` | What is ratified and under what authority |
| 2 | `metafind/data/encode_text_image.py` | The node. `NODE = "n06_encode_text_image"`, `IMPLEMENTS-NODE` at line 3 |
| 3 | `tools/preflight_stage1_text.py` | The B-4 gate. **Must PASS before the run and after it** |
| 4 | `data/outputs/stage1_encoding_protocol.json` | The bound protocol. `load_protocol()` refuses a foreign serializer |
| 5 | `data/outputs/annotation_provenance.json` | The declared registry. **Read-only for this task** |
| 6 | `.claude/rules/experiments.md` | Provenance requirements for a research-relevant run |

---

## 6. Current Relevant State

Master-verified 2026-08-21, read-only.

### Populations — never conflate them

| count | population | what n06 does with it |
|---|---|---|
| **45,955** | n06's work list — `annotations` ∩ `renders_index.jsonl` | **attempted** |
| **45,952** | accepted legacy-v3, validated under `VALIDATOR_VERSION 2` | **encoded successfully** |
| **3** | legacy-v1 residuals — `6c7db00c…`, `8a0192ee…`, `a397b648…` | **quarantined**, no `.npz`. `D0-003` stays UNRESOLVED |

The 3 raise `KeyError: 'width'` inside `serialize_annotation()` and are caught by the encode loop. **Quarantining them is not resolving them.**

### Gates, all green today

```
load_protocol()   PASS -> metafind_v2_cm@8e4b1fcc66c7f48c
preflight         PRE-FLIGHT PASSED   (0 template mismatches, 0 zero-dims, 0 over-77, max 72)
pytest tests/ -q  547 passed
check_graph.py    2275 checks, all pass
```

### Cache

`data/outputs/embeddings/` = 5,276 `.npz` + 5,276 sidecars. **0 cache-valid.** `is_complete()` binds to the serialized text (D10 B-1), so a resume re-encodes every one of them.

`ENCODER_VERSION = 1` — unchanged. It means "the CLIP encoder", and no encoder changed.

### Observed throughput

The 2026-08-17 partial run logged **~189/min**. Over 45,955 that projects to **~4 hours**. A projection, not a guarantee — record the actual rate.

### Known, disclosed, not this task's to fix

- **MIF-D10-3** — the serializer identity is enforced at n06 **only**. `stage1.py:110` and `gallery_index.py:215` load NPZ with no sidecar check; neither file mentions `text_serialization`. Routed to `D3`/`D4`. **Do not fix it here.**
- **G-7** — D10's four comment corrections and the `math.isfinite()` guard placement. **OPEN, not ratified.** Not this task's.
- **F-2** — `sidecar_path()` performs no uid validation. Pre-existing, LOCAL.

---

## 7. Scope

### In Scope

1. **Pre-run gate.** Confirm `load_protocol()` passes and the pre-flight PASSES **before** spending GPU time.
2. **Run n06 to completion** over the full work list.
3. **Verify the output** against the expected populations.
4. **Record experiment provenance** per `.claude/rules/experiments.md`: command, git state, config, seed if any, environment, hardware, duration, throughput, output path.
5. **Post-run gate.** Re-run the pre-flight and confirm the cache is now fully valid.

### Explicit Non-Scope

- ❌ **Do not run n09** (`metafind.data.splits`), training, or gallery indexing.
- ❌ **Do not re-annotate anything.** No `annotate_run` invocation that mutates a record.
- ❌ **Do not delete the 5,276 stale `.npz` or sidecars** unless overwriting them is the natural result of the run. Do not `rm -rf` the directory.
- ❌ **Do not decide or resolve `D0-003`.** The 3 residuals are quarantined, and that is all.
- ❌ **Do not fix MIF-D10-3, G-7, or F-2.**
- ❌ **Do not change the serializer, the template, the protocol, or any ratified artifact.**
- ❌ **Do not use `--force` to paper over a refusal.** If `load_protocol()` or the pre-flight refuses, **stop and report** — the refusal is the guard working.
- ❌ Do not touch `workflow/MASTER.md`, `CONTEXT.md`, `INDEX.md`, `DECISION_LEDGER.md`, or another task's files.

---

## 8. Master's Standing Rulings

**R-A — a refusal is a finding, not an obstacle.** `load_protocol()`, the pre-flight, and the >77 gate exist because this project has already been bitten by silent mismatches. If any refuses, **stop and report a `MASTER-IMPACTING FINDING`.** Do not bypass with `--force`, do not edit the artifact to match, do not adjust a threshold.

**R-B — quarantine is not resolution.** The 3 legacy-v1 residuals will be quarantined. Report the count and the uids. **Do not describe `D0-003` as resolved, closed, or handled.**

**R-C — the run is an experiment.** Its provenance must be reproducible: exact command, `git rev-parse HEAD` plus working-tree state, environment versions, GPU, start/end time, throughput, and the quarantine ledger. A completed process is not a completed experiment.

**R-D — do not delete evidence.** The 5,276 stale sidecars carry the only record of the old text distribution. Overwriting them through the normal encode path is expected; wiping the directory is not.

---

## 9. Allowed Writes / Protected Files

### 9.1 ALLOWED

| Path | For |
|---|---|
| `data/outputs/embeddings/**` | The `.npz` and sidecars n06 produces. **This is the task's output** |
| `data/outputs/logs/` — the n06 run log and its quarantine ledger | Run record |
| `workflow/tasks/D1_n06-reencode/HANDOFF.md`, `CODEX_REVIEW.md` | Required by §16 |
| `workflow/tasks/D1_n06-reencode/TASK.md` | Status line only — `PROPOSED` → `ACTIVE` at start |

### 9.2 PROTECTED — do not write, move, delete, or regenerate

| Path | Why |
|---|---|
| `data/outputs/annotations/**` | **All 45,955 must be byte-identical at completion.** Verify by checksum |
| `data/outputs/annotations_v1_prompt1/**`, `annotations_v2_sample/**`, `annotations_v3_pre_D10/**` | Backups |
| `data/outputs/annotation_provenance.json` | The declared registry. Read-only here |
| `data/outputs/stage1_encoding_protocol.json`, `stage1_hyperparameters.json`, `variant_registry.json` | Ratified by `DL-003`. n06 reads, never writes |
| `data/outputs/checkpoints/**` | Must stay empty — no training in this task |
| `metafind/**`, `tests/**`, `tools/**` | **No code change is in scope.** If you believe one is required, escalate |
| `docs/**` | Untouched |
| `workflow/` — anything outside `tasks/D1_n06-reencode/` | Master's, or another task's |

Anything not listed in 9.1 is protected by default.

---

## 10. Execution Requirements

1. **Gate before GPU.** Run `load_protocol()` and the pre-flight first. If either refuses, stop.
2. Capture the baseline: annotation checksum, embeddings count, checkpoints count, `git rev-parse HEAD`, environment versions, GPU.
3. Record the **exact** command. Do not paraphrase it afterwards.
4. Expect ~4 hours. Report the actual throughput, not the projection.
5. Never mutate an annotation record.
6. Report any Master-impacting discovery immediately.
7. Stop if a required authority decision is missing.

---

## 11. Master-Impacting Finding Rule

Report `MASTER-IMPACTING FINDING` for anything affecting project architecture, accepted research interpretation, a cross-task dependency, another task's contract, milestone feasibility, or a global runtime assumption. Include the finding, the evidence, affected tasks, and whether this task can safely continue.

**Escalate rather than decide:** any gate refusal · any quarantine beyond the expected 3 · any evidence bearing on `D0-003`, MIF-D10-3, or `D0-009`.

---

## 12. Verification Requirements

`PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python`, from the repository root.

### 12.1 Before the run

```bash
$PY -c "from metafind.data.encode_text_image import load_protocol; p=load_protocol(); print('OK', p['text_serialization'])"
$PY tools/preflight_stage1_text.py                                            # must PASS
find data/outputs/annotations -name '*.json' | sort | xargs md5sum | md5sum   # baseline fingerprint
ls data/outputs/embeddings/*.npz | wc -l                                      # 5276 at start
ls data/outputs/checkpoints | wc -l                                           # 0
git rev-parse HEAD; git status --porcelain
$PY -c "import torch;p=torch.cuda.get_device_properties(0);print(p.name, round(p.total_memory/1024**3,1),'GB')"
```

### 12.2 The run

Record the exact command and its full output, including the quarantine ledger.

### 12.3 After the run

```bash
ls data/outputs/embeddings/*.npz | wc -l                                      # expect 45,952
$PY tools/preflight_stage1_text.py                                            # must still PASS
find data/outputs/annotations -name '*.json' | sort | xargs md5sum | md5sum   # must equal the baseline
ls data/outputs/checkpoints | wc -l                                           # still 0
$PY -m pytest tests/ -q                                                       # 547 passed
$PY tools/check_graph.py                                                      # all pass
git status --porcelain
```

**Cache validity — the claim that matters.** Show that every produced embedding is valid under the ratified serializer, and that **0** remain invalid:

```bash
$PY - <<'EOF'
from metafind.data import encode_text_image as E
from metafind import paths
uids = [p.stem for p in sorted(paths.EMBEDDINGS.glob("*.json"))]
valid = [u for u in uids if E.is_complete(u, E.expected_text_for(u))]
print(f"sidecars              {len(uids):>7,}")
print(f"cache-valid           {len(valid):>7,}")
print(f"NOT cache-valid       {len(uids)-len(valid):>7,}   <-- must be 0")
EOF
```

### 12.4 Research fidelity

- Every sidecar records `text_serialization` = the ratified identity. **Zero** carry `metafind_v1_natural`.
- Sample sidecars and confirm the text matches the ratified template — centimetres, no `"A "` article, capitalised category.
- **0** records report `text_truncated`.

---

## 13. Definition of Done

- [ ] **1.** `load_protocol()` and the pre-flight both PASSED **before** GPU time was spent.
- [ ] **2.** n06 ran to completion over the full work list. Exact command recorded.
- [ ] **3.** `data/outputs/embeddings/` holds **45,952** `.npz`.
- [ ] **4.** Exactly **3** assets quarantined — the known legacy-v1 residuals, by uid. Any other quarantine is escalated, not absorbed.
- [ ] **5.** **0** sidecars are NOT cache-valid under the ratified serializer.
- [ ] **6.** **0** sidecars carry `metafind_v1_natural`. **0** report `text_truncated`.
- [ ] **7.** Pre-flight still PASSES after the run.
- [ ] **8.** All 45,955 annotation records byte-identical to the baseline.
- [ ] **9.** `checkpoints/` still empty; n09 and training never invoked.
- [ ] **10.** `pytest tests/ -q` 547 passed; `check_graph.py` all pass.
- [ ] **11.** `git diff` touches only §9.1 paths. **No code change.**
- [ ] **12.** Experiment provenance recorded per `.claude/rules/experiments.md` — command, git state, environment, GPU, duration, throughput, output path, quarantine ledger.
- [ ] **13.** Nothing states or implies that `D0-003` is resolved.
- [ ] **14.** Codex review completed; material findings independently verified by Claude.
- [ ] **15.** `HANDOFF.md` and `CODEX_REVIEW.md` written, including `USER REVIEW INPUT`.

The task owner does not mark the Stage 1 milestone DONE. Master reviews; the user decides.

---

## 14. Codex Review Requirement

Scope to this task. Provide Codex with this `TASK.md`, the run log, the verification output, `DL-001`/`DL-002`/`DL-003`, and known uncertainties.

Ask Codex to attack:

- whether the cache-validity proof is genuine or merely restates `is_complete()`;
- whether any `.npz` could be present-but-wrong — truncated, empty, or from a different serializer — and still counted valid;
- whether the 3 quarantines are the expected ones, and whether any *other* asset silently failed;
- whether the run could have partially used a stale sidecar through a resume path;
- whether anything in the output implies `D0-003` is resolved;
- whether the recorded provenance is actually sufficient to reproduce the run;
- whether the annotation corpus is genuinely untouched, not merely reported as such.

Codex must not be asked merely to confirm.

---

## 15. Claude Verification of Codex Findings

Classify each material finding: `CONFIRMED` · `PLAUSIBLE` · `REJECTED` · `UNVERIFIED`.

If review is unavailable: `CODEX REVIEW UNAVAILABLE`. **Not a PASS**, and it must reach the HANDOFF's `USER REVIEW INPUT`.

---

## 16. Required Handoff

Write `HANDOFF.md` and `CODEX_REVIEW.md` per `workflow/tasks/HANDOFF_TEMPLATE.md`, including §15 `USER REVIEW INPUT`.

Report findings and decisions **separately** (`WORKFLOW.md` §13A).

State explicitly that **no n09 run, no training, no gallery indexing, and no annotation mutation occurred**, and that `D0-003` remains **UNRESOLVED**.

Then stop. Do not start `D3_stage1-train` or n09.

---

## 17. Master Recommendation and the User Review Gate

Master reviews and returns a **MASTER RECOMMENDATION**, not an acceptance.

```
task completion
→ Master integration review
→ MASTER RECOMMENDATION
→ USER REVIEW BRIEF
→ USER decision: APPROVE / REJECT / MODIFY / INVESTIGATE MORE
→ on APPROVE: FINAL ACCEPTED
```

Until the user approves: integration is `AWAITING_USER_REVIEW`, the task is **not** DONE, `D3` is **not** unblocked, and no global state file records the result.

**`D3_stage1-train` needs more than this task.** It also requires n09's `splits.json`, `eval_protocols.json`, and `stage1_protocol.json` — which remain blocked on `D0-002` and `D0-003`. D1's acceptance alone does not unblock D3.
