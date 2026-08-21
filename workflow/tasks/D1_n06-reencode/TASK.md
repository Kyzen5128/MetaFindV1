# D-Task Execution Contract

> Authoritative execution contract for one bounded work package.
> Stay within scope, satisfy the Definition of Done, verify, obtain Codex review, return a HANDOFF to Master.

---

## Task ID

`D1_n06-reencode`

---

## Status

`ACTIVE` — **APPROVED by the user 2026-08-21**, after two rounds of contract review (one `MODIFY` cycle covering the exact command, resume semantics, the complete write surface, the post-run NPZ audit, and D0-009 parallel safety).

The D1 execution conversation sets this to `ACTIVE` at start. That is the only edit to this file the executor may make.

**Baseline at approval:** `git rev-parse HEAD` = `cf234fb`, working tree clean apart from this file. `D0-008` (`DL-001`), `D10` (`DL-002`), `D2a` (`DL-003`) all `USER_APPROVED`.

**On §4's D0-009 constraint:** `D0-009_essgnn-fx-codomain` reported `INVESTIGATION COMPLETE` on 2026-08-21 and is awaiting Master integration review. It is not writing. No separate git worktree exists, so §4's read-only condition applies for the duration of D1's run — and is satisfied by D0-009 having finished. **If D0-009 is re-opened for rework while D1 is running, §4's five conditions bind.**

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

### `D0-009_essgnn-fx-codomain` — **`PARALLEL SAFE: YES WITH WORKTREE ISOLATION`**

User ruling, 2026-08-21. The two have **no scientific dependency**. But D1 is an **experiment** and records `git rev-parse HEAD` plus working-tree state, so a concurrent writer would pollute D1's provenance.

D0-009 may continue researching during the ~4-hour GPU run **only if all five hold**:

| | |
|---|---|
| 1 | It does **not** modify the same working tree D1 is using |
| 2 | It does **not** touch `metafind/**` |
| 3 | It does **not** touch `data/**` |
| 4 | It does **not** touch D1's task files |
| 5 | It does **not** `commit`, `checkout`, or `reset` the worktree D1 is running in |

**Without a separate git worktree, D0-009 is read-only for the duration.** Any `workflow/decisions/` write waits until D1's GPU run has finished, then lands.

This is deliberate: the ~4 hours are usable for research **without** contaminating D1's experiment provenance.

**D1's obligation:** record the working-tree state at run start **and** at run end. If they differ, say so in the HANDOFF — do not silently report only one.

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
| `data/outputs/embeddings/**` | The `.npz`, the `.json` sidecars, and the `.part.npz` / `.json.part` temporaries n06 writes then renames. **This is the task's output** |
| `data/outputs/logs/quarantine_n06_encode_text_image.jsonl` | `runlog.quarantine()` — `runlog.py:146`. The quarantine ledger |
| `data/outputs/logs/run_progress.jsonl` | `runlog.run_progress(NODE)` — `runlog.py:74`, `:86`. **Shared append-only ledger used by every node** |
| `data/outputs/logs/cost_ledger.jsonl` | `runlog.cost_ledger()` — `runlog.py:129`. **Shared append-only ledger used by every node** |
| a run log file under `data/outputs/logs/` | Only if you redirect stdout. Name it in the HANDOFF |

> **The last two are shared, append-only ledgers.** n06 may **append** to them through its normal run path and nothing else. **Do not truncate, rewrite, reorder, deduplicate, or clean them.** Other nodes' entries live there.
>
> This list is the **complete** verified write surface of `n06`, traced through `encode_text_image.py` and `runlog.py`. It replaces an earlier vaguer `data/outputs/logs/` entry that under-declared it — the same defect class as `MIF-2` in `DL-003`.
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
2. Capture the baseline: annotation checksum, embeddings count, checkpoints count, `git rev-parse HEAD`, working-tree state, environment versions, GPU.
3. Use the **production command in §12.2 verbatim**. Do not paraphrase it afterwards.
4. Expect ~4 hours. Report the actual throughput, not the projection.
5. Never mutate an annotation record.
6. Report any Master-impacting discovery immediately.
7. Stop if a required authority decision is missing.

---

### 10.1 Resume / interruption / restart semantics

A four-hour GPU run will sometimes be interrupted. **SSH drop, terminal disconnect, process crash, machine reboot — the response is the same.**

**Do NOT, under any circumstance:**

- ❌ delete `data/outputs/embeddings/`
- ❌ empty or clear the sidecars
- ❌ `rm -rf` anything
- ❌ force a full overwrite from scratch
- ❌ add `--force` to "make sure it's clean"

**Do:** re-issue the **exact same production command** from §12.2. Nothing else.

**Expected resume behaviour** — this is the contract, and it follows from D10's B-1 text binding:

| Asset state | Expected |
|---|---|
| No sidecar / no `.npz` | **encode** |
| Sidecar exists but is **not** cache-valid under the ratified serializer | **encode** |
| One of the original **5,276 stale** sidecars — text mismatch | **encode.** They are invalid by text, exactly as designed |
| Already produced by this run and **cache-valid** | **skip** |

**If the production resume path does not behave this way — STOP and report a `MASTER-IMPACTING FINDING`.** Do not work around it, do not force, do not hand-clean the directory. A resume that re-encodes work already done wastes GPU time; a resume that **skips** something invalid silently corrupts the cache. The second is the one that matters.

**Record for the HANDOFF, per interruption:**

- interruption time
- restart time
- completed count before the restart
- **final total runtime** across all segments

A run that was interrupted three times is still one experiment. Report it as one, with its segments.

**Leftover temporaries.** `n06` writes `<uid>.part.npz` and `<uid>.json.part`, then renames. An interruption can leave one behind. Do not delete them blindly — **count them, report them**, and confirm the resume either overwrites or ignores them. A stray `.part` file is evidence of where the run stopped.

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

### 12.2 The run — the production command

**This is the only entry point. Use it verbatim.**

```bash
/home/kyzen/miniconda3/envs/MetaFind/bin/python -m metafind.data.encode_text_image
```

**No flags.** Specifically **no** `--force`, **no** `--limit`, no debug flag, no alternative entry point, no direct call into `main()` or a helper.

**If you run it under `tmux` or in the background, record both layers:**

1. the outer command — the exact `tmux new-session …` / `nohup …` invocation, including any redirect;
2. the command actually executed inside it — which must be the line above, unaltered.

Record the full output including the quarantine ledger.

> **The executor may not substitute another entry point.** If this command cannot be used, that is a `MASTER-IMPACTING FINDING`, not a reason to improvise.

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

### 12.4 Post-run NPZ integrity audit — **once, after the run completes**

`is_complete()` proves the **metadata contract** is valid. It never opens the `.npz`. This audit proves the **payload is not corrupt**. **Neither substitutes for the other.**

Run it **once**, after the run, not on every cache check — D10 declined per-pass validation on cost (~1.3 GB) and that reasoning still holds for the hot path.

**The contract below is derived from the current writer**, `encode_text_image.py:346-350` and the sidecar record at `:361-372`. It is not invented:

| | Derived from |
|---|---|
| keys `text`, `views`, `image` | `np.savez_compressed(tmp, text=…, views=…, image=…)` |
| dtype `float16` for all three | `.astype(np.float16)` on each |
| `text.shape == (embedding_dim,)` | sidecar `embedding_dim = int(text_vec.shape[0])` |
| `image.shape == (embedding_dim,)` | `pooled = aggregate(view_vecs, …)`, same width |
| `views.shape == (n_views, embedding_dim)` | sidecar `n_views = int(view_vecs.shape[0])` |
| `embedding_dim == 1280` | `EMBED_DIM = 1280`, `ulip_backbone.py:87` |

**Cross-check the `.npz` against each sidecar's own recorded `embedding_dim` and `n_views`** — the writer derives both per asset. Do not hardcode `n_views`.

```bash
$PY - <<'EOF'
import json, numpy as np
from metafind import paths
from metafind.models.ulip_backbone import EMBED_DIM
RATIFIED = "metafind_v2_cm@8e4b1fcc66c7f48c"
bad = []
n = 0
for sc in sorted(paths.EMBEDDINGS.glob("*.json")):
    uid = sc.stem; n += 1
    try:
        rec = json.loads(sc.read_text())
        npz = paths.EMBEDDINGS / f"{uid}.npz"
        if not npz.is_file():
            bad.append((uid, "npz missing")); continue
        z = np.load(npz)
        dim, nv = rec["embedding_dim"], rec["n_views"]
        checks = [
            (set(z.files) == {"text", "views", "image"},      "keys"),
            (all(z[k].dtype == np.float16 for k in z.files),  "dtype"),
            (z["text"].shape  == (dim,),                      "text shape"),
            (z["image"].shape == (dim,),                      "image shape"),
            (z["views"].shape == (nv, dim),                   "views shape"),
            (dim == EMBED_DIM,                                "embedding_dim != EMBED_DIM"),
            (all(np.isfinite(z[k]).all() for k in z.files),   "non-finite values"),
            (all(z[k].size > 0 for k in z.files),             "empty array"),
            (rec["text_serialization"] == RATIFIED,           "serializer"),
            (rec["text_truncated"] is False,                  "text_truncated"),
        ]
        for ok, why in checks:
            if not ok: bad.append((uid, why))
    except Exception as e:
        bad.append((uid, f"{type(e).__name__}: {e}"))
print(f"sidecars audited      {n:>7,}")
print(f"failures              {len(bad):>7,}   <-- must be 0")
for uid, why in bad[:20]: print("   ", uid, why)
EOF
```

Also report any leftover temporaries:

```bash
ls data/outputs/embeddings/*.part.npz data/outputs/embeddings/*.json.part 2>/dev/null | wc -l   # expect 0
```

**Any failure is escalated, not absorbed.** A corrupt payload that passes `is_complete()` is precisely the class of defect this project has been bitten by.

---

### 12.5 Research fidelity

- Every sidecar records `text_serialization` = the ratified identity. **Zero** carry `metafind_v1_natural`.
- Sample sidecars and confirm the text matches the ratified template — centimetres, no `"A "` article, capitalised category.
- **0** records report `text_truncated`.

---

## 13. Definition of Done

- [ ] **1.** `load_protocol()` and the pre-flight both PASSED **before** GPU time was spent.
- [ ] **2.** n06 ran to completion over the full work list using the **§12.2 production command verbatim** — no `--force`, no `--limit`, no substitute entry point. Command recorded, including the outer `tmux`/background layer if one was used.
- [ ] **2a.** If the run was interrupted: every interruption recorded with interruption time, restart time, completed count before restart, and **final total runtime**. Resume used the identical command. Nothing was deleted or force-overwritten. Leftover `.part` files counted and reported.
- [ ] **3.** `data/outputs/embeddings/` holds **45,952** `.npz`.
- [ ] **4.** Exactly **3** assets quarantined — the known legacy-v1 residuals, by uid. Any other quarantine is escalated, not absorbed.
- [ ] **5.** **0** sidecars are NOT cache-valid under the ratified serializer.
- [ ] **5a.** **Post-run NPZ integrity audit (§12.4) reports 0 failures** over all 45,952 — keys, `float16` dtype, shapes cross-checked against each sidecar's own `embedding_dim` / `n_views`, `embedding_dim == EMBED_DIM`, all values finite, no empty array, correct serializer, `text_truncated` false. **0** leftover `.part` files. This is a separate claim from item 5 and does not substitute for it.
- [ ] **6.** **0** sidecars carry `metafind_v1_natural`. **0** report `text_truncated`.
- [ ] **7.** Pre-flight still PASSES after the run.
- [ ] **8.** All 45,955 annotation records byte-identical to the baseline.
- [ ] **9.** `checkpoints/` still empty; n09 and training never invoked.
- [ ] **10.** `pytest tests/ -q` 547 passed; `check_graph.py` all pass.
- [ ] **11.** `git diff` touches only §9.1 paths. **No code change.** The two shared ledgers were **appended to only** — not truncated, rewritten, or cleaned.
- [ ] **11a.** Working-tree state recorded at run start **and** run end. If they differ, the difference is reported and attributed.
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
