# Codex Review — D10_stage1-encoding-contract

## Review metadata

| | |
|---|---|
| Reviewer | Codex (`codex-cli` 0.148.0), adversarial-review mode |
| Round 1 session | `01a02074-fc0a-7263-b481-534054d64572` |
| Round 2 session | `01a02087-4a5b-7ef3-a283-d70167b6b36b` |
| Scope | working-tree diff at commit `35a3dfb` |
| Brief given | `TASK.md`, `HANDOFF.md` context, D0-008 §11.2/§11.3/§12.3, the full verification output, and nine explicit attack targets from `TASK.md` §14 |
| Instruction | **attack, do not confirm**; run commands rather than reading only |
| Round 1 verdict | `needs-attention` — "No-ship" |
| Round 2 | targeted re-review of the six fixes; see §Round 2 |

Codex was given the forbidden operations explicitly (no `n06`, no
`resolve_stage1.main()`) and respected both.

**Round 1 completed successfully. This is a real review, not an unavailability
report.** It ran its own corpus enumerations, its own in-memory mutation
experiments, and its own re-derivation of the arithmetic.

---

## Findings, classified

Severity is Codex's; the classification and the verification are mine.

### C-1 · `CONFIRMED` · MAJOR · **fixed**
> Single-probe identity ignores serializer branches
> (`metafind/models/resolve_stage1.py`)

Codex mutated `PLACEMENT_PHRASES[("onFloor",)]` in memory: the emitted string
changed and `text_serialization_id()` did not.

**Verified independently.** Reproduced exactly:

```
string changed:   True
identity changed: False
MAX_CATEGORY_CHARS exercised by probe: False
```

My own comment claimed the probe covered "any character cap". It did not cover
`MAX_CATEGORY_CHARS`, and it covered only one of six placement branches. This is
a genuine hole in B-2/B-3: the protocol would have gone on certifying a
serializer that no longer existed.

**Fix.** `SERIALIZATION_PROBE` (one dict) became `serialization_probes()` (ten),
covering `NO_PLACEMENT_PHRASE`, every key in `PLACEMENT_PHRASES`, two unmapped
fallback combinations, a category past `MAX_CATEGORY_CHARS`, a description past
`MAX_DESCRIPTION_CHARS`, four materials against `MAX_MATERIALS`, and integer /
>1-fractional / <1-fractional / zero dimensions. `serialization_id_for()` hashes
every emitted string in order. Identity moved
`metafind_v2_cm@a74abe1414f10997` → `metafind_v2_cm@271b5893f042d43c`.

Post-fix sensitivity, measured over seven knobs:

```
PLACEMENT_PHRASES[onFloor]   identity moves: True
re-add the R-3 dead entry    identity moves: True
NO_PLACEMENT_PHRASE          identity moves: True
MAX_CATEGORY_CHARS           identity moves: True
MAX_DESCRIPTION_CHARS        identity moves: True
MAX_MATERIALS                identity moves: True
TEXT_TEMPLATE                identity moves: True
```

Seven new tests pin this, including one asserting the suite emits every phrase in
`PLACEMENT_PHRASES` plus the no-placement phrase plus the fallback join.

### C-2 · `CONFIRMED` as observation · MAJOR · **deliberately NOT fixed — escalated**
> Downstream cache consumers still accept the retired identity
> (`metafind/train/stage1.py`)

**Verified.** `load_protocols()` (`stage1.py:64-89`) checks `status` and the
hyperparameter hash and never looks at `text_serialization`.
`Stage1Dataset.__getitem__` (`stage1.py:108-119`) does
`np.load(paths.EMBEDDINGS / f"{uid}.npz")` with no sidecar consultation at all.
`gallery_index.py` reuses the same loader. So n10/n11 can consume stale NPZ files
without ever passing through n06's new guard.

**Not fixed here, on contract grounds.** `TASK.md` §7 confines this task to n06,
n05b and their tests, and §Global Constraints in `CONTEXT.md` forbids expanding
into another task's scope. The fix belongs to `D3_stage1-train` / `D4_gallery-index`.

**Not urgent, but not harmless.** n10 cannot run today: `splits.json` and
`stage1_protocol.json` are both absent and `checkpoints/` is empty. The moment
n09 produces them, this gap becomes live.

Escalated as **MIF-D10-3** in `HANDOFF.md`.

### C-3 · `CONFIRMED` · MAJOR · **fixed**
> The pre-flight's ratified oracle shares production semantics
> (`tools/preflight_stage1_text.py`)

I had disclosed this as a limitation. Codex was right that disclosing it was not
enough: it demonstrated that changing the production on-floor phrase moved both
sides of the comparison, so the gate reported zero mismatches on a serializer
that had been edited without authorisation.

**Fix.** The tool now imports nothing from `resolve_stage1` except
`text_serialization_id`. The caps, `_cap()`'s trimming rule, the placement
vocabulary and `NO_PLACEMENT_PHRASE` are transcribed locally as `RATIFIED_*`
constants with a comment explaining that the duplication is the point.

Post-fix mutation evidence, on a record that exercises each knob:

```
clean agreement:                              True
placement phrase     oracle detects mismatch: True
MAX_MATERIALS 3->1   oracle detects mismatch: True
MAX_CATEGORY 40->4   oracle detects mismatch: True
NO_PLACEMENT         oracle detects mismatch: True
TEXT_TEMPLATE unit   oracle detects mismatch: True
```

The full corpus still reports **0 template mismatches** — now against an oracle
that shares no code with the thing it checks.

### C-4 · `CONFIRMED` · MINOR (defence in depth) · **fixed**
> `load_protocol` validates a different callable than n06 can execute
> (`metafind/data/encode_text_image.py`)

`serialize_annotation` and `text_serialization_id` were imported as separate
aliases, so `load_protocol()` certified the resolver's serializer while the encode
loop called the local alias. Rebinding one leaves the other certified.

The realistic likelihood is low — it takes deliberate in-memory rebinding — but
the fix costs nothing and the whole point of B-2 is that the protocol certifies
the callable that will run.

**Fix.** `serialization_id_for(serializer)` takes the callable. `load_protocol()`
passes its own alias: `serialization_id_for(serialize_annotation)`.

### C-5 · `PARTIALLY CONFIRMED` · MAJOR on part (b) · **(b) fixed**
> Pre-flight neither independently proves nor gates cache validity

**(a) "the proof repeats the implementation" — reduced.** `is_complete()` is the
gate n06 will actually apply; a proof that avoided calling it would be proving
something else. `TASK.md` DoD item 2 asks for exactly this ("proven through
completion / cache-validity / pre-flight logic **only**"). Independence was
already supplied by the counterfactual script, and Codex's own independent
enumeration reached the same 0/5,276.

Still, the point had force, so the tool now carries a **second, independent
recount** that reconstructs cache-validity from the sidecar without calling
`is_complete()`, and **fails if the two disagree**. Both report 0.

**(b) "nonzero `cache_valid` is never a failure" — CONFIRMED, and this was a real
hole.** The script could have printed `PRE-FLIGHT PASSED` while stale records were
being classified complete. Fixed: any sidecar judged complete while recording a
different serialization identity is now a hard failure, with the message naming
the two-distribution gallery this contract exists to prevent.

### C-6 · `CONFIRMED` · MINOR · **fixed at the gate, not in the serializer**
> Non-finite dimensions serialize silently and evade pre-flight

**Verified.** `_dim(float("nan"))` → `"nan"`, `_dim(float("inf"))` → `"inf"`.
`validate_annotation()` rejects both, but n06 reads annotation JSON straight off
disk and never revalidates, so a corrupted or hand-edited record could reach the
encoder as `"roughly nan by inf"`.

**Fixed in the pre-flight only.** It now rejects non-finite dimensions and any
value outside n05's own `0.1 … 10000 cm` admission bounds.

**Deliberately NOT fixed in the serializer.** `TASK.md` §8 and D0-008 §12.3 confine
serialization changes to E-1/E-2/S-1/S-2; a `math.isfinite()` guard emits no string
and so is arguably outside that prohibition, but it changes the serializer's
refusal behaviour, and the module already has a precedent guard (empty
`materials`) that Master may prefer to extend deliberately rather than have this
task extend on its own initiative. Recommended to Master in `HANDOFF.md`. Corpus
impact today: **0 records**.

### C-7 · `CONFIRMED` · MAJOR · **fixed**
> A missing `embedding_uri` is judged complete
> (`metafind/data/encode_text_image.py`)

The single best finding of the review, and it is the one thing that was still
capable of defeating B-1.

`Path(rec.get("embedding_uri", ""))` is `Path("")`, which is `PosixPath('.')` —
the working directory — and `.exists()` on it is `True`. Reproduced end to end:

```
no embedding_uri   -> is_complete: True      (before)
empty  uri         -> is_complete: True      (before)
```

So a sidecar with correct current text and no vectors at all was "complete", n06
skipped the asset, and the failure resurfaced later as a `FileNotFoundError`
inside n10's dataloader with nothing pointing back at n06. This bug is
**pre-existing** — it is in the original `is_complete()` too — and D10 is the task
that made it load-bearing.

**Fix.** Require a non-empty `str` and `Path(uri).is_file()`. After:

```
no embedding_uri   -> is_complete: False
empty  uri         -> is_complete: False
dir as uri         -> is_complete: False
uri is None        -> is_complete: False
real npz           -> is_complete: True
```

Corpus impact today: **0 records** — all 5,276 sidecars carry a valid URI.

### C-8 · `CONFIRMED` as fact · LOW · **not a defect; already disclosed**
> The required changed-file allowlist check fails

`docs/graph/README.md` and `TASK.md` are not named in `TASK.md` §9. Both were
already disclosed in `HANDOFF.md` before the review ran:

- `TASK.md` — status `READY` → `ACTIVE`, which the task's own start procedure
  requires;
- `docs/graph/README.md` — one number, forced by `tools/check_graph.py:415`, which
  asserts the README's stated test-function count equals the count in `tests/`.
  Adding tests without updating it makes DoD item 7 fail.

Neither can be avoided while satisfying the rest of the contract. They are for
Master to ratify, not for me to revert.

---

## Findings Codex looked for and did not find

These are attack targets from `TASK.md` §14 that Codex ran and cleared. They are
evidence, not silence.

| Target | Codex's check | Result |
|---|---|---|
| **6 · S-2 on odd leading characters** | ASCII, non-alphabetic, empty, combining-mark, CJK, `é`, `ß` | No defect. Matches the mandated `c[:1].upper() + c[1:]`; the producer rejects empty categories. I reproduced: `"LED lamp"`→`"LED lamp"`, `"3d printer"`→`"3d printer"`, `"élan vital"`→`"Élan vital"`, `"日本刀"` unchanged |
| **7 · did the golden update change more than the four edits?** | Ran HEAD's serializer in memory and compared all 45,952 v3 outputs against an independent HEAD-plus-E-1/E-2/S-1/S-2 construction | **Zero extra changes, zero placement drift.** This is the strongest single piece of evidence that the scope guard held |
| **9 · corpus arithmetic** | Fresh enumeration of the annotation ∩ render intersection | 45,955 / 45,952 / 3, each of the three raising `KeyError: 'width'` — matches my figures exactly |
| **1 · stale sidecars (real data)** | Independent enumeration of all 5,276 | All have mismatched text and valid NPZ URIs; production `is_complete()` returns true for **0** |
| **5 · formatter edge cases** | `0.25→0.2`, `-0.25→-0.2`, `-0.04→-0`, integers and `1000.0` strip `.0` | Confirms round-half-to-even, as I had documented. `-0.04→"-0"` is a new detail; no negative dimension exists in the corpus and n05 bounds dimensions to `≥ 0.1` |

---

## Round 2

Targeted re-review of the six fixes, plus confirmation of the two deliberate
non-fixes. Result recorded below.

Round 2 was briefed with the six fixes and the two deliberate non-fixes, and told
to verify the fixes rather than re-list the originals. Verdict again
`needs-attention`, with **four new findings**, all of which I reproduced and
**all four of which are now fixed**.

Round 2 also independently re-ran `check_graph.py` (2,275 pass), the full-corpus
pre-flight (0 mismatches, 0 zero renders, 0/5,276 cache-valid, exactly one
88-token record), and settled the `WORKFLOW.md` provenance question.

### R2-1 · `CONFIRMED` · MAJOR · **fixed**
> A real serializer change retains the certified identity

The sharpest finding of either round. Codex moved `MAX_DESCRIPTION_CHARS` from
160 to **161** — every probe still truncated at the same word boundary, so the
identity did not move, and `load_protocol()` went on accepting the old protocol.
It named the corpus record that *does* change:
`020a2199c72a4f8eaea8f1212271a1b0` ends `"including plastic."` at 160 and
`"including plastic and."` at 161.

**Reproduced:** `cap 160->161, identity moves: False`.

The underlying point is structural and correct: **a probe suite is a sample, and
a sample cannot cover a continuous parameter.** My round-1 fix made the identity
sensitive to every knob I could think to sample; it could not make it sensitive
to every value those knobs can take.

**Fix.** `serialization_contract()` returns a canonical mapping of every constant
the emitted string depends on — template, all three caps, the placement dict, the
no-placement phrase, the family — and `serialization_id_for()` hashes that
manifest *together with* the emitted probe strings. Constants now move the
identity by value; probes still cover the *logic* (`_dim`, `_capitalise`,
`_cap`, `placement_phrase`), which constants cannot. Identity moved
`metafind_v2_cm@271b5893f042d43c` → **`metafind_v2_cm@8e4b1fcc66c7f48c`**.
Verified: `160 -> 161 moves identity: True`. Two regression tests added, and the
manifest is written into the protocol artifact as `text_serialization_contract`.

### R2-2 · `CONFIRMED` · MAJOR · **fixed**
> Any regular file makes a vector-less sidecar complete

My round-1 fix for C-7 replaced `.exists()` with `.is_file()`. Codex pointed
`embedding_uri` at `AGENTS.md` and `is_complete()` returned true.

**Reproduced:** `unrelated file as embedding_uri -> is_complete: True`.

I had fixed the wrong half. "A file exists there" was never the invariant;
"**this asset's** vectors are there" is.

**Fix.** The URI must `resolve()` to the canonical `EMBEDDINGS/<uid>.npz`.
Verified:

```
foreign file   -> is_complete: False
other uid      -> is_complete: False
canonical      -> is_complete: True
```

**Partially accepted only.** Codex also recommended opening each NPZ and checking
it contains `text`/`views`/`image` at the expected shapes. **Not adopted**: that
reads 1.3 GB on every work-list pass, on every resume, to guard a corruption mode
with zero observed instances. The canonical-path requirement closes the finding
Codex actually demonstrated. Recorded as a residual risk in `HANDOFF.md` rather
than silently dropped.

### R2-3 · `CONFIRMED` · MAJOR · **fixed**
> Pre-flight passes an unserializable `prompt_version:3` record

`expected_text_for()` returns `""` on any exception, and the old loop tested
serializability *before* version — so a v3 record that could not be serialized
fell into the "not v3" bucket and silently left the validated population. Codex
built a v3 record with `width="not-a-number"` and got `PRE-FLIGHT PASSED`,
return code 0, **`total records: 0`**.

A gate that validates nothing and reports success is worse than no gate.

**Fix.** Version is classified first. A v3 record that cannot be serialized is a
hard failure; a non-v3 record that *can* be is also a hard failure (the population
changed shape); and `v3 == 0` is a hard failure in its own right.

**Reproduced and re-verified** in a temp-directory harness that never touches the
corpus: two records, one good v3 and one corrupt v3 →

```
PRE-FLIGHT FAILED -- do not start n06:
  - 1 prompt_version:3 records cannot be serialized at all, e.g. ['bad']
return code: 1
```

### R2-4 · `CONFIRMED` · MINOR · **fixed**
> The token-budget oracle shares the production threshold

The pre-flight imported `TEXT_CONTEXT_LENGTH` from n06. Moving it to 88 would
move the gate with it, and the 88-token record would stop being flagged while
CLIP went on truncating at 77. Exactly the C-3 mistake in the one place I had not
transcribed.

**Fix.** `RATIFIED_TEXT_CONTEXT_LENGTH = 77` locally, used for every comparison,
plus a hard failure if the production constant no longer equals it.

### Round 2's remaining "do not ship"

Codex's round-2 verdict line still reads no-ship, but it was written against the
pre-fix tree — it names R2-1…R2-4 as the reasons, and all four are now closed and
re-verified. The two open items it endorsed are not code defects in this task's
scope:

- the downstream identity gap is "confirmed but **correctly escalated** as D3/D4
  non-scope";
- the README/TASK edits have "**defensible workflow reasons**";
- gate-only dimension validation "remains process-dependent because n06 does not
  invoke it" — accurate, and the reason it is reported to Master rather than
  patched into the serializer under a scope guard.

### Two things round 2 settled that were not findings

**The test count.** Codex reported `468 passed / 5 CUDA-skipped`. Its sandbox has
no visible GPU. In this environment the same command is **476 passed, 0 skipped**
— `CONTEXT.md` §6 records that `test_cuda_smoke.py`'s 5 tests genuinely run here,
and `nvidia-smi` confirms an RTX 5090. Both numbers are correct in their own
environment; the repository figure is the one measured on the project's hardware.

**`workflow/WORKFLOW.md`.** 321 added lines appeared in the working tree during
this session and are **not mine**. Round 1's log never mentions the file. Codex
round 2 attributed it to *Claude session `604d2eb9-e288-4338-a67d-04a4424a16f3`*
— a different session from this one (`52fc23cf-e966-47ac-82b3-fe9af648cb5f`). It
is therefore an external concurrent edit, left completely untouched, and reported
to Master in `HANDOFF.md`.

---

## Net effect on the task

Both rounds changed the work materially. Across the twelve findings, **seven were
holes in the exit criteria this task exists to establish** — C-1, C-3, C-5(b),
C-7, R2-1, R2-2, R2-3 — and three of them (C-7, R2-2, R2-3) were cases where a
guard reported success while doing nothing.

Two findings were fixes to my own fixes: C-7 was closed with `.is_file()` and
R2-2 showed that was still wrong; C-1 was closed with a probe suite and R2-1
showed a sample can never cover a continuous parameter. Both second passes
produced a better invariant than the first — a canonical path, and a contract
manifest — which is the argument for running the second round at all.

Nothing in the review touched the serialized output. The corpus still produces the
same 45,952 strings it did before the review, verified against an oracle that now
shares no code with the serializer.

The review did **not** overturn the ratified template, the corpus arithmetic, or
the cache-validity proof. It corroborated all three independently.

---

## Change made after the review closed

Both Codex rounds ran against a tree in which the B-4 pre-flight **failed** on
`3e91980a…` (88 true tokens), and both correctly reported that as an open item.

After round 2 closed, Kyzen directed that the three CJK records be hand-translated
(MIF-D10-1 in `HANDOFF.md`). That change touches **data only** — three
`description` strings plus three provenance fields per record. It touched no
reviewed source file, and no reviewed finding depends on it.

Re-verified after the change, unreviewed by Codex:

```
pytest tests/ -q          476 passed, 0 skipped
tools/check_graph.py      2275 checks, all pass
preflight                 PRE-FLIGHT PASSED — 0 over-77 (max 72), 0 mismatches,
                          0 zero-dimension renders, 0/5,276 cache-valid
corpus                    45,952 v3 records, 0 CJK, 7 accented English
```

If Master wants the data mutation itself adversarially reviewed, that is a fresh
round against a different kind of change and is not covered here.
