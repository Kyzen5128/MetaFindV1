# D0 Decision — Stage 1 Text Serialization Template (U-15)

> D0 is the Research / Architecture Lead.
> This decision must be evidence-backed, independently reviewed, and accepted by Master before becoming project state.
>
> **Sections 1–5 were prepared by Master** during initialization and are a starting pointer set, not a completed evidence survey. D0 must independently verify them and extend where needed.
> **Sections 6–11 are D0's work and are intentionally empty.** Master has not performed the investigation.
> **Section 12 is filled by Master after review.**

---

## Decision ID

`D0-008_stage1-text-template`

---

## Status

Use only: `OPEN` · `INVESTIGATING` · `REVIEW` · `RECOMMENDED` · `ACCEPTED` · `REWORK` · `REJECTED`

Current:

`USER_APPROVED`

Opened by Master on user approval 2026-08-20. Returned `RECOMMENDED` by D0 2026-08-21.
Master recommendation 2026-08-21: `ACCEPT WITH FOLLOW-UP` — Section 12.
**USER decision 2026-08-21: `MODIFY` → `APPROVE`. FINAL ACCEPTED.** — Sections 13 and 14.

This ratifies the **serialization design**. It does **not** authorise n06 to run: the cache completion/validity gate (§11.2, B-1…B-4) is an execution question owned by `D10_stage1-encoding-contract` and resolved in D10's own integration review. It is explicitly outside what the user approved here.

---

## 1. Question

**Ratify — or reject — the current `TEXT_TEMPLATE` in `metafind/models/resolve_stage1.py:96-100` as the recorded Stage 1 text serialization IMPLEMENTATION CHOICE (U-15).**

The template is the exact string the frozen CLIP text tower sees for every gallery and query object. It determines every text embedding in Stage 1 and therefore every text-conditioned column of Table 1.

Ratification must cover all four of the following, because each is a separable choice the paper does not make:

1. **The template string itself**, currently
   `"{description} A {category} made of {materials}, roughly {width:.0f} by {length:.0f} by {height:.0f} centimetres, {placement}."`
2. **Field order** — currently description → category → materials → dimensions → placement.
3. **Unit and precision** — currently centimetres at `:.0f`.
4. **The three deliberate omissions** — `synset`, `volume`, `mass` are present in the annotation record but excluded from the encoder input (`resolve_stage1.py:102-115`).

The code itself requests this sign-off. `resolve_stage1.py:102` carries:

```
# [U-15, IMPLEMENTATION CHOICE -- CONFIRM BEFORE THE FULL RUN]
```

That marker has never been discharged.

---

## 2. Why This Decision Exists

**Trigger.** Master initialization (2026-08-20) re-verified that `data/outputs/stage1_encoding_protocol.json` records a *different* template from the one `serialize_annotation()` actually emits, and that the in-code confirmation marker is still open.

**What it affects.** `D1_n06-reencode` — a full re-encode over ~45,952 assets, estimated ~4 GPU-hours. Running it against an unratified template risks spending those hours twice.

**What stays blocked until resolved.**

- `D1_n06-reencode` — BLOCKED on this decision.
- `D2_stage1-prereq` — BLOCKED on this decision (among others).
- `D3_stage1-train` — transitively blocked.

**Ordering constraint.** Re-running n05b to refresh the encoding protocol also rewrites `stage1_hyperparameters.json` in the same call (`resolve_stage1.py:443-444`). So this ratification and correction **C-001** (τ = 0.5) must both be settled before n05b runs, or n05b runs twice. See `workflow/MASTER.md` §8.

**Not a new problem.** The artifact-versus-code mismatch was already recorded in `_workflow_old_20260820/任務/A_n06-reencode/TASK.md` (lines 18, 32-37, and acceptance item 133). What is new is the observation that the *ratification* marker, distinct from the artifact refresh, was never addressed.

---

## 3. Decision Scope

### In Scope

- Ratify or reject the current `TEXT_TEMPLATE` string.
- Ratify or reject the current field order.
- Ratify or reject centimetres at `:.0f` precision.
- Ratify or reject the exclusion of `synset`, `volume`, and `mass` from the encoder input.
- Ratify or reject the `PLACEMENT_PHRASES` mapping and `NO_PLACEMENT_PHRASE` (`resolve_stage1.py:116-128`), which produce the `{placement}` slot.
- State the evidence classification of whatever is ratified.
- State what must be true of `stage1_encoding_protocol.json` afterwards so that it describes runtime truth.
- Identify any token-budget risk the template creates against CLIP's 77-token limit.

### Explicit Non-Scope

- **Do not edit `resolve_stage1.py`.** This decision recommends; Master accepts; D1/D2 execute.
- **Do not re-run n05b.** Refreshing `stage1_encoding_protocol.json` is correction **C-002**, executed in D1/D2.
- **Do not run n06** or any encoding job.
- **Do not decide τ.** That is correction **C-001**, already classified as an implementation correction, not a decision. `3experiments.tex:15` settles it: τ = 0.5.
- **Do not decide the disposition of the 3 `prompt_version:1` annotations.** That is `D0-003`.
- **Do not decide `tower_sharing`.** That is `D0-002`.
- Do not re-annotate, or propose changes to, the n05 annotation schema itself. The schema is fixed; only its serialization to text is in question.

---

## 4. Authority / Evidence

> Master-prepared pointers. **D0 must verify each independently and extend the survey.** Classification shown is Master's, and D0 may revise it with evidence.

### Primary Evidence

| Source | Location | What it supports | Class |
|---|---|---|---|
| MetaFind paper | `docs/paper/metafind_source/2methdology.tex` §2.3 | Names the annotation fields; **gives no serialization format**. This is why the template is an IMPLEMENTATION CHOICE rather than a paper requirement | PAPER FACT (as to silence) |
| MetaFind paper | `docs/paper/metafind_source/data-preprocess.png` | The annotation **schema** figure. Previously established to have driven the n05 v3 re-annotation. Whether it constrains serialization is **UNKNOWN** and is part of this investigation | UNKNOWN |
| MetaFind paper | `docs/paper/metafind_source/appendix.tex` | Not yet checked for any serialization statement | UNVERIFIED |

D0 must confirm whether §2.3 is genuinely silent on format, and must check the appendix and figure captions before concluding silence. Absence of evidence is not evidence of absence — see `.claude/rules/research-rigor.md` §3.

### Supporting Evidence

| Source | Location | What it supports | Class |
|---|---|---|---|
| ULIP-2 | `docs/paper/ulip2_source/` | MetaFind builds on ULIP-2; ULIP-2's own text construction is the nearest upstream precedent. Whether MetaFind adopts it is **not established** | UPSTREAM, relevance unverified |
| In-code rationale | `resolve_stage1.py:83-95` | Records why natural prose over labelled records: the frozen CLIP text tower was caption-trained and cannot adapt; the 77-token budget penalises field names | OBSERVED IMPLEMENTATION |
| In-code rationale | `resolve_stage1.py:92-95` | Claims field order follows §2.3's own sentence — "object category, size dimensions, materials, and placement constraints" — with description first. **D0 should check this claim against the actual sentence**, since the implemented order puts description first and materials before dimensions | OBSERVED IMPLEMENTATION, claim unverified |
| In-code rationale | `resolve_stage1.py:102-115` | Records why `synset`, `volume`, `mass` are omitted: identifier-not-language; redundant with the dimensions; no visual grounding and no Objaverse ground truth. Cites a MEASURED r = 0.52-0.62 for size proportions against the mesh bounding box | OBSERVED IMPLEMENTATION; the r figure is **unverified** |
| Golden-string test | `L1-TEXT-SERIALIZATION`, in `tests/test_resolve_stage1.py` | The template is pinned by test. Pinning proves stability, **not** correctness | OBSERVED IMPLEMENTATION |

### Conflicting Evidence

**Artifact contradicts code.** Do not silently reconcile these — record which one the reproduction adopts.

| | Template |
|---|---|
| `data/outputs/stage1_encoding_protocol.json` | `{description} A {category} made of {materials}, roughly {length:.2f} by {width:.2f} by {height:.2f} metres, typically placed {placement}.` |
| `resolve_stage1.py:96-100` (what actually runs) | `{description} A {category} made of {materials}, roughly {width:.0f} by {length:.0f} by {height:.0f} centimetres, {placement}.` |

They differ in **four** ways: unit (metres vs centimetres), precision (`.2f` vs `.0f`), the first two dimension fields are **transposed** (`length, width` vs `width, length`), and the "typically placed" prefix moved out of the template into the placement phrase.

The transposition is the one most easily missed and is research-significant: it changes which number the reader and the encoder see first.

---

## 5. Current Repository State

> Master-verified 2026-08-20 unless marked otherwise.

**What n06 actually uses.** `metafind/data/encode_text_image.py:194` calls `serialize_annotation(annotation)`; `resolve_stage1.py:278` defaults `template=TEXT_TEMPLATE`. **The artifact's `text_template` field is written but never read back.** n06 uses the code constant.

**Verified output, current code on a v3 record:**

```
A black flip-flop with a textured sole and a strap across the footbed.
A flip-flop made of rubber, fabric, roughly 25 by 10 by 4 centimetres,
typically placed on the floor or on other objects.
```

**Verified output cached by the previous partial n06 run**, same uid `000074a334c541878360457c672b6c2e`:

```
A black flip-flop with a textured sole and a strap across the top, featuring
a blue accent stripe. A flip-flop made of rubber, textile, roughly 0.25 by
0.15 by 0.05 metres, typically placed floor.
```

Note the old string also shows a degenerate placement rendering — `"typically placed floor."` — which `PLACEMENT_PHRASES` was introduced to fix.

**Embedding cache.** `data/outputs/embeddings/` holds 5,276 `.npz` — partial and stale. Expected successful output of a full n06 run is **45,952** `.npz` plus **3** quarantine records; see `workflow/MASTER.md` §4 for the count breakdown.

**No gate catches the mismatch.** `tools/check_graph.py` reports 2275 checks, all passing, with the artifact/code divergence in place. `load_protocol()` (`encode_text_image.py:86-106`) validates exactly three fields — `status`, `actual_clip_train_scope`, `image_aggregation` — and **never compares `text_template`**. Verified by reading the function, not inherited from the old task card.

**Test suite.** `python -m pytest tests/ -q` → 442 passed, 0 skipped, 0 deselected. `tests/test_resolve_stage1.py` collects 27. Passing tests establish that the template is stable, **not** that it is correct.

**Token budget.** `encode_text_image.py` counts tokens and flags `text_truncated` at the 77-token CLIP limit. The previous partial run logged **over-length text 0** across its first 5,000 assets. Whether that holds across the full corpus is **UNKNOWN**.

---

## 6. Options

### 6.0 D0 Independent Verification of Sections 1–5

> All work below is read-only. No repository code, artifact, or dataset was modified. No encoding job was run.
> Code state: `git rev-parse HEAD` = `1837477`, working tree modified only in `workflow/` (Master's files) plus this decision file.
> Python: `/home/kyzen/miniconda3/envs/MetaFind/bin/python` (conda env `MetaFind`).

#### V1 — Is §2.3 genuinely silent on serialization format? **PARTLY. §4's framing is too broad.**

**Silent on format: CONFIRMED.** Every `.tex` file in `docs/paper/metafind_source/` was searched for
`templat|serial|prompt|caption|format|string|token|centimet|metre|meter|cm|dimension|GPT-4o|VLM|annotat`:
`2methdology.tex` (134 lines), `3experiments.tex` (143), `4backgound.tex` (12), `appendix.tex` (125), `neurips_2025.tex` (127).
No serialization format, no template, no annotation prompt, and no unit label appears anywhere. `appendix.tex` is entirely the SE(3) equivariance proof (lines 23–68) plus qualitative room figures (95–124) and contains nothing about text construction.
→ **PAPER FACT (as to silence on the serialization format).**

**Not silent on two of the four in-scope dimensions.** `data-preprocess.png` was read directly (not inherited). Its "Structured Detailed Description" box prints:

```
{"annotations": {
"category": "robot", "synset": "robot.n.01", "width": 30, "length": 30,
"height": 40, "volume": 36000, "mass": 2.5, "description": "A small cubic-shaped
robot with a smiling screen face, two antennae on top, and rounded side arms and
feet with spring-like connectors.", "materials": ["metal", "glass", "plastic"],
"onCeiling": false, "onWall": false, "onFloor": true, "onObject": true}}
```

Two constraints follow that §4 did not credit:

- **Scale.** `30 × 30 × 40 = 36000` closes **only** in cubic centimetres. This is MetaFind-internal arithmetic and does not depend on the Holodeck schema match recorded at `metafind/data/annotate.py:115-143`. A 30-unit robot of mass 2.5 kg is coherent in cm and absurd in metres.
  → the centimetre reading is an **INFERENCE with two independent supports (figure arithmetic + upstream schema)**, materially stronger than a bare IMPLEMENTATION CHOICE. It is still not a PAPER FACT: the figure labels no unit.
- **Dimension ordering.** The figure prints **width, length, height** — the order the code emits, and the *opposite* of the order the artifact records.

**Also verified:** `volume` in the figure is exactly `width × length × height`, which independently confirms the redundancy argument at `resolve_stage1.py:109` as a figure-supported fact rather than an assertion.

**Corpus check (OBSERVED DATA).** All 45,952 `prompt_version:3` annotation records carry `"dimension_unit": "cm"`; 0 carry any other unit. So the code's word "centimetres" describes the stored values correctly and the artifact's "metres" is **factually false about the data**, not a stylistic variant.

#### V2 — Does the field-order claim at `resolve_stage1.py:92-95` hold? **NO. REJECTED.**

The comment claims: *"Field order follows paper 2.3's own sentence: 'object category, size dimensions, materials, and placement constraints'."*

The sentence exists verbatim — `2methdology.tex:28`:
> "These annotations provide rich textual descriptions detailing attributes such as **object category, size dimensions, materials, and placement constraints**."

Its order is **category → dimensions → materials → placement**.
The figure caption, `2methdology.tex:24`, independently gives the same order:
> "capturing attributes such as **category, dimensions, materials, and spatial placement constraints**"

The implementation emits **category → materials → dimensions → placement**. Materials and dimensions are transposed relative to both paper statements.

→ **The in-code justification is false as written.** Two independent sentences in the paper put dimensions before materials; the code does not follow either.

Two mitigations, stated so the finding is not overstated: both paper statements use *"such as"*, which marks a non-exhaustive attribute list rather than a prescribed serialization order; and neither sentence is describing a string at all. The correct classification is therefore **§2.3 does not constrain serialization order, and the comment's appeal to it is post-hoc rationalisation** — not that the paper mandates a different order.

#### V3 — The artifact/code divergence. **CONFIRMED, and a fifth difference found.**

The four differences in §4 are confirmed by direct reading of `data/outputs/stage1_encoding_protocol.json` and `resolve_stage1.py:96-100`. On the transposition specifically, §4 flags it as "most easily missed" and is right to — but the resolution runs the other way from what §4 implies: **the code's `width, length, height` matches the paper figure; the artifact's `length, width, height` does not.** The transposition is a defect in the artifact.

**Fifth difference — a provenance hazard §4 did not list.** Both the artifact field `text_serialization` and the code constant `TEXT_SERIALIZATION` are the string `"metafind_v1_natural"`. `encode_text_image.py:233` stamps `protocol["text_serialization"]` into **every** n06 sidecar. Verified in the 5,276 existing sidecars: they record `"text_serialization": "metafind_v1_natural"` next to text reading `"roughly 0.25 by 0.15 by 0.05 metres"`. A fresh run would stamp the *same identifier* onto centimetre text.
→ **The serialization identifier does not discriminate between two materially different text distributions.** The only field that distinguishes them is the verbatim `text` string, which is fortunately also stored. See `MASTER-IMPACTING FINDING MIF-1`.

#### V4 — Is the 77-token budget safe across the full corpus? **RESOLVED. No longer UNKNOWN.**

**Method.** All 45,952 `prompt_version:3` annotations were serialized with the production `serialize_annotation()` and tokenized with `open_clip.get_tokenizer('ViT-bigG-14')`, replicating `Encoder.token_count` (`(tokens != 0).sum()`).

**Proxy validated before use.** The same tokenizer was run over the `text` field of all **5,276** existing n06 sidecars and compared against their recorded `text_tokens`: **0 mismatches in 5,276**. The proxy is exact against the real pipeline, so this is a measurement of n06's own counter, not an approximation of it.

**Result over the full v3 corpus:**

| | tokens |
|---|---|
| median | 49 |
| p95 | 58 |
| p99 | 62 |
| max | 77 |
| records ≥ 77 (`text_truncated`) | **1** of 45,952 |

The single record is `3e91980a22da4c0da975cc8ef776972c`, whose true length is **89** BPE tokens — 12 tokens lost from the tail, which is the placement clause, exactly the failure the comment at `resolve_stage1.py:130-133` anticipates. Its cause is **not template overhead**: the description is partly Chinese (`"a medical device used for injecting or抽吸液体，由塑料或金属制成，带有针头和活塞。"`) and CJK tokenizes near one token per character. n06 flags it (`text_truncated=True`), so it is visible rather than silent.

10 v3 records contain non-ASCII text; 3 contain CJK. See `MIF-2`.

→ **The 77-token budget is safe under the current template.** The residual risk is annotation-language quality, not serialization design.

#### V5 — Is ULIP-2 the relevant upstream precedent? **RELEVANCE NOT ESTABLISHED — and the precedent leans against templating.**

- ULIP-2's language modality is a **raw generated caption**: `docs/paper/ulip2_source/main.tex:677` — BLIP-2-opt6.7B generates 10 descriptions per rendered view, ranked by CLIP score, **top-1 used**. No template and no structured-field serialization anywhere in the paper.
- ULIP-2 explicitly argues *against* the pattern MetaFind's template uses. `main.tex:588` describes ULIP's language modality as "derived by prompting dataset metadata such as descriptive terms and category names into cohesive sentences"; `main.tex:334` then criticises exactly that: "the prompt-based pseudo-captions generated by these methods lack the fine-grained details, and variations that are necessary for comprehensive understanding."
- What MetaFind actually states it takes from ULIP-2 is the **embedding backbone** — `2methdology.tex:14`, "both leveraging the ULIP-2 embedding backbone"; `neurips_2025.tex:100`, "MetaFind builds upon ULIP2 … a tri-modal learning framework". **MetaFind never states that it adopts ULIP-2's text construction.**

→ **UNKNOWN** whether MetaFind adopts ULIP-2's text pipeline. Under the reproduction rule in `.claude/rules/paper-reproduction.md` §4, an upstream detail may be inherited only where there is evidence MetaFind adopts it; there is none here. Recorded as mild upstream evidence *against* templating, decisively outweighed by MetaFind's own §2.3, which states the annotations detail category, dimensions, materials and placement — attributes a raw ULIP-2-style caption would not carry.

#### V6 — The `MEASURED r = 0.52-0.62` figure at `resolve_stage1.py:111`. **UNVERIFIED.**

No script, artifact, test, or log in this repository reproduces this correlation. `raw_bbox_extents` is present in every v3 record, so the claim is reproducible in principle, but reproducing it would be a new experiment and is outside this decision's scope. It is cited in support of *keeping* the size fields and *omitting* `mass`; the omission of `mass` therefore rests partly on an unverified number.
→ **Must not be reported as MEASURED until reproduced.** Recommended follow-up, not a blocker: it argues for the current behaviour, so it cannot be the reason to change it.

---

### 6.A New findings, not present in Sections 1–5

All are OBSERVED DATA / OBSERVED IMPLEMENTATION, measured over the 45,952 `prompt_version:3` records at commit `1837477`.

**N1 — 161 records serialize a real dimension as `0 centimetres`. (0.35%)**
163 non-integer dimension values exist in the corpus (chiefly `0.5` cm heights). `:.0f` renders `0.5` as `"0"`. Actual current output:

```
A circular coin with detailed engravings and text, likely made of metal.
A coin made of metal, roughly 4 by 4 by 0 centimetres, typically placed on top
of other objects.
```

The string asserts a zero dimension for a physical object. This is the serialization **misrepresenting the record it serializes**, for 161 gallery assets.

**N2 — 3,643 records (7.93%) emit an ungrammatical article.**
The template hardcodes `"A {category}"`. Vowel-initial categories produce "A airplane" (696), "A air conditioner" (228), "A air purifier" (207), "A umbrella" (148), "A apple" (129), "A egg" (97), "A octopus" (85), "A orange" (84), "A aircraft" (84)… The stated reason for choosing prose over a labelled record (`resolve_stage1.py:87-91`) is that the frozen tower was **caption-trained**; a construction no caption corpus contains works directly against that rationale.

**N3 — `MAX_PLACEMENT = 2` is defined and never used.**
`resolve_stage1.py:162` defines it; nothing reads it (`grep` over `metafind/` and `tests/`: `MAX_MATERIALS`, `MAX_DESCRIPTION_CHARS`, `MAX_CATEGORY_CHARS` are all consumed at lines 296–302; `MAX_PLACEMENT` is not). The comment at `resolve_stage1.py:141-149` asserts "EVERY variable-length part is bounded, not just the description". The placement clause is not bounded. Corpus impact is nil — exactly **1** record has more than two placement flags true — but this is the third in-code justification that does not describe the code.

**N4 — `PLACEMENT_PHRASES[("onWall", "onCeiling")]` is unreachable.**
`placement_phrase()` builds its key in the fixed order `(onCeiling, onWall, onFloor, onObject)` and retries with `tuple(sorted(on))`; neither can produce `("onWall", "onCeiling")`. Confirmed by rendering the corpus: the 90 records with both flags emit the fallback join `"typically mounted on a ceiling or on a wall"`, never the curated `"typically mounted on a wall or ceiling"`. Output remains grammatical, so this is a dead entry rather than a defect in the text.

**N5 — the caps are near-inert on the real corpus.**
Description capped at 160 chars: 164 records (0.36%). Category capped at 40 chars: **0**. Fourth-or-later material dropped: 52 (0.11%). Placement-flag histogram: 0 flags → 2,127 (these get `NO_PLACEMENT_PHRASE`), 1 → 35,255, 2 → 8,569, 3 → 1, 4 → 0. All 11 observed placement renderings were enumerated and every one reads as grammatical prose.

**N6 — remedy cost, measured (not estimated).**
Re-serializing the whole corpus under candidate variants and re-tokenizing with the validated tokenizer:

| variant | median | p99 | max | ≥77 | records rendering a 0 dimension |
|---|---|---|---|---|---|
| current `:.0f` | 49 | 62 | 77 | 1 | **161** |
| `.1f` with trailing `.0` stripped | 49 | 62 | 77 | 1 | **0** |
| unconditional `.1f` | 55 | 68 | 77 | **4** | 0 |
| stripped `.1f` **+** a/an agreement | 49 | 62 | 77 | 1 | 0 |

Adaptive formatting removes all 161 zero-renderings at **zero token cost**. Unconditional `.1f` costs +6 median tokens and quadruples the truncation count — it is the worse remedy. Article agreement is free.

---

### 6.B Options

Four options. Each is materially distinct and each is raised by evidence above; none is filler.

---

#### Option A — Ratify verbatim

**Description.** Ratify `TEXT_TEMPLATE`, the field order, centimetres at `:.0f`, the `synset`/`volume`/`mass` omissions, and `PLACEMENT_PHRASES`/`NO_PLACEMENT_PHRASE` exactly as they stand. Discharge the `CONFIRM BEFORE THE FULL RUN` marker with no code change. C-002 then rewrites `stage1_encoding_protocol.json` to match the code.

**Evidence supporting.**
- The paper specifies no format (V1), so no option is paper-mandated and the incumbent is not paper-violating.
- Centimetres and `width, length, height` are the figure-supported readings (V1); the code already holds both.
- The token budget is safe: 1 record of 45,952 at the limit, and that one is caused by a CJK description (V4).
- The design intent — natural prose for a caption-trained frozen tower — is sound and is the reason the placement phrasing was rewritten from v1's degenerate `"typically placed floor."`.
- Zero implementation risk; the golden test and all 442 tests stay green; D1 unblocks immediately.

**Evidence against.**
- N1: 161 gallery assets get text asserting a zero dimension. Ratifying makes that a **recorded, deliberate** property of the reproduction rather than an oversight.
- N2: 7.93% of the corpus receives an ungrammatical article, contradicting the very rationale used to justify the template form.
- The in-code justification is inaccurate in three places (V2, N3, and — as dead code — N4). Ratifying the template verbatim also ratifies comments that misdescribe the code and misattribute a claim to §2.3.

**Consequences.** D1 proceeds now. The reproduction report must disclose N1 and N2 as accepted properties of the Stage 1 text distribution.

**Risks.** Low operational, non-zero scientific: 161 assets carry a factually wrong dimension in the only text the gallery tower ever sees for them, and the defect is now on the record as intentional.

---

#### Option B — Ratify the design; require two bounded serialization corrections before the full run **(recommended)**

**Description.** Ratify everything Option A ratifies — prose form, field set, field order, centimetres, the three omissions, the placement phrases — and additionally require, before D1 runs n06, exactly two corrections, both classified as **bug fixes** because in each case the emitted string misdescribes the record it serializes:

1. **Dimension precision.** Stop rendering a stored non-zero dimension as `0`. Measured remedy (N6): format `.1f` and strip a trailing `.0`, giving `"4 by 4 by 0.5 centimetres"`. Zero token cost. Affects 161 records; the other 45,791 strings are **byte-identical to today's**.
2. **Article agreement.** `A` → `An` before a vowel-initial category. Zero token cost. Affects 3,643 records.

Everything else stays byte-identical. Then C-002 rewrites the artifact to the corrected template, and the `text_serialization` identifier is versioned (MIF-1).

**Evidence supporting.**
- N1 and N2 are measured defects, not preferences. A 0.5 cm coin described as "0 centimetres" is the serialization contradicting `data/outputs/annotations/*.json`, which is the higher-authority object.
- N6 measures both remedies as free: median 49, p99 62, max 77, 1 record ≥77 — identical to today. The token-budget argument that justifies the template's terseness is not weakened.
- Fixing before the run is strictly cheaper than after: n06 is ~4 GPU-hours, and any later fix invalidates the entire embedding cache, the gallery index, and any Stage 1 checkpoint trained on it.
- Preserves every element the figure supports (cm, `w,l,h` order) and every element with a defensible rationale.

**Evidence against.**
- Changes the string for 3,795 records (8.26% of the corpus), so it is not a no-op: the golden-string test `L1-TEXT-SERIALIZATION` must be updated deliberately, which is exactly what that test exists to force.
- Neither defect is paper-driven. Both remedies are IMPLEMENTATION CHOICES; the paper is silent, so a reviewer could argue the incumbent is equally admissible.
- Adds a small code change to D1/D2's critical path before ~4 GPU-hours of work.

**Consequences.** One narrow edit to `resolve_stage1.py` (the template string plus a two-line article helper and a one-line dimension formatter), one golden-string update, then C-001 + C-002 + n05b + n06 in the existing order.

**Risks.** Low. The change is bounded, measurable, and fully re-verifiable by re-running the corpus serialization before spending GPU time. The residual risk is scope creep — see the guard in §11.

---

#### Option C — Reject the field order; re-serialize in the paper's stated attribute order

**Description.** Move dimensions before materials so the sentence follows `2methdology.tex:28` and `:24` literally: description → category → dimensions → materials → placement.

**Evidence supporting.** Two independent paper statements use that order (V2), and the current in-code comment already claims to follow it. Adopting it would make the claim true.

**Evidence against.**
- Both statements say "such as" — a non-exhaustive attribute list, not a serialization order. Neither sentence describes a string.
- It would move **every** text embedding in the corpus for a reason the paper does not actually assert, in exchange for no measurable benefit.
- `tests/test_resolve_stage1.py:60-69` uses precisely this swap as its *negative injection* — the codebase already treats it as the canonical example of a silent, harmful edit.

**Consequences.** Every one of 45,952 strings changes; the golden string changes; no measurable gain.

**Risks.** Converting an INFERENCE about English prose ordering into a pseudo-requirement. This is the failure `.claude/rules/paper-reproduction.md` §3 names directly.

---

#### Option D — Reject templating; use the VLM description alone as the text modality

**Description.** Feed `annotation["description"]` to the frozen text tower with no structured tail, mirroring ULIP-2's use of a raw generated caption.

**Evidence supporting.** The nearest upstream precedent (V5): ULIP-2 uses a raw BLIP-2 caption and explicitly criticises metadata-templated "pseudo-captions". A defensible reading of `2methdology.tex:28` — "These annotations provide rich textual descriptions" — is that the description *is* the text modality.

**Evidence against.**
- The same sentence says those descriptions detail "object category, size dimensions, materials, and placement constraints". The v3 `description` field does **not** systematically carry dimensions, materials, or placement — verified across the corpus. Description-only would drop attributes §2.3 states are present.
- MetaFind never states it adopts ULIP-2's text construction; it states it adopts the ULIP-2 **backbone** (V5). Inheriting the text pipeline would violate `.claude/rules/paper-reproduction.md` §4.
- It would discard the placement constraint, which the layout-aware objective depends on, from the text modality entirely.

**Consequences.** A materially different Stage 1 text distribution and a weaker text tower for layout-conditioned retrieval.

**Risks.** High. Recorded for completeness and as the option Codex should attack hardest; not recommended.

---

## 7. Analysis

**Against primary evidence.** No option is paper-mandated, because the paper specifies no serialization format (V1, PAPER FACT as to silence). The paper does constrain two things the four options treat differently: the dimension *scale* (figure arithmetic → centimetres) and the dimension *ordering as the authors print it* (width, length, height). A and B both honour both; C leaves them intact but reorders the surrounding clauses on evidence that does not support the weight placed on it; D discards the fields entirely. Option C's supporting evidence collapses on inspection — "such as" marks a non-exhaustive list, and the codebase's own negative-injection test treats that exact swap as the canonical silent-harm edit. Option D contradicts §2.3's own description of what the annotations contain.

**Against reproducibility.** The template determines every text embedding, hence every text-conditioned column of Table 1. The reproducibility requirement is not that the template be optimal — it cannot be, since the paper does not specify one — but that it be **recorded, stable, and truthfully described**. Options A and B both satisfy that once C-002 lands. A leaves three inaccurate in-code justifications (V2, N3, N4) standing next to a ratified constant, which is a traceability defect in itself: a future reader who trusts `resolve_stage1.py:93-95` will believe the field order is paper-derived when it is not. Under Option B those comments must be corrected as part of the same edit.

**Against implementation impact.** A is zero-cost. B is one narrow edit whose effect is fully measured *before* any GPU time: 3,795 of 45,952 strings change, the remaining 42,157 are byte-identical, and token statistics are unchanged (N6). C changes all 45,952 strings for no measured benefit. D is a redesign.

**Against downstream dependencies.** The decisive asymmetry is ordering cost. Any template change after n06 invalidates 45,952 cached embeddings (~4 GPU-hours), the gallery index built from them, and any Stage 1 checkpoint trained on them — `gallery_index.py` fingerprints the checkpoint, so a drifted encoder yields self-consistent wrong numbers with no error. Fixing before the run costs one edit; fixing after costs the whole chain. That argues for resolving every known text defect **now**, which is Option B, and against the "ratify now, revisit later" posture of Option A.

**Against scientific validity.** N1 is the load-bearing finding. Text reading "roughly 4 by 4 by 0 centimetres" is not a stylistic imperfection: for 161 gallery assets the only text the tower ever sees asserts a false property, and it asserts it about the size fields, which §2.3 names explicitly. Under `.claude/rules/code-changes.md` §6, existing behaviour is OBSERVED IMPLEMENTATION and is not correct merely because it exists and its tests pass. The correct classification of the `:.0f`-on-0.5 behaviour is a **bug**: the serialization contradicts the annotation record, a higher-authority object in the same repository. N2 is weaker — CLIP tolerates "A airplane" — but it is free to fix and it directly contradicts the caption-trained rationale used to justify the template's form.

**Against uncertainty.** Two uncertainties survive every option and must not be hidden. The `MEASURED r = 0.52-0.62` figure (V6) is unreproduced in this repository; it supports keeping the current behaviour, so it cannot justify changing it, but it must not be reported as MEASURED. And the `text_serialization` identifier does not discriminate template versions (V3/MIF-1) — a hazard that survives C-002 unless the identifier is versioned, because the artifact will then *silently* describe the fresh run correctly while the 5,276 stale sidecars carry the same label over different text.

**Where the balance lands.** A and B differ on exactly one question: is emitting `0 centimetres` for a 0.5 cm object acceptable in a ratified reproduction artifact? It affects 0.35% of the corpus, the remedy is measured at zero token cost and zero effect on 99.6% of the strings, and the window in which it is cheap to fix closes the moment n06 starts. Ratifying it verbatim would be recording a known defect as a deliberate scientific choice for no benefit.

---

## 8. Recommended Decision

**Recommendation: Option B — ratify the design, with two bounded serialization corrections required before n06.**

Ratify as recorded IMPLEMENTATION CHOICE (U-15):

1. **Template form** — natural prose, single sentence, description first. RATIFY.
2. **Field set** — description, category, materials, dimensions, placement. RATIFY.
3. **Field order** — description → category → materials → dimensions → placement. RATIFY as an IMPLEMENTATION CHOICE, **and correct the comment at `resolve_stage1.py:93-95`**, which misattributes the order to §2.3. The paper does not constrain serialization order.
4. **Unit** — centimetres. RATIFY, and **upgrade the classification**: this is an INFERENCE supported by the figure's own arithmetic (`30×30×40=36000`) plus the corpus's `dimension_unit: "cm"`, not a bare free choice.
5. **Dimension ordering `width, length, height`** — RATIFY; it matches the figure. The artifact's `length, width` is the defect.
6. **Precision** — **REJECT `:.0f` as written.** Adopt a formatter that does not render a stored non-zero dimension as `0`. Measured remedy: `.1f` with a trailing `.0` stripped. Classification: **bug fix**, because the current output contradicts the annotation record.
7. **Article** — **REJECT the hardcoded `"A"`.** Adopt a/an agreement. Classification: **bug fix** against the template's own stated caption-trained rationale.
8. **Omission of `synset`, `volume`, `mass`** — RATIFY. `synset` is an identifier, not language; `volume` is confirmed redundant by the figure's own arithmetic; `mass` has no visual grounding. Record that the supporting `r = 0.52-0.62` figure is **unverified in this repository**.
9. **`PLACEMENT_PHRASES` / `NO_PLACEMENT_PHRASE`** — RATIFY. All 11 renderings observed on the real corpus are grammatical, and the four-boolean rendering is a genuine improvement over v1's degenerate `"typically placed floor."`. Record N4 (the unreachable `("onWall","onCeiling")` entry) as a known cosmetic dead branch; fixing it changes 90 strings and is **optional**, at Master's discretion.
10. **`MAX_PLACEMENT`** — record N3: defined, unused, and contradicted by its own comment. Corpus impact nil. No behavioural change required; the comment should stop claiming a bound that does not exist.

**Confidence:** Moderate-to-high on the recommendation; high on the underlying measurements.
High confidence attaches to what was measured directly on the full corpus with a tokenizer validated exactly against the pipeline (V1, V2, V4, N1, N2, N5, N6). Moderate confidence attaches to the judgement call — whether N1 warrants a pre-run edit rather than a disclosed limitation — because the paper is silent and both readings are admissible.

**Evidence classification of what is ratified:**

| Element | Class |
|---|---|
| Serialization format exists at all | **IMPLEMENTATION CHOICE** — paper is silent (PAPER FACT as to silence) |
| Field set | **PAPER FACT** — the five attributes are named in `2methdology.tex:28` and printed in Figure 2 |
| Field order | **IMPLEMENTATION CHOICE** — paper does not constrain it; prior claim to §2.3 rejected |
| Centimetres | **INFERENCE**, doubly supported (figure arithmetic + upstream schema); corroborated by OBSERVED DATA (`dimension_unit: "cm"` in all 45,952 records) |
| `width, length, height` ordering | **INFERENCE from Figure 2**, which prints that order |
| Dimension precision (proposed change) | **BUG FIX**, remedy is an IMPLEMENTATION CHOICE |
| Article agreement (proposed change) | **BUG FIX**, remedy is an IMPLEMENTATION CHOICE |
| Omission of `synset` / `volume` / `mass` | **IMPLEMENTATION CHOICE**; `volume`'s redundancy is figure-confirmed, `mass`'s rationale rests partly on an **UNVERIFIED** number |
| Placement phrasing | **IMPLEMENTATION CHOICE** |
| 77-token safety | **OBSERVED DATA** — 1 of 45,952 at the limit, cause is a CJK description |

**Reason.** The paper mandates no template, so ratification is about recording a defensible, stable, truthfully described choice. The current template is defensible in form, in field set, in unit, and in dimension ordering — the last two better supported than Section 4 credited. It is not defensible in two measured particulars where the emitted string contradicts the record it serializes, and both remedies are measured at zero token cost, leave 91.7% of the corpus byte-identical, and are cheap only in the window before ~4 GPU-hours of encoding closes it.

> **Section 8 is the PRE-REVIEW recommendation and is retained for the audit trail. Codex review (§9) materially corrected its classifications, its completeness, and its sufficiency. §11 supersedes it.**

---

## 9. Codex Adversarial Review

**Status: COMPLETED.** Two rounds, `codex exec --sandbox read-only`, `codex-cli 0.148.0`, model `gpt-5.6-sol`, reasoning effort xhigh, workdir `/home/kyzen/MetaFindV1`, commit `1837477`.

- **Round 1** (session `01a02030-b8dc-7663-9274-b70470568896`) exhausted its budget re-reading paper sources and terminated before writing findings. It is **not** counted as a review, but it did land one substantive hit before stopping, recorded as C-0 below.
- **Round 2** ran to completion and produced a full findings report (73,969 tokens). Round 2 was explicitly instructed to attack the judgement calls rather than re-verify the paper survey, and to accept C-0 as already conceded.

Codex was briefed adversarially: it was told its job was to attack, given the project's evidence-classification rules and authority hierarchy, handed all my measurements, and asked ten specific attack questions. It was told not to be agreeable. It was not shown Sections 6–8 as a document to approve — it was given the claims as targets.

**Codex verdict: `BLOCKED BY UNKNOWN`. Disposition: REJECT Option B as written. Do not launch n06.**

Findings as Codex reported them:

| # | Severity | Finding |
|---|---|---|
| C-0 | MAJOR | `30×30×40 = 36000` is **unit-invariant** arithmetic. Without a printed unit it cannot establish centimetres specifically; at most the depicted robot's scale makes centimetres plausible. |
| C-1 | **BLOCKER** | C-002 and identifier versioning **do not invalidate the 5,276 stale embeddings.** `is_complete()` checks only `encoder_version` and NPZ existence. A resumed n06 skips metre-derived embeddings while producing centimetre-derived ones for new assets → a mixed gallery. |
| C-2 | **BLOCKER** | The recorded protocol is **not binding** on the actual serializer. `load_protocol()` never validates `text_serialization` or `text_template`; the system can encode with serializer X and label it Y without failing. |
| C-3 | MAJOR | "Token budget is safe" is overstated. A fixed-width tokenizer saturates at 77, so `max = 77` cannot by itself distinguish 77 from 89; and n06 *records* the overflow but **still encodes it** (line 202). Placement-last ordering and character-based caps are serializer design contributions to which field is lost, so the cause is not solely annotation language. |
| C-4 | MAJOR | Calling N1 a **BUG FIX overreaches**. `.0f` behaves exactly as selected; no specification requires positive dimensions to remain positive. It is a *defective IMPLEMENTATION CHOICE* requiring approval, not a bug label that bypasses research governance. |
| C-5 | MAJOR | `.1f`-stripped is not established as the correct numeric policy. Values with >1 decimal stay rounded; Python's ties-to-even renders `0.25` as `0.2`; tenths may imply unsupported precision for VLM-estimated dimensions; the fractional vocabulary beyond "mostly 0.5" was never enumerated. |
| C-6 | MAJOR | **"Volume is redundant" is false at the encoder-input layer.** CLIP is not guaranteed to multiply three numerals; the string rounds the dimensions, destroying exact equivalence; the N1 case implies volume 0 from a record whose volume is positive. Omission may still be right — but not on redundancy grounds. |
| C-7 | MAJOR | `A → An` by first letter is **not** article agreement: "hour", "university", "one-piece", "MRI", "USB" depend on pronunciation. And better grammar does not establish better retrieval; "zero token cost" means no count increase, not zero embedding impact. |
| C-8 | MINOR | V2 is right about the contradiction, but "post-hoc" **asserts motive without evidence**. Post-hoc, stale, or simply mistaken is UNKNOWN. |
| C-9 | MAJOR | Ratifying `PLACEMENT_PHRASES` wholesale contradicts my own N4, and overlooks that the table makes **semantic expansions** — `onWall` → "mounted", `onObject` → "on top", all-false → "no typical placement" — which are IMPLEMENTATION CHOICES, not schema-preserving paraphrases. |
| C-10 | MINOR | More is unbounded than `MAX_PLACEMENT`: individual material strings are uncapped; `_cap()` fails its own word-boundary guarantee for a space-free string (`rsplit` returns the whole mid-word prefix); `_cap()` can append a second period; descriptions ending in `!`, `?`, or `。` still get an ASCII period appended. |
| C-11 | MAJOR | The stale sidecars already **demonstrate** a provenance inconsistency, not merely a future hazard. The old identifier should be **retired as ambiguous**, not reused as a valid cache identity. |
| C-12 | MINOR | "ULIP-2 is outweighed by MetaFind §2.3" is unsupported. §2.3 establishes the attributes exist; it does not establish they should be concatenated into a prompt. Upstream evidence is weakly cautionary and MetaFind is silent — neither side wins. |

---

## 10. Claude Verification of Codex Findings

Each finding was checked against source, code, or a fresh measurement. Measurements below were run read-only at commit `1837477` with `/home/kyzen/miniconda3/envs/MetaFind/bin/python`.

### C-0 — `CONFIRMED` · MAJOR · changes the recommendation's *justification*

Codex is right and I was wrong. `30 × 30 × 40 = 36000` holds in **any** consistent unit; the arithmetic establishes only that `volume = w·l·h`, never which unit. My §6.0 V1 CLAIM A converted an inference into a stronger support than it carries.

**What actually survives**, re-derived: the figure prints `mass: 2.5` alongside `30/30/40`. Under centimetres that is a 0.3 × 0.3 × 0.4 m box at 2.5 kg → ≈ 69 kg/m³, light but physical for a hollow toy robot. Under metres it is a 30 × 30 × 40 m object at 2.5 kg → ≈ 7 × 10⁻⁵ kg/m³, roughly 1/17,000 the density of air. **Physical plausibility of the mass/size pairing**, not the volume arithmetic, is what makes centimetres the defensible reading — and this repository already uses exactly that density argument as a validator (`metafind/data/annotate.py:408-422`).

Revised classification: centimetres is an **INFERENCE**, supported by (i) density plausibility of the figure's own mass/size pairing and (ii) the unstated Holodeck schema match (`annotate.py:115-143`). **Not** supported by the volume arithmetic.

**Crucially, the ratification does not depend on resolving MetaFind's intent at all.** All 45,952 v3 records store `"dimension_unit": "cm"` (OBSERVED DATA, 0 records with any other unit). Whatever the paper meant, the template must say what the corpus holds. The artifact's "metres" is false about our data; the code's "centimetres" is true about it. That argument is independent of C-0 and unaffected by it.

### C-1 — `CONFIRMED` · **BLOCKER** · MASTER-IMPACTING · I missed this entirely

Verified by direct reading, `metafind/data/encode_text_image.py:73-83`:

```python
def is_complete(uid: str) -> bool:
    sc = sidecar_path(uid)
    if not sc.exists():                              return False
    try:    rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):          return False
    if rec.get("encoder_version") != ENCODER_VERSION: return False
    return Path(rec.get("embedding_uri", "")).exists()
```

It compares **nothing** about the text: not `text`, not `text_serialization`, not the template, not the annotation revision. `encode_text_image.py:178-179` builds the work list as `[p for p in annotations if p.stem in renders and (args.force or not is_complete(p.stem))]`.

**Consequence, confirmed:** a plain `python -m metafind.data.encode_text_image` after C-002 lands would skip all 5,276 metre-derived assets as "complete" and encode only the remaining ~40,676 in centimetres. The result is a gallery whose text embeddings come from **two different text distributions**, with no error, no warning, and a `text_serialization` label that reads identically on both halves. `gallery_index.py` fingerprints the checkpoint, not the text, so downstream numbers would be self-consistent and wrong.

This is real, it survives C-002, it survives identifier versioning, and my §8 was wrong to imply C-002 was sufficient. Reported as **`MASTER-IMPACTING FINDING MIF-4`**.

### C-2 — `CONFIRMED` · BLOCKER · already partly in §6.0 V3, correctly sharpened

`load_protocol()` (`encode_text_image.py:86-108`) validates exactly `status`, `actual_clip_train_scope`, and `image_aggregation`. It never touches `text_template` or `text_serialization`. Meanwhile `encode_text_image.py:194` calls `serialize_annotation(annotation)` with the **imported default** template, and line 233 stamps `protocol["text_serialization"]` into the sidecar. Encoder and label are independent.

I had recorded the label ambiguity (V3/MIF-1). Codex is right that **versioning the identifier does not fix it** — an unenforced label is still unenforced, whatever it is called. The protocol must be *bound* to the executed serializer (exact comparison or hash), or the artifact remains decorative. This escalates MIF-1 from "hazard" to "unmet requirement".

### C-3 — `PARTIALLY CONFIRMED` · MAJOR · methodology sound, wording too strong

**Methodological objection — REJECTED, and now closed by measurement.** Codex is right that a padded tokenizer saturates at 77 and cannot distinguish 77 from 89. I therefore re-measured with the **untruncated** BPE path (`SimpleTokenizer.encode`, no padding, +2 for SOT/EOT) over all 45,952 strings:

| | true tokens |
|---|---|
| median | 49 |
| p99 | 62 |
| **true max** | **89** |
| records **> 77** | **1** |
| records **exactly 77** | **0** |

There is exactly one record above the limit and its true length is 89. The saturation ambiguity is fully resolved, not assumed away.

**Substantive objection — CONFIRMED.** "The token budget is safe" was too strong on two counts Codex names correctly:
1. n06 **records and then encodes** the overflow (`encode_text_image.py:198-202`). The comment at `encode_text_image.py:64-65` says "An overflow is RECORDED, never silently encoded" — the behaviour is recorded *and* encoded, so one degraded gallery embedding is produced.
2. Which 12 tokens are lost is a **serializer design consequence**: placement is last, so placement is what CLIP drops. Attributing it purely to "annotation language quality" understates the template's contribution.

Revised wording adopted: **one known truncation, recorded and still encoded.** Its disposition is a decision, not a footnote — see §11.

### C-4 — `CONFIRMED` · MAJOR · classification corrected, conclusion unchanged

Codex is right on governance. `.0f` does exactly what it says; no specification in this repository requires a positive dimension to render as positive; and the serializer is *already* deliberately lossy (caps, omissions). Calling it a "bug fix" would let a research-significant template change bypass the approval that `.claude/rules/research-rigor.md` §2 and `.claude/rules/code-changes.md` §4 require.

**Reclassified: `defective IMPLEMENTATION CHOICE`.** The substantive judgement — that emitting "roughly 4 by 4 by 0 centimetres" for a 0.5 cm object is not something a reproduction should ratify — is unchanged. What changes is that its revision now explicitly requires Kyzen's sign-off rather than being executed as a repair. See §11.

### C-5 — `PARTIALLY CONFIRMED` · MAJOR · the gap was real; now closed by enumeration

Codex correctly identified that I never enumerated the fractional vocabulary. Enumerated now, over all 45,952 v3 records:

**Every non-integer dimension value in the corpus is one of exactly two: `0.5` (156 values) or `0.2` (7 values).** 163 values total, across 161 records.

Against that:
- `format(0.5,'.1f') = '0.5'` and `format(0.2,'.1f') = '0.2'` — both exact. **No value is rounded** by the proposed policy on this corpus.
- The ties-to-even example Codex raises is real Python behaviour (`format(0.25,'.1f') = '0.2'`) but **`0.25` does not occur**. `REJECTED` for this corpus; retained as a valid robustness caveat for any future annotation batch.
- "Tenths imply unsupported precision" — a fair judgement, but the tenths are already *in the record*; rendering them is faithful to it, and 45,791 records still render as bare integers.

`CONFIRMED` residue: the policy must be **written down explicitly**, including its behaviour for fractional values this corpus does not contain, rather than being defined by a format string. Adopted into §11.

### C-6 — `CONFIRMED` · MAJOR · I over-claimed and must withdraw it

My §6.0 V1 said the figure's arithmetic "independently confirms the redundancy argument". That is wrong for the reason Codex gives: algebraic derivability at the *schema* layer is not redundancy at the *encoder-input* layer. A frozen CLIP text tower is not guaranteed to multiply three numerals in a sentence. Worse, the rounding makes it concretely false — the N1 coin renders "4 by 4 by 0", from which no reader recovers the recorded volume of 8.

**Withdrawn.** The omission of `volume` may still be the right call — token cost inside a 77-token budget, and a noisy VLM-estimated quantity — but it is now recorded as an **IMPLEMENTATION CHOICE with unknown retrieval impact**, not as a justified redundancy. The same downgrade applies to `mass` (whose supporting `r = 0.52-0.62` is separately UNVERIFIED, §6.0 V6) and to `synset`.

### C-7 — `CONFIRMED` · MAJOR · remedy under-specified; corpus checked

Codex is right that first-letter substitution is not article agreement. English article choice follows pronunciation: *an* hour, *a* university, *a* one-piece, *an* MRI, *a* USB. A naive vowel test gets all five wrong.

Corpus check: the observed vowel-initial categories are dominated by ordinary vowel-sound words — airplane (696), air conditioner (228), air purifier (207), umbrella (148), apple (129), egg (97), octopus (85), orange (84), aircraft (84), ice cream cone (80), axe (75), earphone (70). No high-frequency `u`-as-"yoo" or silent-`h` category appears in the top band, so a naive rule would be *mostly* right — but "mostly right" is not a specification, and the vocabulary was not exhaustively audited.

Codex's second point also stands: **zero token cost is not zero embedding impact.** Fixing the article moves 3,643 text embeddings; nothing here establishes it moves them in a helpful direction. `UNKNOWN` — and it is the reason this must be Kyzen's call rather than mine. §11 records article-free wording as the alternative that removes the problem instead of approximating a solution to it.

### C-8 — `CONFIRMED` · MINOR

"Post-hoc" attributes motive I cannot evidence. Corrected wording: the comment at `resolve_stage1.py:93-95` is **objectively false as documentation**; whether it is post-hoc rationalisation, stale after an edit, or simply mistaken is `UNKNOWN`. The substantive point is untouched: it supplies no reason to keep *or* change the actual field order.

### C-9 — `PARTIALLY CONFIRMED` · MAJOR

- **Unreachable entry — CONFIRMED**, and it was my own N4; Codex is right that "ratify the dictionary as-is" while recording a dead entry is internally inconsistent. §11 now gives it an explicit disposition instead of a footnote.
- **Semantic expansions — CONFIRMED as a framing correction.** I had classified the whole dictionary as IMPLEMENTATION CHOICE, which is the right class, but I did not *enumerate* what was invented. Enumerated: `onWall` → "mounted" (the schema says only that it can be on a wall, not that it is mounted); `onObject` → "on top of other objects" (the schema does not say *on top*); all-false → "with no typical placement" (an assertion the schema does not make — it asserts four falses, which is not the same as asserting a positive absence). All 11 renderings observed on the corpus were listed in §6.0 N5 and all are grammatical; none is schema-preserving.

### C-10 — `PARTIALLY CONFIRMED` · MINOR · logic real, corpus impact now measured

Codex flagged four defects and correctly noted their corpus impact was unmeasured. Measured now:

| Codex claim | Logic | Corpus impact (45,952 v3) |
|---|---|---|
| individual material strings uncapped | `CONFIRMED` — `resolve_stage1.py:296` caps the *count*, never the length | 22 records have a material string > 20 chars |
| `_cap()` violates its own word-boundary promise on a space-free string | `CONFIRMED` — `text[:limit].rsplit(" ",1)[0]` returns the whole mid-word prefix when there is no space | **0 records** — latent, never fires |
| `_cap()` can append a second period | `CONFIRMED` as reachable logic | **0 records** — latent, never fires |
| non-`.` terminators get an ASCII period appended | `CONFIRMED` — `resolve_stage1.py:298` tests only `endswith(".")` | 1,982 records (4.3%) gain a period; their final characters are ordinary letters, so the addition is correct English. The pathological case Codex names is real but confined to the CJK records — the syringe description ends `活塞。` and becomes `活塞。.` |

So: two of four are latent-only, one is benign at scale, one (uncapped material length) is real but small. Codex's broader conclusion is the one that matters and is `CONFIRMED`: **the in-code safety comments and the current tests cannot be relied on as validation of these properties.** That is now the third and fourth instance of an in-code justification not describing the code (with V2 and N3).

### C-11 — `CONFIRMED` · MAJOR

The 5,276 stale sidecars carry `"text_serialization": "metafind_v1_natural"` next to metre-based text. Verified directly. The identifier already names two different transformations, so it is not a valid cache identity today, before any change. Codex's remedy — **retire the identifier as ambiguous rather than reuse it** — is stronger than my "version it" and is adopted. Folds into MIF-1/MIF-4.

### C-12 — `CONFIRMED` · MINOR

My §6.0 V5 said ULIP-2's counter-precedent was "decisively outweighed by MetaFind's own §2.3". Codex is right that §2.3 establishes only that the annotations *contain* those attributes; it does not establish that they must be **concatenated into one prompt** for the encoder. Corrected: upstream evidence is **weakly cautionary**, MetaFind is **silent**, and neither settles it. Templating is therefore neither mandated nor prohibited — which is exactly why U-15 is an IMPLEMENTATION CHOICE requiring sign-off, and Option D remains a recorded, non-recommended alternative rather than a refuted one.

### Findings I reject or reduce

Only two, both narrow and both stated above: C-0's implication that the centimetre reading collapses (it survives on density plausibility plus, decisively, the corpus's own `dimension_unit: "cm"`), and C-5's `0.25` example (real Python behaviour, absent from this corpus). Everything else Codex raised is confirmed in whole or in substance.

### Net effect on the recommendation

Codex did not overturn the direction — it overturned the **classification, the completeness, and the sufficiency** of §8:

- §8 called two changes "bug fixes". They are defective IMPLEMENTATION CHOICES needing Kyzen's approval (C-4).
- §8 ratified the omissions partly on a redundancy argument that does not hold at the encoder-input layer (C-6).
- §8 said the token budget was "safe" when one embedding is knowingly degraded (C-3).
- §8 treated C-002 plus identifier versioning as enough to make the artifact describe runtime truth. **It is not** (C-1, C-2, C-11) — and that is a blocker on D1 that D0-008's acceptance does not clear.

---

## 11. Final Recommendation to Master

> **User decisions received from Kyzen, 2026-08-21.** E-1, E-2, and E-3 as escalated in the pre-decision draft are now **RESOLVED BY THE USER** and are recorded below as decided policy, not as open questions. The cache-validity finding (MIF-4) is elevated to **BLOCKER** on Kyzen's instruction.
> Two follow-up points that Kyzen's original wording did not settle, **S-1** and **S-2**, were escalated with a measured recommendation each and were **ACCEPTED by Kyzen, 2026-08-21** (「都接受」). They are recorded below as decided policy. **No open questions remain in this decision.**

### Decision proposed

**RATIFY the Stage 1 text serialization design, incorporating the three user decisions below, plus four documentation/dead-code corrections.**

**`D1_n06-reencode` remains BLOCKED after this decision is accepted.** Per Kyzen's instruction, the cache completion/validity defect is a **BLOCKER** that must be fixed before any full n06 run.

---

### 11.1 User decisions — RESOLVED, not open

#### **E-1 · Dimension precision — ADOPTED**

> Kyzen: *"採用。小於 1 cm 的尺寸保留必要小數，不得被 `:.0f` 捨入為 0。"*
> (Adopt. Dimensions below 1 cm keep the decimals they need; they must not be rounded to 0 by `:.0f`.)

**Policy.** Reject `:.0f`. A stored dimension must never render as `0`. Adopted formatter: render at one decimal place and strip a trailing `.0`, so integers stay bare and fractional values keep their decimal.

| stored | current `:.0f` | adopted |
|---|---|---|
| `25.0` | `25` | `25` |
| `0.5` | **`0`** | `0.5` |
| `0.2` | **`0`** | `0.2` |

Verified output on the actual record that motivated this:

```
A circular coin with detailed engravings and text, likely made of metal.
coin made of metal, roughly 4 by 4 by 0.5 centimetres, typically placed on
top of other objects.
```

**Class:** revised **IMPLEMENTATION CHOICE**, approved by the user. Not a bug fix — `.0f` behaved as written; the choice itself was defective (Codex C-4, confirmed).

**Corpus effect:** 161 records stop asserting a false zero dimension. **Zero token cost** — see §11.4.

##### **S-1 — RESOLVED: the formatter applies to every value, not only below 1 cm**

Kyzen's original wording scoped the rule to dimensions *below* 1 cm. Full enumeration of every non-integer dimension value in the corpus showed one value outside that scope:

| value | occurrences | below 1 cm? |
|---|---|---|
| `0.5` | 155 | yes |
| `0.2` | 7 | yes |
| **`2.5`** | **1** | **no** |

Under a literal reading that value would still render `2` and lose 0.5 cm — 20% of the dimension. Escalated with the recommendation to apply the formatter uniformly.

**Kyzen accepted, 2026-08-21.** The formatter applies to **every** dimension value regardless of magnitude. No `< 1 cm` threshold branch is to be implemented. This matches the stated intent (*保留必要小數*), needs no special case, and is robust to future annotation batches that may contain fractional values this corpus does not.

The affected record is `f348e7704ddc4216b39f3fc399bec0e6`, verified output:

```
a circular dartboard with concentric rings and a central bullseye, featuring
multiple darts embedded in it. Dartboard made of plastic, wood, roughly 25 by
25 by 2.5 centimetres, typically mounted on a wall.
```

#### **E-2 · Article — ADOPTED**

> Kyzen: *"採用。移除固定的 `\"A \"`，不要實作 a/an heuristic。"*
> (Adopt. Remove the fixed `"A "`; do not implement an a/an heuristic.)

**Policy.** Delete the hardcoded `"A "` from the template. **No pronunciation or vowel-letter heuristic is to be implemented** — this is an explicit user constraint, and it directly answers Codex C-7, which showed a first-letter rule is wrong for "hour", "university", "MRI", "USB".

Template clause becomes `{category} made of {materials}` rather than `A {category} made of {materials}`.

**Class:** revised **IMPLEMENTATION CHOICE**, approved by the user.

**Corpus effect:** all 3,643 ungrammatical "A airplane" / "A umbrella" / "A apple" strings are eliminated — by removing the construction rather than by approximating a fix to it.

##### **S-2 — RESOLVED: the category's first character is capitalised**

Removing `"A "` left the second sentence opening on a bare lowercase noun:

```
A black flip-flop with a textured sole and a strap across the footbed.
flip-flop made of rubber, fabric, roughly 25 by 10 by 4 centimetres, ...
```

On the template's own stated rationale — the frozen tower is caption-trained (`resolve_stage1.py:87-91`) — that is the residue of the very problem E-2 removes. Escalated with the recommendation to capitalise.

**Kyzen accepted, 2026-08-21.** The category's first character is upper-cased: `category[:1].upper() + category[1:]`.

This **fully honours the E-2 constraint**: capitalisation is not an a/an heuristic, carries no pronunciation model, and adds no vocabulary-dependent branch. Verified output:

```
A black flip-flop with a textured sole and a strap across the footbed.
Flip-flop made of rubber, fabric, roughly 25 by 10 by 4 centimetres,
typically placed on the floor or on other objects.
```

**Measured cost: zero.** Identical token statistics to the un-capitalised variant.

#### **E-3 · The truncated record — ADOPTED (re-annotate)**

> Kyzen: *"重新標註那 1 筆中文 record，使其符合目前英文 annotation 規則。"*
> (Re-annotate the one Chinese record so it conforms to the current English annotation rules.)

**Target:** `3e91980a22da4c0da975cc8ef776972c`. Current description is part English, part Chinese:
`"a medical device used for injecting or抽吸液体，由塑料或金属制成，带有针头和活塞。"`
True length **89 tokens** against CLIP's 77 → 12 tokens lost from the tail, which is the placement clause. n06 records `text_truncated=True` **and encodes it anyway** (`encode_text_image.py:198-202`).

**Action:** re-annotate that single record to English under the current v3 prompt, then re-verify its token count before n06. **Owned by the task that runs n05 re-annotation (D1/D2), not by D0.** This is a data mutation and must not be performed under this decision.

**Expected outcome:** the corpus reaches **0 records over the 77-token limit**, so no gallery embedding is knowingly degraded.

##### Residual, not re-opened

Kyzen's decision covers the one truncated record. The corpus contains **10** v3 records with non-ASCII text, **3** of them CJK. The other two CJK records are **not** truncated and are therefore outside the defect E-3 addresses. Recorded under `MIF-2` as a data-quality item for D0-003's owner; deliberately **not** re-opened here.

---

### 11.2 BLOCKER — cache completion / validity, on Kyzen's instruction

> Kyzen: *"將 resumed n06 cache validity 問題視為 BLOCKER。在 n06 全量執行前，必須先修正 cache completion / validity 判斷，確保舊 metres-based embeddings 不會和新的 centimetres-based embeddings 混用。"*
> (Treat the resumed-n06 cache-validity problem as a BLOCKER. Before the full n06 run, the cache completion/validity check must be fixed so that old metres-based embeddings cannot be mixed with new centimetres-based ones.)

**Confirmed defect.** `metafind/data/encode_text_image.py:73-83`:

```python
def is_complete(uid: str) -> bool:
    sc = sidecar_path(uid)
    if not sc.exists():                               return False
    try:    rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):           return False
    if rec.get("encoder_version") != ENCODER_VERSION: return False
    return Path(rec.get("embedding_uri", "")).exists()
```

It compares nothing about the text: not `text`, not `text_serialization`, not the template, not the annotation revision. The work list (`encode_text_image.py:178-179`) is `[p for p in annotations if p.stem in renders and (args.force or not is_complete(p.stem))]`.

**Consequence if unfixed:** a resumed run skips all **5,276** metre-derived embeddings as "complete" and encodes only the remaining **~40,676** in centimetres — a gallery built from two text distributions, with no error, no warning, and an identical `text_serialization` label on both halves. `gallery_index.py` fingerprints the checkpoint, not the text, so Table 1 would be self-consistent and wrong.

**Compounding defect.** `load_protocol()` (`encode_text_image.py:86-108`) validates only `status`, `actual_clip_train_scope`, and `image_aggregation`. It never checks `text_serialization` or `text_template`, while `encode_text_image.py:194` serializes with the **imported default** and line 233 stamps the protocol's identifier regardless. The system can encode with serializer X and label it Y without failing.

**Blocker exit criteria — all four must hold before a full n06 run:**

| | Requirement |
|---|---|
| B-1 | Every one of the 5,276 stale sidecars is treated as **incomplete** — via a bumped `ENCODER_VERSION`, a fresh embeddings namespace, or `is_complete()` binding to the serialized text. Deleting them is **not** required and is not authorised here. |
| B-2 | `load_protocol()` **refuses to run** when the protocol's recorded template does not match the executed serializer — exact comparison or hash. An unenforced record is decorative. |
| B-3 | `"metafind_v1_natural"` is **retired** as a cache identity. It already names two different transformations, so versioning it is insufficient (Codex C-11). |
| B-4 | A pre-flight check confirms **0 records** render a zero dimension, **0 records** exceed 77 true tokens, and the serialized output matches the ratified template — run before GPU time, costs seconds. |

**Ownership:** D1 / Master. **Not** executed under D0-008.

---

### 11.3 Ratified as recorded IMPLEMENTATION CHOICE (U-15)

| # | Element | Disposition | Class |
|---|---|---|---|
| 1 | Natural prose, single sentence, description first | **RATIFY** | IMPLEMENTATION CHOICE |
| 2 | Field set: description, category, materials, dimensions, placement | **RATIFY** | field set is PAPER FACT (`2methdology.tex:28` + Figure 2); concatenating it into one prompt is an IMPLEMENTATION CHOICE |
| 3 | Field order: description → category → materials → dimensions → placement | **RATIFY** | IMPLEMENTATION CHOICE. The paper does not constrain serialization order |
| 4 | Unit: centimetres | **RATIFY** | INFERENCE (density plausibility of Figure 2's mass/size pairing + unstated Holodeck schema match), **decisively corroborated by OBSERVED DATA** — all 45,952 v3 records store `dimension_unit: "cm"` |
| 5 | Dimension ordering `width, length, height` | **RATIFY** | INFERENCE from Figure 2, which prints that order. The artifact's `length, width` is the defect |
| 6 | Dimension precision | **REVISED per E-1** | IMPLEMENTATION CHOICE, user-approved |
| 7 | Article | **REVISED per E-2** — `"A "` removed, no a/an heuristic | IMPLEMENTATION CHOICE, user-approved |
| 8 | Omission of `synset`, `volume`, `mass` | **RATIFY**, on revised grounds | IMPLEMENTATION CHOICE with **unknown retrieval impact**. The "volume is redundant" justification is **WITHDRAWN** (Codex C-6, confirmed): a frozen text tower is not guaranteed to multiply three numerals, and rounding destroyed exact equivalence anyway. `mass`'s rationale rests partly on an **UNVERIFIED** `r = 0.52-0.62` |
| 9 | `PLACEMENT_PHRASES` / `NO_PLACEMENT_PHRASE` | **RATIFY**, with the semantic expansions named | IMPLEMENTATION CHOICE. `onWall`→"mounted", `onObject`→"on top of other objects", all-false→"with no typical placement" are inventions, not schema-preserving paraphrases |

**Resulting ratified template — final, all user decisions applied:**

```
{description} {Category} made of {materials}, roughly {W} by {L} by {H} centimetres, {placement}.
```

where `{Category}` is the category with its first character upper-cased (S-2), and `W`/`L`/`H` render at one decimal with a trailing `.0` stripped, applied uniformly at every magnitude (S-1).

**Resulting golden string** for `L1-TEXT-SERIALIZATION`, computed from the existing `GOLDEN_ANNOTATION` in `tests/test_resolve_stage1.py`:

```
A wooden dining chair with a slatted back and four tapered legs. Dining chair
made of wood, fabric, roughly 50 by 45 by 90 centimetres, typically placed on
the floor.
```

#### Required corrections — documentation and dead code

| # | Correction | Class | Behavioural impact |
|---|---|---|---|
| R-1 | Correct the field-order comment at `resolve_stage1.py:93-95`. It is **objectively false as documentation**: the paper's order is category → *dimensions* → materials; the code emits category → *materials* → dimensions. Whether it is post-hoc, stale, or mistaken is UNKNOWN (Codex C-8) | documentation | none |
| R-2 | Correct the "EVERY variable-length part is bounded" claim at `resolve_stage1.py:141-149`, and either enforce or delete the unused `MAX_PLACEMENT` (`resolve_stage1.py:162`) | documentation / dead code | at most 1 record |
| R-3 | Fix or deliberately delete the unreachable `PLACEMENT_PHRASES[("onWall","onCeiling")]` entry. Fixing changes 90 strings; deleting changes 0 | IMPLEMENTATION CHOICE | 90 records, or 0 |
| R-4 | Retire `"metafind_v1_natural"` as a cache identity — folded into blocker item **B-3** | reproducibility contract | all sidecars |

---

### 11.4 Measured effect of the decided template

Full corpus, 45,952 `prompt_version:3` records, serialized read-only at commit `1837477`. Token counts use the **untruncated** BPE path (`SimpleTokenizer.encode` + 2 for SOT/EOT), so no count is masked by the tokenizer's 77-wide padding. The padded counter was separately validated at **5,276 / 5,276 exact agreement** against the recorded `text_tokens` of the existing n06 sidecars.

| | current template | **final decided** (E-1 + E-2 + S-1 + S-2) |
|---|---|---|
| median tokens | 49 | **48** |
| p99 tokens | 62 | **61** |
| true max tokens | 89 | **88** |
| records over 77 | 1 | **1** → **0** after E-3 |
| records rendering a `0` dimension | **161** | **0** |
| records with an ungrammatical article | **3,643** | **0** |

Measured on the final template with S-1 and S-2 applied. Capitalisation and the uniform formatter are both token-neutral, so these figures are the accepted configuration's, not an approximation of it.

**The decided template costs nothing and is marginally shorter than the incumbent.** Every measured defect is eliminated.

**Note for the golden-string test.** Unlike the pre-review draft, **all 45,952 strings change** — E-2 removes `"A "` from every one. `L1-TEXT-SERIALIZATION` must be updated deliberately, which is precisely what that test exists to force.

---

### Why

The paper mandates no template — `PAPER FACT` as to silence, verified across all five `.tex` files including the appendix, plus Figure 2. Ratification is therefore not about optimality, which is unreachable, but about recording a **defensible, stable, and truthfully described** choice.

The incumbent is defensible in form, field set, unit, and dimension ordering — the last two better supported than Section 4 credited, since Figure 2 prints `width, length, height` and the corpus stores centimetres. It was **not** truthfully described: four separate in-code justifications do not match the code (field order, the "everything is bounded" claim, the dead placement entry, and the withdrawn redundancy argument). And it was defective in two measured particulars where the emitted string misstated the record it serialized. Kyzen's E-1 and E-2 resolve both at zero cost.

The decisive constraint, however, is not the template. It is that **the reproduction cannot currently guarantee which serializer produced which embedding**, and a resumed n06 would silently build a two-distribution gallery. Kyzen has classified that as a BLOCKER, which is the correct call: it is the only finding here that can produce confident wrong numbers with no error anywhere in the chain.

### Remaining uncertainty

- Whether the decided template retrieves **better** than the incumbent — `UNKNOWN`, and unresolvable without a controlled Stage 1 comparison that is not budgeted. Zero token cost is not zero embedding impact (Codex C-7, confirmed). What *is* established is that the decided template removes 161 false statements and 3,643 ungrammatical constructions from the encoder input.
- `r = 0.52-0.62` at `resolve_stage1.py:111` — **UNVERIFIED** in this repository. It supports current behaviour, so it cannot justify changing it, but it must not be reported as MEASURED.
- Whether MetaFind's authors serialized at all, or fed the VLM description directly — `UNKNOWN`. Option D (§6.B) is recorded, not recommended, and not refuted (Codex C-12).
- Centimetres as MetaFind's *intent* — `INFERENCE` only. Centimetres as *this corpus's* unit — `OBSERVED DATA`, certain.
- The retrieval impact of omitting `synset` / `volume` / `mass` — `UNKNOWN` now that the redundancy argument is withdrawn.
- Latent defects with zero current corpus impact: `_cap()`'s word-boundary guarantee fails on a space-free string (0 records), `_cap()` can emit a doubled period (0 records), individual material strings are uncapped (22 records exceed 20 chars).

### Required repository changes if accepted

**None by D0.** This decision recommends only; no code, artifact, or dataset was modified. In dependency order:

1. **Clear the BLOCKER (§11.2, B-1…B-4).** D1 / Master. Nothing downstream is trustworthy until this holds.
2. **Re-annotate `3e91980a22da4c0da975cc8ef776972c`** to English under the v3 prompt (E-3), then re-verify its token count. Owner: the n05 re-annotation task.
3. **Apply E-1, E-2, S-1, S-2, and R-1…R-4** to `resolve_stage1.py`. All serialization policy is now user-decided; no further sign-off is required for the template itself.
4. **Update `L1-TEXT-SERIALIZATION`** deliberately, and add the coverage Codex found missing: sub-centimetre dimensions, the absent-article form, the wall+ceiling combination, protocol/serializer mismatch, and cache invalidation.
5. **Run the §11.4 pre-flight measurement** as a gate before GPU time — seconds, and now reproducible.
6. **Then** C-001 (τ = 0.5) and C-002 together through n05b — they must land in one call (`resolve_stage1.py:443-444`) — and only then n06.

### Tasks affected

`D1_n06-reencode` (blocked by §11.2, not merely by this decision), `D2_stage1-prereq`, `D3_stage1-train`, `D4` gallery index (would inherit any mixed-distribution gallery), `D7` Table 1 (every text-conditioned column).

### MASTER-IMPACTING FINDINGS

- **MIF-1 — the serialization identifier is not a cache identity.** `"metafind_v1_natural"` already labels both metre and centimetre text. `load_protocol()` never validates `text_serialization` or `text_template`; `encode_text_image.py:233` stamps it regardless. Versioning is insufficient — it must be **bound or retired**. Folded into blocker items B-2/B-3. Affects D1, D2, D4, D7.
- **MIF-2 — 10 v3 records carry non-ASCII descriptions, 3 CJK.** One is the corpus's only truncation and is resolved by E-3; the other two CJK records are not truncated and remain open. Adjacent to D0-003 but distinct. Affects D0-003 scope, D1.
- **MIF-3 — `workflow/CONTEXT.md` §5 and Section 4 of this file understate the centimetre evidence** (INFERENCE with density + schema support, plus decisive OBSERVED DATA) **and overstate the volume-redundancy argument, which is withdrawn.** Affects the reproduction report's wording. Master owns CONTEXT.md; D0 did not edit it.
- **MIF-4 — `is_complete()` does not invalidate stale embeddings** (`encode_text_image.py:73-83`). **Elevated to BLOCKER by Kyzen, 2026-08-21.** See §11.2. Affects D1, D2, D3, D4, D7.

### Confidence

**High** on the measurements: full corpus, no sampling, tokenizer validated at 5,276/5,276 against the pipeline's own counter, and the untruncated path measured separately so no count is masked by padding.
**High** on the ratification: E-1, E-2, E-3, S-1, and S-2 are all user-decided rather than assumed, and the final template was measured over the full corpus in its accepted configuration.
**No open questions remain in this decision.** What is left is execution (the §11.2 blocker, E-3's re-annotation, R-1…R-4) and the residual uncertainties listed above, none of which is a pending choice.

**Status: `RECOMMENDED`. Returned to Master. Section 12 is Master's and has not been touched.**

---

## 12. Master Resolution

**Reviewed by Master, 2026-08-21.** Master did not accept this handoff on assertion. The load-bearing claims were independently re-verified against the repository at commit `1837477` before resolution; the verification log is recorded below.

---

### Resolution

## `ACCEPT WITH FOLLOW-UP`

The decision is complete, evidence-backed, independently reviewed, and correctly scoped. Acceptance carries the mandatory follow-up items in §12.4, including one **BLOCKER** that D0 itself surfaced and that acceptance of this decision does **not** clear.

---

### 12.1 Master's independent verification

Re-run read-only by Master. Every figure below was reproduced, not read off the decision file.

| Claim | Source checked | Result |
|---|---|---|
| D0 modified no code, data, or artifact | `git status --porcelain`; mtimes of `resolve_stage1.py` (08-19), `encode_text_image.py` (08-17), `test_resolve_stage1.py` (08-19), both protocol artifacts (08-16); embeddings still 5,276; annotations still 45,955; checkpoints still empty | **CONFIRMED** — only `workflow/` files and this decision file exist as changes |
| Section 12 untouched by D0 | Template placeholders intact on arrival | **CONFIRMED** |
| **BLOCKER** — `is_complete()` ignores text | `encode_text_image.py:73-83` read directly | **CONFIRMED** — checks sidecar existence, `encoder_version`, and NPZ existence only. Nothing about `text`, `text_serialization`, or the template |
| `text_serialization` stamped regardless of serializer | `encode_text_image.py:233` | **CONFIRMED** |
| Paper attribute order is category → dimensions → materials | `2methdology.tex:28` verbatim: "object category, size dimensions, materials, and placement constraints"; caption `:24` gives the same order | **CONFIRMED** — the in-code claim at `resolve_stage1.py:93-95` is false as documentation (R-1 justified) |
| `MAX_PLACEMENT` unused | `grep -rn "MAX_PLACEMENT" metafind/ tests/` → one hit, its own definition at `:162` | **CONFIRMED** (R-2 justified) |
| `PLACEMENT_PHRASES[("onWall","onCeiling")]` unreachable | `placement_phrase()` source read: builds `on` in fixed order `(onCeiling, onWall, onFloor, onObject)`, retries `tuple(sorted(on))`. Both yield `("onCeiling","onWall")`; neither can produce the stored key | **CONFIRMED** (R-3 justified) |
| N1 — non-integer dimensions | Full corpus re-scan, 45,952 v3 records | **CONFIRMED EXACTLY** — 163 values across 162 records; distinct set `{0.5: 155, 0.2: 7, 2.5: 1}`; **161** records render a `0` under `:.0f` |
| S-1 — the `2.5` outlier | Same scan | **CONFIRMED** — the 162/161 gap is precisely the `2.5` record, which renders `2` rather than `0`. D0's distinction is exact and is what makes S-1 necessary |
| N2 — vowel-initial categories | Same scan | **CONFIRMED EXACTLY** — 3,643; top: airplane 696, air conditioner 228, air purifier 207, umbrella 148, apple 129 |
| MIF-2 — non-ASCII descriptions | Same scan | **CONFIRMED** — 10 |
| E-3 — the truncated record's true length | `SimpleTokenizer.encode` + 2 on `3e91980a22da4c0da975cc8ef776972c` | **CONFIRMED** — **89** tokens |
| Current golden string | `serialize_annotation(GOLDEN_ANNOTATION)` from `tests/test_resolve_stage1.py` | **CONFIRMED** — matches D0's stated incumbent; D0's proposed replacement differs by exactly E-2 + S-2 |

**Not re-verified by Master, accepted on D0's record:** the full-corpus token distributions (median/p99/max) and the 5,276/5,276 tokenizer-proxy validation. These are expensive to reproduce and are not load-bearing for the resolution — the resolution turns on the defect counts and the blocker, both of which Master reproduced directly.

**Provenance note.** Kyzen's acceptances of E-1, E-2, E-3 and of S-1/S-2 (「都接受」, 2026-08-21) occurred inside the D0 conversation. Their only record is this file. That is adequate provenance for a user ruling, and Master is treating them as decided. If any was misrecorded, say so before D10 executes.

---

### 12.2 Assessment against the review questions

1. **Sections 1–11 complete; Section 12 reserved.** Yes. All eleven sections substantively filled; Section 12 arrived with its template placeholders intact.
2. **Codex adversarial review completed.** Yes. Two rounds recorded with session ID, CLI version, model, reasoning effort, and workdir. Round 1 exhausted its budget and is honestly recorded as **not counting as a review**, while preserving the one finding it landed (C-0). Round 2 completed with 13 findings. Codex was briefed to attack, not confirm, and returned `BLOCKED BY UNKNOWN` / REJECT-as-written — evidence the brief worked.
3. **Findings independently verified.** Yes, C-0 through C-12 individually classified with the verification performed. Only two were rejected or reduced, both narrowly and with stated reasons. D0 **conceded** on C-0, C-1, C-4, C-6 and withdrew its own claims rather than defending them — including retracting the volume-redundancy argument it had originally used as support. That is the correct posture.
4. **Final template and user decisions incorporated.** Yes. E-1, E-2, E-3, S-1, S-2 are all folded into the ratified template in §11.3, with the resulting golden string derived and the effect measured in §11.4.
5. **S-1 and S-2 are formal recommendations.** Yes. Both appear in §11.1 as escalated-then-resolved items with measured justification, and both are reflected in the ratified template and in the §11.4 measurement of the accepted configuration — not appended as afterthoughts.
6. **Is the 45,952-record validation sufficient?** **Yes, for what it is used to support.** It establishes defect counts, the fractional-value vocabulary, token statistics, and zero token cost — all with full-corpus coverage and no sampling. It does **not** establish that the decided template retrieves better, and §11 says so explicitly under Remaining uncertainty. The evidence is correctly scoped to its claim.
7. **Is the n06 cache-validity finding treated as an execution BLOCKER?** Yes, correctly. §11.2 states four exit criteria (B-1…B-4) and — importantly — states that `D1_n06-reencode` **remains BLOCKED after this decision is accepted**. A decision that unblocked its own downstream task by fiat would have been wrong here; this one does not.
8. **Unresolved research / architecture decisions?** **None within D0-008's scope.** Every in-scope item has a disposition. Residual uncertainties are recorded as UNKNOWN rather than as pending choices, which is the correct treatment.

**One residual choice was delegated to Master and is settled below (§12.3).**

---

### 12.3 Master's rulings on delegated items

**R-3 — the unreachable `PLACEMENT_PHRASES[("onWall","onCeiling")]` entry: DELETE it. Do not "fix" it.**

D0 left this at Master's discretion, correctly. Master rules **delete**, on the following grounds:

- The fallback already emits grammatical prose — `"typically mounted on a ceiling or on a wall"` — for all 90 affected records. There is no defect in the output.
- "Fixing" it would change 90 serialized strings **that were not present in the configuration measured in §11.4**. The ratified template's measurements were computed without it. Adopting a change outside the measured configuration would weaken exactly the evidence this decision rests on.
- Deleting changes 0 strings and removes a dead branch that misleads a future reader.

Classification: dead-code removal, no behavioural change. It must **not** be bundled as a silent serialization tweak.

**Scope guard for the follow-up task.** No serialization change beyond E-1, E-2, S-1, and S-2 is authorised. If the implementing task believes another string change is warranted, it reports `MASTER-IMPACTING FINDING` and stops. It does not extend the template.

---

### 12.4 Accepted decision

**RATIFIED as recorded IMPLEMENTATION CHOICE (U-15)** — the Stage 1 text serialization design as specified in §11.3, incorporating user decisions E-1, E-2, E-3, S-1, S-2, with the classifications stated there.

Ratified template:

```
{description} {Category} made of {materials}, roughly {W} by {L} by {H} centimetres, {placement}.
```

`{Category}` = category with first character upper-cased. `W`/`L`/`H` render at one decimal with a trailing `.0` stripped, applied **uniformly at every magnitude** — no `< 1 cm` threshold branch.

**Explicitly carried into project state:**

- Centimetres is an **INFERENCE** as to MetaFind's intent (density plausibility of Figure 2's mass/size pairing, plus the unstated Holodeck schema match) and **OBSERVED DATA** as to this corpus (`dimension_unit: "cm"` in all 45,952 v3 records). The volume-arithmetic support is **withdrawn** — it is unit-invariant.
- The "volume is redundant" justification is **WITHDRAWN**. Omission of `synset` / `volume` / `mass` is ratified as an IMPLEMENTATION CHOICE with **unknown retrieval impact**.
- `resolve_stage1.py:111`'s `r = 0.52-0.62` is **UNVERIFIED** in this repository and must not be reported as MEASURED.
- Serialization order is **not** constrained by the paper. Any future claim that it is must cite new evidence.

**Date:** 2026-08-21

**Affected tasks:** `D1_n06-reencode`, `D2_stage1-prereq`, `D3_stage1-train`, `D4_gallery-index`, `D7_eval-table1`, and the new `D10_stage1-encoding-contract` proposed below.

---

### Required follow-up

Acceptance is conditional on these. None is executed by this decision.

| # | Follow-up | Owner | Gate |
|---|---|---|---|
| F-1 | **BLOCKER** — cache completion/validity, exit criteria B-1…B-4 (§11.2). Stale sidecars must be treated as incomplete; `load_protocol()` must bind to the executed serializer; `"metafind_v1_natural"` retired as a cache identity; pre-flight check in place | `D10` | **Blocks D1. Nothing downstream is trustworthy until this holds** |
| F-2 | Apply E-1, E-2, S-1, S-2 to `resolve_stage1.py`. No other serialization change (§12.3 scope guard) | `D10` | Blocks D1 |
| F-3 | Apply R-1, R-2, and R-3-as-delete. Documentation and dead code only | `D10` | Blocks D1 |
| F-4 | Re-annotate `3e91980a22da4c0da975cc8ef776972c` to English under the v3 prompt (E-3); re-verify its token count | `D10` | Blocks D1. Requires the annotation model to be available — if it is not, report and stop |
| F-5 | Update `L1-TEXT-SERIALIZATION` deliberately, plus the coverage Codex found missing: sub-centimetre dimensions, absent-article form, wall+ceiling combination, protocol/serializer mismatch, cache invalidation | `D10` | Blocks D1 |
| F-6 | Master updates `workflow/CONTEXT.md` §5 per **MIF-3** — record the ratified template, correct the centimetre classification, remove the withdrawn redundancy argument | `MASTER` | Done at acceptance |
| F-7 | **MIF-2** — 2 remaining non-truncated CJK records, 10 non-ASCII total. Route to `D0-003`'s scope. **Not** re-opened here | `MASTER` → `D0-003` | Does not block D1 |
| F-8 | Reproduce or retire `r = 0.52-0.62`. It supports current behaviour, so it cannot justify changing it | deferred | Does not block |
| F-9 | Latent defects with zero current corpus impact: `_cap()` word-boundary failure on space-free strings (0 records), doubled period (0 records), uncapped individual material strings (22 records > 20 chars) | deferred | Does not block |

**Milestone review note.** Per `workflow/WORKFLOW.md` §14, the Stage 1 milestone review must confirm that F-1's exit criteria still hold at the point Stage 1 is declared complete — not merely that they held when D10 was accepted.

---

### Master's assessment of the work

Two things distinguish this decision and are worth recording.

First, D0 **overturned its own Section 8** after Codex review rather than defending it, and said so in the document. The pre-review recommendation was retained for the audit trail and explicitly marked as superseded. The withdrawal of the volume-redundancy argument — which D0 had itself introduced as supporting evidence — is the clearest instance.

Second, the finding that actually matters was not in the decision's original scope at all. `is_complete()` ignoring text (MIF-4) is the one defect here capable of producing **confident wrong numbers with no error anywhere in the chain**: a resumed n06 would build a gallery from two text distributions, `gallery_index.py` fingerprints the checkpoint rather than the text, and Table 1 would come out self-consistent and wrong. It was surfaced by adversarial review, confirmed by direct code reading, escalated rather than absorbed, and correctly left blocking. That is the review process working as designed.

## 13. USER REVIEW BRIEF

Written by Master 2026-08-21 per `workflow/USER_REVIEW_TEMPLATE.md`.

**File:** `workflow/decisions/D0-008_USER_REVIEW.md`

Two revisions:

- **Revision 1** — first brief delivered.
- **Revision 2** — issued after the user returned `MODIFY`. Two corrections plus one binding classification:
  1. **Numbering collision fixed.** Revision 1 used `F1–F7` for findings and `F-1…F-9` for follow-ups in the same document, so "F-1" meant both "the paper is silent" and "the cache-validity blocker". Findings are now `FIND-n`; follow-ups are now `FU-n`.
  2. **Scope corrected.** §6 no longer gates this decision on the n06 execution blocker. `FIND-6` (`is_complete()` ignores text) was surfaced by Codex during *this* decision's review and is retained as a finding here, but its remedy and execution disposition belong to `D10`'s integration review.
  3. **Follow-ups enumerated individually** rather than as an unqualified range.

---

## 14. USER Final Decision

**User action: `MODIFY`, then `APPROVE`. 2026-08-21.**

### 14.1 What the user approved

The Stage 1 text serialization design as specified in §11.3, in principle and in full:

- E-1, E-2, E-3, S-1, S-2
- the final ratified template
- `width → length → height`
- centimetres
- R-3 — **delete** the unreachable `PLACEMENT_PHRASES[("onWall","onCeiling")]` entry
- omission of `synset` / `volume` / `mass`

### 14.2 User's binding modification

> *"synset / volume / mass 的省略請明確記為 IMPLEMENTATION CHOICE，其 retrieval impact 仍為 UNKNOWN。不得表述成 PAPER FACT 或已證明 redundant。"*

The omission of `synset`, `volume`, and `mass` is recorded as an **IMPLEMENTATION CHOICE** whose **retrieval impact is UNKNOWN**.

It must **not** be stated as a PAPER FACT.
It must **not** be described as proven redundant — the redundancy argument was withdrawn under Codex finding C-6.

This wording is binding on all downstream documents, comments, and the reproduction report.

### 14.3 Explicitly outside this approval

The n06 cache completion / validity gate (§11.2, B-1…B-4). It is an execution question owned by `D10_stage1-encoding-contract`. Approving this decision neither clears nor waives it.

### 14.4 Outstanding follow-ups

Carried at approval, enumerated in `D0-008_USER_REVIEW.md` §7:

| Owner | Items |
|---|---|
| `D10` | FU-2 (apply E-1/E-2/S-1/S-2) · FU-3 (apply R-1/R-2/R-3) · FU-4 (re-annotate per E-3) · FU-5 (golden string + coverage) |
| Master | FU-6 (CONTEXT.md §5 per MIF-3 — done at acceptance) · FU-7 (route MIF-2 to D0-003) |
| Deferred | FU-8 (reproduce or retire `r = 0.52-0.62`) · FU-9 (latent zero-impact defects) |
| **Not carried here** | FU-1 — owned by D10 |

### 14.5 Effect

Status → `USER_APPROVED`. **FINAL ACCEPTED.**

Master integrates into global project state and records the entry in `workflow/DECISION_LEDGER.md`.

Master's Section 12 recommendation is retained unchanged alongside this decision.

---

## D0 Operating Rules

D0:

- investigates research, evidence, architecture, and cross-task decisions;
- does not execute unrelated implementation work;
- does not silently update project-wide accepted state;
- does not mark its own recommendation as accepted;
- must return the result to Master;
- must use Codex adversarial review for formal decisions;
- must clearly distinguish paper fact, upstream-supported inference, implementation choice, runtime fact, and unresolved interpretation.

Master remains the final integration owner.

If D0 discovers something that changes project architecture, another task's contract, dependency order, or a shared assumption, report `MASTER-IMPACTING FINDING` with evidence and affected tasks. Do not make a new project-wide decision locally.
