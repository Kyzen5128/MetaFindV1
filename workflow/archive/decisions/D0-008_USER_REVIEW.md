# USER REVIEW BRIEF

**Decision ID:** `D0-008_stage1-text-template`
**Master Recommendation:** `ACCEPT WITH FOLLOW-UP`
**Migration case** — `WORKFLOW.md` §19. Recorded pre-gate on 2026-08-21, now reclassified as a recommendation. Evidence, Codex review, and verification were **not** re-run.

**Revision 2 — 2026-08-21.** User returned `MODIFY`. Two corrections applied, plus one classification the user made binding.

**OUTCOME — 2026-08-21: `APPROVE`. Status `USER_APPROVED`. FINAL ACCEPTED.**
Recorded in the decision file Section 14 and in `workflow/DECISION_LEDGER.md`.

> **Numbering fix.** Revision 1 used `F1–F7` for findings and `F-1…F-9` for follow-ups in the same document. Those collided — "F-1" meant both "the paper is silent" and "the cache-validity blocker". Findings are now **`FIND-n`**; follow-ups are now **`FU-n`**. No other renumbering.

---

## 1. What was found

Findings only. Remedies are in §5.

| # | Finding |
|---|---|
| FIND-1 | The paper specifies **no** text serialization format anywhere. |
| FIND-2 | `stage1_encoding_protocol.json` records a **different template** from the one the code actually runs — 5 differences, including the first two dimension fields being transposed. The code is what n06 uses; the artifact field is written but never read back. |
| FIND-3 | **161 records** render a real dimension as `0 centimetres` (a 0.5 cm coin becomes "roughly 4 by 4 by 0"). |
| FIND-4 | **3,643 records** (7.93%) emit `"A airplane"`-style ungrammatical articles. |
| FIND-5 | **1 record** exceeds CLIP's 77-token limit (true length 89). n06 records the overflow **and encodes it anyway**. |
| FIND-6 | `is_complete()` compares nothing about the text. A resumed n06 would skip the 5,276 old metre-based embeddings as "complete" and encode the rest in centimetres — **no error, no warning, same label on both halves.** `gallery_index.py` fingerprints the checkpoint, not the text, so Table 1 would come out self-consistent and wrong. **Surfaced by Codex during this decision's review; its remedy and execution disposition belong to `D10`'s integration review, not to this decision.** |
| FIND-7 | **Four in-code justifications do not describe the code**: the field-order claim is false; `MAX_PLACEMENT` is defined and never used; one `PLACEMENT_PHRASES` entry is unreachable; the "volume is redundant" argument was withdrawn under review. |

---

## 2. Evidence / provenance

| Finding | Source | Location | Master verified? |
|---|---|---|---|
| FIND-1 | paper | all 5 `.tex` files + Figure 2, searched for template/format/unit terms | **on D0's report** |
| FIND-2 | data + code | `stage1_encoding_protocol.json` vs `resolve_stage1.py:96-100`; `encode_text_image.py:194`, `:233` | **direct** |
| FIND-3 | data | full corpus scan, 45,952 v3 records | **direct** — 163 non-integer values, `{0.5:155, 0.2:7, 2.5:1}`, 161 render `0` |
| FIND-4 | data | same scan | **direct** — 3,643; airplane 696, air conditioner 228, umbrella 148 |
| FIND-5 | runtime | `SimpleTokenizer.encode` on `3e91980a...`; `encode_text_image.py:198-202` | **direct** — 89 tokens |
| FIND-6 | code | `encode_text_image.py:73-83`, `:178-179`, `:86-108` | **direct** — read the function |
| FIND-7 | paper + code | `2methdology.tex:28` and caption `:24` (order = category → **dimensions** → materials; code emits materials first); `resolve_stage1.py:162`; `placement_phrase()` key construction | **direct** |

**Accepted on D0's report, not re-verified by Master:** the full-corpus token distributions (median 49 / p99 62), and the tokenizer proxy validated at 5,276/5,276 against n06's own counter. Neither is load-bearing for this decision.

---

## 3. Claude ↔ Codex disagreement

Codex's verdict was **`BLOCKED BY UNKNOWN` — reject Option B as written, do not launch n06.** D0 conceded on most points rather than defending.

**Two residual disagreements, both narrow:**

| | Codex | D0 / Master | Disposition |
|---|---|---|---|
| C-0 | `30×30×40=36000` is unit-invariant, so it cannot establish centimetres | Conceded — argument **withdrawn**. But centimetres still holds on density plausibility **and decisively on OBSERVED DATA**: all 45,952 records store `dimension_unit: "cm"` | Codex right on the argument; conclusion survives on different evidence |
| C-5 | `.1f` rounding is unproven; Python renders `0.25` as `0.2` | Real Python behaviour, but `0.25` **does not occur** in this corpus | Rejected for this corpus; retained as a caveat for future annotation batches |

**Worth stating plainly:** on the operative question — *do not run n06 yet* — Codex, D0, and Master all converged. That convergence is **not** independent confirmation; all three were given the same brief. It is only notable because the final position moved toward Codex's, not D0's original §8.

---

## 4. Verified conclusion

```
CONFIRMED:   F1–F7. Codex findings C-1, C-2, C-4, C-6, C-7, C-8, C-11, C-12
             (D0 conceded C-0, C-1, C-4, C-6 and withdrew its own claims)
PARTIAL:     C-3 (methodology objection rejected after re-measurement;
                  substantive objection confirmed — one embedding is knowingly degraded)
             C-5, C-9, C-10
REJECTED:    C-0's implication that centimetres collapses; C-5's 0.25 case
UNVERIFIED:  r = 0.52-0.62 at resolve_stage1.py:111 — no script in this repo reproduces it
```

Codex review **ran**. Round 1 exhausted its budget and was honestly excluded from the count; round 2 completed with 13 findings.

---

## 5. Proposed / implemented decisions

| # | Decision | Authority | Classification |
|---|---|---|---|
| 1 | Template form, field set, field order | Master proposes ratify | **IMPLEMENTATION CHOICE** (field *set* is PAPER FACT; concatenating it is a choice) |
| 2 | Unit = centimetres | Master proposes ratify | **INFERENCE** as to the paper's intent · **OBSERVED DATA** as to our corpus |
| 3 | `width, length, height` ordering | Master proposes ratify | **INFERENCE from Figure 2** |
| 4 | Dimension precision — stop rendering `0.5` as `0` | **already your decision (E-1, 2026-08-21)** | IMPLEMENTATION CHOICE, user-approved. *Not* a bug fix — Codex C-4 |
| 5 | Remove `"A "`, no a/an heuristic | **already your decision (E-2)** | IMPLEMENTATION CHOICE, user-approved |
| 6 | Formatter applies at **every** magnitude, no `<1 cm` threshold | **already your decision (S-1, 「都接受」)** | IMPLEMENTATION CHOICE, user-approved |
| 7 | Capitalise the category's first character | **already your decision (S-2, 「都接受」)** | IMPLEMENTATION CHOICE, user-approved |
| 8 | Re-annotate the one Chinese record to English | **already your decision (E-3)** | USER DECISION |
| 9 | Omit `synset` / `volume` / `mass` | **USER DECISION (MODIFY, 2026-08-21)** — omission upheld, classification made binding | **IMPLEMENTATION CHOICE.** Retrieval impact **UNKNOWN**. Must **not** be stated as a PAPER FACT, and must **not** be described as proven redundant — the redundancy argument was withdrawn under Codex C-6 |
| 10 | `PLACEMENT_PHRASES` — ratify, naming the invented expansions (`onWall`→"mounted", all-false→"no typical placement") | Master proposes ratify | IMPLEMENTATION CHOICE, **not** schema-preserving |
| 11 | R-3: **delete** the unreachable entry rather than fix it (fixing would change 90 unmeasured strings) | **MASTER RECOMMENDATION** | dead-code removal |

**Ratified template:**
```
{description} {Category} made of {materials}, roughly {W} by {L} by {H} centimetres, {placement}.
```
Measured effect: median tokens 49 → 48. Zero-dimension renderings 161 → **0**. Ungrammatical articles 3,643 → **0**. Over-limit records 1 → **0** after E-3.

> **Provenance caveat:** E-1, E-2, E-3, S-1, S-2 were decided by you inside the D0 conversation. Their only record is that file. If any was misrecorded, say so now.

---

## 6. Impact

**What this decision settles:** the Stage 1 text serialization design — the template, its field order, its unit, its precision, its article handling, its placement phrasing, and its three omissions.

**What it does not settle:** whether n06 may run. That is an **execution** question owned by `D10_stage1-encoding-contract` and resolved in D10's own integration review. This brief does not ask you to approve or waive any n06 execution gate.

- **Artifacts affected:** `stage1_encoding_protocol.json` (records a template the encoder does not use — FIND-2), and every future Stage 1 text embedding.
- **Tasks affected:** `D10` implements the ratified template. `D1_n06-reencode` and everything downstream of it (`D2` → `D3` → `D4` → `D7`) inherit whatever template is ratified here.
- **Downstream result:** every text-conditioned column of Table 1.

> **You should know:** `D10_stage1-encoding-contract` has **already executed** against this decision, before the gate existed. Approving is partly retrospective. D10 is a separate migration item with its own brief.

---

## 7. Remaining UNKNOWN / unresolved

- `r = 0.52-0.62` (`resolve_stage1.py:111`) — **UNVERIFIED** here. It supports current behaviour, so it cannot justify changing it, but it must not be reported as MEASURED.
- Whether the ratified template **retrieves better** — UNKNOWN. Zero token cost is not zero embedding impact (Codex C-7).
- Whether MetaFind's authors serialized at all, or fed the VLM description directly — UNKNOWN. Option D recorded, not refuted.
- Centimetres as the authors' *intent* — INFERENCE only. As *our corpus's* unit — certain.
- Retrieval impact of omitting `synset`/`volume`/`mass` — UNKNOWN now that redundancy is withdrawn.

### Remaining follow-ups if approved

Enumerated individually. Source: D0-008 decision file §12.4.

**Owned by `D10` — implementing what this decision ratifies:**

| ID | Follow-up |
|---|---|
| FU-2 | Apply E-1, E-2, S-1, S-2 to `resolve_stage1.py` — and nothing else |
| FU-3 | Apply R-1 (correct the false field-order comment), R-2 (`MAX_PLACEMENT`), R-3 (**delete** the unreachable entry) |
| FU-4 | Re-annotate `3e91980a22da4c0da975cc8ef776972c` to English per E-3, then re-verify its token count |
| FU-5 | Update the `L1-TEXT-SERIALIZATION` golden string, plus the coverage Codex found missing |

**Owned by Master:**

| ID | Follow-up |
|---|---|
| FU-6 | Update `CONTEXT.md` §5 per MIF-3 — record the ratified template, correct the centimetre classification, remove the withdrawn redundancy argument. Done at acceptance |
| FU-7 | Route MIF-2 — the 2 remaining non-truncated CJK records (10 non-ASCII total) — into `D0-003`'s scope. Does not block anything |

**Deferred, not blocking:**

| ID | Follow-up |
|---|---|
| FU-8 | Reproduce or retire `r = 0.52-0.62`. It supports current behaviour, so it cannot justify changing it |
| FU-9 | Latent defects with zero current corpus impact: `_cap()` word-boundary failure on a space-free string (0 records), doubled period (0 records), individual material strings uncapped (22 records > 20 chars) |

**Not carried by this decision:**

| ID | Disposition |
|---|---|
| FU-1 | The cache completion / validity blocker (FIND-6). Surfaced during this decision's Codex review, but it is an **n06 execution gate**, not a serialization-design question. **Owned by `D10` and resolved in D10's integration review.** It is not part of what you are approving here |

---

## 8. USER ACTION REQUIRED

**Do you ratify the Stage 1 text serialization design as specified in §5 — the template, `width → length → height`, centimetres, E-1/E-2/E-3/S-1/S-2, the R-3 delete ruling, and the omission of `synset`/`volume`/`mass` as an IMPLEMENTATION CHOICE with UNKNOWN retrieval impact?**

Outstanding if you approve: **FU-2, FU-3, FU-4, FU-5** (D10 implementation) · **FU-6, FU-7** (Master) · **FU-8, FU-9** (deferred).

Explicitly **not** part of this approval: **FU-1**, the n06 execution blocker — that belongs to D10's integration review.

- `APPROVE`
- `REJECT`
- `MODIFY`
- `INVESTIGATE MORE`
