# MetaFindV1 Block Registry

> Ownership layer. **USER decision 2026-08-21/22: two technical blocks, one integrator.**
> Task-level detail stays in `workflow/INDEX.md` and each `workflow/tasks/<id>/TASK.md`.
> Master owns this file.

---

## Roles

| Role | Who | Authority |
|---|---|---|
| **USER** | Kyzen | Final research and project authority. The only role that makes anything FINAL ACCEPTED |
| **MASTER** | — | Global view, integration review, recommendations. Never decides a material question |
| **Block engineer** | one per block | Implements and self-verifies the whole block chain |
| **Block reviewer** | one per block | Independent, read-only, **synchronous**. Attacks the engineer's contract, not just its output |
| **Integrator (接通)** | one | Cross-block interfaces and cross-block research questions. Owns no node |
| **Codex** | — | Adversarial reviewer. Not authority. Every finding is independently verified |

---

## Blocks

### `ULIP2` — object chain

n02 download → n03 pointclouds → n04 renders → n05 annotate → n05b protocol →
n06 encode → n09 splits → n10 Stage 1 train → n11 / G4 / n12 gallery →
**n15 retrieval eval (Table 1)**. Gates G1–G4.

Owns tasks `D14`, `D15`, `D16`, `D1`, `D2`, `D3`, `D4`, `D7`.
Owns decisions `D0-003`, `D0-010`.

**Holds the GPU.** Everything downstream is blocked on its annotation corpus.

### `ESSGNN` — scene chain

n07 scene graphs → n07b modalities → n08 semantic edges → n09b / n09c →
n11b index → n13 Stage 2 train → n14 equivariance probe →
**n15a/b/c → n16 compose → n17 judge (Table 2)**. Gates G6, G7.

Owns tasks `D5`, `D6`, `D8`. Owns decisions `D0-004`, `D0-006`, `D0-007`.

**USER constraint 2026-08-21: code only, no GPU job of any kind** without a new
authorisation. Its input artifacts already exist, and n14 / n11b / n13 / n15a-c / n16 / n17
are all unimplemented, so there is real work that needs no GPU.

### `INTEGRATOR` — 接通

Owns no node. Owns the seams:

1. `splits.json` / `stage1_protocol.json` — ULIP2 writes, everything reads
2. gallery index format + encoder fingerprint — ULIP2 → both tables
3. `stage2_protocol.json`, `procthor_node_embeddings.npz` — ESSGNN → composition
4. the deviation registry in `docs/graph/graph_spec.yaml`
5. `D0-002` `tower_sharing` — affects Stage 1 and Stage 2 feasibility
6. `D0-005` `build_model()` construction path

---

## Where evaluation lives — USER-confirmed 2026-08-22

There is no separate evaluation block. Evaluation follows the chain that produces it:

- **Table 1** (object retrieval) → **ULIP2**
- **Table 2** (scene composition + LLM judge) → **ESSGNN**

---

## Communication rule — USER instruction, binding

> Everything goes through `HANDOFF.md`. In-conversation reporting does not count.

- Need Master? **Write `HANDOFF.md` first, then call Master.**
- Stuck, and it touches another block? Write `HANDOFF.md`. Master or the Integrator picks it up.
- The USER relays work between roles by handing over the `HANDOFF.md`.

```
workflow/blocks/ULIP2/{BLOCK.md, REVIEW.md, HANDOFF.md}
workflow/blocks/ESSGNN/{BLOCK.md, REVIEW.md, HANDOFF.md}
workflow/blocks/INTEGRATOR/{BLOCK.md, HANDOFF.md}
```

Existing `workflow/tasks/D*` directories are unchanged and become internal work items
of their block. Nothing was renamed; no decision file moved.

---

## Skills — where each one is used, by whom

Full policy: **`workflow/SKILLS.md`**. Summary only here.

**Skills are method tools, never authority.** Skill PASS ≠ scientific PASS. Tests PASS ≠
reproduction fidelity. Codex PASS ≠ Block PASS. Reviewer PASS ≠ USER acceptance.

| Layer | Skill | Who calls it |
|---|---|---|
| Block start | `grilling` + `domain-modeling` | Master, with Owner and Reviewer |
| Contract | SPEC, 15 sections (`blocks/SPEC_TEMPLATE.md`) | Owner |
| Build | `tdd` at agreed seams · `research` · `diagnosing-bugs` | Owner |
| Completion claim | `code-review`, **extended to 4 axes** | Owner |
| Independent, synchronous | `research` · `diagnosing-bugs` · `code-review` · `improve-codebase-architecture` | Reviewer |
| Milestone adversarial | Codex | Master requests |
| Acceptance | `grilling`, **one item at a time** | Master → USER |

**Six skills carry `disable-model-invocation: true` and Claude cannot call them** — verified
2026-08-22. `grill-with-docs` and `implement` are thin wrappers Claude reproduces directly;
`improve-codebase-architecture` must be **run by the USER**; `to-spec`, `handoff` and `grill-me`
are not used. Details and reasons: `SKILLS.md` §1 and §4.

### The four axes

`STANDARDS` · `SPEC` · `SOURCE / EVIDENCE` · `SCIENTIFIC / SEMANTIC` — **reported separately,
never merged into one PASS.** Axis 4 assumes the code runs and the tests pass, and asks how the
result could still be scientifically wrong.

### When none of this applies

Internal work items inside an approved SPEC, comments, formatting, read-only investigation, and
re-runs of accepted deterministic steps need **no** grill, spec, review, Codex or USER gate.
We removed fine-grained D-tasks deliberately; do not rebuild them out of skills. `SKILLS.md` §5.

---

## Formal acceptance flow

```
Block Plan
  → USER approves scope
  → Owner implementation + self-verification
       ↕  Reviewer synchronous independent verification
  → 4-axis completion review
  → Codex milestone adversarial review
  → Master integration
  → USER Acceptance Grill          (one material criterion per round)
  → USER FINAL ACCEPTED
```

No step is skipped at a Block milestone. Every step is skipped for an internal work item.

---

## Block state

| Block | State | Engineer | Reviewer | Next |
|---|---|---|---|---|
| `ULIP2` | **ACTIVE** | unassigned | unassigned | the two-model annotation comparison, then D14 Phase 2 |
| `ESSGNN` | **READY — code only** | unassigned | unassigned | `D0-006`, then rewrite n08 for the new LLM |
| `INTEGRATOR` | **READY** | unassigned | n/a | deviation registry: LVIS anchoring and n08's LLM both have no entry |
