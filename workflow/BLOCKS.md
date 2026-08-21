# Blocks — structure, roles and rules

> **The operating protocol.** Who owns what, how work moves, how it is reviewed, how it is
> accepted. Master owns this file.
>
> Project state: `workflow/MASTER.md` · Method: `workflow/SKILLS.md` ·
> Orientation: `workflow/CONTEXT.md` · Decisions in force: `workflow/DECISION_LEDGER.md`

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

Owns the open question `Q-CATEGORY`.

**Holds the GPU.** Everything downstream is blocked on its annotation corpus.

### `ESSGNN` — scene chain

n07 scene graphs → n07b modalities → n08 semantic edges → n09b / n09c →
n11b index → n13 Stage 2 train → n14 equivariance probe →
**n15a/b/c → n16 compose → n17 judge (Table 2)**. Gates G6, G7.

Owns the open questions `Q-ESSGNN-AXIS`, `Q-NODETEXT`, `Q-TABLE2`, `Q-JUDGE-MODEL`,
`Q-N08-MODEL`, `Q-YAW-PLACEMENT`.

**USER constraint: code only, no GPU job of any kind** without a new
authorisation. Its input artifacts already exist, and n14 / n11b / n13 / n15a-c / n16 / n17
are all unimplemented, so there is real work that needs no GPU.

### `INTEGRATOR` — 接通

Owns no node. Owns the seams:

1. `splits.json` / `stage1_protocol.json` — ULIP2 writes, everything reads
2. gallery index format + encoder fingerprint — ULIP2 → both tables
3. `stage2_protocol.json`, `procthor_node_embeddings.npz` — ESSGNN → composition
4. the deviation registry in `docs/graph/graph_spec.yaml`
5. `Q-TOWER` — tower sharing; affects Stage 1 and Stage 2 feasibility
6. `Q-BUILDMODEL` — the Stage 1 model construction path

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

Each block keeps its own evidence under `workflow/blocks/<BLOCK>/evidence/`.

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
Work is owned by whole blocks; do not rebuild a swarm of tiny tickets out of skills. `SKILLS.md` §5.

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

## Finding is not Decision

Two different things. **Report them separately. Never merge them into one sentence.**

```
FINDING    what is true.  Carries evidence: file:line, paper section, a measurement and the
           population it was measured over. Anyone may produce one.
DECISION   what to do about it.  Material decisions belong to the USER.
```

"I found a bug" does **not** mean the finder gets to choose the fix. A finding can be correct
and its proposed remedy rejected; the finding still stands.

**Material — the USER decides:**

paper interpretation · architecture · dataset, annotation or preprocessing semantics ·
training protocol · evaluation protocol · deviations · admitting, dropping or regenerating a
corpus · shared artifact semantics · any rerun that changes scientific output · any assumption
that crosses blocks · model selection.

**Not material — the block owner decides alone:** local refactors inside an approved SPEC, test
scaffolding, logging, comments, and documentation that only makes a description accurate.

**When in doubt, treat it as material.**

## Escalation

A block that hits something affecting shared architecture, a cross-block dependency, an accepted
assumption, milestone feasibility, or a global runtime fact writes a **MASTER-IMPACTING FINDING**
into its `HANDOFF.md`:

```
FINDING     what is true, with evidence
IMPACT      which blocks, artifacts, stages
ASK         exactly what is needed
STATE       can this block safely continue meanwhile? yes / no, and why
```

**Report it. Do not act on it.** Do not rewrite global project state locally.

**Stop-safe rule.** If continuing would require inventing, assuming, or silently choosing
research-critical information, **stop** and report: what is known, what is unknown, the evidence,
why it matters, and what decision is needed.

**Engineering objection.** A block that believes an instruction is wrong says so once, with
evidence, before executing. If the USER reaffirms it, execute the full instruction and record the
objection. New *evidence* against an approved decision is a MASTER-IMPACTING FINDING; preference
is not.

## Status vocabulary

Execution and acceptance are two different facts. **Never merge them.**

```
execution    PLANNED · READY · ACTIVE · BLOCKED · REVIEW · COMPLETE · REWORK
acceptance   —  ·  AWAITING USER  ·  USER APPROVED  ·  USER REJECTED
```

> `execution COMPLETE` **≠** `accepted`.

A block can legitimately be execution `COMPLETE` and acceptance `AWAITING USER` at the same time.
**Done** requires both. `BLOCKED` must name its blocker.

The USER's four responses at acceptance: `ACCEPT` · `REJECT` · `INVESTIGATE MORE` ·
`SHOW MORE EVIDENCE`. A `MODIFY` is recorded as accepted **with the USER's wording as the
decision**, not Master's.

---

## Block state

| Block | State | Engineer | Reviewer | Next |
|---|---|---|---|---|
| `ULIP2` | **ACTIVE** | unassigned | unassigned | milestone 1 — the annotator bake-off |
| `ESSGNN` | **ON HOLD — USER decision 2026-08-22.** Not staffed yet | — | — | when the USER opens it: the Table 2 chain (n15a/b/c, n16, n17, n14), which needs no GPU and no decision |
| `INTEGRATOR` | **ON HOLD — USER decision 2026-08-22.** Not staffed yet | — | n/a | when the USER opens it: the deviation registry, where LVIS anchoring and n08's model both have no entry |
