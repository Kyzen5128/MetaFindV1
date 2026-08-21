# BLOCK — INTEGRATOR (接通)

**State:** `READY` · **Holder:** unassigned · **Opened:** 2026-08-22

---

## 1. Purpose

Own the seams between `ULIP2` and `ESSGNN`. Owns **no node** and runs **no training job**.

When a block hits something it cannot solve alone and that touches the other block, it
writes `HANDOFF.md`; the Integrator picks it up or routes it to Master.

## 2. Owned artifacts — the contracts both blocks depend on

| # | Artifact | Producer → consumer | Why it is a seam |
|---|---|---|---|
| 1 | `splits.json`, `stage1_protocol.json`, `eval_protocols.json` | ULIP2 n09 → everything | `stage1_protocol.json` carries `tower_sharing` and the sha256 of `stage1_hyperparameters.json`; `stage1.py:81-86` refuses to run if the hash does not dereference |
| 2 | gallery index + encoder fingerprint | ULIP2 n11/n12 → Table 1 and Table 2 | `gallery_index.py` writes `gallery_index_<ckpt-sha16>.npz` and cross-checks `stage1_ckpt.json`. **An index built by a drifted encoder produces self-consistent wrong numbers with no error** |
| 3 | `stage2_protocol.json`, `procthor_node_embeddings.npz`, `stage2_positive_map.json` | ESSGNN → composition | Stage 2's side of the same contract |
| 4 | `docs/graph/graph_spec.yaml` deviation registry | both | see §4 |

## 3. Owned decisions

| ID | Question | Why it is cross-block |
|---|---|---|
| `Q-TOWER` | `tower_sharing`: `shared_backbone_separate_fusion` / `fully_shared` / `fully_separate` | Written into `stage1_protocol.json` by ULIP2's n09, but determines whether ESSGNN's Stage 2 can freeze the gallery at all. `fully_shared` cannot reach Stage 2 |
| `Q-BUILDMODEL` | `build_model()` bypasses `Stage1RuntimeConfig`; one backbone; one shared `FusionConfig` object | Makes `fully_separate` unimplementable as written. Conditional on `Q-TOWER` |

## 4. Deviation registry — **two known gaps, both open**

`docs/graph/graph_spec.yaml` is the authoritative numbering. `tools/check_graph.py:373-383`
compares deviation **ids only, never the `what:` text** (`FU-A`), so a deviation whose
description has gone false **passes every gate silently**. That is exactly how `D-2` stayed
wrong until 2026-08-21.

| Gap | State |
|---|---|
| **LVIS category anchoring** — n05 v5 feeds the dataset's ground-truth label into the prompt. The paper has the VLM *generate* the category | Recorded as a DEVIATION in the block's evidence. **No `graph_spec.yaml` entry exists** |
| **n08's LLM** — `semantic_edges_run.py:77` was labelled `# D-2's stand-in`. After the deviation registry split `D-2` (annotation) from `D-8` (scene judging), n08 belongs to **neither id** | **No entry exists** |

Also unresolved and must not be restated as settled: **`D-2`'s stated reason "GPT-4o is
unavailable" was never verified.** OpenAI's deprecation page does not list base `gpt-4o`;
secondary sources disagree; nobody has exercised the API.

## 5. Open handoffs

_None yet._

## 6. Rules

- Reports `FINDING` and `DECISION` separately (`workflow/BLOCKS.md`).
- May recommend; may not decide a material question. Only the USER makes anything FINAL.
- Read-only on both blocks' production files. Writes only its own directory and, on Master's
  instruction, `graph_spec.yaml` and the documents that mirror it.
