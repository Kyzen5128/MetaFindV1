# D0 Decision — n05 category: how the source-dataset ground truth enters annotation

> **Sections 1–5 prepared by Master** as framing and evidence. Master did **not** perform the research.
> **Sections 6–11 are D0's.** **12–14** are Master's / the user's.

---

## Decision ID

`D0-010_n05-category-source`

---

## Status

`OPEN` — raised by Kyzen 2026-08-21 after finding the defect. **BLOCKING a full re-annotation.**

---

## 1. Question

**When n05 annotates an Objaverse-LVIS asset, what role does the dataset's own category label play?**

```
FINDING UNDER EXAMINATION:  The LVIS ground-truth category exists on disk, is
                            never read, and disagrees with Qwen's category on
                            70.0% of the corpus by the "no word in common" floor.
DECISION REQUIRED:          Whether and how LVIS's category enters n05 — and
                            what that does to reproduction fidelity.
```

**And a second question the first one exposed:**

**Is the divergence a missing cross-check, a model-capacity gap, or both?** The paper annotates with **GPT-4o**; this reproduction uses **Qwen2.5-VL-7B-Instruct** (deviation `D-2`).

---

## 2. Why This Decision Exists

A full re-annotation costs **~19.6 GPU-hours** at n05's observed 39/min. **It must not be run twice.** Every question below changes what that run produces.

`D1_n06-reencode` was **stopped mid-run** on the user's order at `2026-08-21T14:15:48`, after 20,053 of 45,952 embeddings. Those embeddings encode categories that may change.

Full finding: `workflow/MIF_n05_category_vs_lvis.md`.

---

## 3. Decision Scope

### In Scope

- Whether MetaFind specifies where `category` comes from.
- Whether LVIS's category should be: prompt input · a cross-check · the value itself · recorded alongside but unused.
- Whether disagreement should trigger the repair loop, quarantine, or a flag.
- **Whether the other fields are equally suspect.** If the model misidentifies the object, `description`, `materials`, `width/length/height`, and `placement` are likely wrong too — this is not a `category`-only defect.
- Whether the GPT-4o → Qwen2.5-VL-7B substitution (`D-2`) is a material cause, and whether a stronger model is available or warranted.
- What agreement gate, if any, a corpus must pass before being called accepted.

### Explicit Non-Scope

- ❌ **Do not run n05, n06, or any GPU job.** This is an audit and a decision.
- ❌ **Do not modify `annotate.py`, `annotate_run.py`, the prompt, or the validator.**
- ❌ **Do not delete or mutate any annotation record.**
- ❌ Do not delete the 20,053 embeddings from the stopped run.
- ❌ Do not decide `D0-003` (the 3 legacy-v1 residuals).
- ❌ Do not re-open `D0-008` (the text template) or `D0-009`.

---

## 4. Authority / Evidence

Authority order: MetaFind main text → appendix/figures → upstream → implementation.

**Master-verified paper evidence:**

| Source | Says |
|---|---|
| `2methdology.tex:28` | "Each asset is rendered from 11 orthogonal viewpoints and **annotated using GPT-4o**. These annotations provide rich textual descriptions detailing attributes such as **object category**, size dimensions, materials, and placement constraints." |
| `neurips_2025.tex:100` | "each rendered from 11 views and **processed with GPT-4o** to generate structured text descriptions" |
| `2methdology.tex:24` (Figure 2 caption) | "passed through a **VLM to generate** structured, detailed annotations, capturing attributes such as **category**, dimensions, materials, and spatial placement constraints" |

**Master's reading, for D0 to test, not to inherit:** the paper appears to have the VLM *generate* the category rather than copy it from LVIS. If so, feeding LVIS's category into the prompt would itself be a deviation — which does **not** make it wrong, but does make it a recorded choice rather than a fix.

**What the paper does not say:** whether the LVIS label is used at all, whether outputs are cross-checked, and what happens on disagreement. **D0 must verify this silence, not assume it.**

---

## 5. Current Repository State

Master-verified 2026-08-21, read-only.

**The ground truth exists and is downloaded but never read:**

- `data/datasets/objaverse-lvis/objaverse_lvis_metadata.json` → `value_to_key_mapping`, **46,207** uid → category.
- `download.py:70` fetches it deliberately. `download.py:330` uses only `lvis.json`'s **keys** as the uid manifest.
- **`value_to_key_mapping` is read by nothing in the pipeline.**

**The model is never told the category:** `build_prompt(n_views)` (`annotate.py:366`) takes only `n_views`.

**The validator cannot catch a wrong category:** `validate_annotation()` (`annotate.py:510`) checks required fields, non-empty strings, English script, synset **shape only**, and numeric bounds. **It never checks semantic correctness.**

**Measured divergence, full corpus, 45,952 v3 records:**

```
identical after squashing      9,551   20.8%
one contains the other         3,588    7.8%
share >= 1 word                  656    1.4%
NOTHING in common             32,157   70.0%   <- floor for divergence
```

Sample: `jeep → air purifier` · `kimono → airplane` · `horned cow → airplane` · `sofa → bookshelf` · `chocolate cake → mosaic` · `vending machine → air purifier` · `microwave oven → fridge`.

**70.0% is a floor for divergence, not a proven error rate.** LVIS labels are themselves imperfect, and the prompt deliberately asks for a specific noun phrase. But no caveat explains `jeep → air purifier`.

**Model deviation `D-2`:** `annotate_run.py:71` — `MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # D-2: stands in for GPT-4o`. The paper says GPT-4o, twice.

**A recorded claim that must be re-examined:** `DL-003` and `CONTEXT.md` §5 call the corpus "legacy-v3 validated under `VALIDATOR_VERSION 2`". That validation covers schema, language, and bounds — **never** category correctness. The wording is easy to misread.

**Cost:** n05 observed 39/min → a full 45,952-asset re-annotation is **~19.6 GPU-hours**.

---

## 6. Options

*D0 fills this.* At minimum, evaluate:

**A.** Keep the VLM independent, as now. Record the divergence as a measured limitation.
**B.** Pass the LVIS category into the prompt as a hint. Changes the text distribution; a deviation from "VLM generates the category" if that reading holds.
**C.** Use LVIS's category as the `category` value; the VLM supplies the rest.
**D.** Generate independently, then cross-check against LVIS; disagreement triggers the repair loop / quarantine / a flag.
**E.** Record both, use one, and report the agreement rate as a corpus-quality metric.

Plus, orthogonally: **should the annotation model change?**

For each: dimensional and pipeline executability, paper fidelity, effect on the Stage 1 text distribution, effect on Table 1 comparability, and GPU cost.

---

## 7. Analysis
## 8. Recommended Decision
## 9. Codex Adversarial Review
## 10. Claude Verification of Codex Findings
## 11. Final Recommendation to Master

*D0 fills these. State FINDING and PROPOSED DECISION separately. Do not choose for the user where the paper is silent.*

---

## 12. Master Integration Recommendation
## 13. USER REVIEW BRIEF
## 14. USER Final Decision

*Master's / the user's.*

---

## D0 Operating Rules

D0 investigates and recommends. It does not decide, does not run GPU jobs, does not modify code, prompts, validators, or data, and does not fill §12–§14.

**The user is the final research authority.**
