# n05 v5 — category-anchored annotation. FINAL DESIGN

**Decided by Master 2026-08-21 under Kyzen's explicit delegation ("你定案好").**
Every number below was measured by Master, read-only. Nothing has been implemented.

---

## The principle

**Category is the root of the record, not one field of thirteen.**

`build_prompt` currently says: *"Estimate its size from what kind of object it is, not from the picture."* So dimensions are **by design** derived from the category. Placement is almost entirely category-derived. Descriptions were observed describing the hallucinated object.

```
category ──┬─→ dimensions   (prompt says: from the category, not the image)
           ├─→ placement    (almost entirely)
           ├─→ description  (observed: describes whatever was hallucinated)
           ├─→ materials    (partly)
           └─→ synset       (entirely)
```

Wrong category is not one bad field. It is a wrong record.

**So: fix the identity first, then let everything hang off it.**

---

## Verified facts this design rests on

| Fact | Measurement |
|---|---|
| LVIS label coverage | **45,952 / 45,952 = 100.00%** |
| LVIS vocabulary | 1,156 classes, median class size 30 |
| LVIS labels that are vague catch-alls | **2.0%** (figurine 125, sculpture 115, motor vehicle 104, toy 80…) |
| Qwen collapses onto priors | top-20 share **22.3%** vs LVIS **7.1%** — 3× concentration. `toy` 1,542 (3.4%), which the prompt explicitly forbids |
| More pixels do **not** help | correlation(best-view occupancy, agreement) = **+0.054**; agreement flat ~28-30% across a 100× range of effective object pixels |
| Mesh proportions available | `raw_bbox_extents`, **46,041 / 46,052 = 100%** |
| **Axis convention** | **Y-up, verified.** Tall objects (n=651) mean normalised `[x .515, y .960, z .402]`; flat objects (n=497) `[x .865, y .318, z .738]`. **`height` = y** |

---

## Decision 1 — `category`: LVIS is the starting point, refinement downward only

**Not** written in verbatim. The model receives it as authoritative context and outputs a category that must be the LVIS label **or a strictly more specific term for the same object**.

- Lateral replacement is forbidden: `motor vehicle → coffee machine` is invalid.
- Downward refinement is required where the images support it: `toy → toy dinosaur`, `motor vehicle → pickup truck`.

**Why not write LVIS in directly.** It would lock in the 2.0% vague labels — 104 assets permanently "motor vehicle", 80 permanently "toy" — and the existing prompt rule ("'toy dinosaur' not 'toy'") exists for a reason: a generic label is a weak retrieval target.

**Classification: IMPLEMENTATION CHOICE / DEVIATION.** The paper has the VLM generate the annotations with GPT-4o (`2methdology.tex:28`, `neurips_2025.tex:100`). Supplying the dataset's own label is a departure and must be recorded as one.

---

## Decision 2 — `identity_confirmed`: flag on the first run, do not quarantine

New boolean field. The model states whether the images are consistent with the catalogued identity.

**First run: record it. Do not quarantine, do not trigger the repair loop, do not drop anything.**

**Why.** We have **no measurement** of LVIS's own error rate. LVIS labels are not clean — observed: `lamb-chop` on a plush panda, `cider` on a barrel (label names the contents, not the object). Setting a quarantine threshold before measuring that rate would be inventing a number.

Flag it, then read the distribution:

- `false` at ~2% → LVIS is a reliable anchor. Move on.
- `false` at ~30% → we have learned something that changes the whole plan.

Either way the run is not lost, because nothing was discarded on an unmeasured rule.

---

## Decision 3 — dimensions: give the exact proportions, ask only for scale

`raw_bbox_extents` is exact, 100% covered, and Y-up is verified. Feed the normalised triple with height identified.

The model then supplies **absolute scale for one dimension** — "how tall is a real apple" — and the rest follows.

**Three unknowns become one**, and the one that remains is exactly the one a category prior answers well.

`width` / `length` assignment across the two horizontal axes (x, z) stays the model's call from the images; the canonical facing of an Objaverse mesh is not determined.

---

## Prompt structure

```
This 3D asset is catalogued in Objaverse-LVIS as: "{lvis_category}"
Treat that identity as correct unless the images clearly contradict it.

Its true proportions, measured from the mesh, are
  height : width : depth  =  {h} : {a} : {b}      (largest = 1.0)
These are exact. Do not re-estimate them from the images.

IMPORTANT: the renders are SCALE-NORMALISED and carry no absolute size.

Return one JSON object:
  "category"            the catalogued identity, or a STRICTLY MORE SPECIFIC
                        term for the same object. Never an unrelated object.
  "identity_confirmed"  true if the images are consistent with the catalogued
                        identity; false if they clearly show something else
  "height"              CENTIMETRES. How tall is a real {lvis_category}?
                        The other two follow from the proportions above.
  "description"         The identity is given. Spend your attention on what
                        makes THIS instance different: colour, style, finish,
                        distinguishing detail.
  "materials", "onFloor"/"onWall"/"onCeiling"/"onObject", ...
```

**GPU cost: unchanged.** Same 11 views, same model, same ~19.6 h.

---

## Decision 4 — `synset` is mapped, not generated

`annotate.py` already admits "a well-formed but invented synset passes here". With `category` anchored to a 1,156-term vocabulary, synset becomes a **one-time lookup table of 1,156 entries**, not a per-asset guess across 45,952.

Removes an entire error class. Build the table once, review it once, apply it deterministically.

---

## What this design deliberately does NOT do

- ❌ **No re-render.** correlation +0.054 says it would not fix the annotations. Revisit only if a cheap A/B contradicts that.
- ❌ **No model change.** GPT-4o is unavailable; deviation `D-2` stands and stays recorded.
- ❌ **No quarantine policy invented ahead of data** (Decision 2).
- ❌ **No claim of paper fidelity.** This is a disclosed DEVIATION that produces a usable corpus, chosen over a paper-shaped one that is quietly wrong.

---

## Honest statement of what is being traded

Qwen's identification errors are being replaced by **LVIS's** identification errors.

That is a good trade — LVIS is human-curated, 100%-covering, and 3× less concentrated than Qwen's output — but it **is** a trade, and `identity_confirmed` is what makes it measurable instead of assumed.

---

## Execution order

```
1. n05 v5   re-annotate 45,952 under this design        ~19.6 GPU-h
2. read identity_confirmed distribution                  minutes
3. n06      re-encode                                    ~4 GPU-h
```

`D1_n06-reencode` stays stopped. The 20,053 embeddings from the halted run are invalidated by any annotation change and must not be reused — `is_complete()` binds to the serialized text, so they invalidate themselves.
