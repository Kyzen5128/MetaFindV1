# MASTER-IMPACTING FINDING — n05 never checks its category against the source dataset

**Raised by:** Kyzen, 2026-08-21. **Verified by Master the same day.**
**Severity:** potentially invalidates the n05 corpus and everything encoded from it.
**Status:** OPEN. **`D1_n06-reencode` is running as this is written.**

---

## The finding

**Objaverse-LVIS ships a ground-truth category for every asset. It is downloaded, sits on disk, and nothing in the pipeline ever looks at it.**

| | Evidence |
|---|---|
| The ground truth exists | `data/datasets/objaverse-lvis/objaverse_lvis_metadata.json` → `value_to_key_mapping`, **46,207** uid → category entries (`"d4c9180a…": "Band Aid"`) |
| It is deliberately downloaded | `download.py:70` fetches **both** `lvis.json` and `objaverse_lvis_metadata.json` |
| Only the uid list is ever used | `download.py:330` reads `lvis.json` **keys** as the uid manifest. `value_to_key_mapping` is read by **nothing** |
| The model is never told the category | `build_prompt(n_views)` (`annotate.py:366`) takes **only** `n_views`. The known category is not passed |
| Nothing ever compares them | zero references to the category mapping outside `download.py` |
| The validator cannot catch it | `validate_annotation()` (`annotate.py:510`) checks required fields present · non-empty strings · English script · synset **shape only** · numeric bounds. **It never checks whether the category is correct.** |

---

## Measured divergence — full corpus, 45,952 v3 records

```
identical after squashing      9,551   20.8%    gasmask / gas mask
one contains the other         3,588    7.8%    cake / chocolate cake
share >= 1 word                  656    1.4%
NOTHING in common             32,157   70.0%
```

Random sample of the 32,157, `LVIS -> Qwen`:

```
jeep            -> air purifier      kimono          -> airplane
horned cow      -> airplane          sofa            -> bookshelf
chocolate cake  -> mosaic            dove            -> pillow
vending machine -> air purifier      microwave oven  -> fridge
roller skate    -> airplane          parking meter   -> screwdriver
garlic          -> snail             lamppost        -> showerhead
trash can       -> cylinder          helicopter      -> industrial machine
```

---

## What the number does and does not mean

**Does NOT mean "70% are Qwen errors."** Honest caveats:

- Objaverse-LVIS labels are themselves imperfect.
- The prompt deliberately asks for a **specific** noun phrase ("sofa" not "furniture"), so some divergence is by design — though LVIS labels are already specific, so this explains little.
- Some assets are legitimately describable more than one way (`martini -> wine glass`).

**Does mean:** 70.0% is a **floor for divergence**, and the sample contains unmistakable gross misidentifications that none of the above explains. **`jeep -> air purifier` is not a labelling philosophy difference.**

The structural fact stands regardless of how the 70% is apportioned: **nothing in this pipeline has ever compared the two, so no one knows which is right, and no gate would have noticed.**

---

## Why this matters now

`category` is serialized into the Stage 1 text template:

```
{description} {Category} made of {materials}, roughly {W} by {L} by {H} centimetres, {placement}.
```

It reaches **every text embedding**, and through them every text-conditioned column of Table 1.

**`D1_n06-reencode` is running now.** At the time of writing: **19,403** `.npz` produced of an expected 45,952.

---

## A claim that now reads dangerously

`DL-003` and `CONTEXT.md` §5 record the corpus as:

> **legacy-v3 corpus validated under `VALIDATOR_VERSION 2`**

That is true and was verified — **but "validated" means schema, language, and bounds. It has never meant the categories are correct.** The wording is easy to misread as semantic validation. Master wrote it; Master flags it.

---

## What is NOT claimed here

- Not claimed that the corpus must be discarded.
- Not claimed that LVIS is right and Qwen is wrong in any specific case.
- Not claimed that n05 must pass the category to the model — that is a research decision, not a defect fix.

---

## Decision required from the user

1. **Halt `D1_n06-reencode`, or let it finish?** It is baking these categories into embeddings now.
2. Should n05 **use** the LVIS category — as prompt input, as a cross-check, or as the value itself?
3. Should a **category-agreement audit** become a gate before any corpus is called accepted?
4. Does the accepted-legacy-v3 wording in `DL-003` / `CONTEXT.md` need qualifying?

**Master has adopted nothing and changed no code, no artifact, and no running process.**
