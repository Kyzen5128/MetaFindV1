# n05 annotation defect — full diagnosis and fix plan

**Raised by Kyzen 2026-08-21. Diagnosed by Master the same day, read-only.**
`D1_n06-reencode` stopped at `14:15:48`, 20,053 / 45,952 npz. Nothing deleted.

---

## Root cause

**Qwen is asked to *identify* an object it cannot see well enough, so it falls back on high-frequency priors. The dataset's own ground-truth label was on disk the whole time and was never read.**

### Evidence 1 — the answers collapse onto common objects

| | distinct categories | top-20 share |
|---|---|---|
| **LVIS ground truth** | 1,156 | **7.1%** |
| **Qwen output** | 3,036 | **22.3%** |

**3× more concentrated than the truth.** Qwen's most frequent answers:

```
toy            1,542   3.4%      <- the prompt explicitly forbids this
bookshelf      1,085   2.4%
pillow           943   2.1%
airplane         696   1.5%
lamp             565   1.2%
teddy bear       548   1.2%
```

LVIS's most common class is `chair` at 453 (1.0%). **Qwen labels 1,085 assets "bookshelf".**

`toy` is the giveaway: `build_prompt` says *"Name the thing itself, not the class it belongs to: … 'toy dinosaur' not 'toy'"*. It is still the single most common answer. That is not a model following instructions on a legible image — that is a model hedging on an illegible one.

### Evidence 2 — more pixels do NOT help

```
best-view occupancy   n     LVIS agreement
   0 -  5%           68        27.9%
   5 - 10%           99        29.3%
  10 - 20%          263        24.0%
  20 - 30%          181        29.8%
  30 -100%          189        29.6%

correlation(occupancy, agreement) = +0.054
```

Effective object pixels vary ~**100×** across this sample. Agreement is **flat**.

**This kills the re-render hypothesis.** Master's first instinct was that 224×224 framing was the cause — objects occupy a mean 16.9% of frame, 31% of assets under 10%. The data says fixing that would not have fixed the annotations. **Re-rendering 46,045 assets would have been wasted GPU time.**

### Evidence 3 — the errors are confident, and they contaminate every field

The model does not hedge — it writes a rich, coherent description of the wrong object:

```
LVIS pinecone       -> "A dark brown hairbrush with a circular handle and bristles"
LVIS mug            -> "A cylindrical pillow with a striped pattern"
LVIS chocolate cake -> "a decorative wall clock with a dark face and gold accents"
LVIS truck          -> "a modern air conditioner unit with multiple vents"
LVIS saddle         -> "A detailed model of a firearm with a wooden stock"
```

So `description`, `materials`, `width/length/height`, and `placement` are **all** derived from a hallucinated object. **This is not a `category` bug. The whole record is contaminated.**

### Evidence 4 — nothing could have caught it

- `objaverse_lvis_metadata.json` → `value_to_key_mapping`: **46,207** uid → category. Downloaded by `download.py:70`. **Read by nothing.**
- `build_prompt(n_views)` (`annotate.py:366`) receives **only** `n_views`.
- `validate_annotation()` (`annotate.py:510`) checks fields present, non-empty, English, synset *shape*, numeric bounds. **Never semantic correctness.**

---

## How big is the real error?

**Not 70%.** That figure was string matching and it overstates.

| measure | value |
|---|---|
| LVIS label matches Qwen `category` | 29.0% |
| LVIS label word appears in Qwen `description` | 28.4% |
| either | 32.2% |
| **neither** | **67.8%** |

Part of the 67.8% is vocabulary, not error: `headset → headphones`, `chair → stool`, `dresser → chest`, `cockroach → insect`, `pastry → donut`, `keg → barrel`, `Bible → book`, `softball → baseball`. Several are cases where Qwen is *more* specific — which is what the prompt asked for.

**But the prior-collapse evidence is independent of any string metric**, and it shows a large genuine error component. The honest statement: **the true error rate is well below 67.8% and far above zero, and it cannot be pinned down without adjudication we have not done.**

---

## The fix

**Stop asking Qwen to identify the object. Give it the LVIS category and ask it to describe.**

| | |
|---|---|
| **What changes** | `build_prompt()` receives the known category. The model's job becomes description, materials, proportions, placement — anchored on a correct identity |
| **Why it works** | It removes the failure mode entirely. The model never has to guess what the thing is, so it cannot collapse onto `toy` / `bookshelf` / `pillow` |
| **Why it fixes more than `category`** | The descriptions are currently *of the wrong object*. Anchoring the identity fixes the whole record |
| **GPU cost** | **Zero extra.** Same 11 views, same model, same ~19.6 h |
| **Coverage** | 46,207 of 46,052 corpus uids have a label. Coverage must be verified per-uid before the run |

### Honest classification

The paper says the VLM generates the annotations, GPT-4o specifically (`2methdology.tex:28`, `neurips_2025.tex:100`). Feeding the dataset label in is a **DEVIATION**, and must be recorded as one — not presented as paper-faithful.

**It is still the right call.** We cannot run GPT-4o. A 7B model guessing identities produces a corpus that reproduces nothing. A deviation that is disclosed beats a corpus that is quietly wrong.

### What NOT to do

- ❌ **Do not re-render.** Evidence 2 says it would not fix this. Revisit only if a cheap A/B says otherwise.
- ❌ **Do not just re-run n05 unchanged.** 19.6 GPU-hours to reproduce the same failure.
- ❌ **Do not silently swap the category in** and leave the descriptions from the old run.

---

## Open questions for the user

1. **Prompt hint or hard value?** Give the label as context and let the model still name it — or write LVIS's label into `category` directly and let the model do the rest?
2. **Disagreement handling.** If the model is given "microwave oven" and still describes a lamp, is that a quarantine, a repair-loop trigger, or a recorded flag?
3. **The 155 uids** in the corpus with no LVIS label — how are they handled?
4. **Resolution.** Evidence says it is not the cause. Test it cheaply anyway before ruling it out for good?
5. **A validation gate.** Should a category-agreement measurement become mandatory before any future corpus is called accepted?

**Master has changed no code, no prompt, no validator, no data.**
