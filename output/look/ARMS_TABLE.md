# Stage 1 retrain arms — official evaluator, seven Table 1 conditions, R@1 (%)

All rows: metafind.eval.run_retrieval (float64 cosine, ties against the model). C = 4,569 dev_val queries vs the 4,569 dev_val gallery; D = the same queries vs the 36,554 train gallery. Paper: 48K-asset Objaverse-LVIS, 20% test queries, gallery size unstated.

| arm | protocol | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|---|
| **paper, MetaFind w/o ESSGNN** | — | **13.8** | **11.7** | **75.1** | **17.2** | **44.5** | **45.8** | **51.7** |
| pilot10b: same_record, v2_cm text, 12-view mean query, raw inputs | C | 78.4 | 95.0 | 92.1 | 98.8 | 99.9 | 98.7 | 100.0 |
| pilot10b | D | 58.0 | 84.6 | 78.8 | 96.5 | 99.6 | 94.1 | 100.0 |
| **P1**: attrs_v1 text, single_view query image, prefusion L2 | C | 34.7 | 56.9 | 86.2 | 87.6 | 99.0 | 92.7 | 99.7 |
| P1 | D | 11.6 | 29.7 | 66.6 | 67.5 | 95.6 | 77.8 | 98.1 |
| P1 / paper | D | 0.84x | 2.54x | 0.89x | 3.93x | 2.15x | 1.70x | 1.90x |

P1 per-epoch dev_val (C shape): text climbs 11.4 -> 34.7, image 17.8 -> 56.9, pc 45.8 -> 86.1, full 82.1 -> 99.7 over 10 epochs. Text and pc are the gallery's own record on the query side, so every condition containing them keeps heading for Eq. 5's trivial solution; only the image differs. P5 (every modality a second observation) is queued for that reason.

## ULIP row hypothesis (Kyzen 2026-09-03 evening): does a category-only text query explain ULIP's 0.1?

Released ULIP-2, no training, gallery = PC embedding, query = raw mean of the available ULIP embeddings, 36,554 gallery, 4,569 queries:

| text arm | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| **paper, ULIP row** | **0.1** | **0.1** | **97.9** | **0.0** | **33.9** | **22.6** | **6.4** |
| category only | 3.8 | 58.4 | 100.0 | 38.6 | 98.8 | 98.6 | 97.0 |
| form-fill (attrs) | 4.7 | 58.4 | 100.0 | 39.7 | 98.7 | 98.6 | 96.6 |
| description only | 24.1 | 58.4 | 100.0 | 52.8 | 98.4 | 98.6 | 96.5 |
| full template | 24.5 | 58.4 | 100.0 | 52.3 | 98.6 | 98.6 | 96.6 |

Category-only moves the text cell from 24.5 to 3.8 (paper 0.1) and is the closest arm there. It moves T+PC and full by nothing: 98.7 / 96.6 in every arm against the paper's 33.9 / 6.4, because a query pc identical to the gallery pc dominates any mean. The paper's fingerprint -- adding text or image to pc HURTS -- cannot come from the text content; it needs the query's pc and image to sit far from the gallery's. INFERENCE; the paper does not say what text either row was fed.
