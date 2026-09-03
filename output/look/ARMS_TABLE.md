# Stage 1 arms vs Table 1 (MetaFind w/o ESSGNN) -- fourteen cells

Official evaluator (float64 cosine, ties against the model). C: 4,569 dev_val queries vs the 4,569 dev_val gallery. D: same queries vs the 36,554 train gallery. Paper: 20% test queries on Objaverse-LVIS, gallery size unstated. Distance = mean over the 14 cells of |ln(ours/paper)|; a ranking device, not a protocol claim.

## D_dev_val_vs_train

| arm | dist | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|---|
| **paper R@1** | 0 | **13.8** | **11.7** | **75.1** | **17.2** | **44.5** | **45.8** | **51.7** |
| **paper R@5** | 0 | **23.1** | **19.2** | **78.0** | **21.8** | **71.3** | **73.1** | **76.5** |
| P4: P1 + ONE shared Fusion (n=36,554) R@1 | 0.54 | 12.0 | 25.0 | 52.3 | 58.5 | 94.4 | 65.8 | 98.0 |
| ↳ R@5 | | 30.8 | 48.7 | 76.0 | 81.8 | 99.0 | 85.9 | 99.8 |
| P1: attrs_v1 text + single_view query + prefusion L2 (n=36,554) R@1 | 0.59 | 11.6 | 29.7 | 66.6 | 67.5 | 95.6 | 77.8 | 98.1 |
| ↳ R@5 | | 31.3 | 56.6 | 88.6 | 89.5 | 99.6 | 94.3 | 99.9 |
| pilot10b: same_record, v2_cm text, 12-view mean, raw inputs (n=36,554) R@1 | 0.91 | 58.0 | 84.6 | 78.8 | 96.5 | 99.6 | 94.1 | 100.0 |
| ↳ R@5 | | 79.0 | 95.6 | 93.2 | 98.9 | 99.9 | 99.1 | 100.0 |

## C_dev_selection

| arm | dist | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|---|
| **paper R@1** | 0 | **13.8** | **11.7** | **75.1** | **17.2** | **44.5** | **45.8** | **51.7** |
| **paper R@5** | 0 | **23.1** | **19.2** | **78.0** | **21.8** | **71.3** | **73.1** | **76.5** |
| P4: P1 + ONE shared Fusion (n=4,569) R@1 | 0.77 | 34.1 | 49.0 | 75.1 | 80.6 | 98.6 | 84.8 | 99.6 |
| ↳ R@5 | | 65.3 | 74.6 | 91.3 | 94.5 | 100.0 | 96.1 | 100.0 |
| P1: attrs_v1 text + single_view query + prefusion L2 (n=4,569) R@1 | 0.83 | 34.7 | 56.9 | 86.1 | 87.6 | 99.0 | 92.7 | 99.7 |
| ↳ R@5 | | 67.7 | 82.1 | 97.9 | 97.6 | 100.0 | 99.2 | 100.0 |
| pilot10b: same_record, v2_cm text, 12-view mean, raw inputs (n=4,569) R@1 | 0.98 | 78.3 | 95.0 | 92.1 | 98.8 | 99.9 | 98.7 | 100.0 |
| ↳ R@5 | | 92.8 | 99.3 | 98.4 | 99.8 | 100.0 | 99.9 | 100.0 |

## ULIP row hypothesis (2026-09-03): does a category-only text query explain ULIP's 0.1?

Released ULIP-2, no training, gallery = PC embedding, query = raw mean of the available embeddings, 36,554 gallery, 4,569 queries (R@1):

| text arm | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| **paper, ULIP row** | **0.1** | **0.1** | **97.9** | **0.0** | **33.9** | **22.6** | **6.4** |
| category only | 3.8 | 58.4 | 100.0 | 38.6 | 98.8 | 98.6 | 97.0 |
| form-fill (attrs) | 4.7 | 58.4 | 100.0 | 39.7 | 98.7 | 98.6 | 96.6 |
| description only | 24.1 | 58.4 | 100.0 | 52.8 | 98.4 | 98.6 | 96.5 |
| full template | 24.5 | 58.4 | 100.0 | 52.3 | 98.6 | 98.6 | 96.6 |

Category-only moves the text cell from 24.5 to 3.8 (paper 0.1) and nothing else: T+PC and full stay 98.7 / 96.6 in every arm against 33.9 / 6.4, because a query pc identical to the gallery pc dominates any mean. The paper's shape -- adding text or image to pc HURTS -- needs the query's pc and image to sit far from the gallery's. INFERENCE; the paper says what neither row was fed.
