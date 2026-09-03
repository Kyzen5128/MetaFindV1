# Stage 1 arms vs Table 1 (MetaFind w/o ESSGNN) -- fourteen cells

Official evaluator (float64 cosine, ties against the model). C: 4,569 dev_val queries vs the 4,569 dev_val gallery. D: same queries vs the 36,554 train gallery. Paper: 20% test queries on Objaverse-LVIS, gallery size unstated.

Two scores per arm, never merged: **level** = mean |ln(ours/paper)| over the 14 cells; **shape** = the same after removing each table's overall level (R@1 and R@5 centred separately), i.e. only the relative pattern of the seven conditions. Plus the interaction ratios Table 1 is most discriminating on. Ranking devices, not protocol claims.

## D_dev_val_vs_train

| arm | level | shape | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|---|---|
| **paper R@1** | 0 | 0 | **13.8** | **11.7** | **75.1** | **17.2** | **44.5** | **45.8** | **51.7** |
| **paper R@5** | | | **23.1** | **19.2** | **78.0** | **21.8** | **71.3** | **73.1** | **76.5** |
| P6: P1 with a fresh query view per step (n=36,554) R@1 | 0.58 | 0.40 | 12.8 | 28.8 | 67.3 | 67.1 | 96.9 | 77.3 | 98.2 |
| ↳ R@5 | | | 32.0 | 55.6 | 88.0 | 89.3 | 99.8 | 93.9 | 99.8 |
| P3: P1 + 12 view tokens into Fusion (n=36,554) R@1 | 0.56 | 0.40 | 10.4 | 24.6 | 61.1 | 59.7 | 94.6 | 72.7 | 98.0 |
| ↳ R@5 | | | 28.9 | 50.0 | 84.9 | 84.6 | 99.3 | 91.4 | 99.8 |
| P5: desc_v1 text; query = alternate description + resampled pc + single view; prefusion L2 (n=36,554) R@1 | 0.66 | 0.40 | 14.3 | 49.4 | 88.5 | 59.5 | 92.8 | 94.0 | 95.5 |
| ↳ R@5 | | | 35.3 | 74.3 | 97.4 | 84.3 | 98.7 | 99.2 | 99.4 |
| P4: P1 + ONE shared Fusion (n=36,554) R@1 | 0.54 | 0.41 | 12.0 | 25.0 | 52.3 | 58.5 | 94.4 | 65.8 | 98.0 |
| ↳ R@5 | | | 30.8 | 48.7 | 76.0 | 81.8 | 99.0 | 85.9 | 99.8 |
| P1: attrs_v1 text + single_view query + prefusion L2 (n=36,554) R@1 | 0.59 | 0.41 | 11.6 | 29.7 | 66.6 | 67.5 | 95.6 | 77.8 | 98.1 |
| ↳ R@5 | | | 31.3 | 56.6 | 88.6 | 89.5 | 99.6 | 94.3 | 99.9 |
| P7: P1 with prefusion L2 OFF (n=36,554) R@1 | 0.61 | 0.44 | 9.5 | 36.1 | 69.1 | 64.6 | 95.6 | 81.7 | 98.5 |
| ↳ R@5 | | | 26.3 | 60.5 | 86.6 | 85.5 | 99.5 | 92.9 | 99.9 |
| pilot10b: same_record, v2_cm text, 12-view mean, raw inputs (n=36,554) R@1 | 0.91 | 0.57 | 58.0 | 84.6 | 78.8 | 96.5 | 99.6 | 94.1 | 100.0 |
| ↳ R@5 | | | 79.0 | 95.6 | 93.2 | 98.9 | 99.9 | 99.1 | 100.0 |

Interaction ratios (R@1):

| arm | T+PC/PC | I+PC/PC | Full/PC | T+I/max(T,I) | R@5/R@1 |
|---|---|---|---|---|---|
| **paper** | **0.59** | **0.61** | **0.69** | **1.25** | **1.40** |
| P6: P1 with a fresh query view per step | 1.44 | 1.15 | 1.46 | 2.33 | 1.25 |
| P3: P1 + 12 view tokens into Fusion | 1.55 | 1.19 | 1.60 | 2.42 | 1.28 |
| P5: desc_v1 text; query = alternate description + resampled pc + single view; prefusion L2 | 1.05 | 1.06 | 1.08 | 1.20 | 1.19 |
| P4: P1 + ONE shared Fusion | 1.81 | 1.26 | 1.87 | 2.34 | 1.29 |
| P1: attrs_v1 text + single_view query + prefusion L2 | 1.44 | 1.17 | 1.47 | 2.27 | 1.25 |
| P7: P1 with prefusion L2 OFF | 1.38 | 1.18 | 1.43 | 1.79 | 1.21 |
| pilot10b: same_record, v2_cm text, 12-view mean, raw inputs | 1.26 | 1.19 | 1.27 | 1.14 | 1.09 |

## C_dev_selection

| arm | level | shape | text | image | pc | text+image | text+pc | image+pc | full |
|---|---|---|---|---|---|---|---|---|---|
| **paper R@1** | 0 | 0 | **13.8** | **11.7** | **75.1** | **17.2** | **44.5** | **45.8** | **51.7** |
| **paper R@5** | | | **23.1** | **19.2** | **78.0** | **21.8** | **71.3** | **73.1** | **76.5** |
| P3: P1 + 12 view tokens into Fusion (n=4,569) R@1 | 0.79 | 0.44 | 32.1 | 50.8 | 82.6 | 83.6 | 98.8 | 89.3 | 99.6 |
| ↳ R@5 | | | 63.8 | 76.9 | 96.1 | 95.6 | 100.0 | 98.1 | 100.0 |
| P4: P1 + ONE shared Fusion (n=4,569) R@1 | 0.77 | 0.44 | 34.1 | 49.0 | 75.1 | 80.6 | 98.6 | 84.8 | 99.6 |
| ↳ R@5 | | | 65.3 | 74.6 | 91.3 | 94.5 | 100.0 | 96.1 | 100.0 |
| P7: P1 with prefusion L2 OFF (n=4,569) R@1 | 0.81 | 0.45 | 30.3 | 60.2 | 84.8 | 84.5 | 99.1 | 92.8 | 99.7 |
| ↳ R@5 | | | 61.4 | 81.1 | 95.4 | 95.8 | 100.0 | 98.4 | 100.0 |
| P1: attrs_v1 text + single_view query + prefusion L2 (n=4,569) R@1 | 0.83 | 0.46 | 34.7 | 56.9 | 86.1 | 87.6 | 99.0 | 92.7 | 99.7 |
| ↳ R@5 | | | 67.7 | 82.1 | 97.9 | 97.6 | 100.0 | 99.2 | 100.0 |
| P6: P1 with a fresh query view per step (n=4,569) R@1 | 0.83 | 0.46 | 35.8 | 56.0 | 86.6 | 87.0 | 99.5 | 92.0 | 99.7 |
| ↳ R@5 | | | 69.9 | 81.1 | 97.6 | 97.4 | 100.0 | 98.9 | 100.0 |
| P5: desc_v1 text; query = alternate description + resampled pc + single view; prefusion L2 (n=4,569) R@1 | 0.87 | 0.48 | 37.9 | 73.0 | 96.6 | 83.5 | 98.1 | 98.8 | 99.1 |
| ↳ R@5 | | | 69.5 | 90.1 | 99.8 | 96.1 | 100.0 | 100.0 | 100.0 |
| pilot10b: same_record, v2_cm text, 12-view mean, raw inputs (n=4,569) R@1 | 0.98 | 0.61 | 78.3 | 95.0 | 92.1 | 98.8 | 99.9 | 98.7 | 100.0 |
| ↳ R@5 | | | 92.8 | 99.3 | 98.4 | 99.8 | 100.0 | 99.9 | 100.0 |

Interaction ratios (R@1):

| arm | T+PC/PC | I+PC/PC | Full/PC | T+I/max(T,I) | R@5/R@1 |
|---|---|---|---|---|---|
| **paper** | **0.59** | **0.61** | **0.69** | **1.25** | **1.40** |
| P3: P1 + 12 view tokens into Fusion | 1.20 | 1.08 | 1.21 | 1.64 | 1.17 |
| P4: P1 + ONE shared Fusion | 1.31 | 1.13 | 1.33 | 1.64 | 1.19 |
| P7: P1 with prefusion L2 OFF | 1.17 | 1.09 | 1.18 | 1.40 | 1.15 |
| P1: attrs_v1 text + single_view query + prefusion L2 | 1.15 | 1.08 | 1.16 | 1.54 | 1.16 |
| P6: P1 with a fresh query view per step | 1.15 | 1.06 | 1.15 | 1.55 | 1.16 |
| P5: desc_v1 text; query = alternate description + resampled pc + single view; prefusion L2 | 1.02 | 1.02 | 1.03 | 1.14 | 1.12 |
| pilot10b: same_record, v2_cm text, 12-view mean, raw inputs | 1.09 | 1.07 | 1.09 | 1.04 | 1.04 |

## ULIP row hypothesis (2026-09-03): does a category-only text query explain ULIP's 0.1?

Released ULIP-2, no training, gallery = PC embedding, query = raw mean of the available embeddings, 36,554 gallery, 4,569 queries (R@1):

| text arm | text | image | pc | T+I | T+PC | I+PC | full |
|---|---|---|---|---|---|---|---|
| **paper, ULIP row** | **0.1** | **0.1** | **97.9** | **0.0** | **33.9** | **22.6** | **6.4** |
| category only | 3.8 | 58.4 | 100.0 | 38.6 | 98.8 | 98.6 | 97.0 |
| form-fill (attrs) | 4.7 | 58.4 | 100.0 | 39.7 | 98.7 | 98.6 | 96.6 |
| description only | 24.1 | 58.4 | 100.0 | 52.8 | 98.4 | 98.6 | 96.5 |
| full template | 24.5 | 58.4 | 100.0 | 52.3 | 98.6 | 98.6 | 96.6 |

Category-only moves the text cell from 24.5 to 3.8 (paper 0.1) and nothing else: T+PC and full stay 98.7 / 96.6 in every arm against 33.9 / 6.4. Per-modality L2 before the mean does not change that either (T+PC 99.3, full 98.0, every text arm): with q = (p + t)/|p + t| and the gallery's own p, the own score (1 + p.t) exceeds every other (p.p_j + t.p_j) unless t prefers asset j over the own asset by MORE than the pc margin 1 - p.p_j, so an uninformative text cannot flip the ranking, only lower every score together. The paper's shape needs the query's pc (or image) to sit far from the gallery's own, or a text that is systematically anti-informative. INFERENCE; the paper says what neither row was fed. No ln-ratio score for this row (paper has a 0.0 cell); read the table.
