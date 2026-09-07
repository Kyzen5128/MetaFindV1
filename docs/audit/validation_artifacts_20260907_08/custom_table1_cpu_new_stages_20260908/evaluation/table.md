# Custom seven-modality retrieval

IMPLEMENTATION CHOICE: fixed same-UID retrieval, not an exact reproduction of the authors' Table 1.
Queries: 2; gallery candidates: 6. Values are R@1 / R@5 (%).
Evaluation role: development comparison; independent test status is not certified.

## same_observation

| Method | T | I | P | T+I | T+P | I+P | T+I+P |
|---|---:|---:|---:|---:|---:|---:|---:|
| ulip2_available_mean | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| stage1 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 |
| stage2_layout_off | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 |

## different_observations

| Method | T | I | P | T+I | T+P | I+P | T+I+P |
|---|---:|---:|---:|---:|---:|---:|---:|
| ulip2_available_mean | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| stage1 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 50.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 |
| stage2_layout_off | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 50.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 | 0.00 / 100.00 |

Stage 2 is evaluated with layout=None and the Stage 1 parent gallery. This does not measure scene/layout benefit.
Different observations share the target mesh/UID; they do not establish statistical independence or semantic substitute relevance.
