# Frozen evaluation protocol

Written **2026-09-01, before Stage 2 has ever run**, at Codex's Step 0. Nothing
below may be changed once a Stage 2 result has been seen. A change after that
point is a new protocol with a new name, recorded as such, and both are
reported.

Every number here was measured today from the artifacts on disk, not recalled.

---

## 0. What is being frozen and why

Two of this project's benchmarks are chosen by us rather than by the paper, so
both are exposed to the failure the ledger keeps naming: picking, after the
fact, whichever protocol makes the method look best. The defence is to fix the
manifest, the seeds, the thresholds and the output schema in advance, and to
treat the resulting resolution as an **eligibility test** -- a protocol that
cannot separate two methods is discarded for being blunt, never chosen for
being flattering.

`R` (literal fidelity against the paper's seven conditions) is NOT frozen here.
It is the paper's protocol, not ours, and it is reported per condition with its
own leakage audit.

---

## 1. E1 -- alternate-observation / sampling-disjoint sensitivity

`RENAMED 2026-09-01.` This was called "observation-disjoint" until Codex
pointed out that only one of its three arms withholds raw visual evidence. The
name now says what it measures.

### 1.1 Query manifest

```
pool         the object test split, splits.json -> object.test
n            9,138 uids
with alt cap 9,132 of 9,138 carry a description_candidate of rank >= 1
gallery      the full 45,692-asset corpus (train 36,554 + test 9,138)
positive     the query's own uid, exactly one per query
```

The 6 uids with no alternate caption are **dropped from the text arm only** and
retained in the image and pc arms. Reported as 9,132 / 9,138 / 9,138, never as
a single n.

### 1.2 The three arms, and what each is honestly disjoint from

```
arm    query observation                     disjoint from the gallery in
------ ------------------------------------- --------------------------------
text   description_candidate, lowest rank    the exact string, NOT the visual
       >= 1, i.e. not the canonical one      evidence: same generator, same
                                             renders. cos to canonical 0.85
image  one held-out view k; the gallery      the raw pixels. This is the only
       aggregates the OTHER views only       arm that withholds evidence
pc     a resample of the same complete mesh  the exact point set, NOT the
       at a different seed                   geometry. cos to canonical 0.944
```

Each arm's correlation to what the gallery holds is **printed in the results
table**, so "independent" is never read as "uncorrelated".

If a genuinely independent pc query is ever needed it must be a partial
view-derived cloud or a simulated scan. Another full-mesh resample will not do,
and this protocol does not claim it does.

### 1.3 Seeds and selection, fixed now

```
caption choice   the lowest-rank candidate with rank >= 1, deterministic,
                 no sampling
view index k     k = uid_seed(uid) mod n_views, deterministic per uid
pc resample seed 20260901, one draw per uid, farthest-point as in n03
```

`AMENDED 2026-09-02, before any E1 number exists.` The view rule was written
here as `crc32(uid) mod n_views` while the query pack the trained checkpoints
already use draws `uid_seed(uid) mod n_views` (`stage1.py`, `QueryPack.view_index`,
with `uid_seed` from `metafind/data/pointclouds.py`). Two rules for one draw
would have let a query see a view the gallery still averaged in. The property
that matters is "deterministic per uid"; both have it, so the rule the
checkpoints were trained with wins and the other is withdrawn.

### 1.4 Controls, all three run every time

```
identity oracle    q = g. Must return R@1 100.00 and zero ties, or the run is
                   void and the numbers are not reported
uid derangement    per direction. Median target rank must land near 22,846
zero-parameter     raw mean(T,I,P) gallery, raw query. The learned model has
                   to beat this or the learning is not doing the work
```

### 1.5 Metrics and schema

`MRR, R@1, R@5, R@10, NDCG@5, median rank`, macro-averaged over queries, one
row per query written to `.jsonl` so the independent recompute
(`tools/probes/independent_metric_recompute.py`, which imports nothing from
`metafind`) can check every cell.

```json
{"uid": "...", "arm": "text|image|pc", "condition": "...",
 "rank": {"<direction>": 1}, "cos_to_canonical": 0.85}
```

---

## 2. E2a -- programmatic scene-replacement validity

The one benchmark here that is genuinely self-match-free: the query is a
layout with a hole in it, and the answer is any asset that could fill the hole.
No annotation, no generator, no caption.

### 2.1 Slot construction

```
houses     scene_splits.json -> test_houses, 2,400 of 12,000, seed 20260816
slot       (house h, node k) with node k REMOVED from the graph
query       the remaining layout: nodes, positions, phys_edges, room_types,
            with k and every edge touching k deleted
gallery     the 1,439 ProcTHOR assets that have a point cloud
            (1,467 minus the 28 with no depth: transparent materials never
            enter Unity's depth prepass. 1.9%, and they cannot be positives)
```

### 2.2 The five constraints, thresholds fixed before any result

A gallery asset `c` is a **positive** for slot `(h, k)` iff all five hold.
Each is computed from ProcTHOR metadata alone and each is reported
**separately as well as jointly**, so a failure can be attributed.

```
C1 category    category(c) == category(node k)
               category comes from the node record; the asset_id -> category
               map is built over all 12,000 graphs and must be single-valued

C2 support     if k has a support parent p in phys_edges.support, then c must
               occur somewhere in the TRAIN houses as a child of some asset
               with category(p). If k has no parent, C2 passes.

C3 bbox fit    for each of the three dimensions of bbox_measured (metres):
                   0.75 <= dim(c) / dim(asset at k) <= 1.25
               bbox_measured, not bbox_reported: the two differ by up to
               5.6 mm and the measured one is what our renders and clouds
               were made from

C4 no overlap  the axis-aligned footprint of c, centred at position(k),
               must not intersect the footprint of any KEPT node in the same
               room by more than 5% of c's own footprint area.
               Axis-aligned because the node record carries no rotation;
               this is a DEVIATION from a true collision test and is
               reported as one

C5 room        c must occur in at least one TRAIN house in a room whose
               room_type equals room_types[room_id(k)]
```

C2 and C5 are computed over the **train** houses only, so a positive set is
never defined using the test houses it is evaluated on.

Slots whose positive set is empty, or is the removed asset alone, are
**excluded and counted** -- an excluded slot is reported, never silently
dropped.

### 2.3 Metrics

Multiple positives, so: `Success@K` (any positive in top K), `Recall@K`,
`mAP@K` for K in 1, 5, 10. Reported alongside the per-constraint pass rate of
the top-1 retrieval, which is what makes a failure inspectable rather than a
single number.

### 2.4 What E2a is not

It does not measure whether a human would like the replacement. It measures
whether the retrieved asset **could physically and semantically occupy the
slot**. That is the claim, and no stronger one may be made from it.

---

## 3. E2b -- graded relevance

Frozen only to this extent, because it is downstream of E2a:

```
qrels        E2a's five constraints give the binary grade. A graded level may
             add material and style agreement from gemma as a SECONDARY
             signal only
circularity  gemma text must NOT be the sole source of a qrel: the retrieval
             model consumes text of the same kind, and a benchmark scored by
             the thing it evaluates measures nothing
thresholds   fixed before comparison, and reported with a sensitivity band
             over +/- one threshold step
```

---

## 4. Candidate pools, stated once

```
E1     45,692 objects. chance R@1 = 0.00219%
E2a     1,439 ProcTHOR assets with a point cloud. chance R@1 = 0.0695%
R       whatever the paper's condition specifies, per condition
```

A pool may not be narrowed for a run without the narrowing appearing in the
reported table.

---

## 5. Provenance required on every reported number

```
git commit and whether the tree was dirty
checkpoint path and sha256
protocol name and this file's commit
seeds actually used
n queries, n gallery, n excluded and why
the three controls' results
```

A number without these is a debug observation, not a result.
