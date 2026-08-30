# n15 on protocol C — what it should cost, and why that is an extrapolation

**Status: `INFERENCE`. Extrapolated, never measured.** No line in this file is a
measurement of n15. Every quantity is derived from a *different* program's timings
(`stage1.evaluate_dev_val` at arm 1's epoch boundaries) under assumptions listed in full
below. Written 2026-08-30, before n15's first execution, so the prediction is on record
before the measurement exists to be compared against it.

---

## First: the ~4.5 minute figure is protocol **B**, not protocol C

The standing estimate in `ULIP2 ENGINEER.md §11` is `≈ 4.5 min`, and it is for
**`B_full_gallery`** — gallery 45,692, query 9,138. It is quoted there under the heading
「協定 B 的編碼成本」.

Protocol C is a different and much smaller job:

```
                    query      gallery    reported
A_test_gallery      test       test          yes      9,138
B_full_gallery      test       full          yes     45,692
C_dev_selection     dev_val    dev_val       NO       4,569
```

`OBSERVED DATA` — `data/outputs/eval_protocols.json`, read 2026-08-30.

Protocol C's gallery is **10.0x smaller than B's** and its query pool **2.0x smaller**.
Carrying B's 4.5 min across to C would overstate C by roughly an order of magnitude.
Recorded here because the two figures are easy to confuse and one of them has already
been asked for under the other's name.

---

## The base measurement, and the derivation that stands on it

Preserved from `ULIP2 ENGINEER.md §11` as written, because the assumptions belong to
that derivation and must not be quietly re-derived:

```
協定 B 的編碼成本 —— 2026-08-30 已由 arm 1 的 epoch 邊界外推（見下），
  但仍未實測。外推值約 4–6 分鐘，比原本以為的小一個數量級。

  推導：arm 1 每個 epoch 邊界 27.1–27.7 秒，內含
    存 323 MB checkpoint + copyfile（NVMe，實測讀 0.1s / sha256 0.1s，寫入同量級 → 約 0.5s）
    dev_val 4,569 筆的完整評估（encode_pc 一次 + 8 次 fusion：1 gallery + 7 query 條件）
  → 約 26 秒 / 4,569 筆 = 5.7 ms/asset

  協定 B  gallery 45,692 筆 × (encode + 1 fusion) ≈ 45,692 × 4.5 ms ≈ 3.4 min
          query   9,138 筆 × (encode + 7 fusion) ≈ 9,138 × 5.7 ms ≈ 0.9 min
          streaming 計分：實測 0.7 s/condition/pass × 7 × 2 pass ≈ 10 s
          合計 ≈ 4.5 min
```

### The three unverified assumptions — reproduced verbatim, still unverified

```
  ⚠ 這是外推不是量測。三個未驗證的假設：
    (a) fusion 的成本與 encode 的比例，我用 8 次拆成 1+7 只是線性假設
    (b) 45,692 筆的資料讀取（點雲 123 KB × 45,692 ≈ 5.6 GB）沒有 IO 瓶頸
        —— 資料在 NVMe（/dev/nvme0n1p2）不是 SMR 碟，這一點已查證
    (c) n15 的 backbone 建構時 grad_checkpointing=False，與訓練不同，未量過差異
```

None of the three has been checked since. `(b)` is stated for B's 45,692 assets; on
protocol C the read is 4,569 x 123 KB ≈ 0.56 GB, so the IO assumption is *weaker load,
same untested claim* — a smaller read does not make the claim measured.

---

## Protocol C, derived from the same rates

```
gallery pass   4,569 × 4.5 ms   ≈ 21 s
query pass     4,569 × 5.7 ms   ≈ 26 s
streaming scoring                ≤ 10 s   (B's 0.7 s/condition/pass × 7 × 2, NOT scaled
                                           down for a 10x smaller gallery — deliberately
                                           left as an over-estimate)
                                 ------
compute subtotal                 ≈ 57 s
+ process startup                UNMEASURED — see below
```

`INFERENCE`. Roughly **one minute of compute**, plus a startup cost nobody has timed.

### Why C costs about twice a dev-val epoch boundary, not once

`OBSERVED IMPLEMENTATION`, `metafind/eval/run_retrieval.py`, `encode_pools`:

```python
gal, _      = embed(gallery_uids, [])
_, per_cond = embed(query_uids, list(QUERY_CONDITIONS))
```

Two `embed` calls, each building its own `Stage1Dataset` + `DataLoader` and each running
`backbone.encode_pc` over its own uid list. On protocol C `query_uids == gallery_uids`
— the same 4,569 assets — and n15 does **not** special-case that, so the point encoder
runs over the dev-val pool **twice**.

`stage1.evaluate_dev_val` encodes the pool **once** and derives the gallery embedding and
all seven query conditions from that single pass. Its ≈26 s is therefore a one-pass
number, and n15's protocol C is approximately two of them.

This is the single largest structural difference between the timing source and the thing
being timed. It is not a defect — n15 must work when the two pools differ, which is the
case for A and B — but it means the ≈26 s base cannot be carried over unchanged, and the
derivation above splits the two passes for exactly this reason.

### What "startup" contains, and why it is not estimated

`OBSERVED IMPLEMENTATION`, `run_retrieval.main`: before any asset is encoded it
constructs `ULIPBackbone`, calls `build_model`, and `load_stage1_checkpoint`. That pulls
in OpenCLIP `ViT-bigG-14` (10.16 GB of safetensors, per arm 1's
`initializers.open_clip.weight_size_bytes`), the ULIP-2 PointBERT initialiser (402 MB)
and the Stage 1 checkpoint (323 MB).

No run has ever timed this in isolation. Arm 1's log conflates it with preloading 31,985
assets into RAM, which n15 does not do. Left as `UNKNOWN` rather than guessed: on a
one-minute compute budget, an unmeasured startup could plausibly be the larger half, and
an invented number here would be the whole estimate.

---

## Running protocol C does not break the seal

`OBSERVED IMPLEMENTATION`, `run_retrieval.check_seal`:

```python
splits_used  = (protocol.get("query_split"), protocol.get("gallery_split"))
touches_test = "test" in splits_used or "full" in splits_used
```

For `C_dev_selection` that tuple is `("dev_val", "dev_val")`, so `touches_test` is
`False`:

* `--unseal` is **not** required, and passing it would change nothing for C;
* the guard's `SystemExit` is not reached;
* `core["sealed_split_read"]` is recorded as `False` in `table1.json`, so the artifact
  states on its face that this run did not spend the test split.

A and B are the opposite case and both need `--unseal` — they query `test`. Nothing in
this file authorises either.

---

## It is also n15's first real execution

`OBSERVED DATA`, 2026-08-30:

* `data/outputs/eval/` **does not exist**. `run_retrieval.main` creates it
  unconditionally (`out.mkdir(parents=True, exist_ok=True)`) before encoding anything, so
  its absence means `main` has never reached that line.
* `ULIP2 ENGINEER.md §11`: 「n15 從未真正執行過。所有測試都是純函式加合成資料」.

⚠ `run_progress.jsonl` holding zero `n15` rows is **not** independent evidence of this.
Until 2026-08-30 n15 wrote no `run_progress` record at all — that was the defect fixed
today — so an n15 run that had happened would also have left zero rows. The empty
`eval/` directory is the evidence; the empty log is a consequence of the bug.

So the first protocol-C run is simultaneously:

1. the first execution of the retrieval evaluator on real data;
2. the first opportunity to replace every number above with a measurement;
3. the first n15 run that will leave a `run_progress` and a `cost_ledger` record, which
   is what makes (2) readable afterwards — `cost_ledger` now carries `wallclock_s`,
   `queries_scored` and `gallery_comparisons` for exactly this comparison.

Being a first execution, it should be read as a smoke run of the evaluator as much as a
measurement of it. A first run that produces a plausible number has demonstrated that the
program runs; it has not demonstrated that the number is right. The negative control
(`--control shuffle_targets`, which must collapse to ~1/n_gallery) is the check that
separates those two, and it is a second run, not this one.

---

## Provenance

| Claim | Class | Source |
|---|---|---|
| protocol C is query dev_val / gallery dev_val / 4,569 | `OBSERVED DATA` | `data/outputs/eval_protocols.json` |
| ≈26 s for a 4,569-asset dev-val evaluation, 5.7 ms/asset | `OBSERVED DATA` (of `evaluate_dev_val`, not of n15) | arm 1 epoch boundaries, `ULIP2 ENGINEER.md §11` |
| the 1+7 fusion split | `INFERENCE`, assumption (a) | unverified |
| no IO bottleneck | `INFERENCE`, assumption (b) | unverified; NVMe placement is verified, the absence of a bottleneck is not |
| `grad_checkpointing=False` cost difference | `UNKNOWN`, assumption (c) | never measured |
| n15 encodes the pool twice on protocol C | `OBSERVED IMPLEMENTATION` | `metafind/eval/run_retrieval.py`, `encode_pools` |
| ≈1 min compute for protocol C | `INFERENCE` | this file; rests on all of the above |
| startup cost | `UNKNOWN` | never timed in isolation |
| C needs no `--unseal` | `OBSERVED IMPLEMENTATION` | `run_retrieval.check_seal` |
| n15 has never run | `OBSERVED DATA` | `data/outputs/eval/` absent |
