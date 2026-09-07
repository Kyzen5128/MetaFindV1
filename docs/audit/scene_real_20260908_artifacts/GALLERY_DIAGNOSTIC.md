# Actual Objaverse staging, G4 and promotion diagnostic

On 2026-09-08, the unchanged production CLIs completed in this isolated data root:

| Phase | Actual entry | Result | Seconds |
| --- | --- | --- | --- |
| n11 | `metafind.train.gallery_index stage1 --device cpu --stage1-ckpt-record .../stage1_best_ckpt.json` | 6 x 1280 staged | 23.64 |
| G4 | `metafind.gates.g4_gallery_freeze` with all defaults | PASS, all 6 sampled | 1.13 |
| n12 | `metafind.train.gallery_index promote` | Published the exact gate-verified index | 0.071 |

The registry is `data/outputs/gallery_index.json`, keyed by actual Stage1 checkpoint SHA `df4e81fee9da89bcd802ddadb2a4048d2411bf169ca48b2077727b691da1d532`. The index SHA is `7e2f94854aa533b20a8b04f580a13a8e318af4ee542e1c10fe12e7e3ef3ec90a`. The real consumer `load_promoted_index_for_checkpoint` verified the index and six-UID membership; the recorded gate SHA was separately checked. See `gallery_result.json`.

The admitted set is exactly the six actual Objaverse UIDs already fixed by `custom_table1_cpu_real_20260907/protocol/protocol.json`. Their old attrs_v1 text/image caches, point clouds and sidecars were copied byte-for-byte. Actual n11 restored the new Stage1 PointBERT and gallery fuser, re-encoded every point cloud, and used the original frozen-CLIP cached text and 12-view image mean. It used no `--limit`. No ProcTHOR asset was substituted for an Objaverse UID. `gallery_preparation.json` binds copied input bytes, parent sources and the six original GLB paths/hashes.

The new `data/outputs/splits.json` defines an **index-only diagnostic corpus** with `admitted_total=6` and the original seed 20260816. Its train/val lists are empty and its test list names all six assets; this partition is not used to train or select a checkpoint. The original parent Stage1 training split, checkpoint and configuration remain unchanged and are separately hash-bound. Two of these six UIDs were parent selection examples, so this diagnostic must not be treated as independent generalization evidence.

G4's existing implementation uses `min(1000,N)`, so its default CLI sampled six rows without changing a gate setting. This does **not** claim that the specification's literal 1000-sample check ran. G3 is still unimplemented; this diagnostic does **not** claim a G3-certified corpus or formal research promotion. No paper metric, scene quality, or completeness claim follows from this result. G4's declared collapse/id-permutation limitations also remain.

All model execution was CPU-only with CUDA/HIP hidden, nice 10, two BLAS threads and offline mode. Each production CLI ran in a separate sequential process; the production source hashes stayed unchanged during every phase. Every large-model process exited before downstream scene preparation was allowed to begin.

The first outer driver execution ended after the three successful production CLIs because its final read-only Python import lacked the repository on `sys.path`. The original failure is preserved in `gallery_execute.log`. Only this isolated driver was corrected, and `gallery_driver.py verify_promoted` subsequently exited zero; no model or producer was rerun. Per-phase started/result records contain the original driver hash, so the later driver edit is not retroactively attributed to those runs.
