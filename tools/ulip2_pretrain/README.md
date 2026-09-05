# ULIP-2 from scratch on MetaFind's corpus, through ULIP's OFFICIAL main.py (unmodified)

Live run directory: `/home/kyzen/metafind/metafind_data_attrs/ulip2_pretrain_run/` (main.py / models / utils are
symlinks into `/home/kyzen/upstream/ULIP`; `data/` holds symlinks to the upstream dataset module,
templates and labels, plus the files copied here). These copies are for provenance; edit the live ones.

- `run.py`: wrapper. Stubs `torch._six` / `pointnet2_ops` / `knn_cuda` / `wandb`, swaps `misc.fps` for the
  pure-torch FPS the vendor copy runs under, registers our two dataset classes, wraps
  `ULIP2_PointBERT_Colored` so OpenCLIP is frozen (PAPER §3.3 over the released builder, which only
  calls `.eval()`), runs the frozen encoders under `no_grad`, and adds per-block gradient checkpointing
  to Point-BERT (memory only). Then `runpy.run_path("main.py")`.
- `metafind_ulip_dataset.py`: `MetaFindObjaverse` (train tuple in ShapeNet's order) and
  `MetaFindObjaverseVal` (zero-shot pool with `lvis_metadata`). Text = attrs_v1 sentence
  (IMPLEMENTATION CHOICE; ULIP-2 used the per-view BLIP-2 caption), image = one random view of 12 on
  the n06 black background, pc = pc_norm(xyz) ++ rgb, 10,000 x 6, no augmentation.
- Split: train 36,554 / holdout 9,138 from `outputs/splits.json` (Kyzen 「8020拆分」).
