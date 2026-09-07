# Isolated real Stage2 CPU diagnostic

This is an execution and restoration smoke, not a canonical research run or a benchmark.

- One untouched real room graph: test_00757, eight distinct assets, 28 semantic edges.
- Eight ProcTHOR gallery assets, all 88 original image PNG bytes copied.
- Node/gallery text is canonical fitted metadata. Relation descriptions explicitly reuse historical category text and 16 exact existing generated sentences. All semantic vectors are newly encoded by the real CLIP tower. Historical LLM weight bytes remain UNKNOWN.
- Parent is the separately completed real Stage1 CPU smoke; its attrs_v1/12-view training configuration is preserved. Transfer to ProcTHOR v3_fit/11 views is an explicit diagnostic distribution change.
- Stage2 recipe is workflow/stage2_hyperparameters_ft_lr5e-5.json with batch size 64→8 only. One epoch, one actual update. Warmup remains active, so the sole update uses 1e-6, not the nominal 5e-5 peak. Seed 20260816 selects layout present on that update.
- No legacy flags, GPU, LLM execution, source-corpus write or queue change. Each large-model phase runs in a fresh process under nice 10, two CPU threads and offline mode.
- Driver train calls production Stage2.main. Observers call the original AdamW.step and checkpoint-payload function unchanged, record actual finite gradients and changed parameter hashes, and capture inference outputs. Restore uses actual parent load, strict child identity and query overlay; all saved tensors and query outputs must match exactly.

Phases: bind --parent-record PATH; semantics; gallery; preflight; train; restore. Do not run phases concurrently. Raw logs and result JSONs are the evidence; existence of this README does not mean a phase passed.

## Observed results, 2026-09-08

All six phases exited successfully. Real semantic encoding took 28.99 s; gallery encoding took 345.09 s; strict preflight took 2.17 s; actual Stage2 training took 21.93 s; separate-process restoration took 22.97 s. The 8-asset gallery excluded no assets; all 28 graph edges resolved to one of the 16 freshly encoded sentences. The maximum absolute difference between n08 node vectors and normalized n11b text vectors was 3.43e-7.

The sole real AdamW update used 1e-6 in both parameter groups. Of 109 optimizer tensors, 106 received finite gradients and 105 changed. Each required component had a finite, nonzero gradient and actual updates: query fusion 26 changed tensors, ESSGNN 78, and lambda 1. The real backbone and gallery were frozen and absent from the optimizer. Initial lambda was 2.3415285110473634. See `optimizer_observation.json`, `preflight_result.json`, and `train.log`.

The new checkpoint is `data/outputs/checkpoints/stage2_full.pt`, SHA-256 `c4347b8c09ec28f2d932749498be04dde04cf3b8769f953bf4f51d4461e0e5ad`; its record is `data/outputs/variant_ckpts.json`, variant `full`. Strict restoration checked parent, input and gallery identities, all 109 saved tensors matched exactly, and all 8 x 1280 **layout-present** query values matched their pre-save values bitwise. See `restore_result.json` and `restore.log`. The shared overlay helper's log mentions layout being unused at evaluation; this diagnostic explicitly calls the training query encoder with `drop_layout=False`, so that wording does not describe the restoration parity check.

Training and restoration phase records bind their actual driver and production source hashes and two-thread/offline environment. Earlier bind, semantic and gallery phases ran before this additional execution-record instrumentation; their logs and encoded source identities are retained, but no retrospective driver hash claim is made for them. All model processes exited before another evaluator was allowed to start.

These observations prove this small execution and restoration path, not paper metric reproduction, generalization, canonical current-relation equivalence, or full training convergence. The upstream test room is used as a training input solely for this diagnostic; there is no evaluation split or reported retrieval score here.
