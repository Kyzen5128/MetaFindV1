# Validation evidence for inspection, September 7–8, 2026

This archive preserves 123 small evidence files from four isolated local runs,
byte for byte. It supports inspection of commands, inputs, observer scripts,
logs, hashes, parameter updates and emitted retrieval ranks. It is **not a
self-contained reproduction bundle**, a formal benchmark, or proof that the
paper's Table 1–3 results have been reproduced.

| Run | Evidence | Scope |
|---|---|---|
| First real custom evaluation | [Execution](custom_table1_cpu_real_20260907/execution.json), [protocol](custom_table1_cpu_real_20260907/protocol/protocol.json), [results](custom_table1_cpu_real_20260907/evaluation/results.json) | Two queries, six candidates; mean and Stage 1; 28 combinations. |
| Real Stage 1 training | [Preparation script](stage1_cpu_real_20260908/prepare.py), [execution](stage1_cpu_real_20260908/execution.json), [verification](stage1_cpu_real_20260908/verification.json) | Two training assets, two selection assets, one real optimizer step. |
| Real Stage 2 training | [Diagnostic description](stage2_cpu_real_20260908/README.md), [driver](stage2_cpu_real_20260908/driver.py), [optimizer observations](stage2_cpu_real_20260908/optimizer_observation.json), [restoration](stage2_cpu_real_20260908/restore_result.json) | One room, eight assets, historical relation sentences, one real optimizer step; layout-present restoration. |
| Custom evaluation of new checkpoints | [Execution](custom_table1_cpu_new_stages_20260908/execution.json), [results table](custom_table1_cpu_new_stages_20260908/evaluation/table.md), [rank checks](custom_table1_cpu_new_stages_20260908/verification.json) | Three methods, two observations, seven modality combinations; 42 combinations and 84 query records. One query overlaps Stage 1 selection. |

[MANIFEST.json](MANIFEST.json) maps each original repository-relative source
path to its archive path, byte count and SHA-256. Embedded provenance and local
paths have not been rewritten. Original checks records elsewhere in
`docs/audit/` retain their original identities and source paths.

The archived scripts retain their as-executed local path assumptions. Moving
them into this directory does not make them runnable here: replay requires an
explicit new output directory, the referenced external datasets, pretrained
models, checkpoint files and compatible environment. Do not execute these
copies expecting them to operate on the archive. The first custom run has no
persisted driver script in its source directory; its execution record retains
the command and environment. Earlier Stage 2 phases lack the later per-phase
driver-hash instrumentation, as documented in that run's README.

Raw corpus files, images, generated query clouds, model files, and the Stage 1
and Stage 2 checkpoints are excluded. Their identities remain in the copied
records. The only included NumPy archive is the 38,198-byte
[before-save query matrix](stage2_cpu_real_20260908/before_save_queries.npz),
containing eight float32 query vectors of width 1,280. It is an observation
artifact, not model weights. The separately generated restored matrix was not
persisted; the restoration script and result record document that comparison.

To verify all archived file hashes from the repository root without loading
models or depending on the original source directories:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import json

root = Path("docs/audit/validation_artifacts_20260907_08")
manifest = json.loads((root / "MANIFEST.json").read_text())
for record in manifest["files"]:
    data = (root / record["archive"]).read_bytes()
    assert len(data) == record["size_bytes"], record["archive"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"], record["archive"]
print(f"Verified {len(manifest['files'])} archived files")
PY
```
