# Real scene execution evidence, September 8, 2026

This is an **inspection archive** of 102 files from
`output/validation/scene_cpu_real_20260908`, preserved byte for byte. It contains
execution records, observer scripts, failed attempts, semantic text records,
composition traces, controls, placement records, a rendered image and independent
geometry checks. It is not a standalone rerun bundle or formal Table 2 quality
evaluation.

The run uses the preceding diagnostic Stage 1 and Stage 2 checkpoints, a small
gallery and three explicit placement slots. The controls preserve their own
traces; equal selected assets do not establish that layout had no numerical
effect, nor does a numerical difference establish scene quality. See the actual
[verification result](verification.json) and [verification script](verify_scene.py)
for the scoped assertions.

| Evidence | Files |
|---|---|
| Gallery generation, freeze check and promotion | [Diagnostic description](GALLERY_DIAGNOSTIC.md), [outer driver log](gallery_execute.log), [driver](gallery_driver.py), [gallery result](gallery_result.json), [index](data/outputs/gallery_index.json), [staging record](data/outputs/gallery_index_staging.json), [G4 result](data/outputs/logs/gates/G4_gallery_freeze.yaml) |
| Earlier preparation failures | [Attempt 00 log](prepare_00.log), [attempt 00 missing semantics](bundle_00/missing_semantics.json), [attempt 01 log](prepare_01.log), [attempt 01 missing semantics](bundle_01/missing_semantics.json) |
| Semantic producer outputs | [First record](semantics_01/record.json), [first sentences](semantics_01/sentences.jsonl), [second record](semantics_02/record.json), [second sentences](semantics_02/sentences.jsonl) |
| Successful composition | [Frozen manifest](bundle_02/manifest.json), [model configuration](bundle_02/model/model.json), [validation composition](bundle_02/validation_composition.json), [actual composition](composition.json) |
| Controls | [Parallel composition](control_parallel/composition.json), [layout-off composition](control_layout_off/composition.json), each with its own frozen manifest and execution record |
| Placement and rendering | [Placement](placement_bundle/placement.json), [Blender log](blender_output/blender.log), [result](blender_output/result.json), [rendered image](blender_output/view_000.png) |
| Independent geometry | [Extracted observations](independent_geometry/extracted.json), [verified geometry](independent_geometry/verified_geometry.json), [trace audit](independent_geometry/trace_audit.json) |
| Verification history | [Earlier failed check](verification_attempt00.json), [earlier log](verification_attempt00.log), [final result](verification.json), [final log](verification.log) |

[MANIFEST.json](MANIFEST.json) lists the original source path, archive path,
byte count and SHA-256 of each archived file. It also lists 55 omitted local
dependencies and 77 external hash references found in the archived records.
Hashes of omitted local files below 100 MB were checked without copying those
files. Larger model files retain their producer-declared hashes, explicitly
distinguished from freshly calculated hashes. External hash references are
preserved metadata; their target files were not re-read during archiving.

All `.pt`, `.npz`, raw GLB, raw query inputs and `scene.blend` files are excluded.
The copied `data/outputs` subtree contains only the gallery index, staging record
and G4 YAML, not the corpus. Small placement annotation JSON files are retained.
The external model-cache symlink was neither copied nor traversed.

Original records, paths and unsuccessful attempts have not been rewritten.
Scripts retain their as-executed machine paths and source-directory assumptions.
They are provided for review, not for execution from this archive. Replaying the
run requires the matching external model weights, raw assets, datasets,
environment, explicit scene inputs and a new output directory. The archive does
not replace those dependencies or certify a formal scene list, judge protocol,
training convergence or research metric.

Verify archived bytes from the repository root without loading a model:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import json

root = Path("docs/audit/scene_real_20260908_artifacts")
manifest = json.loads((root / "MANIFEST.json").read_text())
for record in manifest["files"]:
    data = (root / record["archive"]).read_bytes()
    assert len(data) == record["size_bytes"], record["archive"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"], record["archive"]
print(f"Verified {len(manifest['files'])} archived files")
PY
```
