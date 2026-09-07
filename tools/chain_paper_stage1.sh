#!/bin/bash
# DL-103 paper-first line, after R2 (annotation v10) has finished:
#   exclusions -> n05b protocol -> n06 encode -> n09 splits -> Stage 1 (same_record, same_mean)
#   -> n11 -> G4 -> n12 -> Table 1 row 1 (holdout) -> n11b -> Stage 2 (9,600 houses) -> row 2 -> tables
# Stop and restart waiting shells when changing this file; Bash buffers script input.
set -Eeuo pipefail
trap 'rc=$?; echo "PIPELINE FAILED (line $LINENO, exit $rc)" >&2; exit "$rc"' ERR
cd "${METAFIND_REPO:-/home/kyzen/MetaFindV1}"
PY=${METAFIND_PYTHON:-/home/kyzen/miniconda3/envs/MetaFind/bin/python}
R=${METAFIND_DATA:-/home/kyzen/metafind/metafind_data_paper}
export PYTHONPATH="$PWD" METAFIND_DATA="$R" METAFIND_TEXT_TEMPLATE=v3_fit
O=$R/outputs; L=$O/logs; CK=$O/checkpoints
FILTER="FutureWarning|warnings.warn|enable_nested_tensor|timm.models.layers"
step() { echo; echo "=== $1  $(date '+%F %T')"; }
fail() { echo "PIPELINE FAILED at $1" >&2; exit 1; }
# awk returns success even when every line is a filtered warning.
filter_output() { awk -v pattern="$FILTER" '$0 !~ pattern'; }

step "0 wait for R2"
until grep -q "^=== R2 DONE" $L/r2_annotate_v10.log 2>/dev/null; do sleep 60; done

step "1 apply Kyzen's 21 manual exclusions (2026-08-28) to the new corpus"
$PY - <<'PYEOF'
import hashlib, json, os, re, shutil, stat
from pathlib import Path
from metafind import paths
from metafind.data.splits import ledger_excluded_uids
from metafind.scene.placement import _publish

def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate exclusion ledger key: {key}")
        value[key] = item
    return value

source = Path(os.environ.get("METAFIND_EXCLUSION_LEDGER", "/home/kyzen/metafind/metafind_data/outputs/annotation_exclusions.json")).resolve(strict=True)
source_bytes = source.read_bytes()
ex = json.loads(source_bytes, object_pairs_hook=unique_object)
group = ex["groups"]["manual_review_rejected"]
# Only the approved manual group survives this corpus rebuild. The old n05
# failures are retried by v10; do not copy all 332 historical exclusions.
uids = sorted(ledger_excluded_uids({"groups": {"manual_review_rejected": group},
                                  "excluded_total": group["n"]}))
if any(re.fullmatch(r"[0-9a-f]{32}", uid) is None for uid in uids):
    raise ValueError("manual exclusions must contain explicit Objaverse UIDs")
provenance = {key: ex[key] for key in ("decided_at", "decided_by", "decision", "git_commit")}
if any(not isinstance(value, str) or not value.strip() for value in provenance.values()):
    raise ValueError("source exclusion ledger lacks explicit decision provenance")
ledger = paths.OUTPUTS.resolve(strict=True) / "annotation_exclusions.json"
payload = {
    "schema": "metafind.annotation_exclusions.v1", "accounting_decision": "DL-106",
    "decision": "DL-106: carry approved manual exclusions separately from processing failures",
    "excluded_total": len(uids),
    "groups": {"manual_review_rejected": {"n": len(uids), "uids": uids}},
    "source_ledger": {"path": str(source), "sha256": hashlib.sha256(source_bytes).hexdigest(), **provenance},
}
part = ledger.with_name(ledger.name + ".part")
if ledger.is_symlink() or ledger.resolve() == source or (ledger.exists() and ledger.samefile(source)):
    raise FileExistsError("destination exclusion ledger aliases another file or the source")
if part.exists() or part.is_symlink():
    raise FileExistsError("refusing existing exclusion ledger publication temporary file")
existing = ledger.exists()
if existing:
    info = ledger.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise FileExistsError("destination exclusion ledger must be an independent regular file")
    if json.loads(ledger.read_bytes(), object_pairs_hook=unique_object) != payload:
        raise FileExistsError("destination exclusion ledger conflicts with the bound manual decision")
ann = paths.ANNOTATIONS.resolve(strict=True); out = ann.parent / "annotations_v10_excluded"
if not ann.is_dir() or (out.exists() and not out.is_dir()):
    raise FileExistsError("annotation source/archive must be directories")
pending = [ann / f"{uid}.json" for uid in uids if (ann / f"{uid}.json").exists()]
if any((ann / f"{uid}.json").is_symlink() for uid in uids) or any(not p.is_file() for p in pending):
    raise FileExistsError("manual annotation sources must be regular files")
if out.is_symlink() or any((out / p.name).exists() or (out / p.name).is_symlink() for p in pending):
    raise FileExistsError("refusing to overwrite previously excluded annotations")
if source.read_bytes() != source_bytes:
    raise ValueError("source exclusion ledger changed during preflight")
# Persist E before moving annotations: interruption cannot remove n09's second
# defence against a restored/reannotated rejected asset. No old corpus counts
# or historical n05 failures are copied into the rebuilt corpus.
if not existing:
    _publish(ledger, payload)
out.mkdir(exist_ok=True)
for p in pending:
    shutil.move(str(p), str(out / p.name))
moved = len(pending)
print(f"moved {moved} of {len(uids)} manually rejected assets to {out}")
print(f"manual exclusion ledger: {ledger} (DL-106; source {payload['source_ledger']['sha256']})")
PYEOF
$PY -c "
from metafind.data.annotate_run import rebuild_index; from metafind import paths
print('annotations indexed:', rebuild_index(paths.LOGS / 'annotations_index.jsonl'))" || fail "annotations index"

step "2 n05b: Stage 1 encoding protocol (text = v3_fit sentence within 77 tokens, image = mean of 11 views)"
$PY -m metafind.models.resolve_stage1 --paper-clip-train-scope trainable --actual-clip-train-scope frozen \
   --confidence moderate --decided-by "Claude, DL-103 (paper: full encoder fine-tuning; ViT-bigG cannot be trained on 32 GB)" 2>&1 | filter_output || fail "n05b"

step "3 n06: encode text + image"
$PY -m metafind.data.encode_text_image 2>&1 | filter_output | tail -n 30 || fail "n06"

step "4 n09: splits 80/20 (seed 20260816), Stage 1 protocol"
$PY -m metafind.data.splits --seed 20260816 --decided-by "Claude, DL-103: paper 80/20; select and report on the 20% (Kyzen 2026-09-04)" 2>&1 | filter_output || fail "n09"

step "5 Stage 1: paper-literal line (same record on both towers, image = 11-view mean, 30% query masking), lr 1e-4, 10 epochs, select on the 20%"
$PY -m metafind.train.stage1 --query-observation same_record --query-image-policy same_mean \
   --selection-split holdout --lr 1e-4 --epochs 10 --amp off --out-dir $CK/paper_v10_same_record_lr1e-4 2>&1 | filter_output | tail -n 40 || fail "stage1"
REC=$CK/paper_v10_same_record_lr1e-4/stage1_best_ckpt.json
[ -f $REC ] || fail "stage1 record missing"

step "6 gallery index n11 -> G4 -> n12"
$PY -m metafind.train.gallery_index stage1 --stage1-ckpt-record $REC 2>&1 | filter_output || fail "n11"
$PY -m metafind.gates.g4_gallery_freeze 2>&1 | filter_output || fail "G4"
$PY -m metafind.train.gallery_index promote 2>&1 | filter_output || fail "n12"

step "7 Table 1 row 1 (Stage 1 head): official evaluator, holdout -> holdout and full gallery"
$PY -m metafind.eval.run_retrieval --ckpt-record $REC --protocol A20_holdout_vs_holdout --protocol B_full_gallery \
   --out-dir $O/eval/table1_paper_v10_S1head --unseal 2>&1 | filter_output | tail -n 25 || fail "step 7"

step "DONE (row 1). Stage 2 follows in a separate chain after the room-level graphs (R6)"
