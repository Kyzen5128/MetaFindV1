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
import json, shutil
from pathlib import Path
from metafind import paths
ex = json.load(open("/home/kyzen/metafind/metafind_data/outputs/annotation_exclusions.json"))
uids = ex["groups"]["manual_review_rejected"]["uids"]
ann = paths.ANNOTATIONS.resolve(); out = ann.parent / "annotations_v10_excluded"; out.mkdir(exist_ok=True)
moved = 0
for u in uids:
    p = ann / f"{u}.json"
    if p.exists():
        shutil.move(str(p), str(out / p.name)); moved += 1
print(f"moved {moved} of {len(uids)} manually rejected assets to {out}")
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
