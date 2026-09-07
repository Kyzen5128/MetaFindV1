#!/bin/bash
# DL-103 R6/R8/R9: Stage 2 on the paper-first line, after the Stage 1 chain (row 1) and the
# unified ProcTHOR renders. R6b gemma captions -> metadata text with captions -> n09b protocol
# (graph_unit room) -> n08 semantic edges + node embeddings -> n09c scene splits -> n11b index
# -> Stage 2 (9,600 houses, 1 epoch) -> ProcTHOR probe -> Table 1 both heads (holdout) -> tables.
# Stop and restart waiting shells when changing this file; Bash buffers script input.
set -Eeuo pipefail
trap 'rc=$?; echo "PIPELINE FAILED (line $LINENO, exit $rc)" >&2; exit "$rc"' ERR
cd "${METAFIND_REPO:-/home/kyzen/MetaFindV1}"
PY=${METAFIND_PYTHON:-/home/kyzen/miniconda3/envs/MetaFind/bin/python}
R=${METAFIND_DATA:-/home/kyzen/metafind/metafind_data_paper}
export PYTHONPATH="$PWD" METAFIND_DATA="$R" METAFIND_TEXT_TEMPLATE=v3_fit
O=$R/outputs; L=$O/logs; CK=$O/checkpoints
FILTER="FutureWarning|warnings.warn|enable_nested_tensor|timm.models.layers|Loading weights"
step() { echo; echo "=== $1  $(date '+%F %T')"; }
fail() { echo "PIPELINE FAILED at $1" >&2; exit 1; }
# awk returns success even when every line is a filtered warning.
filter_output() { awk -v pattern="$FILTER" '$0 !~ pattern'; }

step "0 wait: Stage 1 chain (row 1) and the unified ProcTHOR renders"
until grep -q "^=== DONE (row 1)" $L/chain_paper_stage1_20260906.log 2>/dev/null; do sleep 60; done
until grep -q "rendered, .* quarantined" $L/n07b_procthor_modalities_v2.log 2>/dev/null; do sleep 60; done
REC=$CK/paper_v10_same_record_lr1e-4/stage1_best_ckpt.json
[ -f $REC ] || fail "Stage 1 record missing"

step "1 R6b: gemma one-sentence captions for the 1,467 ProcTHOR assets (unified renders)"
$PY tools/procthor_captions.py 2>&1 | filter_output || fail "captions"
$PY tools/procthor_metadata_text.py --captions $O/procthor_captions.json 2>&1 | tail -n 2 || fail "metadata text"

step "2 n09b: Stage 2 protocol (graph_unit room)"
$PY -m metafind.models.resolve_stage2 --decided-by "Claude, DL-103 (room-level graphs; recipe unchanged)" 2>&1 | filter_output || fail "n09b"

step "3 n08: semantic edges (gemma) + node embeddings (frozen text tower)"
for f in sem_edge_cache.json sem_edge_embeddings.npz sem_edge_sentences.jsonl procthor_node_embeddings.json procthor_node_embeddings.npz; do if [ -L "$O/$f" ]; then rm -- "$O/$f"; fi; done
$PY -m metafind.data.semantic_edges_run 2>&1 | filter_output | tail -n 20 || fail "n08"

step "4 n09c: scene splits (80/20 houses, seed 20260816) + coverage"
if [ -L "$O/scene_splits.json" ]; then rm -- "$O/scene_splits.json"; fi
$PY -m metafind.data.scene_splits --seed 20260816 2>&1 | filter_output | tail -n 8 || fail "n09c"

step "5 n11b: Stage 2 gallery index"
$PY -m metafind.train.gallery_index stage2 --stage1-ckpt-record $REC 2>&1 | filter_output | tail -n 8 || fail "n11b"

step "6 Stage 2: ESSGNN + query fusion, both encoders frozen, 9,600 houses, 1 epoch, scene dropout 0.3, bidirectional"
$PY -m metafind.train.stage2 --variant full --stage1-ckpt-record $REC \
  --hyperparameters workflow/stage2_hyperparameters_ft_lr5e-5.json --query-modality-masking none --overwrite 2>&1 | filter_output | tail -n 20 || fail "stage2"
ARM=$CK/stage2_arms/S2_paper_v10_none_ft5e-5_allhouses_room; mkdir -p $ARM
cp "$CK/stage2_full.pt" "$ARM/"
cp "$O/variant_ckpts.json" "$ARM/"
cp "$L/train_stage2_full.jsonl" "$ARM/"
cp "$O/stage2_gallery_index.json" "$ARM/"
$PY - <<'PYEOF'
import json
from pathlib import Path
from metafind import paths
arm = paths.CHECKPOINTS / "stage2_arms/S2_paper_v10_none_ft5e-5_allhouses_room"
rec = json.loads((arm / "variant_ckpts.json").read_text()); rec["full"]["uri"] = str(arm / "stage2_full.pt")
(arm / "variant_ckpts.json").write_text(json.dumps(rec, indent=1)); print("arm record uri ->", rec["full"]["uri"])
PYEOF

step "7 ProcTHOR probe S1 / S2-off / S2-on (300 test houses, room-level context)"
$PY tools/probes/stage2_procthor_retrieval.py --stage1-ckpt-record $REC --stage2-record $ARM/variant_ckpts.json \
  --houses 300 --query-mode none --out output/look/exp_stage2_procthor_retrieval_paper_v10.json 2>&1 | filter_output || echo "FAILED step 7 (probe); Table 1 steps continue"

step "8 Table 1 both heads, holdout -> holdout, own / weak own / partner"
CKPT=$CK/paper_v10_same_record_lr1e-4/stage1_best.pt
$PY tools/probes/exp_type_level_query.py --ckpt $CKPT --query-split holdout --gallery-split holdout --pc-policies canonical \
   --no-sketchfab --out output/look/table1_paper_v10_S1head_holdout.json 2>&1 | filter_output || fail "step 8a"
$PY tools/probes/exp_type_level_query.py --ckpt $CKPT --stage2-state $ARM/stage2_full.pt --query-split holdout --gallery-split holdout --pc-policies canonical \
   --no-sketchfab --out output/look/table1_paper_v10_S2head_holdout.json 2>&1 | filter_output || fail "step 8b"
$PY tools/probes/tabulate_table1_final.py output/look/table1_paper_v10_S1head_holdout.json output/look/table1_paper_v10_S2head_holdout.json > output/look/table1_paper_v10_tables.md
step "DONE (row 2)"
