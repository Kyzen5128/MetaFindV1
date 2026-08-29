#!/bin/bash
# Wait for n05, then run n08 and a Stage 1 smoke -- each step gated on the last.
#
#   nohup bash tools/chain_after_n05.sh > .../logs/chain.log 2>&1 &
#
# Every stage here needs the GPU and n05 is holding 16.8 of 24.5 GB, so the
# point is to queue rather than to parallelise. Nothing starts until the thing
# before it has finished AND been checked: a chain that runs the next step on a
# failed predecessor turns one bad run into three.
#
# `set -e` is deliberately NOT used. Each step reports its own verdict and the
# script decides; an implicit exit would leave no record of which step stopped
# it or why.

# ─────────────────────────────────────────────────────────────────────────────
# RETIRED 2026-08-28 [MASTER]. Superseded by tools/chain_to_stage1.sh.
#
# This chain queues n08 BEFORE a Stage 1 smoke. n08 is a Stage 2 input, and
# PAPER FACT 2methdology.tex:75 / 3experiments.tex:24 put Stage 1 on
# Objaverse-LVIS isolated assets with no layout. Running this would put a
# 2.5-hour scene-graph node on Stage 1's critical path for nothing.
#
# [CORRECTED 2026-08-30] This said "the only production reader of
# sem_edge_cache.json is metafind/train/stage2.py:320". Codex R2 judged that
# FALSE on 2026-08-28 and the correction landed in the sibling
# tools/chain_to_stage1.sh; this file was missed. Both halves were wrong:
#
#   * There are TWO production readers, not one. Grepped 2026-08-30 over
#     metafind/ tools/ tests/, unpiped:
#       - metafind/train/stage2.py, Stage2Data.__init__  (line 379 today)
#       - metafind/data/scene_splits.py, main(), for semantic-edge coverage
#         (line 141 today)
#     Both are on the Stage 2 side, which is why the retirement below still
#     holds -- but it rests on "no Stage 1 reader", never on "one reader".
#   * ":320" was already stale when written; the stage2.py read is at 379.
#     Line numbers rot, so the readers are named by symbol above.
#
# Non-production mentions, for completeness, so the next grep is not a surprise:
# semantic_edges_run.py WRITES it; this file and tools/status.sh only check the
# path exists; tools/audit_claims.py and tools/check_graph.py refer to it.
#
# The file is KEPT, not deleted: it is the origin of that misreading, and the
# ledger points at it as evidence. A comment cannot stop `bash
# tools/chain_after_n05.sh`, so the guard is executable and comes first.
# 78 = EX_CONFIG, so a mistaken run gets a non-zero status rather than silence.
{
  echo "RETIRED 2026-08-28. Superseded by tools/chain_to_stage1.sh."
  echo "This chain puts n08 ahead of Stage 1. n08 belongs to Stage 2; Stage 1"
  echo "trains on Objaverse-LVIS with no scene graphs (2methdology.tex:75)."
  echo "Kept only as evidence of where that misreading came from. Not run."
} >&2
exit 78
# Everything below is preserved unchanged.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

REPO=/home/kyzen/MetaFindV1
PY=$HOME/miniconda3/envs/MetaFind/bin/python
# Roots come from metafind/paths.py, never spelled here. Six scripts used
# to hardcode the previous machine's /mnt/data1/kyzen/MetaFind, so on any
# other checkout they silently observed an empty directory.
eval "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && ${METAFIND_PYTHON:-python3} -m metafind.paths)"
OUT="$METAFIND_OUTPUTS"
LOGS=$OUT/logs

cd "$REPO" || exit 1
mkdir -p "$LOGS"

say() { echo "[$(date '+%F %H:%M:%S')] $*"; }

# Plot before exiting on failure too. A run that went wrong is exactly the one
# worth looking at, and the curves are often what SAYS what went wrong -- a loss
# that never moved, a temperature that ran away. An earlier version put the plot
# call after the last check, so the only path that skipped it was the failing
# one.
curves() { $PY "$REPO/tools/plot_training.py" 2>&1 | sed 's/^/  /'; }

die() {
    say "STOPPED: $*"
    curves
    say "curves (such as they are) -> $OUT/training_curves.html"
    say "nothing after this point ran"
    exit 1
}

# ---------------------------------------------------------------- wait for n05
say "waiting for n05 (annotate_run) to exit"
while pgrep -f "metafind.data.annotate_run" > /dev/null; do sleep 60; done
say "n05's process is gone"

# Gone is not the same as finished. A killed or crashed run also leaves no
# process, and the next steps would happily consume its partial output.
ANN=$(ls "$OUT/annotations" 2>/dev/null | wc -l)
say "annotations on disk: $ANN"
if [ "$ANN" -lt 45000 ]; then
    die "expected ~45,955 annotations, found $ANN -- n05 did not finish"
fi
if tail -50 "$LOGS/n05_full.log" 2>/dev/null | grep -qi "traceback\|CUDA out of memory"; then
    die "n05's log ends in an error; read $LOGS/n05_full.log"
fi

say "waiting 60s for the GPU to be released"
sleep 60
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# ---------------------------------------------------------------------- n08
say "=== n08 semantic edges: 4,242 pairs, ~2.4 h at n05's measured rate ==="
$PY -m metafind.data.semantic_edges_run >> "$LOGS/n08_full.log" 2>&1
rc=$?
[ $rc -eq 0 ] || die "n08 exited $rc -- read $LOGS/n08_full.log"

for f in sem_edge_cache.json sem_edge_embeddings.npz procthor_node_embeddings.json; do
    [ -f "$OUT/$f" ] || die "n08 returned 0 but did not write $f"
done
say "n08 wrote its three artifacts"

# The widths n13 reads come from these files, so a mismatch here is the
# KeyError-shaped failure one stage earlier. Cheap to check, so check.
$PY - <<'CHECK' || die "n08's artifacts are not self-consistent"
import json, numpy as np
from metafind import paths
cache = json.loads((paths.OUTPUTS / "sem_edge_cache.json").read_text())
node = json.loads((paths.OUTPUTS / "procthor_node_embeddings.json").read_text())
vecs = np.load(paths.OUTPUTS / "sem_edge_embeddings.npz")["embeddings"]
arr = np.load(node["uri"])["embeddings"]
assert vecs.shape[1] == int(cache["edge_dim"]), (vecs.shape, cache["edge_dim"])
assert arr.shape[1] == int(node["embedding_dim"]), (arr.shape, node["embedding_dim"])
print(f"  e_ij {vecs.shape}  t_i {arr.shape}  entries {len(cache['entries']):,}")
CHECK

# ------------------------------------------------------------- Stage 1 smoke
# Small on purpose. Its job is to prove the P0 fixes hold against a REAL loaded
# backbone -- three of them (the checkpoint's point-encoder section, the full
# gallery-encoder hash, the ProcTHOR pc_norm) cannot be tested without the 9.5 GB
# of weights, which is why they are marked SPEC ONLY in the audit.
say "=== Stage 1 smoke: 200 assets, 1 epoch ==="
$PY -m metafind.train.stage1 --limit 200 --epochs 1 >> "$LOGS/n10_smoke.log" 2>&1
rc=$?
[ $rc -eq 0 ] || die "Stage 1 smoke exited $rc -- read $LOGS/n10_smoke.log"

$PY - <<'CHECK' || die "the checkpoint is missing a section the optimizer moved"
import json, torch
from metafind import paths
from metafind.train.stage1 import CKPT_SECTIONS
ck = torch.load(paths.CHECKPOINTS / "stage1.pt", map_location="cpu", weights_only=False)
missing = [s for s in CKPT_SECTIONS if s not in ck]
assert not missing, missing
n = {s: sum(v.numel() for v in ck[s].values()) for s in CKPT_SECTIONS}
print("  " + "  ".join(f"{k.split('_')[0]}={v:,}" for k, v in n.items()))
# The whole point of P0-1: the point encoder has to BE here.
assert n["backbone_trainable_state"] > 1_000_000, (
    f"backbone section holds only {n['backbone_trainable_state']:,} parameters "
    "-- PointBERT is not in the checkpoint")
rec = json.loads((paths.CHECKPOINTS / "stage1_ckpt.json").read_text())
print(f"  record: {rec['n_params_saved']:,} params, {rec['size_bytes']/1e6:.0f} MB")
CHECK

# ---------------------------------------------------------------- curves
curves
say "curves -> $OUT/training_curves.html"

say "=== all three steps finished and verified ==="
say "next, and NOT automated: n11 gallery index, then n13"
