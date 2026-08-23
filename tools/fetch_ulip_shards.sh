#!/bin/bash
# Fetch more of ULIP's released objaverse_lvis clouds, so V1.0 can actually
# decide D-12. Only shard 000-009 is on this machine; the measurement was
# designed for a population of 37 and got 1.
#
#   setsid nohup ionice -c3 nice -n19 bash tools/fetch_ulip_shards.sh \
#       > data/outputs/logs/ulip_shards.log 2>&1 < /dev/null &
#
# Runs at idle I/O priority on purpose: the n03/n04/n05 chain owns this disk,
# and this is a yardstick for a 2% question. It must never be the reason the
# main run slows down.
#
# The tarball is deleted right after it extracts. Keeping 40 of them would cost
# 46 GB for files whose contents are already on disk.

set -uo pipefail

REPO=/home/kyzen/MetaFindV1
PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python
N_SHARDS="${N_SHARDS:-40}"
MIN_FREE_GB="${MIN_FREE_GB:-150}"

cd "$REPO" || exit 1
eval "$($PY -m metafind.paths)"
DEST=$METAFIND_OUTPUTS/reference/ulip_npy
STAGE=$METAFIND_OUTPUTS/reference/_ulip_stage
mkdir -p "$DEST" "$STAGE"

say() { echo "[$(date '+%F %H:%M:%S')] $*"; }

for ((i=0; i<N_SHARDS; i++)); do
    SHARD=$(printf "000-%03d" "$i")
    if [[ -d "$DEST/$SHARD" ]]; then
        say "$SHARD already extracted, skipping"
        continue
    fi
    # The main run needs room far more than this yardstick does.
    FREE=$(df -BG --output=avail "$METAFIND_OUTPUTS" | tail -1 | tr -dc '0-9')
    if [[ $FREE -lt $MIN_FREE_GB ]]; then
        say "STOP -- only ${FREE}G free, floor is ${MIN_FREE_GB}G. Got $(ls -d "$DEST"/000-* 2>/dev/null | wc -l) shards."
        exit 0
    fi

    say "$SHARD downloading (${FREE}G free)"
    TAR=$($PY - "$SHARD" "$STAGE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
shard, stage = sys.argv[1], sys.argv[2]
try:
    print(hf_hub_download("SFXX/ULIP", f"ULIP-2/objaverse_lvis/{shard}.tar.gz",
                          repo_type="dataset", local_dir=stage))
except Exception as e:                       # a 404 or a dropped connection is
    print(f"FAILED {type(e).__name__}: {e}", file=sys.stderr)   # not fatal --
    raise SystemExit(1)                                        # the next shard
PY
    ) || { say "$SHARD download failed, moving on"; continue; }

    tar xzf "$TAR" -C "$DEST" \
        && say "$SHARD extracted" \
        || say "$SHARD extract FAILED"
    rm -f "$TAR"
done

rm -rf "$STAGE"
say "DONE -- $(ls -d "$DEST"/000-* 2>/dev/null | wc -l) shards, $(find "$DEST" -name '*.npy' | wc -l) clouds"
