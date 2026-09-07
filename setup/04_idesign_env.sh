#!/usr/bin/env bash
# MetaFind — I-Design 與 LLM 服務環境（graph 的 n15c_prepare_eval_scenes）
#
# 用法：
#   bash setup/04_idesign_env.sh
#
# 這裡建立兩個環境，都刻意與 MetaFind 環境分開：
#
#   IDesign   場景生成。它釘 networkx 2.6 / jsonschema 4.3 / numpy 1.26，
#             裝進 MetaFind 會弄壞 transformers 5 與 torch 2.11。
#             兩者的介面是磁碟上的 JSON 檔，不需要共用直譯器。
#
#   serve     vLLM。它自帶 torch 2.13，同樣不能裝進 MetaFind。
#             放在 /mnt/data1，因為 / 只剩約 100G。
#
# ---------------------------------------------------------------------------
# 兩個實測踩到的坑，寫在這裡才不會被重新踩一次：
#
# 1) I-Design 的 requirements.txt 照抄裝不起來。
#    它寫 `ag2==0.2.0`，但 PyPI 上 ag2 最早是 0.3.2b2 —— 那個 commit
#    (0827ad2 "Migrate from pyautogen to ag2") 只做了機械式改名，沒驗證
#    版本存在。改名前的 `pyautogen==0.2.0` 才是程式實際寫作時的 API。
#
# 2) serve 環境必須是 Python 3.13，不能是 3.11/3.12。
#    vLLM 依賴的 flashinfer 在 comm/fd_exchange.py 用了
#        def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
#    而 `array.array` 直到 Python 3.13 才可下標，該檔又沒有
#    `from __future__ import annotations`，於是 import 期就
#        TypeError: type 'array.array' is not subscriptable
#    移除 flashinfer 不能繞過 —— vLLM 別處硬相依它，只會換成
#    ModuleNotFoundError。用 3.13 是唯一不必改 site-packages 的解法。
#
# 3) MinkowskiEngine / dgl / torch 1.12 全部不需要。
#    I-Design 的 README 與 Dockerfile 都要你裝，但實測 import graph 後，
#    那些只有 retrieve.py 要 —— 而 retrieve.py 正是 MetaFind 取代的元件
#    （它用 OpenShape 對 Objaverse 檢索）。場景生成那半只要
#    autogen + networkx + jsonschema + matplotlib + opencv。
# ---------------------------------------------------------------------------

set -euo pipefail

METAFIND_SETUP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDESIGN_REPO=${IDESIGN_REPO:-"$(dirname "$METAFIND_SETUP_REPO")/upstream/IDesign"}
IDESIGN_COMMIT=7bc891c            # 釘住；I-Design repo 無 LICENSE，不 vendor 進本 repo
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)/patches"
# Roots come from metafind/paths.py, never spelled here. Six scripts used
# to hardcode the previous machine's /mnt/data1/kyzen/MetaFind, so on any
# other checkout they silently observed an empty directory.
eval "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && ${METAFIND_PYTHON:-python3} -m metafind.paths)"
SERVE_PREFIX="$METAFIND_DATA/envs/serve"
DATA_ROOT="$METAFIND_DATA"

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- I-Design
if [ ! -f "$IDESIGN_REPO/IDesign.py" ]; then
    echo "找不到 I-Design：$IDESIGN_REPO"
    echo "  git clone https://github.com/atcelen/IDesign.git $IDESIGN_REPO"
    exit 1
fi
echo "==> I-Design at $IDESIGN_REPO ($(git -C "$IDESIGN_REPO" rev-parse --short HEAD))"

# 我們對 I-Design 的四個 patch（01–03 保留既有差异）：
#   01  把寫死的 "gpt-4" 模型名改成真實的 Qwen 名稱（純命名）
#   02  佈局元素歸位、preposition 對齊 enum、丟棄懸空引用、物件去重（**改變行為**）
#   03  修正迴圈加上上限、每輪換 cache_seed、耗盡時放棄場景（**改變行為**）
#   04  四個 model filter 改讀 METAFIND_IDESIGN_MODEL，支援已決議的真實 Gemma 名稱
#
# 02 與 03 會改變產出的場景與完成率，不是格式調整。每個場景的 sidecar
# 都會記下實際套用了哪些，避免日後被當成接近原版的 I-Design。

# 先確認 revision。patch 是針對這個 commit 做的；HEAD 不同就不該硬套。
ACTUAL_HEAD=$(git -C "$IDESIGN_REPO" rev-parse --short=7 HEAD)
if [ "$ACTUAL_HEAD" != "$IDESIGN_COMMIT" ]; then
    echo "I-Design HEAD 是 $ACTUAL_HEAD，預期 $IDESIGN_COMMIT"
    echo "  patch 是針對 $IDESIGN_COMMIT 做的。請 checkout 該 commit，或重新產生 patch。"
    exit 1
fi

# 04 覆蓋 01 的模型行，必須核對累積終態，不能各自 reverse-check。
# 只補完整前綴之後的 patch；完整鏈略過，受影響檔案的未知修改拒絕。
${METAFIND_PYTHON:-python3} "$METAFIND_SETUP_REPO/tools/idesign_patches.py" apply \
    --repo "$IDESIGN_REPO" --patch-dir "$PATCH_DIR"

if ! conda env list | grep -qE '^IDesign\s'; then
    conda create -n IDesign python=3.10 -y -q
fi
conda activate IDesign
# 版本依 I-Design 的 requirements.txt，但 ag2 -> pyautogen（見上面第 1 點）
pip install -q \
    "pyautogen==0.2.0" \
    "networkx==2.6.3" \
    "jsonschema==4.3.2" \
    "opencv-python==4.8.1.78" \
    "numpy==1.26.4" \
    "flaml==2.1.2" \
    matplotlib
conda deactivate

# ---------------------------------------------------------------- vLLM
echo "==> serve 環境（Python 3.13，見上面第 2 點）"
if [ ! -d "$SERVE_PREFIX" ]; then
    conda create -p "$SERVE_PREFIX" python=3.13 -y -q
fi
conda activate "$SERVE_PREFIX"
TMPDIR="$DATA_ROOT/tmp" pip install -q vllm
python - <<'PY'
import sys, vllm, torch
import flashinfer.comm  # 這行在 Python 3.11 會 TypeError；3.13 正常
print(f"vllm {vllm.__version__} | torch {torch.__version__} | py {sys.version.split()[0]}")
PY
conda deactivate

cat <<EOF

環境就緒。啟動 LLM 服務（I-Design 的規劃 agent 是純文字）：

  conda activate $SERVE_PREFIX
  export HF_HOME=$DATA_ROOT/models/hf-cache
  vllm serve /home/kyzen/metafind/metafind_out/gemma-4-12B-it \\
      --served-model-name gemma-4-12B-it \\
      --max-model-len 16384 --gpu-memory-utilization 0.85 --port 8000

  # 這是服務設定範例，尚未在此流程實跑；需服務環境支援此模型。
  # wrapper 將同一個真實模型名傳到四個 filter、OAI config 與 sidecar。

然後產生場景：

  conda activate IDesign
  python "$METAFIND_SETUP_REPO/tools/idesign_generate.py" \\
      --idesign-repo "$IDESIGN_REPO" --model gemma-4-12B-it --n-scenes 2
EOF
