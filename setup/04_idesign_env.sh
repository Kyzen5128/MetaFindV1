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

IDESIGN_REPO=${IDESIGN_REPO:-/home/kyzen/IDesign}
IDESIGN_COMMIT=7bc891c            # 釘住；I-Design repo 無 LICENSE，不 vendor 進本 repo
SERVE_PREFIX=/mnt/data1/kyzen/MetaFind/envs/serve
DATA_ROOT=/mnt/data1/kyzen/MetaFind

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- I-Design
if [ ! -f "$IDESIGN_REPO/IDesign.py" ]; then
    echo "找不到 I-Design：$IDESIGN_REPO"
    echo "  git clone https://github.com/atcelen/IDesign.git $IDESIGN_REPO"
    exit 1
fi
echo "==> I-Design at $IDESIGN_REPO ($(git -C "$IDESIGN_REPO" rev-parse --short HEAD))"

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
  vllm serve Qwen/Qwen2.5-7B-Instruct \\
      --served-model-name gpt-4 gpt-4-1106-preview gpt-3.5-turbo-1106 \\
      --max-model-len 16384 --gpu-memory-utilization 0.85 --port 8000

  # --served-model-name 讓 I-Design 過濾的模型名直接解析到 Qwen，
  # 因此完全不用改它的原始碼。真實模型 id 會記進每個場景的 sidecar。

然後產生場景：

  conda activate IDesign
  PYTHONPATH=$IDESIGN_REPO python tools/idesign_generate.py --n-scenes 2
EOF
