#!/usr/bin/env bash
# MetaFind — 建立 conda 環境（graph 的 n01_env_bootstrap）
#
# 用法：
#   bash setup/02_conda_env.sh              # 建立 MetaFind 環境
#   bash setup/02_conda_env.sh --name Foo   # 換個環境名
#
# 之後：
#   conda activate MetaFind
#   python setup/03_verify_env.py
#
# 為什麼不直接 pip install -r /home/kyzen/ULIP/requirements.txt：
#   那份鎖的是 2021–2022 的版本（timm==0.4.12、open3d==0.16.0），
#   timm 0.4.12 與 torch 2.9 不相容，open3d 0.16 沒有 py3.11 的 wheel。
#   另見 docs/graph/00_FINDINGS.md F4。

set -euo pipefail

ENV_NAME=MetaFind
PY_VER=3.11
# Roots come from metafind/paths.py, never spelled here. Six scripts used
# to hardcode the previous machine's /mnt/data1/kyzen/MetaFind, so on any
# other checkout they silently observed an empty directory.
eval "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && ${METAFIND_PYTHON:-python3} -m metafind.paths)"
ROOT="$METAFIND_DATA"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) ENV_NAME="$2"; shift 2 ;;
    *) echo "未知參數: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- 前置檢查
command -v conda >/dev/null || { echo "找不到 conda"; exit 1; }

if [[ ! -d "${ROOT}/cache/hf" ]]; then
  echo "找不到 ${ROOT}/cache/hf"
  echo "請先跑 setup/01_storage.sh（或手動建目錄）。"
  exit 1
fi

# cache 導向大硬碟：ViT-bigG-14 約 10GB，/ 只剩 108GB
export HF_HOME="${ROOT}/cache/hf"
export TORCH_HOME="${ROOT}/cache/torch"

# ---------------------------------------------------------------- 建立環境
# 可重跑：環境已存在就跳過建立，直接往下裝套件。
# （安裝中途失敗時要能重跑，不該因為「環境已存在」就整個中斷。）
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> 環境 ${ENV_NAME} 已存在，跳過建立"
else
  echo "==> 建立 conda env: ${ENV_NAME} (python ${PY_VER})"
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi

# conda 的 shell hook 會讀到未定義變數（$PS1 等），在 set -u 下會直接中斷，
# 所以 activate 前後要把 -u 關掉再開回來。
# shellcheck disable=SC1091
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

echo "==> 使用 $(python -V) @ $(which python)"

# 全程用 python -m pip，確保打到的是這個環境的 pip 而不是 base 的
PIP="python -m pip"
$PIP install --upgrade pip

# ---------------------------------------------------------------- 核心相依
echo "==> PyTorch (cu128，對齊本機 driver 580 / CUDA 12.8)"
$PIP install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

echo "==> ULIP-2 backbone"
# open_clip >=2.24 才有 ViT-bigG-14 laion2b_s39b_b160k
# h5py / matplotlib：ULIP 的 import 鏈需要（models/pointbert/misc.py、utils/io.py），
#   不裝的話 ULIP2_PointBERT_Colored 會在 import 階段就掛掉。
$PIP install "open_clip_torch>=2.24" "timm>=1.0" easydict pyyaml termcolor ftfy regex \
             h5py matplotlib

echo "==> 3D 資料處理"
# numpy 不釘版本：torch 2.9 支援 numpy 2.x，硬釘反而可能造成 ABI 不合。
# 真的有套件需要 numpy<2 時，在下面的可選段落處理。
$PIP install trimesh scipy objaverse

echo "==> Qwen 標註（本地 VLM）"
$PIP install "transformers>=4.51" accelerate qwen-vl-utils pillow

echo "==> ProcTHOR / AI2-THOR（場景資料集與 Stage 2 的資產模態）"
# `prior` 下載 procthor-10k；`ai2thor` 是實際渲染它的 Unity 應用。
# 版本不是隨意的：procthor-10k 的目前 revision 要求 AI2-THOR 5.0+，
# 而 F24／F25 的所有實測數字都綁在這個 build 上。
#   ai2thor 5.0.0
#   CloudRendering  thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917
#   procthor-10k    allenai/procthor-10k @ 439193522244720b86d8c81cde2e51e3a4d150cf
# Unity build（約 800 MB）預設下載到 ~/.ai2thor，也就是根目錄。
# 這裡先把它導到資料碟，理由與 HF_HOME 相同（見 01_storage.sh）。
if [[ ! -e "$HOME/.ai2thor" ]]; then
  mkdir -p "${ROOT}/ai2thor"
  ln -s "${ROOT}/ai2thor" "$HOME/.ai2thor"
fi
# EXACT, not a range. `>=5.0,<6` would let a rebuild take 5.1 and change the
# Unity build underneath -- 12,000 houses would still load and headless
# rendering would still work, while the asset database is no longer 1,934 and
# F24/F25 are no longer the same experiment. A pin that only holds the major
# version pins nothing that matters here.
$PIP install "ai2thor==5.0.0" "prior==1.0.3"

echo "==> 工程支撐：sidecar / progress / 測試"
$PIP install tqdm orjson pandas pyarrow filelock pytest pytest-xdist rich

# ---------------------------------------------------------------- 核心煙霧測試
echo "==> 核心煙霧測試"
python - <<'PY'
import torch, open_clip, timm, trimesh, transformers
assert torch.cuda.is_available(), "torch 看不到 CUDA"
print(f"    torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
print(f"    open_clip {open_clip.__version__} / timm {timm.__version__} / transformers {transformers.__version__}")
PY

# ---------------------------------------------------------------- 可選相依
# 渲染與點雲視覺化在 headless 機器上常因缺 EGL/OSMesa 而裝不起來。
# 它們只在 n04_object_prep 的渲染節點用到，03_verify_env.py 完全不需要，
# 所以這裡失敗只警告，不中斷。
echo "==> 可選相依（失敗不中斷）"
set +e

if $PIP install pyrender PyOpenGL PyOpenGL-accelerate; then
  echo "    pyrender OK"
else
  echo "    !! pyrender 失敗。headless 渲染可能要先裝系統套件："
  echo "       sudo apt install -y libgl1 libegl1 libosmesa6 libglib2.0-0"
  echo "       替代方案：Blender headless（見 docs/graph/02_BUILD_STEPS.md Step 1.1）"
fi

if $PIP install open3d; then
  echo "    open3d OK"
else
  echo "    !! open3d 失敗。只影響點雲視覺化，不影響管線。"
fi

set -e

# ---------------------------------------------------------------- 收尾
cat <<EOF

==> 完成。接著跑：

    conda activate ${ENV_NAME}
    python setup/03_verify_env.py

刻意不安裝的兩個套件：
  pointnet2_ops / knn_cuda  ULIP 只有 misc.fps 真的用到 pointnet2，knn_point 已是
                            純 torch。我們用純 torch FPS 取代，不必編譯 CUDA extension。
  faiss                     48K x 1280 x 4B = 246MB，用 torch 精確內積即可，
                            順便消掉一個不確定性來源（graph_spec NS-6）。
EOF
