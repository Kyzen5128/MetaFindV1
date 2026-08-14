#!/usr/bin/env bash
# MetaFind — 建立 conda 環境（graph 的 n01_env_bootstrap）
#
# 用法：  bash setup/02_conda_env.sh
# 之後：  conda activate metafind && python setup/03_verify_env.py
#
# 設計依據見 docs/graph/00_FINDINGS.md 的 F4：
#   ULIP 的 requirements.txt 鎖的是 2023 年的版本（timm==0.4.12、open3d==0.16.0、
#   open-clip-torch==2.24.0），在 torch 2.x / Python 3.11 上不會乾淨安裝，
#   而且 data/dataset_3d.py:544 還 import 了 PyTorch 2.0 已移除的 torch._six。
#   所以我們**不照抄** requirements.txt，改用現代版本組合 + compat patch。

set -euo pipefail

ENV_NAME=metafind
PY_VER=3.11
ROOT=/mnt/data1/kyzen/MetaFind

# ---- cache 導向大硬碟（ViT-bigG-14 約 10GB，不要塞爆 home）
export HF_HOME="${ROOT}/cache/hf"
export TORCH_HOME="${ROOT}/cache/torch"

echo "==> 建立 conda env: ${ENV_NAME} (python ${PY_VER})"
conda create -y -n "${ENV_NAME}" python="${PY_VER}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> PyTorch (cu128，對齊本機 driver 580 / CUDA 12.8)"
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision

echo "==> ULIP-2 backbone 相依"
# open_clip 需要 >=2.24 才有 ViT-bigG-14 laion2b_s39b_b160k
pip install \
    "open_clip_torch>=2.24" \
    "timm>=1.0" \
    easydict pyyaml termcolor ftfy regex

echo "==> 3D 資料處理"
pip install \
    trimesh "numpy<2.3" scipy \
    objaverse \
    pyrender PyOpenGL PyOpenGL-accelerate \
    open3d

echo "==> Qwen 標註（本地 VLM，取代 GPT-4o）"
pip install \
    "transformers>=4.51" accelerate \
    qwen-vl-utils pillow

echo "==> 工程支撐：sidecar / progress / 測試"
pip install \
    tqdm orjson pandas pyarrow filelock \
    pytest pytest-xdist \
    rich

echo
echo "==> 完成。接著跑驗證："
echo "    conda activate ${ENV_NAME}"
echo "    python setup/03_verify_env.py"
echo
echo "注意：以下兩個套件**刻意不安裝** ——"
echo "  pointnet2_ops / knn_cuda : ULIP 只在 misc.fps 真的用到 pointnet2，"
echo "                             knn_point 已是純 torch。我們用純 torch FPS 取代，"
echo "                             不需要編譯任何 CUDA extension。"
echo "  faiss                    : 48K x 1280 x 4B = 246MB，直接用 torch 精確內積，"
echo "                             順便消掉一個不確定性來源（見 graph_spec NS-6）。"
