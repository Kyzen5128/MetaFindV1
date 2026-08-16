"""Every path the project uses, in one place.

Layout under ``data/`` (a symlink to /mnt/data1/kyzen/MetaFind)::

    datasets/           raw data, exactly as the paper names it
      objaverse-lvis/     46,052 GLB assets + the uid manifest
      procthor-10k/       12,000 house layouts as JSONL
    models/             pretrained weights, never trained by us
      ulip2/              the frozen ULIP-2 checkpoint
      hf-cache/           HuggingFace cache: ViT-bigG-14, Qwen2.5-VL
    outputs/            everything we derive; safe to delete and regenerate
      pointclouds/        sampled from the meshes
      renders/            11 views per asset
      annotations/        Qwen captions
      embeddings/         cached 1280-d modality vectors
      scene_graphs/       ProcTHOR -> graph
      checkpoints/        training runs
      logs/               progress, sidecars, gate records
      reference/          held-out ULIP point clouds, for cross-checking

The split that matters: ``datasets/`` and ``models/`` are downloaded and never
written to; ``outputs/`` is entirely reproducible from them. Losing outputs costs
compute, losing datasets costs a re-download.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("METAFIND_DATA", REPO / "data"))

# ---- raw inputs
DATASETS = DATA / "datasets"
OBJAVERSE = DATASETS / "objaverse-lvis"
OBJAVERSE_GLB = OBJAVERSE / "glbs"
LVIS_MANIFEST = OBJAVERSE / "lvis.json"
LVIS_METADATA = OBJAVERSE / "objaverse_lvis_metadata.json"
PROCTHOR = DATASETS / "procthor-10k"

# ---- pretrained weights
MODELS = DATA / "models"
ULIP2_CKPT = (
    MODELS / "ulip2/ULIP-2/pretrained_models"
    / "ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
)
HF_CACHE = MODELS / "hf-cache"

# ---- derived
OUTPUTS = DATA / "outputs"
POINTCLOUDS = OUTPUTS / "pointclouds"
RENDERS = OUTPUTS / "renders"
ANNOTATIONS = OUTPUTS / "annotations"
EMBEDDINGS = OUTPUTS / "embeddings"
SCENE_GRAPHS = OUTPUTS / "scene_graphs"
PROCTHOR_MODALITIES = OUTPUTS / "procthor_modalities"
CHECKPOINTS = OUTPUTS / "checkpoints"
LOGS = OUTPUTS / "logs"
REFERENCE = OUTPUTS / "reference"

ALL_OUTPUT_DIRS = (
    POINTCLOUDS, RENDERS, ANNOTATIONS, EMBEDDINGS,
    SCENE_GRAPHS, CHECKPOINTS, LOGS, REFERENCE,
)


def setup_env() -> None:
    """Point the HuggingFace and torch caches at the data volume.

    Must run before torch, open_clip or transformers are imported: they read
    these variables once, at import. Relying on the shell to export them means a
    non-login shell silently fills the root partition, which has ~100 GB free
    against a ~26 GB model set.

    HF_HUB_DISABLE_XET is set because authenticating switches the Hub onto its
    xet backend, measured here at ~0.6 KB/s against 6 MB/s over plain HTTP.
    """
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("TORCH_HOME", str(HF_CACHE / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def ensure_dirs() -> None:
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable inventory, for status reporting."""

    def size(p: Path) -> str:
        if not p.exists():
            return "缺"
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if total < 1024:
                return f"{total:.0f}{unit}"
            total /= 1024
        return f"{total:.0f}PB"

    rows = [
        ("datasets/objaverse-lvis", OBJAVERSE),
        ("datasets/procthor-10k", PROCTHOR),
        ("models/ulip2", MODELS / "ulip2"),
        ("models/hf-cache", HF_CACHE),
        ("outputs", OUTPUTS),
    ]
    return "\n".join(f"  {name:28s} {size(path):>8s}" for name, path in rows)
