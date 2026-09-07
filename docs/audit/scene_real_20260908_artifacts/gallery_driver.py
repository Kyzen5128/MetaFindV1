"""Isolated actual n11 -> G4 -> n12 diagnostic; never edits the source corpus."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
DATA = HERE / "data"
SOURCE = Path("/home/kyzen/metafind/metafind_data_attrs")
PROTOCOL = REPO / "output/validation/custom_table1_cpu_real_20260907/protocol/protocol.json"
PARENT = REPO / "output/validation/stage1_cpu_real_20260908/data/outputs/checkpoints/actual_cpu_step/stage1_best_ckpt.json"
PYTHON = "/home/kyzen/miniconda3/envs/MetaFind/bin/python"
ENV = {"METAFIND_DATA": str(DATA), "METAFIND_TEXT_TEMPLATE": "attrs_v1",
       "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
       "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HOME": str(SOURCE / "models/hf-cache"),
       "TORCH_HOME": str(SOURCE / "models/hf-cache/torch"),
       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def prepare():
    import numpy as np
    protocol = json.loads(PROTOCOL.read_text())
    ids = protocol["gallery_uids"]
    assert len(ids) == len(set(ids)) == 6 and Path(protocol["data_root"]) == SOURCE
    original_split = SOURCE / "outputs/splits.json"
    split = json.loads(original_split.read_text())
    admitted = set(sum((split["object"].get(k, []) for k in ("train", "val", "test")), []))
    assert set(ids) <= admitted
    parent = json.loads(PARENT.read_text())
    assert sha(parent["uri"]) == parent["sha256"]
    parent_out = PARENT.parents[2]
    DATA.mkdir(exist_ok=False)
    (DATA / "models").symlink_to((SOURCE / "models").resolve(), target_is_directory=True)
    out = DATA / "outputs"
    out.mkdir()
    sources = {str(p): sha(p) for p in (PROTOCOL, PARENT, Path(parent["uri"]), original_split,
                                        parent_out / "splits.json")}
    for sub in ("embeddings", "pointclouds", "annotations"):
        (out / sub).mkdir()
        for uid in ids:
            for ext in ((".json",) if sub == "annotations" else (".json", ".npz")):
                src = SOURCE / "outputs" / sub / (uid + ext)
                raw = src.read_bytes()
                sources[str(src)] = hashlib.sha256(raw).hexdigest()
                (out / sub / src.name).write_bytes(raw)
    for uid in ids:
        with np.load(out / "embeddings" / (uid + ".npz"), allow_pickle=False) as arrays:
            assert all(arrays[k].shape == (1280,) and np.isfinite(arrays[k]).all()
                       for k in ("text", "image"))
        with np.load(out / "pointclouds" / (uid + ".npz"), allow_pickle=False) as arrays:
            assert all(arrays[k].shape == (10000, 3) and np.isfinite(arrays[k]).all()
                       for k in ("xyz", "rgb"))
        side = json.loads((out / "embeddings" / (uid + ".json")).read_text())
        assert side["clip_train_scope"] == "frozen" and side["n_views"] == 12
        assert side["text_serialization"] == protocol["encoding_protocol"]["content"]["text_serialization"]
    for name in ("stage1_encoding_protocol.json", "stage1_protocol.json", "stage1_hyperparameters.json"):
        src = parent_out / name
        raw = src.read_bytes()
        sources[str(src)] = hashlib.sha256(raw).hexdigest()
        (out / name).write_bytes(raw)
    write(out / "splits.json", {
        "classification": "DIAGNOSTIC INDEX CORPUS ONLY: six fixed actual Objaverse assets; not the parent training split, not G3-certified",
        "source": str(original_split), "source_sha256": sources[str(original_split)],
        "selection_protocol": str(PROTOCOL), "selection_protocol_sha256": sources[str(PROTOCOL)],
        "parent_training_split": str(parent_out / "splits.json"),
        "parent_training_split_sha256": sources[str(parent_out / "splits.json")],
        "split_seed": int(split["split_seed"]), "admitted_total": len(ids),
        "object": {"train": [], "val": [], "test": ids},
    })
    meshes = {}
    wanted = set(ids)
    for path in (SOURCE / "datasets/objaverse-lvis/glbs").rglob("*.glb"):
        if path.stem in wanted:
            assert path.stem not in meshes
            meshes[path.stem] = {"path": str(path.resolve()), "sha256": sha(path)}
    assert set(meshes) == wanted
    inputs = {str(p): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    for name, digest in sources.items():
        assert sha(name) == digest
    write(HERE / "gallery_preparation.json", {"classification": "Actual n11/G4/n12 diagnostic on six fixed Objaverse UIDs; no formal G3 or 1000-sample claim",
          "gallery_uids": ids, "parent_checkpoint": parent["sha256"], "sources": sources,
          "inputs": inputs, "raw_glbs": meshes,
          "text_image_rule": "Original attrs_v1 frozen-CLIP cached text and 12-view mean, copied byte-for-byte",
          "point_rule": "Actual restored new Stage1 PointBERT re-encodes copied original 10000-point xyzrgb; cached PC embedding is not used",
          "gate_rule": "Unchanged CLI defaults: sample_size=1000; existing implementation samples min(1000,N)=6. All admitted assets encoded, no --limit.",
          "corpus_limitation": "G3 remains unimplemented and this diagnostic does not claim its corpus certification. Parent split is unchanged."})
    print("prepared six real Objaverse assets", flush=True)


def execute():
    preparation = json.loads((HERE / "gallery_preparation.json").read_text())
    def verify():
        for name, digest in (preparation["sources"] | preparation["inputs"]).items():
            assert sha(name) == digest, name
    verify()
    production = ("metafind/train/gallery_index.py", "metafind/gates/g4_gallery_freeze.py",
                  "metafind/train/stage1.py", "metafind/eval/retrieval.py",
                  "metafind/eval/run_retrieval.py", "metafind/models/ulip_backbone.py",
                  "metafind/models/fusion.py", "metafind/models/dual_tower.py",
                  "docs/graph/validation_plan.yaml")
    commands = [
        ("n11", ["metafind.train.gallery_index", "stage1", "--device", "cpu", "--stage1-ckpt-record", str(PARENT)]),
        ("g4", ["metafind.gates.g4_gallery_freeze"]),
        ("n12", ["metafind.train.gallery_index", "promote"]),
    ]
    for phase, arguments in commands:
        verify()
        cmd = ["nice", "-n", "10", PYTHON, "-m", *arguments]
        record = {"phase": phase, "command": cmd, "environment": ENV,
                  "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "driver_sha256": sha(__file__),
                  "production_sources": {name: sha(REPO / name) for name in production}}
        write(HERE / f"gallery_{phase}_started.json", record)
        started = time.monotonic()
        with (HERE / f"gallery_{phase}.log").open("x") as log:
            result = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, **ENV), stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, seconds=time.monotonic()-started,
                      source_changed=[name for name in production if sha(REPO/name) != record["production_sources"][name]])
        write(HERE / f"gallery_{phase}_result.json", record)
        print(json.dumps({k: record[k] for k in ("phase", "exit_code", "seconds", "source_changed")}), flush=True)
        assert result.returncode == 0 and not record["source_changed"], record
        verify()
    verify_promoted()


def verify_promoted():
    preparation = json.loads((HERE / "gallery_preparation.json").read_text())
    for name, digest in (preparation["sources"] | preparation["inputs"]).items():
        assert sha(name) == digest, name
    os.environ.update(ENV)
    from metafind.train.gallery_index import load_promoted_index_for_checkpoint
    import yaml
    gate = yaml.safe_load((DATA / "outputs/logs/gates/G4_gallery_freeze.yaml").read_text())
    assert gate["verdict"] == "PASS" and gate["is_terminal"] is True
    assert gate["observed"]["self_retrieval"]["sample_size"] == 6
    rec, ids, vectors = load_promoted_index_for_checkpoint(preparation["parent_checkpoint"])
    assert set(ids) == set(preparation["gallery_uids"]) and vectors.shape == (6, 1280)
    assert rec["sha256"] == gate["index_sha256"] and sha(rec["gate_record_uri"]) == rec["gate_record_sha256"]
    write(HERE / "gallery_result.json", {"classification": preparation["classification"],
          "registry": str(DATA / "outputs/gallery_index.json"), "index_record": rec,
          "gate_sample_size": 6, "verified_promoted_shape": list(vectors.shape),
          "formal_G3_claim": False, "formal_1000_sample_claim": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "execute", "verify_promoted"))
    args = parser.parse_args()
    globals()[args.phase]()
