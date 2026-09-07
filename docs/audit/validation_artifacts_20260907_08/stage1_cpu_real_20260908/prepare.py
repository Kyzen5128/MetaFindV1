"""Prepare isolated diagnostic inputs; never write the source corpus."""
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE / "data"
SOURCE = Path("/home/kyzen/metafind/metafind_data_attrs")
os.environ["METAFIND_DATA"] = str(ROOT)
os.environ["METAFIND_TEXT_TEMPLATE"] = "attrs_v1"
from metafind.models.resolve_stage1 import build_hyperparameters


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    ROOT.mkdir(exist_ok=False)
    (ROOT / "models").symlink_to(SOURCE / "models", target_is_directory=True)
    out = ROOT / "outputs"
    out.mkdir()
    originals = {}
    splits_path = SOURCE / "outputs/splits.json"
    old_splits = json.loads(splits_path.read_text())
    train = old_splits["object"]["dev_train"][:2]
    val = old_splits["object"]["dev_val"][:2]
    assert len(set(train + val)) == 4
    originals[str(splits_path)] = digest(splits_path)
    write(out / "splits.json", {
        "classification": "DIAGNOSTIC: two training and two disjoint selection assets, not research scores",
        "source": str(splits_path), "source_sha256": digest(splits_path),
        "object": {"train": train, "dev_train": train, "val": val,
                   "dev_val": val, "holdout": val, "test": [], "train_val": train + val},
    })
    for directory in ("embeddings", "pointclouds", "annotations"):
        (out / directory).mkdir()
        for uid in train + val:
            suffixes = (".json",) if directory == "annotations" else (".json", ".npz")
            for suffix in suffixes:
                src = SOURCE / "outputs" / directory / (uid + suffix)
                originals[str(src)] = digest(src)
                shutil.copy2(src, out / directory / src.name)
    for name in ("stage1_encoding_protocol.json", "stage1_protocol.json", "stage1_hyperparameters.json"):
        src = SOURCE / "outputs" / name
        originals[str(src)] = digest(src)
        shutil.copy2(src, out / name)
    hp_path = out / "stage1_hyperparameters.json"
    before = json.loads(hp_path.read_text())
    values = dict(before["values"], batch_size=2, epochs=1)
    hp = build_hyperparameters("2026-09-08 authorized isolated CPU execution diagnostic; not paper recipe", values)
    write(hp_path, hp)
    train_path = out / "stage1_protocol.json"
    protocol = json.loads(train_path.read_text())
    protocol["hyperparameter_config_hash"] = hp["sha256"]
    protocol["diagnostic_parent_protocol_sha256"] = originals[str(SOURCE / "outputs/stage1_protocol.json")]
    write(train_path, protocol)
    write(HERE / "preparation.json", {
        "classification": "OBSERVED DATA / IMPLEMENTATION CHOICE: isolated genuine CPU trainer smoke; old attrs_v1 corpus, same_record, batch 2, one epoch. Not paper reproduction or meaningful retrieval quality.",
        "train_uids": train, "selection_uids": val,
        "overrides": {"batch_size": {"from": before["values"]["batch_size"], "to": 2},
                      "epochs": {"from": before["values"]["epochs"], "to": 1}},
        "source_sha256": originals,
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "model_location": str((ROOT / "models").resolve()),
    })
    print(json.dumps({"root": str(ROOT), "train": train, "selection": val, "hyperparameter_hash": hp["sha256"]}))


if __name__ == "__main__":
    main()
