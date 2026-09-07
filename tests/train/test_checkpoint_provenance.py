"""A checkpoint must answer, from itself, which run produced it.

[CODEX MAJOR 3 & 4, 2026-08-30] Two failures these pin, both of which pass every
other test in the suite:

* the weights and their sidecar are separate files, and until now only the
  sidecar knew anything -- copy, rename or lose one and the other is orphaned;
* a sidecar is a CLAIM about a file, and nothing downstream checked it, so a
  record naming checkpoint A's provenance could be pointed at checkpoint B's
  bytes and every artifact built from it inherited the wrong identity.

None of this needs a GPU or the real corpus. The metadata path is separable from
the training path, so it is tested with a toy state dict.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from metafind import paths  # noqa: E402
from metafind.train import gallery_index, stage1  # noqa: E402


class FakeBackbone:
    def __init__(self) -> None:
        self.model = nn.Linear(4, 2)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


class FakeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(2.659))


def _training(**over):
    t = {"hyperparameter_config_hash": "b" * 64,
         "train_scope": "point_encoder_and_fuser",
         "_arm_config_hash": "a" * 64,
         "_arm_config": {"learning_rate": 7.5e-4, "epochs": 5},
         "_repeat_index": 1, "_argv": ["--lr", "0.00075", "--seed", "20260830"],
         "_hardware": {"device": "cpu"}, "_lr_horizon": 5, "_epoch_count": 5,
         "_preload": True, "_num_workers": 0,
         "_pools": stage1.pool_provenance(["u1", "u2"], ["v1"]),
         "_initializers": {"open_clip": {"model": "ViT-bigG-14"}}}
    return {**t, **over}


def _save(tmp_path, monkeypatch, seed=20260830, **over):
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    rp = stage1.resolve_run_paths(None)
    b, m, l = FakeBackbone(), nn.Linear(2, 3), FakeLoss()
    record = stage1.save_checkpoint(
        b, m, l, {"sha256": "b" * 64}, {"actual_clip_train_scope": "frozen"},
        _training(**over), seed, epoch=3, rp=rp)
    return rp, record


def test_the_weights_carry_their_own_provenance(tmp_path, monkeypatch):
    """Delete the sidecar and the `.pt` must still say what produced it."""
    rp, record = _save(tmp_path, monkeypatch)
    rp.latest_record.unlink()

    meta = torch.load(rp.latest_checkpoint, weights_only=False)["metadata"]
    assert meta["seed"] == 20260830
    assert meta["repeat_index"] == 1
    assert meta["arm_config_hash"] == "a" * 64
    assert meta["arm_config"]["learning_rate"] == 7.5e-4
    assert "--lr" in meta["argv"] and "0.00075" in meta["argv"]
    assert meta["inputs"]["n_train"] == 2
    assert meta["initializers"]["open_clip"]["model"] == "ViT-bigG-14"


def test_the_embedded_metadata_does_not_contain_its_own_digest(tmp_path, monkeypatch):
    """A file cannot contain its own sha256, and pretending it can is a lie."""
    rp, _ = _save(tmp_path, monkeypatch)
    meta = torch.load(rp.latest_checkpoint, weights_only=False)["metadata"]
    assert "sha256" not in meta and "uri" not in meta


def test_the_sidecar_and_the_weights_agree(tmp_path, monkeypatch):
    import hashlib
    rp, record = _save(tmp_path, monkeypatch)
    on_disk = hashlib.sha256(rp.latest_checkpoint.read_bytes()).hexdigest()
    assert record["sha256"] == on_disk
    meta = torch.load(rp.latest_checkpoint, weights_only=False)["metadata"]
    assert meta["run_id"] == record["run_id"]
    assert meta["arm_config_hash"] == record["arm_config_hash"]


@pytest.mark.parametrize("field,value", [("seed", 20260816), ("repeat_index", 0)])
def test_the_run_variables_reach_the_record(tmp_path, monkeypatch, field, value):
    """`--seed` and `--repeat-index` are what make two runs a declared pair."""
    over = {"_repeat_index": value} if field == "repeat_index" else {}
    _, record = _save(tmp_path, monkeypatch,
                      seed=value if field == "seed" else 20260830, **over)
    assert record[field] == value


# ------------------------------------------------- downstream integrity

def test_a_record_pointed_at_the_wrong_bytes_is_refused(tmp_path, monkeypatch):
    """[CODEX MAJOR 3] The claim was never checked.

    `--out-dir` makes several checkpoints exist at once, so a record and a
    weights file drifting apart stops being theoretical. An index built from
    the wrong bytes would carry the other checkpoint's provenance and nothing
    downstream would ever disagree.
    """
    rp, _ = _save(tmp_path, monkeypatch)
    rp.latest_checkpoint.write_bytes(b"a different checkpoint entirely")
    with pytest.raises(ValueError, match="hashes to"):
        gallery_index.load_checkpoint_record(rp.latest_record)


def test_a_record_whose_weights_are_gone_is_refused(tmp_path, monkeypatch):
    rp, _ = _save(tmp_path, monkeypatch)
    rp.latest_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="separated"):
        gallery_index.load_checkpoint_record(rp.latest_record)


def test_a_matching_record_is_accepted(tmp_path, monkeypatch):
    """The guard must not simply refuse everything."""
    rp, record = _save(tmp_path, monkeypatch)
    assert gallery_index.load_checkpoint_record(rp.latest_record)["sha256"] == \
        record["sha256"]


def test_the_open_clip_record_pairs_one_revision_with_its_own_blob():
    """[CODEX MINOR 2026-08-30] `glob("snapshots/*/*")` took whichever blob it
    found first.

    With more than one cached revision that pairs `refs/main`'s revision with
    another download's weights -- a provenance record whose two halves describe
    different files, which is worse than no record because it looks complete.
    """
    from metafind.train.stage1 import _open_clip_weight_identity
    got = _open_clip_weight_identity()
    if "hf_revision" not in got:
        pytest.skip("no HF cache on this host")
    snapshot = Path(got["hf_cache"]) / "snapshots" / got["hf_revision"]
    assert snapshot.is_dir(), "the recorded revision must have a snapshot"
    weights = snapshot / got["weight_file"]
    assert weights.resolve().name == got["weight_blob_sha256"], (
        "the recorded blob must be the one THIS revision's snapshot points at")
