"""G4_gallery_freeze, and the n12 promotion that must read its record.

Every fixture here is synthetic and small. Building a real 45,692-asset index to
test a gate would test the encoder, take hours, and still not exercise the case
that matters -- a BROKEN index, which the pipeline cannot produce on demand.

The load-bearing test is ``test_legal_ties_pass``. Two assets with identical
embeddings are a correct index, and ``argmax`` returns either of them, so an
implementation written as ``argmax(sim) == target`` rejects a correct index
about half the time. Both spec files carry a ``[CORRECTED]`` note saying exactly
this; this file is where the correction is enforced rather than described.

MEASURED before these tests were written (480 trials over 8 shapes, including
d=1280 / ng=4,569): bit-identical gallery rows tie EXACTLY through
``score_streaming``, 0 spurious failures. That is not a safe assumption in
general -- ``rank_of_target``'s docstring records one-ULP disagreement between
identical columns of a single GEMM -- so it was measured rather than assumed
before a tie test was allowed to depend on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("METAFIND_DATA", str(Path(__file__).resolve().parents[1] / "data"))

from metafind.gates import g4_gallery_freeze as g4      # noqa: E402
from metafind.train import gallery_index as gi          # noqa: E402

DIM = 8
N = 6


def _weights(path: Path, dim: int = DIM) -> str:
    """A checkpoint carrying only what the gate reads: the gallery width."""
    import torch

    torch.save({"tower_trainable_state":
                {"gallery.fusion.mask_tokens": torch.zeros(3, dim)}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, embeddings: np.ndarray, ids: list[str],
             admitted: list[str] | None = None, dim: int = DIM) -> dict:
    """A complete, internally consistent n11 output plus everything G4 reads."""
    admitted = sorted(admitted if admitted is not None else set(ids))
    weights = tmp_path / "stage1.pt"
    ckpt_sha = _weights(weights, dim)
    ckpt_record_path = tmp_path / "stage1_ckpt.json"
    ckpt_record_path.write_text(json.dumps({"uri": str(weights), "sha256": ckpt_sha}))

    record = gi.build_index(embeddings, ids, tmp_path / "index.npz")
    record["gallery_encoder_sha256"] = "e" * 64
    record["stage1_checkpoint_sha256"] = ckpt_sha
    record["stage1_ckpt_record"] = str(ckpt_record_path)

    staging = tmp_path / "gallery_index_staging.json"
    staging.write_text(json.dumps({ckpt_sha: record}))

    splits = tmp_path / "splits.json"
    cut = len(admitted) // 2
    splits.write_text(json.dumps({
        "object": {"train": admitted[:cut], "test": admitted[cut:],
                   "dev_train": [], "dev_val": []},
        "split_seed": 20260816,
        "admitted_total": len(admitted),
    }))
    return {"staging": staging, "splits": splits, "record": tmp_path / "g4.yaml",
            "ckpt_sha": ckpt_sha, "ckpt_record": ckpt_record_path,
            "index": tmp_path / "index.npz", "staging_record": record}


def _healthy(n: int = N, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Distinct vectors: every row is its own nearest neighbour."""
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def _ids(n: int = N) -> list[str]:
    return [f"asset{i:02d}" for i in range(n)]


def _run(fx, **kw) -> tuple[int, dict]:
    rc = g4.run(staging_path=fx["staging"], splits_path=fx["splits"],
                record_path=fx["record"], **kw)
    return rc, yaml.safe_load(fx["record"].read_text())


# --------------------------------------------------------------- happy path

def test_a_healthy_index_passes(tmp_path):
    fx = _fixture(tmp_path, _healthy(), _ids())
    rc, rec = _run(fx)
    assert rc == g4.PASS, rec["observed"]["failures"]
    assert rec["verdict"] == "PASS"
    assert rec["is_terminal"] is True
    assert rec["gate_class"] == "G-CONTAM"
    assert rec["record_kind"] == "gate"
    assert rec["observed"]["self_retrieval"]["n_failed"] == 0
    assert rec["observed"]["self_retrieval"]["sample_size"] == N
    assert rec["observed"]["ids_are_sorted"] is True
    # Recorded, not a pass condition: a healthy index still says what its
    # effective rank was, so a later collapsed one is comparable to it.
    assert rec["observed"]["embedding_health"]["effective_rank"] > 1.0
    assert "effective_rank_centred" in rec["observed"]["embedding_health"]
    assert "collapse-blindness" in rec["observed"]["collapse_note"]


def test_the_record_carries_every_declared_field(tmp_path):
    """The declared schema, not a subset of it that happened to be written."""
    fx = _fixture(tmp_path, _healthy(), _ids())
    _run(fx)
    rec = yaml.safe_load(fx["record"].read_text())
    for field in ("gate_id", "gate_class", "scope", "record_kind", "criterion",
                  "inputs", "observed", "verdict", "rc", "timestamp",
                  "code_revision", "is_terminal", "index_uri", "index_sha256",
                  "staging_record_sha256", "stage1_checkpoint_sha256",
                  "gallery_encoder_sha256", "expected_uid_set_sha256",
                  "sample_uid_sequence_sha256", "runtime_source_sha256"):
        assert field in rec, f"the gate record has no {field!r}"
    assert rec["criterion"], "the criterion was not quoted from the spec"
    assert rec["index_sha256"] == hashlib.sha256(
        fx["index"].read_bytes()).hexdigest()


def test_the_sample_is_fixed_and_its_sequence_is_digested(tmp_path):
    """A seed names a recipe; the digest names what was actually drawn."""
    fx = _fixture(tmp_path, _healthy(n=40), _ids(40))
    _, first = _run(fx, sample_size=10)
    _, second = _run(fx, sample_size=10)
    a = first["observed"]["self_retrieval"]
    b = second["observed"]["self_retrieval"]
    assert a["sample_uid_sequence_sha256"] == b["sample_uid_sequence_sha256"]
    assert a["sample_size"] == 10
    assert first["sample_uid_sequence_sha256"] == a["sample_uid_sequence_sha256"]
    # A different sample must not report the same digest, or the field would be
    # decoration rather than evidence.
    _, third = _run(fx, sample_size=20)
    assert third["observed"]["self_retrieval"]["sample_uid_sequence_sha256"] != \
        a["sample_uid_sequence_sha256"]


# ------------------------------------------------------------------- ties

def test_legal_ties_pass(tmp_path):
    """TWO IDENTICAL EMBEDDINGS ARE A CORRECT INDEX.

    `argmax` returns either id, so `argmax(sim) == target` rejects this index
    for one of the two tied assets. The criterion is `sim(target) == max(sim)`
    with the target in the argmax tie set, and both tied rows satisfy it.
    """
    emb = _healthy()
    emb[3] = emb[1]                       # asset01 and asset03 are identical
    fx = _fixture(tmp_path, emb, _ids())
    rc, rec = _run(fx)
    assert rc == g4.PASS, rec["observed"]["failures"]
    sr = rec["observed"]["self_retrieval"]
    assert sr["n_failed"] == 0
    # The tie is real, not a fixture that quietly failed to create one.
    assert sr["max_tie_count"] >= 1
    assert sr["n_with_ties"] >= 2
    # And a naive argmax implementation would have disagreed on one of them.
    gallery = np.asarray(emb, dtype=np.float64)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    argmax = (gallery @ gallery.T).argmax(axis=1)
    assert (argmax != np.arange(len(emb))).any(), \
        "this fixture no longer produces the tie the test exists for"


# ------------------------------------------------------------------ FAILs

def test_wrong_index_sha_fails(tmp_path):
    fx = _fixture(tmp_path, _healthy(), _ids())
    staging = json.loads(fx["staging"].read_text())
    key = fx["ckpt_sha"]
    staging[key]["sha256"] = "0" * 64
    fx["staging"].write_text(json.dumps(staging))
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    assert rec["observed"]["index_sha256_match"] is False
    assert any("index bytes hash to" in f for f in rec["observed"]["failures"])


def test_wrong_count_fails(tmp_path):
    """The index holds fewer assets than the split admits."""
    fx = _fixture(tmp_path, _healthy(n=4), _ids(4), admitted=_ids(N))
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    obs = rec["observed"]
    assert obs["n_index_ids"] == 4 and obs["n_admitted_unique"] == N
    assert obs["n_missing_from_index"] == 2
    assert any("!= len(admitted)" in f for f in obs["failures"])


def test_extra_asset_not_admitted_fails(tmp_path):
    """The other direction: the index holds something the split does not."""
    ids = _ids(N - 1) + ["intruder"]
    fx = _fixture(tmp_path, _healthy(), ids, admitted=_ids(N))
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    assert rec["observed"]["n_extra_in_index"] == 1
    assert rec["observed"]["extra_examples"] == ["intruder"]
    assert rec["observed"]["n_missing_from_index"] == 1


def test_duplicate_asset_ids_fail(tmp_path):
    ids = _ids()
    ids[4] = ids[0]
    fx = _fixture(tmp_path, _healthy(), ids, admitted=_ids())
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    assert rec["observed"]["n_duplicate_index_ids"] == 1
    assert any("more than once" in f for f in rec["observed"]["failures"])


@pytest.mark.parametrize("bad,label", [(np.nan, "NaN"), (np.inf, "Inf")])
def test_non_finite_values_fail(tmp_path, bad, label):
    emb = _healthy()
    emb[2, 3] = bad
    fx = _fixture(tmp_path, emb, _ids())
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    key = "n_nan_rows" if label == "NaN" else "n_inf_rows"
    assert rec["observed"][key] == 1
    assert any(label in f for f in rec["observed"]["failures"])
    # Self-retrieval must not be reported at all: the scorer cannot normalise
    # these vectors, and a missing verdict is not a passing one.
    assert "self_retrieval" not in rec["observed"]
    assert "self_retrieval_not_run" in rec["observed"]


def test_zero_vector_fails(tmp_path):
    emb = _healthy()
    emb[5] = 0.0
    fx = _fixture(tmp_path, emb, _ids())
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    assert rec["observed"]["n_zero_norm_rows"] == 1
    assert rec["observed"]["zero_norm_examples"] == ["asset05"]


def test_self_retrieval_reports_a_target_that_is_not_the_maximum():
    """The comparison itself, exercised directly, because `run` cannot fail it.

    Under `run` the query IS the gallery row and every vector is unit-length, so
    the self-similarity is the maximum by construction -- see
    `test_the_declared_negative_injection_does_not_fire`. This calls the
    comparison with an UNNORMALISED gallery, where a short vector's similarity
    to itself is genuinely below its similarity to a longer one, and checks that
    the machinery reports it. An `argmax(sim) == target` implementation would
    also report it, so this test alone does not pin the tie-safety; it pins that
    detection works at all.
    """
    gallery = np.array([[1.0, 0.0], [10.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    out = g4._self_retrieval(["a", "b", "c"], gallery, seed=1,
                             sample_size=3, block=4096)
    assert out["n_failed"] == 1
    assert out["failed_examples"][0]["asset_id"] == "a"
    assert out["failed_examples"][0]["n_strictly_higher"] == 1


def test_a_failing_sample_makes_the_gate_fail(tmp_path, monkeypatch):
    """`run`'s handling of a self-retrieval failure, wired end to end."""
    fx = _fixture(tmp_path, _healthy(), _ids())
    monkeypatch.setattr(g4, "_self_retrieval", lambda *a, **k: {
        "sample_size": 6, "sample_seed": 1, "sample_uid_sequence_sha256": "x" * 64,
        "n_target_is_max": 5, "n_zero_strictly_higher": 5, "n_failed": 1,
        "n_with_ties": 0, "max_tie_count": 0,
        "failed_examples": [{"asset_id": "asset02", "target_score": 0.4,
                             "top1_score": 0.9, "n_strictly_higher": 1}]})
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    assert any("do not retrieve themselves" in f
               for f in rec["observed"]["failures"])
    assert "asset02" in rec["observed"]["failures"][0]


def test_the_declared_negative_injection_does_not_fire(tmp_path):
    """⚠ REFUTATION. L2-GALLERY-SELF's negative injection does NOT fail this gate.

    `validation_plan.yaml` L2-GALLERY-SELF declares
    `negative_injection: "shuffle the index id mapping"` and
    `expected_on_injection: "the target's score stops being the maximum and the
    index is not promoted"`. Shuffling the ids against the rows leaves the
    admitted SET, the count, the width and every vector untouched, and the
    criterion scores each gallery ROW against itself -- so the arithmetic is
    bit-identical to the unshuffled index and the gate PASSES.

    Recorded as a test rather than a comment so it cannot rot: if someone
    strengthens the criterion so this injection does fire, this test goes red
    and says which claim changed.
    """
    ids = _ids(8)
    emb = _healthy(n=8)
    shuffled = [ids[i] for i in [3, 1, 7, 0, 5, 2, 6, 4]]
    fx = _fixture(tmp_path, emb, shuffled, admitted=ids)
    rc, rec = _run(fx)
    assert rc == g4.PASS, rec["observed"]["failures"]
    assert rec["observed"]["self_retrieval"]["n_failed"] == 0
    # The shuffle is INVISIBLE to the criterion and VISIBLE in the record. That
    # split is the whole point: the gate does not invent a rule, and a reader of
    # the artifact can still see that the row order left n11's convention.
    assert rec["observed"]["ids_are_sorted"] is False
    assert "shuffle the index id mapping" in rec["observed"]["ids_sorted_note"]


def test_width_disagreeing_with_the_checkpoint_fails(tmp_path):
    fx = _fixture(tmp_path, _healthy(dim=DIM), _ids(), dim=DIM + 1)
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    assert rec["observed"]["checkpoint_embedding_dim"] == DIM + 1
    assert rec["observed"]["index_dim"] == DIM
    assert any("gallery width" in f for f in rec["observed"]["failures"])


def test_checkpoint_sha_disagreeing_with_the_staging_record_fails(tmp_path):
    fx = _fixture(tmp_path, _healthy(), _ids())
    staging = json.loads(fx["staging"].read_text())
    rec_in = staging.pop(fx["ckpt_sha"])
    rec_in["stage1_checkpoint_sha256"] = "9" * 64
    staging["9" * 64] = rec_in
    fx["staging"].write_text(json.dumps(staging))
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    assert any("but" in f and "records" in f for f in rec["observed"]["failures"])


def test_a_checkpoint_record_that_lies_about_its_weights_fails(tmp_path):
    fx = _fixture(tmp_path, _healthy(), _ids())
    rec_json = json.loads(fx["ckpt_record"].read_text())
    rec_json["sha256"] = "7" * 64
    fx["ckpt_record"].write_text(json.dumps(rec_json))
    rc, rec = _run(fx)
    assert rc == g4.FAIL
    assert rec["observed"]["checkpoint_weights_sha256_verified"] is False


# --------------------------------------------------- BLOCKED_EVIDENCE (rc 3)

def test_missing_staging_index_is_blocked_not_passed(tmp_path):
    fx = _fixture(tmp_path, _healthy(), _ids())
    fx["staging"].unlink()
    rc, rec = _run(fx)
    assert rc == g4.BLOCKED_EVIDENCE == 3
    assert rec["verdict"] == "BLOCKED_EVIDENCE"
    assert rec["is_terminal"] is True
    assert "not found" in rec["observed"]["blocked_reason"]


def test_missing_checkpoint_record_is_blocked(tmp_path):
    fx = _fixture(tmp_path, _healthy(), _ids())
    fx["ckpt_record"].unlink()
    rc, rec = _run(fx)
    assert rc == g4.BLOCKED_EVIDENCE
    assert "stage1_ckpt.json" in rec["observed"]["blocked_reason"]


def test_unreadable_npz_is_blocked(tmp_path):
    fx = _fixture(tmp_path, _healthy(), _ids())
    # Corrupt the archive but keep the recorded digest honest, so the failure
    # is "cannot read" and not "does not match".
    fx["index"].write_bytes(b"not a zip file at all")
    staging = json.loads(fx["staging"].read_text())
    staging[fx["ckpt_sha"]]["sha256"] = hashlib.sha256(
        fx["index"].read_bytes()).hexdigest()
    fx["staging"].write_text(json.dumps(staging))
    rc, rec = _run(fx)
    assert rc == g4.BLOCKED_EVIDENCE
    assert "unreadable index" in rec["observed"]["blocked_reason"]


def test_missing_index_file_is_blocked(tmp_path):
    fx = _fixture(tmp_path, _healthy(), _ids())
    fx["index"].unlink()
    rc, rec = _run(fx)
    assert rc == g4.BLOCKED_EVIDENCE


def test_an_ambiguous_staging_map_is_blocked(tmp_path):
    """Two staged indices, one gate record: the gate cannot say which it means."""
    fx = _fixture(tmp_path, _healthy(), _ids())
    staging = json.loads(fx["staging"].read_text())
    staging["f" * 64] = dict(staging[fx["ckpt_sha"]])
    fx["staging"].write_text(json.dumps(staging))
    rc, rec = _run(fx)
    assert rc == g4.BLOCKED_EVIDENCE
    assert "2 entries" in rec["observed"]["blocked_reason"]


def test_a_split_with_no_seed_is_blocked(tmp_path):
    """The fixed sample is defined by the split's seed. No seed, no definition."""
    fx = _fixture(tmp_path, _healthy(), _ids())
    splits = json.loads(fx["splits"].read_text())
    del splits["split_seed"]
    fx["splits"].write_text(json.dumps(splits))
    rc, rec = _run(fx)
    assert rc == g4.BLOCKED_EVIDENCE
    assert "split_seed" in rec["observed"]["blocked_reason"]


def test_an_internal_error_still_leaves_a_terminal_record(tmp_path, monkeypatch):
    """A gate that dies without a record cannot be told from one that never ran."""
    fx = _fixture(tmp_path, _healthy(), _ids())
    monkeypatch.setattr(g4, "_self_retrieval",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rc, rec = _run(fx)
    assert rc == g4.BLOCKED_EVIDENCE
    assert rec["is_terminal"] is True
    assert "internal error in the gate itself" in rec["observed"]["blocked_reason"]
    assert "RuntimeError: boom" in rec["observed"]["blocked_reason"]


def test_a_rerun_pass_does_not_destroy_the_earlier_fail(tmp_path):
    """The history is append-only. A re-run may not erase what it replaces.

    `graph_spec.yaml` declares gate_records as a persistent append-only
    `list[gate_record]`; `validation_plan.yaml` gives one `record_path` holding
    one record. Writing both satisfies both. What neither permits is the
    behaviour they produced together: overwriting `record_path` on a re-run made
    a FAIL vanish the moment it was fixed, and the fixed run is the one nobody
    needs a record of.
    """
    emb = _healthy()
    emb[5] = 0.0                                  # a zero vector -> FAIL
    fx = _fixture(tmp_path, emb, _ids())
    assert _run(fx)[0] == g4.FAIL

    # repair the index in place and re-stage it, then re-run: PASS
    rec = gi.build_index(_healthy(), _ids(), fx["index"])
    staging = json.loads(fx["staging"].read_text())
    staging[fx["ckpt_sha"]]["sha256"] = rec["sha256"]
    fx["staging"].write_text(json.dumps(staging))
    rc, current = _run(fx)
    assert rc == g4.PASS, current["observed"]["failures"]

    assert current["verdict"] == "PASS"           # record_path: the CURRENT one
    history = yaml.safe_load(g4.history_path(fx["record"]).read_text())
    assert [r["verdict"] for r in history] == ["FAIL", "PASS"]   # both, in order
    assert history[-1]["rc"] == 0 and history[0]["rc"] == 2
    # The history entry is the whole record, not a summary of it.
    assert history[-1] == current


def test_a_blocked_run_is_in_the_history_too(tmp_path):
    """PASS, FAIL and BLOCKED alike -- a gate that could not judge is evidence."""
    fx = _fixture(tmp_path, _healthy(), _ids())
    fx["staging"].unlink()
    assert _run(fx)[0] == g4.BLOCKED_EVIDENCE
    history = yaml.safe_load(g4.history_path(fx["record"]).read_text())
    assert [r["verdict"] for r in history] == ["BLOCKED_EVIDENCE"]


def test_an_unparseable_history_is_left_alone(tmp_path):
    """Refusing to clobber is the point; it must not change the verdict either."""
    fx = _fixture(tmp_path, _healthy(), _ids())
    hist = g4.history_path(fx["record"])
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text("this is not a list\n")
    rc, rec = _run(fx)
    assert rc == g4.PASS                          # verdict unaffected
    assert rec["verdict"] == "PASS"               # terminal record still written
    assert hist.read_text() == "this is not a list\n"


def test_the_rc_contract_matches_the_spec():
    """The gate implements the contract validation_plan.yaml declares."""
    assert g4.spec()["rc_contract"] == {"PASS": 0, "FAIL": 2,
                                        "BLOCKED_EVIDENCE": 3, "INVALIDATED": 4}
    assert (g4.PASS, g4.FAIL, g4.BLOCKED_EVIDENCE, g4.INVALIDATED) == (0, 2, 3, 4)


def test_a_fail_does_not_touch_the_staging_index(tmp_path):
    """[on_fail] Do NOT promote, and do not edit the evidence."""
    emb = _healthy()
    emb[5] = 0.0
    fx = _fixture(tmp_path, emb, _ids())
    before = (fx["index"].read_bytes(), fx["staging"].read_bytes())
    rc, _ = _run(fx)
    assert rc == g4.FAIL
    assert (fx["index"].read_bytes(), fx["staging"].read_bytes()) == before


# ------------------------------------------------------------- n12 promotion

@pytest.fixture
def promotable(tmp_path, monkeypatch):
    """A staged index that G4 has already PASSED, with n12 pointed at tmp."""
    fx = _fixture(tmp_path, _healthy(), _ids())
    monkeypatch.setattr(gi, "STAGING_PATH", fx["staging"])
    monkeypatch.setattr(gi, "PROMOTED_PATH", tmp_path / "gallery_index.json")
    rc = g4.run(staging_path=fx["staging"], splits_path=fx["splits"],
                record_path=fx["record"])
    assert rc == g4.PASS
    fx["promoted"] = tmp_path / "gallery_index.json"
    return fx


def test_promotion_accepts_a_genuine_pass(promotable):
    assert gi.promote(promotable["record"]) == 0
    promoted = json.loads(promotable["promoted"].read_text())
    entry = promoted[promotable["ckpt_sha"]]
    assert entry["sha256"] == hashlib.sha256(
        promotable["index"].read_bytes()).hexdigest()
    assert entry["gate_record_uri"] == str(promotable["record"])
    assert entry["gate_record_sha256"] == hashlib.sha256(
        promotable["record"].read_bytes()).hexdigest()
    assert "promoted_at" in entry


def test_promotion_refuses_a_missing_gate_record(promotable):
    promotable["record"].unlink()
    assert gi.promote(promotable["record"]) == 3
    assert not promotable["promoted"].exists()


def test_promotion_refuses_a_fail_record(promotable):
    rec = yaml.safe_load(promotable["record"].read_text())
    rec["verdict"], rec["rc"] = "FAIL", 2
    promotable["record"].write_text(yaml.safe_dump(rec))
    assert gi.promote(promotable["record"]) == 3
    assert not promotable["promoted"].exists()


def test_promotion_refuses_a_non_terminal_record(promotable):
    rec = yaml.safe_load(promotable["record"].read_text())
    rec["is_terminal"] = False
    promotable["record"].write_text(yaml.safe_dump(rec))
    assert gi.promote(promotable["record"]) == 3
    assert not promotable["promoted"].exists()


def test_promotion_refuses_a_record_from_another_gate(promotable):
    rec = yaml.safe_load(promotable["record"].read_text())
    rec["gate_id"] = "G3_object_corpus"
    promotable["record"].write_text(yaml.safe_dump(rec))
    assert gi.promote(promotable["record"]) == 3


def test_promotion_refuses_when_the_verified_bytes_changed(promotable, tmp_path):
    """The index was rebuilt after G4 saw it. Same path, different artifact."""
    gi.build_index(_healthy(seed=99), _ids(), promotable["index"])
    staging = json.loads(promotable["staging"].read_text())
    staging[promotable["ckpt_sha"]]["sha256"] = hashlib.sha256(
        promotable["index"].read_bytes()).hexdigest()
    promotable["staging"].write_text(json.dumps(staging))
    assert gi.promote(promotable["record"]) == 2
    assert not promotable["promoted"].exists()


def test_promotion_refuses_when_the_staging_record_changed(promotable):
    """The .npz is untouched; only the record around it moved."""
    staging = json.loads(promotable["staging"].read_text())
    staging[promotable["ckpt_sha"]]["count"] = 999
    promotable["staging"].write_text(json.dumps(staging))
    assert gi.promote(promotable["record"]) == 2
    assert not promotable["promoted"].exists()


def test_promotion_refuses_a_checkpoint_sha_mismatch(promotable):
    rec = yaml.safe_load(promotable["record"].read_text())
    rec["stage1_checkpoint_sha256"] = "1" * 64
    rec["staging_record_sha256"] = hashlib.sha256(
        promotable["staging"].read_bytes()).hexdigest()
    promotable["record"].write_text(yaml.safe_dump(rec))
    assert gi.promote(promotable["record"]) == 2
    assert not promotable["promoted"].exists()


def test_promotion_keeps_write_once(promotable):
    """A second, DIFFERENT index for the same checkpoint is refused."""
    assert gi.promote(promotable["record"]) == 0
    promoted = json.loads(promotable["promoted"].read_text())
    promoted[promotable["ckpt_sha"]]["sha256"] = "3" * 64
    promotable["promoted"].write_text(json.dumps(promoted))
    assert gi.promote(promotable["record"]) == 2


def test_the_cli_flag_can_no_longer_stand_in_for_the_gate(monkeypatch):
    """`--gate-passed` is gone, not ignored: the old command must fail loudly.

    An argument that is accepted and disregarded still reads as "the gate is
    asserted here", so argparse has to reject it rather than shrug.
    """
    import inspect

    assert "gate_passed" not in inspect.signature(gi.promote).parameters
    monkeypatch.setattr(sys, "argv",
                        ["gallery_index", "promote", "--gate-passed"])
    with pytest.raises(SystemExit) as exc:
        gi.main()
    assert exc.value.code != 0


# --------------------------------------------------- the shared index loader

def test_the_loader_reverifies_bytes_on_every_call(promotable):
    assert gi.promote(promotable["record"]) == 0
    rec, ids, emb = gi.load_promoted_index_for_checkpoint(
        promotable["ckpt_sha"], promotable["promoted"])
    assert ids == _ids() and emb.shape == (N, DIM) and emb.dtype == np.float32
    assert rec["stage1_checkpoint_sha256"] == promotable["ckpt_sha"]
    # Same process, same registry, tampered bytes: the second call must refuse.
    promotable["index"].write_bytes(promotable["index"].read_bytes() + b"\0")
    with pytest.raises(ValueError, match="hashes to"):
        gi.load_promoted_index_for_checkpoint(
            promotable["ckpt_sha"], promotable["promoted"])


def test_the_loader_refuses_an_unknown_checkpoint(promotable):
    assert gi.promote(promotable["record"]) == 0
    with pytest.raises(KeyError):
        gi.load_promoted_index_for_checkpoint("0" * 64, promotable["promoted"])


def test_the_loader_refuses_a_missing_registry(tmp_path):
    with pytest.raises(FileNotFoundError):
        gi.load_promoted_index_for_checkpoint("0" * 64, tmp_path / "nope.json")
