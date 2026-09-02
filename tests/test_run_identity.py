"""What identifies a run: the arm it belongs to, the code that ran, where it wrote.

Every mechanism below exists because a specific claim made on 2026-08-29 or
2026-08-30 turned out to be false and nothing recorded could have contradicted
it. Each test names the claim it refutes.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from metafind import paths, runlog
from metafind.models.stage1_config import UnsupportedProtocol
from metafind.train import stage1

PY = sys.executable


def _values(**over):
    v = {"optimizer": "adamw", "learning_rate": 5e-4, "weight_decay": 0.1,
         "scheduler": "cosine", "batch_size": 64, "epochs": 5, "max_epochs": 250,
         "p_mask": 0.30, "init_temperature": 0.5, "learnable_temperature": False,
         "max_logit_scale": 100.0, "betas": [0.9, 0.98], "eps": 1e-8,
         "warmup_epochs": 1, "lr_start": 1e-6, "lr_end": 1e-5, "seed": 20260816}
    return {**v, **over}


def _training(**over):
    t = {"_epoch_count": 5, "_lr_horizon": 250, "fusion": "transformer",
         "tower_sharing": "shared_backbone_separate_fusion",
         "similarity": "cosine", "train_scope": "point_encoder_and_fuser"}
    return {**t, **over}


def _encoding(**over):
    e = {"image_aggregation": "mean",
         "missing_modality_representation": "learned_token",
         "actual_clip_train_scope": "frozen"}
    return {**e, **over}


def _arm(values=None, training=None, encoding=None, phase="dev"):
    return stage1.arm_config_hash(values or _values(), training or _training(),
                                  encoding or _encoding(), phase)


# ---------------------------------------------------------------- arm identity

def test_absent_override_and_override_to_the_same_value_are_one_arm():
    """`--lr 5e-4` and no `--lr` train the same model.

    The rejected design hashed the override PATCH, which would have split these
    into two experiments and made the sweep's baseline arm incomparable with
    every run that came before the flag existed.
    """
    assert _arm()[0] == _arm(_values(learning_rate=5e-4))[0]


def test_seed_does_not_change_the_arm():
    """Two seeds are repeats of ONE treatment [R-33].

    Folding the seed in would give every repeat its own arm hash, and the paired
    differences the stopping rule is built on would have nothing to pair.
    """
    assert _arm(_values(seed=1))[0] == _arm(_values(seed=2))[0]
    assert "seed" not in _arm()[1]


@pytest.mark.parametrize("field,value", [
    ("learning_rate", 1e-3), ("batch_size", 32), ("p_mask", 0.5),
    # [CODEX MAJOR 2026-08-30] The five below were NOT in the first version's
    # enumerated ARM_FIELDS. Two runs differing only in weight_decay, or in the
    # optimizer betas, or in the warmup, landed on one arm hash and would have
    # been reported as the same experiment.
    ("weight_decay", 0.2), ("betas", [0.9, 0.999]), ("warmup_epochs", 3),
    ("eps", 1e-6), ("lr_start", 1e-5), ("lr_end", 1e-4),
    # `scheduler` was here, asserting only that the hash moved. [CODEX BLOCKER 1
    # 2026-08-30] named that as the wrong assertion: the trainer calls
    # `cosine_schedule` unconditionally, so `scheduler: linear` changed the
    # identity and then annealed on a cosine anyway. It is now REFUSED, and
    # `test_a_declared_value_the_trainer_cannot_honour_is_refused` is where it
    # belongs.
])
def test_a_changed_hyperparameter_changes_the_arm(field, value):
    assert _arm()[0] != _arm(_values(**{field: value}))[0]


@pytest.mark.parametrize("field,value", [
    ("_epoch_count", 10), ("_lr_horizon", 25),
    ("fusion", "mlp"), ("tower_sharing", "fully_shared"),
    # `train_scope: fuser_only` was here, asserting only that the hash moved.
    # [CODEX BLOCKER 2026-08-30] `stage1.py:1324` hardcodes
    # `point_encoder_and_fuser`, so that assertion blessed a recorded recipe the
    # run did not perform -- Table 3's "Train fuser only" is a real ablation
    # this repository does not implement. It is now REFUSED, below.
])
def test_command_line_and_protocol_fields_change_the_arm(field, value):
    """The hole `config_hash` had.

    `--epochs` and `--lr-horizon` are command-line only, so e5, e10 and e25 all
    carried ONE `config_hash` -- the sha256 of an artifact none of them fully
    obeyed. Three different trainings reported the same experiment id. The
    protocol fields were missing for the same reason: they live outside the
    hyperparameter artifact.
    """
    assert _arm()[0] != _arm(training=_training(**{field: value}))[0]


def test_encoding_protocol_fields_change_the_arm():
    """`actual_clip_train_scope: finetuned` was here too, and is now refused.

    Stage 1 reads text and image vectors that n06 computed offline with a frozen
    ViT-bigG-14. No value of that field can change anything inside a Stage 1
    run, so hashing `finetuned` recorded a recipe that cannot happen.
    """
    assert _arm()[0] != _arm(
        encoding=_encoding(missing_modality_representation="zero_pad"))[0]


@pytest.mark.parametrize("source,field,bad", [
    # `fuser_only` has been a real, executed scope since 2026-09-01; `full` is
    # the one the trainer still refuses (ViT-bigG-14's optimizer state does
    # not fit the card), so it is the value that must be refused here.
    ("training", "train_scope", "full"),
    ("encoding", "actual_clip_train_scope", "trainable"),
    ("encoding", "actual_clip_train_scope", "finetuned"),
])
def test_a_hashed_field_the_trainer_hardcodes_is_refused(source, field, bad):
    """[CODEX BLOCKER 2026-08-30] Hashed, and then ignored.

    `stage1.py:1324` builds the backbone with a literal
    `train_scope="point_encoder_and_fuser"`, and the CLIP towers are never in
    the optimizer at all. Both fields entered the arm hash and neither reached
    any branch, so a checkpoint could record an ablation that did not run.

    The frozen-key-set test cannot see this: the field is present in the recipe,
    correctly, and the defect is that nothing downstream reads it.
    """
    kw = {source: (_training if source == "training" else _encoding)(**{field: bad})}
    with pytest.raises(UnsupportedProtocol):
        _arm(**kw)


def test_phase_changes_the_arm():
    """dev trains 31,985 assets and final trains 36,554. Not one experiment."""
    assert _arm(phase="dev")[0] != _arm(phase="final")[0]


def test_a_per_view_draw_refuses_rather_than_silently_excusing_the_worker_count():
    """ARM_EXCLUDED's justification is CONDITIONAL, so it is enforced.

    `--preload` and `num_workers` are excluded from the arm because under the
    resolved `mean` aggregation nothing in `__getitem__` draws. Under
    `random_single_view`, `random` IS seeded per worker and they become
    arm-effective. Leaving the exclusion as a comment would let that change
    arrive with no signal at all.
    """
    with pytest.raises(UnsupportedProtocol):
        _arm(encoding=_encoding(image_aggregation="random_single_view"))


def test_the_resolved_recipe_is_stored_beside_the_digest():
    """A digest alone is unreadable provenance."""
    digest, resolved = _arm()
    assert len(digest) == 64
    assert resolved["learning_rate"] == 5e-4
    assert resolved["epochs"] == 5 and resolved["lr_horizon"] == 250
    assert resolved["weight_decay"] == 0.1


# ------------------------------------------------------------ code identity

def test_two_edits_at_one_revision_are_distinguishable(tmp_path):
    """The failure this field exists for.

    `e25_400w` and `e25_500w` shared `code_revision` 3f6fdde and
    `code_dirty` true, were reported as a clean repeat, and their 0.00123 spread
    was quoted as the noise floor. The trees differed. Revision plus a boolean
    could not have said so.
    """
    def digest():
        runlog._SOURCE_SHA = None
        return runlog.runtime_source_sha256(tmp_path)

    (tmp_path / "m.py").write_text("x = 1\n")
    a = digest()
    (tmp_path / "m.py").write_text("x = 2\n")
    b = digest()
    assert a != b
    runlog._SOURCE_SHA = None


def test_an_untracked_module_is_not_invisible(tmp_path):
    """[CODEX B 2026-08-30] The rejected version hashed `git diff HEAD`.

    A brand-new module that is never `git add`ed does not appear in a diff, yet
    Python imports it exactly like any other. Tracking status has never decided
    what ran, so it does not decide this digest either.
    """
    (tmp_path / "m.py").write_text("x = 1\n")
    runlog._SOURCE_SHA = None
    before = runlog.runtime_source_sha256(tmp_path)
    (tmp_path / "brand_new.py").write_text("y = 2\n")
    runlog._SOURCE_SHA = None
    after = runlog.runtime_source_sha256(tmp_path)
    assert before != after
    runlog._SOURCE_SHA = None


def test_moving_bytes_between_two_files_still_changes_the_digest(tmp_path):
    """Why path and length are framed into the hash rather than concatenated.

    Without the framing, `a.py="ab" b.py="c"` and `a.py="a" b.py="bc"` hash the
    same stream and two different programs report one identity.
    """
    def digest():
        runlog._SOURCE_SHA = None
        return runlog.runtime_source_sha256(tmp_path)

    (tmp_path / "a.py").write_text("ab")
    (tmp_path / "b.py").write_text("c")
    one = digest()
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("bc")
    assert one != digest()
    runlog._SOURCE_SHA = None


def test_the_snapshot_does_not_move_when_the_tree_does(tmp_path):
    """A run edited at hour two must report the program it STARTED with.

    The rejected version first resolved this at the end of epoch one, which
    answers a different question than the field claims to answer.
    """
    (tmp_path / "m.py").write_text("x = 1\n")
    runlog._SOURCE_SHA = None
    first = runlog.runtime_source_sha256(tmp_path)
    (tmp_path / "m.py").write_text("x = 999\n")
    assert runlog.runtime_source_sha256(tmp_path) == first
    runlog._SOURCE_SHA = None


def test_a_source_tree_that_cannot_be_read_reports_none_not_a_digest(tmp_path):
    """[CODEX MINOR 2026-08-30] This test used to assert `is not None`.

    Under a name promising a failure path, it asserted the opposite -- and it
    passed, because `rglob` on a missing directory returns an EMPTY iterator
    rather than raising. The function hashed nothing and returned
    `e3b0c442...`, sha256 of the empty string: a perfectly valid-looking digest
    meaning "there was no source here", which every reader would have compared
    as though it identified a program.

    Zero files is a failure. `None` cannot be mistaken for a digest, and the
    status field says why it is absent.
    """
    runlog._SOURCE_SHA = None
    assert runlog.runtime_source_sha256(tmp_path / "does_not_exist") is None
    assert runlog.runtime_source_status(tmp_path / "does_not_exist") == "unavailable"

    runlog._SOURCE_SHA = None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert runlog.runtime_source_sha256(empty) is None, (
        "an empty tree must not hash to sha256('')")
    runlog._SOURCE_SHA = None


def test_a_readable_source_tree_reports_ok(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n")
    runlog._SOURCE_SHA = None
    assert runlog.runtime_source_status(tmp_path) == "ok"
    assert len(runlog.runtime_source_sha256(tmp_path)) == 64
    runlog._SOURCE_SHA = None


# ------------------------------------------------------------------- run paths

def test_a_fresh_directory_gets_all_four_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    rp = stage1.resolve_run_paths("arm0")
    assert rp.root == (tmp_path / "arm0").resolve()
    assert {t.parent for t in rp.targets()} == {rp.root}
    assert len({t.name for t in rp.targets()}) == 4


def test_an_occupied_destination_stops_the_run_before_it_starts(tmp_path, monkeypatch):
    """[CODEX BLOCKER] `mkdir(exist_ok=True)` accepted a used directory.

    `data/outputs/ladder/e5_RECOVERED/` holds 8 KB of metrics and no weights
    because a later run reached `stage1_best.pt` first, in silence. Two sweep
    arms given one name reproduce that exactly, one directory further down.
    """
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    rp = stage1.resolve_run_paths("arm0")
    rp.best_checkpoint.write_bytes(b"a multi-hour run")
    with pytest.raises(SystemExit, match="refusing to start"):
        stage1.resolve_run_paths("arm0")


def test_the_canonical_destination_is_guarded_too(tmp_path, monkeypatch):
    """Omitting --out-dir is the case that actually lost a run."""
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    stage1.resolve_run_paths(None).latest_checkpoint.write_bytes(b"x")
    with pytest.raises(SystemExit, match="refusing to start"):
        stage1.resolve_run_paths(None)


def test_overwrite_is_the_deliberate_way_past(tmp_path, monkeypatch):
    """A DEAD run's outputs can be replaced on purpose."""
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    first = stage1.resolve_run_paths("arm0")
    first.latest_checkpoint.write_bytes(b"x")
    first.release()                                   # the run ended
    second = stage1.resolve_run_paths("arm0", overwrite=True)
    assert second.root.exists()
    second.release()


def test_a_crashed_run_does_not_wedge_its_directory(tmp_path, monkeypatch):
    """The reason the lock is `flock` and not a file whose existence is the lock.

    This machine hard-reset nine times on 2026-08-29. A lock a dead process
    still owns would have to be cleared by hand, and the first person to do that
    under time pressure clears a LIVE one by mistake. The kernel releases an
    flock when the holder dies, so the retry just works while a live run is
    still protected.
    """
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    crashed = stage1.resolve_run_paths("arm0")
    crashed.release()                                 # stands in for the crash
    assert crashed.lock.exists(), "the file remains; only the lock is gone"
    again = stage1.resolve_run_paths("arm0", overwrite=True)
    again.release()


@pytest.mark.parametrize("bad", ["../escaped", "a/../../escaped"])
def test_a_path_that_leaves_the_checkpoint_root_is_refused(bad, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path / "ckpt")
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(SystemExit, match="outside"):
        stage1.resolve_run_paths(bad)


def test_an_absolute_out_dir_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    with pytest.raises(SystemExit, match="absolute"):
        stage1.resolve_run_paths(str(tmp_path / "elsewhere"))


def test_a_smoke_run_cannot_write_the_canonical_names(tmp_path, monkeypatch):
    """[Codex 2026-08-28] Preserved from `best_paths`, which this replaced.

    `chain_to_stage1.sh` runs Stage 1 as a 200-asset smoke through the same
    dev-val selection. Without the suffix a 10-minute smoke silently replaces a
    multi-hour selection with one chosen over a 200-asset gallery.
    """
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    full = stage1.resolve_run_paths(None)
    smoke = stage1.resolve_run_paths(None, limit=200)
    assert not set(full.targets()) & set(smoke.targets())
    assert all("smoke200" in t.name for t in smoke.targets())


def test_run_paths_cannot_be_moved_after_construction():
    """Frozen: the destination is decided once, before the first batch."""
    rp = stage1.resolve_run_paths(None, overwrite=True)
    try:
        with pytest.raises(Exception):
            rp.root = paths.CHECKPOINTS / "elsewhere"
    finally:
        # Released, or this test holds the real checkpoint directory's lock for
        # the rest of the session and every later test that touches it fails
        # somewhere else. Found exactly that way.
        rp.release()
        # And its reservation removed: this is the real data directory, and a
        # test must not leave a file there that a reader would take for the
        # record of an actual run.
        rp.reservation.unlink(missing_ok=True)
        rp.lock.unlink(missing_ok=True)


# --------------------------------------------------------------- CLI guards

@pytest.mark.parametrize("argv,expect", [
    (["--lr", "-1"], "finite and positive"),
    (["--lr", "inf"], "finite and positive"),
    (["--lr", "nan"], "finite and positive"),
    # [CODEX F] `args.epochs or values["epochs"]` swallowed 0: the operator
    # asked for nothing and got the artifact's full run.
    (["--epochs", "0"], "must be positive"),
    (["--epochs", "-3"], "must be positive"),
    (["--out-dir", "/tmp/elsewhere"], "absolute"),
])
def test_the_cli_refuses_arguments_that_cannot_produce_a_run(argv, expect):
    # --query-observation is required since 2026-09-02; without it argparse
    # exits before any of the guards under test are reached.
    r = subprocess.run([PY, "-m", "metafind.train.stage1",
                        "--query-observation", "same_record", *argv],
                       capture_output=True, text=True, timeout=180,
                       cwd=str(paths.REPO))
    assert r.returncode != 0, r.stdout[-500:]
    assert expect in (r.stderr + r.stdout), (r.stderr + r.stdout)[-800:]


def test_the_parser_accepts_the_three_sweep_flags():
    r = subprocess.run([PY, "-m", "metafind.train.stage1", "--help"],
                       capture_output=True, text=True, timeout=180,
                       cwd=str(paths.REPO))
    for flag in ("--lr ", "--seed ", "--out-dir ", "--repeat-index ", "--overwrite"):
        assert flag in r.stdout, flag


# ------------------------------------------- what must NOT change the arm

def test_raising_the_approved_ceiling_does_not_change_the_arm():
    """[ULIP2 REVIEWER 2026-08-30, MAJOR] `max_epochs` sat in `values`.

    Nothing reads it: `stage1.py` only WARNS when a run exceeds it, and
    `resolve_stage1.py:304` says outright that no production code reads it. So
    raising the ceiling 250 -> 500 left training bit-identical while changing
    every arm hash -- the mirror of the enumeration hole. Under-inclusion gives
    two experiments one identity; over-inclusion strips comparability from every
    arm already run, just as quietly.
    """
    assert _arm()[0] == _arm(_values(max_epochs=500))[0]
    assert "max_epochs" not in _arm()[1]


@pytest.mark.parametrize("field", ["preload", "num_workers", "device"])
def test_execution_details_are_declared_excluded_not_merely_absent(field):
    """[ULIP2 REVIEWER 2026-08-30, MAJOR] The declaration has to be true.

    These three were excluded only by never being merged into `values`, while
    `arm_config_hash`'s own error message sent the reader to `ARM_EXCLUDED` to
    find them. This batch also established `values["learning_rate"] = args.lr`
    as the way to fold a flag in, so the next `values["preload"] = args.preload`
    would enter every arm hash with nothing to stop it.
    """
    assert field in stage1.ARM_EXCLUDED
    assert _arm()[0] == _arm(_values(**{field: "anything"}))[0]


# ------------------------------------- fields the trainer actually obeys

def test_the_masking_rule_changes_the_arm():
    """[CODEX BLOCKER 1, 2026-08-30] The field the enumeration forgot.

    `allow_all_masked` reaches `sample_modality_mask` at `stage1.py:1310` and
    decides whether a query with every modality dropped may occur. Two runs
    differing in it are two different trainings, and they shared one arm hash.
    """
    assert _arm()[0] != _arm(training=_training(allow_all_masked=False))[0]


def test_the_text_serialization_changes_the_arm():
    """The template that produced the cached text embeddings.

    Change it and every text vector the run consumes is different, while the
    uid pool -- the only other thing recorded about the inputs -- is identical.
    """
    assert _arm()[0] != _arm(encoding=_encoding(text_serialization="v8"))[0]


def test_rewording_a_note_about_the_paper_does_not_change_the_arm():
    """The over-inclusion side of the same coin.

    `paper_clip_train_scope_basis` is prose about what MetaFind claims, not
    about what this run does. Hashing it would give one experiment a second
    identity every time someone improved the wording, which strips
    comparability from every arm already run -- just as quietly as the
    under-inclusion failure above.
    """
    reworded = _encoding(paper_clip_train_scope_basis="reworded, same meaning")
    assert _arm()[0] == _arm(encoding=reworded)[0]


@pytest.mark.parametrize("field,bad", [
    ("optimizer", "sgd"), ("scheduler", "linear"), ("similarity", "dot"),
])
def test_a_declared_value_the_trainer_cannot_honour_is_refused(field, bad):
    """[CODEX BLOCKER 1] A recipe the record asserts but the run did not follow.

    `torch.optim.AdamW`, `cosine_schedule` and normalize-then-matmul are all
    called unconditionally. Setting `scheduler: linear` changed the arm hash and
    then annealed on a cosine anyway -- the checkpoint would have recorded a
    recipe that did not happen. Refusing is the only way the declaration stays
    true; an enforcement comment is not enforcement.
    """
    with pytest.raises(UnsupportedProtocol):
        _arm(_values(**{field: bad}), _training(**{field: bad}))


# ---------------------------------------------------------- pool identity

def test_the_pool_records_order_as_well_as_membership():
    """[CODEX MAJOR 4] Sorted membership cannot see a reordering.

    The shuffle permutes POSITIONS, so at one seed a different input order
    yields different batches and therefore different in-batch negatives for the
    contrastive loss. Two runs with the same assets in a different order are not
    the same condition, and only the sequence digest says so.
    """
    a = stage1.pool_provenance(["u1", "u2", "u3"], ["v1"])
    b = stage1.pool_provenance(["u3", "u1", "u2"], ["v1"])
    assert a["train_uid_set_sha256"] == b["train_uid_set_sha256"]
    assert a["train_uid_sequence_sha256"] != b["train_uid_sequence_sha256"]
    assert a["n_train"] == 3 and a["n_selection"] == 1


def test_a_final_phase_pool_is_recorded_too():
    """`save_best` hashed the pools, and only a dev run reaches `save_best`.

    A `--phase final` checkpoint -- the one a paper number comes from -- carried
    no pool identity at all.
    """
    p = stage1.pool_provenance(["u1", "u2"], [])
    assert p["n_selection"] == 0
    assert len(p["selection_uid_sequence_sha256"]) == 64


# ------------------------------------------------- simultaneous claimants

def test_two_live_processes_cannot_claim_one_directory(tmp_path, monkeypatch):
    """[CODEX BLOCKER 2, 2026-08-30] Checking is not claiming.

    The existence check passes for both callers -- nothing is written until the
    end of epoch one, minutes later -- and then they share the same `.part` and
    the same final names. The reservation is created with `O_CREAT | O_EXCL`,
    which the kernel makes atomic, so exactly one caller wins.

    It also survives a crash the four-target check cannot see: a run killed
    before its first save leaves no checkpoint, so the next run saw an empty
    directory and started on top of an attempt that may still have been alive.
    This machine hard-reset nine times on 2026-08-29.
    """
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    first = stage1.resolve_run_paths("arm0")
    assert first.reservation.exists() and first.lock.exists()
    # While the first is LIVE, nothing else gets in -- with or without
    # --overwrite. [CODEX MAJOR 2026-08-30] the previous version let
    # --overwrite unlink the reservation and both callers succeeded; my test
    # missed it because its second call did not pass overwrite=True.
    for kw in ({}, {"overwrite": True}):
        with pytest.raises(SystemExit, match="another process holds"):
            stage1.resolve_run_paths("arm0", **kw)
    # Once it is gone -- which the kernel does on process exit, including a hard
    # reset -- the directory is claimable again, so a crash does not wedge it.
    first.release()
    second = stage1.resolve_run_paths("arm0", overwrite=True)
    assert second.reservation.exists()
    second.release()


def test_the_reservation_says_what_claimed_it(tmp_path, monkeypatch):
    """A reservation naming nobody is a lock, not provenance."""
    import json as _json
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    rp = stage1.resolve_run_paths("arm0")
    claim = _json.loads(rp.reservation.read_text())
    assert claim["run_id"] == runlog.run_id()
    assert "argv" in claim and "started_at" in claim


# --------------------------------------------- the recipe's frozen key set

# Every key the arm recipe contains, reviewed one at a time on 2026-08-30. This
# list is the review, not a mirror of the code: adding a field to any of the
# three artifacts breaks the test below, and the only way to fix it is to decide
# whether the new field is a TREATMENT (add it here) or POLICY (add it to one of
# the *_EXCLUDED tuples with a stated reason).
ARM_RECIPE_KEYS = {
    # hyperparameter artifact, whole, minus ARM_EXCLUDED
    "optimizer", "learning_rate", "weight_decay", "scheduler", "batch_size",
    "p_mask", "init_temperature", "learnable_temperature", "max_logit_scale",
    "betas", "eps", "warmup_epochs", "lr_start", "lr_end",
    # command line only -- the hole `config_hash` had
    "epochs", "lr_horizon", "phase", "train_scope",
    # stage1_protocol.json, whole, minus TRAINING_EXCLUDED
    "training.fusion", "training.tower_sharing", "training.similarity",
    "training.allow_all_masked",
    # stage1_encoding_protocol.json, whole, minus ENCODING_EXCLUDED
    "encoding.image_aggregation", "encoding.missing_modality_representation",
    "encoding.actual_clip_train_scope", "encoding.text_serialization",
    "encoding.text_serialization_family", "encoding.text_serialization_contract",
    "encoding.text_serialization_probes", "encoding.text_template",
}


def test_the_recipe_key_set_is_frozen():
    """[ULIP2 REVIEWER 2026-08-30] The class of defect `ENFORCED_SINGLETONS` cannot see.

    Fail-closed catches "a value this trainer does not support". It does not
    catch **a field that is declared, supported, hashed, and read by nothing** —
    which is exactly what `max_epochs` was, and we found that one by hand.
    The next new artifact field will do it again.

    Freezing the key set makes that impossible to miss: any field added to any
    of the three artifacts breaks this test, and the only way past it is to
    decide, in writing, whether the field is a treatment or a policy. Cheaper
    than a resolver refactor and it stops the whole category.
    """
    import json as _json
    from metafind import paths as _paths
    hp = _json.loads((_paths.OUTPUTS / "stage1_hyperparameters.json").read_text())
    tr = _json.loads((_paths.OUTPUTS / "stage1_protocol.json").read_text())
    en = _json.loads((_paths.OUTPUTS / "stage1_encoding_protocol.json").read_text())
    _, recipe = stage1.arm_config_hash(
        hp["values"], {**tr, "_epoch_count": 5, "_lr_horizon": 5}, en, "dev")

    added = set(recipe) - ARM_RECIPE_KEYS
    gone = ARM_RECIPE_KEYS - set(recipe)
    assert not added, (
        f"{sorted(added)} entered the arm recipe unreviewed. Decide whether each "
        "is a TREATMENT (add to ARM_RECIPE_KEYS) or POLICY (add to ARM_EXCLUDED / "
        "TRAINING_EXCLUDED / ENCODING_EXCLUDED with a stated reason).")
    assert not gone, (
        f"{sorted(gone)} left the arm recipe. Two different trainings may now "
        "share one identity -- the failure `allow_all_masked` already caused.")


# ------------------------------------------------- input content provenance

def test_the_pointcloud_digest_comes_from_bytes_not_from_the_sidecars_claim(
        tmp_path, monkeypatch):
    """[CODEX MAJOR 2026-08-30] A sidecar is a claim, and claims were the thing
    this whole batch stopped trusting.

    The first version hashed the sha256 the n07 sidecar CLAIMED. Edit the `.npz`
    and leave the sidecar alone and the digest does not move -- a
    complete-looking provenance record describing bytes that are no longer
    there. Measured cost of doing it properly: 1,500 clouds in 0.2 s.
    """
    pc, emb = tmp_path / "pc", tmp_path / "emb"
    pc.mkdir(); emb.mkdir()
    monkeypatch.setattr(paths, "POINTCLOUDS", pc)
    monkeypatch.setattr(paths, "EMBEDDINGS", emb)
    import hashlib as _h
    import json as _j
    (pc / "u1.npz").write_bytes(b"original cloud")
    (pc / "u1.json").write_text(_j.dumps(
        {"sha256": _h.sha256(b"original cloud").hexdigest()}))
    (emb / "u1.npz").write_bytes(b"vectors")

    before = stage1.input_content_digest(["u1"])
    assert before["n_pointcloud_sidecar_mismatch"] == 0

    (pc / "u1.npz").write_bytes(b"a DIFFERENT cloud")   # sidecar untouched
    after = stage1.input_content_digest(["u1"])
    assert after["content_sha256"] != before["content_sha256"], (
        "changing the cloud must change the digest")
    assert after["n_pointcloud_sidecar_mismatch"] == 1, (
        "the sidecar's stale claim must be reported, not silently trusted")
