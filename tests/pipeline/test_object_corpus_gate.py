"""CPU evidence tests for G3's existing corpus/protocol criteria, not training."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from metafind.gates import g3_object_corpus as gate
from metafind.data.splits import build_eval_protocols
from metafind.models.stage1_config import canonical_hyperparameter_hash


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def corpus(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "logs").mkdir()
    write_json(outputs / "annotation_exclusions.json", {"excluded_total": 0, "groups": {}})
    manifest = tmp_path / "lvis.json"
    uids = [f"u{i:03}" for i in range(100)]
    write_json(manifest, {uid: f"glbs/{uid}.glb" for uid in uids})
    values = {"optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": 0.01,
              "scheduler": "cosine", "batch_size": 8, "epochs": 1,
              "max_epochs": 10, "p_mask": 0.3, "decay_mask_tokens": False,
              "init_temperature": 0.07, "learnable_temperature": True,
              "max_logit_scale": 100.0, "betas": [0.9, 0.999], "eps": 1e-8,
              "warmup_epochs": 0, "lr_start": 1e-4, "lr_end": 0.0, "seed": 17}
    hp = {"values": values, "sha256": canonical_hyperparameter_hash(values)}
    training = {"status": "resolved", "fusion": "transformer",
                "tower_sharing": "shared_backbone_separate_fusion",
                "allow_all_masked": True, "similarity": "cosine",
                "hyperparameter_config_hash": hp["sha256"]}
    encoding = {"status": "resolved", "text_serialization": "v3_fit@fixture",
                "image_aggregation": "mean", "paper_clip_train_scope": "frozen",
                "actual_clip_train_scope": "frozen", "missing_modality_representation": "learned_token",
                "paper_clip_train_scope_basis": "explicit fixture interpretation",
                "paper_clip_train_scope_confidence": "moderate"}
    for name, obj in [("stage1_hyperparameters", hp), ("stage1_protocol", training),
                      ("stage1_encoding_protocol", encoding)]:
        write_json(outputs / f"{name}.json", obj)

    def admit(selected):
        train, val, test = selected[:-20], selected[-20:-10], selected[-10:]
        write_json(outputs / "splits.json", {"object": {"train": train, "val": val,
            "test": test, "holdout": val + test, "train_val": train + val,
            "dev_train": train, "dev_val": val}, "split_seed": 17, "scheme": "80/10/10"})
        write_json(outputs / "eval_protocols.json", build_eval_protocols(train, test, val, val + test))

    admit(uids)
    return {"outputs": outputs, "manifest": manifest, "uids": uids, "admit": admit,
            "record": tmp_path / "results/G3_object_corpus.yaml"}


def run(corpus, **kwargs):
    rc = gate.run(corpus["outputs"], corpus["manifest"], corpus["record"], **kwargs)
    return rc, yaml.safe_load(corpus["record"].read_text())


def change(corpus, filename, fn):
    path = corpus["outputs"] / f"{filename}.json"
    value = json.loads(path.read_text())
    fn(value)
    write_json(path, value)


def set_mask(corpus, value):
    hp_path = corpus["outputs"] / "stage1_hyperparameters.json"
    hp = json.loads(hp_path.read_text())
    hp["values"]["p_mask"] = value
    hp["sha256"] = canonical_hyperparameter_hash(hp["values"])
    write_json(hp_path, hp)
    change(corpus, "stage1_protocol", lambda p: p.update(hyperparameter_config_hash=hp["sha256"]))


def row(uid, **kwargs):
    return {"uid": uid, "stage": "n04_render_views", "failure_class": "DETERMINISTIC_INPUT",
            "exception_type": "ValueError", "exception_msg": "broken mesh",
            "code_revision": "fixture-revision", "timestamp": "2026-09-08T00:00:00Z", **kwargs}


def quarantine(corpus, records, suffix=""):
    # Mirror the producer's per-node files; mixed fixtures must not put a scene
    # row in an object writer's file and accidentally test an impossible seam.
    owners = {}
    for record in records:
        owner = record.get("stage")
        if owner not in gate.OBJECT_STAGES | gate.SCENE_STAGES:
            owner = "n04_render_views"
        owners.setdefault(owner, []).append(record)
    owners = owners or {"n04_render_views": []}
    paths_written = []
    for owner, rows in owners.items():
        path = corpus["outputs"] / f"logs/quarantine_{owner}{suffix}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in rows))
        paths_written.append(path)
    return paths_written[0]


def manual_exclusion(corpus, uid):
    write_json(corpus["outputs"] / "annotation_exclusions.json", {
        "decision": "fixture explicit rejection", "excluded_total": 1,
        "groups": {"manual_review_rejected": {"n": 1, "uids": [{"uid": uid, "model": "fixture"}]}}})


def test_complete_corpus_passes_with_aliases_and_six_protocols(corpus):
    rc, record = run(corpus)
    assert rc == 0 and record["verdict"] == "PASS"
    assert record["observed"]["accounting"]["set_conservation"]
    assert record["observed"]["admitted"]["count"] == 100
    assert len(record["observed"]["evaluation_protocols"]) == 6
    assert record["observed"]["leakage_count"] == 0
    assert "actual trainer" in record["observed"]["coverage_limit"]
    spec = next(item for item in yaml.safe_load(gate.SPEC_PATH.read_text())["level_3_gates"]
                if item["gate_id"] == gate.GATE_ID)
    assert set(spec["record_fields"]) <= record.keys()
    assert record["criterion"] == spec["criterion"].strip()
    assert record["is_terminal"] is True
    assert record["inputs"]["manifest"]["sha256"] == hashlib.sha256(corpus["manifest"].read_bytes()).hexdigest()


def test_duplicate_retry_rows_recovery_and_historical_guards_do_not_inflate_quarantine(corpus):
    corpus["admit"](corpus["uids"][2:])
    rows = [row("u000"), row("u000"), row("u001"), row("u099")]
    rows += [row(uid, exception_msg=gate.HISTORICAL_GUARD_PREFIX + " (snapshot)") for uid in corpus["uids"]]
    path = quarantine(corpus, rows)
    quarantine(corpus, [row("u000")], ".v5_intel")
    before = path.read_bytes()
    rc, record = run(corpus)
    assert rc == 0
    q = record["observed"]["quarantine"]
    assert q["quarantined_uids"] == ["u000", "u001"]
    assert q["duplicate_failure_rows"] == 2
    assert q["recovered"]["count"] == 1
    assert q["historical_guard_rows_excluded"] == 100
    assert record["observed"]["accounting"]["quarantine_rate"] == 0.02
    assert path.read_bytes() == before
    assert len(record["inputs"]["quarantine_logs"]) == 2


def test_no_real_failures_are_invented_for_unaccounted_assets(corpus):
    corpus["admit"](corpus["uids"][1:])
    rc, record = run(corpus)
    assert rc == 2
    assert record["observed"]["quarantine"]["quarantined_uids"] == []
    assert record["observed"]["accounting"]["unexplained_missing"]["examples"] == ["u000"]
    assert "missing is not a quarantine reason" in str(record["observed"]["failures"])


def test_historical_guard_is_exact_prefix_and_n04_only(corpus):
    corpus["admit"](corpus["uids"][2:])
    quarantine(corpus, [row("u000", exception_msg="prefix: " + gate.HISTORICAL_GUARD_PREFIX),
                        row("u001", stage="n06_encode_text_image", exception_msg=gate.HISTORICAL_GUARD_PREFIX)])
    rc, record = run(corpus)
    assert rc == 0
    assert record["observed"]["quarantine"]["quarantined_uids"] == ["u000", "u001"]
    assert record["observed"]["quarantine"].get("historical_guard_rows_excluded", 0) == 0


def test_historical_guard_alone_does_not_account_for_missing_uid(corpus):
    corpus["admit"](corpus["uids"][1:])
    quarantine(corpus, [row("u000", exception_msg=gate.HISTORICAL_GUARD_PREFIX)])
    rc, record = run(corpus)
    assert rc == 2 and record["observed"]["accounting"]["quarantined_count"] == 0


def test_real_quarantine_limit_is_not_relaxed(corpus):
    corpus["admit"](corpus["uids"][3:])
    quarantine(corpus, [row(uid) for uid in corpus["uids"][:3]])
    rc, record = run(corpus)
    assert rc == 2
    assert record["observed"]["accounting"]["quarantine_rate"] == 0.03
    assert "exceeds existing 0.02" in str(record["observed"]["failures"])


@pytest.mark.parametrize("field", gate.QUARANTINE_FIELDS[2:])
def test_missing_exception_provenance_blocks_instead_of_fabricating(corpus, field):
    corpus["admit"](corpus["uids"][1:])
    failure = row("u000")
    del failure[field]
    quarantine(corpus, [failure])
    rc, record = run(corpus)
    assert rc == 3 and field in str(record["observed"]["blocked_reasons"])


def test_scene_failures_and_global_initialization_are_not_object_quarantine(corpus):
    quarantine(corpus, [row("house-id", stage="n07_scene_graphs"),
                        row("u000", stage="n07b_procthor_asset_modalities"),
                        row(None, stage="n06_encode_text_image", phase="initialization")])
    rc, record = run(corpus)
    assert rc == 0
    q = record["observed"]["quarantine"]
    assert q["quarantined"]["count"] == 0
    assert q["ignored_non_object_stages"] == {"n07_scene_graphs": 1, "n07b_procthor_asset_modalities": 1}
    assert q["initialization_rows_not_asset_failures"] == 1


def test_scene_failure_cannot_cover_missing_object(corpus):
    corpus["admit"](corpus["uids"][1:])
    quarantine(corpus, [row("u000", stage="n07b_procthor_asset_modalities")])
    assert run(corpus)[0] == 2


@pytest.mark.parametrize("inject", [
    lambda p: p["object"]["train"].append(p["object"]["test"][0]),
    lambda p: p["object"]["test"].append(p["object"]["test"][0]),
    lambda p: p["object"]["holdout"].pop(),
    lambda p: p["object"]["dev_val"].pop(),
    lambda p: p["object"]["train_val"].pop(),
])
def test_leakage_duplicates_and_alias_drift_fail(corpus, inject):
    change(corpus, "splits", inject)
    assert run(corpus)[0] == 2


def test_equal_counts_cannot_hide_a_foreign_admitted_uid(corpus):
    corpus["admit"](["foreign"] + corpus["uids"][1:])
    rc, record = run(corpus)
    assert rc == 2
    accounting = record["observed"]["accounting"]
    assert accounting["manifest_count"] == accounting["admitted_count"] == 100
    assert accounting["unexpected"]["examples"] == ["foreign"]
    assert accounting["unexplained_missing"]["examples"] == ["u000"]


def test_quarantine_uid_outside_manifest_fails(corpus):
    quarantine(corpus, [row("foreign")])
    assert run(corpus)[0] == 2


@pytest.mark.parametrize("failure_exists", [False, True])
def test_approved_manual_exclusions_are_separate_and_not_double_counted(corpus, failure_exists):
    corpus["admit"](corpus["uids"][1:])
    manual_exclusion(corpus, "u000")
    if failure_exists:
        quarantine(corpus, [row("u000")])
    rc, record = run(corpus)
    assert rc == 0
    assert record["observed"]["annotation_exclusions"]["manual_uids"] == ["u000"]
    assert record["observed"]["accounting"]["quarantined_count"] == 0
    assert record["observed"]["accounting"]["manual_excluded_count"] == 1
    assert record["observed"]["accounting"]["pairwise_disjoint"]
    assert record["observed"]["accounting"]["set_conservation"]
    assert record["observed"]["quarantine"]["manual_failure_overlap"]["count"] == int(failure_exists)
    assert record["observed"]["accounting"]["unexplained_missing"]["count"] == 0
    assert not record["observed"]["blocked_reasons"]


def test_readmitted_excluded_asset_is_a_definite_failure(corpus):
    manual_exclusion(corpus, "u000")
    rc, record = run(corpus)
    assert rc == 2
    assert record["observed"]["accounting"]["excluded_but_admitted"]["count"] == 1
    assert not record["observed"]["accounting"]["pairwise_disjoint"]


def test_n05_exclusion_ledger_alone_is_not_real_exception_evidence(corpus):
    corpus["admit"](corpus["uids"][1:])
    write_json(corpus["outputs"] / "annotation_exclusions.json", {"excluded_total": 1,
        "groups": {"n05_quarantine": {"n": 1, "uids": ["u000"]}}})
    rc, record = run(corpus)
    assert rc == 2 and record["observed"]["accounting"]["quarantined_count"] == 0
    quarantine(corpus, [row("u000", stage="n05_annotate")])
    assert run(corpus)[0] == 0


@pytest.mark.parametrize("filename", ["annotation_exclusions", "splits", "eval_protocols", "stage1_encoding_protocol",
                                      "stage1_protocol", "stage1_hyperparameters"])
def test_missing_core_input_blocks_and_writes_terminal_record(corpus, filename):
    (corpus["outputs"] / f"{filename}.json").unlink()
    rc, record = run(corpus)
    assert rc == 3 and record["is_terminal"] and filename in str(record["observed"]["blocked_reasons"])


def test_missing_manifest_and_log_directory_block(corpus):
    corpus["manifest"].unlink()
    (corpus["outputs"] / "logs").rmdir()
    rc, record = run(corpus)
    assert rc == 3
    assert "manifest" in str(record["observed"]["blocked_reasons"])
    assert "quarantine log directory" in str(record["observed"]["blocked_reasons"])


@pytest.mark.parametrize("filename,inject,rc", [
    ("stage1_encoding_protocol", lambda p: p.pop("paper_clip_train_scope_basis"), 3),
    ("stage1_encoding_protocol", lambda p: p.update(paper_clip_train_scope_basis=" "), 3),
    ("stage1_encoding_protocol", lambda p: p.update(paper_clip_train_scope_confidence="certain"), 2),
    ("stage1_encoding_protocol", lambda p: p.update(text_serialization="UNKNOWN"), 3),
    ("stage1_encoding_protocol", lambda p: p.update(status="pending"), 3),
    ("stage1_protocol", lambda p: p.update(allow_all_masked="true"), 2),
    ("stage1_protocol", lambda p: p.update(fusion="invented"), 2),
    ("stage1_protocol", lambda p: p.update(similarity="dot_product"), 2),
    ("stage1_protocol", lambda p: p.pop("tower_sharing"), 3),
    ("stage1_hyperparameters", lambda p: p["values"].pop("learning_rate"), 3),
    ("stage1_hyperparameters", lambda p: p["values"].update(learning_rate=0.1), 2),
    ("stage1_protocol", lambda p: p.update(hyperparameter_config_hash="0" * 64), 2),
])
def test_protocol_and_canonical_hash_checks(corpus, filename, inject, rc):
    change(corpus, filename, inject)
    assert run(corpus)[0] == rc


@pytest.mark.parametrize("mask,variant,rc", [(0.3, None, 0), (0.1, None, 2), (0.5, None, 2),
    (0.1, "dropout_10", 0), (0.5, "dropout_50", 0), (0.1, "dropout_50", 2),
    (0.3, "dropout_10", 2), (True, None, 2)])
def test_main_probability_and_explicit_table3_variants(corpus, mask, variant, rc):
    set_mask(corpus, mask)
    assert run(corpus, variant=variant)[0] == rc


def test_zero_padding_known_ablation_not_rejected_by_stale_runtime_config(corpus):
    change(corpus, "stage1_encoding_protocol", lambda p: p.update(missing_modality_representation="zero_pad"))
    assert run(corpus)[0] == 0


@pytest.mark.parametrize("inject,rc", [
    (lambda p: p.pop("B_full_gallery"), 3),
    (lambda p: p["A_test_gallery"].pop("query_split"), 3),
    (lambda p: p["B_full_gallery"].update(gallery_size=48000), 2),
    (lambda p: p["B_full_gallery"].update(gallery_split="unknown"), 2),
    (lambda p: p["A_test_gallery"].update(query_split="train"), 2),
    (lambda p: p["C_dev_selection"].update(gallery_split="train"), 2),
    (lambda p: p["C_dev_selection"].update(query_size=999), 2),
])
def test_evaluation_scope_and_sizes_are_checked_not_only_presence(corpus, inject, rc):
    change(corpus, "eval_protocols", inject)
    assert run(corpus)[0] == rc


def test_legacy_dev_partition_supported_without_counting_aliases_twice(corpus):
    train, test = corpus["uids"][:80], corpus["uids"][80:]
    dev_train, dev_val = train[:70], train[70:]
    write_json(corpus["outputs"] / "splits.json", {"object": {"train": train,
        "test": test, "dev_train": dev_train, "dev_val": dev_val}, "split_seed": 17})
    write_json(corpus["outputs"] / "eval_protocols.json", build_eval_protocols(train, test, dev_val))
    assert run(corpus)[0] == 0


def test_duplicate_manifest_keys_are_not_lost_by_json_parser(corpus):
    corpus["manifest"].write_text('{"u000":"a", "u000":"b"}')
    rc, record = run(corpus)
    assert rc == 2 and "duplicate JSON object key" in str(record["observed"]["failures"])


def test_failed_record_history_is_preserved_on_retry(corpus):
    change(corpus, "stage1_protocol", lambda p: p.update(status="pending"))
    assert run(corpus)[0] == 3
    change(corpus, "stage1_protocol", lambda p: p.update(status="resolved"))
    assert run(corpus)[0] == 0
    history = yaml.safe_load(corpus["record"].with_suffix(".history.yaml").read_text())
    assert [record["rc"] for record in history] == [3, 0]


def test_cli_rc_and_run_progress_agree_and_write_only_isolated_outputs(corpus, tmp_path):
    (corpus["outputs"] / "stage1_protocol.json").unlink()
    before = {path: path.read_bytes() for path in corpus["outputs"].glob("*.json")}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    canonical = tmp_path / "untouched_canonical_root"
    canonical.mkdir()
    env["METAFIND_DATA"] = str(canonical)
    result = subprocess.run([sys.executable, "-B", "-m", "metafind.gates.g3_object_corpus",
        "--outputs", str(corpus["outputs"]), "--manifest", str(corpus["manifest"]),
        "--record", str(corpus["record"])], env=env, capture_output=True, text=True)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "BLOCKED_EVIDENCE" in result.stdout
    progress = [json.loads(line) for line in (corpus["record"].parent / "run_progress.jsonl").read_text().splitlines()]
    assert progress[-1]["stage"] == gate.GATE_ID and progress[-1]["rc"] == 3
    assert progress[-1]["status"] == "FAILED"
    assert not (corpus["outputs"] / "logs/run_progress.jsonl").exists()
    assert not list(canonical.iterdir())
    assert all(path.read_bytes() == blob for path, blob in before.items())


def test_changed_spec_rc_contract_blocks(corpus, tmp_path):
    spec = yaml.safe_load(gate.SPEC_PATH.read_text())
    next(entry for entry in spec["level_3_gates"] if entry["gate_id"] == gate.GATE_ID)["rc_contract"]["PASS"] = 99
    path = tmp_path / "changed_spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    assert run(corpus, spec_path=path)[0] == 3


@pytest.mark.parametrize("value,rc", [(None, 2), ({}, 2), ("", 2), ("UNKNOWN", 3)])
def test_malformed_or_unresolved_manifest_source_is_not_certified(corpus, value, rc):
    manifest = json.loads(corpus["manifest"].read_text())
    manifest["u000"] = value
    write_json(corpus["manifest"], manifest)
    assert run(corpus)[0] == rc


@pytest.mark.parametrize("field,value,rc", [
    ("optimizer", "UNKNOWN", 3), ("optimizer", False, 2),
    ("scheduler", {}, 2), ("batch_size", "8", 2), ("seed", True, 2),
    ("learning_rate", "0.1", 2), ("decay_mask_tokens", "false", 2),
    ("betas", [], 2), ("betas", ["0.9", 0.999], 2),
])
def test_unknown_or_malformed_hyperparameter_values_do_not_pass_with_valid_hash(corpus, field, value, rc):
    hp_path = corpus["outputs"] / "stage1_hyperparameters.json"
    hp = json.loads(hp_path.read_text())
    hp["values"][field] = value
    hp["sha256"] = canonical_hyperparameter_hash(hp["values"])
    write_json(hp_path, hp)
    change(corpus, "stage1_protocol", lambda p: p.update(hyperparameter_config_hash=hp["sha256"]))
    assert run(corpus)[0] == rc


def test_unknown_quarantine_scope_is_not_silently_ignored(corpus):
    quarantine(corpus, [row("u000", stage="UNKNOWN")])
    rc, record = run(corpus)
    assert rc == 3 and "unknown quarantine stage" in str(record["observed"]["blocked_reasons"])


def test_truncated_quarantine_evidence_is_blocked(corpus):
    quarantine(corpus, []).write_text('{"uid":')
    assert run(corpus)[0] == 3


def test_missing_split_object_is_evidence_absence(corpus):
    write_json(corpus["outputs"] / "splits.json", {"split_seed": 17})
    assert run(corpus)[0] == 3


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink", "history", "part"])
def test_record_writer_never_overwrites_input_even_via_alias(corpus, tmp_path, alias_kind):
    original = corpus["manifest"].read_bytes()
    record_path = tmp_path / "gate.yaml"
    if alias_kind == "direct":
        record_path = corpus["manifest"]
    elif alias_kind == "symlink":
        record_path.symlink_to(corpus["manifest"])
    elif alias_kind == "hardlink":
        os.link(corpus["manifest"], record_path)
    elif alias_kind == "history":
        record_path.with_suffix(".history.yaml").symlink_to(corpus["manifest"])
    else:
        record_path.with_suffix(".yaml.part").symlink_to(corpus["manifest"])
    assert gate.run(corpus["outputs"], corpus["manifest"], record_path) == 3
    assert corpus["manifest"].read_bytes() == original


@pytest.mark.parametrize("mutation", ["append", "new_log", "new_ledger", "protocol"])
def test_inputs_that_drift_during_read_cannot_produce_pass(corpus, monkeypatch, mutation):
    log = quarantine(corpus, [row("u000")])  # recovered, so initially valid
    real = gate._protocols

    def mutate_after_input_reads(*args):
        real(*args)
        if mutation == "append":
            with log.open("a") as stream:
                stream.write(json.dumps(row("u001")) + "\n")
        elif mutation == "new_log":
            quarantine(corpus, [row("u001")], ".additional")
        elif mutation == "new_ledger":
            manual_exclusion(corpus, "u001")
        else:
            change(corpus, "stage1_protocol", lambda p: p.update(status="pending"))

    monkeypatch.setattr(gate, "_protocols", mutate_after_input_reads)
    rc, record = run(corpus)
    assert rc == 3 and "during gate execution" in str(record["observed"]["blocked_reasons"])


@pytest.mark.parametrize("alias_kind,target", [("direct", "record"),
    *[(kind, target) for kind in ("symlink", "hardlink")
      for target in ("record", "history", "record_part", "history_part")]])
def test_cli_progress_cannot_alias_gate_outputs(corpus, tmp_path, alias_kind, target):
    isolated = tmp_path / "cli_outputs"
    isolated.mkdir()
    progress = isolated / "run_progress.jsonl"
    original = b'{"stage":"prior-run","rc":3}\n'
    progress.write_bytes(original)
    record = isolated / "gate.yaml"
    history = isolated / "gate.history.yaml"
    destinations = {"record": record, "history": history,
                    "record_part": isolated / "gate.yaml.part",
                    "history_part": isolated / "gate.history.yaml.part"}
    if alias_kind == "direct":
        record = progress
    elif alias_kind == "symlink":
        destinations[target].symlink_to(progress)
    else:
        os.link(progress, destinations[target])
    before = {path: path.read_bytes() for path in isolated.iterdir()}
    result = subprocess.run([sys.executable, "-B", "-m", "metafind.gates.g3_object_corpus",
        "--outputs", str(corpus["outputs"]), "--manifest", str(corpus["manifest"]),
        "--record", str(record)], capture_output=True, text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES=""))
    assert result.returncode == 3, result.stdout + result.stderr
    assert "BLOCKED_EVIDENCE" in result.stdout
    assert {path: path.read_bytes() for path in isolated.iterdir()} == before
    assert progress.read_bytes() == original


def test_cli_new_progress_record_collision_is_refused_before_creating_logs(corpus, tmp_path):
    record = tmp_path / "new_outputs/run_progress.jsonl"
    result = subprocess.run([sys.executable, "-B", "-m", "metafind.gates.g3_object_corpus",
        "--outputs", str(corpus["outputs"]), "--manifest", str(corpus["manifest"]),
        "--record", str(record)], capture_output=True, text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES=""))
    assert result.returncode == 3, result.stdout + result.stderr
    assert "BLOCKED_EVIDENCE" in result.stdout
    assert not record.parent.exists()


@pytest.mark.parametrize("owner", sorted(gate.OBJECT_STAGES))
def test_current_runlog_writer_object_records_are_accepted(corpus, monkeypatch, owner):
    corpus["admit"](corpus["uids"][1:])
    monkeypatch.setattr(gate.paths, "LOGS", corpus["outputs"] / "logs")
    path = gate.runlog.quarantine(owner, [{"uid": "u000", "failure_class": "DETERMINISTIC_INPUT",
        "exception_type": "ValueError", "exception_msg": "actual writer fixture"}])
    actual = json.loads(path.read_text())
    assert actual["stage"] == owner
    assert type(actual["timestamp"]) is float
    assert actual["code_revision"]
    rc, record = run(corpus)
    assert rc == 0 and record["observed"]["quarantine"]["quarantined_uids"] == ["u000"]


def test_current_procthor_phase_writer_is_excluded_by_verified_scene_owner(corpus, monkeypatch):
    monkeypatch.setattr(gate.paths, "LOGS", corpus["outputs"] / "logs")
    path = gate.runlog.quarantine("n07b_procthor_asset_modalities", [{
        "asset_id": "u000", "failure_class": "DETERMINISTIC_INPUT", "stage": "render",
        "exception_type": "ValueError", "exception_msg": "actual n07b writer fixture"}])
    actual = json.loads(path.read_text())
    assert actual["stage"] == "render" and "uid" not in actual
    rc, record = run(corpus)
    assert rc == 0
    assert record["observed"]["quarantine"]["ignored_non_object_phases"] == {
        "n07b_procthor_asset_modalities/render": 1}
    assert record["observed"]["quarantine"]["quarantined_uids"] == []
    corpus["admit"](corpus["uids"][1:])
    assert run(corpus)[0] == 2  # A scene render failure cannot account for an object deficit.


@pytest.mark.parametrize("owner,stage,rc", [
    ("n04_render_views", "render", 3),
    ("n07b_procthor_asset_modalities", "n04_render_views", 2),
    ("n04_render_views", "n05_annotate", 2),
])
def test_owner_and_row_scope_mismatch_is_not_silently_ignored(corpus, monkeypatch, owner, stage, rc):
    monkeypatch.setattr(gate.paths, "LOGS", corpus["outputs"] / "logs")
    gate.runlog.quarantine(owner, [row("u000", stage=stage)])
    assert run(corpus)[0] == rc


@pytest.mark.parametrize("timestamp,rc", [(1725753600.5, 0), (1725753600, 0),
    ("2026-09-08T00:00:00Z", 0), (None, 3), (False, 3), ("UNKNOWN", 3), ({}, 3)])
def test_quarantine_timestamp_accepts_epoch_or_iso_but_not_malformed(corpus, monkeypatch, timestamp, rc):
    corpus["admit"](corpus["uids"][1:])
    monkeypatch.setattr(gate.paths, "LOGS", corpus["outputs"] / "logs")
    gate.runlog.quarantine("n04_render_views", [row("u000", timestamp=timestamp)])
    assert run(corpus)[0] == rc


def test_current_n04_run_and_batch_sentinels_are_not_asset_uids(corpus, monkeypatch):
    monkeypatch.setattr(gate.paths, "LOGS", corpus["outputs"] / "logs")
    gate.runlog.quarantine("n04_render_views", [
        {"uid": "__run", "failure_class": "RESOURCE", "exception_type": "SystemicFailure",
         "exception_msg": "actual n04 run sentinel fixture"},
        {"uid": "__batch_500", "failure_class": "RESOURCE", "exception_type": "BrokenProcessPool",
         "exception_msg": "actual n04 batch sentinel fixture"},
    ])
    rc, record = run(corpus)
    assert rc == 0
    q = record["observed"]["quarantine"]
    assert q["systemic_rows_not_asset_failures"] == 2
    assert q["quarantined_uids"] == []
    assert [item["uid"] for item in q["systemic_rows"]] == ["__run", "__batch_500"]


@pytest.mark.parametrize("uid,exception_type", [("__run", "ValueError"),
                                               ("__batch_invalid", "BrokenProcessPool")])
def test_sentinel_exclusion_requires_exact_writer_signature(corpus, monkeypatch, uid, exception_type):
    monkeypatch.setattr(gate.paths, "LOGS", corpus["outputs"] / "logs")
    gate.runlog.quarantine("n04_render_views", [row(uid, failure_class="RESOURCE", exception_type=exception_type)])
    rc, record = run(corpus)
    assert rc == 2
    assert record["observed"]["quarantine"]["quarantined_uids"] == [uid]


def test_only_quarantine_rate_has_two_percent_limit_manual_rate_is_separate(corpus):
    corpus["admit"](corpus["uids"][12:])
    write_json(corpus["outputs"] / "annotation_exclusions.json", {"excluded_total": 10,
        "groups": {"manual_review_rejected": {"n": 10, "uids": corpus["uids"][:10]}}})
    quarantine(corpus, [row(uid) for uid in corpus["uids"][:12]])
    rc, record = run(corpus)
    accounting = record["observed"]["accounting"]
    assert rc == 0
    assert accounting["quarantined_count"] == 2 and accounting["manual_excluded_count"] == 10
    assert accounting["quarantine_rate"] == 0.02
    assert accounting["manual_excluded_rate"] == 0.1
    assert accounting["total_excluded_rate"] == 0.12
    assert accounting["admitted_count"] + accounting["quarantined_count"] + accounting["manual_excluded_count"] == accounting["manifest_count"]
    assert accounting["pairwise_disjoint"] and accounting["set_conservation"]


def test_unknown_manual_group_stays_blocked_and_does_not_become_E_or_Q(corpus):
    corpus["admit"](corpus["uids"][1:])
    write_json(corpus["outputs"] / "annotation_exclusions.json", {"excluded_total": 1,
        "groups": {"unapproved_filter": {"n": 1, "uids": ["u000"]}}})
    rc, record = run(corpus)
    assert rc == 3
    assert record["observed"]["accounting"]["manual_excluded_count"] == 0
    assert record["observed"]["accounting"]["quarantined_count"] == 0
    assert not record["observed"]["accounting"]["set_conservation"]
    assert record["observed"]["annotation_exclusions"]["unapproved_groups"] == ["unapproved_filter"]


def test_historical_n05_ledger_does_not_override_actual_recovery(corpus):
    write_json(corpus["outputs"] / "annotation_exclusions.json", {"excluded_total": 1,
        "groups": {"n05_quarantine": {"n": 1, "uids": ["u000"]}}})
    quarantine(corpus, [row("u000", stage="n05_annotate")])
    rc, record = run(corpus)
    assert rc == 0
    assert record["observed"]["quarantine"]["recovered"]["count"] == 1
    assert record["observed"]["accounting"]["quarantined_count"] == 0
    assert record["observed"]["accounting"]["manual_excluded_count"] == 0


def test_manual_exclusion_cannot_introduce_foreign_manifest_uid(corpus):
    manual_exclusion(corpus, "foreign")
    rc, record = run(corpus)
    assert rc == 2
    assert record["observed"]["accounting"]["unexpected"]["examples"] == ["foreign"]


def test_exclusion_evidence_must_explicitly_define_groups(corpus):
    write_json(corpus["outputs"] / "annotation_exclusions.json", {})
    assert run(corpus)[0] == 3
