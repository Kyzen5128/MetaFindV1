"""Exercise the queued shell scripts without models or real corpus writes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_chain(tmp_path, stage, failure, archive_missing=None):
    root = Path(__file__).resolve().parents[2]
    data = tmp_path / "data"
    logs = data / "outputs" / "logs"
    logs.mkdir(parents=True)
    (logs / "r2_annotate_v10.log").write_text("=== R2 DONE\n")
    (logs / "chain_paper_stage1_20260906.log").write_text("=== DONE (row 1)\n")
    (logs / "n07b_procthor_modalities_v2.log").write_text("1467 rendered, 0 quarantined\n")
    ck = data / "outputs" / "checkpoints" / "paper_v10_same_record_lr1e-4"
    ck.mkdir(parents=True)
    (ck / "stage1_best_ckpt.json").write_text("{}")
    if archive_missing is not None:
        archive_sources = [ck.parent / "stage2_full.pt", data / "outputs" / "variant_ckpts.json",
                           logs / "train_stage2_full.jsonl", data / "outputs" / "stage2_gallery_index.json"]
        for artifact in archive_sources:
            if artifact.name != archive_missing:
                artifact.write_text("{}")
    stub = tmp_path / "python_stub"
    stub.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["STUB_CALLS"], "a") as fh:
    fh.write(json.dumps(args) + "\\n")
if args == ["-"]:
    sys.stdin.read()
print("FutureWarning")
if os.environ["STUB_FAIL"] in args:
    raise SystemExit(23)
''')
    stub.chmod(0o700)
    calls = tmp_path / "calls.jsonl"
    env = {**os.environ, "METAFIND_REPO": str(tmp_path),
           "METAFIND_PYTHON": str(stub), "METAFIND_DATA": str(data),
           "STUB_CALLS": str(calls), "STUB_FAIL": failure}
    result = subprocess.run(["bash", str(root / "tools" / f"chain_paper_stage{stage}.sh")],
                            env=env, capture_output=True, text=True, timeout=10)
    seen = [json.loads(line) for line in calls.read_text().splitlines()]
    return result, seen


@pytest.mark.parametrize("failure,forbidden", [
    ("metafind.models.resolve_stage1", "metafind.data.encode_text_image"),
    ("metafind.data.encode_text_image", "metafind.data.splits"),
    ("metafind.train.stage1", "metafind.train.gallery_index"),
    ("metafind.eval.run_retrieval", None)])
def test_stage1_stops_on_producer_failure_even_if_output_is_filtered(tmp_path, failure, forbidden):
    result, calls = run_chain(tmp_path, 1, failure)
    assert result.returncode != 0
    assert any(failure in args for args in calls)
    assert forbidden is None or all(forbidden not in args for args in calls)
    assert "=== DONE" not in result.stdout
    assert "PIPELINE FAILED" in result.stderr


def test_successful_warning_only_commands_are_not_mistaken_for_failure(tmp_path):
    result, calls = run_chain(tmp_path, 1, "not-a-command")
    assert result.returncode == 0, result.stderr
    assert any("metafind.eval.run_retrieval" in args for args in calls)
    assert "=== DONE (row 1)" in result.stdout


def test_stage2_stops_at_semantic_encoder_failure(tmp_path):
    result, calls = run_chain(tmp_path, 2, "metafind.data.semantic_edges_run")
    assert result.returncode != 0
    assert any("metafind.data.semantic_edges_run" in args for args in calls)
    assert all("metafind.data.scene_splits" not in args for args in calls)
    assert "=== DONE" not in result.stdout


@pytest.mark.parametrize("missing", ["stage2_full.pt", "variant_ckpts.json",
                                     "train_stage2_full.jsonl", "stage2_gallery_index.json"])
def test_stage2_does_not_probe_an_old_archive_after_a_copy_failure(tmp_path, missing):
    result, calls = run_chain(tmp_path, 2, "not-a-command", archive_missing=missing)
    assert result.returncode != 0
    assert any("metafind.train.stage2" in args for args in calls)
    assert all("tools/probes/stage2_procthor_retrieval.py" not in args for args in calls)
    assert "=== DONE" not in result.stdout


def execute_manual_exclusions(tmp_path, *, source_path=None, prelude=""):
    """Execute the actual shell heredoc against files in a disposable corpus."""
    root = Path(__file__).resolve().parents[2]
    script = (root / "tools" / "chain_paper_stage1.sh").read_text()
    body = script.split("$PY - <<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]
    ledger = source_path or tmp_path / "exclusions.json"
    source = ledger.read_bytes()
    env = {**os.environ, "METAFIND_DATA": str(tmp_path / "data"), "METAFIND_EXCLUSION_LEDGER": str(ledger),
           "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-"], input=prelude + "\n" + body, env=env,
                            capture_output=True, text=True, timeout=10)
    assert ledger.read_bytes() == source
    return result


def run_manual_exclusions(tmp_path, manual, *, destinations=(), count=None, setup=None, prelude=""):
    data = tmp_path / "data"
    ann = data / "outputs" / "annotations"
    ann.mkdir(parents=True)
    for uid in ("1" * 32, "2" * 32, "3" * 32, "4" * 32):
        (ann / f"{uid}.json").write_bytes(uid.encode())
    archive = ann.parent / "annotations_v10_excluded"
    if destinations:
        archive.mkdir()
        for uid in destinations:
            (archive / f"{uid}.json").write_bytes(b"previously reviewed")
    ledger = tmp_path / "exclusions.json"
    ledger.write_text(json.dumps({"decided_at": "2026-08-28T14:44:25+08:00", "decided_by": "Kyzen",
        "decision": "fixture manual review decision; old failures are retried", "git_commit": "a" * 40,
        "corpus_before": 99, "corpus_after": 98, "rendered_assets": 100, "groups": {
        "manual_review_rejected": {"n": len(manual) if count is None else count, "uids": manual},
        "n05_quarantine": {"n": 1, "uids": ["3" * 32]}},
        "excluded_total": len(manual) + 1}))
    if setup is not None:
        setup(ledger, ann, archive)
    result = execute_manual_exclusions(tmp_path, prelude=prelude)
    return result, ann, archive


def test_manual_exclusion_heredoc_handles_records_and_keeps_retried_failures(tmp_path):
    result, ann, archive = run_manual_exclusions(
        tmp_path, [{"uid": "1" * 32, "lvis": "chair"}, "2" * 32, {"uid": "5" * 32}])
    assert result.returncode == 0, result.stderr
    assert "moved 2 of 3" in result.stdout
    assert sorted(p.stem for p in ann.iterdir()) == ["3" * 32, "4" * 32]
    assert sorted(p.stem for p in archive.iterdir()) == ["1" * 32, "2" * 32]
    for uid in ("1" * 32, "2" * 32):
        assert (archive / f"{uid}.json").read_bytes() == uid.encode()
    persisted = json.loads((ann.parent / "annotation_exclusions.json").read_bytes())
    assert persisted["schema"] == "metafind.annotation_exclusions.v1"
    assert persisted["accounting_decision"] == "DL-106"
    assert persisted["excluded_total"] == 3
    assert persisted["groups"] == {"manual_review_rejected": {"n": 3, "uids": ["1" * 32, "2" * 32, "5" * 32]}}
    assert not {"corpus_before", "corpus_after", "rendered_assets"} & persisted.keys()
    source = tmp_path / "exclusions.json"
    source_value = json.loads(source.read_bytes())
    assert persisted["source_ledger"] == {
        "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        **{k: source_value[k] for k in ("decided_at", "decided_by", "decision", "git_commit")}}


@pytest.mark.parametrize("manual,count", [
    ([{"uid": "1" * 32}, {"uid": "../invalid"}], None),
    ([{"uid": "1" * 32}], 2),
    ([{"lvis": "missing uid"}], None),
])
def test_manual_exclusion_validates_all_entries_before_moving(tmp_path, manual, count):
    result, ann, archive = run_manual_exclusions(tmp_path, manual, count=count)
    assert result.returncode != 0
    assert len(list(ann.iterdir())) == 4
    assert not archive.exists()
    assert not (ann.parent / "annotation_exclusions.json").exists()


def test_manual_exclusion_does_not_overwrite_or_partially_move_on_conflict(tmp_path):
    result, ann, archive = run_manual_exclusions(
        tmp_path, ["1" * 32, {"uid": "2" * 32}], destinations=["2" * 32])
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert len(list(ann.iterdir())) == 4
    assert not (archive / f"{'1' * 32}.json").exists()
    assert (archive / f"{'2' * 32}.json").read_bytes() == b"previously reviewed"
    assert not (ann.parent / "annotation_exclusions.json").exists()


@pytest.mark.parametrize("reformat", [False, True])
def test_manual_ledger_same_value_is_reused_without_rewriting(tmp_path, reformat):
    result, ann, archive = run_manual_exclusions(tmp_path, [{"uid": "1" * 32}])
    assert result.returncode == 0, result.stderr
    target = ann.parent / "annotation_exclusions.json"
    if reformat:
        target.write_text(json.dumps(json.loads(target.read_bytes()), sort_keys=True, separators=(",", ":")))
    before = target.read_bytes(), target.stat().st_ino, target.stat().st_mtime_ns
    moved = (archive / f"{'1' * 32}.json").read_bytes()
    rerun = execute_manual_exclusions(tmp_path)
    assert rerun.returncode == 0, rerun.stderr
    assert "moved 0 of 1" in rerun.stdout
    assert (target.read_bytes(), target.stat().st_ino, target.stat().st_mtime_ns) == before
    assert (archive / f"{'1' * 32}.json").read_bytes() == moved


@pytest.mark.parametrize("fault", ["different", "symlink", "hardlink", "part", "part_symlink", "directory"])
def test_manual_ledger_conflicts_rejected_before_any_move(tmp_path, fault):
    def setup(source, ann, archive):
        target = ann.parent / "annotation_exclusions.json"
        if fault == "different": target.write_text('{"groups": {}}')
        elif fault == "symlink": target.symlink_to(source)
        elif fault == "hardlink": os.link(source, target)
        elif fault == "part": target.with_suffix(".json.part").write_bytes(b"unfinished evidence")
        elif fault == "part_symlink": target.with_suffix(".json.part").symlink_to(source)
        else: target.mkdir()
    result, ann, archive = run_manual_exclusions(tmp_path, ["1" * 32], setup=setup)
    assert result.returncode != 0
    assert len(list(ann.iterdir())) == 4
    assert not archive.exists()
    target = ann.parent / "annotation_exclusions.json"
    if fault == "different": assert target.read_text() == '{"groups": {}}'
    if fault == "part": assert target.with_suffix(".json.part").read_bytes() == b"unfinished evidence"


@pytest.mark.parametrize("missing", ["decided_at", "decided_by", "decision", "git_commit"])
def test_manual_source_requires_decision_provenance_before_publication(tmp_path, missing):
    def setup(source, ann, archive):
        value = json.loads(source.read_bytes())
        del value[missing]
        source.write_text(json.dumps(value))
    result, ann, archive = run_manual_exclusions(tmp_path, ["1" * 32], setup=setup)
    assert result.returncode != 0
    assert len(list(ann.iterdir())) == 4
    assert not archive.exists()
    assert not (ann.parent / "annotation_exclusions.json").exists()


def test_persisted_manual_ledger_keeps_restored_asset_out_of_n09(tmp_path, monkeypatch):
    from metafind.data import splits
    result, ann, archive = run_manual_exclusions(tmp_path, [{"uid": "1" * 32}])
    assert result.returncode == 0, result.stderr
    # Simulate a restored/reannotated sidecar and a complete producer index:
    # n09 must apply the persisted ledger, not rely on the earlier move.
    restored = ann / f"{'1' * 32}.json"
    restored.write_bytes((archive / restored.name).read_bytes())
    logs = ann.parent / "logs"
    logs.mkdir()
    uids = ["1" * 32, "2" * 32, "3" * 32, "4" * 32]
    for name in ("pointclouds", "renders", "annotations"):
        (logs / f"{name}_index.jsonl").write_text("".join(json.dumps({"uid": uid}) + "\n" for uid in uids))
    manifest = tmp_path / "lvis.json"
    manifest.write_text(json.dumps({uid: f"{uid}.glb" for uid in uids}))
    monkeypatch.setattr(splits.paths, "OUTPUTS", ann.parent)
    monkeypatch.setattr(splits.paths, "LOGS", logs)
    monkeypatch.setattr(splits.paths, "LVIS_MANIFEST", manifest)
    assert splits.admitted_uids() == uids[1:]
    assert restored.exists()


def test_manual_ledger_published_before_first_move_so_interruption_preserves_e(tmp_path):
    prelude = """
import shutil
def interrupted_move(*args, **kwargs):
    raise RuntimeError('injected first-move interruption')
shutil.move = interrupted_move
"""
    result, ann, archive = run_manual_exclusions(tmp_path, ["1" * 32], prelude=prelude)
    assert result.returncode != 0 and "injected first-move interruption" in result.stderr
    assert len(list(ann.iterdir())) == 4 and list(archive.iterdir()) == []
    persisted = json.loads((ann.parent / "annotation_exclusions.json").read_bytes())
    from metafind.data.splits import ledger_excluded_uids
    assert ledger_excluded_uids(persisted) == {"1" * 32}
    retry = execute_manual_exclusions(tmp_path)
    assert retry.returncode == 0, retry.stderr
    assert not (ann / f"{'1' * 32}.json").exists()


def test_manual_source_change_during_preflight_cannot_publish_or_move(tmp_path):
    # Return changed bytes on the second source read without touching real
    # fixture bytes. This exercises the producer's final source recheck.
    prelude = """
import os
from pathlib import Path
original_read_bytes = Path.read_bytes
source_reads = 0
def changed_second_read(path):
    global source_reads
    raw = original_read_bytes(path)
    if str(path) == os.environ['METAFIND_EXCLUSION_LEDGER']:
        source_reads += 1
        if source_reads > 1:
            return raw + b' '
    return raw
Path.read_bytes = changed_second_read
"""
    result, ann, archive = run_manual_exclusions(tmp_path, ["1" * 32], prelude=prelude)
    assert result.returncode != 0 and "changed during preflight" in result.stderr
    assert len(list(ann.iterdir())) == 4
    assert not archive.exists()
    assert not (ann.parent / "annotation_exclusions.json").exists()
