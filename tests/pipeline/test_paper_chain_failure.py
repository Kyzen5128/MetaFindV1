"""Exercise the queued shell scripts without models or real corpus writes."""
import json
import os
from pathlib import Path
import subprocess

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
