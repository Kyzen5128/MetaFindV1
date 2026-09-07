"""Planner batch failures must never masquerade as a completed evaluation set."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import idesign_generate as runner


def test_documented_direct_script_with_upstream_only_pythonpath(tmp_path):
    result = subprocess.run(
        [sys.executable, str(runner.REPO / "tools/idesign_generate.py"), "--help"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": "/home/kyzen/upstream/IDesign"},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--scene-spec-file" in result.stdout


@pytest.mark.parametrize("overrides", [
    {"room_dimensions": [1, 2, 0]}, {"room_dimensions": [1, float("nan"), 3]},
    {"room_dimensions": [1, True, 3]}, {"room_dimensions": [1, 2]},
    {"prompt": " "}, {"source": []}, {"n_objects": 0}, {"n_objects": True},
    {"seed": False}, {"seed": 1.5},
])
def test_bad_batch_rejected_before_endpoint_or_output(tmp_path, monkeypatch, overrides):
    spec = {"prompt": "a room", "room_dimensions": [3, 4, 2.4],
            "n_objects": 2, "seed": None, "source": "test", **overrides}
    path, out = tmp_path / "spec.jsonl", tmp_path / "out"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(runner, "endpoint_model_id", lambda *_: pytest.fail("endpoint queried"))
    assert runner.main(["--scene-spec-file", str(path), "--n-scenes", "1", "--out", str(out)]) == 2
    assert not out.exists()


@pytest.mark.parametrize("count", [0, -1, 3])
def test_invalid_smoke_count_has_no_side_effects(tmp_path, monkeypatch, count):
    out = tmp_path / "out"
    monkeypatch.setattr(runner, "endpoint_model_id", lambda *_: pytest.fail("endpoint queried"))
    assert runner.main(["--n-scenes", str(count), "--out", str(out)]) == 2
    assert not out.exists()


def test_existing_results_not_overwritten(tmp_path, monkeypatch):
    marker = tmp_path / "index.json"
    marker.write_bytes(b"previous scene evidence")
    monkeypatch.setattr(runner, "endpoint_model_id", lambda *_: pytest.fail("endpoint queried"))
    assert runner.main(["--out", str(tmp_path)]) == 2
    assert marker.read_bytes() == b"previous scene evidence"


@pytest.mark.parametrize("fail_one", [True, False])
def test_batch_status_reports_partial_failure(tmp_path, monkeypatch, fail_one):
    repo = tmp_path / "IDesign"
    repo.mkdir()
    (repo / "IDesign.py").write_text("# isolated planner seam")
    monkeypatch.setattr(runner, "verify_patches", lambda *_: [{"name": "test", "applied": True, "sha256": "a"*64}])
    monkeypatch.setattr(runner, "endpoint_model_id", lambda *_: "fixture-model")

    def run_one(repo, workdir, prompt, dims, n_objects, seed=None):
        if fail_one and workdir.name == "scene_0001":
            raise RuntimeError("planner failed")
        (workdir / "scene_graph.json").write_text("[]")
        return {"n_objects_returned": 2, "n_objects_positioned": 2, "wallclock_s": .1}

    monkeypatch.setattr(runner, "run_one", run_one)
    out = tmp_path / "batch"
    rc = runner.main(["--idesign-repo", str(repo), "--out", str(out)])
    status = json.loads((out / "run_status.json").read_text())
    assert rc == int(fail_one)
    assert status["status"] == ("incomplete" if fail_one else "complete")
    assert status["requested"] == 2
    assert status["succeeded"] == 2 - int(fail_one)
    assert status["failed"] == int(fail_one)
    assert len(json.loads((out / "index.json").read_text())) == status["succeeded"]
