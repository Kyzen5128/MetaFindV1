"""Every metrics row carries the run and the code state that produced it.

The defect these cover, measured on 2026-08-29: `train_stage1.jsonl` held six
Stage 1 attempts -- four killed by machine crashes -- with nothing on any row
to say which attempt it belonged to, and `stage1_best_ckpt.json` named a commit
(`fdfd6a8`) at which its own checkpoint could not have been produced, because
the run used uncommitted gradient checkpointing.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from metafind import paths, runlog


@pytest.fixture()
def fresh(monkeypatch, tmp_path):
    """A runlog whose caches are empty and whose LOGS points at tmp_path."""
    monkeypatch.setattr(runlog, "_RUN_ID", None)
    monkeypatch.setattr(runlog, "_REVISION", None)
    monkeypatch.setattr(runlog, "_DIRTY", None)
    monkeypatch.setattr(paths, "LOGS", tmp_path)
    return tmp_path


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_every_train_metrics_row_carries_run_and_code_state(fresh):
    runlog.train_metrics("t", epoch=0, step=20, loss=1.0)
    runlog.train_metrics("t", epoch=0, step=40, loss=0.9)
    rows = _rows(fresh / "train_t.jsonl")
    assert len(rows) == 2
    for r in rows:
        assert r["run_id"], "a row with no run_id cannot be attributed"
        assert "code_revision" in r
        assert "code_dirty" in r


def test_run_id_is_stable_inside_a_run(fresh):
    runlog.train_metrics("t", step=20)
    runlog.train_metrics("t", step=40)
    ids = {r["run_id"] for r in _rows(fresh / "train_t.jsonl")}
    assert len(ids) == 1, f"one process must emit one run_id, got {ids}"


def test_two_runs_appending_to_one_file_stay_separable(fresh, monkeypatch):
    """The reader-facing property. This is the whole point of the change.

    Before the fix the only separator was `step` jumping backwards, so a run
    that crashed and restarted produced a file whose rows could not be split
    without guessing. Here run A reaches step 40 and dies; run B restarts from
    step 20. Grouping by run_id must recover exactly the two runs -- and note
    that the step sequence 20,40,20 is genuinely ambiguous on its own, which is
    why the assertion below is on run_id and not on step.
    """
    runlog.train_metrics("t", step=20, loss=3.0)
    runlog.train_metrics("t", step=40, loss=2.9)
    a = runlog._RUN_ID
    monkeypatch.setattr(runlog, "_RUN_ID", None)          # new process
    runlog.train_metrics("t", step=20, loss=3.0)
    b = runlog._RUN_ID

    assert a != b, "a restart must not reuse the previous run's id"
    grouped = {}
    for r in _rows(fresh / "train_t.jsonl"):
        grouped.setdefault(r["run_id"], []).append(r["step"])
    assert grouped == {a: [20, 40], b: [20]}


def test_run_id_differs_across_processes(fresh):
    """pid alone collides after a pid wrap; time alone collides inside a second.

    Two interpreters started back to back is the case the ladder script
    actually produces, so it is the one worth spending a subprocess on.
    """
    code = "from metafind import runlog; print(runlog.run_id())"
    out = [subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(paths.REPO)).stdout.strip()
           for _ in range(2)]
    assert all(out), f"run_id() produced nothing: {out}"
    assert out[0] != out[1], f"two processes shared a run_id: {out}"


def test_code_dirty_reports_the_tree_not_a_constant(fresh):
    """A field that is always False is worse than no field: it reads as proof.

    Asserting the CURRENT tree's state would make this test track whoever last
    edited the repo, so it asserts the mechanism instead -- that the answer is
    read from `git status` output rather than hardcoded.
    """
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=" M metafind/x.py\n", stderr="")

    import metafind.runlog as rl
    real = rl.subprocess.run
    rl.subprocess.run = fake_run
    try:
        assert rl.code_dirty() is True
        assert "status" in seen["cmd"] and "--porcelain" in seen["cmd"]

        rl._DIRTY = None
        rl.subprocess.run = lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr="")
        assert rl.code_dirty() is False
    finally:
        rl.subprocess.run = real


def test_code_dirty_is_none_when_git_cannot_answer(fresh):
    """None and False must not collapse: one is "clean", one is "unknown"."""
    import metafind.runlog as rl
    real = rl.subprocess.run
    rl.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(OSError("no git"))
    try:
        assert rl.code_dirty() is None
    finally:
        rl.subprocess.run = real
