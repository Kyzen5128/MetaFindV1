"""Tests for metafind/runlog.py.

The durable record of a run is what a resume and every later audit read. A
record that disagrees with what happened is worse than no record, because
nothing downstream can tell it is wrong.
"""

from __future__ import annotations

import json

import pytest

# --- the record must agree with what actually happened -----------------------

def _rows(tmp_path):
    p = tmp_path / "run_progress.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_a_node_that_fails_by_returning_is_not_recorded_as_a_success(monkeypatch, tmp_path):
    """Measured on disk before this fix: a pass that returned 2, halted the
    chain and never started n05 was recorded `status: SUCCESS, rc: 0`.

    A context manager cannot see its body's return value, so the body has to
    say. What it must never do is stay silent and be called a success.
    """
    from metafind import runlog

    monkeypatch.setattr(runlog.paths, "LOGS", tmp_path)

    def node() -> int:
        with runlog.run_progress("n_test") as progress:
            progress.rc = 2
            return 2

    assert node() == 2
    exit_row = _rows(tmp_path)[-1]
    assert exit_row["status"] == "FAILED"
    assert exit_row["rc"] == 2


def test_a_node_that_returns_zero_is_still_a_success(monkeypatch, tmp_path):
    """The negative case: the handle defaults to 0, so nothing that succeeds
    has to remember to say so."""
    from metafind import runlog

    monkeypatch.setattr(runlog.paths, "LOGS", tmp_path)
    with runlog.run_progress("n_test"):
        pass
    exit_row = _rows(tmp_path)[-1]
    assert exit_row["status"] == "SUCCESS"
    assert exit_row["rc"] == 0


def test_an_exception_outranks_whatever_the_body_set(monkeypatch, tmp_path):
    """A body that sets rc = 0 and then raises has still failed."""
    from metafind import runlog

    monkeypatch.setattr(runlog.paths, "LOGS", tmp_path)
    with pytest.raises(RuntimeError):
        with runlog.run_progress("n_test") as progress:
            progress.rc = 0
            raise RuntimeError("boom")
    exit_row = _rows(tmp_path)[-1]
    assert exit_row["status"] == "FAILED"
    assert exit_row["rc"] == 1


# --- the positional half of R-32, guarded structurally ------------------------

def test_no_node_returns_non_zero_after_its_run_progress_block_closes():
    """`progress.rc` closes the syntactic hole; this closes the positional one.

    A `return 2` INSIDE the block can set the handle first. A `return 2` AFTER
    it cannot: the row is already on disk saying SUCCESS. That is how R-32
    survived in n03 and n04 after the handle was added -- `return 0 if done or
    not todo else 2` sat thirty lines below the block in both, and a run that
    produced nothing halted the chain while its durable record said it worked.

    An architectural test rather than a per-node one, because the next node
    written will have the same shape available and nothing else would notice.
    """
    import ast
    import pathlib

    ALLOWED = {
        # renders.py's systemic path leaves the block by RAISING
        # SystemicFailure, so the row already says FAILED / rc 1 before this
        # return is reached. 3 is the process exit code, not the record.
        ("metafind/data/renders.py", "3"),
    }

    offenders = []
    for path in sorted(pathlib.Path("metafind").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            blocks = [w for w in ast.walk(fn)
                      if isinstance(w, ast.With)
                      and any(isinstance(i.context_expr, ast.Call)
                              and getattr(i.context_expr.func, "attr", "") == "run_progress"
                              for i in w.items)]
            if not blocks:
                continue
            end = max(max(getattr(n, "lineno", w.lineno) for n in ast.walk(w))
                      for w in blocks)
            for ret in [n for n in ast.walk(fn) if isinstance(n, ast.Return)]:
                if ret.lineno <= end:
                    continue
                src = ast.unparse(ret.value) if ret.value is not None else "None"
                if src in ("0", "None", "progress.rc"):
                    continue
                if (str(path), src) in ALLOWED:
                    continue
                offenders.append(f"{path}:{ret.lineno} return {src}")

    assert offenders == [], (
        "these return non-zero after run_progress has already written its row, "
        "so the record says SUCCESS for a failed run: " + "; ".join(offenders))
