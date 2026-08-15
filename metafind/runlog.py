"""Write the three channels every node is declared to write.

Each helper is NAMED for the channel it writes -- ``run_progress``,
``cost_ledger``, ``quarantine`` -- so that a node's source literally contains
the channel names it claims in the registry, and the structural check that
compares the two is testing something rather than a comment.

``run_progress``, ``cost_ledger`` and ``quarantine`` are declared on almost
every node in the registry, and neither implemented node wrote any of them:
n02 and n03 both listed them in ``writes`` and produced nothing. That is the
same defect the spec reviews kept finding one level up -- a contract that is
declared and not executed -- so it gets one place to live rather than being
re-invented per node.

``run_progress`` in particular is not decoration: L2-RESUME asserts that a
killed and restarted stage does not re-consume work, and there is nothing to
resume from without a record of what finished.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from metafind import paths

__all__ = ["code_revision", "cost_ledger", "quarantine", "run_progress"]


def code_revision() -> str:
    """The commit the artifact was produced by. Required in quarantine records."""
    try:
        return subprocess.run(
            ["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return "unknown"


def _append(path: Path, record: dict[str, Any]) -> None:
    """Append one JSONL row and fsync it.

    A plain buffered append is not a record: the bytes sit in the page cache,
    and the crash these files exist to survive is exactly the event that
    discards them. Appending a single line under O_APPEND is atomic for writes
    below PIPE_BUF, so no temp-and-rename is needed -- but the fsync is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def run_progress(name: str, attempt: int = 1):
    """Bracket a node's execution and emit its run_progress record.

    Writes on the way out whatever happens, including on an exception, because
    a stage that vanished without a record is indistinguishable from one that
    never started -- and that is exactly the state a resume has to interpret.
    """
    started = time.time()
    # A record on the way IN as well as out. The exit record is written in a
    # `finally`, which does not run under kill -9 or a power loss -- so a run
    # that died left no trace of itself at all, and a resume could not tell
    # "never started" from "started and was killed after writing 8,000 files".
    _append(paths.LOGS / "run_progress.jsonl", {
        "stage": name, "status": "RUNNING", "started_at": started,
        "ended_at": None, "attempt": attempt, "rc": None,
        "stdout_broken": False, "code_revision": code_revision(),
    })
    rc = 0
    try:
        yield
    except BaseException:
        rc = 1
        raise
    finally:
        _append(paths.LOGS / "run_progress.jsonl", {
            "stage": name,
            "status": "SUCCESS" if rc == 0 else "FAILED",
            "started_at": started,
            "ended_at": time.time(),
            "attempt": attempt,
            "rc": rc,
            # A pipe that closed under us loses output without losing exit
            # status, so the two are recorded separately.
            "stdout_broken": not os.isatty(1) and not _stdout_writable(),
            "code_revision": code_revision(),
        })


def _stdout_writable() -> bool:
    try:
        os.fstat(1)
        return True
    except OSError:
        return False


def cost_ledger(**resources: float) -> None:
    """Append to cost_ledger. merge is numeric_add, so partial runs accumulate."""
    _append(paths.LOGS / "cost_ledger.jsonl",
            {"timestamp": time.time(), **{k: float(v) for k, v in resources.items()}})


def quarantine(stage_name: str, records: list[dict[str, Any]]) -> Path | None:
    """Append quarantine entries in the shape the channel type declares.

    The type is
    ``{uid, stage, failure_class, exception_type, exception_msg, code_revision,
    timestamp}``; n03's first version wrote ``node`` instead of ``stage`` and
    omitted failure_class and code_revision entirely, so G3 -- which reads this
    channel to compute quarantine_rate -- would have been parsing a different
    shape than the one it was specified against.
    """
    if not records:
        return None
    rev, now = code_revision(), time.time()
    path = paths.LOGS / f"quarantine_{stage_name}.jsonl"
    for r in records:
        _append(path, {"stage": stage_name, "code_revision": rev, "timestamp": now, **r})
    return path
