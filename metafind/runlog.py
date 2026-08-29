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

import hashlib
import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from metafind import paths

__all__ = ["code_dirty", "code_revision", "cost_ledger",
           "quarantine", "runtime_source_sha256", "runtime_source_status",
           "run_id", "run_progress"]


_REVISION: str | None = None
_DIRTY: bool | None = None
_RUN_ID: str | None = None
_UNAVAILABLE = object()   # sentinel: tried, failed. Never a digest.
_SOURCE_SHA: object | None = None
_RUN_CONTEXT: dict[str, Any] = {}


def code_revision() -> str:
    """The commit this RUN was produced by. Required in quarantine records.

    [R-19, corrected 2026-08-23 -- ULIP2 Reviewer] This shelled out on every
    call. Six roles and Codex commit into this repository while a node is
    running, so a single n04 pass recorded **13 different revisions**, none of
    which was "the commit that produced these artifacts" -- they were whatever
    HEAD happened to be at each write, mostly other people's documentation.

    The field claims to describe THE RUN, so it is resolved once, at first use,
    and reused. That is the same rule `renders.implementation_fingerprint`
    already follows: capture in the parent, hand the value down, never re-derive
    it per record.

    A cached wrong-looking value is still better than the alternative here: if
    HEAD moves mid-run, one revision that was true at the start beats thirteen
    that were each true for a few seconds and jointly describe nothing.
    """
    global _REVISION
    if _REVISION is None:
        try:
            _REVISION = subprocess.run(
                ["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
            _REVISION = "unknown"
    return _REVISION


def code_dirty() -> bool | None:
    """Whether the working tree carried uncommitted changes when this run began.

    [ULIP2 ENGINEER 2026-08-29, approved by MASTER as a bug fix] `code_revision`
    alone is not provenance. The 5-epoch dev run of 2026-08-29 06:17 was
    produced by `ulip_backbone.py` and `stage1.py` in a MODIFIED working tree --
    gradient checkpointing, which does not exist at HEAD `fdfd6a8` -- and
    nothing in `train_stage1.jsonl` or `stage1_best_ckpt.json` recorded that.
    The checkpoint therefore claimed a commit that could not have produced it,
    and would not have run at all at that commit (batch 64 OOMs without
    checkpointing). A commit hash beside a dirty tree is a false provenance
    claim, not a partial one.

    Resolved once and cached for the same reason `code_revision` is: the answer
    is meant to describe THE RUN, not the instant of each write.

    `None` means the question could not be answered (no git, timeout), which is
    deliberately distinct from `False`.
    """
    global _DIRTY
    if _DIRTY is None:
        try:
            r = subprocess.run(
                ["git", "-C", str(paths.REPO), "status", "--porcelain"],
                capture_output=True, text=True, timeout=5)
            _DIRTY = bool(r.stdout.strip()) if r.returncode == 0 else None
        except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
            _DIRTY = None
    return _DIRTY


def runtime_source_sha256(root: Path | None = None) -> str | None:
    """One digest over the source that actually decides what this run produces.

    [ULIP2 REVIEWER R-33 + CODEX 2026-08-30] Replaces a first attempt that
    hashed `git diff HEAD` plus untracked `*.py` under the package. Codex named
    the defect precisely: that mixture was neither a worktree forensic patch nor
    a runtime program identity. Its tracked half swept in documentation and test
    edits from the whole repository, while its untracked half admitted only
    package Python -- so the same digest answered two different questions, badly.

    The question worth answering is the second one, because it is the one that
    was got wrong on real data: `e25_400w` and `e25_500w` shared revision
    `3f6fdde` and `code_dirty=true`, were reported as a clean repeat, and their
    0.00123 spread was quoted as the noise floor. Their checkpoint records do
    not even carry the same FIELDS, so the program had changed between them.
    That noise floor is withdrawn.

    So: every existing `*.py` under one explicit root, hashed as
    `path\0length\0content` -- the framing is not decoration, without it a
    rename that shifts bytes between two files leaves the digest unchanged.
    Tracked and untracked cross the same boundary, and git is not consulted at
    all: what ran is what is on disk, and a file's tracking status has never
    changed what Python imported.

    Snapshotted at first call, which `main` makes before training. A long run
    whose worktree is edited at hour two must still report the program it
    started with -- the previous version first resolved this at the end of epoch
    one, which is a different claim than the one the field makes.

    Not a substitute for `code_revision`: this says WHAT ran, that says which
    published commit it is nearest. Both are recorded.
    """
    global _SOURCE_SHA
    if _SOURCE_SHA is None:
        try:
            base = root or (paths.REPO / "metafind")
            files = [] if not base.is_dir() else [
                f for f in sorted(base.rglob("*.py"))
                if "__pycache__" not in f.parts]
            # [CODEX MINOR 2026-08-30] `rglob` on a missing directory returns an
            # EMPTY iterator, not an error, so the old code hashed nothing and
            # returned e3b0c442... -- sha256 of the empty string. A perfectly
            # valid-looking digest meaning "there was no source here at all",
            # which every reader would have compared as if it identified a
            # program. Zero files is a failure, and it has to say so.
            if not files:
                raise FileNotFoundError(f"no *.py under {base}")
            h = hashlib.sha256()
            for f in files:
                b = f.read_bytes()
                h.update(str(f.relative_to(base)).encode())
                h.update(b"\0")
                h.update(str(len(b)).encode())
                h.update(b"\0")
                h.update(b)
            _SOURCE_SHA = h.hexdigest()
        except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
            # [ULIP2 REVIEWER 2026-08-30] Cached, not returned bare: a bare
            # `return None` left `_SOURCE_SHA` unset, so a later call recomputed
            # against a worktree that may have been edited since. The docstring
            # promises a snapshot at first call; a retry is a different promise.
            #
            # [CODEX MINOR 2026-08-30] The sentinel is NOT a hash-shaped string.
            # `"unavailable"` sat in a field named `..._sha256`, where anything
            # that reads digests as opaque strings would compare it, index it,
            # or print it as one. `None` cannot be mistaken for a digest, and
            # `runtime_source_status` says why it is absent.
            _SOURCE_SHA = _UNAVAILABLE
    return None if _SOURCE_SHA is _UNAVAILABLE else _SOURCE_SHA


def runtime_source_status(root: Path | None = None) -> str:
    """`ok` if the source snapshot was taken, `unavailable` if it could not be.

    Distinct from `runtime_source_sha256() is None` only in being explicit: a
    null digest with no status reads as "nobody tried".
    """
    runtime_source_sha256(root)
    return "unavailable" if _SOURCE_SHA is _UNAVAILABLE else "ok"


def run_id() -> str:
    """Identifier for THIS process's run, stamped on every metrics row.

    [ULIP2 ENGINEER 2026-08-29] `train_metrics` appends, and the file is named
    for the stage rather than the run, so every attempt at a stage lands in one
    file. On 2026-08-29 six Stage 1 attempts -- four of them killed by machine
    crashes -- shared `train_stage1.jsonl` with nothing to tell them apart. The
    only way to separate them was to notice `step` jumping backwards, and a
    reader doing the obvious thing (`tail`) sees the tail of the LAST attempt
    grafted onto the body of an earlier one and reads it as one curve.

    Stamped per ROW, not written as a header: the reader this protects is the
    one running `tail`, who never sees a header.
    """
    global _RUN_ID
    if _RUN_ID is None:
        # time+pid alone is not unique: `test_two_runs_appending_to_one_file_
        # stay_separable` went red on it. A crash-restart inside the same second
        # can reuse a pid, and that is the exact case this field exists for --
        # the 2026-08-29 crashes restarted Stage 1 four times. The timestamp is
        # kept because it makes ids sort into run order when reading a mixed
        # file; the pid because it ties a row to a live process during a run;
        # the random suffix because only it makes the id actually unique.
        _RUN_ID = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    return _RUN_ID


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


class Progress:
    """The handle `run_progress` yields, so a node can report a non-zero rc.

    A context manager cannot see the value its body returns. `rc = 0` plus
    `except BaseException: rc = 1` therefore recorded SUCCESS for every failure
    that came back as a RETURN rather than a raise -- and a `return 2` leaves
    the block normally.

    Measured in run_progress.jsonl: a pass that returned 2, halted the chain and
    never started n05 is on disk as `status: SUCCESS, rc: 0`. The durable record
    of a failure said the opposite of what happened.

    Nodes that fail by returning set `progress.rc` before returning:

        with runlog.run_progress(NODE) as progress:
            if not path.exists():
                progress.rc = 2
                return 2
    """

    __slots__ = ("rc",)

    def __init__(self) -> None:
        self.rc = 0


@contextmanager
def run_progress(name: str, attempt: int = 1):
    """Bracket a node's execution and emit its run_progress record.

    Writes on the way out whatever happens, including on an exception, because
    a stage that vanished without a record is indistinguishable from one that
    never started -- and that is exactly the state a resume has to interpret.

    Yields a `Progress`; set its `rc` before returning non-zero from inside the
    block, or the record will say the run succeeded.
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
    progress = Progress()
    raised = False
    try:
        yield progress
    except BaseException:
        raised = True
        raise
    finally:
        # An exception always means failure; otherwise the body's own rc stands.
        rc = 1 if raised else int(progress.rc)
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


def set_run_context(**fields: Any) -> None:
    """Fields every subsequent metrics row carries. Called once, by the trainer."""
    _RUN_CONTEXT.update(fields)


def train_metrics(run: str, **fields: Any) -> None:
    """One row per logged training step, for plotting afterwards.

    Separate from ``cost_ledger`` because these are per-STEP and there are
    thousands of them, while the ledger is one row per node run. Separate from
    stdout because a printed line is not data: "loss 3.9214" scrolls past and
    cannot be plotted, differenced, or compared against the previous run.

    ``run`` names the file, so Stage 1, a Stage 2 variant and a re-run at a new
    seed each land in their own series and can be drawn on one axis.

    Every field is optional and written as given. A metric that only exists in
    one stage -- ``lambda`` in Stage 2, ``acc_g2q`` in the bidirectional loss --
    simply appears in the rows where it applies, and the plotter skips the rest.
    """
    _append(paths.LOGS / f"train_{run}.jsonl",
            {"ts": time.time(), "run_id": run_id(),
             "code_revision": code_revision(), "code_dirty": code_dirty(),
             "runtime_source_sha256": runtime_source_sha256(),
             "runtime_source_status": runtime_source_status(),
             # [CODEX 2026-08-30] Stamped from run context, not passed per call
             # site: a loss curve that cannot say which arm and which seed
             # produced it is unusable for the paired-difference analysis the
             # stopping rule is built on, and joining it to a checkpoint record
             # first is a step a reader running `tail` will not take.
             **_RUN_CONTEXT, **fields})


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


# Set once per process. A worker imports its modules at spawn and cannot change
# them afterwards, so one check covers every task that worker will ever run.
_FINGERPRINT_VERIFIED = False


def implementation_fingerprint(*modules, extra_files=()) -> dict[str, str]:
    """sha256 of every source file that decides what a node produces.

    A pool with `max_tasks_per_child` set respawns workers mid-run, and each new
    worker **re-imports**. Editing one of these files while a job is running
    therefore changes what that job produces, silently. Measured 2026-08-22 on
    `n04`: roughly 1,700 assets were rendered with a geometry fix and stamped
    with the version that predated it, **no field in any artifact revealed it**,
    and the corpus was discarded because no subset could be identified by
    inspection.

    The version field caught that by accident, not by design -- had the bump
    landed WITH the fix, the corpus would have been uniformly stamped and every
    gate would have passed. That is the case this exists for.

    Callers pass their own module list. `meshload` belongs in every one that
    loads geometry: it owns `FRAME_CORRECTION`, so a change there moves every
    asset while the calling module does not differ by a byte.
    """
    import hashlib

    out: dict[str, str] = {}
    for mod in modules:
        path = Path(mod.__file__)
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # `extra_files` is for source that decides the output but cannot be imported
    # here -- n04's vendored Blender script runs inside Blender's own Python and
    # importing it in this process would execute it. Its bytes still choose every
    # pixel, so they belong in the fingerprint.
    for path in extra_files:
        path = Path(path)
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def verify_fingerprint(expected: dict[str, str] | None, *modules, extra_files=()) -> None:
    """Abort this worker if its source differs from what the run started on.

    The point is not to forbid editing a module. It is that a run's behaviour
    must not be able to change without the artifacts saying so.
    """
    global _FINGERPRINT_VERIFIED
    if not expected or _FINGERPRINT_VERIFIED:
        return
    actual = implementation_fingerprint(*modules, extra_files=extra_files)
    if actual != expected:
        drift = sorted(k for k in expected if actual.get(k) != expected[k])
        raise RuntimeError(
            f"implementation changed while the run was in progress: {', '.join(drift)}. "
            "This worker would write artifacts the rest of the corpus does not share, "
            "and no sidecar field would show it. Restart the run."
        )
    _FINGERPRINT_VERIFIED = True
