"""Refuse to let a bad gallery index become the shared immutable artifact.

# IMPLEMENTS-NODE: G4_gallery_freeze

Writes ``gate_records`` -- the YAML at the gate's declared ``record_path`` --
and ``run_progress``. It writes NOTHING else: on FAIL it does not delete, move
or rewrite the staging index, and it never touches the promoted registry. The
spec's ``on_fail`` says rebuild, and a gate that edited the evidence would be
the last thing to have touched the artifact it judged.

Why this gate exists at all
---------------------------
[validation_plan.yaml, gate_class G-CONTAM] The gallery index is the one
artifact every table reads. A corrupted one lowers every metric uniformly and
leaves no signature -- there is no cell that looks wrong, only a whole table
that is quietly lower than it should be.

Ties are LEGAL and the naive check is wrong
-------------------------------------------
[validation_plan.yaml `[CORRECTED]`; node_registry.yaml `[CORRECTED]`] The
self-retrieval criterion was once ``recall@1 == 1.0``. That is not tie-safe:
two assets with identical embeddings make ``argmax`` return either id, so a
CORRECT index can score below 1.0 and a BROKEN one can reach it. The criterion
is now "the target's similarity EQUALS the maximum and the target is a member
of the argmax tie set", which is checked here as two facts about the same row:

* ``target_score == top1_score`` -- own equals the max, so the target is in the
  argmax tie set by definition; and
* ``rank - tie_count - 1 == 0`` -- nothing scores strictly higher.

``score_streaming`` reads ``own`` out of the SAME GEMM in the SAME block shape
as the comparisons, so these are exact comparisons between numbers produced by
one arithmetic path. Ranks are NOT used: ``rank`` counts ties against the
model, which is right for a retrieval metric and wrong for an identity check --
a legally tied index would fail it.

What this criterion CANNOT see
------------------------------
**This criterion cannot detect a collapsed index** -- a fully collapsed gallery
satisfies it in every trial, because every vector ties for the maximum.
MEASURED 2026-08-30 on this scorer: 120 of 120 trials at d=8/ng=30,
d=1280/ng=999 and d=1280/ng=4,569 passed with every row identical. The
``[CORRECTED]`` change to ``validation_plan.yaml`` bought tie-safety and paid in
collapse-blindness -- the OLD ``recall@1 == 1.0`` would have failed a collapsed
index, because rank counts ties against the model. ``effective_rank`` and
``effective_rank_centred`` are recorded for that reason and are **NOT pass
conditions**: whether the criterion itself should change is a specification
decision, not this module's. Do not read a G4 PASS as evidence the index is
healthy.

Arithmetic is n15's, imported, never re-implemented
---------------------------------------------------
``normalize_for_scoring`` and ``score_streaming`` come from the evaluation
modules. A gate that scored the index with its own similarity code would be
certifying an index against arithmetic no reported number uses. The float64
choice is measured -- see ``run_retrieval.encode_pools.norm`` -- and inherited
here rather than restated.

rc 4 (INVALIDATED) is declared upstream and is deliberately never emitted
------------------------------------------------------------------------
``rc_contract`` in ``validation_plan.yaml`` declares ``INVALIDATED: 4``, and the
gate's stated criterion contains no condition that produces it. Inventing a
trigger would put a verdict in the record that no specified condition justifies,
and a later reader would take its existence as evidence this gate can detect
something it cannot. It stays declared and unused until the criterion says
otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from metafind import paths, runlog
from metafind.eval.retrieval import normalize_for_scoring
from metafind.eval.run_retrieval import embedding_health, score_streaming
from metafind.train import gallery_index as gi

paths.setup_env()

GATE_ID = "G4_gallery_freeze"
GATE_CLASS = "G-CONTAM"
RECORD_PATH = paths.LOGS / "gates" / f"{GATE_ID}.yaml"
SPEC_PATH = paths.REPO / "docs" / "graph" / "validation_plan.yaml"

PASS, FAIL, BLOCKED_EVIDENCE, INVALIDATED = 0, 2, 3, 4
VERDICT = {PASS: "PASS", FAIL: "FAIL",
           BLOCKED_EVIDENCE: "BLOCKED_EVIDENCE", INVALIDATED: "INVALIDATED"}
RC_CONTRACT = {"PASS": PASS, "FAIL": FAIL,
               "BLOCKED_EVIDENCE": BLOCKED_EVIDENCE, "INVALIDATED": INVALIDATED}

SAMPLE_SIZE = 1000            # [validation_plan.yaml] "1000 sampled gallery vectors"
EXAMPLES = 5                  # how many offending ids to name in the record

ROW_ORDER_NOTE = (
    "n11 (gallery_index.main) builds ids = sorted(train + test) and appends "
    "vectors in that same loop, so a correct staging index has sorted ids. "
    "Recorded because L2-GALLERY-SELF's declared negative injection -- shuffle "
    "the index id mapping -- does NOT fail the self-retrieval criterion: that "
    "criterion scores each gallery row against itself, so shuffling ids "
    "against rows leaves the arithmetic bit-identical. This is NOT a pass "
    "condition; sorted row order is n11's convention and the stated criterion "
    "does not mention it. Nothing inside the index can detect a shuffle that "
    "leaves the ids sorted -- that needs a re-encode against the checkpoint."
)

COLLAPSE_NOTE = (
    "This criterion cannot detect a collapsed index -- a fully collapsed "
    "gallery satisfies it in every trial, because every vector ties for the "
    "maximum. The [CORRECTED] change to validation_plan.yaml bought tie-safety "
    "and paid in collapse-blindness. effective_rank is recorded here for that "
    "reason and is NOT a pass condition."
)


class _Blocked(Exception):
    """Evidence is missing or unreadable: rc 3, never rc 0, never a skip."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_lines(seq) -> str:
    """Digest of a sequence of ids, one per line.

    Used two ways and the difference is the point: over ``sorted(uids)`` it
    identifies a SET, over a sample in its own order it identifies a SEQUENCE.
    A seed alone does not make a sample auditable -- it names a recipe, and the
    recipe's output depends on the RNG implementation. The digest names the
    result.
    """
    h = hashlib.sha256()
    for item in seq:
        h.update(str(item).encode())
        h.update(b"\n")
    return h.hexdigest()


def spec(spec_path: Path | None = None) -> dict:
    """This gate's own entry in the validation plan.

    The criterion is QUOTED from the spec rather than restated in code, so the
    record says what the gate was run against and the two cannot drift. A gate
    whose code and spec disagree about its rc contract cannot honour either.
    """
    path = Path(spec_path) if spec_path else SPEC_PATH
    try:
        gates = yaml.safe_load(path.read_text())["level_3_gates"]
    except Exception as exc:  # noqa: BLE001 -- any unreadable spec is blocked
        raise _Blocked(f"cannot read the gate spec at {path}: {exc}") from exc
    for g in gates:
        if g.get("gate_id") == GATE_ID:
            if g.get("rc_contract") != RC_CONTRACT:
                raise _Blocked(
                    f"{path} declares rc_contract {g.get('rc_contract')} but "
                    f"this gate implements {RC_CONTRACT}")
            return g
    raise _Blocked(f"{path} has no gate {GATE_ID}")


def checkpoint_embedding_dim(weights_uri: Path) -> int:
    """The width the checkpoint's GALLERY tower emits, read off the checkpoint.

    ``gallery.fusion.mask_tokens`` is ``(len(MODALITIES), cfg.dim)`` and
    ``metafind/models/fusion.py`` constructs it UNCONDITIONALLY for all five
    fusion kinds -- "always constructed so a checkpoint can be loaded under
    either setting" -- so its last axis is the model width whatever
    ``FusionConfig.kind`` was. ``build_model`` passes one ``dim`` to both towers.

    Read from the checkpoint rather than from ``ulip_backbone.EMBED_DIM``
    because the question is what THESE weights emit, not what the current
    source would build. MEASURED 2026-08-30 on the real 323 MB
    ``stage1.pt``: ``mmap=True, weights_only=True`` returned in 0.01 s. That is
    evidence the load is not eager; it is not a claim about torch's internals.
    """
    import torch

    try:
        state = torch.load(Path(weights_uri), map_location="cpu",
                           weights_only=True, mmap=True)["tower_trainable_state"]
        return int(state["gallery.fusion.mask_tokens"].shape[-1])
    except Exception as exc:  # noqa: BLE001
        raise _Blocked(
            f"cannot read the gallery width from {weights_uri}: {exc}. The "
            "checkpoint must carry gallery.fusion.mask_tokens; without it the "
            "index's width cannot be compared to anything.") from exc


def _admitted_ids(splits_path: Path) -> tuple[list[str], dict]:
    """The admitted asset set, DERIVED from the split artifact.

    Never a literal. 45,692 is a fact about today's corpus, and a gate carrying
    it would keep passing an index of the wrong size the day the corpus moves,
    while its own criterion says "len(admitted)".
    """
    try:
        splits = json.loads(Path(splits_path).read_text())
        listed = list(splits["object"]["train"]) + list(splits["object"]["test"])
    except Exception as exc:  # noqa: BLE001
        raise _Blocked(f"cannot read the admitted set from {splits_path}: "
                       f"{exc}") from exc
    if not isinstance(splits.get("split_seed"), int):
        # The sample seed comes from the split artifact, so a split with no seed
        # leaves the "fixed 1000" undefined. Blocked, not defaulted: a silently
        # chosen seed would make the recorded sample digest unreproducible.
        raise _Blocked(f"{splits_path} has no integer split_seed; the fixed "
                       "sample has no reproducible definition")
    admitted = sorted(set(listed))
    return admitted, {
        "splits_path": str(splits_path),
        "n_train": len(splits["object"]["train"]),
        "n_test": len(splits["object"]["test"]),
        "n_admitted_listed": len(listed),
        "n_admitted_unique": len(admitted),
        "splits_admitted_total": splits.get("admitted_total"),
        "split_seed": splits.get("split_seed"),
    }


def _self_retrieval(ids: list[str], gallery: np.ndarray, seed: int,
                    sample_size: int, block: int) -> dict:
    """[L2-GALLERY-SELF] Each sampled vector must retrieve itself, ties allowed.

    The sample is fixed by ``seed`` and pinned by ``sample_uid_sequence_sha256``:
    the seed says how it was drawn, the digest says what was drawn. Sorted into
    index order so the digest identifies the sampled SET independently of the
    draw order.
    """
    n = gallery.shape[0]
    k = min(sample_size, n)
    sample = np.sort(np.random.default_rng(seed).permutation(n)[:k])
    sample_uids = [ids[i] for i in sample]

    scored = score_streaming(gallery[sample], gallery,
                             sample.astype(np.int64), block=block)

    target_is_max = scored["target_score"] == scored["top1_score"]
    strictly_higher = scored["rank"] - scored["tie_count"] - 1
    ok = target_is_max & (strictly_higher == 0)
    bad = np.nonzero(~ok)[0]
    return {
        "sample_size": int(k),
        "sample_seed": int(seed),
        "sample_uid_sequence_sha256": _sha256_lines(sample_uids),
        "n_target_is_max": int(target_is_max.sum()),
        "n_zero_strictly_higher": int((strictly_higher == 0).sum()),
        "n_failed": int(bad.size),
        # Ties are LEGAL. Recorded so a PASS says how many there were rather
        # than leaving "did any occur?" unanswerable after the fact.
        "n_with_ties": int((scored["tie_count"] > 0).sum()),
        "max_tie_count": int(scored["tie_count"].max()) if k else 0,
        "failed_examples": [
            {"asset_id": sample_uids[i],
             "target_score": float(scored["target_score"][i]),
             "top1_score": float(scored["top1_score"][i]),
             "n_strictly_higher": int(strictly_higher[i])}
            for i in bad[:EXAMPLES]],
    }


def history_path(record_path: Path) -> Path:
    """The append-only history beside a gate record. Derived, so they move together."""
    return record_path.with_name(record_path.stem + ".history.yaml")


def _append_history(record_path: Path, record: dict) -> None:
    """Append this record to the persistent history. Never a verdict, never an rc.

    THE TWO SPECS DESCRIBE THIS CHANNEL DIFFERENTLY, and writing both files
    satisfies both rather than picking a winner. Do not collapse this to one:

    * ``graph_spec.yaml`` ``gate_records``: ``type: "list[gate_record]"``,
      ``merge: append``, ``lifetime: persistent``.
    * ``validation_plan.yaml``: one ``record_path``, one record's
      ``record_fields``.

    Neither is wrong on its own terms. What is indefensible under EITHER is the
    behaviour they produced together: ``record_path`` alone is overwritten, so a
    FAIL followed by a re-run PASS destroyed the FAIL -- and the project's
    experiment rules say in as many words that a failed run is evidence and must
    not be silently discarded. Round 2 on the real 45,692-asset index can
    produce exactly that sequence. So ``record_path`` keeps the CURRENT terminal
    record in the shape the validation plan describes -- n12 reads that and only
    that -- and every record, PASS, FAIL and BLOCKED alike, is also appended
    here as the declared ``list[gate_record]``.

    Read-modify-write through ``gi._write`` (temp-and-rename plus fsync) rather
    than an O_APPEND line: a gate record is far larger than PIPE_BUF, so an
    append is not atomic, and the one thing this file must never do is lose a
    record. The rewrite is O(n) in the number of gate runs, which is a handful.

    A history that does not parse as a list is LEFT ALONE. Overwriting it would
    be this function committing the exact deletion it exists to prevent. It says
    so on stdout and returns; it does not change the verdict either way.
    """
    path = history_path(record_path)
    prior: list = []
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            print(f"  WARNING: {path} is not readable YAML ({exc}); leaving it "
                  "untouched. This record is in the terminal record and NOT in "
                  "the history.", flush=True)
            return
        if not isinstance(loaded, list):
            print(f"  WARNING: {path} does not hold a list; leaving it "
                  "untouched. This record is in the terminal record and NOT in "
                  "the history.", flush=True)
            return
        prior = loaded
    gi._write(path, prior + [record], dump=yaml.safe_dump)


def run(staging_path: Path | None = None, splits_path: Path | None = None,
        record_path: Path | None = None, spec_path: Path | None = None,
        sample_size: int = SAMPLE_SIZE, block: int = 4096) -> int:
    """Judge the staging index and write the gate record. Returns the rc.

    Fail-closed everywhere: an unreadable input is BLOCKED_EVIDENCE (3), never
    a PASS and never a silent skip. [L1-GATE-NORECORD] a missing record counts
    as not passed, so the record is written for every verdict including 3.
    """
    staging_path = Path(staging_path) if staging_path else gi.STAGING_PATH
    splits_path = Path(splits_path) if splits_path else paths.OUTPUTS / "splits.json"
    record_path = Path(record_path) if record_path else RECORD_PATH

    failures: list[str] = []
    observed: dict = {}
    inputs: dict = {"staging_index": str(staging_path), "splits": str(splits_path)}
    criterion = ""
    scope = str(staging_path)
    self_retrieval: dict | None = None

    try:
        criterion = str(spec(spec_path)["criterion"]).strip()

        # ---- evidence: the staging record -------------------------------
        if not staging_path.exists():
            raise _Blocked(f"{staging_path} not found -- run n11 first")
        try:
            staging = json.loads(staging_path.read_text())
        except json.JSONDecodeError as exc:
            raise _Blocked(f"{staging_path} is not readable JSON: {exc}") from exc
        if len(staging) != 1:
            # n11 replaces the whole file, so exactly one entry is what it
            # writes. More than one means something else edited it, and this
            # gate would have to choose which index its verdict covers.
            raise _Blocked(
                f"{staging_path} holds {len(staging)} entries; a gate record "
                "names one index and cannot cover an ambiguous set")
        stage1_sha, staged = next(iter(staging.items()))
        staging_record_sha256 = _sha256_file(staging_path)
        inputs["staging_record_sha256"] = staging_record_sha256
        scope = f"gallery_index_staging[{stage1_sha}]"
        observed["staging_record_sha256"] = staging_record_sha256
        observed["staging_map_key"] = stage1_sha
        for field in ("uri", "sha256", "dim", "count",
                      "stage1_checkpoint_sha256", "gallery_encoder_sha256",
                      "stage1_ckpt_record"):
            if field not in staged:
                raise _Blocked(
                    f"the staging record has no {field!r}; it cannot say which "
                    "artifact this gate would be certifying")
        observed["index_uri"] = str(staged["uri"])
        observed["staging_record_dim"] = int(staged["dim"])
        observed["staging_record_count"] = int(staged["count"])
        observed["gallery_encoder_sha256"] = str(staged["gallery_encoder_sha256"])
        observed["stage1_checkpoint_sha256_in_staging"] = \
            str(staged["stage1_checkpoint_sha256"])
        inputs["index_uri"] = str(staged["uri"])
        if stage1_sha != staged["stage1_checkpoint_sha256"]:
            failures.append(
                f"the staging map is keyed {stage1_sha[:12]}... but the record "
                f"inside says {staged['stage1_checkpoint_sha256'][:12]}...")

        # ---- 1. index bytes hash to what the staging record claims -------
        index_uri = Path(staged["uri"])
        if not index_uri.exists():
            raise _Blocked(f"the staging record names {index_uri}, which does "
                           "not exist")
        on_disk = _sha256_file(index_uri)
        observed["index_sha256_on_disk"] = on_disk
        observed["index_sha256_in_staging_record"] = str(staged["sha256"])
        observed["index_sha256_match"] = bool(on_disk == staged["sha256"])

        ids: list[str] | None = None
        embeddings: np.ndarray | None = None
        if not observed["index_sha256_match"]:
            failures.append(
                f"index bytes hash to {on_disk[:12]}... but the staging record "
                f"records {str(staged['sha256'])[:12]}...")
        else:
            try:
                ids, embeddings = gi.verified_index(staged, str(staging_path))
            except gi.IndexUnreadable as exc:
                # Nobody knows whether this index was good: rc 3, not rc 2.
                raise _Blocked(f"unreadable index {index_uri}: {exc}") from exc
            except ValueError as exc:
                # The record describes an index other than the one it points at
                # -- a contract violation, not missing evidence.
                failures.append(f"staging record does not describe its index: "
                                f"{exc}")

        # ---- 2. the checkpoint chain -------------------------------------
        ckpt_record_path = Path(staged["stage1_ckpt_record"])
        inputs["stage1_ckpt_record"] = str(ckpt_record_path)
        ckpt_record = None
        try:
            # Reused, not re-implemented: this already refuses a record whose
            # sha256 does not match the weights it names.
            ckpt_record = gi.load_checkpoint_record(ckpt_record_path)
        except FileNotFoundError as exc:
            raise _Blocked(str(exc)) from exc
        except ValueError as exc:
            failures.append(f"checkpoint record does not match its weights: "
                            f"{exc}")
        if ckpt_record is not None:
            observed["checkpoint_uri"] = str(ckpt_record["uri"])
            observed["checkpoint_record_sha256"] = str(ckpt_record["sha256"])
            observed["checkpoint_weights_sha256_verified"] = True
            if staged["stage1_checkpoint_sha256"] != ckpt_record["sha256"]:
                failures.append(
                    f"the staging record was built from checkpoint "
                    f"{str(staged['stage1_checkpoint_sha256'])[:12]}... but "
                    f"{ckpt_record_path} records "
                    f"{str(ckpt_record['sha256'])[:12]}...")
        else:
            observed["checkpoint_weights_sha256_verified"] = False

        # ---- 5. width equals the checkpoint's gallery width ---------------
        if ckpt_record is not None:
            ckpt_dim = checkpoint_embedding_dim(Path(ckpt_record["uri"]))
            observed["checkpoint_embedding_dim"] = ckpt_dim
            if embeddings is not None:
                observed["index_dim"] = int(embeddings.shape[1])
                if int(embeddings.shape[1]) != ckpt_dim:
                    failures.append(
                        f"index width {embeddings.shape[1]} != the "
                        f"checkpoint's gallery width {ckpt_dim}")
            if int(staged["dim"]) != ckpt_dim:
                failures.append(
                    f"the staging record claims dim {staged['dim']} != the "
                    f"checkpoint's gallery width {ckpt_dim}")

        # ---- 3 + 4. ids are unique and are exactly the admitted set -------
        admitted, split_obs = _admitted_ids(splits_path)
        observed.update(split_obs)
        observed["expected_uid_set_sha256"] = _sha256_lines(admitted)
        if split_obs["n_admitted_listed"] != split_obs["n_admitted_unique"]:
            failures.append(
                f"{splits_path} lists {split_obs['n_admitted_listed']} ids for "
                f"{split_obs['n_admitted_unique']} unique assets; the admitted "
                "set is not well defined")
        if split_obs["splits_admitted_total"] != split_obs["n_admitted_unique"]:
            failures.append(
                f"{splits_path} records admitted_total="
                f"{split_obs['splits_admitted_total']} but train+test hold "
                f"{split_obs['n_admitted_unique']} unique assets")

        if ids is not None:
            counts = Counter(ids)
            dupes = sorted(u for u, c in counts.items() if c > 1)
            missing = sorted(set(admitted) - counts.keys())
            extra = sorted(counts.keys() - set(admitted))
            observed.update({
                "n_index_ids": len(ids),
                "n_unique_index_ids": len(counts),
                "index_uid_set_sha256": _sha256_lines(sorted(counts)),
                "n_duplicate_index_ids": len(dupes),
                "duplicate_examples": dupes[:EXAMPLES],
                "n_missing_from_index": len(missing),
                "missing_examples": missing[:EXAMPLES],
                "n_extra_in_index": len(extra),
                "extra_examples": extra[:EXAMPLES],
            })
            if dupes:
                failures.append(f"{len(dupes)} asset id(s) appear more than "
                                f"once, e.g. {dupes[:EXAMPLES]}")
            if missing:
                failures.append(f"{len(missing)} admitted asset(s) are absent "
                                f"from the index, e.g. {missing[:EXAMPLES]}")
            if extra:
                failures.append(f"{len(extra)} indexed asset(s) are not "
                                f"admitted, e.g. {extra[:EXAMPLES]}")
            if len(ids) != len(admitted):
                failures.append(f"index count {len(ids)} != len(admitted) "
                                f"{len(admitted)}")
            # RECORDED, NOT A PASS CONDITION. `gallery_index.main` builds
            # `ids = sorted(train + test)` and appends `vectors` in that same
            # loop, so a correct staging index has sorted ids. This is the one
            # cheap thing that makes L2-GALLERY-SELF's declared negative
            # injection -- "shuffle the index id mapping" -- VISIBLE, because
            # the self-retrieval criterion cannot see it: the criterion scores
            # each gallery row against itself, so shuffling ids against rows
            # leaves its arithmetic bit-identical. Not a criterion, because
            # sorted row order is n11's convention and the gate's stated
            # criterion never mentions it; failing on it would be this module
            # inventing a rule.
            observed["ids_are_sorted"] = bool(ids == sorted(ids))
            observed["ids_sorted_note"] = ROW_ORDER_NOTE

        # ---- 6. no NaN, no Inf, no zero vectors ---------------------------
        if embeddings is not None:
            x = embeddings.astype(np.float64)
            nan_rows = np.nonzero(np.isnan(x).any(axis=1))[0]
            inf_rows = np.nonzero(np.isinf(x).any(axis=1))[0]
            # The same definition normalize_for_scoring uses, so a vector this
            # accepts is a vector the scorer can normalise.
            zero_rows = np.nonzero(np.linalg.norm(x, axis=1) == 0)[0]
            observed.update({
                "n_nan_rows": int(nan_rows.size),
                "n_inf_rows": int(inf_rows.size),
                "n_zero_norm_rows": int(zero_rows.size),
                "nan_examples": [ids[i] for i in nan_rows[:EXAMPLES]],
                "inf_examples": [ids[i] for i in inf_rows[:EXAMPLES]],
                "zero_norm_examples": [ids[i] for i in zero_rows[:EXAMPLES]],
            })
            for name, rows in (("NaN", nan_rows), ("Inf", inf_rows),
                               ("zero", zero_rows)):
                if rows.size:
                    failures.append(
                        f"{rows.size} {name} vector(s), e.g. "
                        f"{[ids[i] for i in rows[:EXAMPLES]]}")

            # ---- 7. self-retrieval, ties allowed --------------------------
            if nan_rows.size or inf_rows.size or zero_rows.size:
                observed["self_retrieval_not_run"] = (
                    "the index carries NaN, Inf or zero vectors; the scorer "
                    "refuses to normalise them")
            else:
                gallery = normalize_for_scoring(x)
                self_retrieval = _self_retrieval(
                    ids, gallery, seed=int(split_obs["split_seed"]),
                    sample_size=sample_size, block=block)
                observed["self_retrieval"] = self_retrieval
                # RECORDED, NOT A PASS CONDITION. See COLLAPSE_NOTE.
                observed["collapse_note"] = COLLAPSE_NOTE
                observed["embedding_health"] = embedding_health(gallery)
                if self_retrieval["n_failed"]:
                    failures.append(
                        f"{self_retrieval['n_failed']} of "
                        f"{self_retrieval['sample_size']} sampled vectors do "
                        f"not retrieve themselves, e.g. "
                        f"{[f['asset_id'] for f in self_retrieval['failed_examples']]}")
        else:
            observed["self_retrieval_not_run"] = "the index could not be read"

        rc = FAIL if failures else PASS
    except _Blocked as exc:
        rc = BLOCKED_EVIDENCE
        observed["blocked_reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        # A gate that dies without a record is indistinguishable from a gate
        # that never ran, and [L1-GATE-NORECORD] makes both "not passed" -- but
        # only one of them says why. The exception is written into the record
        # rather than swallowed, and rc 3 still blocks promotion. This is NOT a
        # PASS path and it is NOT a FAIL: an internal error means nobody knows
        # whether the index was good.
        rc = BLOCKED_EVIDENCE
        observed["blocked_reason"] = (
            f"internal error in the gate itself: {type(exc).__name__}: {exc}")

    observed["failures"] = failures
    record = {
        "gate_id": GATE_ID,
        "gate_class": GATE_CLASS,
        "scope": scope,
        "record_kind": "gate",
        "criterion": criterion,
        "inputs": inputs,
        "observed": observed,
        "verdict": VERDICT[rc],
        "rc": rc,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_revision": runlog.code_revision(),
        "code_dirty": runlog.code_dirty(),
        "runtime_source_sha256": runlog.runtime_source_sha256(),
        "runtime_source_status": runlog.runtime_source_status(),
        # This writer only emits final verdicts -- it has no provisional path --
        # so this is true in every record it produces. n12 checks it anyway,
        # because the field is part of the record contract and a record from any
        # other writer has to satisfy it too.
        "is_terminal": True,
        # Promoted to top level as well as living inside `observed`: n12 reads
        # these four and must not have to guess where a verdict put them.
        "index_uri": observed.get("index_uri"),
        "index_sha256": observed.get("index_sha256_on_disk"),
        "staging_record_sha256": observed.get("staging_record_sha256"),
        "stage1_checkpoint_sha256":
            observed.get("stage1_checkpoint_sha256_in_staging"),
        "gallery_encoder_sha256": observed.get("gallery_encoder_sha256"),
        "expected_uid_set_sha256": observed.get("expected_uid_set_sha256"),
        "sample_uid_sequence_sha256":
            (self_retrieval or {}).get("sample_uid_sequence_sha256"),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    gi._write(record_path, record, dump=yaml.safe_dump)
    _append_history(record_path, record)
    print(f"{GATE_ID}: {record['verdict']} (rc {rc}) -> {record_path}",
          flush=True)
    for line in failures:
        print(f"  FAIL: {line}", flush=True)
    if "blocked_reason" in observed:
        print(f"  BLOCKED: {observed['blocked_reason']}", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staging", default=None,
                    help="staging index record; defaults to n11's output")
    ap.add_argument("--splits", default=None,
                    help="split artifact the admitted set is derived from")
    ap.add_argument("--record", default=None,
                    help="where to write the gate record; defaults to the "
                         "path declared in validation_plan.yaml")
    ap.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = ap.parse_args()
    with runlog.run_progress(GATE_ID) as progress:
        progress.rc = run(staging_path=args.staging, splits_path=args.splits,
                          record_path=args.record, sample_size=args.sample_size)
    # `progress.rc`, not a separate `rc`: the run_progress row is written from
    # this attribute as the block closes, so returning it is the only spelling
    # where the durable record and the exit code cannot disagree. A gate whose
    # FAIL is logged as SUCCESS is worse than no gate.
    return progress.rc


if __name__ == "__main__":
    raise SystemExit(main())
