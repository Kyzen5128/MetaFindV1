"""Judge object corpus accounting and resolved Stage 1 input protocols.

# IMPLEMENTS-NODE: G3_object_corpus

Writes gate_records (current YAML plus history); the CLI also writes run_progress.

This is the existing G-INVALID gate, not a corpus producer. It reads a frozen
manifest, splits, protocols and real quarantine records; it never manufactures
an exception for a missing asset. The 2% limit is the project validation rule,
not a paper value. Object scope excludes ProcTHOR failures. DL-026's historical
n04 runtime-guard refusals are reported separately and do not quarantine assets.

The user-approved 2026-09-08 accounting keeps manual_review_rejected as E,
separate from admitted A and true quarantine Q = failure UIDs - A - E.
A, Q and E must be pairwise disjoint and their union must equal the unchanged
manifest M. Only Q / M has the existing 2% limit; manual and total exclusion
rates are reported independently. Unapproved exclusion groups remain blocked.
The complete E set is bound to the original approved decision bytes, not to
a caller-provided group name or self-reported provenance hash.

PASS certifies these input checks, not cache contents, trainer execution or
paper-level experimental reproduction. INVALIDATED (4) is declared but has no
specified trigger. This module never emits it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import get_args

import yaml

from metafind import paths, runlog
from metafind.data.splits import corpus_uids, ledger_excluded_uids
from metafind.gates.g4_gallery_freeze import _append_history
from metafind.models.dual_tower import TOWER_SHARING
from metafind.models.fusion import FusionKind
from metafind.models.stage1_config import (
    ABLATION_P_MASK, ENCODING_FIELDS, PAPER_P_MASK, PER_VIEW_AGGREGATIONS,
    PRECOMPUTABLE_AGGREGATIONS, REQUIRED_HYPERPARAMETERS, TRAINING_FIELDS,
    canonical_hyperparameter_hash,
)
from metafind.train.gallery_index import _write

GATE_ID = "G3_object_corpus"
GATE_CLASS = "G-INVALID"
PASS, FAIL, BLOCKED_EVIDENCE, INVALIDATED = 0, 2, 3, 4
RC_CONTRACT = {"PASS": PASS, "FAIL": FAIL, "BLOCKED_EVIDENCE": BLOCKED_EVIDENCE,
               "INVALIDATED": INVALIDATED}
SPEC_PATH = paths.REPO / "docs/graph/validation_plan.yaml"
APPROVED_EXCLUSIONS_PATH = paths.REPO / "workflow/annotation_exclusions_20260828.json"
APPROVED_EXCLUSIONS_SHA256 = "39eff095ea6283601d57db7618d39ba2355db694e16b711ca82d5ddc59ef3058"
APPROVAL_METADATA = ("decided_at", "decided_by", "decision", "git_commit")
HISTORICAL_GUARD_PREFIX = "implementation changed while the run was in progress"
OBJECT_STAGES = {"n03_sample_pointclouds", "n04_render_views", "n05_annotate",
                 "n06_encode_text_image"}
SCENE_STAGES = {"n07_scene_graphs", "n07b_procthor_asset_modalities", "n08_semantic_edges"}
QUARANTINE_FIELDS = ("uid", "stage", "failure_class", "exception_type",
                     "exception_msg", "code_revision", "timestamp")


class _Blocked(Exception):
    pass


class _Violation(Exception):
    pass


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _set_evidence(uids: set[str]) -> dict:
    return {"count": len(uids), "sha256": _sha("".join(
        uid + "\n" for uid in sorted(uids)).encode()), "examples": sorted(uids)[:10]}


def _object_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _Violation(f"duplicate JSON object key {key!r}")
        out[key] = value
    return out


def _json(blob: bytes, label: str):
    try:
        return json.loads(blob, object_pairs_hook=_object_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              _Violation(f"non-finite JSON value {value}")))
    except (ValueError, UnicodeError) as exc:
        raise _Blocked(f"cannot parse {label}: {exc}") from exc


def _read(path: Path, inputs: dict, label: str):
    try:
        blob = path.read_bytes()
    except OSError as exc:
        inputs[label] = {"path": str(path), "error": str(exc)}
        raise _Blocked(f"cannot read {label} at {path}: {exc}") from exc
    inputs[label] = {"path": str(path), "sha256": _sha(blob), "bytes": len(blob)}
    return _json(blob, str(path))


def _output_conflict(destinations: list[Path], sources: list[Path]) -> str | None:
    for destination in destinations:
        for source in sources:
            if (destination.resolve() == source.resolve()
                    or (destination.exists() and source.exists() and destination.samefile(source))):
                return f"output {destination} aliases input {source}"
    return None


def _verify_inputs(inputs: dict) -> None:
    """Do not certify a collection of snapshots that changed while being read."""
    for evidence in inputs.values():
        entries = evidence if isinstance(evidence, list) else [evidence]
        for entry in entries:
            path = Path(entry["path"])
            if "sha256" in entry and _sha(path.read_bytes()) != entry["sha256"]:
                raise _Blocked(f"input changed during gate execution: {path}")
            if entry.get("present") is False and path.exists():
                raise _Blocked(f"previously absent input appeared during gate execution: {path}")
            if "files" in entry and sorted(str(p) for p in path.glob("quarantine_*.jsonl")) != entry["files"]:
                raise _Blocked(f"quarantine log inventory changed during gate execution: {path}")


def _dict(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise _Violation(f"{label} must be a JSON object")
    return value


def _uids(value, label: str) -> list[str]:
    if not isinstance(value, list):
        raise _Violation(f"{label} must be a UID list")
    if any(not isinstance(uid, str) or not uid.strip() or uid != uid.strip()
           or "\n" in uid or "\r" in uid or "\0" in uid for uid in value):
        raise _Violation(f"{label} contains an invalid UID")
    if len(value) != len(set(value)):
        raise _Violation(f"{label} contains duplicate UIDs")
    return value


def _splits(value: dict, observed: dict) -> tuple[dict[str, list[str]], set[str]]:
    if "object" not in value:
        raise _Blocked("splits missing object pools")
    obj = _dict(value.get("object"), "splits.object")
    if missing := {"train", "test"} - obj.keys():
        raise _Blocked(f"splits.object missing {sorted(missing)}")
    pools = {name: _uids(uids, f"splits.object.{name}") for name, uids in obj.items()}
    primary = ["train", "test"] + (["val"] if "val" in pools else [])
    leaks = {}
    for i, left in enumerate(primary):
        for right in primary[i + 1:]:
            overlap = set(pools[left]) & set(pools[right])
            if overlap:
                leaks[f"{left}/{right}"] = _set_evidence(overlap)
    observed["leakage"] = leaks
    observed["leakage_count"] = sum(item["count"] for item in leaks.values())
    if leaks:
        raise _Violation(f"object split leakage: {list(leaks)}")
    admitted = set(corpus_uids(obj))
    if not admitted:
        raise _Violation("admitted object corpus is empty")
    # Alias overlaps are intentional; their membership must match their source.
    expected = {"train_val": set(pools["train"]) | set(pools.get("val", [])),
                "holdout": set(pools["test"]) | set(pools.get("val", []))}
    if "val" in pools:
        expected.update(dev_train=set(pools["train"]), dev_val=set(pools["val"]))
    elif "dev_train" in pools or "dev_val" in pools:
        if "dev_train" not in pools or "dev_val" not in pools:
            raise _Blocked("legacy development split must define both dev_train and dev_val")
        if set(pools["dev_train"]) & set(pools["dev_val"]):
            raise _Violation("legacy dev_train/dev_val leakage")
        if set(pools["dev_train"]) | set(pools["dev_val"]) != set(pools["train"]):
            raise _Violation("legacy dev_train + dev_val != train")
    for name, expected_uids in expected.items():
        if name in pools and set(pools[name]) != expected_uids:
            raise _Violation(f"split alias {name} does not match its primary pools")
    for name, uids in pools.items():
        if not set(uids) <= admitted:
            raise _Violation(f"split pool {name} contains UIDs outside admitted")
    if "full" in pools and set(pools["full"]) != admitted:
        raise _Violation("split full does not equal admitted")
    pools["full"] = corpus_uids(obj)
    observed["split_pools"] = {name: _set_evidence(set(uids)) for name, uids in pools.items()}
    observed["admitted"] = _set_evidence(admitted)
    return pools, admitted


def _approved_manual(inputs: dict, observed: dict) -> dict:
    """DL-106 carries the original decision, including all its manual UIDs.

    The trusted path/hash are code-owned; no corpus file or CLI flag can appoint
    a new approval source. Historical n05 failures in that source are not E.
    """
    try:
        blob = APPROVED_EXCLUSIONS_PATH.read_bytes()
    except OSError as exc:
        inputs["approved_manual_decision"] = {"path": str(APPROVED_EXCLUSIONS_PATH), "error": str(exc)}
        raise _Blocked(f"cannot read approved manual decision: {exc}") from exc
    digest = _sha(blob)
    inputs["approved_manual_decision"] = {"path": str(APPROVED_EXCLUSIONS_PATH),
        "sha256": digest, "bytes": len(blob)}
    if digest != APPROVED_EXCLUSIONS_SHA256:
        raise _Blocked("approved manual decision bytes differ from the pinned sha256")
    ledger = _dict(_json(blob, "approved manual decision"), "approved manual decision")
    metadata = {key: ledger.get(key) for key in APPROVAL_METADATA}
    if any(not isinstance(value, str) or not value.strip() for value in metadata.values()):
        raise _Blocked("approved manual decision lacks decision metadata")
    group = _dict(ledger.get("groups", {}).get("manual_review_rejected"),
                  "approved manual decision group")
    members = [entry.get("uid") if isinstance(entry, dict) else entry
               for entry in group.get("uids", [])]
    _uids(members, "approved manual decision UIDs")
    approved = ledger_excluded_uids({"groups": {"manual_review_rejected": group},
                                     "excluded_total": group.get("n")})
    observed["manual_approval"] = {"decision": "DL-106 carries the 2026-08-28 manual decision",
        "metadata": metadata, "approved": _set_evidence(approved), "approved_uids": sorted(approved)}
    return {"uids": approved, "sha256": digest, "metadata": metadata}


def _exclusions(path: Path, inputs: dict, observed: dict,
                approval: dict | None = None) -> tuple[set[str], set[str]]:
    if not path.exists():
        inputs["annotation_exclusions"] = {"path": str(path), "present": False}
        observed["annotation_exclusions"] = {"status": "UNKNOWN", "note": "ledger missing; E is not known to be empty"}
        raise _Blocked(f"annotation_exclusions ledger missing: {path}; explicit empty groups required when E is empty")
    ledger = _dict(_read(path, inputs, "annotation_exclusions"), "annotation_exclusions")
    if "groups" not in ledger:
        raise _Blocked("annotation_exclusions missing explicit groups")
    try:
        excluded = ledger_excluded_uids(ledger)
    except (TypeError, ValueError, AttributeError) as exc:
        raise _Violation(f"invalid exclusion ledger: {exc}") from exc
    groups = _dict(ledger["groups"], "annotation_exclusions.groups")
    manual, unapproved, historical = set(), set(), set()
    unapproved_groups = []
    for name, group in groups.items():
        _dict(group, f"exclusion group {name}")
        members = [entry.get("uid") if isinstance(entry, dict) else entry
                   for entry in group.get("uids", [])]
        _uids(members, f"annotation_exclusions.groups.{name}.uids")
        if name == "manual_review_rejected":
            manual.update(members)
        elif name == "n05_quarantine":
            # Historical ledger membership is not an exception record and does
            # not override recovery. In particular, do not carry the old 311
            # failures into the rebuilt corpus without actual failure evidence.
            historical.update(members)
        elif members:
            unapproved.update(members)
            unapproved_groups.append(name)
    observed["annotation_exclusions"] = {**_set_evidence(excluded),
        "manual_count": len(manual), "manual_uids": sorted(manual),
        "unapproved_count": len(unapproved), "unapproved_groups": sorted(unapproved_groups),
        "unapproved_uids": sorted(unapproved), "historical_n05_ledger": _set_evidence(historical),
        "decision": ledger.get("decision"),
        "formal_accounting": "A union Q union E = M; Q = true failure UIDs - A - E",
        "manual_group": "manual_review_rejected", "accounting_decision_date": "2026-09-08"}
    if "source_ledger" in ledger:
        source = _dict(ledger["source_ledger"], "annotation_exclusions.source_ledger")
        if any(not isinstance(source.get(key), str) or not source[key].strip()
               for key in ("path", "sha256", *APPROVAL_METADATA)):
            raise _Blocked("annotation_exclusions.source_ledger lacks decision provenance")
        if ledger.get("accounting_decision") != "DL-106":
            raise _Violation("annotation_exclusions source ledger does not declare DL-106")
        if approval is not None and (source["sha256"] != approval["sha256"]
                or any(source[key] != approval["metadata"][key] for key in APPROVAL_METADATA)):
            raise _Violation("annotation_exclusions source_ledger disagrees with the pinned decision")
        # The producer's original absolute path is provenance, not permission to
        # read a new authority file. The verified archived bytes establish it.
        observed["annotation_exclusions"]["source_ledger"] = source
    return manual, unapproved


def _quarantine(log_dir: Path, admitted: set[str], manual: set[str],
                inputs: dict, observed: dict) -> set[str]:
    if not log_dir.is_dir():
        raise _Blocked(f"quarantine log directory missing: {log_dir}")
    records = []
    counters = Counter()
    ignored_stages = Counter()
    ignored_phases = Counter()
    systemic_rows = []
    input_logs = []
    inputs["quarantine_logs"] = input_logs
    log_paths = sorted(log_dir.glob("quarantine_*.jsonl"))
    inputs["quarantine_log_directory"] = {"path": str(log_dir), "files": [str(p) for p in log_paths]}
    for path in log_paths:
        # runlog names the file for its stage_name argument, while an individual
        # record may override `stage` with a phase (n07b explicitly uses render).
        # Archived .v5_intel etc. logs retain that same owner before the first dot.
        owner = path.name.removeprefix("quarantine_").split(".", 1)[0]
        if owner not in OBJECT_STAGES | SCENE_STAGES:
            raise _Blocked(f"quarantine log has unknown producer owner: {path}")
        try:
            blob = path.read_bytes()
        except OSError as exc:
            raise _Blocked(f"cannot read quarantine log {path}: {exc}") from exc
        input_logs.append({"path": str(path), "sha256": _sha(blob), "bytes": len(blob)})
        for lineno, line in enumerate(blob.splitlines(), 1):
            if not line.strip():
                continue
            row = _dict(_json(line, f"{path}:{lineno}"), f"{path}:{lineno}")
            stage = row.get("stage")
            if not isinstance(stage, str) or not stage:
                raise _Blocked(f"{path}:{lineno} lacks quarantine stage")
            if owner in SCENE_STAGES:
                permitted = {owner}
                if owner == "n07b_procthor_asset_modalities":
                    permitted.add("render")  # procthor_modalities.main's explicit phase
                if stage not in permitted:
                    raise _Violation(f"{path}:{lineno} producer owner {owner} disagrees with row stage {stage!r}")
                ignored_stages[owner] += 1
                if stage != owner:
                    ignored_phases[f"{owner}/{stage}"] += 1
                continue
            if stage not in OBJECT_STAGES:
                raise _Blocked(f"{path}:{lineno} has unknown quarantine stage {stage!r}")
            if stage != owner:
                raise _Violation(f"{path}:{lineno} producer owner {owner} disagrees with row stage {stage!r}")
            counters["object_rows"] += 1
            if (stage == "n04_render_views"
                    and isinstance(row.get("exception_msg"), str)
                    and row["exception_msg"].startswith(HISTORICAL_GUARD_PREFIX)):
                counters["historical_guard_rows_excluded"] += 1
                continue
            uid = row.get("uid")
            if (owner == "n04_render_views" and row.get("failure_class") == "RESOURCE"
                    and ((uid == "__run" and row.get("exception_type") == "SystemicFailure")
                         or (isinstance(uid, str) and uid.startswith("__batch_")
                             and uid.removeprefix("__batch_").isdigit()
                             and row.get("exception_type") == "BrokenProcessPool"))):
                # renders.main writes these run/batch sentinels explicitly.
                # They diagnose a run, not a mesh; never add them to the UID set.
                counters["systemic_rows_not_asset_failures"] += 1
                systemic_rows.append({"source": f"{path}:{lineno}", "uid": uid,
                                      "exception_type": row["exception_type"]})
                continue
            if uid is None and row.get("phase") == "initialization":
                counters["initialization_rows_not_asset_failures"] += 1
                continue
            if not isinstance(uid, str) or not uid.strip():
                raise _Blocked(f"{path}:{lineno} cannot identify failed asset UID")
            records.append((uid, row, f"{path}:{lineno}"))
    all_failed = {uid for uid, _, _ in records}
    quarantined = all_failed - admitted - manual
    observed["quarantine"] = {**dict(counters), "ignored_non_object_stages": dict(ignored_stages),
        "ignored_non_object_phases": dict(ignored_phases), "systemic_rows": systemic_rows,
        "failure_uid_count_before_recovery": len(all_failed),
        "duplicate_failure_rows": len(records) - len(all_failed),
        "recovered": _set_evidence(all_failed & admitted),
        "manual_failure_overlap": _set_evidence(all_failed & manual),
        "quarantined": _set_evidence(quarantined), "quarantined_uids": sorted(quarantined),
        "historical_guard_prefix": HISTORICAL_GUARD_PREFIX,
        "historical_guard_decision": "workflow/DECISION_LEDGER.md DL-026"}
    for uid, row, label in records:
        if uid not in quarantined:
            continue
        missing = [field for field in QUARANTINE_FIELDS if field != "timestamp"
                   if not isinstance(row.get(field), str) or not row[field].strip()]
        timestamp = row.get("timestamp")
        valid_timestamp = type(timestamp) in (int, float) and math.isfinite(timestamp)
        if isinstance(timestamp, str):
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                valid_timestamp = True
            except ValueError:
                pass
        if not valid_timestamp:
            missing.append("timestamp")
        if missing:
            raise _Blocked(f"real quarantine evidence {label} ({uid}) missing/invalid {missing}")
    return quarantined


def _choices(protocol: dict, fields: tuple, label: str) -> None:
    if protocol.get("status") != "resolved":
        raise _Blocked(f"{label}.status is not resolved")
    missing = [name for name in fields if name not in protocol or protocol[name] is None
               or (isinstance(protocol[name], str)
                   and protocol[name].strip().lower() in {"", "unknown", "unresolved", "tbd"})]
    if missing:
        raise _Blocked(f"{label} missing explicit choices: {missing}")


def _protocols(encoding: dict, training: dict, hp: dict, variant: str | None,
               observed: dict) -> None:
    _choices(encoding, ENCODING_FIELDS, "stage1_encoding_protocol")
    _choices(training, TRAINING_FIELDS, "stage1_protocol")
    basis = encoding.get("paper_clip_train_scope_basis")
    if not isinstance(basis, str) or not basis.strip():
        raise _Blocked("paper_clip_train_scope_basis is missing or empty")
    if "paper_clip_train_scope_confidence" not in encoding:
        raise _Blocked("paper_clip_train_scope_confidence is missing")
    if encoding["paper_clip_train_scope_confidence"] not in {"low", "moderate", "high"}:
        raise _Violation("paper_clip_train_scope_confidence is not low/moderate/high")
    enums = [(encoding, "paper_clip_train_scope", ("frozen", "trainable")),
             (encoding, "actual_clip_train_scope", ("frozen", "trainable")),
             (encoding, "image_aggregation", PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS),
             (encoding, "missing_modality_representation", ("learned_token", "zero_pad")),
             (training, "fusion", get_args(FusionKind)),
             (training, "tower_sharing", TOWER_SHARING),
             (training, "similarity", ("cosine",))]
    for protocol, key, choices in enums:
        if protocol[key] not in choices:
            raise _Violation(f"unsupported {key}={protocol[key]!r}; implemented choices {choices}")
    if not isinstance(encoding["text_serialization"], str):
        raise _Violation("text_serialization must name a serialization")
    if type(training["allow_all_masked"]) is not bool:
        raise _Violation("allow_all_masked must be a boolean")
    if "values" not in hp or "sha256" not in hp:
        raise _Blocked("stage1_hyperparameters missing values/sha256")
    values = _dict(hp["values"], "stage1_hyperparameters.values")
    if missing := set(REQUIRED_HYPERPARAMETERS) - values.keys():
        raise _Blocked(f"stage1_hyperparameters.values missing {sorted(missing)}")
    if any(value is None or (isinstance(value, str) and value.strip().lower()
           in {"", "unknown", "unresolved", "tbd"}) for value in values.values()):
        raise _Blocked("stage1_hyperparameters.values contains an unspecified value")
    for name in ("optimizer", "scheduler"):
        if not isinstance(values[name], str):
            raise _Violation(f"hyperparameter {name} must name a choice")
    for name in ("batch_size", "epochs", "max_epochs", "warmup_epochs", "seed"):
        if type(values[name]) is not int:
            raise _Violation(f"hyperparameter {name} must be an integer")
    for name in ("learning_rate", "weight_decay", "init_temperature", "max_logit_scale",
                 "eps", "lr_start", "lr_end"):
        if (type(values[name]) not in (int, float) or not math.isfinite(values[name])):
            raise _Violation(f"hyperparameter {name} must be finite numeric")
    for name in ("decay_mask_tokens", "learnable_temperature"):
        if type(values[name]) is not bool:
            raise _Violation(f"hyperparameter {name} must be boolean")
    betas = values["betas"]
    if (not isinstance(betas, list) or len(betas) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in betas)):
        raise _Violation("hyperparameter betas must contain two finite numbers")
    digest = canonical_hyperparameter_hash(values)
    observed["hyperparameters"] = {"canonical_sha256": digest,
        "artifact_sha256_field": hp["sha256"],
        "training_hash": training["hyperparameter_config_hash"],
        "required_fields": list(REQUIRED_HYPERPARAMETERS), "variant": variant,
        "p_mask": values["p_mask"]}
    if digest != hp["sha256"] or digest != training["hyperparameter_config_hash"]:
        raise _Violation("canonical hyperparameter hash differs from artifact or stage1_protocol")
    p_mask = values["p_mask"]
    if isinstance(p_mask, bool) or not isinstance(p_mask, (float, int)) or not math.isfinite(p_mask):
        raise _Violation("p_mask must be a finite numeric probability")
    if variant is None and p_mask != PAPER_P_MASK:
        raise _Violation(f"main p_mask must be {PAPER_P_MASK}; got {p_mask}")
    if variant is not None and ABLATION_P_MASK.get(p_mask) != variant:
        raise _Violation(f"declared dropout variant {variant!r} does not match p_mask={p_mask}")
    observed["protocols_resolved"] = True


def _evaluation(protocols: dict, pools: dict, observed: dict) -> None:
    required = {"A_test_gallery": ("test", "test"), "B_full_gallery": ("test", "full")}
    if missing := required.keys() - protocols.keys():
        raise _Blocked(f"eval_protocols missing {sorted(missing)}")
    checked = {}
    for name, protocol in protocols.items():
        _dict(protocol, f"eval_protocols.{name}")
        if missing := {"query_split", "gallery_split", "gallery_size"} - protocol.keys():
            raise _Blocked(f"eval_protocols.{name} missing {sorted(missing)}")
        query, gallery = protocol["query_split"], protocol["gallery_split"]
        if not isinstance(query, str) or not isinstance(gallery, str):
            raise _Violation(f"eval_protocols.{name} scopes must name split pools")
        if query not in pools or gallery not in pools:
            raise _Violation(f"eval_protocols.{name} references unknown scope {query}/{gallery}")
        if name in required and (query, gallery) != required[name]:
            raise _Violation(f"eval_protocols.{name} differs from its declared A/B scopes")
        if not pools[query] or not pools[gallery]:
            raise _Violation(f"eval_protocols.{name} has an empty query/gallery scope")
        if (type(protocol["gallery_size"]) is not int
                or protocol["gallery_size"] != len(pools[gallery])):
            raise _Violation(f"eval_protocols.{name} gallery_size does not match {gallery}")
        if "query_size" in protocol and (type(protocol["query_size"]) is not int
                or protocol["query_size"] != len(pools[query])):
            raise _Violation(f"eval_protocols.{name} query_size does not match {query}")
        if not set(pools[query]) <= set(pools[gallery]):
            raise _Violation(f"eval_protocols.{name} gallery omits query GT UIDs")
        checked[name] = {"query_split": query, "gallery_split": gallery,
                         "query_size": len(pools[query]), "gallery_size": len(pools[gallery])}
    observed["evaluation_protocols"] = checked


def run(outputs_path: Path | None = None, manifest_path: Path | None = None,
        record_path: Path | None = None, *, quarantine_dir: Path | None = None,
        exclusions_path: Path | None = None, variant: str | None = None,
        spec_path: Path | None = None) -> int:
    """Write current + append-history records; return the declared gate RC.

    Missing evidence is BLOCKED (3), known contradictions are FAIL (2), and
    definite failures take precedence when both are observed. No inputs change.
    Explicit outputs/record paths make isolated CPU fixtures and dry audits safe.
    An output path aliasing an input is refused with RC 3 and no record written:
    there is no safe destination in that invocation, and missing record != PASS.
    """
    outputs = Path(outputs_path) if outputs_path is not None else paths.OUTPUTS
    manifest_path = Path(manifest_path) if manifest_path is not None else paths.LVIS_MANIFEST
    record_path = Path(record_path) if record_path is not None else outputs / "logs/gates/G3_object_corpus.yaml"
    quarantine_dir = Path(quarantine_dir) if quarantine_dir is not None else outputs / "logs"
    exclusions_path = Path(exclusions_path) if exclusions_path is not None else outputs / "annotation_exclusions.json"
    inputs, observed, failures, blocked = {}, {}, [], []
    observed["coverage_limit"] = ("Input accounting/protocol checks only. Does not validate cache tensors, "
        "actual trainer modules/optimizer execution, scene leakage, or paper experimental results. "
        "No split ratio, rounded corpus size, annotation quality or new promotion threshold is imposed.")

    def check(label, fn):
        try:
            return fn()
        except _Violation as exc:
            failures.append(f"{label}: {exc}")
        except _Blocked as exc:
            blocked.append(f"{label}: {exc}")
        except Exception as exc:  # A gate bug/unreadable evidence must never look like PASS.
            blocked.append(f"{label}: internal/unreadable evidence {type(exc).__name__}: {exc}")
        return None

    def read_spec():
        path = Path(spec_path) if spec_path is not None else SPEC_PATH
        blob = path.read_bytes()
        inputs["specification"] = {"path": str(path), "sha256": _sha(blob)}
        entries = yaml.safe_load(blob)["level_3_gates"]
        entry = next(g for g in entries if g["gate_id"] == GATE_ID)
        if entry["rc_contract"] != RC_CONTRACT or entry["gate_class"] != GATE_CLASS:
            raise _Blocked("specification RC contract or gate class differs from implementation")
        return str(entry["criterion"]).strip()

    criterion = check("specification", read_spec) or ""
    manifest = check("manifest", lambda: _dict(_read(manifest_path, inputs, "manifest"), "manifest"))
    if manifest is not None:
        check("manifest", lambda: _uids(list(manifest), "manifest"))
        # n01's persisted artifact is UID -> source-relative mesh path. Do not
        # bless an unknown/null value as a complete manifest just by counting keys.
        for uid, source in manifest.items():
            if not isinstance(source, str) or not source.strip():
                failures.append(f"manifest: {uid} has no valid source path string")
            elif source.strip().lower() in {"unknown", "unresolved", "tbd"}:
                blocked.append(f"manifest: {uid} has an unresolved source path")
        observed["manifest"] = _set_evidence(set(manifest))
        if not manifest:
            failures.append("manifest: manifest is empty")
    split_value = check("splits", lambda: _dict(_read(outputs / "splits.json", inputs, "splits"), "splits"))
    split_result = check("splits", lambda: _splits(split_value, observed)) if split_value is not None else None
    approval = check("manual approval", lambda: _approved_manual(inputs, observed))
    exclusions = check("exclusions", lambda: _exclusions(exclusions_path, inputs, observed, approval))
    if approval is not None and exclusions is not None:
        manual = exclusions[0]
        missing_approved, unapproved_manual = approval["uids"] - manual, manual - approval["uids"]
        observed["manual_approval"].update(matches=manual == approval["uids"],
            missing=_set_evidence(missing_approved), unexpected=_set_evidence(unapproved_manual))
        if missing_approved or unapproved_manual:
            failures.append("manual approval: E differs from the complete pinned approved UID set "
                            f"({len(missing_approved)} missing, {len(unapproved_manual)} unexpected)")
    if exclusions is not None and exclusions[1]:
        blocked.append("unapproved exclusion groups: cannot classify as manual_excluded or fabricate quarantine")
    if split_result is not None:
        pools, admitted = split_result
        manual = exclusions[0] if exclusions is not None else set()
        quarantined = check("quarantine", lambda: _quarantine(quarantine_dir, admitted, manual, inputs, observed))
        if manifest is not None and quarantined is not None and exclusions is not None:
            manual, unapproved = exclusions
            missing = set(manifest) - admitted - quarantined - manual
            unexpected = (admitted | quarantined | manual | unapproved) - set(manifest)
            unexplained = missing - unapproved
            rate = len(quarantined) / len(manifest) if manifest else None
            observed["accounting"] = {"admitted_count": len(admitted), "quarantined_count": len(quarantined),
                "manual_excluded_count": len(manual), "manual_excluded": _set_evidence(manual),
                "manifest_count": len(manifest), "quarantine_rate": rate, "quarantine_limit": 0.02,
                "manual_excluded_rate": len(manual) / len(manifest) if manifest else None,
                "total_excluded_rate": len(quarantined | manual) / len(manifest) if manifest else None,
                "set_conservation": admitted | quarantined | manual == set(manifest),
                "pairwise_disjoint": not (admitted & quarantined or admitted & manual or quarantined & manual),
                "missing": _set_evidence(missing), "unexplained_missing": _set_evidence(unexplained),
                "unexpected": _set_evidence(unexpected), "excluded_but_admitted": _set_evidence(manual & admitted)}
            if unexpected:
                failures.append(f"accounting: {len(unexpected)} unexpected UID(s) outside manifest")
            if unexplained:
                failures.append(f"accounting: {len(unexplained)} unexplained missing UID(s); missing is not a quarantine reason")
            if manual & admitted:
                failures.append(f"accounting: {len(manual & admitted)} manually excluded UID(s) are admitted")
            if rate is not None and rate > 0.02:
                failures.append(f"accounting: quarantine_rate={rate} exceeds existing 0.02 limit")
        evals = check("evaluation", lambda: _dict(_read(outputs / "eval_protocols.json", inputs, "eval_protocols"), "eval_protocols"))
        if evals is not None:
            check("evaluation", lambda: _evaluation(evals, pools, observed))
    documents = [check(name, lambda name=name: _dict(_read(outputs / f"{name}.json", inputs, name), name))
                 for name in ("stage1_encoding_protocol", "stage1_protocol", "stage1_hyperparameters")]
    if all(document is not None for document in documents):
        check("stage1 protocols", lambda: _protocols(*documents, variant, observed))
    check("input stability", lambda: _verify_inputs(inputs))
    observed.update(failures=failures, blocked_reasons=blocked, declared_variant=variant)
    rc = FAIL if failures else BLOCKED_EVIDENCE if blocked else PASS
    record = {"gate_id": GATE_ID, "gate_class": GATE_CLASS, "scope": str(manifest_path.resolve()),
        "record_kind": "gate", "criterion": criterion, "inputs": inputs, "observed": observed,
        "verdict": next(name for name, value in RC_CONTRACT.items() if value == rc), "rc": rc,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_revision": runlog.code_revision(),
        "code_dirty": runlog.code_dirty(), "runtime_source_sha256": runlog.runtime_source_sha256(),
        "runtime_source_status": runlog.runtime_source_status(),
        "gate_source_sha256": _sha(Path(__file__).read_bytes()), "is_terminal": True}
    source_paths = [manifest_path, exclusions_path, APPROVED_EXCLUSIONS_PATH,
                    Path(spec_path) if spec_path else SPEC_PATH]
    source_paths += [outputs / name for name in ("splits.json", "eval_protocols.json",
                     "stage1_encoding_protocol.json", "stage1_protocol.json", "stage1_hyperparameters.json")]
    source_paths += list(quarantine_dir.glob("quarantine_*.jsonl"))
    history = record_path.with_name(record_path.stem + ".history.yaml")
    destinations = [record_path, history, record_path.with_suffix(record_path.suffix + ".part"),
                    history.with_suffix(history.suffix + ".part")]
    if conflict := _output_conflict(destinations, source_paths):
        print(f"{GATE_ID}: BLOCKED_EVIDENCE (rc 3); no safe record written: {conflict}", flush=True)
        return BLOCKED_EVIDENCE
    record_path.parent.mkdir(parents=True, exist_ok=True)
    _write(record_path, record, dump=yaml.safe_dump)
    _append_history(record_path, record)
    print(f"{GATE_ID}: {record['verdict']} (rc {rc}) -> {record_path}", flush=True)
    for message in failures:
        print(f"  FAIL: {message}", flush=True)
    for message in blocked:
        print(f"  BLOCKED: {message}", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, default=paths.OUTPUTS)
    parser.add_argument("--manifest", type=Path, default=paths.LVIS_MANIFEST)
    parser.add_argument("--record", type=Path,
                        help="gate record; if explicit, run_progress.jsonl is written beside it too")
    parser.add_argument("--quarantine-dir", type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--variant", choices=sorted(ABLATION_P_MASK.values()))
    args = parser.parse_args()
    # --record is the output-isolation option even when --outputs is a read-only
    # corpus. run() itself writes only the gate record and its history.
    paths.LOGS = args.record.parent if args.record is not None else args.outputs / "logs"
    sources = [args.manifest, args.exclusions or args.outputs / "annotation_exclusions.json",
               APPROVED_EXCLUSIONS_PATH, SPEC_PATH]
    sources += [args.outputs / name for name in ("splits.json", "eval_protocols.json",
                "stage1_encoding_protocol.json", "stage1_protocol.json", "stage1_hyperparameters.json")]
    sources += list((args.quarantine_dir or args.outputs / "logs").glob("quarantine_*.jsonl"))
    if conflict := _output_conflict([paths.LOGS / "run_progress.jsonl"], sources):
        print(f"{GATE_ID}: BLOCKED_EVIDENCE (rc 3); no safe progress output: {conflict}", flush=True)
        return BLOCKED_EVIDENCE
    # Both writers must own distinct files, including their atomic-write
    # intermediates. Otherwise the context's final JSON append corrupts the
    # YAML gate record even though the gate and process both report PASS.
    record = args.record if args.record is not None else args.outputs / "logs/gates/G3_object_corpus.yaml"
    history = record.with_name(record.stem + ".history.yaml")
    gate_outputs = [record, history, record.with_suffix(record.suffix + ".part"),
                    history.with_suffix(history.suffix + ".part")]
    if conflict := _output_conflict([paths.LOGS / "run_progress.jsonl"], gate_outputs):
        print(f"{GATE_ID}: BLOCKED_EVIDENCE (rc 3); gate/progress outputs collide: {conflict}", flush=True)
        return BLOCKED_EVIDENCE
    with runlog.run_progress(GATE_ID) as progress:
        progress.rc = run(args.outputs, args.manifest, args.record,
                          quarantine_dir=args.quarantine_dir,
                          exclusions_path=args.exclusions, variant=args.variant)
    return progress.rc


if __name__ == "__main__":
    raise SystemExit(main())
