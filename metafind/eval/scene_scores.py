"""Validate externally supplied scene ratings and report their denominators.

This is a partial n17/n20 component, not a judge or a formal Table 2 runner.
It never calls a model, chooses prompts/cameras/scenes, or certifies rating
quality. Method labels and judge execution are external declarations. The
verified claim is that supplied scores refer to these exact scene artifacts.

CLI: ``python -m metafind.eval.scene_scores import --manifest FILE
--protocol FILE --records FILE --out FILE``; then ``aggregate --scores FILE
--out FILE``. All outputs are new files; inputs are never overwritten.

The frozen protocol requires schema/status, the exact DIMENSIONS/score_range,
model {id, revision}, prompt, generation, view_policy and provenance. The frozen
manifest requires schema/status, scene_ids, method_ids, provenance and outcomes:
one entry per (method_id, scene_id). Complete outcomes carry composition,
placement and result path+sha256 references, plus explicit judged_camera_ids.
Incomplete outcomes carry stage/reason. The submission requires schema,
protocol_sha256, manifest_sha256 and records. Each scored/failed record names
method_id, scene_id and inputs (the three artifact SHA256s and selected camera
SHA256s). Scored records carry four scores and a raw_response path+sha256;
failed records carry a reason. Omitted records remain missing, never zero.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re

from PIL import Image

from metafind.scene import placement

DIMENSIONS = ("overall_aesthetic_and_atmosphere", "color_scheme_and_material_choices",
              "scene_coherence", "realism_and_3d_geometric_consistency")
PROTOCOL_SCHEMA = "metafind.scene_judge_protocol.v1"
MANIFEST_SCHEMA = "metafind.scene_evaluation_manifest.v1"
SUBMISSION_SCHEMA = "metafind.scene_score_submission.v1"
SCHEMA = "metafind.scene_scores.v1"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(raw):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=invalid)
    # This also refuses numeric overflow, e.g. a JSON number written as 1e999.
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _fields(value, names, label):
    placement._fields(value, names, label)


def _ids(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be an explicit nonempty list")
    for item in value:
        placement._identity(item, label)
    if len(set(value)) != len(value):
        raise ValueError(f"duplicate {label}")
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected lowercase SHA256")
    return value


def _snapshot(path, checked):
    path = Path(path).resolve()
    raw = path.read_bytes()
    rec = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    checked.append(rec)
    return {**rec, "content": _decode(raw)}


def _read_ref(record, base, checked):
    _fields(record, "path sha256", "artifact reference")
    placement._identity(record["path"], "artifact path")
    _digest(record["sha256"])
    path = Path(record["path"])
    path = (path if path.is_absolute() else base / path).resolve()
    normalized = {"path": str(path), "sha256": record["sha256"]}
    raw = placement._artifact(normalized, base)
    checked.append(normalized)
    return raw, normalized


def _recheck(checked):
    seen = {}
    for record in checked:
        path, expected = record["path"], record["sha256"]
        if path in seen:
            if seen[path] != expected:
                raise ValueError(f"conflicting input identities: {path}")
            continue
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"input changed during score validation: {path}")
        seen[path] = expected


def _protocol(value):
    _fields(value, "schema status dimensions score_range model prompt generation view_policy provenance", "judge protocol")
    if value["schema"] != PROTOCOL_SCHEMA or value["status"] != "frozen":
        raise ValueError("requires a frozen explicit judge protocol")
    if (value["dimensions"] != list(DIMENSIONS) or value["score_range"] != [1, 5]
            or any(type(x) is not int for x in value["score_range"])):
        raise ValueError("requires the four MetaFind dimensions on the 1–5 scale")
    _fields(value["model"], "id revision", "judge model")
    for name in ("id", "revision"):
        placement._identity(value["model"][name], "judge " + name)
    if value["model"]["id"].rstrip("/").split("/")[-1] != "gemma-4-12B-it":
        raise ValueError("this component follows the approved Gemma judge decision (D-8)")
    for name in ("prompt", "view_policy"):
        placement._identity(value[name], name)
    for name in ("generation", "provenance"):
        placement._provenance(value[name])


def _completed(outcome, base, checked):
    inputs, parsed, refs = {}, {}, {}
    for key in ("composition", "placement", "result"):
        raw, ref = _read_ref(outcome[key], base, checked)
        parsed[key], refs[key] = _decode(raw), ref
        inputs[key + "_sha256"] = ref["sha256"]
    comp, manifest, result = (parsed[k] for k in ("composition", "placement", "result"))
    if (comp.get("schema") != "metafind.scene_composition.v1" or comp.get("status") != "complete"
            or comp.get("room_id") != outcome["scene_id"]):
        raise ValueError("completed composition must match the declared scene_id/room_id")
    # Reuse the actual placement reader, including slots, assets and snapshots.
    validated = placement.load_placement(Path(refs["placement"]["path"]),
                                         expected_sha256=refs["placement"]["sha256"])
    if validated != manifest or manifest["sources"]["composition"]["sha256"] != inputs["composition_sha256"]:
        raise ValueError("placement is not bound to this composition")
    if manifest["sources"]["composition"]["content"] != comp:
        raise ValueError("placement composition snapshot differs")
    if manifest["render"] is None:
        raise ValueError("scene ratings require explicit rendered views")
    if (result.get("schema") != placement.SCHEMA or result.get("status") != "complete"
            or result.get("manifest_sha256") != inputs["placement_sha256"]):
        raise ValueError("Blender result is incomplete or belongs to another placement")
    if (result.get("room") != manifest["room"] or result.get("render_config") != manifest["render"]
            or result.get("implementation") != manifest["implementation"] or result.get("device") != "CPU"):
        raise ValueError("Blender result differs from the frozen placement/render configuration")
    instances = result.get("instances")
    if not isinstance(instances, list) or len(instances) != len(manifest["instances"]):
        raise ValueError("Blender result has missing/extra instances")
    for actual, expected in zip(instances, manifest["instances"]):
        if (actual.get("asset_id") != expected["asset_id"] or actual.get("slot") != expected["slot"]
                or actual.get("slot_id") != expected["slot"]["new_object_id"]):
            raise ValueError("Blender instance differs from the composed slot/asset")
    # Track assets for the final drift check as well as the reader's first check.
    for asset in manifest["assets"].values():
        for key in ("mesh", "annotation"):
            _read_ref({k: asset[key][k] for k in ("path", "sha256")},
                      Path(refs["placement"]["path"]).parent, checked)
    result_base = Path(refs["result"]["path"]).parent
    _read_ref(result["blend"], result_base, checked)
    renders = result.get("renders")
    cameras = [c["id"] for c in manifest["render"]["cameras"]]
    if not isinstance(renders, list) or [r.get("camera_id") for r in renders] != cameras:
        raise ValueError("rendered camera identities/order differ from the render configuration")
    rendered = {}
    for render in renders:
        _fields(render, "camera_id path sha256", "rendered view")
        raw, _ = _read_ref({k: render[k] for k in ("path", "sha256")}, result_base, checked)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or list(image.size) != manifest["render"]["resolution"]:
                raise ValueError("render must be a PNG with the declared resolution")
            image.verify()
        rendered[render["camera_id"]] = render["sha256"]
    chosen = _ids(outcome["judged_camera_ids"], "judged_camera_ids")
    if not set(chosen) <= set(rendered):
        raise ValueError("judged camera was not rendered")
    inputs["renders"] = {camera: rendered[camera] for camera in chosen}
    inputs["judged_camera_ids"] = chosen
    return inputs


def _build(manifest_path, protocol_path, records_path):
    checked = []
    sources = {name: _snapshot(path, checked) for name, path in
               (("manifest", manifest_path), ("protocol", protocol_path), ("submission", records_path))}
    manifest, protocol, submission = (sources[k]["content"] for k in ("manifest", "protocol", "submission"))
    _protocol(protocol)
    _fields(manifest, "schema status scene_ids method_ids outcomes provenance", "evaluation manifest")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["status"] != "frozen":
        raise ValueError("requires a frozen evaluation manifest")
    scenes = _ids(manifest["scene_ids"], "scene_ids")
    methods = _ids(manifest["method_ids"], "method_ids")
    placement._provenance(manifest["provenance"])
    expected = {(method, scene) for method in methods for scene in scenes}
    cases = {}
    if not isinstance(manifest["outcomes"], list):
        raise ValueError("outcomes must explicitly cover every declared method/scene")
    for outcome in manifest["outcomes"]:
        if not isinstance(outcome, dict):
            raise ValueError("outcome must be an object")
        for name in ("method_id", "scene_id"):
            placement._identity(outcome.get(name), name)
        key = (outcome.get("method_id"), outcome.get("scene_id"))
        if key not in expected or key in cases:
            raise ValueError("unknown/duplicate method-scene outcome")
        status = outcome.get("status")
        common = "method_id scene_id status "
        if status == "complete":
            _fields(outcome, common + "composition placement result judged_camera_ids", "complete outcome")
            inputs = _completed(outcome, Path(sources["manifest"]["path"]).parent, checked)
            cases[key] = {"method_id": key[0], "scene_id": key[1], "composition_status": status,
                          "score_status": "missing", "inputs": inputs}
        elif status == "incomplete":
            _fields(outcome, common + "stage reason", "incomplete outcome")
            for name in ("stage", "reason"):
                placement._identity(outcome[name], name)
            cases[key] = {"method_id": key[0], "scene_id": key[1], "composition_status": status,
                          "score_status": "not_eligible", "stage": outcome["stage"], "reason": outcome["reason"]}
        else:
            raise ValueError("outcome must explicitly be complete or incomplete")
    if set(cases) != expected:
        raise ValueError("outcomes do not cover the declared scene denominator")
    _fields(submission, "schema protocol_sha256 manifest_sha256 records", "score submission")
    if (submission["schema"] != SUBMISSION_SCHEMA
            or submission["protocol_sha256"] != sources["protocol"]["sha256"]
            or submission["manifest_sha256"] != sources["manifest"]["sha256"]):
        raise ValueError("score submission is not bound to this protocol and manifest")
    if not isinstance(submission["records"], list):
        raise ValueError("score records must be a list")
    seen = set()
    for record in submission["records"]:
        if not isinstance(record, dict):
            raise ValueError("score record must be an object")
        for name in ("method_id", "scene_id"):
            placement._identity(record.get(name), name)
        key = (record.get("method_id"), record.get("scene_id"))
        if key not in cases or key in seen:
            raise ValueError("unknown/duplicate score record")
        seen.add(key)
        case = cases[key]
        if case["composition_status"] != "complete":
            raise ValueError("incomplete scenes cannot receive scores")
        if record.get("inputs") != case["inputs"]:
            raise ValueError("score inputs differ from this scene's exact artifacts/views")
        common = "method_id scene_id status inputs "
        if record.get("status") == "scored":
            _fields(record, common + "scores raw_response", "scored record")
            _fields(record["scores"], " ".join(DIMENSIONS), "scores")
            for dim, value in record["scores"].items():
                placement._number(value, dim)
                if not 1 <= value <= 5:
                    raise ValueError(f"{dim} score must be in [1,5]")
            raw, response = _read_ref(record["raw_response"], Path(sources["submission"]["path"]).parent, checked)
            if not raw.strip():
                raise ValueError("a scored record requires a nonempty raw response")
            case.update(score_status="scored", scores=record["scores"], raw_response=response)
        elif record.get("status") == "failed":
            _fields(record, common + "reason", "failed score record")
            placement._identity(record["reason"], "judge failure reason")
            case.update(score_status="failed", reason=record["reason"])
        else:
            raise ValueError("score status must be scored or failed; missing records remain missing")
    _recheck(checked)
    return {"schema": SCHEMA, "status": "validated", "classification": "OBSERVED DATA",
            "scope": "External rating import; no judge execution or rating-quality attestation; partial n17/n20 only",
            "sources": sources, "scene_ids": scenes, "method_ids": methods,
            "cases": [cases[(m, s)] for m in methods for s in scenes],
            "limitations": ["Method labels and judge identity/execution are external declarations.",
                            "Raw response bytes are bound, not interpreted or certified as the source of scores.",
                            "No formal 200-scene protocol or paper metric reproduction is certified."],
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def import_scores(manifest_path: Path, protocol_path: Path, records_path: Path, out_path: Path) -> Path:
    """Validate all external inputs, then publish a new rating artifact."""
    out_path = Path(out_path)
    if out_path.exists() or out_path.is_symlink():
        raise FileExistsError(out_path)
    result = _build(manifest_path, protocol_path, records_path)
    placement._publish(out_path, result)
    return out_path


def load_scores(path: Path, *, expected_sha256=None) -> dict:
    """Revalidate source bytes and all dependencies; refuse edited imported data."""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("scene score artifact changed before aggregation")
    value = _decode(raw)
    if value.get("schema") != SCHEMA or value.get("status") != "validated":
        raise ValueError("requires a validated scene score artifact")
    sources = value["sources"]
    current = _build(sources["manifest"]["path"], sources["protocol"]["path"], sources["submission"]["path"])
    if current != value:
        raise ValueError("score artifact differs from its verified inputs or implementation")
    return value


def aggregate_scores(path: Path) -> dict:
    """Preserve the declared denominator and distinguish absent/failed ratings."""
    path = Path(path).resolve()
    score_ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    value = load_scores(path, expected_sha256=score_ref["sha256"])
    methods = {}
    for method in value["method_ids"]:
        cases = [r for r in value["cases"] if r["method_id"] == method]
        complete = [r for r in cases if r["composition_status"] == "complete"]
        scored = [r for r in complete if r["score_status"] == "scored"]
        counts = Counter(r["score_status"] for r in complete)
        means = {dim: math.fsum(r["scores"][dim] for r in scored)/len(scored)
                 if scored else INSUFFICIENT for dim in DIMENSIONS}
        methods[method] = {
            "status": "complete" if len(scored) == len(cases) else "partial",
            "n_total": len(cases), "n_complete": len(complete), "n_incomplete": len(cases)-len(complete),
            "completion_rate": len(complete)/len(cases), "n_scored": len(scored),
            "n_judge_failed": counts["failed"], "n_missing_scores": counts["missing"],
            "mean_over_scored": means,
            "mean_over_complete": means if len(scored) == len(complete) else {dim: INSUFFICIENT for dim in DIMENSIONS},
            "human": {dim: INSUFFICIENT for dim in DIMENSIONS},
            "failures": [r for r in cases if r["composition_status"] != "complete" or r["score_status"] != "scored"],
        }
    return {"schema": "metafind.scene_score_summary.v1", "classification": "OBSERVED DATA",
            "scope": "Conditional imported scores, not formal Table 2 or judge-quality evidence",
            "scores_artifact": score_ref,
            "protocol_sha256": value["sources"]["protocol"]["sha256"],
            "manifest_sha256": value["sources"]["manifest"]["sha256"],
            "submission_sha256": value["sources"]["submission"]["sha256"],
            "dimensions": list(DIMENSIONS), "scene_ids": value["scene_ids"], "methods": methods,
            "human_policy": "D-4: no human evaluation; INSUFFICIENT_EVIDENCE",
            "aggregation": "Arithmetic mean per dimension. mean_over_complete is unavailable if any completed scene lacks a valid score; mean_over_scored is explicitly conditional. Incomplete scenes are never imputed."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    importer = subs.add_parser("import", help="Validate explicit protocol, scene outcomes and external ratings")
    for name in ("manifest", "protocol", "records", "out"):
        importer.add_argument("--" + name, type=Path, required=True)
    aggregate = subs.add_parser("aggregate", help="Revalidate then summarize imported ratings")
    aggregate.add_argument("--scores", type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "import":
        import_scores(args.manifest, args.protocol, args.records, args.out)
    else:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(args.out)
        placement._publish(args.out, aggregate_scores(args.scores))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
