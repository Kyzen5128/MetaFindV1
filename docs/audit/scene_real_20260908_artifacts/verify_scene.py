"""Independently tally real scene replay, controls, scores and unchanged sources."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def inputs(folder):
    m = read(folder / "manifest.json")
    record = m["inputs"]
    p = folder / record["path"]
    assert sha(p) == record["sha256"]
    return torch.load(io.BytesIO(p.read_bytes()), map_location="cpu", weights_only=True)


def main():
    source = inputs(WORK / "bundle_02")
    request = read(WORK / "request_02.json")
    base = read(WORK / "composition.json")
    assert base == read(WORK / "bundle_02/validation_composition.json")
    gallery = source["gallery"].numpy().astype(np.float64)
    gallery /= np.sqrt((gallery * gallery).sum(axis=1, keepdims=True))
    assert np.isfinite(gallery).all()
    groups = {}
    for name, relative, context_sizes, changes in (
        ("iterative", "composition.json", [0, 1, 2], {}),
        ("parallel", "control_parallel/composition.json", [0, 0, 0], {"mode": "parallel"}),
        ("layout_off", "control_layout_off/composition.json", [0, 1, 2], {"use_layout": False}),
    ):
        result = read(WORK / relative)
        if changes:
            control = inputs((WORK / relative).parent)
            assert equal(control, dict(source, **changes))
        assert result["status"] == "complete" and len(result["trace"]) == 3
        assert result["gallery_ids"] == source["gallery_ids"]
        assert [len(t["context"]["node_ids"]) for t in result["trace"]] == context_sizes
        assert [n["slot"] for n in result["final_graph"]["nodes"]] == [q["slot"] for q in request["queries"]]
        assert len(result["final_graph"]["nodes"]) == 3
        selected, deltas, residual_norms = [], [], []
        for i, t in enumerate(result["trace"]):
            q = np.array(t["query_embedding"], dtype=np.float64)
            assert q.shape == (1280,) and np.isfinite(q).all()
            scores = gallery @ (q / np.sqrt((q*q).sum()))
            order = np.argsort(-scores, kind="stable")[:5]
            assert [r["asset_id"] for r in t["top5"]] == [source["gallery_ids"][j] for j in order]
            np.testing.assert_allclose([r["score"] for r in t["top5"]], scores[order], atol=1e-14, rtol=0)
            assert t["selected_asset_id"] == t["top5"][0]["asset_id"]
            assert t["selected_node_text"] == source["asset_texts"][t["selected_asset_id"]]["text"]
            assert t["fused_norm"] == base["trace"][i]["fused_norm"]
            selected.append(t["selected_asset_id"])
            deltas.append(float(np.linalg.norm(q - np.array(base["trace"][i]["query_embedding"]))))
            residual_norms.append(t["lambda_layout_norm"])
        if name == "iterative":
            assert residual_norms[0] == 0 and all(v > 0 for v in residual_norms[1:])
        else:
            assert residual_norms == [0, 0, 0] and deltas[0] == 0 and all(v > 0 for v in deltas[1:])
        groups[name] = {"selected": selected, "context_sizes": context_sizes,
                        "residual_norms": residual_norms, "delta_from_iterative": deltas,
                        "sha256": sha(WORK / relative)}
    cpu = read(REPO / "docs/audit/reproduction_scene_real_20260908_cpu_execution.json")
    changed = [p for p, h in cpu["source_sha256"].items() if sha(REPO / p) != h]
    assert not changed and cpu["exit_code"] == 0
    phases = {}
    for path in sorted(WORK.glob("*_execution.json")):
        value = read(path)
        if "phase" not in value:
            continue
        assert not value["source_changed"]
        assert all(sha(REPO / p) == h for p, h in value["source_sha256"].items())
        phases[value["phase"]] = {k: value[k] for k in ("exit_code", "elapsed_seconds")}
    for label in ("00", "01"):
        assert not (WORK / f"bundle_{label}/manifest.json").exists()
        assert phases[f"prepare_{label}"]["exit_code"] == 1
    assert all(r["exit_code"] == 0 for k, r in phases.items() if k not in ("prepare_00", "prepare_01"))
    for label in ("01", "02"):
        r = read(WORK / f"semantics_{label}/record.json")
        assert r["generated_successes"] == 1 and r["generated_degraded"] == 0
        assert r["llm"]["invoked"] is True
        assert all(sha(Path(p)) == h for p, h in r["sources"].items())
    image_path = WORK / "blender_output/view_000.png"
    with Image.open(image_path) as image:
        assert image.size == (512, 512) and image.mode == "RGBA"
    geometry = read(WORK / "independent_geometry/verified_geometry.json")
    record = {"classification": "OBSERVED DATA: finite real-model three-slot execution; not quality or formal Table2 evidence",
              "script_sha256": sha(Path(__file__)), "groups": groups, "phase_results": phases,
              "three_composition_score_tallies": "PASS", "prepare_publication_fail_closed": "PASS",
              "original_replay_exact": True, "control_input_differences_only_declared": True,
              "cpu_sources_verified": len(cpu["source_sha256"]), "cpu_source_changed": changed,
              "image": {"sha256": sha(image_path), "size": [512, 512]},
              "independent_geometry_sha256": sha(WORK / "independent_geometry/verified_geometry.json"),
              "geometry_summary": geometry}
    with (WORK / "verification.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({"groups": groups, "cpu_sources": len(cpu["source_sha256"]), "verification": "PASS"}))


if __name__ == "__main__":
    main()
