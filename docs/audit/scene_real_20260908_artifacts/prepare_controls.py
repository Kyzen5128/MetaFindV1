"""Declare two CPU replay controls using identical frozen raw-encoded inputs."""
import copy
import hashlib
import io
import json
from pathlib import Path

import torch

WORK = Path(__file__).resolve().parent
BASE = WORK / "bundle_02"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    raw_manifest = (BASE / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    raw_inputs = (BASE / manifest["inputs"]["path"]).read_bytes()
    assert sha(raw_inputs) == manifest["inputs"]["sha256"]
    weights = BASE / manifest["model"]["weights"]["path"]
    assert sha(weights.read_bytes()) == manifest["model"]["weights"]["sha256"]
    inputs = torch.load(io.BytesIO(raw_inputs), map_location="cpu", weights_only=True)
    for name, change in (("parallel", {"mode": "parallel"}),
                         ("layout_off", {"use_layout": False})):
        out = WORK / ("control_" + name)
        out.mkdir(exist_ok=False)
        altered = dict(inputs, **change)
        torch.save(altered, out / "inputs.pt")
        m = copy.deepcopy(manifest)
        m["model"]["weights"]["path"] = str(weights)
        m["inputs"] = {"path": "inputs.pt", "sha256": sha((out / "inputs.pt").read_bytes())}
        m["provenance"]["diagnostic_control"] = {
            "parent_manifest_path": str(BASE / "manifest.json"),
            "parent_manifest_sha256": sha(raw_manifest), "changes": change,
            "script_sha256": sha(Path(__file__).read_bytes()),
            "classification": "IMPLEMENTATION CHOICE: same encoded tensors/model and three declared slots; no quality claim",
        }
        with (out / "manifest.json").open("x") as stream:
            json.dump(m, stream, indent=2)
    print("Prepared explicit parallel and layout-off controls; did not execute models")


if __name__ == "__main__":
    main()
