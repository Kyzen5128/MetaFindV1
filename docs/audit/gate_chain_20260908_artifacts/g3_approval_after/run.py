"""CPU-only approval regression and stable-corpus read-only gate diagnosis.

Writes only this new evidence directory. Original fixtures, corpus, annotation,
approval source and historical archives are read-only.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BEFORE = OUT.parent / "g3_approval_before"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = REPO / "metafind/gates/g3_object_corpus.py"
    source_sha = sha(source)
    approval = REPO / "workflow/annotation_exclusions_20260828.json"
    approval_sha = sha(approval)
    historical = yaml.safe_load((REPO / "docs/audit/corpus_20260908_artifacts/g3_final/historical/G3_object_corpus.yaml").read_bytes())
    manifest = Path(historical["inputs"]["manifest"]["path"])
    cases = [(label, BEFORE / label / "outputs", BEFORE / label / "lvis.json", 2)
             for label in ("group_name_only", "true_source_hash_wrong_uid")]
    cases.append(("historical", Path("/home/kyzen/metafind/metafind_data/outputs"), manifest, 0))
    results = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1")
    for label, inputs, manifest, expected in cases:
        dest = OUT / label
        dest.mkdir(exist_ok=False)
        record = dest / "G3_object_corpus.yaml"
        command = [sys.executable, "-B", "-m", "metafind.gates.g3_object_corpus",
                   "--outputs", str(inputs), "--manifest", str(manifest), "--record", str(record)]
        start = time.time()
        result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
        (dest / "stdout.log").write_text(result.stdout)
        (dest / "stderr.log").write_text(result.stderr)
        gate = yaml.safe_load(record.read_bytes())
        input_matches = []
        for evidence in gate["inputs"].values():
            for item in evidence if isinstance(evidence, list) else [evidence]:
                if "sha256" in item:
                    input_matches.append(sha(Path(item["path"])) == item["sha256"])
        entry = {"case": label, "command": command, "cwd": str(REPO),
                 "started_epoch": start, "elapsed_seconds": time.time() - start,
                 "expected_rc": expected, "actual_rc": result.returncode,
                 "verdict": gate["verdict"], "record_sha256": sha(record),
                 "source_sha256": source_sha, "approval_sha256": approval_sha,
                 "all_consumed_input_hashes_still_match": all(input_matches),
                 "manual_approval": gate["observed"]["manual_approval"],
                 "accounting": gate["observed"]["accounting"]}
        (dest / "execution.json").write_text(json.dumps(entry, indent=2) + "\n")
        assert result.returncode == expected, (label, result.stdout, result.stderr)
        assert gate["rc"] == expected and input_matches and all(input_matches)
        results.append(entry)
    assert sha(source) == source_sha and sha(approval) == approval_sha
    summary = {"scope": "Two original synthetic counterexamples replayed; one unchanged stable corpus. No formal paper corpus PASS claimed.",
               "source_sha256": source_sha, "approval_sha256": approval_sha,
               "source_and_approval_unchanged": True, "cases": results}
    with (OUT / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"cases": {r["case"]: r["verdict"] for r in results},
                      "source_sha256": source_sha, "all_inputs_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
