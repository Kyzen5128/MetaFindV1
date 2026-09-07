"""Exercise the public evaluator with both newly trained diagnostic stages."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VALIDATION = HERE.parent
PYTHON = "/home/kyzen/miniconda3/envs/MetaFind/bin/python"
PROTOCOL = VALIDATION / "custom_table1_cpu_real_20260907/protocol/protocol.json"
PARENT = VALIDATION / "stage1_cpu_real_20260908/data/outputs/checkpoints/actual_cpu_step/stage1_best_ckpt.json"
CHILD = VALIDATION / "stage2_cpu_real_20260908/data/outputs/variant_ckpts.json"
ENV = {
    "METAFIND_DATA": "/home/kyzen/metafind/metafind_data_attrs", "METAFIND_TEXT_TEMPLATE": "attrs_v1",
    "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
}
COMMAND = ["nice", "-n", "10", PYTHON, "-m", "metafind.eval.custom_table1",
           "--protocol", str(PROTOCOL), "--stage1-record", str(PARENT), "--stage2-record", str(CHILD),
           "--out-dir", str(HERE / "evaluation"), "--device", "cpu", "--batch-size", "2", "--block", "4", "--seed", "20260908"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    return {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO / "metafind").rglob("*.py")) if "vendor" not in p.parts}


def main():
    started = time.monotonic()
    protocol = json.loads(PROTOCOL.read_text())
    prep = json.loads((VALIDATION / "stage1_cpu_real_20260908/preparation.json").read_text())
    record = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": COMMAND, "environment": ENV, "source_sha256": source_hashes(),
              "classification": "Real public custom evaluator diagnostic using both new one-step checkpoints, 2 query / 6 gallery. Not research quality or an independent test.",
              "frozen_protocol": {"path": str(PROTOCOL), "sha256": digest(PROTOCOL)},
              "parent_record_sha256": digest(PARENT), "child_record_sha256": digest(CHILD),
              "query_overlap_with_stage1_train": sorted(set(protocol["query_uids"]) & set(prep["train_uids"])),
              "query_overlap_with_stage1_selection": sorted(set(protocol["query_uids"]) & set(prep["selection_uids"]))}
    (HERE / "execution_started.json").write_text(json.dumps(record, indent=2) + "\n")
    with (HERE / "execution.log").open("x") as log:
        result = subprocess.run(COMMAND, cwd=REPO, env=dict(os.environ, **ENV), stdout=log, stderr=subprocess.STDOUT)
    after = source_hashes()
    record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - started,
                  ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_changed=[k for k in sorted(record["source_sha256"].keys() | after.keys()) if record["source_sha256"].get(k) != after.get(k)])
    (HERE / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("exit_code", "elapsed_seconds", "source_changed")}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
