"""Run the actual CLI in a child process and retain the command and outcome."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PYTHON = "/home/kyzen/miniconda3/envs/MetaFind/bin/python"
ENV = {
    "METAFIND_DATA": str(HERE / "data"), "METAFIND_TEXT_TEMPLATE": "attrs_v1",
    "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
}
COMMAND = ["nice", "-n", "10", PYTHON, "-m", "metafind.train.stage1", "--device", "cpu",
           "--epochs", "1", "--preload", "--amp", "off", "--phase", "dev", "--selection-split", "dev_val",
           "--query-observation", "same_record", "--query-image-policy", "same_mean",
           "--train-scope", "point_encoder_and_fuser", "--out-dir", "actual_cpu_step"]


def source_hashes():
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((REPO / "metafind").rglob("*.py")) if "vendor" not in p.parts}


def main():
    started = time.monotonic()
    record = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": COMMAND, "environment": ENV, "source_sha256": source_hashes(),
              "classification": "Actual CPU Stage 1 CLI diagnostic; two training + two selection assets, one optimizer step. No mocked models or scorers. Not paper training or Table 1 scores."}
    (HERE / "execution_started.json").write_text(json.dumps(record, indent=2) + "\n")
    with (HERE / "execution.log").open("x") as log:
        result = subprocess.run(COMMAND, cwd=REPO, env=dict(os.environ, **ENV), stdout=log, stderr=subprocess.STDOUT)
    after = source_hashes()
    record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - started,
                  ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_changed=[k for k in sorted(record["source_sha256"].keys() | after.keys())
                                  if record["source_sha256"].get(k) != after.get(k)])
    (HERE / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("exit_code", "elapsed_seconds", "source_changed")}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
