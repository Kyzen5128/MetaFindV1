"""Retain actual CLI outcome and source snapshot for each isolated scene phase."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[2]
ENV = {"METAFIND_DATA": str(WORK / "data"), "METAFIND_TEXT_TEMPLATE": "attrs_v1",
       "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
       "LIBGL_ALWAYS_SOFTWARE": "1", "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
       "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json"}


def snapshot():
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((REPO / "metafind").rglob("*.py")) if "vendor" not in p.parts}


def main():
    name, module, *args = sys.argv[1:]
    if not name.replace("_", "").isalnum():
        raise ValueError("phase name must be alphanumeric/underscores")
    command = ["nice", "-n", "10", sys.executable, "-m", module, *args]
    record = {"phase": name, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": command, "environment": ENV, "source_sha256": snapshot(),
              "classification": "Real isolated CPU scene diagnostic; declared three slots, six admitted assets, one-step checkpoints; not formal Table2."}
    with (WORK / f"{name}_started.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    start = time.monotonic()
    with (WORK / f"{name}.log").open("x") as stream:
        run = subprocess.run(command, cwd=REPO, env=dict(os.environ, **ENV), stdout=stream, stderr=subprocess.STDOUT)
    after = snapshot()
    record.update(exit_code=run.returncode, elapsed_seconds=time.monotonic()-start,
                  ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_changed=[p for p,h in record["source_sha256"].items() if after.get(p)!=h])
    with (WORK / f"{name}_execution.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({k:record[k] for k in ("phase", "exit_code", "elapsed_seconds", "source_changed")}))
    raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
