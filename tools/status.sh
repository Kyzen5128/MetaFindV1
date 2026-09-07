#!/bin/bash
# Read-only status; accepts the same arguments as tools/status.py.
# bash tools/status.sh --data /path/to/corpus [--json]
set -euo pipefail
TASK_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${METAFIND_PYTHON:-python3}" -B "$TASK_REPO/tools/status.py" "$@"
