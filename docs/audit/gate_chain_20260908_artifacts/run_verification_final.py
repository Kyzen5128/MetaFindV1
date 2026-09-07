"""Capture the final CPU integration run with source hashes and exact scope."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/home/kyzen/MetaFindV1')
OUT = Path(__file__).resolve().parent
PYTHON = '/home/kyzen/miniconda3/envs/MetaFind/bin/python'
ENV = dict(PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
           HIP_VISIBLE_DEVICES='', PYOPENGL_PLATFORM='egl', LIBGL_ALWAYS_SOFTWARE='1',
           MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',
           __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
           METAFIND_DATA='/home/kyzen/metafind/metafind_data')


def snapshot():
    files = [p for folder in ('metafind', 'tools', 'tests', 'setup')
             for p in (ROOT / folder).rglob('*') if p.suffix in {'.py', '.sh', '.patch'} and p.is_file()]
    files.append(ROOT / 'workflow/annotation_exclusions_20260828.json')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


command = [PYTHON, '-m', 'pytest', 'tests', '-q', '-ra', '-p', 'no:cacheprovider',
           '--ignore=tests/gpu', '--ignore=tests/hooks']
before = snapshot()
start = time.time()
log = OUT / 'integration_cpu_final.log'
with log.open('xb') as stream:
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENV},
                            stdout=stream, stderr=subprocess.STDOUT)
after = snapshot()
record = {'command': command, 'environment': ENV, 'cwd': str(ROOT),
          'started_at': start, 'finished_at': time.time(), 'exit_code': result.returncode,
          'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'source_sha256_before': before, 'source_sha256_after': after,
          'source_unchanged': before == after,
          'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
          'scope': 'CPU integration only. GPU/hook groups excluded; no formal training or paper table result.'}
(OUT / 'integration_cpu_final.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'exit_code': result.returncode, 'source_unchanged': before == after,
                  'source_files': len(before), 'seconds': record['finished_at'] - start,
                  'last_lines': log.read_text().splitlines()[-3:]}))
raise SystemExit(result.returncode if before == after else 99)
