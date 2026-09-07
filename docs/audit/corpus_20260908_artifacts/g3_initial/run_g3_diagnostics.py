"""Read actual corpora; keep all gate/progress writes in this diagnostic tree."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import yaml

ROOT = Path('/home/kyzen/MetaFindV1')
HERE = Path(__file__).resolve().parent
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
       'HIP_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
results = []
for label, data in (('paper_in_progress', Path('/home/kyzen/metafind/metafind_data_paper')),
                    ('historical_stable', Path('/home/kyzen/metafind/metafind_data'))):
    out = HERE / label
    out.mkdir(exist_ok=False)
    record = out / 'G3_object_corpus.yaml'
    command = ['/home/kyzen/miniconda3/envs/MetaFind/bin/python', '-m',
               'metafind.gates.g3_object_corpus', '--outputs', str(data/'outputs'),
               '--manifest', str(data/'datasets/objaverse-lvis/lvis.json'),
               '--record', str(record)]
    started = time.time()
    p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    result = {'label': label, 'command': command, 'returncode': p.returncode,
              'stdout': p.stdout, 'stderr': p.stderr, 'elapsed_seconds': time.time()-started,
              'gate_source_sha256': hashlib.sha256((ROOT/'metafind/gates/g3_object_corpus.py').read_bytes()).hexdigest(),
              'gate_record_sha256': hashlib.sha256(record.read_bytes()).hexdigest() if record.exists() else None}
    (out/'execution.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    assert p.returncode in (0, 2, 3, 4), p.stderr
    value = yaml.safe_load(record.read_bytes())
    assert value['rc'] == p.returncode and value['is_terminal']
    results.append({'label': label, 'returncode': p.returncode, 'verdict': value['verdict'],
                    'accounting': value['observed'].get('accounting'),
                    'failures': value['observed']['failures'],
                    'blocked': value['observed']['blocked_reasons']})
    print(json.dumps(results[-1], ensure_ascii=False, indent=2), flush=True)
(HERE/'g3_diagnostics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
