"""Record the real graph-checker CLI and its exact input/source snapshot."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time

root = Path('/home/kyzen/MetaFindV1')
record_path = root/'docs/audit/reproduction_corpus_20260908_delivery_graph.json'
log = record_path.with_suffix('.log')
assert not record_path.exists() and not log.exists(), 'delivery graph evidence already exists'


def snapshot():
    files = list((root/'docs/graph').glob('*'))
    files += list((root/'docs/audit').glob('*.md'))
    inventory = root/'docs/audit/formula_inventory_validation.json'
    if inventory.exists():
        files.append(inventory)
    for directory in ('metafind', 'tools', 'setup', 'tests'):
        files += [p for p in (root/directory).rglob('*.py')
                  if 'vendor' not in p.relative_to(root).parts and '__pycache__' not in p.parts]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files)) if p.is_file() and not p.is_symlink()}


environment = {'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
               'HIP_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
record = {'command': ['/home/kyzen/miniconda3/envs/MetaFind/bin/python', 'tools/check_graph.py'],
          'environment': environment, 'started_at': time.time(), 'inputs_sha256': snapshot(),
          'test_file_count': len(list((root/'tests').rglob('test_*.py'))),
          'test_def_count': sum(len(re.findall(r'^def test_', p.read_text(), re.M))
                                for p in (root/'tests').rglob('test_*.py')),
          'scope': 'Graph specification/source declaration checks; not runtime execution or paper fidelity.',
          'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with log.open('x') as stream:
    process = subprocess.run(record['command'], cwd=root, env={**os.environ, **environment},
                             stdout=stream, stderr=subprocess.STDOUT)
record.update(returncode=process.returncode, elapsed_seconds=time.time()-record['started_at'],
              inputs_sha256_after=snapshot(), log_path=str(log),
              log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
record['inputs_unchanged'] = record['inputs_sha256'] == record['inputs_sha256_after']
output = log.read_text()
match = re.search(r'^(\d+) checks$', output, re.M)
record['checks'] = int(match.group(1)) if match else None
match = re.search(r'^channels (\d+)  nodes (\d+)  edges (\d+)  gates (\d+)  L1 (\d+)  L2 (\d+)$', output, re.M)
record['graph_counts'] = dict(zip(('channels', 'nodes', 'edges', 'gates', 'L1', 'L2'), map(int, match.groups()))) if match else {}
with record_path.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(output, end='')
print(json.dumps({'returncode': record['returncode'], 'inputs_unchanged': record['inputs_unchanged'],
                  'checks': record['checks'], 'record': str(record_path)}, indent=2))
assert record['inputs_unchanged'], 'graph inputs changed during the check'
raise SystemExit(process.returncode)
