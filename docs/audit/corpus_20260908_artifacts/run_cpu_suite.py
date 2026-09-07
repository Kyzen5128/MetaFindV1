from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

root = Path('/home/kyzen/MetaFindV1')
prior = json.loads((root/'docs/audit/reproduction_scene_real_20260908_cpu_execution.json').read_text())
source_names = sorted(set(prior['source_sha256']) | {
    'metafind/eval/scene_scores.py', 'metafind/gates/g3_object_corpus.py',
    'tests/eval/test_scene_scores.py', 'tests/pipeline/test_object_corpus_gate.py'})


def hashes():
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_names}


record = {'command': [prior['command'][0], '-m', 'pytest', 'tests', '-q', '-ra', '-p',
                      'no:cacheprovider', '--ignore=tests/gpu', '--ignore=tests/hooks',
                      '--basetemp', '/tmp/metafind_corpus_20260908_cpu'],
          'environment': prior['environment'], 'started_at': time.time(),
          'source_sha256': hashes()}
log = root/'docs/audit/reproduction_corpus_20260908_cpu.log'
with log.open('x') as stream:
    p = subprocess.run(record['command'], cwd=root,
                       env={**os.environ, **record['environment']}, stdout=stream, stderr=subprocess.STDOUT)
record.update(returncode=p.returncode, elapsed_seconds=time.time()-record['started_at'],
              source_sha256_after=hashes(), log_path=str(log),
              log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
record['source_unchanged'] = record['source_sha256'] == record['source_sha256_after']
(root/'docs/audit/reproduction_corpus_20260908_cpu_execution.json').write_text(json.dumps(record, indent=2)+'\n')
print(log.read_text()[-9000:])
assert record['source_unchanged'], 'source changed during suite'
raise SystemExit(p.returncode)
