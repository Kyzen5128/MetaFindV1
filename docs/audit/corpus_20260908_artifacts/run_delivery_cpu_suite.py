"""Fresh CPU delivery verification; never overwrites earlier run evidence."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import subprocess
import time

root = Path('/home/kyzen/MetaFindV1')
prior = json.loads((root/'docs/audit/reproduction_corpus_20260908_final_cpu_execution.json').read_text())
log = root/'docs/audit/reproduction_corpus_20260908_delivery_cpu.log'
execution = root/'docs/audit/reproduction_corpus_20260908_delivery_cpu_execution.json'
assert not log.exists() and not execution.exists(), 'delivery evidence already exists'


def sources():
    names = set(prior['source_sha256'])
    for directory in ('metafind', 'tools', 'setup', 'tests'):
        names.update(str(p.relative_to(root)) for p in (root/directory).rglob('*')
                     if p.is_file() and not p.is_symlink() and p.suffix in {'.py', '.sh', '.patch'}
                     and 'vendor' not in p.relative_to(root).parts and '__pycache__' not in p.parts)
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)}


tree = ast.parse((root/'tools/dump_rules.py').read_text())
rule_paths = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in {'OUT', 'PREAMBLE'}:
            rule_paths[name] = str(eval(compile(ast.Expression(node.value), '<path assignment>', 'eval'), {'ROOT': root}))
assert rule_paths == {'OUT': str(root/'docs/history/RULES_SNAPSHOT.md'),
                      'PREAMBLE': str(root/'docs/history/_rules_preamble.md')}
assert all(Path(p).is_file() for p in rule_paths.values())
test_files = sorted((root/'tests').rglob('test_*.py'))
test_defs = sum(len(re.findall(r'^def test_', p.read_text(), re.M)) for p in test_files)
record = {
    'command': [prior['command'][0], '-m', 'pytest', 'tests', '-q', '-ra', '-p',
                'no:cacheprovider', '--ignore=tests/gpu', '--ignore=tests/hooks',
                '--basetemp', '/tmp/metafind_corpus_20260908_delivery_cpu'],
    'environment': prior['environment'], 'started_at': time.time(),
    'source_sha256': sources(), 'source_scope': 'Previous 253 source files plus current Python, shell and patch files under metafind/tools/setup/tests; vendor excluded; no model files hashed.',
    'test_file_count': len(test_files), 'test_def_count': test_defs,
    'dump_rules_check': {'method': 'AST parse and evaluation of OUT/PREAMBLE path assignments only; generator never executed', 'resolved_paths': rule_paths},
    'runner_path': str(Path(__file__).resolve()),
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
with log.open('x') as stream:
    process = subprocess.run(record['command'], cwd=root,
                             env={**os.environ, **record['environment']},
                             stdout=stream, stderr=subprocess.STDOUT)
record.update(returncode=process.returncode, elapsed_seconds=time.time()-record['started_at'],
              source_sha256_after=sources(), log_path=str(log),
              log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
record['source_unchanged'] = record['source_sha256'] == record['source_sha256_after']
summary = next((line for line in reversed(log.read_text().splitlines())
                if re.search(r'\d+ passed', line)), '')
record['pytest_summary'] = summary
with execution.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps({'returncode': process.returncode, 'elapsed_seconds': record['elapsed_seconds'],
                  'summary': summary, 'source_unchanged': record['source_unchanged'],
                  'source_file_count': len(record['source_sha256']),
                  'log_path': str(log), 'execution_path': str(execution)}, indent=2), flush=True)
assert record['source_unchanged'], 'source changed during suite'
raise SystemExit(process.returncode)
