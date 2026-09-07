"""One-shot reviewed replacement of two identified waiting shells, never R2."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/home/kyzen/MetaFindV1')
HERE = Path(__file__).resolve().parent
LOGS = Path('/home/kyzen/metafind/metafind_data_paper/outputs/logs')
JOBS = [
    (1, 2685619, '79771743', 'd18f7accff486a68cdc537ca0d35fa01545b2c8af30b0ab54387843fbda36fb6',
     '0f14b536074703f8afdaf6c686b58519ac8ea170af97581b6e12b84c8f6c02d9'),
    (2, 2494552, '77088621', 'cbd33e69e88c599876e0807fdd336bb8fea018268374c03237278441eed165a0',
     'f90676478ba09bf73dc8b080daa596e9e2e892941cef48483d273fd9a4da90a7'),
]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def identity(pid):
    proc = Path(f'/proc/{pid}')
    stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19],
            'argv': (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
            'cwd': str((proc / 'cwd').resolve()), 'stdout': os.readlink(proc / 'fd/1')}


def children(pid):
    return [int(s) for s in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def r2_waiting():
    state = None
    for line in (LOGS / 'r2_annotate_v10.log').read_text().splitlines():
        if line.startswith('=== R2 START'):
            state = 'waiting'
        elif line.startswith(('=== R2 EXIT', '=== R2 DONE')):
            state = 'terminal'
    return state == 'waiting'


receipt = HERE / 'waiting_chain_deployment.json'
assert not receipt.exists(), 'one-shot deployment already has a receipt'
record = {'started_at': time.time(), 'scope': 'Replace only verified waiting S1/S2 shells', 'jobs': []}
protected = {str(pid): identity(pid) for pid in (2043422, 2043427)}
assert protected['2043422']['start_ticks'] == '67087548'
assert protected['2043427']['start_ticks'] == '67087549'
assert protected['2043427']['argv'][-4:] == ['-m', 'metafind.data.annotate_run', '--prompt-mode', 'figure2_v10']
record['protected_before'] = protected
plans = []
for stage, old, ticks, old_sha, new_sha in JOBS:
    live = LOGS / f'chain_paper_stage{stage}_20260906.sh'
    source = ROOT / f'tools/chain_paper_stage{stage}.sh'
    before = identity(old)
    assert before['start_ticks'] == ticks and before['cwd'] == str(ROOT)
    assert before['argv'] in (['bash', str(live)], ['/bin/bash', str(live)])
    assert before['stdout'] == str(live.with_suffix('.log'))
    assert not live.is_symlink() and live.stat().st_nlink == 1
    assert os.stat(f'/proc/{old}/fd/255').st_ino == live.stat().st_ino
    old_bytes, new_bytes = live.read_bytes(), source.read_bytes()
    assert sha(old_bytes) == old_sha and sha(new_bytes) == new_sha
    subprocess.run(['bash', '-n', str(source)], check=True)
    env = dict(s.split('=', 1) for s in Path(f'/proc/{old}/environ').read_bytes().decode().split('\0') if s)
    assert env['METAFIND_DATA'] == '/home/kyzen/metafind/metafind_data_paper'
    assert r2_waiting()
    owned = children(old)
    assert len(owned) == 1 and identity(owned[0])['argv'] == ['sleep', '60']
    plans.append((stage, old, live, source, before, env, old_bytes, new_bytes))

stopped = None
try:
    for stage, old, live, source, before, env, old_bytes, new_bytes in plans:
        assert identity(old) == before and r2_waiting()
        assert source.read_bytes() == new_bytes and live.read_bytes() == old_bytes
        backup = HERE / f'chain_paper_stage{stage}_before_wait_fix.sh'
        with backup.open('xb') as stream:
            stream.write(old_bytes)
        candidate = live.with_name(live.name + '.validated-wait-fix-20260908.new')
        with candidate.open('xb') as stream:
            stream.write(new_bytes)
        candidate.chmod(live.stat().st_mode & 0o777)
        job = {'stage': stage, 'before': before, 'old_sha256': sha(old_bytes),
               'new_sha256': sha(new_bytes), 'backup': str(backup), 'launcher': str(live),
               'environment_preserved': {k: env.get(k) for k in ('METAFIND_DATA', 'METAFIND_TEXT_TEMPLATE', 'PYTHONPATH')}}
        record['jobs'].append(job)
        os.kill(old, signal.SIGSTOP)
        stopped = old
        time.sleep(0.1)
        assert identity(old) == before and r2_waiting()
        owned = children(old)
        assert len(owned) == 1
        child = identity(owned[0])
        assert child['argv'] == ['sleep', '60']
        job['terminated_wait_child'] = child
        os.kill(old, signal.SIGTERM)
        os.kill(owned[0], signal.SIGTERM)
        os.kill(old, signal.SIGCONT)
        stopped = None
        for _ in range(100):
            if not alive(old):
                break
            time.sleep(0.05)
        assert not alive(old), 'refuse duplicate waiting shell'
        os.replace(candidate, live)
        with live.with_suffix('.log').open('ab', buffering=0) as log:
            log.write(f'\n=== REVIEWED WAITING STAGE{stage} RESTART 2026-09-08: last-attempt failure propagation\n'.encode())
            proc = subprocess.Popen(['/bin/bash', str(live)], cwd=ROOT, env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        job['new_pid'] = proc.pid
        time.sleep(0.5)
        assert proc.poll() is None and r2_waiting()
        owned = children(proc.pid)
        assert len(owned) == 1 and identity(owned[0])['argv'] == ['sleep', '60']
        assert live.read_bytes() == source.read_bytes() == new_bytes
        assert {str(pid): identity(pid) for pid in (2043422, 2043427)} == protected
        job['after'] = identity(proc.pid)
        job['status'] = 'verified_waiting'
    record['protected_after'] = {str(pid): identity(pid) for pid in (2043422, 2043427)}
    record.update(status='verified_waiting', annotation_restarted=False, r2_wrapper_restarted=False)
except BaseException as exc:
    record.update(status='deployment_error', error=repr(exc))
    raise
finally:
    if stopped is not None:
        os.kill(stopped, signal.SIGCONT)
    record['finished_at'] = time.time()
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': record['status'], 'new_pids': [j['new_pid'] for j in record['jobs']],
                  'annotation_pid_unchanged': 2043427, 'r2_wrapper_unchanged': 2043422}))
