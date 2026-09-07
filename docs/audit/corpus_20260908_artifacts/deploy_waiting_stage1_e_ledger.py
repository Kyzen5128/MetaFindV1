"""One reviewed deployment of the waiting shell; never restart annotation."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/home/kyzen/MetaFindV1')
HERE = Path(__file__).resolve().parent
LIVE = Path('/home/kyzen/metafind/metafind_data_paper/outputs/logs/chain_paper_stage1_20260906.sh')
SOURCE = ROOT / 'tools/chain_paper_stage1.sh'
OLD, ANNOTATION, STAGE2 = 2668419, 2043427, 2494552


def identity(pid):
    proc = Path(f'/proc/{pid}')
    stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19],
            'argv': (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0'),
            'cwd': str((proc / 'cwd').resolve()),
            'stdout': os.readlink(proc / 'fd/1')}


def children(pid):
    return [int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]


def waiting_marker_absent():
    with (LIVE.parent / 'r2_annotate_v10.log').open() as stream:
        return not any(line.startswith('=== R2 DONE') for line in stream)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


record = {'started_at': time.time(), 'scope': 'Restart only verified waiting Stage 1 shell after code review and isolated tests'}
receipt = HERE / 'stage1_e_ledger_deployment.json'
assert not receipt.exists(), 'do not repeat a deployment'
before = {str(pid): identity(pid) for pid in (OLD, ANNOTATION, STAGE2)}
record['before'] = before
assert {pid: before[str(pid)]['start_ticks'] for pid in (OLD, ANNOTATION, STAGE2)} == {
    OLD: '79568402', ANNOTATION: '67087549', STAGE2: '77088621'}
assert before[str(OLD)]['argv'] == ['/bin/bash', str(LIVE)]
assert before[str(OLD)]['cwd'] == str(ROOT)
assert before[str(OLD)]['stdout'] == str(LIVE.with_suffix('.log'))
assert before[str(ANNOTATION)]['argv'][-4:] == ['-m', 'metafind.data.annotate_run', '--prompt-mode', 'figure2_v10']
assert before[str(ANNOTATION)]['cwd'] == str(ROOT)
assert str(Path(f'/proc/{OLD}/fd/255').resolve()) == str(LIVE)
assert os.stat(f'/proc/{OLD}/fd/255').st_ino == LIVE.stat().st_ino
assert not LIVE.is_symlink()
old_bytes = LIVE.read_bytes()
assert sha(old_bytes) == '77c8a588371e8d32bf2b8e88a79451f16b01bff46c114e46e7237de1c77fb8fb'
new_bytes = SOURCE.read_bytes()
assert sha(new_bytes) == 'd18f7accff486a68cdc537ca0d35fa01545b2c8af30b0ab54387843fbda36fb6'
subprocess.run(['bash', '-n', str(SOURCE)], check=True)
env = dict(item.split('=', 1) for item in Path(f'/proc/{OLD}/environ').read_bytes().decode().split('\0') if item)
assert env['METAFIND_DATA'] == '/home/kyzen/metafind/metafind_data_paper'
record['environment_preserved'] = {k: env.get(k) for k in ('METAFIND_DATA', 'METAFIND_TEXT_TEMPLATE', 'PYTHONPATH')}
assert waiting_marker_absent()
backup = HERE / 'chain_paper_stage1_before_e_ledger.sh'
with backup.open('xb') as stream:
    stream.write(old_bytes)
candidate = LIVE.with_name(LIVE.name + '.validated-e-ledger-20260908.new')
with candidate.open('xb') as stream:
    stream.write(new_bytes)
candidate.chmod(LIVE.stat().st_mode & 0o777)
record.update(old_sha256=sha(old_bytes), new_sha256=sha(new_bytes), backup=str(backup), launcher=str(LIVE))
stopped = False
try:
    assert identity(OLD) == before[str(OLD)] and waiting_marker_absent()
    os.kill(OLD, signal.SIGSTOP)
    stopped = True
    time.sleep(0.1)
    assert identity(OLD) == before[str(OLD)] and waiting_marker_absent()
    owned = children(OLD)
    assert len(owned) == 1
    child = identity(owned[0])
    assert child['argv'] == ['sleep', '60'], child['argv']
    record['terminated_wait_child'] = child
    # TERM is queued while Bash is stopped. End only its verified sleep child,
    # then resume Bash to deliver TERM; never signal a group or annotation PID.
    os.kill(OLD, signal.SIGTERM)
    os.kill(owned[0], signal.SIGTERM)
    os.kill(OLD, signal.SIGCONT)
    stopped = False
    for _ in range(100):
        if not alive(OLD):
            break
        time.sleep(0.05)
    assert not alive(OLD), 'old waiting shell did not exit; do not start a duplicate'
    os.replace(candidate, LIVE)
    with LIVE.with_suffix('.log').open('ab', buffering=0) as log:
        log.write(b'\n=== REVIEWED WAITING STAGE1 RESTART 2026-09-08: durable manual exclusion E ledger (DL-106)\n')
        process = subprocess.Popen(['/bin/bash', str(LIVE)], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record['new_pid'] = process.pid
    time.sleep(0.4)
    assert process.poll() is None
    assert waiting_marker_absent()
    new_children = children(process.pid)
    assert len(new_children) == 1 and identity(new_children[0])['argv'] == ['sleep', '60']
    assert LIVE.read_bytes() == SOURCE.read_bytes() == new_bytes
    assert identity(ANNOTATION) == before[str(ANNOTATION)]
    assert identity(STAGE2) == before[str(STAGE2)]
    record['after'] = {str(pid): identity(pid) for pid in (process.pid, ANNOTATION, STAGE2)}
    record.update(status='verified_waiting', annotation_restarted=False,
                  stage2_restarted=False, annotation_artifacts_modified=False)
except BaseException as error:
    record.update(status='deployment_error', error=repr(error))
    raise
finally:
    if stopped:
        os.kill(OLD, signal.SIGCONT)
    record['finished_at'] = time.time()
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': record['status'], 'new_pid': record['new_pid'],
                  'new_sha256': record['new_sha256'], 'annotation_pid_unchanged': ANNOTATION,
                  'stage2_pid_unchanged': STAGE2}, indent=2))
