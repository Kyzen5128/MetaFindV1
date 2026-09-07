"""Bind the previous real render to the new importer with NO ratings/model run."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/home/kyzen/MetaFindV1')
HERE = Path(__file__).resolve().parent / 'scene_score_handoff'
HERE.mkdir(exist_ok=False)
SCENE = ROOT / 'output/validation/scene_cpu_real_20260908'
PY = '/home/kyzen/miniconda3/envs/MetaFind/bin/python'
from metafind.eval.scene_scores import DIMENSIONS


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(name, value):
    path = HERE / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return path


composition = SCENE / 'composition.json'
result = SCENE / 'blender_output/result.json'
room_id = json.loads(composition.read_bytes())['room_id']
cameras = [r['camera_id'] for r in json.loads(result.read_bytes())['renders']]
protocol = write('diagnostic_protocol.json', {
    'schema': 'metafind.scene_judge_protocol.v1', 'status': 'frozen',
    'dimensions': list(DIMENSIONS), 'score_range': [1, 5],
    'model': {'id': 'gemma-4-12B-it', 'revision': 'NOT_EXECUTED: artifact validation fixture only'},
    'prompt': 'DIAGNOSTIC ONLY. No judge request is made. No prompt or rubric for formal evaluation is selected.',
    'generation': {'executed': False},
    'view_policy': 'Artifact validation fixture: enumerate existing renders; no image is sent to a judge.',
    'provenance': {'classification': 'IMPLEMENTATION CHOICE',
                   'scope': 'Non-evaluation CLI diagnostic with zero ratings, not a research protocol decision.'}})
manifest = write('manifest.json', {
    'schema': 'metafind.scene_evaluation_manifest.v1', 'status': 'frozen',
    'scene_ids': [room_id], 'method_ids': ['diagnostic_real_one_step_s1_s2'],
    'provenance': {'source': str(SCENE), 'scope': 'Previous actual 3-slot CPU render; zero scores; no generalization claim.'},
    'outcomes': [{'method_id': 'diagnostic_real_one_step_s1_s2', 'scene_id': room_id,
                  'status': 'complete', 'composition': ref(composition),
                  'placement': ref(SCENE / 'placement_bundle/placement.json'),
                  'result': ref(result), 'judged_camera_ids': cameras}]})
submission = write('empty_submission.json', {
    'schema': 'metafind.scene_score_submission.v1',
    'protocol_sha256': ref(protocol)['sha256'], 'manifest_sha256': ref(manifest)['sha256'],
    'records': []})
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
       'HIP_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
commands = [
    [PY, '-m', 'metafind.eval.scene_scores', 'import', '--manifest', str(manifest),
     '--protocol', str(protocol), '--records', str(submission), '--out', str(HERE / 'imported.json')],
    [PY, '-m', 'metafind.eval.scene_scores', 'aggregate', '--scores', str(HERE / 'imported.json'),
     '--out', str(HERE / 'summary.json')]]
records = []
for i, command in enumerate(commands):
    started = time.time()
    p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    records.append({'command': command, 'returncode': p.returncode,
                    'stdout': p.stdout, 'stderr': p.stderr, 'elapsed_seconds': time.time()-started})
    write(f'command_{i}.json', records[-1])
    assert p.returncode == 0, p.stderr
summary = json.loads((HERE / 'summary.json').read_bytes())
method = summary['methods']['diagnostic_real_one_step_s1_s2']
assert (method['n_total'], method['n_complete'], method['n_scored'], method['n_missing_scores']) == (1, 1, 0, 1)
for field in ('mean_over_complete', 'mean_over_scored', 'human'):
    assert set(method[field].values()) == {'INSUFFICIENT_EVIDENCE'}
write('verification.json', {'status': 'verified', 'timestamp': time.time(),
    'scope': 'True prior composition/placement/GLB/annotation/Blender/render inputs; NO judge or rating',
    'expected_counts': {'n_total': 1, 'n_complete': 1, 'n_scored': 0, 'n_missing_scores': 1},
    'all_means_and_human': 'INSUFFICIENT_EVIDENCE', 'judge_executed': False,
    'source': ref(ROOT / 'metafind/eval/scene_scores.py'),
    'inputs': {p.name: ref(p) for p in (protocol, manifest, submission)},
    'outputs': {p.name: ref(p) for p in (HERE / 'imported.json', HERE / 'summary.json')}})
print(json.dumps({'status': 'verified', 'n_total': 1, 'n_complete': 1,
                  'n_scored': 0, 'n_missing_scores': 1, 'judge_executed': False}, indent=2))
