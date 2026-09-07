"""Exercise the final heredoc with real input bytes copied into a temporary corpus."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,time
ROOT=Path('/home/kyzen/MetaFindV1'); HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/kyzen/metafind/metafind_data/outputs/annotation_exclusions.json')
ANN=Path('/home/kyzen/metafind/metafind_data_paper/outputs/annotations').resolve(strict=True)
PYTHON='/home/kyzen/miniconda3/envs/MetaFind/bin/python'
sha=lambda raw: hashlib.sha256(raw).hexdigest()
script=(ROOT/'tools/chain_paper_stage1.sh').read_bytes()
assert sha(script)=='d18f7accff486a68cdc537ca0d35fa01545b2c8af30b0ab54387843fbda36fb6'
body=script.decode().split("$PY - <<'PYEOF'\n",1)[1].split('\nPYEOF',1)[0]
source_bytes=SOURCE.read_bytes(); value=json.loads(source_bytes)
uids=sorted(item['uid'] if isinstance(item,dict) else item for item in value['groups']['manual_review_rejected']['uids'])
assert len(set(uids))==len(uids)==21
copied={uid:(ANN/f'{uid}.json').read_bytes() for uid in uids if (ANN/f'{uid}.json').is_file()}
record={'scope':'real source bytes; all writes isolated in TemporaryDirectory; no actual annotation mutation',
        'timestamp':time.time(),'script_sha256':sha(script),'source_path':str(SOURCE),'source_sha256':sha(source_bytes),
        'manual_count':21,'copied_annotation_sha256':{k:sha(v) for k,v in copied.items()}}
with tempfile.TemporaryDirectory(prefix='metafind_manual_e_ledger_') as temp:
    base=Path(temp); ann=base/'outputs/annotations'; ann.mkdir(parents=True)
    for uid,raw in copied.items(): (ann/f'{uid}.json').write_bytes(raw)
    retried=value['groups']['n05_quarantine']['uids'][0]
    assert retried not in uids
    (ann/f'{retried}.json').write_bytes(b'historical failure retried; isolated fixture')
    env={**os.environ,'METAFIND_DATA':str(base),'METAFIND_EXCLUSION_LEDGER':str(SOURCE),'PYTHONPATH':str(ROOT),
         'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':''}
    def run():
        r=subprocess.run([PYTHON,'-'],input=body,env=env,cwd=ROOT,capture_output=True,text=True,timeout=30)
        assert r.returncode==0,r.stderr
        return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    record['first']=run()
    ledger=base/'outputs/annotation_exclusions.json'; payload=json.loads(ledger.read_bytes())
    assert payload['groups']=={'manual_review_rejected':{'n':21,'uids':uids}}
    assert payload['excluded_total']==21 and payload['accounting_decision']=='DL-106'
    assert payload['source_ledger']['sha256']==sha(source_bytes)
    assert not {'corpus_before','corpus_after','rendered_assets'} & payload.keys()
    stat=ledger.stat(); first=(ledger.read_bytes(),stat.st_ino,stat.st_mtime_ns)
    assert all(not (ann/f'{uid}.json').exists() for uid in uids)
    assert all((base/'outputs/annotations_v10_excluded'/f'{uid}.json').read_bytes()==raw for uid,raw in copied.items())
    assert (ann/f'{retried}.json').exists()
    record['repeat']=run()
    stat=ledger.stat()
    assert (ledger.read_bytes(),stat.st_ino,stat.st_mtime_ns)==first
    assert 'moved 0 of 21' in record['repeat']['stdout']
    record.update(published_ledger=payload,ledger_sha256=sha(first[0]),repeated_ledger_identity_unchanged=True,
                  available_copied_and_moved=len(copied),historical_failure_retained=True)
assert SOURCE.read_bytes()==source_bytes
assert all((ANN/f'{uid}.json').read_bytes()==raw for uid,raw in copied.items())
record.update(real_inputs_unchanged=True,real_annotation_written=False,status='PASS')
with (HERE/'manual_e_ledger_real_copy.json').open('x') as f: json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:record[k] for k in ['status','manual_count','available_copied_and_moved','real_inputs_unchanged']},indent=2))
