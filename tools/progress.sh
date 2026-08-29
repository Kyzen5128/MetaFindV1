#!/usr/bin/env bash
# 一眼看訓練進度。Kyzen 2026-08-29 要的。
cd "$(dirname "$0")/.." || exit 1
$HOME/miniconda3/envs/MetaFind/bin/python - <<'PY'
import json, time, subprocess
L='data/outputs/logs/'
rows=[json.loads(l) for l in open(L+'train_stage1.jsonl') if l.strip()]
if not rows: print("還沒有訓練紀錄"); raise SystemExit
rid=rows[-1]['run_id']; run=[r for r in rows if r.get('run_id')==rid]
t0,t1,s0,s1=run[0]['ts'],run[-1]['ts'],run[0]['step'],run[-1]['step']
ep=max(r['epoch'] for r in run)+1
TOT=None
try:
    out=subprocess.run(['journalctl','--user','-u','metafind-train-600w','--no-pager','-o','cat'],
                       capture_output=True,text=True).stdout
    for l in out.splitlines():
        if 'steps/epoch' in l:
            n=int(l.split('epochs')[0].split(',')[-1]); TOT=n*499
except Exception: pass
TOT=TOT or 12475
ms=(t1-t0)/max(1,s1-s0)*1000; left=(TOT-s1)*ms/1000
stale=time.time()-t1
print(f"進度   第 {s1:,} 步 / {TOT:,}   ({s1/TOT*100:.1f}%)   第 {ep} 輪")
print(f"速度   {ms:.0f} ms/step   {ms*499/60000:.1f} 分/輪")
print(f"已跑   {(t1-t0)/60:.0f} 分     剩 {left/60:.0f} 分   預計 {time.strftime('%H:%M', time.localtime(t1+left))} 完成")
print(f"loss   {run[0]['loss']:.3f} -> {run[-1]['loss']:.3f}   grad {run[-1]['grad_norm']:.3f}   tau {run[-1]['tau']}")
print(f"最後寫入 {time.strftime('%H:%M:%S', time.localtime(t1))}"
      + ("   ⚠️ 超過 60 秒沒動靜，可能已經死了" if stale>60 else "   ✅ 正常"))
dv=[json.loads(l) for l in open(L+'train_stage1_dev_val.jsonl') if l.strip()]
mine=[r for r in dv if r.get('run_id')==rid and r.get('n_gallery')==4569]
if mine:
    print("每輪分數  " + "  ".join(f"{r['epoch']+1}:{r['mean_R@1']:.4f}" for r in mine))
PY
echo "----"
nvidia-smi --query-gpu=power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader | sed 's/^/GPU   /'
uptime | sed 's/^/開機  /'
