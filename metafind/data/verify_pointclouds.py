"""Recompute point-cloud completeness from the sidecars, not from the summary.

Graph node: the evidence L2-COMPLETE and gate G2 rely on.

``runs/progress/pointclouds.json`` is a derived summary written by the fetcher.
The per-item sidecars are the durable record, so completeness is recomputed from
them here. That separation matters: a bug in the summariser (there was one --
resumed shards were counted twice) must not be able to make a genuine shortfall
invisible.

Checks, all assertions about content rather than existence:

* every uid in ``lvis.json`` is accounted for exactly once
* ``admitted + quarantined == len(lvis.json)``
* every admitted uid has a file on disk whose sha256 matches its sidecar
* every quarantined uid carries a real exception type and message

Usage::

    python -m metafind.data.verify_pointclouds
    python -m metafind.data.verify_pointclouds --deep    # rehash every file
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
LVIS_JSON = DATA / "sources/objaverse-lvis/ULIP_Objaverse_Triplets/lvis.json"
OUT_DIR = DATA / "artifacts/pointclouds"
SIDECAR_DIR = DATA / "runs/sidecars/pointclouds"


def load_sidecars() -> tuple[dict[str, dict], list[str]]:
    """Return {uid: last record} and any malformed lines encountered."""
    records: dict[str, dict] = {}
    problems: list[str] = []
    for path in sorted(SIDECAR_DIR.glob("*.jsonl")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line is expected after a kill; anything else is not.
                problems.append(f"{path.name}:{n} unparseable")
                continue
            uid = rec.get("uid")
            if uid is None:
                problems.append(f"{path.name}:{n} has no uid")
                continue
            records[uid] = rec  # later record wins; a retry supersedes its failure
    return records, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", action="store_true", help="rehash every admitted file")
    args = ap.parse_args()

    lvis = json.loads(LVIS_JSON.read_text())
    wanted = set(lvis)
    records, problems = load_sidecars()

    admitted = {u for u, r in records.items() if r.get("status") == "admitted"}
    quarantined = {u for u, r in records.items() if r.get("status") == "quarantined"}
    unseen = wanted - admitted - quarantined
    unexpected = (admitted | quarantined) - wanted

    on_disk = {p.stem for p in OUT_DIR.glob("*.npy")} if OUT_DIR.exists() else set()
    missing_files = admitted - on_disk
    orphan_files = on_disk - admitted

    print("point-cloud completeness (recomputed from sidecars)")
    print("=" * 62)
    print(f"  lvis.json 參照      : {len(wanted)}")
    print(f"  admitted            : {len(admitted)}")
    print(f"  quarantined         : {len(quarantined)}")
    print(f"  尚未處理            : {len(unseen)}")
    print(f"  磁碟上的 .npy       : {len(on_disk)}")

    failures: list[str] = []
    if unexpected:
        failures.append(f"{len(unexpected)} uid 不在 lvis.json 內: {sorted(unexpected)[:5]}")
    if missing_files:
        failures.append(f"{len(missing_files)} 個 admitted 沒有對應檔案: {sorted(missing_files)[:5]}")
    if orphan_files:
        failures.append(f"{len(orphan_files)} 個檔案沒有 admitted 紀錄: {sorted(orphan_files)[:5]}")

    # A quarantine record without a real reason is as bad as no record: the point
    # of the sidecar is that a shortfall stays diagnosable months later.
    reasonless = [
        u for u in quarantined
        if not records[u].get("exception_type") or not records[u].get("exception_msg")
    ]
    if reasonless:
        failures.append(f"{len(reasonless)} 筆 quarantine 沒有真實原因: {reasonless[:5]}")

    if args.deep and admitted:
        print("\n  重算 sha256 ...")
        mismatched = []
        for i, uid in enumerate(sorted(admitted), 1):
            p = OUT_DIR / f"{uid}.npy"
            if not p.exists():
                continue
            if hashlib.sha256(p.read_bytes()).hexdigest() != records[uid].get("sha256"):
                mismatched.append(uid)
            if i % 5000 == 0:
                print(f"    {i}/{len(admitted)}", flush=True)
        print(f"    完成，{len(mismatched)} 個不符")
        if mismatched:
            failures.append(f"{len(mismatched)} 個檔案內容與 sidecar 的 sha256 不符: {mismatched[:5]}")

    if problems:
        print(f"\n  sidecar 格式問題 {len(problems)} 筆（收尾被中斷時各檔最多 1 筆屬正常）:")
        for p in problems[:5]:
            print(f"    {p}")
        if len(problems) > len(list(SIDECAR_DIR.glob('*.jsonl'))):
            failures.append(f"{len(problems)} 筆 sidecar 無法解析，超過中斷可解釋的數量")

    print()
    if unseen:
        print(f"  尚未完成 —— 還有 {len(unseen)} 個未處理，抓取仍在進行中")
        by_shard = collections.Counter(str(lvis[u]).split("/")[0] for u in unseen)
        print(f"  分布於 {len(by_shard)} 個 shard")
        return 3  # BLOCKED_EVIDENCE: incomplete, not wrong

    if failures:
        print("  FAIL")
        for f in failures:
            print(f"    - {f}")
        return 2

    print(f"  PASS  admitted + quarantined = {len(admitted)} + {len(quarantined)} = {len(wanted)}")
    print(f"        quarantine 率 = {len(quarantined) / len(wanted) * 100:.3f}%  (G2 上限 2%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
