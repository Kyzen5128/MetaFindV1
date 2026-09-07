"""Read-only Linux process and corpus inventory; no models or GPU calls.

File presence and matching annotation contracts do not establish completion,
current image inputs, validation gates, or permission to start another job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def invocation(argv: list[str]) -> dict | None:
    """Recognise actual Python -m jobs / shell chain scripts, never mentions."""
    if not argv:
        return None
    executable = Path(argv[0]).name
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
        for i, arg in enumerate(argv[1:], 1):
            if arg == "-m" and i + 1 < len(argv):
                module = argv[i + 1]
                if module.startswith("metafind."):
                    return {"kind": "module", "target": module, "arguments": argv[i + 2:]}
                return None
            if arg not in ("-u", "-B", "-E", "-I", "-s", "-S", "-O", "-OO", "-q"):
                return None
    elif executable in ("bash", "sh"):
        for i, arg in enumerate(argv[1:], 1):
            if arg in ("-e", "-u", "-x", "--"):
                continue
            if not arg.startswith("-") and re.fullmatch(r"chain_[\w.-]+\.sh", Path(arg).name):
                return {"kind": "chain", "target": arg, "arguments": argv[i + 1:]}
            return None
    return None


def option(arguments: list[str], name: str, default=None):
    value = default
    for i, arg in enumerate(arguments):
        if arg == name:
            value = arguments[i + 1] if i + 1 < len(arguments) else None
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def processes(repo: Path, proc_root: Path = Path("/proc")) -> dict:
    """Only disclose selected metadata for jobs whose cwd is this checkout.

    cwd is observable, Python import origin is not. An inherited METAFIND_REPO
    does not establish module ownership and is not used by metafind.paths.
    """
    jobs, unavailable = [], 0
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        return {"jobs": [], "error": str(exc), "unavailable": 0}
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            argv = [a for a in (entry / "cmdline").read_bytes().decode().split("\0") if a]
            job = invocation(argv)
            if job is None:
                continue
            cwd = (entry / "cwd").resolve(strict=True)
            env = dict(s.split("=", 1) for s in (entry / "environ").read_bytes().decode().split("\0") if "=" in s)
            if cwd != repo.resolve():
                continue
            # Linux exposes the initial environment. A shell may later set its
            # corpus; do not infer future commands or resources from its name.
            data = env.get("METAFIND_DATA")
            if data:
                path = Path(data)
                data = str((path if path.is_absolute() else cwd / path).resolve())
            elif job["kind"] == "module":
                data = str((repo / "data").resolve())
            try:
                stdout = os.readlink(entry / "fd" / "1")
            except OSError:
                stdout = None
            metadata = {}
            if job["target"] == "metafind.data.annotate_run":
                mode = option(job["arguments"], "--prompt-mode", "v9")
                metadata = {"prompt_mode": mode if mode in ("v9", "figure2_v10") else None,
                            "writes_annotation_arm": any(a == "--arm" or a.startswith("--arm=")
                                                         for a in job["arguments"])}
            # Do not disclose arbitrary command arguments (e.g. model URLs).
            jobs.append({"pid": int(entry.name), "kind": job["kind"], "target": job["target"],
                         **metadata, "cwd": str(cwd), "data": data, "stdout": stdout})
        except (OSError, UnicodeError, ValueError):
            # Vanishing / unreadable /proc entries are incomplete observations,
            # not evidence that a job finished successfully or needs a restart.
            unavailable += 1
    return {"jobs": sorted(jobs, key=lambda j: j["pid"]), "error": None,
            "unavailable": unavailable}


def records(directory: Path) -> dict:
    try:
        if not directory.is_dir():
            return {"path": str(directory), "count": None, "error": "directory missing"}
        count = sum(p.is_file() and p.suffix == ".json" for p in directory.iterdir())
        return {"path": str(directory), "count": count, "error": None}
    except OSError as exc:
        return {"path": str(directory), "count": None, "error": str(exc)}


def annotations(directory: Path, mode: str | None) -> dict:
    result = records(directory)
    result.update(contracts={}, invalid_records=0, prompt_mode=mode,
                  expected_contract=None, matching_contract_records=None,
                  scope="Contract labels only; not producer completion or image-identity validation.")
    if result["error"]:
        return result
    if mode == "figure2_v10":
        from metafind.data.annotate_v10 import contract_id
        result["expected_contract"] = contract_id()
    elif mode == "v9":
        from metafind.data.annotate import annotation_contract_id
        result["expected_contract"] = annotation_contract_id()
    counts = Counter()
    for path in directory.glob("*.json"):
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict):
                raise ValueError("annotation record must be an object")
            contract = record.get("annotation_contract")
            counts[contract if isinstance(contract, str) else "<missing>"] += 1
        except (OSError, ValueError, UnicodeError):
            result["invalid_records"] += 1
    result["contracts"] = dict(sorted(counts.items()))
    if result["expected_contract"]:
        result["matching_contract_records"] = counts[result["expected_contract"]]
    return result


def render_index(path: Path) -> dict:
    result = {"path": str(path), "unique_uids": None, "duplicate_rows": 0, "error": None}
    try:
        seen = set()
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                uid = json.loads(line)["uid"]
                if not isinstance(uid, str) or not uid:
                    raise ValueError("render index UID must be a nonempty string")
                result["duplicate_rows"] += int(uid in seen)
                seen.add(uid)
        result["unique_uids"] = len(seen)
    except (OSError, ValueError, UnicodeError, KeyError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def snapshot(data: Path, mode: str | None = None, *, proc_root=Path("/proc")) -> dict:
    data = data.resolve()
    live = processes(REPO, proc_root)
    matching = [j for j in live["jobs"] if j["target"] == "metafind.data.annotate_run"
                and j["data"] == str(data) and not j["writes_annotation_arm"]]
    modes = {j["prompt_mode"] for j in matching}
    if mode is None and len(modes) == 1:
        mode = modes.pop()
    output = data / "outputs"
    checkpoint_records = sorted(str(p) for p in (output / "checkpoints").rglob("*.json")
                                if p.name in ("stage1_best_ckpt.json", "variant_ckpts.json"))
    if (output / "variant_ckpts.json").is_file():
        checkpoint_records.append(str(output / "variant_ckpts.json"))
    return {"schema": "metafind.status.v1", "observed_at": datetime.now(timezone.utc).isoformat(),
            "repo": str(REPO), "data": str(data), "processes": live,
            "annotations": annotations(output / "annotations", mode),
            "render_index": render_index(output / "logs" / "renders_index.jsonl"),
            "record_inventory": {n: records(output / n) for n in
                                 ("pointclouds", "renders", "embeddings", "scene_graphs", "procthor_modalities")},
            "semantic_files_present": {n: (output / n).is_file() for n in
                                       ("sem_edge_cache.json", "sem_edge_embeddings.npz", "procthor_node_embeddings.json", "procthor_node_embeddings.npz")},
            "checkpoint_records_present": checkpoint_records,
            "scope": "Read-only, non-atomic inventory. Presence does not establish completion, current inputs, gate success, or a runnable queue."}


def display(report: dict) -> str:
    lines = [f"觀測時間 {report['observed_at']}", f"資料根目錄 {report['data']}", "", "目前程序（cwd 為本 checkout）："]
    for job in report["processes"]["jobs"]:
        label = "等候／執行鏈" if job["kind"] == "chain" else "執行程序"
        lines.append(f"  PID {job['pid']} {label} {job['target']}")
        lines.append(f"    corpus: {job['data'] or 'UNKNOWN（初始環境未提供）'}")
        if job["stdout"]:
            lines.append(f"    stdout: {job['stdout']}")
    if not report["processes"]["jobs"]:
        lines.append("  此次未觀測到；不代表其他 checkout 沒有工作。")
    if report["processes"]["error"] or report["processes"]["unavailable"]:
        lines.append(f"  無法讀取的程序數: {report['processes']['unavailable']}；error: {report['processes']['error']}")
    ann = report["annotations"]
    lines.extend(["", "檔案盤點（不代表完成或驗證通過）：",
                  f"  annotation JSON: {ann['count']}；解析失敗: {ann['invalid_records']}；error: {ann['error']}",
                  f"  contract 分布: {ann['contracts']}"])
    if ann["expected_contract"]:
        lines.append(f"  {ann['prompt_mode']} contract 相符: {ann['matching_contract_records']}（只核對契約標記）")
    idx = report["render_index"]
    lines.append(f"  render index UID: {idx['unique_uids']}；重複列: {idx['duplicate_rows']}；error: {idx['error']}")
    for name, item in report["record_inventory"].items():
        lines.append(f"  {name}: {item['count']}" + (f" ({item['error']})" if item["error"] else ""))
    for name, present in report["semantic_files_present"].items():
        lines.append(f"  {name}: {'存在，未驗證' if present else '未找到'}")
    lines.append(f"  checkpoint records: {len(report['checkpoint_records_present'])}（未載入權重）")
    lines.extend(f"    {p}" for p in report["checkpoint_records_present"])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("METAFIND_DATA", REPO / "data")))
    parser.add_argument("--prompt-mode", choices=("v9", "figure2_v10"),
                        help="Otherwise infer only from live annotation on this corpus.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = snapshot(args.data, args.prompt_mode)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else display(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
