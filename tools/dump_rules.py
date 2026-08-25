#!/usr/bin/env python3
"""Collect every live rule file into one copyable snapshot.

The five rule files under `.claude/rules/` are excluded from version control by
`.gitignore:56`, along with the rest of `.claude/`. They are loaded into every
Claude Code session, so they govern the work, but nothing in the repo records
what they said at a given commit. This script produces that record.

Run it after editing any rule file:

    python3 tools/dump_rules.py

Output: docs/RULES_SNAPSHOT.md, which carries a sha256 prefix per source file so
a reader can tell whether the snapshot still matches the live rules.
"""
from __future__ import annotations

import hashlib
import io
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "RULES_SNAPSHOT.md"

SOURCES = [
    ("專案指令", "CLAUDE.md"),
    ("規則 1：研究嚴謹度", ".claude/rules/research-rigor.md"),
    ("規則 2：論文復現", ".claude/rules/paper-reproduction.md"),
    ("規則 3：實驗", ".claude/rules/experiments.md"),
    ("規則 4：程式碼變更", ".claude/rules/code-changes.md"),
    ("規則 5：上游查找", ".claude/rules/upstream-lookup.md"),
]

# Prose that is not mechanically derivable from the source files. Regenerating
# the snapshot must not silently drop it, so it lives here rather than being
# hand-pasted into the output each time.
PREAMBLE = ROOT / "docs" / "_rules_preamble.md"


def head_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def main() -> int:
    missing = [p for _, p in SOURCES if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"rule files missing, refusing to write a partial snapshot: {missing}")

    parts: list[str] = []
    parts.append("# MetaFindV1 規則快照\n\n")
    parts.append(f"repo commit `{head_sha()}`　·　由 `tools/dump_rules.py` 產生\n\n")
    parts.append("**不要直接編輯本檔。** 要改請改下列來源檔，再重新執行該腳本。\n\n---\n\n")

    parts.append("## 生效中的規則檔\n\n")
    parts.append("| # | 名稱 | 來源路徑 | 行數 | sha256(前16) |\n|---|---|---|---|---|\n")
    for i, (name, path) in enumerate(SOURCES, 1):
        raw = (ROOT / path).read_bytes()
        lines = raw.decode("utf-8").count("\n") + 1
        parts.append(f"| {i} | {name} | `{path}` | {lines} | "
                     f"`{hashlib.sha256(raw).hexdigest()[:16]}` |\n")
    parts.append(
        "\n`.claude/` 依 `.gitignore:56` 不進版控，五個規則檔都是本機檔案，"
        "每次 session 由 Claude Code 自動載入。本快照是它們唯一的可攜副本。\n"
    )

    if PREAMBLE.exists():
        parts.append("\n---\n\n")
        parts.append(PREAMBLE.read_text(encoding="utf-8"))

    for name, path in SOURCES:
        parts.append(f"\n\n{'=' * 78}\n## {name}\n\n來源：`{path}`\n\n{'=' * 78}\n\n")
        parts.append((ROOT / path).read_text(encoding="utf-8"))
        parts.append("\n")

    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} from {len(SOURCES)} rule files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
