"""The research-authority guard, tested from outside.

`.claude/hooks/research_authority_guard.py` is gitignored, so the guard itself
has no version history and no diff to review. This file is tracked, so what the
guard is SUPPOSED to do lives in git even though the guard does not.

The Bash cases exist because of a specific failure on 2026-08-30: the guard
watched Write/Edit/NotebookEdit only, and `.claude/settings.json` was edited
that night by a python heredoc inside a Bash call. The guard did not fire, did
not warn, and left no trace. Nobody was working around it -- nobody knew the
door was there. `test_bash_heredoc_write_to_settings_is_blocked` is that exact
command, kept as a regression.

Point the tests at a candidate guard before installing it:

    METAFIND_GUARD=/path/to/candidate.py pytest tests/test_research_authority_guard.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD = Path(os.environ.get("METAFIND_GUARD",
                            REPO / ".claude/hooks/research_authority_guard.py"))

BLOCK = 2
ALLOW = 0

pytestmark = pytest.mark.skipif(
    not GUARD.exists(),
    reason=f"guard not present at {GUARD} (it is gitignored; see module docstring)")


def run(payload, allow_override=False):
    env = dict(os.environ)
    env.pop("METAFIND_ALLOW_AUTHORITY_EDIT", None)
    # A candidate guard run from outside .claude/hooks/ would resolve REPO to
    # the wrong directory and allow everything. Pin it, or the suite passes a
    # guard that guards nothing.
    env["METAFIND_GUARD_REPO"] = str(REPO)
    if allow_override:
        env["METAFIND_ALLOW_AUTHORITY_EDIT"] = "1"
    proc = subprocess.run([sys.executable, str(GUARD)],
                          input=json.dumps(payload), capture_output=True,
                          text=True, env=env, timeout=30)
    return proc.returncode


def write_call(path):
    return {"cwd": str(REPO), "hook_event_name": "PreToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": path, "content": "x"}}


def bash_call(command):
    return {"cwd": str(REPO), "hook_event_name": "PreToolUse",
            "tool_name": "Bash", "tool_input": {"command": command}}


# --------------------------------------------------------------- Write/Edit

@pytest.mark.parametrize("path", [
    "CLAUDE.md",
    ".claude/CLAUDE.md",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/rules/research-rigor.md",
    ".claude/hooks/research_authority_guard.py",
    ".claude/skills/paper-audit/SKILL.md",
    "metafind/vendor/egnn_clean.py",
    "docs/paper/metafind_source/2methdology.tex",
])
def test_write_to_protected_path_is_blocked(path):
    assert run(write_call(path)) == BLOCK


@pytest.mark.parametrize("path", [
    "tools/measure_dtype_effect.py",
    "metafind/train/stage1.py",
    "workflow/DECISION_LEDGER.md",
    "docs/METAFIND_NOTEBOOK.md",
    # not a substring match: these three have tripped naive matching before
    "metafind/compat/shim.py",
    "docs/CLAUDE_NOTES.md",
    "docs/paper/metafind_source/SOURCE_MANIFEST.json",
])
def test_write_to_ordinary_path_is_allowed(path):
    assert run(write_call(path)) == ALLOW


def test_override_lets_a_protected_write_through():
    assert run(write_call(".claude/settings.json"), allow_override=True) == ALLOW


def test_missing_path_fails_closed():
    assert run({"cwd": str(REPO), "tool_name": "Write", "tool_input": {}}) == BLOCK


# --------------------------------------------------------------------- Bash

def test_bash_heredoc_write_to_settings_is_blocked():
    """The exact shape that went through unseen on 2026-08-30."""
    command = (
        "cd /home/kyzen/MetaFindV1 && python3 - <<'PY'\n"
        "import json, pathlib\n"
        "p = pathlib.Path('.claude/settings.json')\n"
        "d = json.loads(p.read_text())\n"
        "d['env']['X'] = '1'\n"
        "p.write_text(json.dumps(d))\n"
        "PY"
    )
    assert run(bash_call(command)) == BLOCK


@pytest.mark.parametrize("command", [
    "echo x > .claude/settings.json",
    "cp /tmp/x .claude/hooks/research_authority_guard.py",
    "rm .claude/rules/research-rigor.md",
    "sed -i s/a/b/ CLAUDE.md",
    "mv /tmp/x metafind/vendor/egnn_clean.py",
    "tee CLAUDE.md < /tmp/x",
    "git checkout -- .claude/settings.json",
])
def test_bash_write_to_protected_path_is_blocked(command):
    assert run(bash_call(command)) == BLOCK


@pytest.mark.parametrize("command", [
    # reading a protected file must stay free, or the guard becomes noise
    "cat CLAUDE.md",
    "head -20 .claude/rules/experiments.md",
    "grep -n foo .claude/settings.json",
    "git log --oneline -3",
    "ls -la .claude/hooks/",
    # ordinary work that happens to use a write marker
    "echo hi > /tmp/scratch.txt",
    "rm -f workflow/tmp_note.md",
    "python3 -c 'import json; json.dump({}, open(\"/tmp/a.json\", \"w\"))'",
])
def test_bash_allowed(command):
    assert run(bash_call(command)) == ALLOW


def test_prose_that_quotes_a_write_command_is_not_a_write():
    """A ledger entry describing the guard must remain writable.

    The first commit after the Bash check shipped was blocked by it: the ledger
    text quoted `cp ... .claude/rules/...` as prose inside a heredoc, and the
    guard read the whole command as one string. A guard that stops you writing
    about the guard is noise, and noise is what gets switched off.
    """
    command = (
        "cd /home/kyzen/MetaFindV1\n"
        "cat >> workflow/DECISION_LEDGER.md <<'EOF'\n"
        "The fix was verified live: `cp /tmp/nope .claude/rules/research-rigor.md`\n"
        "is now blocked, and `rm .claude/settings.json` would be too.\n"
        "EOF"
    )
    assert run(bash_call(command)) == ALLOW


def test_python_heredoc_that_really_writes_is_still_blocked():
    """The exemption above must not reopen the hole it sits next to."""
    command = (
        "cd /home/kyzen/MetaFindV1\n"
        "python3 - <<'PY'\n"
        "import pathlib\n"
        "pathlib.Path('.claude/rules/research-rigor.md').write_text('gone')\n"
        "PY"
    )
    assert run(bash_call(command)) == BLOCK


def test_bash_override_lets_it_through():
    assert run(bash_call("echo x > .claude/settings.json"),
               allow_override=True) == ALLOW


def test_bash_with_no_command_is_not_treated_as_a_write():
    assert run({"cwd": str(REPO), "tool_name": "Bash", "tool_input": {}}) == ALLOW


# ------------------------------------------------------- the stated ceiling

def test_the_heuristic_is_documented_as_evadable():
    """Not a behaviour test. It pins the honesty of the docstring.

    The Bash check is a heuristic. If someone later deletes the paragraph that
    says so, this fails and they have to decide deliberately whether the claim
    changed or only the wording did.
    """
    text = GUARD.read_text()
    assert "HEURISTIC" in text
    assert "not a sandbox" in text
