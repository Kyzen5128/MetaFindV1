"""Inspect/apply the cumulative I-Design compatibility chain using pinned bytes.

Only exact prefixes are accepted for files touched by these patches. Other
checkout files are neither compared nor modified. No upstream code is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

PINNED_COMMIT = "7bc891c72e45f36f6461f5848c3c052723faede4"


class PatchStateError(ValueError):
    """The checkout is not a known cumulative patch state."""


def _git(repo, *args, data=None):
    result = subprocess.run(["git", "-C", str(repo), *args], input=data,
                            capture_output=True)
    if result.returncode:
        raise PatchStateError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _patch_paths(raw):
    # These shipped patches modify existing ordinary files; reject a changed
    # patch format instead of guessing how additions/renames should be checked.
    old, new = [], []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("--- "):
            if not line.startswith("--- a/"):
                raise PatchStateError("patch must modify existing files")
            old.append(line[6:])
        elif line.startswith("+++ "):
            if not line.startswith("+++ b/"):
                raise PatchStateError("patch must modify existing files")
            new.append(line[6:])
    if not old or old != new:
        raise PatchStateError("patch additions/deletions/renames are unsupported")
    for name in old:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
            raise PatchStateError(f"unsafe patch path: {name}")
    return set(old)


def _read_files(repo, names):
    contents = {}
    for name in names:
        path = repo / name
        if any(p.is_symlink() for p in (path, *path.parents) if p != repo.parent):
            raise PatchStateError(f"patched source cannot be a symlink: {name}")
        if not path.is_file():
            raise PatchStateError(f"patched source is missing: {name}")
        contents[name] = path.read_bytes()
    return contents


def _state(repo, patch_dir):
    repo, patch_dir = Path(repo).resolve(), Path(patch_dir).resolve()
    if _git(repo, "rev-parse", "HEAD").decode().strip() != PINNED_COMMIT:
        raise PatchStateError(f"I-Design HEAD must be pinned at {PINNED_COMMIT}")
    patches = [(p, p.read_bytes()) for p in sorted(patch_dir.glob("idesign-*.patch"))]
    if not patches:
        raise PatchStateError("no I-Design compatibility patches found")
    names = set().union(*(_patch_paths(raw) for _, raw in patches))
    actual = _read_files(repo, names)
    expected = {name: _git(repo, "show", f"{PINNED_COMMIT}:{name}") for name in names}
    matches = [0] if actual == expected else []
    with tempfile.TemporaryDirectory(prefix="metafind_idesign_chain_") as directory:
        scratch = Path(directory)
        for name, raw in expected.items():
            (scratch / name).parent.mkdir(parents=True, exist_ok=True)
            (scratch / name).write_bytes(raw)
        for count, (_, raw) in enumerate(patches, 1):
            _git(scratch, "apply", "-", data=raw)
            if actual == _read_files(scratch, names):
                matches.append(count)
        final = _read_files(scratch, names)
    if len(matches) != 1:
        raise PatchStateError(
            "patched source files do not match a unique pinned patch prefix; "
            "refusing mixed/conflicting/unknown modifications (no files changed)"
        )
    prefix = matches[0]
    report = {
        "pinned_commit": PINNED_COMMIT, "applied_prefix": prefix,
        "status": "complete" if prefix == len(patches) else "partial" if prefix else "pristine",
        "patches": [{"name": p.stem, "applied": i < prefix,
                     "sha256": hashlib.sha256(raw).hexdigest()}
                    for i, (p, raw) in enumerate(patches)],
    }
    return repo, patches, actual, report, final


def inspect_chain(repo: Path, patch_dir: Path) -> dict:
    """Read-only check, including patches superseded by later chain entries."""
    return _state(repo, patch_dir)[3]


def apply_chain(repo: Path, patch_dir: Path) -> dict:
    """Complete an exact known prefix; preserve all unrelated checkout files."""
    repo, patches, before, report, final = _state(repo, patch_dir)
    remaining = patches[report["applied_prefix"]:]
    if not remaining:
        return report
    # Several patches may edit the same lines. Applying them in one `git apply`
    # does not reliably compose them, so produce one native Git diff between
    # the inspected prefix and the sequentially built final snapshot.
    with tempfile.TemporaryDirectory(prefix="metafind_idesign_apply_") as directory:
        scratch = Path(directory)
        for label, snapshot in (("before", before), ("after", final)):
            for name, raw in snapshot.items():
                path = scratch / label / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        diff = subprocess.run(["git", "-C", str(scratch), "diff", "--no-index",
                               "--binary", "--", "before", "after"], capture_output=True)
        if diff.returncode != 1 or not diff.stdout:
            raise PatchStateError("could not build a nonempty cumulative patch delta")
        _git(repo, "apply", "-p2", "--check", "-", data=diff.stdout)
        if _read_files(repo, before) != before:
            raise PatchStateError("I-Design source changed during patch preflight")
        _git(repo, "apply", "-p2", "-", data=diff.stdout)
    if _read_files(repo, final) != final:
        raise PatchStateError("I-Design source differs from the final patch snapshot")
    # Provenance describes the exact frozen patch bytes used above, even if
    # another process later replaces a .patch file under the same filename.
    report.update(status="complete", applied_prefix=len(patches))
    for patch in report["patches"]:
        patch["applied"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "apply"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = (inspect_chain if args.action == "inspect" else apply_chain)(args.repo, args.patch_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"I-Design patch chain refused: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
