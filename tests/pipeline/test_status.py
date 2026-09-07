"""Status must observe real jobs and distinguish presence from completion."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from tools import status


def fake_process(tmp_path, pid, argv, *, data=None, repo=None):
    repo = repo or status.REPO
    directory = tmp_path / str(pid)
    directory.mkdir()
    (directory / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
    env = {"METAFIND_REPO": str(repo), "DO_NOT_DISCLOSE_SECRET": "private-test-value"}
    if data is not None:
        env["METAFIND_DATA"] = str(data)
    (directory / "environ").write_bytes(("\0".join(f"{k}={v}" for k, v in env.items()) + "\0").encode())
    (directory / "cwd").symlink_to(repo, target_is_directory=True)
    return directory


@pytest.mark.parametrize("argv", [
    ["bash", "-c", "python -m metafind.data.annotate_run"],
    ["python", "-c", "print('hi')", "-m", "metafind.train.stage1"],
    ["vim", "chain_paper_stage1_20260906.sh"],
    ["rg", "metafind.data.annotate_run"],
    ["python", "notes.py", "-m", "metafind.train.stage1"],
])
def test_mentions_are_not_live_jobs(argv):
    assert status.invocation(argv) is None


def test_actual_jobs_are_scoped_by_checkout_and_keep_each_corpus(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    a, b = tmp_path / "data a", tmp_path / "data b"
    fake_process(tmp_path, 10, ["/usr/bin/python3.11", "-B", "-m", "metafind.data.annotate_run", "--prompt-mode", "figure2_v10"], data=a)
    fake_process(tmp_path, 11, ["/bin/bash", "/logs/chain_paper_stage1_20260906.sh"], data=a)
    fake_process(tmp_path, 12, ["bash", "/logs/chain_paper_stage2_20260906.sh"], data=b)
    fake_process(tmp_path, 13, ["python", "-m", "metafind.train.stage1"], data=b, repo=other)
    result = status.processes(status.REPO, tmp_path)
    assert [j["pid"] for j in result["jobs"]] == [10, 11, 12]
    assert [j["data"] for j in result["jobs"]] == [str(a), str(a), str(b)]
    assert "private-test-value" not in json.dumps(result)


def test_unreadable_live_process_does_not_claim_completion(tmp_path):
    fake = fake_process(tmp_path, 21, ["python", "-m", "metafind.train.stage2"])
    (fake / "environ").unlink()
    result = status.processes(status.REPO, tmp_path)
    assert result["jobs"] == []
    assert result["unavailable"] == 1
    assert status.processes(status.REPO, tmp_path / "missing")["error"]


def test_inherited_repo_variable_does_not_relabel_another_checkout(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    fake = fake_process(tmp_path, 22, ["python", "-m", "metafind.train.stage2"], repo=other)
    (fake / "environ").write_bytes(f"METAFIND_REPO={status.REPO}\0".encode())
    assert status.processes(status.REPO, tmp_path)["jobs"] == []


def test_last_prompt_option_matches_argparse_and_other_arguments_are_private(tmp_path):
    args = ["--prompt-mode", "v9", "--prompt-mode=figure2_v10", "--model", "https://user:private-test-value@example.invalid/model"]
    fake_process(tmp_path, 23, ["python", "-m", "metafind.data.annotate_run", *args])
    result = status.processes(status.REPO, tmp_path)
    assert result["jobs"][0]["prompt_mode"] == "figure2_v10"
    assert "private-test-value" not in json.dumps(result)
    assert "arguments" not in result["jobs"][0]


def test_infer_live_v10_only_on_the_selected_corpus(tmp_path):
    from metafind.data.annotate import annotation_contract_id
    from metafind.data.annotate_v10 import contract_id

    proc, data = tmp_path / "proc", tmp_path / "data"
    proc.mkdir()
    ann = data / "outputs" / "annotations"
    ann.mkdir(parents=True)
    (ann / "v10.json").write_text(json.dumps({"annotation_contract": contract_id()}))
    (ann / "v9.json").write_text(json.dumps({"annotation_contract": annotation_contract_id()}))
    (ann / "partial.json").write_text('{"annotation_contract":')
    fake_process(proc, 30, ["python", "-m", "metafind.data.annotate_run", "--prompt-mode=figure2_v10"], data=data)
    fake_process(proc, 31, ["python", "-m", "metafind.data.annotate_run"], data=tmp_path / "other-data")
    result = status.snapshot(data, proc_root=proc)
    assert result["annotations"]["prompt_mode"] == "figure2_v10"
    assert result["annotations"]["matching_contract_records"] == 1
    assert result["annotations"]["invalid_records"] == 1
    assert "not producer completion" in result["annotations"]["scope"]
    assert status.snapshot(data, "v9", proc_root=proc)["annotations"]["expected_contract"] == annotation_contract_id()


def test_bakeoff_or_conflicting_modes_do_not_select_a_corpus_contract(tmp_path):
    proc, data = tmp_path / "proc", tmp_path / "data"
    proc.mkdir()
    (data / "outputs" / "annotations").mkdir(parents=True)
    fake_process(proc, 40, ["python", "-m", "metafind.data.annotate_run", "--prompt-mode", "figure2_v10", "--arm", "trial"], data=data)
    assert status.snapshot(data, proc_root=proc)["annotations"]["prompt_mode"] is None
    fake_process(proc, 41, ["python", "-m", "metafind.data.annotate_run", "--prompt-mode", "figure2_v10"], data=data)
    fake_process(proc, 42, ["python", "-m", "metafind.data.annotate_run", "--prompt-mode", "v9"], data=data)
    assert status.snapshot(data, proc_root=proc)["annotations"]["prompt_mode"] is None


def test_inventory_follows_directory_links_but_counts_only_json_files(tmp_path):
    target = tmp_path / "actual"
    target.mkdir()
    (target / "a.json").write_text("{}")
    (target / "a.npz").write_bytes(b"array")
    (target / "directory.json").mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    assert status.records(link)["count"] == 1
    assert status.records(tmp_path / "missing")["count"] is None


def test_render_uid_count_is_unique_and_malformed_index_is_unknown(tmp_path):
    path = tmp_path / "renders_index.jsonl"
    path.write_text('{"uid":"A"}\n\n{"uid":"A"}\n{"uid":"B"}\n')
    assert status.render_index(path)["unique_uids"] == 2
    assert status.render_index(path)["duplicate_rows"] == 1
    path.write_text('{"uid":"A"}\n{"uid":')
    result = status.render_index(path)
    assert result["unique_uids"] is None and result["error"]


def test_wrapper_from_another_cwd_preserves_paths_and_does_not_write_corpus(tmp_path):
    data = tmp_path / "corpus with spaces"
    output = data / "outputs"
    output.mkdir(parents=True)
    (output / "sem_edge_cache.json").write_text('{}')
    ck = output / "checkpoints" / "run"
    ck.mkdir(parents=True)
    (ck / "stage1_best_ckpt.json").write_text('{}')
    before = {str(p.relative_to(data)): (p.stat().st_mtime_ns, p.read_bytes()) for p in data.rglob('*') if p.is_file()}
    result = subprocess.run(["bash", str(status.REPO / "tools/status.sh"), "--data", str(data), "--json"],
                            cwd=tmp_path, env={**os.environ, "METAFIND_PYTHON": os.sys.executable},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["data"] == str(data)
    assert record["semantic_files_present"]["sem_edge_cache.json"] is True
    assert record["semantic_files_present"]["sem_edge_embeddings.npz"] is False
    assert record["checkpoint_records_present"] == [str(ck / "stage1_best_ckpt.json")]
    assert "存在，未驗證" in status.display(record)
    assert {str(p.relative_to(data)): (p.stat().st_mtime_ns, p.read_bytes()) for p in data.rglob('*') if p.is_file()} == before
