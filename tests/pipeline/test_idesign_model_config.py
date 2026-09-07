"""Apply shipped patches to pinned upstream bytes; import actual config modules.

Only AutoGen's file filter/type definitions are stubbed. No agent is created,
endpoint contacted, model loaded, or real upstream checkout modified.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
PINNED = "7bc891c72e45f36f6461f5848c3c052723faede4"
FILES = ("agents.py", "corrector_agents.py", "refiner_agents.py", "schemas.py", "IDesign.py", "utils.py")


@pytest.fixture
def checkout(tmp_path):
    upstream = Path(os.environ.get("METAFIND_IDESIGN_TEST_REPO", "/home/kyzen/upstream/IDesign"))
    if not (upstream / ".git").exists():
        pytest.skip("pinned I-Design checkout unavailable; no download")
    original = {}
    for name in FILES:
        proc = subprocess.run(["git", "-C", str(upstream), "show", f"{PINNED}:{name}"], capture_output=True)
        if proc.returncode:
            pytest.skip("pinned I-Design source object unavailable locally; no download")
        original[name] = proc.stdout
        (tmp_path / name).write_bytes(proc.stdout)
    patches = sorted((REPO / "setup/patches").glob("idesign-*.patch"))
    assert [p.name[:10] for p in patches] == ["idesign-01", "idesign-02", "idesign-03", "idesign-04"]
    before_04 = {}
    for patch in patches:
        if patch.name.startswith("idesign-04"):
            before_04 = {p.name: p.read_bytes() for p in tmp_path.glob("*.py")}
        subprocess.run(["git", "-C", str(tmp_path), "apply", str(patch)], check=True, capture_output=True)
    # 04 must not change normalization or the approved bounded-corrector patch.
    for name in ("IDesign.py", "utils.py", "schemas.py"):
        assert (tmp_path / name).read_bytes() == before_04[name]
    assert b"MAX_CORRECTION_ROUNDS = 12" in (tmp_path / "IDesign.py").read_bytes()
    return tmp_path


@pytest.fixture
def config_imports(checkout, monkeypatch):
    monkeypatch.chdir(checkout)
    (checkout / "OAI_CONFIG_LIST.json").write_text(json.dumps([
        {"model": "gemma-4-12B-it", "api_key": "unused-fixture", "base_url": "http://not-contacted.invalid/v1"},
        {"model": "qwen2.5-7b-instruct", "api_key": "unused-fixture", "base_url": "http://not-contacted.invalid/v1"},
    ]))
    calls = []
    autogen = ModuleType("autogen")
    def config_list_from_json(path, filter_dict):
        calls.append(filter_dict)
        return [entry for entry in json.loads(Path(path).read_text()) if entry["model"] in filter_dict["model"]]
    autogen.config_list_from_json = config_list_from_json
    monkeypatch.setitem(sys.modules, "autogen", autogen)
    monkeypatch.setitem(sys.modules, "autogen.agentchat", ModuleType("autogen.agentchat"))
    for mod_name, class_name in (("agent", "Agent"), ("user_proxy_agent", "UserProxyAgent"),
                                 ("assistant_agent", "AssistantAgent"), ("groupchat", "GroupChat")):
        name = "autogen.agentchat." + mod_name
        stub = ModuleType(name)
        setattr(stub, class_name, type(class_name, (), {}))
        monkeypatch.setitem(sys.modules, name, stub)
    def load(name):
        spec = importlib.util.spec_from_file_location(name, checkout / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    load("schemas")
    return load, calls


def test_all_four_filters_use_actual_gemma_and_keep_upstream_config(config_imports, monkeypatch):
    monkeypatch.setenv("METAFIND_IDESIGN_MODEL", "gemma-4-12B-it")
    load, calls = config_imports
    agents = load("agents")
    corrector = load("corrector_agents")
    refiner = load("refiner_agents")
    assert calls == [{"model": ["gemma-4-12B-it"]}] * 4
    for module in (agents, corrector, refiner):
        for name, cfg in vars(module).items():
            if name.startswith("gpt4") and isinstance(cfg, dict):
                assert {c["model"] for c in cfg["config_list"]} == {"gemma-4-12B-it"}
                assert cfg["cache_seed"] == 42 and cfg["timeout"] == 600
    assert agents.gpt4_config["temperature"] == .7
    assert agents.gpt4_config["top_p"] == 1.
    assert agents.gpt4_json_engineer_config["temperature"] == 0.
    for module in (agents, corrector, refiner):
        assert module.gpt4_json_config["config_list"][0]["response_format"] == {"type": "json_object"}
    assert corrector.gpt4_config["temperature"] == refiner.gpt4_config["temperature"] == 0.


@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.parametrize("name", ["agents", "corrector_agents", "refiner_agents"])
def test_model_identity_is_mandatory_no_silent_default(config_imports, monkeypatch, value, name):
    if value is None:
        monkeypatch.delenv("METAFIND_IDESIGN_MODEL", raising=False)
    else:
        monkeypatch.setenv("METAFIND_IDESIGN_MODEL", value)
    load, calls = config_imports
    with pytest.raises(RuntimeError, match="METAFIND_IDESIGN_MODEL must name the actual served"):
        load(name)
    assert calls == []


def test_explicit_model_override_is_not_aliased(config_imports, monkeypatch):
    # Compatibility supports exact declared model IDs; it does not select the
    # research model. The project wrapper supplies its approved Gemma default.
    monkeypatch.setenv("METAFIND_IDESIGN_MODEL", " qwen2.5-7b-instruct ")
    load, calls = config_imports
    agents = load("agents")
    assert calls == [{"model": ["qwen2.5-7b-instruct"]}] * 2
    assert agents.gpt4_config["config_list"][0]["model"] == "qwen2.5-7b-instruct"


@pytest.fixture
def chain_checkout(tmp_path):
    """A local-only Git checkout borrowing pinned objects without editing them."""
    upstream = Path(os.environ.get("METAFIND_IDESIGN_TEST_REPO", "/home/kyzen/upstream/IDesign"))
    if not (upstream / ".git").exists():
        pytest.skip("pinned I-Design checkout unavailable; no download")
    check = subprocess.run(["git", "-C", str(upstream), "cat-file", "-e", PINNED], capture_output=True)
    if check.returncode:
        pytest.skip("pinned I-Design source object unavailable locally; no download")
    root = tmp_path / "checkout"
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
    objects = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "--git-path", "objects"], text=True).strip()
    objects = (upstream / objects).resolve()
    (root / ".git/objects/info/alternates").write_text(str(objects) + "\n")
    subprocess.run(["git", "-C", str(root), "update-ref", "HEAD", PINNED], check=True, capture_output=True)
    for name in FILES:
        (root / name).write_bytes(subprocess.check_output(["git", "-C", str(root), "show", f"{PINNED}:{name}"]))
    return root


@pytest.mark.parametrize("prefix", [0, 1, 2, 3, 4])
def test_cumulative_prefix_completion_and_reentrant_setup_helper(chain_checkout, prefix):
    from tools.idesign_generate import verify_patches
    from tools.idesign_patches import inspect_chain

    patches = sorted((REPO / "setup/patches").glob("idesign-*.patch"))
    for patch in patches[:prefix]:
        subprocess.run(["git", "-C", str(chain_checkout), "apply", str(patch)], check=True, capture_output=True)
    # A real upstream file outside the patch chain and an extra user file must
    # survive both initial application and a repeat invocation byte for byte.
    schema = chain_checkout / "schemas.py"
    schema.write_bytes(schema.read_bytes() + b"\n# unrelated upstream customization\n")
    extra = chain_checkout / "local_notes.txt"
    extra.write_bytes(b"user evidence\n")
    preserved = {p: p.read_bytes() for p in (schema, extra)}
    before = inspect_chain(chain_checkout, REPO / "setup/patches")
    assert before["applied_prefix"] == prefix
    assert before["status"] == ("pristine" if prefix == 0 else "complete" if prefix == 4 else "partial")
    assert [p["applied"] for p in verify_patches(REPO / "setup/patches", chain_checkout)] == [i < prefix for i in range(4)]
    # This is the identical CLI called by setup/04_idesign_env.sh; no conda,
    # pip, model endpoint or real upstream mutation is involved.
    command = [sys.executable, str(REPO / "tools/idesign_patches.py"), "apply",
               "--repo", str(chain_checkout), "--patch-dir", str(REPO / "setup/patches")]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(first.stdout)
    assert report["status"] == "complete" and report["applied_prefix"] == 4
    assert all(p["applied"] for p in report["patches"])
    assert verify_patches(REPO / "setup/patches", chain_checkout) == report["patches"]
    contents = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in chain_checkout.glob("*.py")}
    second = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(second.stdout) == report
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in contents} == contents
    assert {p: p.read_bytes() for p in preserved} == preserved


@pytest.mark.parametrize("conflict", ["mixed", "extra_change", "missing", "symlink"])
def test_unknown_patch_state_refused_before_any_write(chain_checkout, conflict):
    from tools.idesign_generate import verify_patches
    from tools.idesign_patches import PatchStateError, apply_chain, inspect_chain

    patches = sorted((REPO / "setup/patches").glob("idesign-*.patch"))
    for patch in patches:
        subprocess.run(["git", "-C", str(chain_checkout), "apply", str(patch)], check=True, capture_output=True)
    target = chain_checkout / "agents.py"
    if conflict == "mixed":
        target.write_bytes(subprocess.check_output(["git", "-C", str(chain_checkout), "show", f"{PINNED}:agents.py"]))
    elif conflict == "extra_change":
        target.write_bytes(target.read_bytes() + b"\n# unknown modification in a patched file\n")
    else:
        raw = target.read_bytes()
        target.unlink()
        if conflict == "symlink":
            other = chain_checkout / "external.py"
            other.write_bytes(raw)
            target.symlink_to(other)
    before = {p: (p.is_symlink(), p.read_bytes()) for p in chain_checkout.glob("*.py")}
    with pytest.raises(PatchStateError):
        inspect_chain(chain_checkout, REPO / "setup/patches")
    with pytest.raises(PatchStateError):
        apply_chain(chain_checkout, REPO / "setup/patches")
    assert verify_patches(REPO / "setup/patches", chain_checkout) == []
    assert {p: (p.is_symlink(), p.read_bytes()) for p in chain_checkout.glob("*.py")} == before
    command = [sys.executable, str(REPO / "tools/idesign_patches.py"), "apply",
               "--repo", str(chain_checkout), "--patch-dir", str(REPO / "setup/patches")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2 and "patch chain refused" in result.stderr
    assert {p: (p.is_symlink(), p.read_bytes()) for p in chain_checkout.glob("*.py")} == before
