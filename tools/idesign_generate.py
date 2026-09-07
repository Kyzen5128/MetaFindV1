#!/usr/bin/env python3
"""Drive I-Design to produce the evaluation scenes paper section 3.3 calls for.

Runs in the `IDesign` conda environment, NOT in `MetaFind`: I-Design pins
networkx 2.6 / jsonschema 4.3 / numpy 1.26, which would break transformers and
torch in the training environment. The interface between the two is a JSON file
on disk, so they never need to share an interpreter.

    conda activate IDesign
    PYTHONPATH=<idesign_repo> python tools/idesign_generate.py --n-scenes 2

The default planner is Gemma served under its actual model ID, following the
recorded all-LLM Gemma decision (workflow/DECISION_LEDGER.md:2087). This remains
a deviation from upstream GPT-4. Patches 01--03 carry the previously selected
model/graph/retry adaptations; patch 04 makes all model filters configurable.

Two properties of the upstream setup, checked against I-Design's paper 4.1:

* temperature 0.7 and top_p 1.0 are what the paper specifies, and I-Design's
  own agents.py already sets both, so they are inherited rather than assumed.
* Upstream requests response_format=json_object. Whether a particular local
  endpoint honours it requires runtime evidence; setting that field alone
  does not establish constrained decoding or valid scene output.

An earlier version served Qwen under the alias `gpt-4` so that I-Design's
hardcoded `filter_dict={"model": ["gpt-4"]}` would resolve without touching its
source. That was a bad trade: every log line and config file then said `gpt-4`
while nothing of the sort was running, and it misled a reader within minutes.
I-Design is patched instead: patch 04 reads METAFIND_IDESIGN_MODEL at all four
filter sites. Config, endpoint, and sidecar use the same real model name.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import json
import math
import traceback
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# A direct script invocation puts tools/, rather than the checkout, on sys.path.
# Keep the documented IDesign-only PYTHONPATH invocation usable from any CWD.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from metafind import paths

DEFAULT_OUT = paths.OUTPUTS / "idesign"
DEFAULT_IDESIGN_REPO = Path(os.environ.get("IDESIGN_REPO", REPO.parent / "upstream" / "IDesign"))

# [CORRECTED] These used to be "A creative vibrant livingroom" and "An aged
# archive room" at [4.0, 4.0, 2.5] with n=15 and n=12 -- invented, while the
# file claimed nothing was being invented here.
#
# I-Design's paper PUBLISHES its evaluation prompts: Table 4 lists 20 minimal
# prompts with room dimensions, Table 5 lists 40 elaborate ones across four
# categories. Two of those are used below verbatim, so the smoke run exercises
# the pipeline on inputs its authors actually ran.
#
# Still UNKNOWN, and still not invented here:
#   * MetaFind 3.3's "200 randomly sampled scenes" is NOT this list -- the paper
#     publishes 60 prompts in total. Where MetaFind's 200 came from is U-27.
#   * `n`, the object count I-Design requires, is given nowhere. Table 1 reports
#     NObj (12.7 bedroom, 23.6 living room) as an OUTPUT of the runs, not an
#     input to them. The values below are ours and are labelled as such.
SMOKE_PROMPTS = [
    # I-Design Table 4 #1 (minimal), verbatim prompt and dimensions. n is ours.
    ("Design me a bedroom.", [3.0, 4.0, 2.4], 10),
    # I-Design Table 4 #11 (minimal), verbatim prompt and dimensions. n is ours.
    ("Design me a living room.", [4.0, 5.0, 2.8], 10),
]


def endpoint_model_id(base_url: str) -> str:
    """Ask the server what it is really serving, and record it."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=10) as r:
            data = json.load(r)
        return ",".join(sorted({m["id"] for m in data.get("data", [])}))
    except Exception as exc:  # noqa: BLE001 -- recorded, never fatal
        return f"unavailable: {exc}"


def verify_patches(patch_dir: Path, repo: Path) -> list[dict]:
    """Verify cumulative pinned source states, including superseded patches."""
    from tools.idesign_patches import inspect_chain

    try:
        return inspect_chain(repo, patch_dir)["patches"]
    except (OSError, ValueError) as exc:
        print(f"I-Design patch chain refused: {exc}", file=sys.stderr)
        return []


def write_config(workdir: Path, base_url: str, api_key: str, model: str) -> None:
    """I-Design's agents.py reads OAI_CONFIG_LIST.json from the CWD at import."""
    workdir.mkdir(parents=True, exist_ok=True)
    models = [model]
    (workdir / "OAI_CONFIG_LIST.json").write_text(
        json.dumps(
            [{"model": m, "api_key": api_key, "base_url": base_url} for m in models],
            indent=2,
        )
    )


def run_one(
    idesign_repo: Path,
    workdir: Path,
    prompt: str,
    dims: list[float],
    n_objects: int,
    seed: int | None = None,
) -> dict:
    """One scene, in its own CWD. Returns the sidecar record."""
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Imported here, after the CWD is correct, because agents.py loads the
    # config at module scope.
    previous_path = sys.path[:]
    sys.path.insert(0, str(idesign_repo))
    prev_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        from IDesign import IDesign  # noqa: PLC0415
        import agents  # noqa: PLC0415
        from utils import ROOM_LAYOUT_ELEMENTS  # noqa: PLC0415

        if seed is not None:
            # What this does and does not control.
            #
            # cache_seed is autogen's CACHE NAMESPACE, so this gives a scene
            # reproducible cache provenance and makes retries select
            # consistently. Nothing in this repo shows it reaching vLLM's
            # sampling RNG, so it must not be described as controlling
            # generation end to end.
            #
            # Incomplete by construction: corrector_agents.py and
            # refiner_agents.py each build their own module-local gpt4_config,
            # which this does not reach. Recorded rather than papered over.
            random.seed(seed)
            for name in ("gpt4_config", "gpt4_prev_config", "gpt4_json_config",
                         "gpt4_json_engineer_config"):
                if isinstance(getattr(agents, name, None), dict):
                    getattr(agents, name)["cache_seed"] = seed

        design = IDesign(
            no_of_objects=n_objects, user_input=prompt, room_dimensions=dims
        )
        design.create_initial_design()
        design.correct_design()
        design.refine_design()
        design.create_object_clusters(verbose=False)
        design.backtrack(verbose=False)
        design.to_json("scene_graph.json")
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = previous_path

    scene = json.loads((workdir / "scene_graph.json").read_text())
    if not isinstance(scene, list) or any(not isinstance(o, dict) for o in scene):
        raise ValueError("planner scene_graph must be an object list")
    objects = [o for o in scene if o.get("new_object_id") not in ROOM_LAYOUT_ELEMENTS]
    placed = [o for o in objects if "position" in o]
    if not objects or len(placed) != len(objects):
        raise ValueError("planner did not position every returned object")
    return {
        "prompt": prompt,
        "room_dimensions": dims,
        "n_objects_requested": n_objects,
        "n_objects_returned": len(objects),
        "n_objects_positioned": len(placed),
        "n_room_priors": len(scene) - len(objects),
        "n_scene_nodes": len(scene),
        "wallclock_s": round(time.time() - started, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def scene_specs(path: Path | None, n_scenes: int) -> list[dict]:
    """Validate the complete requested batch before creating files or calling a model."""
    if n_scenes < 1:
        raise ValueError("--n-scenes must be a positive integer")
    if path is None:
        if n_scenes > len(SMOKE_PROMPTS):
            raise ValueError(
                f"--n-scenes {n_scenes} exceeds the {len(SMOKE_PROMPTS)} smoke prompts; "
                "pass --scene-spec-file with explicit prompt, room_dimensions, n_objects, seed, source"
            )
        return [{"prompt": p, "room_dimensions": d[:], "n_objects": n, "seed": None,
                 "source": "smoke"} for p, d, n in SMOKE_PROMPTS[:n_scenes]]
    specs = []
    required = {"prompt", "room_dimensions", "n_objects", "seed", "source"}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        spec = json.loads(line)
        if not isinstance(spec, dict) or not required <= spec.keys():
            raise ValueError(f"spec line {line_no}: requires {sorted(required)}")
        if any(not isinstance(spec[k], str) or not spec[k].strip() for k in ("prompt", "source")):
            raise ValueError(f"spec line {line_no}: prompt/source must be nonempty strings")
        dims = spec["room_dimensions"]
        if not isinstance(dims, list) or len(dims) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
            for x in dims
        ):
            raise ValueError(f"spec line {line_no}: room_dimensions needs three finite positive numbers")
        if type(spec["n_objects"]) is not int or spec["n_objects"] < 1:
            raise ValueError(f"spec line {line_no}: n_objects must be a positive integer")
        if spec["seed"] is not None and type(spec["seed"]) is not int:
            raise ValueError(f"spec line {line_no}: seed must be an integer or null")
        specs.append(spec)
    if len(specs) < n_scenes:
        raise ValueError(f"--n-scenes {n_scenes} but {path} has {len(specs)} specs; refusing to truncate")
    return specs[:n_scenes]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idesign-repo", type=Path, default=DEFAULT_IDESIGN_REPO)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="local-vllm")
    ap.add_argument(
        "--model",
        default="gemma-4-12B-it",
        help="Real endpoint model ID; patch 04 reads the same name for all agents.",
    )
    ap.add_argument("--n-scenes", type=int, default=len(SMOKE_PROMPTS))
    ap.add_argument(
        "--scene-spec-file",
        type=Path,
        help="JSONL of scene specs; required for anything larger than a smoke run.",
    )
    args = ap.parse_args(argv)
    args.model = args.model.strip()
    if not args.model:
        print("--model must name the actual served model", file=sys.stderr)
        return 2
    try:
        specs = scene_specs(args.scene_spec_file, args.n_scenes)
    except (OSError, ValueError) as exc:
        print(f"Invalid scene specification: {exc}", file=sys.stderr)
        return 2
    args.idesign_repo = args.idesign_repo.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        print(f"Refusing to overwrite existing output: {args.out}", file=sys.stderr)
        return 2

    if not (args.idesign_repo / "IDesign.py").exists():
        print(f"I-Design not found at {args.idesign_repo}", file=sys.stderr)
        return 2

    revision = subprocess.run(
        ["git", "-C", str(args.idesign_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    applied_patches = verify_patches(
        Path(__file__).resolve().parents[1] / "setup" / "patches", args.idesign_repo
    )
    for rec in applied_patches:
        mark = "applied" if rec["applied"] else "NOT APPLIED"
        print(f"  patch {rec['name']}: {mark} ({rec['sha256'][:12]}...)")
    if not applied_patches or not all(r["applied"] for r in applied_patches):
        print(
            "Refusing to generate: the clone is missing patches this repo ships. "
            "Run setup/04_idesign_env.sh.",
            file=sys.stderr,
        )
        return 2

    served = endpoint_model_id(args.base_url)
    print(f"I-Design {revision[:8]} | endpoint serves: {served}")
    args.out.mkdir(parents=True, exist_ok=False)
    records, failures = [], 0
    os.environ["METAFIND_IDESIGN_MODEL"] = args.model

    for i, spec in enumerate(specs):
        prompt, dims, n_obj = spec["prompt"], spec["room_dimensions"], spec["n_objects"]
        scene_id = f"scene_{i:04d}"
        workdir = args.out / scene_id
        write_config(workdir, args.base_url, args.api_key, args.model)
        print(f"\n=== {scene_id}: {prompt!r} ===", flush=True)
        try:
            rec = run_one(
                args.idesign_repo, workdir, prompt, dims, n_obj, seed=spec["seed"]
            )
        except Exception as exc:  # noqa: BLE001 -- one bad scene must not stop the batch
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            (workdir / "failure.txt").write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )
            continue
        rec |= {
            "scene_id": scene_id,
            "idesign_revision": revision,
            # Every patch applied, not just the cosmetic one. 02 and 03 change
            # BEHAVIOUR -- 02 moves layout references, canonicalises
            # prepositions, drops dangling ids and deduplicates objects; 03
            # bounds the correction loops, varies the cache seed per retry and
            # abandons a scene on exhaustion. Both change which scenes exist and
            # what they contain, so a sidecar naming only patch 01 would tell a
            # later reader the scenes came from near-stock I-Design.
            "idesign_patches": applied_patches,
            "scene_source": spec["source"],
            "seed": spec["seed"],
            # Planner replacement follows the recorded all-LLM Gemma decision.
            "planner_model": args.model,
            "planner_endpoint_serves": served,
        }
        (workdir / "sidecar.json").write_text(json.dumps(rec, indent=2))
        records.append(rec)
        print(
            f"  ok: {rec['n_objects_returned']} objects, "
            f"{rec['n_objects_positioned']} positioned, {rec['wallclock_s']}s"
        )

    (args.out / "index.json").write_text(json.dumps(records, indent=2))
    (args.out / "run_status.json").write_text(json.dumps({
        "schema": "metafind.idesign_batch.v1",
        "status": "complete" if not failures else "incomplete",
        "requested": len(specs), "succeeded": len(records), "failed": failures,
        "scene_ids": [r["scene_id"] for r in records],
    }, indent=2))
    print(f"\n{len(records)} generated, {failures} failed -> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
