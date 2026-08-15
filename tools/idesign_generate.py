#!/usr/bin/env python3
"""Drive I-Design to produce the evaluation scenes paper section 3.3 calls for.

Runs in the `IDesign` conda environment, NOT in `MetaFind`: I-Design pins
networkx 2.6 / jsonschema 4.3 / numpy 1.26, which would break transformers and
torch in the training environment. The interface between the two is a JSON file
on disk, so they never need to share an interpreter.

    conda activate IDesign
    PYTHONPATH=<idesign_repo> python tools/idesign_generate.py --n-scenes 2

The LLM is Qwen served over an OpenAI-compatible endpoint (deviation D-5).

An earlier version served Qwen under the alias `gpt-4` so that I-Design's
hardcoded `filter_dict={"model": ["gpt-4"]}` would resolve without touching its
source. That was a bad trade: every log line and config file then said `gpt-4`
while nothing of the sort was running, and it misled a reader within minutes.
I-Design is patched instead -- setup/patches/idesign-01-qwen-model-name.patch,
four filter sites across three files -- so the model name is honest end to end.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import json
import traceback
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT = Path("/mnt/data1/kyzen/MetaFind/outputs/idesign")

# Paper 3.3 gives no prompt list -- it says only "200 randomly sampled scenes".
# These two exist to prove the pipeline runs (R-01); the real list is part of
# the composition protocol decision (U-21) and is not invented here.
SMOKE_PROMPTS = [
    ("A creative vibrant livingroom", [4.0, 4.0, 2.5], 15),
    ("An aged archive room", [4.0, 4.0, 2.5], 12),
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
    """Which patches are ACTUALLY applied in the external checkout.

    Listing the .patch files in this repository says what we ship, not what the
    clone contains -- a sidecar built from that would claim provenance the
    artifacts may not have. `git apply --reverse --check` succeeds only if the
    change is already present, so it answers the question that matters. The
    hash is recorded too, because a patch file can change under a stable name.
    """
    out = []
    for patch in sorted(patch_dir.glob("idesign-*.patch")):
        applied = (
            subprocess.run(
                ["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)],
                capture_output=True,
            ).returncode
            == 0
        )
        out.append(
            {
                "name": patch.stem,
                "applied": applied,
                "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            }
        )
    return out


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
    sys.path.insert(0, str(idesign_repo))
    prev_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        from IDesign import IDesign  # noqa: PLC0415
        import agents  # noqa: PLC0415

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

    scene = json.loads((workdir / "scene_graph.json").read_text())
    placed = [o for o in scene if isinstance(o, dict) and "position" in o]
    return {
        "prompt": prompt,
        "room_dimensions": dims,
        "n_objects_requested": n_objects,
        "n_objects_returned": len(scene),
        "n_objects_positioned": len(placed),
        "wallclock_s": round(time.time() - started, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idesign-repo", type=Path, default=Path("/home/kyzen/IDesign"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="local-vllm")
    ap.add_argument(
        "--model",
        default="qwen2.5-7b-instruct",
        help="Must match both the patched filter_dict and vLLM --served-model-name.",
    )
    ap.add_argument("--n-scenes", type=int, default=len(SMOKE_PROMPTS))
    ap.add_argument(
        "--scene-spec-file",
        type=Path,
        help="JSONL of scene specs; required for anything larger than a smoke run.",
    )
    args = ap.parse_args()

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
    served = endpoint_model_id(args.base_url)
    print(f"I-Design {revision[:8]} | endpoint serves: {served}")
    for rec in applied_patches:
        mark = "applied" if rec["applied"] else "NOT APPLIED"
        print(f"  patch {rec['name']}: {mark} ({rec['sha256'][:12]}...)")
    if not all(r["applied"] for r in applied_patches):
        print(
            "Refusing to generate: the clone is missing patches this repo ships. "
            "Run setup/04_idesign_env.sh.",
            file=sys.stderr,
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    records, failures = [], 0

    if args.scene_spec_file:
        specs = [
            json.loads(line)
            for line in args.scene_spec_file.read_text().splitlines()
            if line.strip()
        ]
        required = {"prompt", "room_dimensions", "n_objects", "seed", "source"}
        for i, d in enumerate(specs):
            if missing := required - d.keys():
                print(f"spec line {i}: missing {sorted(missing)}", file=sys.stderr)
                return 2
        # Never silently deliver fewer scenes than asked for. Paper 3.3's figure
        # is 200; a run that quietly produced 150 and reported a mean would be
        # answering a different question.
        if len(specs) < args.n_scenes:
            print(
                f"--n-scenes {args.n_scenes} but {args.scene_spec_file} has "
                f"{len(specs)} specs. Refusing to truncate.",
                file=sys.stderr,
            )
            return 2
        specs = specs[: args.n_scenes]
    else:
        # Refuse to stretch two smoke prompts into an evaluation set. Cycling
        # them would hand back 100 copies of each and call it "200 randomly
        # sampled scenes" (paper 3.3), which it is not. The real list is part of
        # the composition protocol decision (U-21/U-27) and is not invented here.
        if args.n_scenes > len(SMOKE_PROMPTS):
            print(
                f"--n-scenes {args.n_scenes} exceeds the {len(SMOKE_PROMPTS)} smoke "
                "prompts. Pass --scene-spec-file with one JSON object per line "
                '({"prompt", "room_dimensions", "n_objects", "seed", "source"}) '
                "derived from the resolved composition_protocol.",
                file=sys.stderr,
            )
            return 2
        specs = [
            {"prompt": p, "room_dimensions": d, "n_objects": n, "seed": None,
             "source": "smoke"}
            for p, d, n in SMOKE_PROMPTS[: args.n_scenes]
        ]

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
            # D-5: I-Design's planner is GPT-4 upstream; here it is Qwen.
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
    print(f"\n{len(records)} generated, {failures} failed -> {args.out}")
    return 1 if failures and not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
