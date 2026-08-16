"""Layout-aware fine-tuning: train the query fuser and the ESSGNN.

INCOMPLETE -- deliberately NOT carrying an `IMPLEMENTS-NODE` marker yet.

The sample construction, the unique-positive batcher and the context-graph
builder are written and testable. The training loop is NOT finished: encode_query
references `modalities` and `text_vectors` that main() does not yet load, so this
module cannot run.

The marker is a claim, and the README's implementation count is computed from it.
Adding it now would make the status table say 18 of 31 nodes work when one of
them does not -- which is the exact class of false status this project keeps
finding. It goes on when the smoke run passes.

Writes ``variant_ckpts``, ``variant_status``, ``run_progress``, ``cost_ledger``
and ``degraded_flags``.

The sample
----------

[U-08d] One leave-one-out sample per eligible ProcTHOR object instance. The
target is REMOVED from the graph before ESSGNN sees it -- that is the
load-bearing part, and it is load-bearing in a way that hides: leaving the
target in lets the layout encoder read the answer off its own input, the loss
falls faster, and a falling loss is what learning looks like.

[U-08a] The positive is the target's OWN assetId, encoded by the frozen Stage 1
gallery tower and looked up in stage2_gallery_index. No ProcTHOR-to-Objaverse
correspondence exists and none is needed.

[U-08e] The sampler admits each positive_asset_id at most once per batch.
665,320 train instances draw on 1,467 assets, so without the constraint 99.3% of
batches at size 64 carry a duplicate -- and a frozen encoder gives one assetId
ONE embedding, so the duplicate is a negative bit-identical to the positive. The
gradient would be asking the model to separate two identical vectors.

What trains
-----------

[PAPER 2.6] "Only the query-side fuser and the ESSGNN module are updated; the
gallery encoder is frozen." Both halves are enforced and both are checked --
L1-STAGE2-QUERY-ENCODERS-FROZEN exists because the query POINT encoder is the
dangerous one: Stage 1 trains it, so it arrives here with requires_grad already
True and would keep training unless something turns it off.

[Eq. 7a/7b] Bidirectional, unlike Stage 1's Eq. 5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from metafind import paths, runlog

paths.setup_env()

NODE = "n13_train_stage2"
TRAINER_VERSION = 1

CKPT_DIR = paths.CHECKPOINTS
VARIANT_STATUS = paths.OUTPUTS / "variant_status.json"
VARIANT_CKPTS = paths.OUTPUTS / "variant_ckpts.json"


def load_stage2_protocols() -> tuple[dict, dict, dict]:
    def read(name: str, writer: str) -> dict:
        path = paths.OUTPUTS / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run {writer} first")
        proto = json.loads(path.read_text())
        if isinstance(proto, dict) and proto.get("status") not in (None, "resolved"):
            raise ValueError(f"{name} is {proto.get('status')!r}")
        return proto

    return (read("stage2_protocol.json", "n09b_resolve_stage2_protocol"),
            read("essgnn_edge_protocol.json", "n09b_resolve_stage2_protocol"),
            read("essgnn_arch_protocol.json", "n09b_resolve_stage2_protocol"))


def enumerate_samples(train_houses: list[str], eligible: set[str]) -> list[tuple[str, int, str]]:
    """[U-08d] Every eligible instance of every train house, enumerated ONCE.

    Returns ``(house_id, node_index, asset_id)``. The enumeration is written
    down rather than drawn per epoch because ``instance_resampling`` is fixed:
    a rerun at the same seed must train on the same tuples, and "the same seed"
    means nothing if the sample set itself is redrawn.
    """
    samples = []
    for house_id in train_houses:
        path = paths.SCENE_GRAPHS / f"{house_id}.json"
        if not path.exists():
            continue
        graph = json.loads(path.read_text())
        for node in graph["nodes"]:
            asset_id = str(node["asset_id"])
            if asset_id in eligible:
                samples.append((house_id, node["index"], asset_id))
    return samples


def unique_positive_batches(samples: list[tuple[str, int, str]], batch_size: int,
                            rng: np.random.Generator) -> list[list[int]]:
    """[U-08e] Batches in which no assetId appears twice.

    Greedy over a shuffled order: a sample whose asset is already in the current
    batch is deferred rather than dropped, so every instance is still used
    within the epoch. Dropping them would silently reweight the corpus toward
    rare assets, which is a different experiment.
    """
    order = rng.permutation(len(samples))
    pending, batches, current, seen = list(order), [], [], set()
    while pending:
        deferred = []
        for idx in pending:
            asset = samples[idx][2]
            if asset in seen:
                deferred.append(idx)
                continue
            current.append(int(idx))
            seen.add(asset)
            if len(current) == batch_size:
                batches.append(current)
                current, seen = [], set()
        if not deferred:
            break
        if len(deferred) == len(pending) and not current:
            # Everything left collides with everything left, which can only
            # happen if fewer distinct assets remain than the batch needs.
            break
        pending = deferred
    if current:
        batches.append(current)
    return batches


def build_context_graph(graph: dict, target_index: int, edge_dim: int,
                        sem_cache: dict, text_map: dict, missing_token: np.ndarray):
    """[U-08d] The house graph MINUS the target.

    Node indices are remapped because ESSGNN indexes edges positionally; leaving
    a hole would make every edge past the target point at the wrong node, and
    nothing downstream would object -- the shapes stay valid.
    """
    keep = [n for n in graph["nodes"] if n["index"] != target_index]
    remap = {n["index"]: k for k, n in enumerate(keep)}

    pos = np.array([n["position"] for n in keep], dtype=np.float32)
    rows, cols, attrs = [], [], []
    for i, j in graph["sem_edge_ids"]:
        if i == target_index or j == target_index:
            continue
        ai = str(graph["nodes"][i]["asset_id"])
        aj = str(graph["nodes"][j]["asset_id"])
        vec = sem_cache.get(_edge_key(ai, aj, text_map))
        # [U-30] A missing semantic edge gets the learned token, never zeros:
        # a zero vector is a valid point in the space and indistinguishable
        # downstream from a real relation.
        attrs.append(missing_token if vec is None else vec)
        # [U-19] symmetric, matching what n07 stored
        rows += [remap[i], remap[j]]
        cols += [remap[j], remap[i]]
        attrs.append(attrs[-1])
    edge_index = np.array([rows, cols], dtype=np.int64) if rows else np.zeros((2, 0), np.int64)
    edge_attr = (np.stack(attrs).astype(np.float32) if attrs
                 else np.zeros((0, edge_dim), np.float32))
    return keep, pos, edge_index, edge_attr


def _edge_key(a: str, b: str, text_map: dict) -> str:
    from metafind.data.semantic_edges import cache_key

    meta = text_map["_meta"]
    return cache_key(text_map[a], text_map[b], meta["prompt_version"],
                     meta["llm_model"], meta["text_encoder_version"])


def encode_query(model, backbone, graph: dict, target_index: int, asset_id: str,
                 drop_layout: bool, device: str, edge_dim: int,
                 sem_cache: dict, text_map: dict, missing_token):
    """One leave-one-out query: the target's modalities plus its scene context.

    The target is removed from the graph here and nowhere else, so the removal
    cannot be skipped by a caller. [U-08d] Leaving it in would let ESSGNN read
    the answer off its own input; the loss would fall and nothing downstream
    would distinguish that from learning.
    """
    import torch
    from PIL import Image

    rec = modalities[asset_id]
    with torch.no_grad():
        # The query encoders are frozen in Stage 2 (2.6); only the fuser and
        # the ESSGNN move, so these three cost no graph.
        text = backbone.encode_text([rec["text"]])
        views = backbone.encode_image(torch.stack([
            backbone.preprocess(Image.open(v).convert("RGB"))
            for v in rec["view_paths"]]))
        image = views.mean(dim=0, keepdim=True)
        cloud = np.load(rec["pointcloud_uri"])["xyz"].astype(np.float32)
        pc = np.concatenate([cloud, np.full_like(cloud, 0.5)], axis=1)[None]
        pc_vec = backbone.encode_pc(torch.from_numpy(pc))

    layout = None
    if not drop_layout:
        keep, pos, edge_index, edge_attr = build_context_graph(
            graph, target_index, edge_dim, sem_cache, text_map, missing_token)
        if keep:
            node_feat = torch.stack([
                torch.from_numpy(text_vectors[str(n["asset_id"])]) for n in keep
            ]).to(device)
            layout = model.query.encode_layout(
                node_feat,
                torch.from_numpy(pos).to(device),
                torch.from_numpy(edge_index).to(device),
                torch.from_numpy(edge_attr).to(device))

    embeds = {"text": text, "image": image, "pc": pc_vec}
    # [2.4] the query side may drop the point cloud; here it is present because
    # target_eligibility required one. `present=None` means all three.
    return model.query(embeds, present=None, layout=layout,
                       drop_layout=None)[0]


def trainable_state_dict(model) -> dict:
    """[L1-CKPT-TRAINABLE-ONLY] Same rule as Stage 1, and it matters more here:
    Stage 2 has ELEVEN variants, so a whole-state_dict save is 112 GB."""
    return {name: p.detach().cpu()
            for name, p in model.named_parameters() if p.requires_grad}


def freeze_for_stage2(model, backbone) -> dict:
    """[PAPER 2.6] Only the query fuser and the ESSGNN move.

    The query POINT encoder is the trap. Stage 1 trains it, so it arrives with
    requires_grad True; nothing about Stage 2's code would fail if it kept
    training, and the gallery index -- built from the Stage 1 weights -- would
    quietly stop matching what the query side produces.
    """
    model.freeze_gallery(True)
    for p in backbone.parameters() if hasattr(backbone, "parameters") else []:
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        trains = name.startswith("query.fusion") or name.startswith("query.layout_encoder") \
            or name.endswith("lam") or "lambda" in name
        p.requires_grad_(trains)
    return {name: p.requires_grad for name, p in model.named_parameters()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit-houses", type=int)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from metafind.train.stage1 import build_model, load_protocols
    from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    encoding, training, hyperparameters = load_protocols()
    stage2, edge_proto, arch_proto = load_stage2_protocols()
    values = hyperparameters["values"]

    positive_map = json.loads((paths.OUTPUTS / "stage2_positive_map.json").read_text())
    index_record = json.loads((paths.OUTPUTS / "stage2_gallery_index.json").read_text())
    ckpt = json.loads((paths.CHECKPOINTS / "stage1_ckpt.json").read_text())

    # [G6] The index must come from the checkpoint this run loads. Comparing
    # here as well as at the gate: a gate verdict is a record of the past, and
    # the index can be rebuilt between the verdict and the run.
    gallery_index = np.load(index_record["uri"])
    id_to_row = {a: i for i, a in enumerate(gallery_index["ids"].tolist())}
    gallery_vecs = torch.from_numpy(gallery_index["embeddings"]).to(args.device)

    scene_splits = json.loads((paths.OUTPUTS / "scene_splits.json").read_text())
    train_houses = scene_splits["train_houses"]
    if args.limit_houses:
        train_houses = train_houses[: args.limit_houses]

    eligible = set(positive_map) & set(id_to_row)
    samples = enumerate_samples(train_houses, eligible)
    if not samples:
        print("no eligible sample; check stage2_positive_map and the gallery index",
              flush=True)
        return 2

    seed = values["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    backbone = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    model, _ = build_model(encoding, training, hyperparameters)
    state = torch.load(ckpt["uri"], map_location="cpu")
    model.load_state_dict(state["trainable_state"], strict=False)
    model.to(args.device)
    grads = freeze_for_stage2(model, backbone)

    loss_fn = MetaFindContrastiveLoss(ContrastiveConfig(
        # [Eq. 7a/7b] symmetric, unlike Stage 1's Eq. 5
        bidirectional=True,
        learnable_temperature=values["learnable_temperature"],
        init_temperature=values["init_temperature"],
        max_logit_scale=values["max_logit_scale"])).to(args.device)

    params = [p for p in list(model.parameters()) + list(loss_fn.parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=values["learning_rate"],
                            weight_decay=values["weight_decay"])

    epochs = args.epochs or values["epochs"]
    batches = unique_positive_batches(samples, values["batch_size"], rng)
    print(f"{len(samples):,} samples over {len(train_houses):,} houses, "
          f"{len(batches):,} batches/epoch, {sum(grads.values()):,} trainable tensors",
          flush=True)

    started, step = time.time(), 0
    with runlog.run_progress(NODE):
        for epoch in range(epochs):
            model.train()
            for batch in batches:
                # [U-32 / 2.6] "omitted in 30% of BATCHES" -- ONE draw per batch,
                # so every sample in a dropped batch loses the layout term and
                # every sample in a kept batch has it. L1-SCENE-DROPOUT-30
                # asserts the within-batch variance of this mask is zero.
                drop = bool(rng.random() < stage2_dropout)

                queries, positives = [], []
                for idx in batch:
                    house_id, target_index, asset_id = samples[idx]
                    graph = graphs[house_id]
                    q = encode_query(model, backbone, graph, target_index,
                                     asset_id, drop, args.device,
                                     edge_dim, sem_cache, text_map,
                                     missing_token)
                    queries.append(q)
                    positives.append(gallery_vecs[id_to_row[asset_id]])

                q = torch.stack(queries)
                g = torch.stack(positives)
                out = loss_fn(q, g)

                opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                opt.step()
                step += 1

                if step % 50 == 0:
                    print(f"  epoch {epoch} step {step}: loss {out['loss'].item():.4f}, "
                          f"tau {loss_fn.temperature.item():.4f}, "
                          f"layout {'dropped' if drop else 'used'}", flush=True)

            record = {
                "variant_id": args.variant,
                "uri": str(CKPT_DIR / f"stage2_{args.variant}.pt"),
                "trainable_only": True,
                "n_params_saved": sum(v.numel() for v in trainable_state_dict(model).values()),
            }
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"trainable_state": trainable_state_dict(model),
                    "trainer_version": TRAINER_VERSION,
                    "variant": args.variant}, record["uri"])
        record["sha256"] = hashlib.sha256(Path(record["uri"]).read_bytes()).hexdigest()
        record["size_bytes"] = Path(record["uri"]).stat().st_size

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1), steps=step)
    print(f"\n{args.variant}: {record['n_params_saved']:,} params, "
          f"{record['size_bytes'] / 1e6:.0f} MB -> {record['uri']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
