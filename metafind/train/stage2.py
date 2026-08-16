"""Layout-aware fine-tuning: train the query fuser and the ESSGNN.

STILL NOT CARRYING AN `IMPLEMENTS-NODE` MARKER. The loop is complete -- sample
construction, batching, context graphs, forward, Eq. 7/8, backward, checkpoint --
but it has never been executed, because it needs `stage1_ckpt` (n10 has not run)
and `sem_edge_cache` (n08 has not run).

The marker is a claim, and the README's implementation count is computed from it.
It goes on when a smoke run passes, not when the code looks finished. An earlier
version of this docstring said the loop referenced names main() never loaded;
that was true then and is not now, and the marker stayed off for the different
reason above.

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

# [PAPER 2.6] "the layout vector e_layout is omitted in 30% of batches". A
# SEPARATE constant from stage1_config.PAPER_P_MASK, which is 2.6's other 30% --
# per-modality masking in Stage 1. Equal values, different mechanisms; sharing
# one symbol coupled Table 3's p_mask sweep to scene dropout.
PAPER_SCENE_DROPOUT = 0.30

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
    pending, batches, current, seen = [int(i) for i in order], [], [], set()
    while pending:
        deferred = []
        for idx in pending:
            asset = samples[idx][2]
            if asset in seen:
                deferred.append(idx)
                continue
            current.append(idx)
            seen.add(asset)
            if len(current) == batch_size:
                batches.append(current)
                current, seen = [], set()
        # A pass that placed NOTHING means every remaining sample collides with
        # the batch being filled. Flushing it short and clearing `seen` is what
        # makes progress; without this the loop spins forever on a corpus where
        # one asset dominates -- MEASURED: five samples of one asset hung the
        # test suite until it was killed at 120 s.
        if len(deferred) == len(pending):
            if not current:
                break
            batches.append(current)
            current, seen = [], set()
        pending = deferred
    if current:
        batches.append(current)
    return batches


def build_context_graph(graph: dict, target_index: int, edge_dim: int,
                        sem_cache: dict, text_map: dict):
    """[U-08d] The house graph MINUS the target.

    Node indices are remapped because ESSGNN indexes edges positionally; leaving
    a hole would make every edge past the target point at the wrong node, and
    nothing downstream would object -- the shapes stay valid.

    Returns ``(keep, pos, edge_index, edge_attr, edge_missing)``. The last is a
    bool mask, not a filled-in token: ``essgnn_edge_protocol`` records
    ``semantic_missing_representation = learned_missing_token``, and a token
    substituted here would be a constant, not something the model learns. The
    substitution happens inside ``ESSGNN.forward`` from an ``nn.Parameter``.

    Rows of ``edge_attr`` under the mask are zeros ONLY as a placeholder that
    never reaches an MLP -- ``torch.where`` replaces them before the first
    layer. That is not a zero-fill in the L1-SEMEDGE-NO-ZEROFILL sense, and the
    mask travelling beside the array is what keeps the distinction checkable.
    """
    keep = [n for n in graph["nodes"] if n["index"] != target_index]
    remap = {n["index"]: k for k, n in enumerate(keep)}

    pos = np.array([n["position"] for n in keep], dtype=np.float32)
    rows, cols, attrs, missing = [], [], [], []
    for i, j in graph["sem_edge_ids"]:
        if i == target_index or j == target_index:
            continue
        ai = str(graph["nodes"][i]["asset_id"])
        aj = str(graph["nodes"][j]["asset_id"])
        vec = sem_cache.get(_edge_key(ai, aj, text_map))
        attrs.append(np.zeros(edge_dim, np.float32) if vec is None else vec)
        missing.append(vec is None)
        # [U-19] symmetric, matching what n07 stored
        rows += [remap[i], remap[j]]
        cols += [remap[j], remap[i]]
        attrs.append(attrs[-1])
        missing.append(missing[-1])
    edge_index = np.array([rows, cols], dtype=np.int64) if rows else np.zeros((2, 0), np.int64)
    edge_attr = (np.stack(attrs).astype(np.float32) if attrs
                 else np.zeros((0, edge_dim), np.float32))
    edge_missing = np.array(missing, dtype=bool)
    return keep, pos, edge_index, edge_attr, edge_missing


def _edge_key(a: str, b: str, text_map: dict) -> str:
    from metafind.data.semantic_edges import cache_key

    meta = text_map["_meta"]
    return cache_key(text_map[a], text_map[b], meta["prompt_version"],
                     meta["llm_model"], meta["text_encoder_version"])


def encode_query(model, backbone, graph: dict, target_index: int, asset_id: str,
                 drop_layout: bool, device: str, data: "Stage2Data"):
    """One leave-one-out query: the target's modalities plus its scene context.

    The target is removed from the graph here and nowhere else, so the removal
    cannot be skipped by a caller. [U-08d] Leaving it in would let ESSGNN read
    the answer off its own input; the loss would fall and nothing downstream
    would distinguish that from learning.
    """
    import torch
    from PIL import Image

    rec = data.modalities[asset_id]
    with torch.no_grad():
        # The query encoders are frozen in Stage 2 (2.6); only the fuser and
        # the ESSGNN move, so these three cost no graph.
        text = backbone.encode_text([rec["text"]])
        views = backbone.encode_image(torch.stack([
            backbone.preprocess(Image.open(v).convert("RGB"))
            for v in rec["view_paths"]]))
        image = views.mean(dim=0, keepdim=True)
        # [P0-4] same normalisation as the Stage 2 gallery -- query and gallery
        # must see one input distribution or the retrieval compares two spaces.
        cloud = np.load(rec["pointcloud_uri"])["xyz"].astype(np.float32)
        pc_vec = backbone.encode_pc(torch.from_numpy(prepare_depth_shell(cloud)))

    layout = None
    if not drop_layout:
        keep, pos, edge_index, edge_attr, edge_missing = build_context_graph(
            graph, target_index, data.edge_dim, data.sem_cache, data.text_map)
        # [P1] `keep` alone, NOT `keep and edges > 0`. A context of one object,
        # or of several with no cached relation between them, is still a
        # context: with E=0 every message sum is empty, the layers reduce to the
        # identity, and Pooling({h^(0)}) is a real -- if weak -- layout vector.
        # Requiring an edge silently converted "sparse scene" into "no scene",
        # which is the layout-free case Table 1 evaluates separately.
        if keep:
            node_feat = torch.stack([
                torch.from_numpy(data.node_vectors[str(n["asset_id"])]) for n in keep
            ]).to(device)
            layout = model.query.encode_layout(
                node_feat,
                torch.from_numpy(pos).to(device),
                torch.from_numpy(edge_index).to(device),
                torch.from_numpy(edge_attr).to(device),
                edge_missing=torch.from_numpy(edge_missing).to(device))

    embeds = {"text": text, "image": image, "pc": pc_vec}
    # [2.4] the query side may drop the point cloud; here it is present because
    # target_eligibility required one. `present=None` means all three.
    return model.query(embeds, present=None, layout=layout,
                       drop_layout=None)[0]


class Stage2Data:
    """Everything the trainer reads, loaded once and passed explicitly.

    Explicit rather than module-level because the first draft of encode_query
    reached for `modalities` and `text_vectors` as free names, and Python was
    happy to compile it -- an AST scan of the function's free variables is what
    found them, not the interpreter.
    """

    def __init__(self, device: str) -> None:
        import numpy as np

        self.modalities = {}
        for path in sorted(paths.PROCTHOR_MODALITIES.glob("*.json")):
            rec = json.loads(path.read_text())
            self.modalities[rec["asset_id"]] = rec

        node = json.loads((paths.OUTPUTS / "procthor_node_embeddings.json").read_text())
        arr = np.load(node["uri"])
        self.node_vectors = {a: v for a, v in
                             zip(arr["ids"].tolist(), arr["embeddings"])}
        self.edge_dim = int(node["embedding_dim"])

        cache = json.loads((paths.OUTPUTS / "sem_edge_cache.json").read_text())
        emb = np.load(paths.OUTPUTS / "sem_edge_embeddings.npz")
        vecs = emb["embeddings"]
        self.sem_cache = {}
        for key, entry in cache["entries"].items():
            if entry.get("degraded") or entry.get("embedding_uri") is None:
                continue
            self.sem_cache[key] = vecs[int(entry["embedding_uri"].rsplit("#", 1)[1])]

        text = json.loads((paths.OUTPUTS / "procthor_object_text.json").read_text())
        self.text_map = {a: rec["text"] for a, rec in text.items()}
        self.text_map["_meta"] = {
            "prompt_version": cache["prompt_version"],
            "llm_model": cache["llm_model"],
            "text_encoder_version": cache["text_encoder_version"],
        }

        # [U-30] The missing-edge token lives on ESSGNN as an nn.Parameter and
        # is genuinely learned; it used to be a seeded vector here, which the
        # protocol already called `learned_missing_token` and which got no
        # gradient, entered no optimizer and reached no checkpoint.
        # build_context_graph now emits a bool mask instead.

    def graphs_for(self, house_ids) -> dict:
        return {h: json.loads((paths.SCENE_GRAPHS / f"{h}.json").read_text())
                for h in house_ids
                if (paths.SCENE_GRAPHS / f"{h}.json").exists()}


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
    # `backbone` is a ULIPBackbone, which has no `.parameters()` -- the guard
    # this replaces was `for p in backbone.parameters() if hasattr(...)`, and
    # since the attribute does not exist the loop iterated over nothing. It read
    # as a freeze and was a no-op. Harmless only while the backbone was built
    # with train_scope="fuser_only"; the moment it was built trainable -- which
    # it must be, to receive Stage 1's point encoder -- the paper's "only the
    # query-side fusion layer and the ESSGNN module are updated" was violated
    # silently, and the query point encoder drifted away from the frozen gallery
    # index for the whole run.
    backbone.set_train_scope("fuser_only")
    if not backbone.is_frozen():
        raise RuntimeError(
            "the ULIP backbone is still trainable or in train() mode; 2.6 says "
            "only the query fuser and the ESSGNN are updated in Stage 2")
    for name, p in model.named_parameters():
        trains = name.startswith("query.fusion") or name.startswith("query.layout_encoder") \
            or name.endswith("layout_weight")
        p.requires_grad_(trains)
    return {name: p.requires_grad for name, p in model.named_parameters()}


def build_stage2_model(encoding: dict, training: dict, hyperparameters: dict,
                       arch_proto: dict, edge_proto: dict):
    """[P0-3] The Stage 2 dual tower -- WITH the ESSGNN branch.

    Stage 1's ``build_model`` sets ``use_layout=False``, which is right there:
    2.6 puts the layout encoder in Stage 2, and building it during Stage 1 would
    put an untrained ESSGNN in the optimizer and in the checkpoint. Reusing that
    constructor here left ``query.layout_encoder`` as None while ``encode_query``
    called ``encode_layout``, and Eq. 6's lambda did not exist at all -- so
    "Stage 2" would have been Stage 1 with a different loss, or a crash,
    depending on which line ran first.

    The ESSGNN config comes from the resolved ``essgnn_arch_protocol`` via
    ``ESSGNNConfig.from_protocol``; hand-writing one here is what lets a run
    drift from the architecture G6 approved.
    """
    from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
    from metafind.models.essgnn import ESSGNNConfig
    from metafind.models.fusion import FusionConfig
    from metafind.models.ulip_backbone import EMBED_DIM

    zero_pad = encoding["missing_modality_representation"] == "zero_pad"
    fusion = FusionConfig(kind=training["fusion"], dim=EMBED_DIM, zero_pad=zero_pad)
    essgnn = ESSGNNConfig.from_protocol(
        arch_proto,
        # U-20: the t_i encoder is unstated by the paper (C's S6), so its width
        # is read from what n08 actually wrote rather than assumed.
        node_feat_dim=int(edge_proto["node_embedding_dim"]),
        edge_feat_dim=int(edge_proto["edge_embedding_dim"]),
        # Eq. 6 ADDS e_layout to the fused query, so this is not a free choice.
        out_dim=EMBED_DIM,
    )
    return MetaFindDualTower(DualTowerConfig(
        dim=EMBED_DIM, tower_sharing=training["tower_sharing"],
        query_fusion=fusion, gallery_fusion=fusion,
        use_layout=True, essgnn=essgnn,
        init_lambda=float(hyperparameters["values"].get("init_lambda", 1.0))))


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
    from metafind.models.ulip_backbone import (
        BackboneConfig, ULIPBackbone, prepare_depth_shell)

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

    data = Stage2Data(args.device)
    eligible = set(positive_map) & set(id_to_row) & set(data.modalities)
    samples = enumerate_samples(train_houses, eligible)
    if not samples:
        print("no eligible sample; check stage2_positive_map and the gallery index",
              flush=True)
        return 2

    seed = values["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # [P0-1] point_encoder_and_fuser so the checkpoint's fine-tuned PointBERT
    # has somewhere to land; freeze_for_stage2 turns it off afterwards.
    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    # [P0-3] build_stage2_model, NOT Stage 1's build_model. Stage 1 builds with
    # use_layout=False -- correct there, since 2.6 puts ESSGNN in Stage 2 -- so
    # reusing it here left query.layout_encoder as None while encode_query below
    # calls encode_layout, which raises. Eq. 6's lambda was absent too.
    model = build_stage2_model(encoding, training, hyperparameters,
                               arch_proto, edge_proto)
    loss_fn = MetaFindContrastiveLoss(ContrastiveConfig(
        # [Eq. 7/8] symmetric, unlike Stage 1's Eq. 5
        bidirectional=True,
        learnable_temperature=values["learnable_temperature"],
        init_temperature=values["init_temperature"],
        max_logit_scale=values["max_logit_scale"]))

    # Restore into all three, INCLUDING loss_fn: Stage 1's learned temperature
    # carries into Stage 2 as its initialisation. The paper says nothing about
    # tau across stages, so this is [IMPLEMENTATION CHOICE] -- but it is the one
    # that makes Stage 2 a fine-tune rather than a restart, and starting from
    # 0.07 again would discard a value Stage 1 spent its whole run learning.
    # The ESSGNN and lambda are new here and correctly absent from the
    # checkpoint; load_stage1_checkpoint checks coverage per module, so their
    # absence from a Stage 1 file is not mistaken for a dropped tensor.
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]),
                           new_prefixes=("query.layout_encoder", "layout_weight"))
    model.to(args.device)
    loss_fn.to(args.device)
    grads = freeze_for_stage2(model, backbone)

    params = [p for p in list(model.parameters()) + list(loss_fn.parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=values["learning_rate"],
                            weight_decay=values["weight_decay"])

    graphs = data.graphs_for({h for h, _, _ in samples})
    # [P1] NOT values["p_mask"]. Both rates are 30% in the paper, which is
    # exactly what made the alias invisible -- but they are different
    # mechanisms: 2.6 masks each MODALITY independently at 30% in Stage 1, and
    # drops the layout vector in 30% of BATCHES in Stage 2. Table 3 sweeps
    # p_mask; under the alias, a p_mask=0.10 row would silently have moved
    # scene dropout to 10% too, and the ablation would no longer be an ablation
    # of one thing.
    stage2_dropout = float(values.get("scene_dropout", PAPER_SCENE_DROPOUT))
    epochs = args.epochs or values["epochs"]
    # Batching is redrawn per epoch, inside the loop. Computing it once here
    # meant every epoch saw the identical partition, so a given sample met the
    # same negatives every time -- with 1,467 assets and a unique-positive
    # constraint, that is a much smaller set of contrasts than the corpus
    # supports. `rng` is seeded, so the sequence of partitions is still
    # reproducible; it is just no longer constant.
    print(f"{len(samples):,} samples over {len(train_houses):,} houses, "
          f"{sum(grads.values()):,} trainable tensors", flush=True)

    started, step = time.time(), 0
    with runlog.run_progress(NODE):
        for epoch in range(epochs):
            model.train()
            batches = unique_positive_batches(samples, values["batch_size"], rng)
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
                                     asset_id, drop, args.device, data)
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
