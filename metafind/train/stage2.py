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
import pathlib
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

# [AUDIT 2026-09-04 F3] How the Stage 2 QUERY is built, per sample:
#   none       the target's text, image and point cloud all present (what every
#              Stage 2 run before 2026-09-04 did). Pilot 2 measured that the
#              layout term then has nothing to add: S2-on = S2-off, lambda
#              does not move -- the answer is already in the query.
#   text_only  text present, image and pc absent (mask tokens). Figure 1's
#              query is a text box plus the scene graph; sec. 2.4 "It accepts
#              any subset of modalities and can be augmented with a layout-aware
#              vector".
#   stage1     sec. 2.6's independent 30% masking continued into Stage 2, at
#              least one modality kept. The paper does not say Stage 2 keeps it
#              (Kyzen's item 33); an ablation.
# The paper defines none of these for Stage 2 training. IMPLEMENTATION CHOICE,
# recorded in the checkpoint as query_modality_masking.
QUERY_MASKING_MODES = ("none", "text_only", "stage1")


def query_present(mode: str, rng, p_mask: float = 0.30):
    """A (1, 3) bool presence mask for one query, or None under `none`."""
    import torch
    if mode == "none":
        return None
    if mode == "text_only":
        return torch.tensor([[True, False, False]])
    keep = rng.random(3) >= p_mask
    if not keep.any():
        keep[int(rng.integers(3))] = True
    return torch.tensor(keep.reshape(1, 3))

CKPT_DIR = paths.CHECKPOINTS
VARIANT_STATUS = paths.OUTPUTS / "variant_status.json"
VARIANT_CKPTS = paths.OUTPUTS / "variant_ckpts.json"


def verify_recorded_artifact(record: dict, label: str,
                             rebuild_hint: str) -> Path:
    """Return the recorded path only after its bytes match the recorded SHA.

    Stage 2 previously performed this check inline only for its gallery index,
    which made the scientific seam difficult to exercise without constructing
    the whole training entry point.  Keeping the check as a pure helper gives
    wrong-bytes, missing-file, and missing-digest cases direct negative tests.
    """
    uri = record.get("uri")
    if not uri:
        raise ValueError(f"the {label} record carries no uri. {rebuild_hint}")
    artifact = Path(uri)
    if not artifact.exists():
        raise FileNotFoundError(
            f"the {label} record names {artifact}, which does not exist. "
            f"{rebuild_hint}")
    claimed = record.get("sha256")
    if claimed is None:
        raise ValueError(
            f"{artifact}'s record carries no sha256 and cannot be verified. "
            f"{rebuild_hint}")
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual != claimed:
        raise ValueError(
            f"{artifact} hashes to {actual[:16]}... but its record claims "
            f"{claimed[:16]}.... The {label} has changed since it was recorded; "
            f"{rebuild_hint}")
    return artifact


def load_stage2_protocols() -> tuple[dict, dict, dict]:
    def read(name: str, writer: str) -> dict:
        path = paths.OUTPUTS / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run {writer} first")
        proto = json.loads(path.read_text())
        if isinstance(proto, dict) and proto.get("status") not in (None, "resolved"):
            raise ValueError(f"{name} is {proto.get('status')!r}")
        return proto

    stage2 = read("stage2_protocol.json", "n09b_resolve_stage2_protocol")
    edge = read("essgnn_edge_protocol.json", "n09b_resolve_stage2_protocol")
    arch = read("essgnn_arch_protocol.json", "n09b_resolve_stage2_protocol")
    validate_graph_unit(stage2.get("graph_unit"))

    # These decisions are implemented as fixed code in this module and in
    # build_context_graph / ESSGNN, not read from the protocol at run time.
    # Until now the protocol was loaded and never compared, so editing it
    # changed nothing while the run went on citing it. Refuse any value the
    # code does not implement.
    from metafind.models.resolve_stage2 import ASSET_MODALITIES
    implemented = {
        # [DL-104] A protocol without `asset_modalities` predates Kyzen's ruling
        # and would default n11b and this trainer back to encoding the ProcTHOR
        # cloud with no error; refuse it, like every other field the code fixes.
        "asset_modalities": (list(ASSET_MODALITIES), stage2),
        "scene_dropout_granularity": ("batch", stage2),
        "target_removed_before_essgnn": (True, stage2),
        "batch_positive_uniqueness": (True, stage2),
        "directionality": ("symmetric", edge),
        "semantic_missing_representation": ("learned_missing_token", edge),
        "physical_relation_encoding": ("neighbourhood_only", edge),
    }
    for field, (want, proto) in implemented.items():
        if proto.get(field) != want:
            raise ValueError(
                f"protocol field {field} = {proto.get(field)!r}, but the Stage 2 "
                f"code implements only {want!r}. The run would cite a protocol it "
                "does not follow; change the code or the protocol, not neither.")

    # `status: resolved` is a CLAIM, and until now nothing checked it on the way
    # in. n09b validates what it writes, but an artifact written before the code
    # moved keeps its old contents and its old status: the file on disk said
    # `resolved` for two days while `ESSGNNConfig.from_protocol` refused it for a
    # missing `architecture_family`, and the failure would only have surfaced
    # after Stage 1 finished and Stage 2 started building a model.
    #
    # Widths are placeholders -- they come from the checkpoint and the edge
    # encoder at real construction time, and none of the fields being checked
    # here depend on them.
    from metafind.models.essgnn import ESSGNNConfig

    try:
        # The three widths here are SCAFFOLDING for a round-trip probe, not
        # values anything trains with. The live path reads them off the
        # artifacts instead: `node_feat_dim=data.node_dim` and
        # `edge_feat_dim=data.edge_dim` at the bottom of main(), sourced from
        # procthor_node_embeddings.json and sem_edge_cache.
        #
        # [U-20] Kyzen decided 2026-08-27 that both encoders are OpenCLIP
        # ViT-bigG-14, 1280-d (METAFIND_NOTEBOOK.md:937). **That decision is NOT
        # represented here and cannot be** -- `node_feat_dim` and
        # `edge_feat_dim` are not keys of essgnn_arch_protocol.json at all, so
        # there is nothing for a probe to check them against.
        # `01_GRAPH_SPEC.md:1123` item 201 required exactly that -- both widths
        # into the protocol BEFORE n13 was implemented. n13 is 680 lines. The
        # debt is unpaid, and changing these scaffolding numbers does not pay it.
        ESSGNNConfig.from_protocol(arch, node_feat_dim=1, edge_feat_dim=1, out_dim=1)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"essgnn_arch_protocol.json claims status={arch.get('status')!r} but "
            f"cannot build an ESSGNNConfig: {exc}. The artifact is stale relative "
            "to the code -- re-run `python -m metafind.models.resolve_stage2`."
        ) from exc

    return stage2, edge, arch


def validate_graph_unit(graph_unit: str) -> None:
    if graph_unit not in ("room", "house"):
        raise ValueError(f"Stage 2 requires an explicit graph_unit of 'room' or "
                         f"'house', got {graph_unit!r}")


def validate_graph_scope(graph: dict, graph_unit: str, label: str = "graph") -> None:
    """Legacy sidecars are usable only under an explicitly selected house scope."""
    validate_graph_unit(graph_unit)
    actual = graph.get("graph_unit")
    legacy_house = (actual is None and graph.get("builder_version", 1) == 1)
    if actual != graph_unit and not (graph_unit == "house" and legacy_house):
        raise ValueError(f"{label} graph_unit={actual!r} disagrees with Stage 2 "
                         f"graph_unit={graph_unit!r}; rebuild the graph inputs or "
                         "explicitly select the legacy house protocol")


_STAGE2_INPUT_FILES = {
    "stage2": "stage2_protocol.json", "edge": "essgnn_edge_protocol.json",
    "arch": "essgnn_arch_protocol.json", "scene_splits": "scene_splits.json",
    "positive_map": "stage2_positive_map.json", "object_text": "procthor_object_text.json",
    "node_record": "procthor_node_embeddings.json", "semantic_cache": "sem_edge_cache.json",
    "semantic_embeddings": "sem_edge_embeddings.npz", "gallery_record": "stage2_gallery_index.json",
}


def capture_stage2_input_identity(stage2: dict, edge: dict, arch: dict,
                                  train_houses: list[str]) -> dict:
    """Bind the input bytes before loading arrays; verify again after loading.

    Paths identify replay sources, while digests detect replacements at the same
    paths. Ordered house IDs preserve the split prefix selected by --limit-houses.
    Protocol snapshots allow consumers to reconstruct the run without substituting
    whatever protocols happen to be in their current environment.
    """
    def identity(path):
        path = Path(path).absolute()
        return {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    protocol_names = {"stage2": "stage2_protocol.json",
                      "edge": "essgnn_edge_protocol.json",
                      "arch": "essgnn_arch_protocol.json"}
    protocols = {"stage2": stage2, "edge": edge, "arch": arch}
    for name, filename in protocol_names.items():
        if json.loads((paths.OUTPUTS / filename).read_text()) != protocols[name]:
            raise ValueError(f"{filename} changed after the protocol was loaded")
    files = {name: paths.OUTPUTS / filename for name, filename in _STAGE2_INPUT_FILES.items()}
    node = json.loads(files["node_record"].read_text())
    files["node_embeddings"] = Path(node["uri"])
    split_houses = json.loads(files["scene_splits"].read_text())["train_houses"]
    if list(train_houses) != split_houses[:len(train_houses)]:
        raise ValueError("scene_splits changed after the training house prefix was selected")
    return {"version": 1, "graph_unit": stage2["graph_unit"],
            "train_houses": list(train_houses),
            "protocols": json.loads(json.dumps(protocols)),
            "artifacts": {name: identity(path) for name, path in files.items()},
            "graphs": {house: identity(paths.SCENE_GRAPHS / f"{house}.json")
                       for house in train_houses},
            "modality_records_root": str(paths.PROCTHOR_MODALITIES.absolute()),
            "modality_records": {path.stem: identity(path)
                                 for path in sorted(paths.PROCTHOR_MODALITIES.glob("*.json"))}}


def verify_stage2_input_identity(record: dict, *, current_inputs: bool = False) -> dict:
    """Validate a new Stage 2 record's replay inputs; legacy records must opt out.

    This is also called between loading the training inputs and the first forward,
    so a concurrently rewritten cache cannot be saved under a stale identity.
    """
    identity = record.get("input_identity")
    if not isinstance(identity, dict) or identity.get("version") != 1:
        raise ValueError("Stage 2 checkpoint has no supported input_identity; "
                         "legacy replay must be selected explicitly")
    validate_graph_unit(identity.get("graph_unit"))
    if current_inputs:
        # Stage2Data reads paths.OUTPUTS, not the archived paths in this record.
        # Verifying an intact old data root must not authorize loading a new one.
        expected = {name: paths.OUTPUTS / filename
                    for name, filename in _STAGE2_INPUT_FILES.items()}
        expected["node_embeddings"] = Path(json.loads(
            (paths.OUTPUTS / "procthor_node_embeddings.json").read_text())["uri"])
        for name, path in expected.items():
            if Path(identity["artifacts"][name]["uri"]).resolve() != path.resolve():
                raise ValueError(f"Stage 2 current input path differs from its record: {name}")
        for house, artifact in identity["graphs"].items():
            if Path(artifact["uri"]).resolve() != (paths.SCENE_GRAPHS / f"{house}.json").resolve():
                raise ValueError(f"Stage 2 current graph path differs from its record: {house}")
        if Path(identity["modality_records_root"]).resolve() != paths.PROCTHOR_MODALITIES.resolve():
            raise ValueError("Stage 2 current modality record root differs from its record")
    if set(identity["graphs"]) != set(identity["train_houses"]):
        raise ValueError("Stage 2 input_identity does not cover its train_houses")
    current_modalities = {path.stem for path in
                          Path(identity["modality_records_root"]).glob("*.json")}
    if current_modalities != set(identity["modality_records"]):
        raise ValueError("Stage 2 modality record membership changed since it was recorded")
    for group in ("artifacts", "graphs", "modality_records"):
        for name, artifact in identity[group].items():
            verify_recorded_artifact(artifact, f"Stage 2 {group}/{name}",
                                     "Restore the recorded input bytes before replay.")
    for name, protocol in identity["protocols"].items():
        path = Path(identity["artifacts"][name]["uri"])
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f"Stage 2 {name} protocol snapshot disagrees with its artifact")
    if identity["protocols"]["stage2"]["graph_unit"] != identity["graph_unit"]:
        raise ValueError("Stage 2 input_identity graph_unit disagrees with its protocol")
    return identity


def enumerate_samples(train_houses: list[str], eligible: set[str], *,
                      graph_unit: str = "room") -> list[tuple[str, int, str]]:
    """[U-08d] Every eligible instance of every train house, enumerated ONCE.

    Returns ``(house_id, node_index, asset_id)``. The enumeration is written
    down rather than drawn per epoch because ``instance_resampling`` is fixed:
    a rerun at the same seed must train on the same tuples, and "the same seed"
    means nothing if the sample set itself is redrawn.
    """
    samples = []
    for house_id in train_houses:
        path = paths.SCENE_GRAPHS / f"{house_id}.json"
        graph = json.loads(path.read_text())
        validate_graph_scope(graph, graph_unit, str(path))
        for node in graph["nodes"]:
            asset_id = str(node["asset_id"])
            if asset_id in eligible:
                samples.append((house_id, node["index"], asset_id))
    return samples


# [MEASURED 2026-09-04, first real Stage 2 run] unique_positive_batches on
# 99,945 samples over 1,500 houses gave 1,529 full batches and then 374 with
# fewer than 14 samples -- 69 of size 1, 205 of size 2 -- all drawn from the
# two most-placed assets (Cellphone_6 x200, Pencil_6 x131), because once every
# other asset is spent a batch can hold at most one of each survivor. A batch
# of one has InfoNCE loss exactly 0 (one class), a batch of two has one
# negative; the log showed 'loss 0.0000' for the last 300 steps and the
# optimizer still stepped on them. Eq. 7/8 need B negatives to mean anything.
MIN_BATCH = 8


def usable_batches(batches: list[list[int]], min_size: int = MIN_BATCH):
    """Drop the degenerate tail. Returns (kept, n_dropped_batches, n_dropped_samples)."""
    kept = [b for b in batches if len(b) >= min_size]
    dropped = [b for b in batches if len(b) < min_size]
    return kept, len(dropped), sum(len(b) for b in dropped)


def training_batches(samples, batch_size: int, rng):
    """Keep the established sampler/tail policy, but require an actual update."""
    batches, dropped, dropped_samples = usable_batches(
        unique_positive_batches(samples, batch_size, rng))
    if not batches:
        raise ValueError(f"Stage 2 has no usable batches (minimum {MIN_BATCH}); "
                         f"{len(samples)} samples would produce zero optimizer steps. "
                         "Increase the sample pool or batch size before training.")
    return batches, dropped, dropped_samples


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
                        sem_cache: dict, text_map: dict, *, graph_unit: str = "room"):
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
    # [DL-103, scene_graphs BUILDER_VERSION 2] Under a room-unit graph the context is the
    # target's ROOM minus the target (paper: room-level scenes). A target whose room could
    # not be parsed gets no context, which encode_query treats as layout absent. Legacy
    # house-unit graphs require an explicitly selected house protocol.
    validate_graph_scope(graph, graph_unit)
    if graph_unit == "room":
        room = graph["nodes"][target_index].get("room_id")
        keep = [n for n in graph["nodes"]
                if n["index"] != target_index and room is not None and n.get("room_id") == room]
    else:
        keep = [n for n in graph["nodes"] if n["index"] != target_index]
    remap = {n["index"]: k for k, n in enumerate(keep)}

    pos = np.array([n["position"] for n in keep], dtype=np.float32)
    rows, cols, attrs, missing = [], [], [], []
    for i, j in graph["sem_edge_ids"]:
        if i == target_index or j == target_index or i not in remap or j not in remap:
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


def derive_init_lambda(model, samples, graphs, data, ratio: float,
                       device: str, n: int = 64,
                       arch_protocol: dict | None = None) -> dict:
    """Eq. 6's lambda_0, from the ratio Kyzen ruled and a norm measured NOW.

    [MASTER DECISION 2026-09-03, under Kyzen's delegation.] He ruled 0.1
    (DL-077 item 8). DL-078 turned that into the literal 9.0 by multiplying it
    by a fused-query norm of 91.4 measured once, through a Stage 1 checkpoint
    that is twelve-view-era and archived as non-comparable. The reasoning was
    right and the artefact was a constant welded to a retired measurement:
    after the scheduled Stage 1 retrain, 9.0 is a tenth of a norm that no
    longer exists, and nothing would notice, because the fusion output is never
    normalised before the loss and its scale is a free property of whichever
    checkpoint is loaded.

    So the ratio is stored and the product is derived, here, under `no_grad`,
    from the first `n` samples of THIS run with THIS checkpoint. The median
    rather than the mean: one asset with an outlying fused norm should not move
    the initialisation of a scalar that governs the whole layout term.

    Both numbers go into the run record. That is the point -- a reader can see
    which norm produced which lambda instead of taking a ledger entry's word.
    """
    import statistics

    import torch

    # [ULIP2 REVIEWER MINOR 1] `lam = ratio * ||fused||` is the intended tenth
    # ONLY IF `||e_layout|| == 1`, because Eq. 6 is `fused + lam * e_layout`.
    # It is 1 today, and only because pooling is `normalised_sum`
    # (`essgnn.py:661-663`, `s / (s.norm() + 1e-12)`). `essgnn.Pool` legally
    # admits mean, sum and max, and under `sum` -- approved hours before
    # normalised_sum -- the layout term measured 27x the fused query at init,
    # so this same formula would land at ~2.7x rather than a tenth. The
    # arithmetic was correct by a protocol value nothing read.
    pooling = (arch_protocol or {}).get("pooling")
    if pooling != "normalised_sum":
        raise SystemExit(
            f"init_lambda_ratio assumes a unit-norm layout vector, which is "
            f"what `normalised_sum` produces. essgnn_arch_protocol records "
            f"pooling={pooling!r}, so `ratio x ||fused||` is not a ratio of "
            "anything. Pin `init_lambda` to a literal, or derive it against a "
            "measured ||e_layout|| for this pooling.")

    # [ULIP2 REVIEWER MINOR 2] `modules_in_eval`, not a single flag plus
    # `model.query.train()`. That restore RECURSES while the state it must put
    # back is per-module -- which is the measurement that made the helper exist
    # on 2026-08-28. Safe here today only because nothing evals a query
    # submodule independently; the helper makes it safe by construction.
    from metafind.train.stage1 import modules_in_eval

    norms = []
    with modules_in_eval(model.query), torch.no_grad():
        for house, target_index, asset_id in samples[:n]:
            g = graphs.get(house)
            if g is None:
                continue
            # `drop_layout=True` so ESSGNN does not run and lambda does not
            # enter: what is wanted is ||Fusion(e_T, e_I, e_P)|| alone, the
            # thing the layout term will be a tenth OF.
            q = encode_query(model, g, target_index, asset_id,
                             drop_layout=True, device=device, data=data)
            norms.append(float(q.norm()))
    if not norms:
        raise SystemExit(
            "could not measure a single fused-query norm, so lambda_0 cannot be "
            "derived from the ratio. Pin `init_lambda` in stage2_protocol.json "
            "if a literal is intended.")
    med = statistics.median(norms)
    return {"init_lambda": ratio * med, "init_lambda_ratio": ratio,
            "fused_query_norm_median": med, "fused_query_norm_n": len(norms),
            "fused_query_norm_min": min(norms), "fused_query_norm_max": max(norms),
            # The assumption the arithmetic rests on, recorded beside the
            # result rather than left implicit.
            "pooling": pooling, "layout_norm_assumed": 1.0,
            "derived_at": "stage2_start", "basis": "median over the first "
            f"{len(norms)} samples of this run, drop_layout=True, no_grad",
            "sample_is_a_positional_prefix": True}


def initialise_layout_lambda(model, stage2: dict, samples, graphs, data,
                             device: str, *, arch_protocol: dict) -> dict | None:
    """Initialise Eq. 6 after parent restore, or record no scalar without layout."""
    import torch

    # A shared protocol can pin lambda for the Full arm. The no-layout arm
    # has no such parameter, so it must not publish an initializer for one.
    if model.query.layout_encoder is None:
        return None

    pinned = stage2.get("init_lambda")
    if pinned is None:
        record = derive_init_lambda(
            model, samples, graphs, data,
            float(stage2["init_lambda_ratio"]), device,
            arch_protocol=arch_protocol)
        print(f"lambda_0 = {record['init_lambda']:.4f} "
              f"= {record['init_lambda_ratio']} x median ||Fusion|| "
              f"{record['fused_query_norm_median']:.2f} "
              f"(n={record['fused_query_norm_n']}, range "
              f"{record['fused_query_norm_min']:.2f}"
              f"-{record['fused_query_norm_max']:.2f})", flush=True)
    else:
        record = {"init_lambda": float(pinned), "init_lambda_ratio": None,
                  "basis": "pinned literal in stage2_protocol.json; no "
                           "measurement taken"}
        print(f"lambda_0 = {pinned} (pinned literal, not derived)", flush=True)
    with torch.no_grad():
        model.query.layout_weight.fill_(float(record["init_lambda"]))
    return record


def encode_query(model, graph: dict, target_index: int, asset_id: str,
                 drop_layout: bool, device: str, data: "Stage2Data",
                 present=None):
    """One leave-one-out query: the target's modalities plus its scene context.

    The target is removed from the graph here and nowhere else, so the removal
    cannot be skipped by a caller. Leaving it in would let ESSGNN read the
    answer off its own input; the loss would fall and nothing downstream would
    distinguish that from learning.

    The target's three modality vectors are LOOKED UP, not re-encoded. Stage 2
    freezes the whole ULIP-2 backbone (paper section 2.7: only the query-side
    fusion layer and the ESSGNN are updated), so the text / image / point-cloud
    vector of a given asset is a constant for the whole run. They are read from
    the Stage 2 gallery index, which the same frozen backbone wrote; see
    ``load_asset_modality_vectors``. Re-encoding them per step -- eleven
    ViT-bigG image forwards per sample -- was what put one epoch at days.
    """
    import torch

    if data.asset_vectors is None:
        raise ValueError("Stage2Data.asset_vectors is not set; load them with "
                         "load_asset_modality_vectors(gallery_index) first")
    vec = data.asset_vectors[asset_id]
    # [DL-104] Only the DECLARED modalities have a vector (ProcTHOR: text, image).
    # An undeclared one is None here, which the query fusion reads as absent and
    # fills with its Stage-1-trained mask token -- sec. 2.6's masked embedding,
    # the same path a Stage 1 query takes when a modality is masked out.
    embeds = {m: (torch.from_numpy(vec[m]).to(device).unsqueeze(0) if m in vec else None)
              for m in ("text", "image", "pc")}
    if present is not None:
        # A caller's mask (text_only / stage1) may name a slot the asset never
        # had; clear it, and keep the text if that empties the row.
        present = present.clone()
        for i, m in enumerate(("text", "image", "pc")):
            if embeds[m] is None:
                present[:, i] = False
        empty = ~present.any(dim=1)
        first = next(i for i, m in enumerate(("text", "image", "pc")) if embeds[m] is not None)
        present[empty, first] = True

    layout = None
    # `layout_encoder is None` is the "w/o Layout Context" row, and it is a
    # different condition from `drop_layout`: scene dropout suppresses the term
    # for 30% of batches in a model that HAS one, while this variant has none at
    # all. Checking only drop_layout sent that row into encode_layout, which
    # raises on a tower built without the branch.
    if not drop_layout and model.query.layout_encoder is not None:
        keep, pos, edge_index, edge_attr, edge_missing = build_context_graph(
            graph, target_index, data.edge_dim, data.sem_cache, data.text_map,
            graph_unit=data.graph_unit)
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

    # [2.4] the query side may drop the point cloud. `present=None` means all
    # three (the pre-2026-09-04 construction); under query_modality_masking
    # text_only / stage1 the caller passes a (1, 3) mask and the absent slots
    # take the learned mask tokens, as in Stage 1 -- the target's own image and
    # cloud then no longer answer the query by themselves, which is what gives
    # Eq. 6's layout term work to do.
    return model.query(embeds, present=present, layout=layout)[0]


class Stage2Data:
    """Everything the trainer reads, loaded once and passed explicitly.

    Explicit rather than module-level because the first draft of encode_query
    reached for `modalities` and `text_vectors` as free names, and Python was
    happy to compile it -- an AST scan of the function's free variables is what
    found them, not the interpreter.
    """

    def __init__(self, device: str, *, graph_unit: str = "room",
                 allow_legacy_semantic_inputs: bool = False) -> None:
        import io
        import numpy as np
        from metafind.data.semantic_edges import relation_text_for
        from metafind.data.semantic_provenance import (read_verified_node_source,
                                                       read_verified_edge_source)
        from metafind.train.gallery_index import verified_source_bytes

        validate_graph_unit(graph_unit)
        self.graph_unit = graph_unit

        # Filled by main from the Stage 2 gallery index: asset_id -> the frozen
        # backbone's text / image / pc vectors. See load_asset_modality_vectors.
        self.asset_vectors: dict[str, dict[str, np.ndarray]] | None = None
        self.modalities = {}
        for path in sorted(paths.PROCTHOR_MODALITIES.glob("*.json")):
            rec = json.loads(path.read_text())
            self.modalities[rec["asset_id"]] = rec

        node = json.loads((paths.OUTPUTS / "procthor_node_embeddings.json").read_text())
        # Consume exactly the hashed bytes: reopening the path after a digest
        # check could load replaced vectors under the earlier file's identity.
        if not node.get("uri") or not node.get("sha256"):
            raise ValueError("node embedding record requires uri and sha256; re-run n08")
        with np.load(io.BytesIO(verified_source_bytes(node)), allow_pickle=False) as stored:
            arr = dict(stored)
        text = json.loads((paths.OUTPUTS / "procthor_object_text.json").read_text())
        node_source = read_verified_node_source(
            node, text, arr, allow_legacy=allow_legacy_semantic_inputs)
        self.node_vectors = {a: v for a, v in
                             zip(arr["ids"].tolist(), arr["embeddings"])}
        # t_i's width and e_ij's width are read from the two artifacts SEPARATELY
        # and checked against the arrays. `self.edge_dim` used to be assigned
        # from the NODE record's `embedding_dim` -- the two happen to come from
        # the same text encoder today, so it worked, and it would have gone on
        # working right up until someone changed one of them. The paper fixes
        # neither: t_i's encoder is unstated (U-20 / C's S6) and e_ij's is only
        # "e.g., CLIP or BERT" (U-06).
        self.node_dim = int(node["embedding_dim"])
        if arr["embeddings"].shape[1] != self.node_dim:
            raise ValueError(
                f"procthor_node_embeddings declares {self.node_dim}-d but the "
                f"array is {arr['embeddings'].shape[1]}-d")

        cache = json.loads((paths.OUTPUTS / "sem_edge_cache.json").read_text())
        edge_path = paths.OUTPUTS / "sem_edge_embeddings.npz"
        if cache.get("source_identity") is not None:
            artifact = cache.get("embedding_artifact")
            if not isinstance(artifact, dict) or not artifact.get("uri") or not artifact.get("sha256") \
                    or Path(artifact["uri"]).resolve() != edge_path.resolve():
                raise ValueError("semantic edge embedding artifact does not identify the loaded file")
            edge_bytes = verified_source_bytes(artifact)
        else:
            # Legacy source policy stays explicit below; reading once does not
            # manufacture a source digest for an unbound historical artifact.
            edge_bytes = edge_path.read_bytes()
        with np.load(io.BytesIO(edge_bytes), allow_pickle=False) as stored:
            emb = dict(stored)
        edge_source = read_verified_edge_source(
            cache, emb, allow_legacy=allow_legacy_semantic_inputs)
        if node_source["status"] == "verified" and edge_source["status"] == "verified" \
                and node_source["encoder_identity"] != edge_source["encoder_identity"]:
            raise ValueError("node and edge semantic caches use different text encoders")
        self.semantic_source_status = {
            "node": node_source["status"], "edge": edge_source["status"],
            "encoder_match": ("verified" if node_source["status"] == edge_source["status"] == "verified"
                              else "legacy_unbound"),
        }
        self.semantic_encoder_identity = (node_source["encoder_identity"]
                                         if self.semantic_source_status["encoder_match"] == "verified"
                                         else None)
        vecs = emb["embeddings"]
        self.edge_dim = int(cache["edge_dim"])
        if vecs.shape[1] != self.edge_dim:
            raise ValueError(
                f"sem_edge_cache declares edge_dim {self.edge_dim} but "
                f"sem_edge_embeddings is {vecs.shape[1]}-d")
        self.sem_cache = {}
        # The uri's "#<row>" is a positional pointer into the npz. The npz
        # also stores the key of every row; compare them, so a cache written
        # against a different embedding file (a crash between the two writes
        # in the semantic-edge run) is caught here rather than served.
        emb_keys = emb["keys"].tolist() if "keys" in emb else None
        for key, entry in cache["entries"].items():
            if entry.get("degraded") or entry.get("embedding_uri") is None:
                continue
            row = int(entry["embedding_uri"].rsplit("#", 1)[1])
            if emb_keys is not None and emb_keys[row] != key:
                raise ValueError(
                    f"sem_edge_cache entry {key[:16]}... points at row {row}, "
                    f"which sem_edge_embeddings labels {emb_keys[row][:16]}.... "
                    "The cache and the embeddings are from different runs.")
            self.sem_cache[key] = vecs[row]

        # Relation cache descriptions and node-encoder text are separate inputs.
        # Node vectors above retain rec["text"]; cache lookup follows its producer.
        self.text_map = {a: relation_text_for(rec) for a, rec in text.items()}
        self.text_map["_meta"] = {
            "prompt_version": cache["prompt_version"],
            "llm_model": cache["llm_model"],
            "text_encoder_version": cache["text_encoder_version"],
        }

        self._assert_text_and_edges_agree()

        # [U-30] The missing-edge token lives on ESSGNN as an nn.Parameter and
        # is genuinely learned; it used to be a seeded vector here, which the
        # protocol already called `learned_missing_token` and which got no
        # gradient, entered no optimizer and reached no checkpoint.
        # build_context_graph now emits a bool mask instead.

    def _assert_text_and_edges_agree(self, n_houses: int = 600,
                                     tolerance: float = 0.001) -> None:
        """Refuse a corpus whose node text and semantic edges came from
        different generations of the text rule.

        [MASTER DECISION 2026-09-03, under Kyzen's delegation of the four open
        items. The DECISION was to DEFER the repair; this is what makes
        deferring safe.]

        `_edge_key` hashes the DESCRIPTIONS, so regenerating
        `procthor_object_text.json` without re-running the edge stage in the
        same operation changes 146 assetIds' keys, the lookup misses, `vec is
        None`, `edge_missing = True`, and the learned missing token substitutes.
        Nothing raises. Training proceeds. Table 2 and Table 3 are wrong.
        Measured over 200 houses and 66,603 edges: 11.56% of edges would go
        missing that way.

        Why the repair is deferred rather than run. `REPRODUCTION_PROTOCOL_20260903`
        §十三 says LLM semantic-edge generation waits until the Stage 2 protocol
        is complete, and 問題 10 -- what `t_i` should be built from at all --
        is still UNRESOLVED. Repairing the CURRENT rule now would spend the GPU
        on a rule that may not survive that answer, and nothing between here and
        PHASE 6 reads the node text. `tools/repair_procthor_node_text.py` is
        written, dry-run verified, and refuses to apply.

        Why a check and not a note. "Remember to run them together" is the class
        of instruction that fails exactly once and silently. This measures the
        real miss rate over a sample of real houses, and a run whose text and
        edges have drifted apart cannot start. Today it measures 0.00%.
        """
        import glob
        import random as _random

        files = sorted(glob.glob(str(paths.SCENE_GRAPHS / "*.json")))
        if not files:
            raise SystemExit(
                f"no scene graphs under {paths.SCENE_GRAPHS}, so the text-vs-edge "
                "agreement cannot be checked. A guard that skips itself when its "
                "input is missing is not a guard. Run n07 first.")
        # [ULIP2 REVIEWER MINOR 3] Three things changed together, and the
        # reviewer's answer to "can a run pass the sample and still be broken"
        # was yes -- bounded inside the sample, unbounded outside it.
        #
        #   * NOT a fixed seed. `Random(0)` drew the SAME 1.4% of houses on
        #     every run, so drift confined to the other 98.6% was invisible
        #     PERMANENTLY rather than probabilistically. An unseeded draw makes
        #     it a real sample: the chance of missing it falls with every run
        #     instead of staying at one.
        #   * 600 houses, not 200. 5.0% of the 12,000, ~200k edges.
        #   * tolerance 0.001, not 0.01. 1% was a round number ten times the
        #     measured baseline; the only legitimate miss source is the
        #     degraded / null-uri entries dropped from `sem_cache` above, and
        #     that rate is 0.00% today. 0.1% still clears the realistic failure
        #     -- a global text regeneration measures 11.56% -- by 100x.
        #
        # `if not files: return` was the same defect one level up: the guard
        # skipped itself exactly when it had nothing to check.
        sample = _random.Random().sample(files, min(n_houses, len(files)))
        asked = missing = 0
        for f in sample:
            g = json.loads(pathlib.Path(f).read_text())
            by_index = {n["index"]: n["asset_id"] for n in g["nodes"]}
            for i, j in g.get("sem_edge_ids") or []:
                a, b = by_index.get(i), by_index.get(j)
                if a is None or b is None:
                    continue
                asked += 1
                if _edge_key(a, b, self.text_map) not in self.sem_cache:
                    missing += 1
        if not asked:
            raise SystemExit(
                f"the {len(sample)} sampled scene graphs carry no sem_edge_ids "
                "at all, so text/edge agreement could not be measured. This is "
                "the twin of the empty-corpus case above and reachable the same "
                "way: a partial or aborted n07 edge stage -- which is exactly "
                "the two-stages-run-separately hazard this guard exists for. "
                "Re-run n07, or restore the scene graphs the edges were built "
                "for.")
        rate = missing / asked
        print(f"  text/edge agreement: {missing:,} of {asked:,} sampled edges "
              f"missing ({rate:.3%}) over {len(sample)} houses", flush=True)
        if rate > tolerance:
            raise SystemExit(
                f"{missing:,} of {asked:,} semantic edges ({rate:.2%}) sampled "
                f"over {len(sample)} houses have no cache entry.\n"
                "TWO things produce this and the remedies differ:\n"
                "  1. The node text and the semantic edges are from different "
                "generations of the text rule. `_edge_key` hashes the "
                "descriptions, so a text regenerated without re-running the "
                "edge stage silently substitutes the learned missing token and "
                "Table 2/3 come out wrong with nothing raised. Repair them "
                "TOGETHER with `tools/repair_procthor_node_text.py --apply`, or "
                "restore the text the cache was built under.\n"
                "  2. The cache legitimately lacks those entries -- degraded or "
                "null-uri rows are dropped from `sem_cache` when it loads. That "
                "rate was 0.00% when this threshold was set; if it is no longer, "
                "the edges need regenerating, not the text.\n"
                "Check which by looking at the dropped-entry count in this "
                "run's own load, above.")

    def graphs_for(self, house_ids) -> dict:
        graphs = {}
        for house in house_ids:
            path = paths.SCENE_GRAPHS / f"{house}.json"
            graph = json.loads(path.read_text())
            validate_graph_scope(graph, self.graph_unit, str(path))
            graphs[house] = graph
        return graphs


def load_asset_modality_vectors(gallery_index, declared=("text", "image", "pc"),
                                ) -> dict[str, dict[str, np.ndarray]]:
    """The frozen backbone's text / image / point-cloud vector for every gallery asset.

    Read from the Stage 2 gallery index, which stores them beside the fused
    gallery vector (see ``gallery_index.build_index``'s ``extra``). They come
    from the same frozen ULIP-2 backbone, the same inputs and the same eval
    mode as the fused vector, so for the query side they are exactly the
    numbers ``encode_query`` used to recompute per step.

    An index built before the arrays were stored is refused rather than
    silently re-encoded: the only encoder that could re-encode is the one this
    function exists to stop running per step, and a refusal names the fix
    (rebuild the index) instead of hiding a days-long slowdown.
    """
    declared = tuple(declared)
    missing = [k for k in declared if k not in gallery_index]
    if missing:
        raise ValueError(
            f"the Stage 2 gallery index has no raw modality arrays {missing}. "
            "It was built before they were stored, or under a different "
            "`asset_modalities`; rebuild it with "
            "`python -m metafind.train.gallery_index stage2 ...`.")
    ids = gallery_index["ids"].tolist()
    return {a: {m: gallery_index[m][i] for m in declared}
            for i, a in enumerate(ids)}


def trainable_state_dict(model) -> dict:
    """[L1-CKPT-TRAINABLE-ONLY] Same rule as Stage 1, and it matters more here:
    Stage 2 has ELEVEN variants, so a whole-state_dict save is 112 GB."""
    return {name: p.detach().cpu()
            for name, p in model.named_parameters() if p.requires_grad}


def stage2_checkpoint_payload(model, loss_fn, record: dict) -> dict:
    """Bind reconstruction and input provenance to the checkpoint's own bytes."""
    # The checkpoint may be archived without changing its bytes. Its location
    # and the digest/size of the finished payload belong only in the sidecar.
    metadata = {key: value for key, value in record.items()
                if key not in {"uri", "sha256", "size_bytes"}}
    return {"trainable_state": trainable_state_dict(model),
            "loss_trainable_state": trainable_state_dict(loss_fn),
            "trainer_version": TRAINER_VERSION,
            "variant": record["variant_id"],
            "metadata": json.loads(json.dumps(metadata))}


def freeze_for_stage2(model, backbone, query_modality_masking: str = "none",
                      asset_modalities=("text", "image", "pc")) -> dict:
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
        # With no modality masking in Stage 2 every query slot is always
        # present, the fusion's mask tokens are never selected, and their
        # gradient is identically zero every step. A parameter that cannot
        # learn must not sit in the optimizer, where weight decay would erode
        # the values Stage 1 learned for it (measured: zero grad, nonzero
        # delta per step). It is restored from Stage 1 and left alone.
        # [DL-104] With an undeclared modality (ProcTHOR has no point cloud) that
        # slot is absent in EVERY query, its mask token is selected every step,
        # and the token is part of the fusion layer the paper says Stage 2 trains.
        if query_modality_masking == "none" and len(asset_modalities) == 3 \
                and name.endswith("fusion.mask_tokens"):
            trains = False
        p.requires_grad_(trains)
    return {name: p.requires_grad for name, p in model.named_parameters()}


def load_variant(variant_id: str, ckpt_record: dict, *, training: dict | None = None,
                 encoding: dict | None = None, values: dict | None = None) -> dict:
    """Resolve a Table 3 row into settings, and refuse the ones we cannot run.

    `--variant` used to reach only the checkpoint FILENAME. Every row trained
    and saved happily as `stage2_<id>.pt`, and every one of them was the full
    model -- ten files, ten different names, identical architectures, and an
    ablation table whose rows would have differed only by training noise. There
    is no error to notice; the table simply would not have meant anything.

    Two classes of row have to be separated, because only one of them is ours:

    * Stage 2 fields -- `layout_encoder` -- are applied here.
    * Stage 1 fields -- `train_scope`, `dropout`, `fusion`,
      `missing_modality_representation` -- were baked into the checkpoint by
      n10. This function CHECKS them against the loaded checkpoint rather than
      applying them, because applying them here would produce a model whose
      towers were trained under one setting and fine-tuned under another.
    """
    path = paths.OUTPUTS / "variant_registry.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run n05b_resolve_stage1_encoding first")
    registry = {v["variant_id"]: v for v in json.loads(path.read_text())}
    if variant_id not in registry:
        raise ValueError(
            f"unknown variant {variant_id!r}; the registry holds "
            f"{sorted(registry)}")
    variant = registry[variant_id]

    # [L1-ABLATION-INFERENCE-ONLY] "w/o iterative retrieval" is not a model.
    # Algorithm 1 runs at inference (2.7), so that row is the Full checkpoint
    # evaluated with composition_mode=parallel. Training it would answer a
    # question Table 3 is not asking.
    if not variant["requires_training"]:
        raise ValueError(
            f"variant {variant_id!r} has requires_training=False and reuses "
            f"{variant['reuses_ckpt']!r}; it is an INFERENCE setting "
            f"(composition_mode={variant['composition_mode']!r}), not a run")

    if variant["layout_encoder"] == "gat":
        raise NotImplementedError(
            "the 'w/ Layout Context (GAT)' row needs a GAT layout encoder, and "
            "only ESSGNN is implemented. Running it with ESSGNN would put the "
            "wrong label on a real result.")

    # Stage 1 fields: verify, do not apply.
    parent_scope = (training or {}).get("train_scope", ckpt_record.get("train_scope"))
    if variant["train_scope"] and variant["train_scope"] != parent_scope:
        raise ValueError(
            f"variant {variant_id!r} needs a Stage 1 checkpoint trained with "
            f"train_scope={variant['train_scope']!r}, but stage1_ckpt records "
            f"{parent_scope!r}. Re-run n10 for this variant.")
    # The other three Stage 1 fields were named in the docstring as checked
    # and were not: a `fusion_mean` run built the transformer fusion from the
    # Stage 1 protocol, restored the transformer checkpoint, trained, and
    # saved the result as stage2_fusion_mean.pt. Same failure the paragraph
    # above describes, one field over. Each is compared against the artifact
    # that actually decides it.
    want_fusion = variant.get("fusion") or (training or {}).get("fusion")
    if training is not None and want_fusion != training["fusion"]:
        raise ValueError(
            f"variant {variant_id!r} needs fusion={want_fusion!r} but the Stage 1 "
            f"protocol (and therefore the checkpoint) is {training['fusion']!r}. "
            "Train Stage 1 for this variant first.")
    if values is not None and variant.get("dropout") is not None \
            and abs(float(variant["dropout"]) - float(values["p_mask"])) > 1e-9:
        raise ValueError(
            f"variant {variant_id!r} needs p_mask={variant['dropout']} but the "
            f"Stage 1 recipe records p_mask={values['p_mask']}. Train Stage 1 "
            "for this variant first.")
    want_missing = variant.get("missing_modality_representation", "learned_token")
    if encoding is not None and want_missing != encoding["missing_modality_representation"]:
        raise ValueError(
            f"variant {variant_id!r} needs missing_modality_representation="
            f"{want_missing!r} but the encoding protocol is "
            f"{encoding['missing_modality_representation']!r}. Train Stage 1 "
            "for this variant first.")
    return variant


def build_stage2_model(encoding: dict, training: dict, hyperparameters: dict,
                       arch_proto: dict, *, node_feat_dim: int, edge_feat_dim: int,
                       use_layout: bool = True, init_lambda: float = 1.0):
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
    from metafind.models.ulip_backbone import EMBED_DIM
    from metafind.train.stage1 import fusion_config_for

    # [BUG FIX 2026-09-04] The Stage 2 tower must be the Stage 1 tower plus the
    # layout branch -- same FusionConfig, or the Stage 1 weights are loaded into
    # a fusion that pre-processes its inputs differently. This built
    # FusionConfig(kind, dim, zero_pad) only, so a parent trained with
    # prefusion_norm=True (P1) had its fusion fed raw vectors here (pc norm 139
    # against text 37): the first Stage 2 pilot started at loss 2.65 instead of
    # the parent's ~0.8, never improved, and moved the fusion by a median
    # relative distance of 0.55 trying to cope -- and the evaluator's Stage 2
    # path, which calls this builder, scored that head the same wrong way
    # (protocol C pc-only 17.0 against the parent's 86.1). Mirror
    # stage1.build_model exactly.
    fusion = fusion_config_for(encoding, training)
    gallery_fusion = fusion_config_for(encoding, training, gallery=True)
    essgnn = ESSGNNConfig.from_protocol(
        arch_proto,
        # MEASURED from n08's artifacts, NOT read from a protocol. An earlier
        # version took these from `essgnn_edge_protocol["node_embedding_dim"]`
        # and `["edge_embedding_dim"]`, which nothing writes -- n09b's
        # EDGE_DECISIONS holds topology / physical_relation_encoding /
        # semantic_missing_representation / directionality and no widths. The
        # first real n13 run would have raised KeyError before building
        # anything, and no test reached this path because n13 has never run.
        #
        # A protocol is the right home for a DECISION. A width is a measurement
        # of what an encoder emitted, so it belongs to the artifact.
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        # Eq. 6 ADDS e_layout to the fused query, so this is not a free choice.
        out_dim=EMBED_DIM,
    )
    return MetaFindDualTower(DualTowerConfig(
        dim=EMBED_DIM, tower_sharing=training["tower_sharing"],
        query_fusion=fusion, gallery_fusion=gallery_fusion,
        # [Table 3] "w/o Layout Context" is this flag. It is the one ablation
        # field Stage 2 owns; the rest were fixed when n10 trained the towers.
        use_layout=use_layout, essgnn=essgnn if use_layout else None,
        # Eq. 6's lambda starts here. The paper calls it "a learnable scalar"
        # and gives no initial value; it lives in stage2_protocol (a Stage 2
        # decision), not in the Stage 1 recipe it used to be read from.
        init_lambda=float(init_lambda)))


def restore_stage1_for_stage2(backbone, model, parent_values: dict,
                              stage2_values: dict, checkpoint_path, *,
                              query_backbone=None):
    """Restore the parent's loss scope before applying the child's temperature.

    ``parent_values`` must come from ``effective_stage1_model_inputs`` for the
    verified parent checkpoint. Fixed Stage 1 losses have no trainable state,
    so restoring directly into a learnable child incorrectly requires a new
    parameter in the parent. Conversely, loading a learned parent directly into
    a fixed child silently overwrites that child's requested fixed temperature.

    A learnable child retains the existing parent-initialisation choice (raw
    scale, before either stage's clamp). A fixed child always uses its own
    recipe. Both the requested and the effective initial temperatures are
    recorded so that a clamped inherited scale cannot impersonate the recipe.
    """
    import torch
    from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss
    from metafind.train.stage1 import load_stage1_checkpoint

    keys = ("learnable_temperature", "init_temperature", "max_logit_scale")
    for name, values in (("Stage 1", parent_values), ("Stage 2", stage2_values)):
        if missing := [key for key in keys if key not in values or values[key] is None]:
            raise ValueError(f"{name} lacks bound temperature settings: {missing}")
    parent_loss = MetaFindContrastiveLoss(ContrastiveConfig(
        bidirectional=False, **{key: parent_values[key] for key in keys}))
    parent_fixed_scale = parent_loss.logit_scale.detach().clone()
    load_stage1_checkpoint(
        backbone, model, parent_loss, checkpoint_path,
        new_prefixes=("query.layout_encoder", "query.layout_weight"),
        query_backbone=query_backbone)
    if (not parent_loss.cfg.learnable_temperature
            and not torch.equal(parent_loss.logit_scale, parent_fixed_scale)):
        raise ValueError("Stage 1 fixed temperature state disagrees with its bound recipe")
    child_loss = MetaFindContrastiveLoss(ContrastiveConfig(
        bidirectional=True, **{key: stage2_values[key] for key in keys}))
    source = "stage2_recipe"
    if child_loss.cfg.learnable_temperature:
        with torch.no_grad():
            child_loss.logit_scale.copy_(parent_loss.logit_scale)
        source = ("stage1_checkpoint" if parent_loss.cfg.learnable_temperature
                  else "stage1_fixed_recipe")
    scale = child_loss.logit_scale.detach().exp()
    effective_scale = (scale.clamp(max=child_loss.cfg.max_logit_scale)
                       if child_loss.cfg.learnable_temperature else scale)
    temperature_init = {
        "source": source,
        "requested_init_temperature": stage2_values["init_temperature"],
        "raw_initial_temperature": scale.reciprocal().item(),
        "effective_initial_temperature": effective_scale.reciprocal().item(),
        "initial_logit_scale": child_loss.logit_scale.detach().item(),
        "learnable_temperature": child_loss.cfg.learnable_temperature,
        "max_logit_scale": child_loss.cfg.max_logit_scale,
    }
    return child_loss, temperature_init


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit-houses", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage1-ckpt-record", default=None,
                    help="Stage 1 checkpoint record to initialise from. "
                         "Defaults to the canonical stage1_ckpt.json.")
    ap.add_argument("--allow-legacy-gallery-index", action="store_true",
                    help="explicitly accept missing legacy source identity or a v1 "
                         "encoder hash; incomplete producer provenance is recorded")
    ap.add_argument("--allow-legacy-semantic-inputs", action="store_true",
                    help="explicitly accept n08 node/edge caches without bound source identity; "
                         "records legacy_unbound instead of verified provenance")
    # Stage 2's training recipe (optimizer, learning rate, weight decay, batch
    # size, epochs, seed, temperature, scene dropout) used to be read from the
    # STAGE 1 hyperparameter artifact without anyone saying so: the paper gives
    # no Stage 2 recipe, and inheriting Stage 1's silently made it look decided.
    # It must now be named on the command line. Passing Stage 1's file is
    # allowed -- that is the current, undecided fallback -- but the record then
    # says so, and the choice is visible in the launch command.
    ap.add_argument("--overwrite", action="store_true",
                    help="permit replacing an existing stage2_<variant>.pt and "
                         "its record. Without it an occupied destination stops "
                         "the run before the first batch.")
    ap.add_argument("--query-modality-masking", default=None,
                    choices=QUERY_MASKING_MODES,
                    help="override stage2_protocol.query_modality_masking for "
                         "this run; recorded in the checkpoint")
    ap.add_argument("--hyperparameters", required=True,
                    help="JSON with a `values` block (same schema as "
                         "stage1_hyperparameters.json) giving Stage 2's optimizer, "
                         "learning_rate, weight_decay, batch_size, epochs, seed, "
                         "init_temperature, learnable_temperature, max_logit_scale "
                         "and optionally scene_dropout (paper: 0.30). Its path and "
                         "sha256 go into the checkpoint record.")
    args = ap.parse_args()

    import torch
    from metafind.train.stage1 import (
        effective_stage1_model_inputs, load_protocols,
        load_stage1_model_config, model_input_snapshot, stage1_backbone_kwargs)
    from metafind.models.ulip_backbone import (
        BackboneConfig, ULIPBackbone, prepare_depth_shell)
    from metafind.train.gallery_index import (verified_stage2_index, verify_gallery_encoder,
                                              verify_stage2_gallery_sources)

    # Stage 1's encoding and training protocols still decide the fusion
    # architecture Stage 2 restores into; only the RECIPE comes from the file
    # named on the command line.
    encoding, training, _stage1_hyperparameters = load_protocols()
    stage2, edge_proto, arch_proto = load_stage2_protocols()
    hp_path = Path(args.hyperparameters)
    hyperparameters = json.loads(hp_path.read_text())
    if "values" not in hyperparameters:
        raise ValueError(f"{hp_path} has no `values` block")
    values = hyperparameters["values"]
    for key in ("optimizer", "learning_rate", "weight_decay", "batch_size",
                "epochs", "seed", "init_temperature", "learnable_temperature",
                "max_logit_scale"):
        if key not in values:
            raise ValueError(f"{hp_path} values block is missing {key!r}")
    hp_sha256 = hashlib.sha256(hp_path.read_bytes()).hexdigest()
    hp_is_stage1 = hp_sha256 == _stage1_hyperparameters.get("sha256")
    dest = CKPT_DIR / f"stage2_{args.variant}.pt"
    if dest.exists() and not args.overwrite:
        raise SystemExit(f"{dest} already exists; the path is fixed per variant, "
                         "so a second run would overwrite the first's weights "
                         "and record. Pass --overwrite to replace them.")
    # Whether the query side sees Stage 1's per-modality masking during Stage 2.
    # The paper states scene dropout for Stage 2 and says nothing about
    # modality masking there, so this is a recorded choice, not a paper fact:
    # "none" is what every Stage 2 run so far did (all three modalities always
    # present); "p_mask" would be an ablation and is not implemented yet.
    query_masking = stage2.get("query_modality_masking", "none")
    if args.query_modality_masking is not None:
        print(f"query_modality_masking {query_masking!r} (protocol) overridden by "
              f"--query-modality-masking {args.query_modality_masking!r}; recorded",
              flush=True)
        query_masking = args.query_modality_masking
    if query_masking not in QUERY_MASKING_MODES:
        raise SystemExit(f"query_modality_masking {query_masking!r}; known "
                         f"{QUERY_MASKING_MODES}")
    print(f"hyperparameters {hp_path}  sha256 {hp_sha256[:12]}"
          + ("  (this is the STAGE 1 artifact: Stage 2 recipe not yet decided, "
             "inheriting Stage 1's values)" if hp_is_stage1 else ""), flush=True)

    scene_splits = json.loads((paths.OUTPUTS / "scene_splits.json").read_text())
    train_houses = scene_splits["train_houses"]
    if args.limit_houses:
        train_houses = train_houses[: args.limit_houses]
    input_identity = capture_stage2_input_identity(stage2, edge_proto, arch_proto, train_houses)

    positive_map = json.loads((paths.OUTPUTS / "stage2_positive_map.json").read_text())
    index_record = json.loads((paths.OUTPUTS / "stage2_gallery_index.json").read_text())
    # [CODEX MAJOR 2026-08-30] See gallery_index.load_checkpoint_record: a
    # run-specific Stage 1 checkpoint must be able to reach Stage 2 without
    # being copied over the canonical name.
    ckpt = json.loads(Path(getattr(args, "stage1_ckpt_record", None)
                           or paths.CHECKPOINTS / "stage1_ckpt.json").read_text())
    ckpt = load_stage1_model_config(ckpt["uri"], ckpt)
    encoding, training, _stage1_hyperparameters = effective_stage1_model_inputs(
        ckpt, encoding, training, _stage1_hyperparameters)

    # [G6] The index must come from the checkpoint this run loads. Comparing
    # here as well as at the gate: a gate verdict is a record of the past, and
    # the index can be rebuilt between the verdict and the run.
    #
    # [CODEX MAJOR 2026-08-30] The comment above stood for weeks over no
    # comparison at all -- the index carried no statement of which checkpoint
    # produced it, so there was nothing to compare. Both halves now exist.
    built_from = index_record.get("stage1_checkpoint_sha256")
    if built_from is None:
        raise ValueError(
            f"{paths.OUTPUTS / 'stage2_gallery_index.json'} predates "
            "`stage1_checkpoint_sha256` and cannot say which checkpoint built "
            "it. Rebuild the index (n11) before running Stage 2.")
    if built_from != ckpt["sha256"]:
        raise ValueError(
            f"the gallery index was built from checkpoint {built_from[:16]}... "
            f"but this run loads {ckpt['sha256'][:16]}.... Queries and gallery "
            "would come from different encoders. Rebuild the index.")
    # [DL-104, ESSGNN REVIEWER MAJOR 1] The index must also have been built under
    # the SAME modality declaration: a three-array index (gallery fused WITH the
    # cloud) under a text+image protocol passes every sha check and trains a
    # pc-absent query against pc-bearing gallery vectors. An index written before
    # the field is read as all three.
    asset_modalities = tuple(stage2["asset_modalities"])
    if training["tower_sharing"] == "fully_separate" and "pc" in asset_modalities:
        raise ValueError("Stage 2 gallery caches do not contain the separate query "
                         "point path; fully_separate Stage 2 requires cached query "
                         "point vectors before point-cloud modalities can be used")
    index_declared = tuple(index_record.get("modality_completeness", {})
                           .get("declared_modalities", ["text", "image", "pc"]))
    if index_declared != asset_modalities:
        raise ValueError(
            f"the gallery index was built with asset_modalities {index_declared} but "
            f"the protocol declares {asset_modalities}. Rebuild the index (n11b).")
    weights = Path(ckpt["uri"])
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != ckpt["sha256"]:
        raise ValueError(
            f"{weights} hashes to {actual[:16]}... but its record claims "
            f"{ckpt['sha256'][:16]}.... Record and weights have diverged.")

    # [CODEX MAJOR 2026-08-30] `build_index` has always recorded the `.npz`'s
    # sha256 and nothing ever read it. An index overwritten or truncated after
    # the record was written still loaded, while the record went on claiming the
    # old digest AND the producer checkpoint -- so the linkage check above would
    # pass over bytes neither of them describes.
    gallery_ids, gallery_embeddings, gallery_index = verified_stage2_index(
        index_record, ckpt["sha256"], asset_modalities,
        allow_legacy_sources=args.allow_legacy_gallery_index)
    id_to_row = {a: i for i, a in enumerate(gallery_ids)}
    gallery_vecs = torch.from_numpy(gallery_embeddings).to(args.device)

    data = Stage2Data(args.device, graph_unit=stage2["graph_unit"],
                      allow_legacy_semantic_inputs=args.allow_legacy_semantic_inputs)
    data.asset_vectors = load_asset_modality_vectors(gallery_index, asset_modalities)
    eligible = set(positive_map) & set(id_to_row) & set(data.modalities)
    samples = enumerate_samples(train_houses, eligible, graph_unit=stage2["graph_unit"])
    if not samples:
        print("no eligible sample; check stage2_positive_map and the gallery index",
              flush=True)
        return 2

    graphs = data.graphs_for({h for h, _, _ in samples})
    verify_stage2_input_identity({"input_identity": input_identity}, current_inputs=True)

    seed = values["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Validate the first real partition before allocating the large backbone.
    # Reuse it in epoch zero so validation does not consume an extra RNG draw.
    first_batches = training_batches(samples, values["batch_size"], rng)

    # Restore the parent's actual trainable subset. A fuser-only checkpoint has
    # no fine-tuned point state; requiring it would reject a valid parent.
    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope=training["train_scope"],
                                           **stage1_backbone_kwargs(ckpt)))
    backbone_q = (backbone.clone_point_path()
                  if training["tower_sharing"] == "fully_separate" else None)
    # [P0-3] build_stage2_model, NOT Stage 1's build_model. Stage 1 builds with
    # use_layout=False -- correct there, since 2.6 puts ESSGNN in Stage 2 -- so
    # reusing it here left query.layout_encoder as None while encode_query below
    # calls encode_layout, which raises. Eq. 6's lambda was absent too.
    variant = load_variant(args.variant, ckpt, training=training,
                           encoding=encoding, values=_stage1_hyperparameters["values"])
    use_layout = variant["layout_encoder"] is not None
    if "init_lambda" not in stage2 and "init_lambda_ratio" not in stage2:
        raise ValueError(
            "stage2_protocol.json records neither init_lambda nor "
            "init_lambda_ratio; re-run `python -m metafind.models.resolve_stage2` "
            "so Eq. 6's starting value is a recorded decision, not a default")
    # A pinned literal WINS and no measurement is taken; that path exists so a
    # literal stays expressible. Otherwise lambda_0 is derived below, after the
    # model exists, from the ratio and a norm measured on this checkpoint.
    pinned = stage2.get("init_lambda")
    model = build_stage2_model(encoding, training, hyperparameters, arch_proto,
                               node_feat_dim=data.node_dim,
                               edge_feat_dim=data.edge_dim,
                               use_layout=use_layout,
                               init_lambda=float(pinned) if pinned is not None
                               else 1.0)
    if training["freeze_gallery"]:
        model.freeze_gallery(True)

    # Eq. 7/8 changes directionality, not the fixed Stage 2 recipe. Only a
    # learnable child inherits the parent's raw scale as an implementation choice.
    loss_fn, temperature_init = restore_stage1_for_stage2(
        backbone, model, _stage1_hyperparameters["values"], values,
        Path(ckpt["uri"]), query_backbone=backbone_q)
    model.to(args.device)
    loss_fn.to(args.device)
    grads = freeze_for_stage2(model, backbone, query_modality_masking=query_masking,
                              asset_modalities=asset_modalities)
    if backbone_q is not None:
        backbone_q.set_train_scope("fuser_only")
        if not backbone_q.is_frozen():
            raise RuntimeError("the restored query point backbone is not frozen in Stage 2")
    verify_gallery_encoder(index_record, backbone, model, parent_checkpoint=ckpt,
                           allow_legacy=args.allow_legacy_gallery_index,
                           declared_modalities=asset_modalities)

    # Same optimizer construction as Stage 1: ULIP's rule puts biases, norms
    # and every 0-/1-D tensor (that includes Eq. 6's lambda and the missing-
    # edge token) in the no-decay group, and the artifact's betas/eps are used
    # rather than torch's defaults. A flat AdamW(weight_decay=...) decayed all
    # of them, and decayed the fusion mask tokens -- which under
    # query_modality_masking="none" receive a zero gradient every step -- by
    # (1 - lr*wd) per step with nothing pushing back (measured, see
    # output/look/stage2_smoke_seven_checks.json "optimizer_audit").
    from metafind.train.stage1 import weight_decay_groups
    opt = torch.optim.AdamW(
        weight_decay_groups(list(model.named_parameters())
                            + list(loss_fn.named_parameters()),
                            values["weight_decay"],
                            # [ULIP2 REVIEWER MINOR 4] Was omitted, so Stage 2
                            # took the default while Stage 1 read the artifact.
                            # LIVE since DL-104: with the pc undeclared the mask
                            # tokens train, only the pc row ever gets a gradient,
                            # and decay would erode the text/image rows with
                            # nothing pushing back (checked below).
                            decay_mask_tokens=bool(
                                values["decay_mask_tokens"])),
        lr=values["learning_rate"], betas=tuple(values["betas"]),
        eps=values["eps"])

    if grads.get("query.fusion.mask_tokens") and query_masking == "none" \
            and bool(values["decay_mask_tokens"]):
        raise ValueError(
            "decay_mask_tokens is true while the mask tokens train under "
            "query_modality_masking none: the rows never selected (text, image) "
            "have an exactly-zero gradient and would only shrink. Set it false "
            "in the Stage 2 hyperparameters or mask those modalities too.")

    # Eq. 6's lambda_0, derived AFTER the checkpoint is restored and BEFORE the
    # first optimizer step, so the norm it is a tenth of is this run's own.
    # Placed here rather than beside `build_stage2_model` because it needs the
    # restored weights and the graphs; the model was built with a placeholder
    # and the parameter is overwritten in place below.
    lambda_record = initialise_layout_lambda(
        model, stage2, samples, graphs, data, args.device,
        arch_protocol=arch_proto)

    # [P1] NOT values["p_mask"]. Both rates are 30% in the paper, which is
    # exactly what made the alias invisible -- but they are different
    # mechanisms: 2.6 masks each MODALITY independently at 30% in Stage 1, and
    # drops the layout vector in 30% of BATCHES in Stage 2. Table 3 sweeps
    # p_mask; under the alias, a p_mask=0.10 row would silently have moved
    # scene dropout to 10% too, and the ablation would no longer be an ablation
    # of one thing.
    stage2_dropout = float(values.get("scene_dropout", PAPER_SCENE_DROPOUT))
    # [AUDIT 2026-09-04 F3] Stage 1's recipe has a warmup and a cosine decay;
    # Stage 2's optimizer ran flat, and the first 50 steps at the flat rate
    # cost the probe batch 11 points of in-batch top-1. Optional, recipe-driven:
    # warmup_frac (of all steps) and lr_end; absent = flat, as before.
    warmup_frac = float(values.get("warmup_frac", 0.0))
    lr_start = float(values.get("lr_start", 1e-6))
    lr_end = float(values.get("lr_end", values["learning_rate"]))
    lr_schedule = None
    from metafind.train.stage1 import cosine_schedule
    epochs = args.epochs or values["epochs"]
    if epochs <= 0:
        raise SystemExit(f"epochs must be positive, got {epochs}")
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
            epoch_start_step = step
            # The QUERY tower only. `model.train()` recursed into the gallery
            # tower and undid freeze_gallery's eval(); harmless while the loop
            # never calls the gallery, but a promise in freeze_gallery's
            # docstring that the code then broke.
            model.query.train()
            batches, n_small, n_small_samples = (first_batches if epoch == 0 else
                training_batches(samples, values["batch_size"], rng))
            if epoch == 0:
                print(f"  {len(batches):,} batches kept; {n_small} with fewer than "
                      f"{MIN_BATCH} samples dropped ({n_small_samples:,} samples, the "
                      "over-placed assets' leftovers)", flush=True)
            if epoch == 0 and warmup_frac > 0:
                total = epochs * len(batches)
                warm = max(1, int(round(warmup_frac * total)))
                lr_schedule = cosine_schedule(float(values["learning_rate"]), lr_end,
                                              1, total, 0, lr_start)
                # cosine_schedule's warmup is in epochs; do the warmup in steps
                # here so a one-epoch fine-tune can still warm up briefly.
                ramp = np.linspace(lr_start, float(values["learning_rate"]), warm)
                lr_schedule = np.concatenate([ramp, lr_schedule[warm:]])
                print(f"  lr schedule: warmup {warm} steps from {lr_start:g}, cosine "
                      f"{values['learning_rate']:g} -> {lr_end:g} over {total} steps",
                      flush=True)
            for batch in batches:
                if lr_schedule is not None:
                    lr_now = float(lr_schedule[min(step, len(lr_schedule) - 1)])
                    for group in opt.param_groups:
                        group["lr"] = lr_now
                # [U-32 / 2.6] "omitted in 30% of BATCHES" -- ONE draw per batch,
                # so every sample in a dropped batch loses the layout term and
                # every sample in a kept batch has it. L1-SCENE-DROPOUT-30
                # asserts the within-batch variance of this mask is zero.
                drop = bool(rng.random() < stage2_dropout)

                queries, positives = [], []
                for idx in batch:
                    house_id, target_index, asset_id = samples[idx]
                    graph = graphs[house_id]
                    present = query_present(query_masking, rng)
                    if present is not None:
                        present = present.to(args.device)
                    q = encode_query(model, graph, target_index,
                                     asset_id, drop, args.device, data,
                                     present=present)
                    queries.append(q)
                    positives.append(gallery_vecs[id_to_row[asset_id]])

                q = torch.stack(queries)
                g = torch.stack(positives)
                out = loss_fn(q, g)

                opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                opt.step()
                step += 1

                if step % 20 == 0:
                    runlog.train_metrics(
                        f"stage2_{args.variant}", epoch=epoch, step=step,
                        loss=round(out["loss"].item(), 6),
                        loss_q2g=round(out["loss_q2g"].item(), 6),
                        loss_g2q=round(out["loss_g2q"].item(), 6),
                        acc_q2g=round(out.get("acc_q2g", torch.tensor(0.0)).item(), 6),
                        tau=round(out["temperature"].item(), 6),
                        # Eq. 6's learnable scalar. Watching it is the cheapest
                        # read on whether the layout branch is contributing at
                        # all: if lambda decays toward zero the model is
                        # learning to ignore ESSGNN, which is a result about the
                        # method and not a bug to hide.
                        **({"lam": round(model.query.lam.item(), 6)}
                           if model.query.layout_encoder is not None else {}),
                        layout_dropped=int(drop))
                if step % 50 == 0:
                    print(f"  epoch {epoch} step {step}: loss {out['loss'].item():.4f}, "
                          f"tau {out['temperature'].item():.4f}, "
                          f"layout {'dropped' if drop else 'used'}", flush=True)

            if step == epoch_start_step:
                raise RuntimeError("Stage 2 epoch performed no optimizer steps; refusing checkpoint")
            record = {
                "variant_id": args.variant,
                "use_layout": use_layout,
                "uri": str(CKPT_DIR / f"stage2_{args.variant}.pt"),
                "trainable_only": True,
                # What this run actually trained with, so a later reader does
                # not have to guess which artifact supplied the recipe.
                "stage1_checkpoint_sha256": ckpt["sha256"],
                "stage1_checkpoint_uri": str(ckpt["uri"]),
                "stage1_model_inputs": model_input_snapshot(
                    encoding, training, _stage1_hyperparameters, model),
                "gallery_index_sha256": index_record["sha256"],
                "allow_legacy_gallery_index": args.allow_legacy_gallery_index,
                "allow_legacy_semantic_inputs": args.allow_legacy_semantic_inputs,
                "semantic_source_status": data.semantic_source_status,
                "gallery_source_status": ("verified" if index_record.get("source_identity")
                                          else "legacy_unbound"),
                "layout_input_dims": {"node_feat_dim": data.node_dim,
                                      "edge_feat_dim": data.edge_dim},
                "graph_unit": stage2["graph_unit"],
                "stage2_protocol": input_identity["protocols"]["stage2"],
                "edge_protocol": input_identity["protocols"]["edge"],
                "input_identity": input_identity,
                "samples_sha256": hashlib.sha256(
                    json.dumps(samples, separators=(",", ":")).encode()).hexdigest(),
                "hyperparameters_uri": str(hp_path),
                "hyperparameters_sha256": hp_sha256,
                "hyperparameters": hyperparameters,
                "hyperparameters_are_stage1_artifact": hp_is_stage1,
                "temperature_init": temperature_init,
                "effective_values": {
                    "learning_rate": values["learning_rate"],
                    "weight_decay": values["weight_decay"],
                    "batch_size": values["batch_size"],
                    "epochs": epochs,
                    "seed": seed,
                    "scene_dropout": stage2_dropout,
                    "warmup_frac": warmup_frac,
                    "lr_start": lr_start if warmup_frac > 0 else None,
                    "lr_end": lr_end if warmup_frac > 0 else None,
                    "init_temperature": temperature_init["effective_initial_temperature"],
                    "learnable_temperature": values["learnable_temperature"],
                },
                # Eq. 6's starting lambda AND how it was obtained. Without the
                # second half, a finished run's lambda_0 is unrecoverable from
                # its own record -- which is what DL-078 left behind, and the
                # ESSGNN Reviewer's MAJOR-3.
                "lambda_init": lambda_record,
                "samples_per_epoch": len(samples),
                "min_batch_size": MIN_BATCH,
                "small_batches_dropped_per_epoch": n_small,
                "small_batch_samples_dropped_per_epoch": n_small_samples,
                "train_houses": len(train_houses),
                "n_params_saved": sum(
                    v.numel() for m in (model, loss_fn)
                    for v in trainable_state_dict(m).values()),
            }
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        # `loss_trainable_state` is here for the same reason Stage 1 has it, and
        # the omission was the same bug one stage later: `learnable_temperature`
        # defaults True, so tau IS in this optimizer -- `params` above takes
        # loss_fn.parameters() -- and saving only `model` trained it all run and
        # then dropped it. Nothing errors; the file is simply missing a tensor
        # that moved.
        record["arch_protocol"] = {k: arch_proto[k] for k in sorted(arch_proto)
                                   if not k.startswith("decided")}
        # [AUDIT 2026-09-04 D-2] h0_mode, coords_agg, edge_proj_dim and
        # normalize_coord_diff are pinned in code (essgnn.PRIMARY_INTERPRETATION),
        # not in the protocol artifact; a checkpoint must still say which it ran.
        from metafind.models.essgnn import PRIMARY_INTERPRETATION
        record["primary_interpretation"] = dict(PRIMARY_INTERPRETATION)
        record["query_modality_masking"] = query_masking
        record["code_revision"] = runlog.code_revision()
        record["code_dirty"] = runlog.code_dirty()
        record["steps"] = step
        # Source files can change during a long training run. Never publish a
        # fresh checkpoint claiming that a stale gallery matches current inputs.
        verify_stage2_gallery_sources(index_record, gallery_index,
                                      allow_legacy=args.allow_legacy_gallery_index)
        torch.save(stage2_checkpoint_payload(model, loss_fn, record), record["uri"])
        record["sha256"] = hashlib.sha256(Path(record["uri"]).read_bytes()).hexdigest()
        record["size_bytes"] = Path(record["uri"]).stat().st_size
        # The record used to be built, printed and dropped: a stage2_<variant>.pt
        # existed on disk with nothing saying which checkpoint, index, recipe,
        # seed or arch protocol produced it. Written per variant, merged into
        # one file, and an existing entry for the same variant is only replaced
        # under --overwrite (checked before training started, above).
        existing = (json.loads(VARIANT_CKPTS.read_text())
                    if VARIANT_CKPTS.exists() else {})
        existing[args.variant] = record
        tmp = VARIANT_CKPTS.with_suffix(".json.part")
        tmp.write_text(json.dumps(existing, indent=1))
        tmp.replace(VARIANT_CKPTS)

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1), steps=step)
    print(f"\n{args.variant}: {record['n_params_saved']:,} params, "
          f"{record['size_bytes'] / 1e6:.0f} MB -> {record['uri']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
